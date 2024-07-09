# This code is modified heavily from Mamba implementation
# Copyright (c) 2023, Tri Dao.

"""We want triton==2.1.0 for this
"""

import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat


@triton.heuristics({"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])})
@triton.jit
def _selective_scan_update_kernel(
    # Pointers to matrices
    kv_state_ptr, k_state_ptr, x_ptr, B_ptr, C_ptr, out_ptr,
    # Matrix dimensions
    dim, dstate,
    # Strides
    stride_kv_state_batch, stride_kv_state_dim, stride_kv_state_dstate,
    stride_k_state_batch, stride_k_state_dstate,
    stride_x_batch, stride_x_dim,
    stride_B_batch, stride_B_dstate,
    stride_C_batch, stride_C_dstate,
    stride_out_batch, stride_out_dim,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    kv_state_ptr += pid_b * stride_kv_state_batch
    k_state_ptr += pid_b * stride_k_state_batch
    x_ptr += pid_b * stride_x_batch
    B_ptr += pid_b * stride_B_batch
    C_ptr += pid_b * stride_C_batch

    out_ptr += pid_b * stride_out_batch

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    kv_state_ptrs = kv_state_ptr + (offs_m[:, None] * stride_kv_state_dim + offs_n[None, :] * stride_kv_state_dstate)
    k_state_ptrs = k_state_ptr + offs_n * stride_k_state_dstate
    x_ptrs = x_ptr + offs_m * stride_x_dim
    B_ptrs = B_ptr + offs_n * stride_B_dstate
    C_ptrs = C_ptr + offs_n * stride_C_dstate
    out_ptrs = out_ptr + offs_m * stride_out_dim

    kv_state = tl.load(kv_state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)
    k_state = tl.load(k_state_ptrs, mask=offs_n < dstate, other=0.0)

    x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0)
    B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0)
    C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0)

    kv_state = kv_state + B[None, :] * x[:, None]
    tl.store(kv_state_ptrs, kv_state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))
    
    k_state = k_state + B
    tl.store(k_state_ptrs, k_state, mask=offs_n < dstate)
    
    num = tl.sum(kv_state * C[None, :], axis=1)
    # den = tl.sum(k_state * C, axis=1) + 1e-6

    tl.store(out_ptrs, num, mask=offs_m < dim)


def selective_state_update(
    kv_state, 
    k_state,
    x, 
    B, 
    C
):
    """
    Argument:
        state: (batch, dim, dstate)
        x: (batch, dim)
        dt: (batch, dim)
        A: (dim, dstate)
        B: (batch, dstate)
        C: (batch, dstate)
        D: (dim,)
        z: (batch, dim)
        dt_bias: (dim,)
    Return:
        out: (batch, dim)
    """
    batch, dim, dstate = kv_state.shape
    assert x.shape == (batch, dim)
    assert B.shape == (batch, dstate)
    assert C.shape == B.shape

    out = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE_M']), batch)
    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    BLOCK_SIZE_M, num_warps = ((32, 4) if dstate <=  16
                               else ((16, 4) if dstate <= 32 else
                                     ((8, 4) if dstate <= 64 else
                                      ((4, 4) if dstate <= 128 else
                                       ((4, 8))))))
    BLOCK_SIZE_M, num_warps = (4, 8)

    with torch.cuda.device(x.device.index):
        _selective_scan_update_kernel[grid](
            kv_state, k_state, x, B, C, out,
            dim, dstate,
            kv_state.stride(0), kv_state.stride(1), kv_state.stride(2),
            k_state.stride(0), k_state.stride(1),
            x.stride(0), x.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE_M,
            num_warps=num_warps,
        )
    return out


def selective_state_update_ref(kv_state, k_state, q, k, v, denom: bool=True, eps: float=1e-6):
    """
    Argument:
        kv_state: (batch, d_model, dstate)
        k_state: (batch, d_state)
        q: (batch, d_model)
        k: (batch, d_model)
        v: (batch, d_model)

    Return:
        out: (batch, d_model)
    """
    k_state += k
    kv_state += torch.einsum("bf,bd->bdf", k, v)
    num = torch.einsum("bf,bdf->bd", q, kv_state)

    if denom:
        den = torch.einsum("bf,bf->b", q, k_state) + eps
        return num / den.unsqueeze(-1)
    else:
        return num

def based_update_ref(
    kv_state: torch.Tensor, 
    k_state: torch.Tensor, 
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    eps: float
):
    """
    Compute linear attention with recurrent view
    """
    b, h, l, d = q.shape
    assert l == 1, f'q.shape is {q.shape} but should be ({b}, {h}, 1, {d})'
    # Expand dims for broadcasting to compute linear attention
    k, v =  k.unsqueeze(-2), v.unsqueeze(-1)
    kv_state += k[:, :, -1:] * v[:, :, -1:]
    k_state  += k[:, :, -1:]

    # Compute linear attention

    # q.shape = torch.Size([128, 16, 1, 1, 601])
    # kv_state.shape = torch.Size([128, 16, 1, 64, 601])
    # k_stae.shape = torch.Size([128, 16, 1, 1, 601])
    num = torch.einsum(
        "bhlf,bhldf->bhld", q, kv_state
    )

    den = torch.einsum(
        "bhlf,bhldf->bhl", q, k_state
    ) + eps
    y = num / den.unsqueeze(-1)
    return y



