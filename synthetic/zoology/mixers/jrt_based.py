"""
Linear attention in Based. 
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import opt_einsum as oe
from einops import rearrange
from typing import Optional, Tuple
from pydantic import validate_call

from zoology.utils import import_from_str

try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, LlamaRotaryEmbedding
except:
    print(f"Failed to import LlamaRotaryEmbedding... ")

try:
    import sys
    sys.path.append('/var/cr05_data/sim_data/code/release/based/train/')
    from csrc.causal_dot_prod import causal_dot_product  # linear attention cuda kernel
    print(f"Succesfully imported the causal dot product kernel... ")
except:
    causal_dot_product = None
    print(f"Failed to import the causal dot product kernel... ")


def init_feature_map(feature_map: str='none', **kwargs: any):
    """
    Initialize query and key mapping for linear attention
    """
    if feature_map in [None, 'none', 'identity']:
        from zoology.mixers.feature_maps.base import FeatureMap
        return FeatureMap(**kwargs)
    # Taylor series approximations to exp(x)
    elif feature_map == 'taylor_exp':
        from zoology.mixers.feature_maps.taylor import TaylorExp
        return TaylorExp(**kwargs) 
    elif feature_map == "performer":
        from zoology.mixers.feature_maps.performer import PerformerFeatureMap
        return PerformerFeatureMap(**kwargs)
    elif feature_map == "cosformer":
        from zoology.mixers.feature_maps.cosformer import CosFormerFeatureMap
        return CosFormerFeatureMap(**kwargs)
    elif feature_map == "pos_elu":
        from zoology.mixers.feature_maps.base import PosELU
        return PosELU(**kwargs)
    elif feature_map == "all_poly":
        from zoology.mixers.feature_maps.all_poly import AllPolyMap
        return AllPolyMap(**kwargs)
    else:
        feature_map = import_from_str(feature_map)
        return feature_map(**kwargs)   


class Based(nn.Module):
    
    @validate_call
    def __init__(
        self,
        d_model: int,
        l_max: int = 2048,
        feature_dim: int = 16,
        num_key_value_heads: int = 12,
        num_heads: int = 12,
        feature_name: "str" = "taylor_exp",
        feature_kwargs: dict = {},
        eps: float = 1e-12,
        causal: bool = True,
        apply_rotary: bool = False,
        rope_theta: int=10000.0,
        train_view: str = "linear",

        enc_length: int = None,
        dec_length: int = None,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.train_view = train_view

        # linear attention 
        self.feature_name = feature_name
        self.feature_dim = feature_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_heads = num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.d_model // self.num_key_value_heads
        self.causal=causal
        print(f"{self.causal=}")

        feature_map_kwargs = {
            'input_dim': self.feature_dim,
            'head_dim_idx': -1,
            'temp': 1.,
            'eps': 1e-12,
            **feature_kwargs
        }
        self.feature_map = init_feature_map(feature_map=self.feature_name, **feature_map_kwargs)
        self.proj_q = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_k = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_v = nn.Linear(self.d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.proj_o = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)
        self.dropout = nn.Identity()
        self.eps = eps

        # parameters
        self.apply_rotary = apply_rotary
        self.rope_theta = rope_theta
        self.q_shape = [self.num_heads, self.feature_dim]
        self.k_shape = [self.num_key_value_heads, self.feature_dim]
        self.v_shape = [self.num_key_value_heads, self.head_dim]
        if self.apply_rotary:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.feature_dim,
                max_position_embeddings=self.l_max,
                base=self.rope_theta,
            )
        
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        self.enc_length = enc_length
        self.dec_length = dec_length
        # self.proj_k_enc = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        # self.proj_v_enc = nn.Linear(self.d_model, self.num_heads_kv * self.head_dim, bias=False)


    def process_qkv(
        self, 
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ):
        """
        Get Q, K, V tensors from hidden_states, e.g., by applying projections, 
        positional embeddings, KV cache
        -> Follow the original LlamaAttention API
        """
        b, l, _ = hidden_states.size()
        q, k, v = self.proj_q(hidden_states), self.proj_k(hidden_states), self.proj_v(hidden_states)
        
        # Following HF Llama source code to get (b, h, l, d)
        q = q.view(b, l, *self.q_shape).transpose(1, 2)
        k = k.view(b, l, *self.k_shape).transpose(1, 2)
        v = v.view(b, l, *self.v_shape).transpose(1, 2)
        
        kv_seq_len = k.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
            
        # Apply rotary embeddings
        if position_ids is None:
            position_ids = torch.arange(
                kv_seq_len, dtype=torch.long, device=hidden_states.device
            )
            position_ids = position_ids.unsqueeze(0).expand((b, kv_seq_len))
            cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # KV cache
        if past_key_value is not None:
            # Reuse k, v, self_attention
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
            
        past_key_value = (k, v) if use_cache else None

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        return q, k, v, kv_seq_len


    def forward(
        self, 
        hidden_states: torch.Tensor, 
        past_key_value: Optional[Tuple[torch.Tensor]] = None, 
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        *args, **kwargs
    ):
        """
        x (torch.Tensor): tensor of shape (b, d, l)
        y (torch.Tensor): tensor of shape (b, d, l)
        """
        # hidden_states = hidden_states.transpose(1, 2)
        b, l, d = hidden_states.size()
        if self.apply_rotary:
            assert d == self.d_model, f'Hidden_states.shape should be size {(b, l, d)} but is shape {hidden_states.shape}'
            q, k, v, kv_seq_len = self.process_qkv(hidden_states, past_key_value, position_ids, use_cache)
        else:
            q, k, v = self.proj_q(hidden_states), self.proj_k(hidden_states), self.proj_v(hidden_states)
            q = q.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)
            k = k.view(b, l, self.num_key_value_heads, self.feature_dim).transpose(1, 2)
            v = v.view(b, l, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Encoder part
        enc_length = l #min(self.enc_length, l) # In inference, enc_length could be > seqlen
        x_enc = hidden_states #[:, :enc_length]
        k_enc, v_enc = k.clone(), v.clone() #self.proj_k(x_enc), self.proj_v(x_enc)
        # k_enc = k_enc.view(b, enc_length, self.num_heads, self.feature_dim).transpose(1, 2)
        # v_enc = v_enc.view(b, enc_length, self.num_heads, self.head_dim).transpose(1, 2)

        return self.parallel_forward(hidden_states, q, k, v, k_enc, v_enc)


    def parallel_forward(self, x, q, k, v, k_enc=None, v_enc=None):
        batch, seqlen = x.shape[:2]
        q, k = self.feature_map(q), self.feature_map(k)
        k_enc = self.feature_map(k_enc)

        # Denominator
        k_state = k_enc.sum(dim=2, keepdim=True) +  k.cumsum(2)
        z = 1 / ((q * k_state).sum(dim=-1))

        # Standard attention
        A_qk = torch.einsum("bhnd,bhmd->bhnm", q, k) 
        try:
            A_qk = torch.tril(A_qk)
        except:
            cumsum_matrix = torch.tril(torch.ones((seqlen, seqlen), device=A_qk.device))
            A_qk = A_qk * cumsum_matrix
        y1 = torch.einsum("bhnm,bhme->bhne", A_qk.to(q.dtype), v.to(q.dtype))
        y1 = y1 * z[..., None]
        output_1 = rearrange(y1, 'b h l d -> b l (h d)')

        # Cross attention
        A_qk_2 = torch.einsum("bhnd,bhmd->bhnm", q, k_enc)
        y2 = torch.einsum("bhnm,bhme->bhne", A_qk_2.to(q.dtype), v_enc.to(q.dtype))
        y2 = y2 * z[..., None]
        output_2 = rearrange(y2, 'b h l d -> b l (h d)')

        # Encoder attention
        output = output_1 + output_2
        output = self.proj_o(output)
        return output    
    
    def state_size(self, sequence_length: int=2048):
        return (
            # numerator
            self.num_key_value_heads * self.head_dim * self.feature_map.expanded_size() +
            # denominator 
            self.num_key_value_heads * self.feature_map.expanded_size()
        )



