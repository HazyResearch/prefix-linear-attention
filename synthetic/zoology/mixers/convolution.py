import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


def fftconv_ref(u, k, D, dropout_mask, gelu=True, k_rev=None, circular=False):
    # u.shape:   B H L
    seqlen = u.shape[-1]
    if circular:
        fft_size = seqlen
    else:
        fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)

    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u * D

    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)
    

class ShortConvolution(nn.Module):
    """
    Simple wrapper around nn.Conv1d that accepts dimension last. 
    """

    def __init__(
        self, 
        d_model: int,
        kernel_size: int,
        causal: bool = True,
        expand_dim: int = 1,
    ): 
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.causal = causal
        self.expand_dim = expand_dim

        if not self.causal:
            self.conv = nn.Conv1d(
                in_channels=d_model * self.expand_dim,
                out_channels=d_model * self.expand_dim,
                kernel_size=kernel_size,
                groups=d_model * self.expand_dim,
                padding=kernel_size-1, 
                padding_mode='circular',
            )
        else:
            self.conv = nn.Conv1d(
                in_channels=d_model * self.expand_dim,
                out_channels=d_model * self.expand_dim,
                kernel_size=kernel_size,
                groups=d_model * self.expand_dim,
                padding=kernel_size - 1,
            )

        if self.expand_dim:
            self.expand = nn.Linear(d_model, d_model * self.expand_dim)
            self.compress = nn.Linear(d_model * self.expand_dim, d_model)

    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (b, l, d) tensor
        Returns: 
            y: (b, l, d) tensor
        """
        l = x.size(1)
        if self.expand_dim:
            x = self.expand(x)

        y = self.conv(x.transpose(1, 2))[..., :l].transpose(1, 2)

        if self.expand_dim:
            y = self.compress(y)

        return y 


    def state_size(self, **kwargs):
        if self.causal:
            return self.d_model * self.kernel_size * self.expand_dim
        else:
            return self.d_model * self.kernel_size * self.expand_dim


class LongConvolution(nn.Module):
    """
    LongConvolution applies a convolution operation on the input tensor using a fixed 
    filter of length l_max.
    The filter is learned during training and is applied using FFT convolution.
    Args:
        d_model (int): The number of expected features in the input and output.
        l_max (int): The maximum sequence length.
    Returns:
        y: (b, l, d) tensor
    """
    def __init__(
        self,
        d_model: int,
        l_max: int,
        **kwargs,
    ):
        """
        Initializes the LongConvolution module.
        Args:
            d_model (int): The number of expected features in the input and output.
            l_max (int): The maximum sequence length.
        """
        super().__init__()
        self.d_model = d_model 
        self.filter = nn.Parameter(torch.randn(self.d_model, l_max), requires_grad=True)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Applies the LongConvolution operation on the input tensor.
        Args:
            x: (b, l, d) tensor
        Returns: 
            y: (b, l, d) tensor
        """
        x = x.transpose(1, 2)
        y = fft_conv(x, self.filter, dropout_mask=None, gelu=False)
        y = y.transpose(1, 2)
        return y.to(dtype=x.dtype)

    def state_size(self, sequence_length: int):
        return self.d_model * sequence_length


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, seq_len: int, **kwargs):
        """Complex exponential positional embeddings for implicit long convolution filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.z = nn.Parameter(z, requires_grad=False)

    def forward(self, L):
        return self.z[:, :L]


class ImplicitLongConvolution(nn.Module):
    """
    Long convolution with implicit filter parameterized by an MLP.

    Args:
        d_model (int): The number of expected features in the input and output.
        l_max (int): The maximum sequence length.
        d_emb (int, optional): The dimension of the positional embeddings. Must be odd and greater or equal to 3 (time, sine and cosine). Defaults to 3.
        d_hidden (int, optional): The number of features in the hidden layer of the MLP. Defaults to 16.

    Attributes:
        pos_emb (PositionalEmbedding): The positional embedding layer.
        mlp (nn.Sequential): The MLP that parameterizes the implicit filter.

    """

    
    def __init__(
        self,
        d_model: int,
        l_max: int,
        d_emb: int=3, 
        d_hidden: int = 16,
        causal: bool = True,
        **kwargs,
    ):
        """
        Long convolution with implicit filter parameterized by an MLP.

        
        """
        super().__init__()
        self.d_model = d_model 
        self.d_emb = d_emb 
        self.causal = causal

        assert (
            d_emb % 2 != 0 and d_emb >= 3
        ), "d_emb must be odd and greater or equal to 3 (time, sine and cosine)"
        self.pos_emb = PositionalEmbedding(d_emb, l_max)

        # final linear layer
        self.mlp = nn.Sequential(
            nn.Linear(d_emb, d_hidden),
            torch.nn.ReLU(),
            nn.Linear(d_hidden, d_model),
        )

        if not self.causal:
            self.mlp_rev = nn.Sequential(
                nn.Linear(d_emb, d_hidden),
                torch.nn.ReLU(),
                nn.Linear(d_hidden, d_model),
            )


    def filter(self, l: int, *args, **kwargs):
        k = self.mlp(self.pos_emb(l))
        return k.transpose(1, 2)
    

    def filter_rev(self, l: int, *args, **kwargs):
        k = self.mlp_rev(self.pos_emb(l))
        return k.transpose(1, 2)


    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Args:
            x: (b, l, d) tensor
        Returns: 
            y: (b, l, d) tensor
        """
        x = x.transpose(1, 2)

        if self.causal:
            k_rev = self.filter_rev(x.shape[-1])
            k_rev = rearrange(k_rev, "c l d -> c d l")[0] # `c` is always 1 by default
            y = self.filter_fn(x, k, dropout_mask=None, gelu=False, k_rev=k_rev)
        else:
            k_rev = None
            k = self.filter(x.shape[-1])
            y = fft_conv(x, k, dropout_mask=None, gelu=False)

        y = y.transpose(1, 2)
        return y.to(dtype=x.dtype)
    

    def state_size(self, sequence_length: int):
        if self.causal:
            return self.d_model * sequence_length
        else:
            assert 0, print(f"State size not implemented for non-causal convolution")
    
