# Modified from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import glob
import json
import math
import os
import types
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version, logging
from torch import nn

from ..dist import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    usp_attn_forward, xFuserLongContextAttention)
from ..utils.cfg_optimization import cfg_skip
from .attention_utils import attention
from .cache_utils import TeaCache
from .wan_camera_adapter import SimpleAdapter


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# modified from https://github.com/thu-ml/RIFLEx/blob/main/riflex_utils.py
@amp.autocast(enabled=False)
def get_1d_rotary_pos_embed_riflex(
    pos: Union[np.ndarray, int],
    dim: int,
    theta: float = 10000.0,
    use_real=False,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
    L_test_scale: Optional[int] = None,
):
    """
    RIFLEx: Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        k (`int`, *optional*, defaults to None): the index for the intrinsic frequency in RoPE
        L_test (`int`, *optional*, defaults to None): the number of frames for inference
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    freqs = 1.0 / torch.pow(theta,
        torch.arange(0, dim, 2).to(torch.float64).div(dim))

    # === Riflex modification start ===
    # Reduce the intrinsic frequency to stay within a single period after extrapolation (see Eq. (8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply by 0.9 to keep the extrapolated length below 90% of a single period.
    if k is not None:
        freqs[k-1] = 0.9 * 2 * torch.pi / L_test
    # === Riflex modification end ===
    if L_test_scale is not None:
        freqs[k-1] = freqs[k-1] / L_test_scale

    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


@amp.autocast(enabled=False)
@torch.compiler.disable()
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


def rope_apply_qk(q, k, grid_sizes, freqs):
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)
    return q, k


@amp.autocast(enabled=False)
@torch.compiler.disable()
def rope_apply_1d(x, seq_lens, freqs):
    n = x.size(2)
    output = []
    for i, seq_len in enumerate(seq_lens.tolist()):
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float32).reshape(seq_len, n, -1, 2))
        freqs_i = freqs[:seq_len].view(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


def rope_apply_qk_1d(q, k, seq_lens, freqs):
    q = rope_apply_1d(q, seq_lens, freqs)
    k = rope_apply_1d(k, seq_lens, freqs)
    return q, k


def build_branch_layer_pairing(num_layers, branch_layers, strategy="dilated", power=1.5):
    if branch_layers < 0:
        raise ValueError("branch_layers must be non-negative")
    if branch_layers == 0:
        return []
    if branch_layers > num_layers:
        raise ValueError(
            f"branch_layers ({branch_layers}) must be <= num_layers ({num_layers})")
    if branch_layers == 1:
        return [0]

    if strategy == "first":
        raw = torch.arange(branch_layers, dtype=torch.float64)
    elif strategy == "uniform":
        raw = torch.linspace(0, num_layers - 1, branch_layers,
                             dtype=torch.float64)
    elif strategy == "dilated":
        raw = torch.linspace(0, 1, branch_layers, dtype=torch.float64)
        raw = raw.pow(power) * (num_layers - 1)
    else:
        raise ValueError(
            f"Unsupported branch pairing strategy: {strategy}")

    pairing = []
    last = -1
    for idx, value in enumerate(raw.round().long().tolist()):
        min_value = last + 1
        max_value = num_layers - (branch_layers - idx)
        value = min(max(value, min_value), max_value)
        pairing.append(value)
        last = value
    return pairing


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def project_qkv(self, x, dtype=torch.bfloat16):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
        k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
        v = self.v(x.to(dtype)).view(b, s, n, d)
        return q, k, v

    def apply_rope(self, q, k, seq_lens, grid_sizes, freqs):
        return rope_apply_qk(q, k, grid_sizes, freqs)

    def project_output(self, x):
        x = x.flatten(2)
        x = self.o(x)
        return x

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        q, k, v = self.project_qkv(x, dtype)
        q, k = self.apply_rope(q, k, seq_lens, grid_sizes, freqs)

        x = attention(
            q.to(dtype), 
            k.to(dtype), 
            v=v.to(dtype),
            k_lens=seq_lens,
            window_size=self.window_size)
        x = x.to(dtype)

        # output
        x = self.project_output(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)

        # compute attention
        x = attention(
            q.to(dtype), 
            k.to(dtype), 
            v.to(dtype), 
            k_lens=context_lens
        )
        x = x.to(dtype)

        # output
        x = self.project_output(x)
        return x


class WanSimulationSelfAttention(WanSelfAttention):

    def apply_rope(self, q, k, seq_lens, grid_sizes, freqs):
        return rope_apply_qk_1d(q, k, seq_lens, freqs)

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, t=0):
        q, k, v = self.project_qkv(x, dtype)
        q, k = self.apply_rope(q, k, seq_lens, grid_sizes, freqs)

        x = attention(
            q.to(dtype),
            k.to(dtype),
            v=v.to(dtype),
            q_lens=seq_lens,
            k_lens=seq_lens,
            window_size=self.window_size,
        )
        x = x.to(dtype)
        x = self.project_output(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img.to(dtype))).view(b, -1, n, d)
        v_img = self.v_img(context_img.to(dtype)).view(b, -1, n, d)

        img_x = attention(
            q.to(dtype), 
            k_img.to(dtype), 
            v_img.to(dtype), 
            k_lens=None
        )
        img_x = img_x.to(dtype)
        # compute attention
        x = attention(
            q.to(dtype), 
            k.to(dtype), 
            v.to(dtype), 
            k_lens=context_lens
        )
        x = x.to(dtype)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        # compute query, key, value
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)
        # compute attention
        x = attention(q.to(dtype), k.to(dtype), v.to(dtype), k_lens=context_lens)
        # output
        x = x.flatten(2)
        x = self.o(x.to(dtype))
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
    'cross_attn': WanCrossAttention,
}


def _chunk_modulation(modulation, e):
    if e.dim() > 3:
        e = (modulation.unsqueeze(0) + e).chunk(6, dim=2)
        e = [element.squeeze(2) for element in e]
    else:
        e = (modulation + e).chunk(6, dim=1)
    return e


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def get_modulation(self, e):
        return _chunk_modulation(self.modulation, e)

    def self_attention(self, x, e, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, t=0):
        e = self.get_modulation(e)
        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)
        y = self.self_attn(temp_x, seq_lens, grid_sizes, freqs, dtype, t=t)
        x = x + y * e[2]
        return x, e

    def cross_attn_ffn(self, x, context, context_lens, e, dtype=torch.bfloat16, t=0):
        x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype, t=t)

        temp_x = self.norm2(x) * (1 + e[4]) + e[3]
        temp_x = temp_x.to(dtype)

        y = self.ffn(temp_x)
        x = x + y * e[5]
        return x

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dtype=torch.bfloat16,
        t=0,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        x, e = self.self_attention(
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            dtype=dtype,
            t=t,
        )
        x = self.cross_attn_ffn(
            x,
            context,
            context_lens,
            e,
            dtype=dtype,
            t=t,
        )
        return x


class WanSimulationAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 qk_norm=True,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSimulationSelfAttention(dim, num_heads, (-1, -1),
                                                    qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def get_modulation(self, e):
        return _chunk_modulation(self.modulation, e)

    def self_attention(self, x, e, seq_lens, freqs, dtype=torch.bfloat16, t=0):
        e = self.get_modulation(e)
        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)
        y = self.self_attn(temp_x, seq_lens, None, freqs, dtype, t=t)
        x = x + y * e[2]
        return x, e

    def ffn_only(self, x, e, dtype=torch.bfloat16):
        temp_x = self.norm2(x) * (1 + e[4]) + e[3]
        temp_x = temp_x.to(dtype)
        y = self.ffn(temp_x)
        x = x + y * e[5]
        return x

    def forward(self, x, e, seq_lens, freqs, dtype=torch.bfloat16, t=0):
        x, e = self.self_attention(x, e, seq_lens, freqs, dtype=dtype, t=t)
        x = self.ffn_only(x, e, dtype=dtype)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        if e.dim() > 2:
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            e = [e.squeeze(2) for e in e]
        else:
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class SimulationHead(nn.Module):

    def __init__(self, dim, out_dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.eps = eps

        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        if e.dim() > 2:
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            e = [element.squeeze(2) for element in e]
        else:
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)

        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens



class WanTransformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    # ignore_for_config = [
    #     'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    # ]
    # _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        downscale_factor_control_adapter=8,
        add_ref_conv=False,
        in_dim_ref_conv=16,
        cross_attn_type=None,
        add_simulation_branch=False,
        simulation_state_dim=6,
        simulation_cond_dim=0,
        simulation_out_dim=None,
        simulation_num_layers=8,
        simulation_pairing=None,
        simulation_pairing_strategy="dilated",
        simulation_pairing_power=1.5,
        simulation_max_seq_len=4096,
        simulation_separate_time_embedding=True,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        # assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.add_simulation_branch = add_simulation_branch
        self.simulation_state_dim = simulation_state_dim
        self.simulation_cond_dim = simulation_cond_dim
        self.simulation_num_layers = simulation_num_layers
        self.simulation_pairing_strategy = simulation_pairing_strategy
        self.simulation_pairing_power = simulation_pairing_power
        self.simulation_max_seq_len = simulation_max_seq_len
        self.simulation_separate_time_embedding = simulation_separate_time_embedding

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        if cross_attn_type is None:
            cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.d = d
        self.dim = dim
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1
        )

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:], downscale_factor=downscale_factor_control_adapter)
        else:
            self.control_adapter = None

        if add_ref_conv:
            self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.ref_conv = None

        self.simulation_input_embedding = None
        self.simulation_time_embedding = None
        self.simulation_time_projection = None
        self.simulation_blocks = nn.ModuleList()
        self.simulation_head = None
        self.simulation_freqs = None
        self.simulation_layer_mapping = {}

        if add_simulation_branch:
            if simulation_num_layers > num_layers:
                raise ValueError(
                    f"simulation_num_layers ({simulation_num_layers}) must be <= num_layers ({num_layers})")
            self.simulation_out_dim = simulation_state_dim if simulation_out_dim is None else simulation_out_dim
            self.simulation_input_dim = simulation_state_dim + simulation_cond_dim
            self.simulation_input_embedding = nn.Sequential(
                nn.LayerNorm(self.simulation_input_dim),
                nn.Linear(self.simulation_input_dim, dim),
                nn.GELU(approximate='tanh'),
                nn.Linear(dim, dim),
            )
            if simulation_separate_time_embedding:
                self.simulation_time_embedding = nn.Sequential(
                    nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
                self.simulation_time_projection = nn.Sequential(
                    nn.SiLU(), nn.Linear(dim, dim * 6))

            self.simulation_blocks = nn.ModuleList([
                WanSimulationAttentionBlock(dim, ffn_dim, num_heads, qk_norm,
                                            eps)
                for _ in range(simulation_num_layers)
            ])
            if simulation_pairing is None:
                simulation_pairing = build_branch_layer_pairing(
                    num_layers,
                    simulation_num_layers,
                    strategy=simulation_pairing_strategy,
                    power=simulation_pairing_power,
                )
            else:
                simulation_pairing = [int(layer_idx)
                                      for layer_idx in simulation_pairing]
                if len(simulation_pairing) != simulation_num_layers:
                    raise ValueError(
                        "simulation_pairing length must equal simulation_num_layers")
                if sorted(simulation_pairing) != simulation_pairing:
                    raise ValueError(
                        "simulation_pairing must be sorted in ascending order")
                if len(set(simulation_pairing)) != len(simulation_pairing):
                    raise ValueError(
                        "simulation_pairing must contain unique layer indices")
                if simulation_pairing[0] < 0 or simulation_pairing[-1] >= num_layers:
                    raise ValueError(
                        "simulation_pairing values must be within [0, num_layers)")

            self.simulation_pairing = simulation_pairing
            self.simulation_layer_mapping = {
                video_idx: sim_idx
                for sim_idx, video_idx in enumerate(simulation_pairing)
            }
            self.simulation_head = SimulationHead(dim, self.simulation_out_dim,
                                                  eps)
            self.simulation_freqs = rope_params(
                max(2, simulation_max_seq_len), d)
        else:
            self.simulation_out_dim = simulation_state_dim if simulation_out_dim is None else simulation_out_dim
            self.simulation_input_dim = simulation_state_dim + simulation_cond_dim
            self.simulation_pairing = []

        self.teacache = None
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None
        self.gradient_checkpointing = False
        self.all_gather = None
        self.sp_world_size = 1
        self.sp_world_rank = 0
        self.init_weights()

    def _set_gradient_checkpointing(self, *args, **kwargs):
        if "value" in kwargs:
            self.gradient_checkpointing = kwargs["value"]
            if hasattr(self, "motioner") and hasattr(self.motioner, "gradient_checkpointing"):
                self.motioner.gradient_checkpointing = kwargs["value"]
        elif "enable" in kwargs:
            self.gradient_checkpointing = kwargs["enable"]
            if hasattr(self, "motioner") and hasattr(self.motioner, "gradient_checkpointing"):
                self.motioner.gradient_checkpointing = kwargs["enable"]
        else:
            raise ValueError("Invalid set gradient checkpointing")

    def enable_teacache(
        self,
        coefficients,
        num_steps: int,
        rel_l1_thresh: float,
        num_skip_start_steps: int = 0,
        offload: bool = True,
    ):
        self.teacache = TeaCache(
            coefficients, num_steps, rel_l1_thresh=rel_l1_thresh, num_skip_start_steps=num_skip_start_steps, offload=offload
        )

    def share_teacache(
        self,
        transformer = None,
    ):
        self.teacache = transformer.teacache

    def disable_teacache(self):
        self.teacache = None

    def enable_cfg_skip(self, cfg_skip_ratio, num_steps):
        if cfg_skip_ratio != 0:
            self.cfg_skip_ratio = cfg_skip_ratio
            self.current_steps = 0
            self.num_inference_steps = num_steps
        else:
            self.cfg_skip_ratio = None
            self.current_steps = 0
            self.num_inference_steps = None

    def share_cfg_skip(
        self,
        transformer = None,
    ):
        self.cfg_skip_ratio = transformer.cfg_skip_ratio
        self.current_steps = transformer.current_steps
        self.num_inference_steps = transformer.num_inference_steps

    def disable_cfg_skip(self):
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None

    def enable_riflex(
        self,
        k = 6,
        L_test = 66,
        L_test_scale = 4.886,
    ):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                get_1d_rotary_pos_embed_riflex(1024, self.d - 4 * (self.d // 6), use_real=False, k=k, L_test=L_test, L_test_scale=L_test_scale),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6))
            ],
            dim=1
        ).to(device)

    def disable_riflex(self):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                rope_params(1024, self.d - 4 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6))
            ],
            dim=1
        ).to(device)

    def enable_multi_gpus_inference(self,):
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        self.all_gather = get_sp_group().all_gather

        # For normal model.
        for block in self.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn)

        # For vace model.
        if hasattr(self, 'vace_blocks'):
            for block in self.vace_blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)

    def _ensure_simulation_freqs(self, seq_len, device):
        if self.simulation_freqs is None:
            return None
        if self.simulation_freqs.device != device and torch.device(type="meta") != device:
            self.simulation_freqs = self.simulation_freqs.to(device)
        if seq_len > self.simulation_freqs.size(0):
            self.simulation_freqs = rope_params(seq_len, self.d).to(device)
        return self.simulation_freqs

    def _concat_simulation_features(self, state, cond=None):
        if state.size(-1) != self.simulation_state_dim:
            raise ValueError(
                f"Expected simulation state dim {self.simulation_state_dim}, got {state.size(-1)}")

        if cond is None and self.simulation_cond_dim > 0:
            cond = state.new_zeros(*state.shape[:-1], self.simulation_cond_dim)

        if cond is not None:
            if cond.size(-1) != self.simulation_cond_dim:
                raise ValueError(
                    f"Expected simulation cond dim {self.simulation_cond_dim}, got {cond.size(-1)}")
            state = torch.cat([state, cond], dim=-1)

        if state.size(-1) != self.simulation_input_dim:
            raise ValueError(
                f"Expected simulation input dim {self.simulation_input_dim}, got {state.size(-1)}")
        return state

    def _prepare_simulation_inputs(self, simulation_states, simulation_cond=None):
        if self.simulation_input_embedding is None:
            raise ValueError("Simulation branch is not initialized")

        simulation_is_tensor = isinstance(simulation_states, torch.Tensor)
        simulation_was_squeezed = False

        if simulation_is_tensor:
            if simulation_states.dim() == 3:
                simulation_states = simulation_states.unsqueeze(0)
                simulation_was_squeezed = True
            if simulation_states.dim() != 4:
                raise ValueError(
                    "simulation_states must have shape [B, T, N, C] or [T, N, C]")

            if simulation_cond is not None:
                if not isinstance(simulation_cond, torch.Tensor):
                    raise ValueError(
                        "simulation_cond must match the container type of simulation_states")
                if simulation_cond.dim() == 3:
                    simulation_cond = simulation_cond.unsqueeze(0)
                if simulation_cond.shape[:-1] != simulation_states.shape[:-1]:
                    raise ValueError(
                        "simulation_cond must match simulation_states on [B, T, N]")

            simulation_inputs = self._concat_simulation_features(
                simulation_states, simulation_cond)
            batch_size, steps, num_points, _ = simulation_inputs.shape
            simulation_hidden = self.simulation_input_embedding(
                simulation_inputs).reshape(batch_size, steps * num_points,
                                           self.dim)
            simulation_seq_lens = torch.full(
                (batch_size,),
                steps * num_points,
                dtype=torch.long,
                device=simulation_hidden.device,
            )
            simulation_shapes = [(int(steps), int(num_points))] * batch_size
            return (
                simulation_hidden,
                simulation_seq_lens,
                simulation_shapes,
                simulation_is_tensor,
                simulation_was_squeezed,
            )

        if not isinstance(simulation_states, (list, tuple)):
            raise ValueError(
                "simulation_states must be a tensor or a list/tuple of tensors")

        if simulation_cond is not None and not isinstance(simulation_cond,
                                                          (list, tuple)):
            raise ValueError(
                "simulation_cond must be a tensor when simulation_states is a tensor, or a list/tuple otherwise")

        simulation_hidden = []
        simulation_seq_lens = []
        simulation_shapes = []

        for idx, state in enumerate(simulation_states):
            if state.dim() != 3:
                raise ValueError(
                    "Each simulation state tensor must have shape [T, N, C]")
            cond = None if simulation_cond is None else simulation_cond[idx]
            if cond is not None and cond.shape[:-1] != state.shape[:-1]:
                raise ValueError(
                    "Each simulation_cond tensor must match the corresponding simulation_states tensor on [T, N]")

            state = self._concat_simulation_features(state, cond)
            steps, num_points, _ = state.shape
            hidden = self.simulation_input_embedding(state).reshape(-1,
                                                                    self.dim)
            simulation_hidden.append(hidden)
            simulation_seq_lens.append(hidden.size(0))
            simulation_shapes.append((int(steps), int(num_points)))

        max_simulation_seq_len = max(simulation_seq_lens)
        simulation_hidden = torch.stack([
            torch.cat([
                hidden,
                hidden.new_zeros(max_simulation_seq_len - hidden.size(0),
                                 hidden.size(1))
            ],
                      dim=0) for hidden in simulation_hidden
        ])
        simulation_seq_lens = torch.tensor(
            simulation_seq_lens,
            dtype=torch.long,
            device=simulation_hidden.device,
        )
        return (
            simulation_hidden,
            simulation_seq_lens,
            simulation_shapes,
            simulation_is_tensor,
            simulation_was_squeezed,
        )

    def _build_simulation_time_embeddings(self, t, simulation_t=None):
        if simulation_t is None:
            simulation_t = t if t.dim() == 1 else t[:, -1]
        elif simulation_t.dim() != 1:
            simulation_t = simulation_t[:, -1]

        time_embedding = self.time_embedding
        time_projection = self.time_projection
        if self.simulation_time_embedding is not None:
            time_embedding = self.simulation_time_embedding
            time_projection = self.simulation_time_projection

        with amp.autocast(dtype=torch.float32):
            simulation_e = time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, simulation_t).float())
            simulation_e0 = time_projection(simulation_e).unflatten(
                1, (6, self.dim))

        return simulation_e, simulation_e0

    def _joint_attention(self, video_attn, simulation_attn, video_x,
                         simulation_x, seq_lens, simulation_seq_lens,
                         grid_sizes, simulation_freqs, dtype):
        video_q, video_k, video_v = video_attn.project_qkv(video_x, dtype)
        simulation_q, simulation_k, simulation_v = simulation_attn.project_qkv(
            simulation_x, dtype)

        video_q, video_k = video_attn.apply_rope(video_q, video_k, seq_lens,
                                                 grid_sizes, self.freqs)
        simulation_q, simulation_k = simulation_attn.apply_rope(
            simulation_q, simulation_k, simulation_seq_lens, None,
            simulation_freqs)

        joint_q = torch.cat([video_q, simulation_q], dim=1)
        joint_k = torch.cat([video_k, simulation_k], dim=1)
        joint_v = torch.cat([video_v, simulation_v], dim=1)
        joint_seq_lens = seq_lens.to(joint_q.device) + simulation_seq_lens.to(
            joint_q.device)

        joint_hidden = attention(
            joint_q.to(dtype),
            joint_k.to(dtype),
            joint_v.to(dtype),
            q_lens=joint_seq_lens,
            k_lens=joint_seq_lens,
            window_size=(-1, -1),
        )
        joint_hidden = joint_hidden.to(dtype)

        video_hidden = joint_hidden[:, :video_x.size(1)]
        simulation_hidden = joint_hidden[:, video_x.size(1):]

        video_hidden = video_attn.project_output(video_hidden)
        simulation_hidden = simulation_attn.project_output(simulation_hidden)
        return video_hidden, simulation_hidden

    def _forward_joint_branch_block(
        self,
        block,
        simulation_block,
        x,
        simulation_hidden,
        e0,
        simulation_e0,
        seq_lens,
        simulation_seq_lens,
        grid_sizes,
        context,
        context_lens,
        simulation_freqs,
        dtype=torch.bfloat16,
        t=0,
    ):
        video_e = block.get_modulation(e0)
        simulation_e = simulation_block.get_modulation(simulation_e0)

        video_input = block.norm1(x) * (1 + video_e[1]) + video_e[0]
        simulation_input = simulation_block.norm1(simulation_hidden) * (
            1 + simulation_e[1]) + simulation_e[0]

        video_input = video_input.to(dtype)
        simulation_input = simulation_input.to(dtype)

        video_y, simulation_y = self._joint_attention(
            block.self_attn,
            simulation_block.self_attn,
            video_input,
            simulation_input,
            seq_lens,
            simulation_seq_lens,
            grid_sizes,
            simulation_freqs,
            dtype,
        )

        x = x + video_y * video_e[2]
        simulation_hidden = simulation_hidden + simulation_y * simulation_e[2]

        x = block.cross_attn_ffn(
            x,
            context,
            context_lens,
            video_e,
            dtype=dtype,
            t=t,
        )
        simulation_hidden = simulation_block.ffn_only(
            simulation_hidden, simulation_e, dtype=dtype)
        return x, simulation_hidden

    def _forward_transformer_blocks(
        self,
        x,
        e0,
        seq_lens,
        grid_sizes,
        context,
        context_lens,
        dtype,
        t,
        simulation_hidden=None,
        simulation_e0=None,
        simulation_seq_lens=None,
        simulation_freqs=None,
    ):
        use_simulation_branch = simulation_hidden is not None
        block_kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            dtype=dtype,
            t=t,
        )

        for layer_idx, block in enumerate(self.blocks):
            simulation_idx = self.simulation_layer_mapping.get(
                layer_idx) if use_simulation_branch else None

            if simulation_idx is None:
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {
                        "use_reentrant": False
                    } if is_torch_version(">=", "1.11.0") else {}
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        e0,
                        seq_lens,
                        grid_sizes,
                        self.freqs,
                        context,
                        context_lens,
                        dtype,
                        t,
                        **ckpt_kwargs,
                    )
                else:
                    x = block(x, **block_kwargs)
                continue

            simulation_block = self.simulation_blocks[simulation_idx]
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_joint_forward(video_block, branch_block,
                                         **static_kwargs):
                    def custom_forward(video_x, branch_x, video_e, branch_e):
                        return self._forward_joint_branch_block(
                            video_block,
                            branch_block,
                            video_x,
                            branch_x,
                            video_e,
                            branch_e,
                            **static_kwargs,
                        )

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {
                    "use_reentrant": False
                } if is_torch_version(">=", "1.11.0") else {}
                x, simulation_hidden = torch.utils.checkpoint.checkpoint(
                    create_joint_forward(
                        block,
                        simulation_block,
                        seq_lens=seq_lens,
                        simulation_seq_lens=simulation_seq_lens,
                        grid_sizes=grid_sizes,
                        context=context,
                        context_lens=context_lens,
                        simulation_freqs=simulation_freqs,
                        dtype=dtype,
                        t=t,
                    ),
                    x,
                    simulation_hidden,
                    e0,
                    simulation_e0,
                    **ckpt_kwargs,
                )
            else:
                x, simulation_hidden = self._forward_joint_branch_block(
                    block,
                    simulation_block,
                    x,
                    simulation_hidden,
                    e0,
                    simulation_e0,
                    seq_lens,
                    simulation_seq_lens,
                    grid_sizes,
                    context,
                    context_lens,
                    simulation_freqs,
                    dtype=dtype,
                    t=t,
                )

        return x, simulation_hidden

    def _unpack_simulation_tokens(self, simulation_hidden, simulation_shapes,
                                  simulation_is_tensor,
                                  simulation_was_squeezed):
        outputs = []
        for hidden, (steps, num_points) in zip(simulation_hidden,
                                               simulation_shapes):
            outputs.append(hidden[:steps * num_points].view(
                steps, num_points, -1))

        if simulation_is_tensor:
            outputs = torch.stack(outputs)
            if simulation_was_squeezed:
                outputs = outputs[0]
            return outputs
        return outputs

    @cfg_skip()
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        y_camera=None,
        full_ref=None,
        subject_ref=None,
        cond_flag=True,
        simulation_states=None,
        simulation_cond=None,
        simulation_t=None,
        return_simulation=False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            cond_flag (`bool`, *optional*, defaults to True):
                Flag to indicate whether to forward the condition input
            simulation_states (Tensor or List[Tensor], *optional*):
                Noisy simulation states with shape [B, T, N, C_state] or per-sample [T, N, C_state]
            simulation_cond (Tensor or List[Tensor], *optional*):
                Per-point simulation conditions concatenated with `simulation_states`
            simulation_t (Tensor, *optional*):
                Optional diffusion timestep override for the simulation branch
            return_simulation (`bool`, *optional*, defaults to False):
                If True, also return the denoised simulation states

        Returns:
            Tensor or Tuple[Tensor, Tensor/List[Tensor]]:
                Denoised video latents, and optionally denoised simulation states
        """
        # Wan2.2 don't need a clip.
        # if self.model_type == 'i2v':
        #     assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if isinstance(x, torch.Tensor):
            dtype = x.dtype
        elif len(x) > 0:
            dtype = x[0].dtype
        else:
            raise ValueError("x must be a tensor or a non-empty list of tensors")
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        use_simulation_branch = simulation_states is not None
        if use_simulation_branch and not self.add_simulation_branch:
            raise ValueError(
                "simulation_states were provided but add_simulation_branch is disabled in the model config")
        if use_simulation_branch and self.sp_world_size > 1:
            raise NotImplementedError(
                "Simulation branch does not yet support sequence-parallel inference")

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # add control adapter
        if self.control_adapter is not None and y_camera is not None:
            y_camera = self.control_adapter(y_camera)
            x = [u + v for u, v in zip(x, y_camera)]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])

        x = [u.flatten(2).transpose(1, 2) for u in x]
        if self.ref_conv is not None and full_ref is not None:
            full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
            grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            seq_len += full_ref.size(1)
            x = [torch.concat([_full_ref.unsqueeze(0), u], dim=1) for _full_ref, u in zip(full_ref, x)]
            if t.dim() != 1 and t.size(1) < seq_len:
                pad_size = seq_len - t.size(1)
                last_elements = t[:, -1].unsqueeze(1)
                padding = last_elements.repeat(1, pad_size)
                t = torch.cat([padding, t], dim=1)

        if subject_ref is not None:
            subject_ref_frames = subject_ref.size(2)
            subject_ref = self.patch_embedding(subject_ref).flatten(2).transpose(1, 2)
            grid_sizes = torch.stack([torch.tensor([u[0] + subject_ref_frames, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            seq_len += subject_ref.size(1)
            x = [torch.concat([u, _subject_ref.unsqueeze(0)], dim=1) for _subject_ref, u in zip(subject_ref, x)]
            if t.dim() != 1 and t.size(1) < seq_len:
                pad_size = seq_len - t.size(1)
                last_elements = t[:, -1].unsqueeze(1)
                padding = last_elements.repeat(1, pad_size)
                t = torch.cat([t, padding], dim=1)
        
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        simulation_hidden = None
        simulation_e = None
        simulation_e0 = None
        simulation_seq_lens = None
        simulation_shapes = None
        simulation_is_tensor = False
        simulation_was_squeezed = False
        simulation_freqs = None

        if use_simulation_branch:
            (
                simulation_hidden,
                simulation_seq_lens,
                simulation_shapes,
                simulation_is_tensor,
                simulation_was_squeezed,
            ) = self._prepare_simulation_inputs(
                simulation_states, simulation_cond)

            if simulation_hidden.size(0) != x.size(0):
                if x.size(0) % simulation_hidden.size(0) != 0:
                    raise ValueError(
                        "simulation_states batch size must either match x batch size or divide it exactly")
                repeat_factor = x.size(0) // simulation_hidden.size(0)
                simulation_hidden = torch.cat(
                    [simulation_hidden] * repeat_factor, dim=0)
                simulation_seq_lens = simulation_seq_lens.repeat(
                    repeat_factor)
                simulation_shapes = simulation_shapes * repeat_factor
                simulation_was_squeezed = False

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            if t.dim() != 1:
                if t.size(1) < seq_len:
                    pad_size = seq_len - t.size(1)
                    last_elements = t[:, -1].unsqueeze(1)
                    padding = last_elements.repeat(1, pad_size)
                    t = torch.cat([t, padding], dim=1)
                bt = t.size(0)
                ft = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            ft).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            else:
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))

            # assert e.dtype == torch.float32 and e0.dtype == torch.float32
            # e0 = e0.to(dtype)
            # e = e.to(dtype)

        if use_simulation_branch:
            simulation_e, simulation_e0 = self._build_simulation_time_embeddings(
                t, simulation_t)
            simulation_freqs = self._ensure_simulation_freqs(
                int(simulation_seq_lens.max().item()), simulation_hidden.device)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            if t.dim() != 1:
                e0 = torch.chunk(e0, self.sp_world_size, dim=1)[self.sp_world_rank]
                e = torch.chunk(e, self.sp_world_size, dim=1)[self.sp_world_rank]

        use_teacache = self.teacache is not None and not use_simulation_branch

        # TeaCache
        if use_teacache:
            if cond_flag:
                if t.dim() != 1:
                    modulated_inp = e0[:, -1, :]
                else:
                    modulated_inp = e0
                skip_flag = self.teacache.cnt < self.teacache.num_skip_start_steps
                if skip_flag:
                    self.should_calc = True
                    self.teacache.accumulated_rel_l1_distance = 0
                else:
                    if cond_flag:
                        rel_l1_distance = self.teacache.compute_rel_l1_distance(self.teacache.previous_modulated_input, modulated_inp)
                        self.teacache.accumulated_rel_l1_distance += self.teacache.rescale_func(rel_l1_distance)
                    if self.teacache.accumulated_rel_l1_distance < self.teacache.rel_l1_thresh:
                        self.should_calc = False
                    else:
                        self.should_calc = True
                        self.teacache.accumulated_rel_l1_distance = 0
                self.teacache.previous_modulated_input = modulated_inp
                self.teacache.should_calc = self.should_calc
            else:
                self.should_calc = self.teacache.should_calc
        
        # TeaCache
        if use_teacache:
            if not self.should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()
                x, simulation_hidden = self._forward_transformer_blocks(
                    x,
                    e0,
                    seq_lens,
                    grid_sizes,
                    context,
                    context_lens,
                    dtype,
                    t,
                    simulation_hidden=simulation_hidden,
                    simulation_e0=simulation_e0,
                    simulation_seq_lens=simulation_seq_lens,
                    simulation_freqs=simulation_freqs,
                )
                    
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            x, simulation_hidden = self._forward_transformer_blocks(
                x,
                e0,
                seq_lens,
                grid_sizes,
                context,
                context_lens,
                dtype,
                t,
                simulation_hidden=simulation_hidden,
                simulation_e0=simulation_e0,
                simulation_seq_lens=simulation_seq_lens,
                simulation_freqs=simulation_freqs,
            )

        # head
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.head), x, e, **ckpt_kwargs)
        else:
            x = self.head(x, e)

        simulation_output = None
        if use_simulation_branch:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {
                    "use_reentrant": False
                } if is_torch_version(">=", "1.11.0") else {}
                simulation_hidden = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.simulation_head),
                    simulation_hidden,
                    simulation_e,
                    **ckpt_kwargs,
                )
            else:
                simulation_hidden = self.simulation_head(
                    simulation_hidden, simulation_e)
            simulation_output = self._unpack_simulation_tokens(
                simulation_hidden,
                simulation_shapes,
                simulation_is_tensor,
                simulation_was_squeezed,
            )

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        if self.ref_conv is not None and full_ref is not None:
            full_ref_length = full_ref.size(1)
            x = x[:, full_ref_length:]
            grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        if subject_ref is not None:
            subject_ref_length = subject_ref.size(1)
            x = x[:, :-subject_ref_length]
            grid_sizes = torch.stack([torch.tensor([u[0] - subject_ref_frames, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if use_teacache and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        if return_simulation:
            return x, simulation_output
        return x


    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        if self.simulation_time_embedding is not None:
            for m in self.simulation_time_embedding.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
        if self.simulation_head is not None:
            nn.init.zeros_(self.simulation_head.head.weight)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={},
        low_cpu_mem_usage=False, torch_dtype=torch.bfloat16
    ):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            for key in transformer_additional_kwargs["dict_mapping"]:
                transformer_additional_kwargs[transformer_additional_kwargs["dict_mapping"][key]] = config[key]

        if low_cpu_mem_usage:
            try:
                import re

                from diffusers import __version__ as diffusers_version
                if diffusers_version >= "0.33.0":
                    from diffusers.models.model_loading_utils import \
                        load_model_dict_into_meta
                else:
                    from diffusers.models.modeling_utils import \
                        load_model_dict_into_meta
                from diffusers.utils import is_accelerate_available
                if is_accelerate_available():
                    import accelerate
                
                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **transformer_additional_kwargs)

                param_device = "cpu"
                if os.path.exists(model_file):
                    state_dict = torch.load(model_file, map_location="cpu")
                elif os.path.exists(model_file_safetensors):
                    from safetensors.torch import load_file, safe_open
                    state_dict = load_file(model_file_safetensors)
                else:
                    from safetensors.torch import load_file, safe_open
                    model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
                    state_dict = {}
                    print(model_files_safetensors)
                    for _model_file_safetensors in model_files_safetensors:
                        _state_dict = load_file(_model_file_safetensors)
                        for key in _state_dict:
                            state_dict[key] = _state_dict[key]

                if model.state_dict()['patch_embedding.weight'].size() != state_dict['patch_embedding.weight'].size():
                    model.state_dict()['patch_embedding.weight'][:, :state_dict['patch_embedding.weight'].size()[1], :, :] = state_dict['patch_embedding.weight'][:, :model.state_dict()['patch_embedding.weight'].size()[1], :, :]
                    model.state_dict()['patch_embedding.weight'][:, state_dict['patch_embedding.weight'].size()[1]:, :, :] = 0
                    state_dict['patch_embedding.weight'] = model.state_dict()['patch_embedding.weight']

                filtered_state_dict = {}
                for key in state_dict:
                    if key in model.state_dict() and model.state_dict()[key].size() == state_dict[key].size():
                        filtered_state_dict[key] = state_dict[key]
                    else:
                        print(f"Skipping key '{key}' due to size mismatch or absence in model.")
                        
                model_keys = set(model.state_dict().keys())
                loaded_keys = set(filtered_state_dict.keys())
                missing_keys = model_keys - loaded_keys

                def initialize_missing_parameters(missing_keys, model_state_dict, torch_dtype=None):
                    initialized_dict = {}
                    
                    with torch.no_grad():
                        for key in missing_keys:
                            param_shape = model_state_dict[key].shape
                            param_dtype = torch_dtype if torch_dtype is not None else model_state_dict[key].dtype
                            if 'weight' in key:
                                if any(norm_type in key for norm_type in ['norm', 'ln_', 'layer_norm', 'group_norm', 'batch_norm']):
                                    initialized_dict[key] = torch.ones(param_shape, dtype=param_dtype)
                                elif 'embedding' in key or 'embed' in key:
                                    initialized_dict[key] = torch.randn(param_shape, dtype=param_dtype) * 0.02
                                elif 'head' in key or 'output' in key or 'proj_out' in key:
                                    initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                                elif len(param_shape) >= 2:
                                    initialized_dict[key] = torch.empty(param_shape, dtype=param_dtype)
                                    nn.init.xavier_uniform_(initialized_dict[key])
                                else:
                                    initialized_dict[key] = torch.randn(param_shape, dtype=param_dtype) * 0.02
                            elif 'bias' in key:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                            elif 'running_mean' in key:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                            elif 'running_var' in key:
                                initialized_dict[key] = torch.ones(param_shape, dtype=param_dtype)
                            elif 'num_batches_tracked' in key:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=torch.long)
                            else:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                            
                    return initialized_dict

                if missing_keys:
                    print(f"Missing keys will be initialized: {sorted(missing_keys)}")
                    initialized_params = initialize_missing_parameters(
                        missing_keys, 
                        model.state_dict(), 
                        torch_dtype
                    )
                    filtered_state_dict.update(initialized_params)

                if diffusers_version >= "0.33.0":
                    # Diffusers has refactored `load_model_dict_into_meta` since version 0.33.0 in this commit:
                    # https://github.com/huggingface/diffusers/commit/f5929e03060d56063ff34b25a8308833bec7c785.
                    load_model_dict_into_meta(
                        model,
                        filtered_state_dict,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )
                else:
                    model._convert_deprecated_attention_blocks(filtered_state_dict)
                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        filtered_state_dict,
                        device=param_device,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )

                    if cls._keys_to_ignore_on_load_unexpected is not None:
                        for pat in cls._keys_to_ignore_on_load_unexpected:
                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                    if len(unexpected_keys) > 0:
                        print(
                            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                        )
                
                return model
            except Exception as e:
                print(
                    f"The low_cpu_mem_usage mode is not work because {e}. Use low_cpu_mem_usage=False instead."
                )
        
        model = cls.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file, safe_open
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for _model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]
        
        if model.state_dict()['patch_embedding.weight'].size() != state_dict['patch_embedding.weight'].size():
            model.state_dict()['patch_embedding.weight'][:, :state_dict['patch_embedding.weight'].size()[1], :, :] = state_dict['patch_embedding.weight'][:, :model.state_dict()['patch_embedding.weight'].size()[1], :, :]
            model.state_dict()['patch_embedding.weight'][:, state_dict['patch_embedding.weight'].size()[1]:, :, :] = 0
            state_dict['patch_embedding.weight'] = model.state_dict()['patch_embedding.weight']
        
        tmp_state_dict = {} 
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
                
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)
        
        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")
        
        model = model.to(torch_dtype)
        return model


class Wan2_2Transformer3DModel(WanTransformer3DModel):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    # ignore_for_config = [
    #     'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    # ]
    # _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True
    
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        downscale_factor_control_adapter=8,
        add_ref_conv=False,
        in_dim_ref_conv=16,
        add_simulation_branch=False,
        simulation_state_dim=6,
        simulation_cond_dim=0,
        simulation_out_dim=None,
        simulation_num_layers=8,
        simulation_pairing=None,
        simulation_pairing_strategy="dilated",
        simulation_pairing_power=1.5,
        simulation_max_seq_len=4096,
        simulation_separate_time_embedding=True,
    ):
        r"""
        Initialize the diffusion model backbone.
        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            in_channels=in_channels,
            hidden_size=hidden_size,
            add_control_adapter=add_control_adapter,
            in_dim_control_adapter=in_dim_control_adapter,
            downscale_factor_control_adapter=downscale_factor_control_adapter,
            add_ref_conv=add_ref_conv,
            in_dim_ref_conv=in_dim_ref_conv,
            cross_attn_type="cross_attn",
            add_simulation_branch=add_simulation_branch,
            simulation_state_dim=simulation_state_dim,
            simulation_cond_dim=simulation_cond_dim,
            simulation_out_dim=simulation_out_dim,
            simulation_num_layers=simulation_num_layers,
            simulation_pairing=simulation_pairing,
            simulation_pairing_strategy=simulation_pairing_strategy,
            simulation_pairing_power=simulation_pairing_power,
            simulation_max_seq_len=simulation_max_seq_len,
            simulation_separate_time_embedding=simulation_separate_time_embedding,
        )
        
        if hasattr(self, "img_emb"):
            del self.img_emb
