"""
LiteratiQuant: 1-bit symmetric group quantization for weights and KV cache.

Pure optimization-based method: no vector quantization, no rotation, no bias.
Each group of G weights is represented as:
    w_q[i] = sign(w[i]) * scale_group

Storage per group of G=128:
    128 sign bits (packed) + 1 FP16 scale = 1.125 bits/element
    → 14x compression vs FP16

Training uses Straight-Through Estimator (STE) so gradients flow through
the non-differentiable sign/scale quantization step.

Inspired by symmetric 1-bit group quantization formats (Q1_0_g128).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


# ── STE primitives ──────────────────────────────────────────────────

class SignSTE(torch.autograd.Function):
    """Sign function with Straight-Through Estimator for backward pass."""

    @staticmethod
    def forward(ctx, x):
        signs = torch.sign(x)
        signs[signs == 0] = 1.0  # tie-break zeros to +1
        return signs

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient through unchanged
        return grad_output


sign_ste = SignSTE.apply


class ScaleClampSTE(torch.autograd.Function):
    """Clamp scales to positive values with STE for backward pass."""

    @staticmethod
    def forward(ctx, x, min_val):
        return x.clamp(min=min_val)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


scale_clamp_ste = ScaleClampSTE.apply


# ── Core quantize / dequantize ──────────────────────────────────────

def quantize_literati(w: torch.Tensor, scales: torch.Tensor,
                      group_size: int = 128) -> torch.Tensor:
    """
    Forward quantization: w → sign(w) * scale_group.

    Args:
        w: (..., C) weight tensor
        scales: (..., num_groups) per-group scales (learnable or computed)
        group_size: elements per group (default 128)

    Returns:
        w_q: (..., C) quantized tensor (same shape as w)
    """
    C = w.shape[-1]
    G = group_size

    # Pad if needed
    pad = (G - C % G) % G
    if pad > 0:
        w = F.pad(w, (0, pad))

    # Reshape to groups
    w_groups = w.reshape(*w.shape[:-1], -1, G)  # (..., num_groups, G)

    # Sign quantization (STE)
    signs = sign_ste(w_groups)  # (..., num_groups, G)

    # Apply per-group scales
    s = scale_clamp_ste(scales, 1e-8).unsqueeze(-1)  # (..., num_groups, 1)
    w_q = signs * s  # (..., num_groups, G)

    # Flatten back and trim padding
    w_q = w_q.reshape(*w_q.shape[:-2], -1)
    if pad > 0:
        w_q = w_q[..., :C]

    return w_q


def compute_scales_mean_abs(w: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Compute per-group scales as mean(|w|) — classic approach."""
    C = w.shape[-1]
    G = group_size
    pad = (G - C % G) % G
    if pad > 0:
        w = F.pad(w, (0, pad))
    w_groups = w.reshape(*w.shape[:-1], -1, G)
    return w_groups.abs().mean(dim=-1)  # (..., num_groups)


def compute_scales_optimal(w: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """
    MSE-optimal scale via L1 projection onto sign basis.

    Closed-form: s = dot(w, sign(w)) / G = sum(|w|) / G = mean(|w|)
    BUT with outlier clipping this differs from naive mean_abs because
    clipping changes the effective distribution.

    Actually for sign quantization, the MSE-optimal scale is:
        s* = argmin_s ||w - sign(w)*s||^2
           = (w^T sign(w)) / (sign(w)^T sign(w))
           = sum(w * sign(w)) / G
           = sum(|w|) / G
           = mean(|w|)

    So we need the CLIPPING to make a difference. This function applies
    percentile clipping THEN computes the optimal scale.
    """
    C = w.shape[-1]
    G = group_size
    pad = (G - C % G) % G
    if pad > 0:
        w = F.pad(w, (0, pad))
    w_groups = w.reshape(*w.shape[:-1], -1, G)

    # Clip outliers at 99th percentile per group
    abs_vals = w_groups.abs()
    clip_val = torch.quantile(abs_vals.float(), 0.99, dim=-1, keepdim=True)
    w_clipped = w_groups.clamp(-clip_val, clip_val)

    # Optimal scale on clipped values
    signs = torch.sign(w_clipped)
    signs[signs == 0] = 1.0
    scales = (w_clipped * signs).sum(dim=-1) / G  # = mean(|w_clipped|)

    return scales


def clip_outliers(w: torch.Tensor, group_size: int = 128,
                  percentile: float = 0.99) -> torch.Tensor:
    """
    Clip outliers per group to given percentile.

    KV activations have heavy tails — a few extreme values dominate the
    group scale and poison sign quantization for all other elements.
    Clipping before sign extraction reduces this.
    """
    C = w.shape[-1]
    G = group_size
    pad = (G - C % G) % G
    if pad > 0:
        w = F.pad(w, (0, pad))
    w_groups = w.reshape(*w.shape[:-1], -1, G)

    abs_vals = w_groups.abs().float()
    clip_val = torch.quantile(abs_vals, percentile, dim=-1, keepdim=True)
    w_clipped = w_groups.clamp(-clip_val, clip_val)

    w_clipped = w_clipped.reshape(*w_clipped.shape[:-2], -1)
    if pad > 0:
        w_clipped = w_clipped[..., :C]
    return w_clipped


def quantize_literati_v2(w: torch.Tensor, group_size: int = 128,
                          percentile: float = 0.99) -> torch.Tensor:
    """
    Improved 1-bit quantization with outlier clipping + optimal scale.

    1. Clip outliers per group (99th percentile)
    2. Compute MSE-optimal scale on clipped values
    3. sign(w_clipped) * scale

    This is the "zero-cost fix" — no training, no calibration data,
    just better statistics.
    """
    C = w.shape[-1]
    G = group_size
    pad = (G - C % G) % G
    if pad > 0:
        w = F.pad(w, (0, pad))

    w_groups = w.reshape(*w.shape[:-1], -1, G)

    # Clip outliers
    abs_vals = w_groups.abs().float()
    clip_val = torch.quantile(abs_vals, percentile, dim=-1, keepdim=True)
    w_clipped = w_groups.clamp(-clip_val, clip_val)

    # Signs from clipped values (outlier signs are more stable)
    signs = torch.sign(w_clipped)
    signs[signs == 0] = 1.0

    # Optimal scale: project ORIGINAL values onto clipped signs
    # This preserves more magnitude info than clipping the scale too
    scales = (w_groups * signs).sum(dim=-1, keepdim=True) / G

    w_q = signs * scales.clamp(min=1e-8)

    w_q = w_q.reshape(*w_q.shape[:-2], -1)
    if pad > 0:
        w_q = w_q[..., :C]
    return w_q


def quantize_literati_v3(w: torch.Tensor, group_size: int = 128,
                          percentile: float = 0.99,
                          n_iter: int = 3) -> torch.Tensor:
    """
    Iterative 1-bit quantization with alternating scale/sign optimization.

    Alternates between:
    1. Fix signs → optimize scale (closed-form)
    2. Fix scale → re-assign signs based on scale (may flip some)

    3 iterations is enough to converge. This can flip signs for elements
    near zero where the initial sign was ambiguous.
    """
    C = w.shape[-1]
    G = group_size
    pad = (G - C % G) % G
    if pad > 0:
        w = F.pad(w, (0, pad))

    w_groups = w.reshape(*w.shape[:-1], -1, G)

    # Clip outliers
    abs_vals = w_groups.abs().float()
    clip_val = torch.quantile(abs_vals, percentile, dim=-1, keepdim=True)
    w_clipped = w_groups.clamp(-clip_val, clip_val)

    # Initialize signs from clipped values
    signs = torch.sign(w_clipped)
    signs[signs == 0] = 1.0

    for _ in range(n_iter):
        # Optimal scale given signs: s = (w^T signs) / G
        scales = (w_groups * signs).sum(dim=-1, keepdim=True) / G
        scales = scales.clamp(min=1e-8)

        # Re-assign signs given scale: sign(w) may differ from sign(w/s)
        # but for symmetric 1-bit, sign doesn't change with positive scale
        # However, we can use the residual to flip ambiguous signs:
        # If |w_i| < threshold, the sign is unreliable.
        # Re-sign based on which assignment minimizes group error.
        residual = w_groups - signs * scales
        # Try flipping each sign and keep if it reduces error
        alt_signs = -signs
        alt_residual = w_groups - alt_signs * scales
        # Flip where alternative is better (element-wise)
        flip_mask = alt_residual.abs() < residual.abs()
        signs = torch.where(flip_mask, alt_signs, signs)

    # Final scale
    scales = (w_groups * signs).sum(dim=-1, keepdim=True) / G
    scales = scales.clamp(min=1e-8)
    w_q = signs * scales

    w_q = w_q.reshape(*w_q.shape[:-2], -1)
    if pad > 0:
        w_q = w_q[..., :C]
    return w_q


def compute_group_stats(w: torch.Tensor, group_size: int = 128):
    """Compute per-group mean and scale (for asymmetric mode)."""
    C = w.shape[-1]
    G = group_size
    pad = (G - C % G) % G
    if pad > 0:
        w = F.pad(w, (0, pad))
    w_groups = w.reshape(*w.shape[:-1], -1, G)
    offsets = w_groups.mean(dim=-1)                    # (..., num_groups)
    centered = w_groups - offsets.unsqueeze(-1)
    scales = centered.abs().mean(dim=-1)               # (..., num_groups)
    return scales, offsets


def quantize_literati_asymmetric(w: torch.Tensor, scales: torch.Tensor,
                                  offsets: torch.Tensor,
                                  group_size: int = 128) -> torch.Tensor:
    """
    Asymmetric quantization: w_q = sign(w - offset) * scale + offset.

    Centers each group before sign quantization, reducing MSE for
    non-zero-mean groups (common in real activations).

    Storage per group: G sign bits + 1 FP16 scale + 1 FP16 offset
        = (G + 32) / G bits/element = 1.25 bits/elem for G=128
    """
    C = w.shape[-1]
    G = group_size
    pad = (G - C % G) % G
    if pad > 0:
        w = F.pad(w, (0, pad))

    w_groups = w.reshape(*w.shape[:-1], -1, G)
    o = offsets.unsqueeze(-1)
    s = scale_clamp_ste(scales, 1e-8).unsqueeze(-1)

    centered = w_groups - o
    signs = sign_ste(centered)
    w_q = signs * s + o

    w_q = w_q.reshape(*w_q.shape[:-2], -1)
    if pad > 0:
        w_q = w_q[..., :C]
    return w_q


def quantize_literati_2bit(w: torch.Tensor, scales: torch.Tensor,
                            group_size: int = 128) -> torch.Tensor:
    """
    2-bit symmetric quantization: levels {-3s, -s, +s, +3s}.

    No codebook — just 4 uniformly-spaced levels with shared scale.
    Storage per group: G*2 bits + 1 FP16 scale = 2.125 bits/elem for G=128.
    Compression: 16 / 2.125 ≈ 7.5x vs FP16.
    """
    C = w.shape[-1]
    G = group_size
    pad = (G - C % G) % G
    if pad > 0:
        w = F.pad(w, (0, pad))

    w_groups = w.reshape(*w.shape[:-1], -1, G)
    s = scale_clamp_ste(scales, 1e-8).unsqueeze(-1)

    # Nearest-level assignment: levels at -3s, -s, +s, +3s
    # Decision boundaries: -2s, 0, +2s
    normalized = w_groups / s
    # Map to nearest level index {0, 1, 2, 3} → values {-3, -1, +1, +3}
    level_idx = torch.clamp(torch.round((normalized + 3) / 2), 0, 3)
    level_vals = level_idx * 2 - 3  # {-3, -1, +1, +3}

    # STE: use level_vals in forward, pass gradient through
    w_q = level_vals * s
    # STE gradient: w_q is a function of w through the rounding,
    # but rounding kills gradients. Apply STE manually:
    w_q = w_groups + (w_q - w_groups).detach()

    w_q = w_q.reshape(*w_q.shape[:-2], -1)
    if pad > 0:
        w_q = w_q[..., :C]
    return w_q


def compute_scales_for_2bit(w: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Optimal scale for 2-bit uniform: s = mean(|w|) / 2."""
    C = w.shape[-1]
    G = group_size
    pad = (G - C % G) % G
    if pad > 0:
        w = F.pad(w, (0, pad))
    w_groups = w.reshape(*w.shape[:-1], -1, G)
    # For levels {-3s, -s, +s, +3s}, optimal s ≈ mean(|w|) / 2
    return w_groups.abs().mean(dim=-1) / 2.0


def pack_signs(w: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Pack sign bits into uint8 for storage. Returns (..., num_groups, G//8)."""
    C = w.shape[-1]
    G = group_size
    pad = (G - C % G) % G
    if pad > 0:
        w = F.pad(w, (0, pad))
    w_groups = w.reshape(*w.shape[:-1], -1, G)
    # 1 if positive, 0 if negative (matches GGUF convention)
    bits = (w_groups > 0).to(torch.uint8)
    # Pack 8 bits per byte
    packed = torch.zeros(*bits.shape[:-1], G // 8, dtype=torch.uint8, device=w.device)
    for i in range(8):
        packed |= bits[..., i::8] << i
    return packed


def unpack_signs(packed: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Unpack sign bits from uint8. Returns (..., num_groups, G) as ±1.0 float."""
    G = group_size
    signs = torch.zeros(*packed.shape[:-1], G, dtype=torch.float32, device=packed.device)
    for i in range(8):
        bits = ((packed >> i) & 1).float()
        signs[..., i::8] = bits * 2.0 - 1.0  # 0 → -1, 1 → +1
    return signs


# ── LiteratiQuantMSE ────────────────────────────────────────────────

class LiteratiQuantMSE(nn.Module):
    """
    Group quantizer with extreme compression.

    Modes:
        'symmetric':   sign(x) * scale           — 1.125 bits/elem, 14.2x
        'asymmetric':  sign(x-offset)*scale+off  — 1.250 bits/elem, 12.8x
        '2bit':        4-level {-3s,-s,+s,+3s}   — 2.125 bits/elem, 7.5x

    All modes: no codebook, no rotation, pure optimization-based.
    Compatible with the RotorQuant/IsoQuant/PlanarQuant API.
    """

    def __init__(self, d: int, group_size: int = 128,
                 mode: str = 'symmetric', device: str = 'cpu'):
        """
        Args:
            d: vector dimension
            group_size: elements per quantization group (default 128)
            mode: 'symmetric' (1-bit), 'asymmetric' (1-bit+offset), '2bit'
            device: torch device
        """
        super().__init__()
        self.d = d
        self.group_size = group_size
        self.mode = mode
        self.device = device

        self.n_groups = (d + group_size - 1) // group_size
        self.d_padded = self.n_groups * group_size

        if mode == 'symmetric':
            self.bits_per_element = 1.0 + 16.0 / group_size
        elif mode == 'asymmetric':
            self.bits_per_element = 1.0 + 32.0 / group_size  # scale + offset
        elif mode == '2bit':
            self.bits_per_element = 2.0 + 16.0 / group_size
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Quantize vectors.

        x: (..., d)
        Returns: (x_quantized, info_dict)
        """
        norms = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms

        if self.mode == 'symmetric':
            scales = compute_scales_mean_abs(x_unit, self.group_size)
            x_q = quantize_literati(x_unit, scales, self.group_size)
            info = {'scales': scales, '_norms': norms.squeeze(-1), '_x_q_unit': x_q}

        elif self.mode == 'asymmetric':
            scales, offsets = compute_group_stats(x_unit, self.group_size)
            x_q = quantize_literati_asymmetric(x_unit, scales, offsets, self.group_size)
            info = {'scales': scales, 'offsets': offsets,
                    '_norms': norms.squeeze(-1), '_x_q_unit': x_q}

        elif self.mode == '2bit':
            scales = compute_scales_for_2bit(x_unit, self.group_size)
            x_q = quantize_literati_2bit(x_unit, scales, self.group_size)
            info = {'scales': scales, '_norms': norms.squeeze(-1), '_x_q_unit': x_q}

        return x_q, info

    def dequantize(self, indices_dict: Dict) -> torch.Tensor:
        """Reconstruct from stored quantized representation."""
        x_hat = indices_dict['_x_q_unit']
        norms = indices_dict['_norms']
        if norms.dim() < x_hat.dim():
            norms = norms.unsqueeze(-1)
        return x_hat * norms

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Full quantize-dequantize round-trip."""
        x_q, info = self.quantize(x)
        x_hat = self.dequantize(info)
        return x_hat, info

    def compression_ratio(self) -> float:
        """Compression vs FP16 (16 bits/element)."""
        return 16.0 / self.bits_per_element


# ── LiteratiQuantRotated (rotation + 1-bit) ────────────────────────

class LiteratiQuantRotated(nn.Module):
    """
    Rotation-enhanced 1-bit quantizer: rotate → sign × scale → inverse rotate.

    Combines the decorrelation power of IsoQuant quaternion rotations
    with extreme 1-bit compression. Rotation makes per-group distributions
    more uniform, dramatically reducing MSE at 1-bit.

    Storage: ~1.125 bits/element (sign bits + scale) + tiny rotation overhead
    """

    def __init__(self, d: int, group_size: int = 128,
                 seed: int = 42, device: str = 'cpu'):
        super().__init__()
        self.d = d
        self.group_size = group_size
        self.device = device

        self.n_groups = (d + group_size - 1) // group_size
        self.d_padded = self.n_groups * group_size
        self.bits_per_element = 1.0 + 16.0 / group_size

        from .isoquant import IsoQuantMSE
        self._iso = IsoQuantMSE(d, bits=1, seed=seed, mode='fast', device=device)

    def _rotate(self, x_unit: torch.Tensor) -> torch.Tensor:
        """Apply forward quaternion rotation (decorrelate coordinates)."""
        v = self._iso._embed(x_unit)
        v_rot = self._iso._rotate(v)
        return self._iso._extract(v_rot)

    def _unrotate(self, x_rot: torch.Tensor) -> torch.Tensor:
        """Apply inverse quaternion rotation."""
        v = self._iso._embed(x_rot)
        v_inv = self._iso._unrotate(v)
        return self._iso._extract(v_inv)

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        norms = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms

        # Rotate to decorrelate
        x_rot = self._rotate(x_unit)

        # 1-bit sign + scale
        scales = compute_scales_mean_abs(x_rot, self.group_size)
        x_q_rot = quantize_literati(x_rot, scales, self.group_size)

        # Inverse rotate
        x_q = self._unrotate(x_q_rot)

        return x_q, {
            'scales': scales,
            '_norms': norms.squeeze(-1),
            '_x_q_unit': x_q,
        }

    def dequantize(self, indices_dict: Dict) -> torch.Tensor:
        x_hat = indices_dict['_x_q_unit']
        norms = indices_dict['_norms']
        if norms.dim() < x_hat.dim():
            norms = norms.unsqueeze(-1)
        return x_hat * norms

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        x_q, info = self.quantize(x)
        x_hat = self.dequantize(info)
        return x_hat, info

    def compression_ratio(self) -> float:
        return 16.0 / self.bits_per_element


# ── LiteratiQuantLinear (QAT drop-in) ──────────────────────────────

class LiteratiQuantLinear(nn.Module):
    """
    Drop-in nn.Linear replacement with 1-bit group-128 quantization.

    Forward pass uses quantized weights (sign * scale).
    Backward pass uses STE so full-precision latent weights receive gradients.

    Storage (inference): in_features * 1.125 bits per output row.
    """

    def __init__(self, in_features: int, out_features: int,
                 group_size: int = 128, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Full-precision latent weights (quantized on-the-fly during forward)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Learnable per-group scales
        num_groups = (in_features + group_size - 1) // group_size
        self.scales = nn.Parameter(torch.ones(out_features, num_groups) * 0.1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def _quantize_weight(self) -> torch.Tensor:
        """Quantize weight on every forward (STE-compatible)."""
        return quantize_literati(self.weight, self.scales, self.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = self._quantize_weight()
        return F.linear(x, w_q, self.bias)

    @torch.no_grad()
    def compute_scales_from_weights(self):
        """Re-initialize scales from current weight magnitudes."""
        self.scales.data.copy_(compute_scales_mean_abs(self.weight, self.group_size))

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'group_size={self.group_size}, '
                f'bits_per_weight={1.0 + 16.0 / self.group_size:.3f}')


# ── LiteratiQuantEmbedding (QAT drop-in) ───────────────────────────

class LiteratiQuantEmbedding(nn.Module):
    """
    Drop-in nn.Embedding replacement with 1-bit group-128 quantization.

    Token embeddings are quantized on-the-fly during forward.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 group_size: int = 128, padding_idx: Optional[int] = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size
        self.padding_idx = padding_idx

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

        num_groups = (embedding_dim + group_size - 1) // group_size
        self.scales = nn.Parameter(torch.ones(num_embeddings, num_groups) * 0.1)

        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)
                self.scales[padding_idx].fill_(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        w_q = quantize_literati(self.weight, self.scales, self.group_size)
        return F.embedding(input_ids, w_q, self.padding_idx)

    @torch.no_grad()
    def compute_scales_from_weights(self):
        self.scales.data.copy_(compute_scales_mean_abs(self.weight, self.group_size))

    def extra_repr(self) -> str:
        return (f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, '
                f'group_size={self.group_size}')


# ── LiteratiQuantKVCache ────────────────────────────────────────────

class LiteratiQuantKVCache(nn.Module):
    """
    KV cache compression using 1-bit group quantization.

    Each KV vector is stored as packed sign bits + FP16 scale per group.
    Compatible with the RotorQuantKVCache / TurboQuantKVCache interface.

    At group_size=128, this gives 14x compression vs FP16 KV cache.
    For head_dim=128 (standard in Llama/Qwen), each head = 1 group = perfect alignment.
    """

    def __init__(self, d: int, group_size: int = 128, device: str = 'cpu'):
        super().__init__()
        self.d = d
        self.group_size = group_size
        self.device = device
        self.n_groups = (d + group_size - 1) // group_size

        # Compressed cache storage (populated on insert)
        self._packed_signs = None  # (batch, n_heads, seq_len, n_groups, G//8) uint8
        self._scales = None        # (batch, n_heads, seq_len, n_groups) fp16
        self._norms = None         # (batch, n_heads, seq_len) fp32
        self._seq_len = 0

    def compress(self, x: torch.Tensor) -> Dict:
        """
        Compress KV vectors to 1-bit representation.

        x: (batch, n_heads, seq_len, d) or (batch, n_heads, d)
        Returns dict with packed_signs, scales, norms.
        """
        if x.dim() == 3:
            x = x.unsqueeze(2)  # add seq_len dim

        norms = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms

        scales = compute_scales_mean_abs(x_unit, self.group_size)
        packed = pack_signs(x_unit, self.group_size)

        return {
            'packed_signs': packed,
            'scales': scales.to(torch.float16),
            'norms': norms.squeeze(-1),
        }

    def decompress(self, compressed: Dict) -> torch.Tensor:
        """
        Decompress 1-bit representation back to dense vectors.

        Returns: (batch, n_heads, seq_len, d)
        """
        signs = unpack_signs(compressed['packed_signs'], self.group_size)
        scales = compressed['scales'].float().unsqueeze(-1)  # (..., n_groups, 1)
        norms = compressed['norms']

        # Reconstruct: sign * scale per group
        x_unit = signs * scales
        x_unit = x_unit.reshape(*x_unit.shape[:-2], -1)  # flatten groups
        x_unit = x_unit[..., :self.d]  # trim padding

        if norms.dim() < x_unit.dim():
            norms = norms.unsqueeze(-1)

        return x_unit * norms

    def insert(self, x: torch.Tensor):
        """Insert new KV vectors into the compressed cache."""
        compressed = self.compress(x)

        if self._packed_signs is None:
            self._packed_signs = compressed['packed_signs']
            self._scales = compressed['scales']
            self._norms = compressed['norms']
        else:
            self._packed_signs = torch.cat([self._packed_signs, compressed['packed_signs']], dim=2)
            self._scales = torch.cat([self._scales, compressed['scales']], dim=2)
            self._norms = torch.cat([self._norms, compressed['norms']], dim=2)

        self._seq_len = self._packed_signs.shape[2]

    def get_all(self) -> torch.Tensor:
        """Decompress and return entire cache."""
        if self._packed_signs is None:
            return None
        return self.decompress({
            'packed_signs': self._packed_signs,
            'scales': self._scales,
            'norms': self._norms,
        })

    def clear(self):
        self._packed_signs = None
        self._scales = None
        self._norms = None
        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def memory_bytes(self) -> int:
        """Approximate memory usage of the compressed cache."""
        if self._packed_signs is None:
            return 0
        sign_bytes = self._packed_signs.numel()  # uint8
        scale_bytes = self._scales.numel() * 2   # fp16
        norm_bytes = self._norms.numel() * 4     # fp32
        return sign_bytes + scale_bytes + norm_bytes


# ── Model replacement utility ──────────────────────────────────────

def literati_replace(model: nn.Module, group_size: int = 128,
                     skip_names: Optional[set] = None) -> nn.Module:
    """
    Recursively replace all nn.Linear and nn.Embedding modules with
    LiteratiQuant variants for QAT training.

    Args:
        model: any nn.Module (e.g. Qwen3ForCausalLM)
        group_size: elements per quantization group
        skip_names: set of module names to skip (e.g. {'lm_head'})

    Returns:
        model with replaced modules (in-place)
    """
    if skip_names is None:
        skip_names = set()

    for name, module in list(model.named_children()):
        if name in skip_names:
            continue

        if isinstance(module, nn.Linear):
            new_mod = LiteratiQuantLinear(
                module.in_features, module.out_features,
                group_size, bias=module.bias is not None
            )
            with torch.no_grad():
                new_mod.weight.data.copy_(module.weight.data)
                new_mod.compute_scales_from_weights()
                if module.bias is not None:
                    new_mod.bias.data.copy_(module.bias.data)
            setattr(model, name, new_mod)

        elif isinstance(module, nn.Embedding):
            new_mod = LiteratiQuantEmbedding(
                module.num_embeddings, module.embedding_dim,
                group_size, padding_idx=module.padding_idx
            )
            with torch.no_grad():
                new_mod.weight.data.copy_(module.weight.data)
                new_mod.compute_scales_from_weights()
            setattr(model, name, new_mod)

        else:
            literati_replace(module, group_size, skip_names)

    return model


# ── GGUF export helpers ─────────────────────────────────────────────

def export_literati_to_gguf_tensors(model: nn.Module) -> Dict[str, Dict]:
    """
    Extract LiteratiQuant weights as GGUF-compatible Q1_0_g128 tensors.

    Returns dict of {layer_name: {'packed_signs': uint8, 'scales': fp16, 'shape': tuple}}
    suitable for writing to GGUF format.
    """
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, (LiteratiQuantLinear, LiteratiQuantEmbedding)):
            w = module.weight.data
            scales = compute_scales_mean_abs(w, module.group_size)
            packed = pack_signs(w, module.group_size)
            result[name] = {
                'packed_signs': packed.cpu(),
                'scales': scales.to(torch.float16).cpu(),
                'shape': w.shape,
                'group_size': module.group_size,
            }
    return result
