"""
Triton kernels for LiteratiQuant: GPU-accelerated 1-bit group quantization.

Kernels:
  1. triton_literati_quantize — compute sign + mean_abs scale per group
  2. triton_literati_dequantize — reconstruct: sign * scale
  3. triton_literati_fused — quantize → store → dequantize in one pass

The simplest kernel in the family: no rotation, no codebook lookup.
Just sign extraction + scale computation per group.

Compared to other Triton kernels:
  - IsoQuant: 16 FMAs per 4D group (quaternion multiply + codebook)
  - PlanarQuant: 4 FMAs per 2D pair (Givens rotation + codebook)
  - LiteratiQuant: 1 compare + 1 multiply per element (sign + scale)
"""

import torch
import triton
import triton.language as tl


# ── Fused quantize-dequantize kernel ─────────────────────────────────

@triton.jit
def _literati_fused_kernel(
    input_ptr, output_ptr,
    scales_ptr,         # output: (batch_size, n_groups) per-group scales
    batch_size, emb_dim,
    n_groups: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    """
    Fused 1-bit quantize + dequantize.

    For each group of GROUP_SIZE elements:
      1. Compute scale = mean(|x|)
      2. Compute signs = sign(x)
      3. Output = signs * scale

    This is the full round-trip; for inference with pre-stored signs,
    use the dequantize-only kernel instead.
    """
    pid_batch = tl.program_id(0)
    pid_group = tl.program_id(1)

    batch_offset = pid_batch * BLOCK_BATCH
    group_offset = pid_group * GROUP_SIZE

    batch_range = batch_offset + tl.arange(0, BLOCK_BATCH)
    batch_mask = batch_range < batch_size

    elem_range = group_offset + tl.arange(0, GROUP_SIZE)
    elem_mask = elem_range < emb_dim

    # Load group elements: (BLOCK_BATCH, GROUP_SIZE)
    offsets = batch_range[:, None] * emb_dim + elem_range[None, :]
    mask = batch_mask[:, None] & elem_mask[None, :]

    vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Compute per-group scale = mean(|vals|)
    abs_vals = tl.abs(vals)
    # Sum across GROUP_SIZE dimension, then divide
    group_sum = tl.sum(abs_vals, axis=1)  # (BLOCK_BATCH,)
    scale = group_sum / GROUP_SIZE  # (BLOCK_BATCH,)
    # Clamp scale to avoid division by zero
    scale = tl.maximum(scale, 1e-8)

    # Store scales
    scale_offsets = batch_range * n_groups + pid_group
    tl.store(scales_ptr + scale_offsets, scale, mask=batch_mask)

    # Compute signs and reconstruct: sign(x) * scale
    signs = tl.where(vals >= 0, 1.0, -1.0)
    output = signs * scale[:, None]

    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_literati_fused(x: torch.Tensor, group_size: int = 128):
    """
    Triton-accelerated fused 1-bit quantize + dequantize.

    Args:
        x: (batch, dim) input tensor
        group_size: elements per group (default 128)

    Returns:
        x_q: (batch, dim) quantized-dequantized tensor
        scales: (batch, n_groups) per-group scales
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensors"
    batch_size, emb_dim = x.shape
    n_groups = (emb_dim + group_size - 1) // group_size

    # Pad if needed
    pad = n_groups * group_size - emb_dim
    if pad > 0:
        x = torch.nn.functional.pad(x, (0, pad))

    output = torch.empty_like(x)
    scales = torch.empty(batch_size, n_groups, device=x.device, dtype=x.dtype)

    BLOCK_BATCH = min(32, batch_size)
    grid = (
        (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH,
        n_groups,
    )

    _literati_fused_kernel[grid](
        x, output, scales,
        batch_size, x.shape[1],
        n_groups=n_groups,
        GROUP_SIZE=group_size,
        BLOCK_BATCH=BLOCK_BATCH,
    )

    if pad > 0:
        output = output[:, :emb_dim]

    return output, scales


# ── Dequantize-only kernel (for inference with stored signs) ─────────

@triton.jit
def _literati_dequant_kernel(
    signs_ptr,          # (batch_size, n_groups, GROUP_SIZE) as ±1.0
    scales_ptr,         # (batch_size, n_groups) fp16/fp32
    output_ptr,
    batch_size, emb_dim,
    n_groups: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    """Reconstruct from stored signs + scales."""
    pid_batch = tl.program_id(0)
    pid_group = tl.program_id(1)

    batch_offset = pid_batch * BLOCK_BATCH
    group_offset = pid_group * GROUP_SIZE

    batch_range = batch_offset + tl.arange(0, BLOCK_BATCH)
    batch_mask = batch_range < batch_size

    elem_range = tl.arange(0, GROUP_SIZE)

    # Load signs: (BLOCK_BATCH, GROUP_SIZE)
    sign_offsets = batch_range[:, None] * (n_groups * GROUP_SIZE) + pid_group * GROUP_SIZE + elem_range[None, :]
    signs = tl.load(signs_ptr + sign_offsets, mask=batch_mask[:, None], other=0.0)

    # Load scales: (BLOCK_BATCH,)
    scale_offsets = batch_range * n_groups + pid_group
    scales = tl.load(scales_ptr + scale_offsets, mask=batch_mask)

    # Reconstruct
    output = signs * scales[:, None]

    # Store to output
    out_offsets = batch_range[:, None] * emb_dim + (group_offset + elem_range[None, :])
    out_mask = batch_mask[:, None] & ((group_offset + elem_range[None, :]) < emb_dim)
    tl.store(output_ptr + out_offsets, output, mask=out_mask)


def triton_literati_dequantize(signs: torch.Tensor, scales: torch.Tensor,
                                emb_dim: int, group_size: int = 128):
    """
    Triton-accelerated dequantize from stored signs + scales.

    Args:
        signs: (batch, n_groups, group_size) as ±1.0 float
        scales: (batch, n_groups) per-group scales
        emb_dim: original embedding dimension
        group_size: elements per group

    Returns:
        x_q: (batch, emb_dim) reconstructed tensor
    """
    assert signs.is_cuda, "Triton kernel requires CUDA tensors"
    batch_size = signs.shape[0]
    n_groups = signs.shape[1]

    output = torch.empty(batch_size, emb_dim, device=signs.device, dtype=scales.dtype)

    BLOCK_BATCH = min(32, batch_size)
    grid = (
        (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH,
        n_groups,
    )

    _literati_dequant_kernel[grid](
        signs, scales.float(), output,
        batch_size, emb_dim,
        n_groups=n_groups,
        GROUP_SIZE=group_size,
        BLOCK_BATCH=BLOCK_BATCH,
    )

    return output


# ── Quantize-only kernel (extract signs + scales) ────────────────────

@triton.jit
def _literati_quantize_kernel(
    input_ptr,
    signs_out_ptr,      # (batch_size, n_groups, GROUP_SIZE) as ±1.0
    scales_out_ptr,     # (batch_size, n_groups)
    batch_size, emb_dim,
    n_groups: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    """Extract signs and compute scales (no reconstruction)."""
    pid_batch = tl.program_id(0)
    pid_group = tl.program_id(1)

    batch_offset = pid_batch * BLOCK_BATCH
    group_offset = pid_group * GROUP_SIZE

    batch_range = batch_offset + tl.arange(0, BLOCK_BATCH)
    batch_mask = batch_range < batch_size

    elem_range = group_offset + tl.arange(0, GROUP_SIZE)
    elem_mask = elem_range < emb_dim

    # Load input
    offsets = batch_range[:, None] * emb_dim + elem_range[None, :]
    mask = batch_mask[:, None] & elem_mask[None, :]
    vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Signs
    signs = tl.where(vals >= 0, 1.0, -1.0)

    # Scale = mean(|vals|)
    abs_vals = tl.abs(vals)
    group_sum = tl.sum(abs_vals, axis=1)
    scale = group_sum / GROUP_SIZE
    scale = tl.maximum(scale, 1e-8)

    # Store signs
    sign_offsets = batch_range[:, None] * (n_groups * GROUP_SIZE) + pid_group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    tl.store(signs_out_ptr + sign_offsets, signs, mask=batch_mask[:, None])

    # Store scales
    scale_offsets = batch_range * n_groups + pid_group
    tl.store(scales_out_ptr + scale_offsets, scale, mask=batch_mask)


def triton_literati_quantize(x: torch.Tensor, group_size: int = 128):
    """
    Triton-accelerated quantize: extract signs + compute scales.

    Args:
        x: (batch, dim) input tensor
        group_size: elements per group

    Returns:
        signs: (batch, n_groups, group_size) as ±1.0 float
        scales: (batch, n_groups) per-group scales
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensors"
    batch_size, emb_dim = x.shape
    n_groups = (emb_dim + group_size - 1) // group_size

    pad = n_groups * group_size - emb_dim
    if pad > 0:
        x = torch.nn.functional.pad(x, (0, pad))

    signs = torch.empty(batch_size, n_groups, group_size, device=x.device, dtype=x.dtype)
    scales = torch.empty(batch_size, n_groups, device=x.device, dtype=x.dtype)

    BLOCK_BATCH = min(32, batch_size)
    grid = (
        (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH,
        n_groups,
    )

    _literati_quantize_kernel[grid](
        x, signs, scales,
        batch_size, x.shape[1],
        n_groups=n_groups,
        GROUP_SIZE=group_size,
        BLOCK_BATCH=BLOCK_BATCH,
    )

    return signs, scales
