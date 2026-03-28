"""
Triton kernels for RotorQuant: GPU-accelerated Clifford algebra quantization.

Kernels:
  1. rotor_sandwich (forward)  — R x R̃ via sparse Cl(3,0) geometric product
  2. rotor_full_fused          — embed→rotor→quantize→unrotor→extract pipeline
  3. fused_attention_scores    — Q@K^T directly on grade-aware compressed keys
  4. rotor_inverse_sandwich    — R̃ x R (dequantize path)

These replace the CUDA C++ kernels in csrc/rotor_fused_kernel.cu with portable,
auto-tuned Triton code that works on both NVIDIA and AMD GPUs.

IMPORTANT: The rotor sandwich R x R̃ requires two DIFFERENT products:
  - R * x   (rotor on LEFT)  — _gp_rotor_mv
  - temp * R̃ (rotor on RIGHT) — _gp_mv_rotor
These are NOT the same in non-commutative Clifford algebra.
"""

import torch
import triton
import triton.language as tl
import math
from typing import Optional


# ============================================================================
# Sparse geometric product helpers for Cl(3,0) rotors
#
# A rotor R has only 4 non-zero components: [s, 0, 0, 0, b12, b13, b23, 0]
# (scalar + bivector grades). We exploit this sparsity for ~28 FMAs vs 64.
# ============================================================================

@triton.jit
def _gp_rotor_mv(
    s, p12, p13, p23,
    x0, x1, x2, x3, x4, x5, x6, x7,
):
    """Sparse geometric product: rotor * multivector (rotor on LEFT).

    Computes R * x where R = [s, 0, 0, 0, p12, p13, p23, 0].
    """
    r0 = s * x0 - p12 * x4 - p13 * x5 - p23 * x6
    r1 = s * x1 + p12 * x2 + p13 * x3 + p23 * x7
    r2 = s * x2 - p12 * x1 + p23 * x3 - p13 * x7
    r3 = s * x3 - p13 * x1 - p23 * x2 + p12 * x7
    r4 = s * x4 + p12 * x0 + p13 * x6 - p23 * x5
    r5 = s * x5 + p13 * x0 - p12 * x6 + p23 * x4
    r6 = s * x6 + p23 * x0 + p12 * x5 - p13 * x4
    r7 = s * x7 - p23 * x1 + p13 * x2 - p12 * x3
    return r0, r1, r2, r3, r4, r5, r6, r7


@triton.jit
def _gp_mv_rotor(
    x0, x1, x2, x3, x4, x5, x6, x7,
    s, p12, p13, p23,
):
    """Sparse geometric product: multivector * rotor (rotor on RIGHT).

    Computes x * R where R = [s, 0, 0, 0, p12, p13, p23, 0].
    This is DIFFERENT from R * x in non-commutative Clifford algebra.
    """
    r0 = s * x0 - p12 * x4 - p13 * x5 - p23 * x6
    r1 = s * x1 - p12 * x2 - p13 * x3 + p23 * x7
    r2 = s * x2 + p12 * x1 - p23 * x3 - p13 * x7
    r3 = s * x3 + p13 * x1 + p23 * x2 + p12 * x7
    r4 = s * x4 + p12 * x0 + p23 * x5 - p13 * x6
    r5 = s * x5 + p13 * x0 - p23 * x4 + p12 * x6
    r6 = s * x6 + p23 * x0 + p13 * x4 - p12 * x5
    r7 = s * x7 + p23 * x1 - p13 * x2 + p12 * x3
    return r0, r1, r2, r3, r4, r5, r6, r7


@triton.jit
def _quantize_nearest(val, centroids_ptr, n_levels: tl.constexpr):
    """Find nearest centroid for a scalar value."""
    best_val = tl.load(centroids_ptr)
    best_dist = tl.abs(val - best_val)
    for i in tl.static_range(1, n_levels):
        c = tl.load(centroids_ptr + i)
        d = tl.abs(val - c)
        mask = d < best_dist
        best_dist = tl.where(mask, d, best_dist)
        best_val = tl.where(mask, c, best_val)
    return best_val


# ============================================================================
# Kernel: Fused rotor sandwich (forward)
#   Input:  vectors (..., emb_dim)
#   Output: multivectors (..., n_groups, 8)
#
#   embed(x) → R x R̃  =  (R * x) * R̃
# ============================================================================

@triton.jit
def _rotor_sandwich_kernel(
    input_ptr, rotors_ptr, output_ptr,
    batch_size, emb_dim, n_groups: tl.constexpr,
    stride_in_b, stride_in_d,
    stride_out_b, stride_out_g, stride_out_c,
    BLOCK_G: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_groups

    # Load rotor: R = [s, p12, p13, p23]
    r_s   = tl.load(rotors_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
    r_p12 = tl.load(rotors_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
    r_p13 = tl.load(rotors_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
    r_p23 = tl.load(rotors_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

    # Embed: load 3 vector components per group
    d0 = g_offs * 3
    v1 = tl.load(input_ptr + pid_b * stride_in_b + d0 * stride_in_d,
                  mask=g_mask & (d0 < emb_dim), other=0.0)
    v2 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 1) * stride_in_d,
                  mask=g_mask & ((d0 + 1) < emb_dim), other=0.0)
    v3 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 2) * stride_in_d,
                  mask=g_mask & ((d0 + 2) < emb_dim), other=0.0)

    z = tl.zeros_like(v1)

    # Step 1: temp = R * x (rotor on LEFT)
    t0, t1, t2, t3, t4, t5, t6, t7 = _gp_rotor_mv(
        r_s, r_p12, r_p13, r_p23,
        z, v1, v2, v3, z, z, z, z,
    )

    # Step 2: result = temp * R̃ (rotor on RIGHT, reverse negates bivectors)
    o0, o1, o2, o3, o4, o5, o6, o7 = _gp_mv_rotor(
        t0, t1, t2, t3, t4, t5, t6, t7,
        r_s, -r_p12, -r_p13, -r_p23,
    )

    # Store output multivectors [batch, n_groups, 8]
    out_base = pid_b * stride_out_b + g_offs * stride_out_g
    tl.store(output_ptr + out_base + 0 * stride_out_c, o0, mask=g_mask)
    tl.store(output_ptr + out_base + 1 * stride_out_c, o1, mask=g_mask)
    tl.store(output_ptr + out_base + 2 * stride_out_c, o2, mask=g_mask)
    tl.store(output_ptr + out_base + 3 * stride_out_c, o3, mask=g_mask)
    tl.store(output_ptr + out_base + 4 * stride_out_c, o4, mask=g_mask)
    tl.store(output_ptr + out_base + 5 * stride_out_c, o5, mask=g_mask)
    tl.store(output_ptr + out_base + 6 * stride_out_c, o6, mask=g_mask)
    tl.store(output_ptr + out_base + 7 * stride_out_c, o7, mask=g_mask)


def triton_rotor_sandwich(
    input: torch.Tensor,     # [batch, emb_dim]
    rotors: torch.Tensor,    # [n_groups, 4] packed as [s, b12, b13, b23]
) -> torch.Tensor:
    """Apply rotor sandwich R x R̃ using Triton.

    Args:
        input: Vectors [batch, emb_dim] (float32 or float16)
        rotors: Packed rotors [n_groups, 4] with [scalar, e12, e13, e23]

    Returns:
        Multivectors [batch, n_groups, 8]
    """
    batch_size, emb_dim = input.shape
    n_groups = rotors.shape[0]

    input_f32 = input.float().contiguous()
    rotors_f32 = rotors.float().contiguous()

    output = torch.empty(batch_size, n_groups, 8,
                         device=input.device, dtype=torch.float32)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 128)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _rotor_sandwich_kernel[grid](
        input_f32, rotors_f32, output,
        batch_size, emb_dim, n_groups,
        input_f32.stride(0), input_f32.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_G=BLOCK_G,
    )

    return output.to(input.dtype)


# ============================================================================
# Kernel: Fused RotorQuant full pipeline
#   embed → R x R̃ → quantize → R̃ x R → extract
#
#   Single kernel launch for the entire quantize-dequantize cycle.
# ============================================================================

@triton.jit
def _rotor_full_fused_kernel(
    input_ptr, output_ptr,
    rotors_ptr,
    c_scalar_ptr, c_vector_ptr, c_bivector_ptr, c_trivector_ptr,
    batch_size, emb_dim,
    n_groups: tl.constexpr,
    n_scalar: tl.constexpr,
    n_vector: tl.constexpr,
    n_bivector: tl.constexpr,
    n_trivector: tl.constexpr,
    stride_in_b, stride_in_d,
    stride_out_b, stride_out_d,
    BLOCK_G: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_groups

    # Load rotor
    r_s   = tl.load(rotors_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
    r_p12 = tl.load(rotors_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
    r_p13 = tl.load(rotors_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
    r_p23 = tl.load(rotors_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

    # Embed
    d0 = g_offs * 3
    v1 = tl.load(input_ptr + pid_b * stride_in_b + d0 * stride_in_d,
                  mask=g_mask & (d0 < emb_dim), other=0.0)
    v2 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 1) * stride_in_d,
                  mask=g_mask & ((d0 + 1) < emb_dim), other=0.0)
    v3 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 2) * stride_in_d,
                  mask=g_mask & ((d0 + 2) < emb_dim), other=0.0)

    z = tl.zeros_like(v1)

    # Forward sandwich: temp = R * x, rotated = temp * R̃
    t0, t1, t2, t3, t4, t5, t6, t7 = _gp_rotor_mv(
        r_s, r_p12, r_p13, r_p23, z, v1, v2, v3, z, z, z, z)
    o0, o1, o2, o3, o4, o5, o6, o7 = _gp_mv_rotor(
        t0, t1, t2, t3, t4, t5, t6, t7, r_s, -r_p12, -r_p13, -r_p23)

    # Grade-aware quantization — vector only (e1, e2, e3)
    # Scalar/bivector: always zero after sandwich of grade-1 input
    # Trivector: non-zero but NEVER READ by extract (only grade-1 is extracted)
    # Dropping trivector saves 25% of indices with zero MSE impact
    q0 = z  # scalar: always zero
    q1 = _quantize_nearest(o1, c_vector_ptr, n_vector)
    q2 = _quantize_nearest(o2, c_vector_ptr, n_vector)
    q3 = _quantize_nearest(o3, c_vector_ptr, n_vector)
    q4 = z  # bivector: always zero
    q5 = z
    q6 = z
    q7 = z  # trivector: non-zero but unused by extract

    # Inverse sandwich: temp2 = R̃ * q, final = temp2 * R
    t0, t1, t2, t3, t4, t5, t6, t7 = _gp_rotor_mv(
        r_s, -r_p12, -r_p13, -r_p23, q0, q1, q2, q3, q4, q5, q6, q7)
    f0, f1, f2, f3, f4, f5, f6, f7 = _gp_mv_rotor(
        t0, t1, t2, t3, t4, t5, t6, t7, r_s, r_p12, r_p13, r_p23)

    # Extract grade-1 back to output
    tl.store(output_ptr + pid_b * stride_out_b + d0 * stride_out_d,
             f1, mask=g_mask & (d0 < emb_dim))
    tl.store(output_ptr + pid_b * stride_out_b + (d0 + 1) * stride_out_d,
             f2, mask=g_mask & ((d0 + 1) < emb_dim))
    tl.store(output_ptr + pid_b * stride_out_b + (d0 + 2) * stride_out_d,
             f3, mask=g_mask & ((d0 + 2) < emb_dim))


def triton_rotor_full_fused(
    input: torch.Tensor,
    rotors: torch.Tensor,
    c_scalar: torch.Tensor,
    c_vector: torch.Tensor,
    c_bivector: torch.Tensor,
    c_trivector: torch.Tensor,
) -> torch.Tensor:
    """Fused RotorQuant pipeline: normalize→embed→rotor→quantize→unrotor→extract→rescale.

    Single kernel launch for the full quantize-dequantize roundtrip.
    Only quantizes non-zero grades (vector + trivector); scalar and bivector
    are always zero after sandwich of grade-1 input and are skipped.
    """
    batch_size, emb_dim = input.shape
    n_groups = rotors.shape[0]

    # Norm separation: quantize unit vectors, store norms
    input_f32 = input.float()
    norms = input_f32.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    input_f32 = (input_f32 / norms).contiguous()
    rotors_f32 = rotors.float().contiguous()
    # Scalar/bivector centroids not used (zero grades), but kernel signature needs them
    c_s = c_scalar.float().contiguous() if c_scalar is not None else c_vector.float().contiguous()
    c_v = c_vector.float().contiguous()
    c_b = c_bivector.float().contiguous() if c_bivector is not None else c_vector.float().contiguous()
    c_t = c_trivector.float().contiguous() if c_trivector is not None else c_vector.float().contiguous()

    output = torch.empty_like(input_f32)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 128)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _rotor_full_fused_kernel[grid](
        input_f32, output, rotors_f32,
        c_s, c_v, c_b, c_t,
        batch_size, emb_dim, n_groups,
        len(c_s), len(c_v), len(c_b), len(c_t),
        input_f32.stride(0), input_f32.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_G=BLOCK_G,
    )

    # Rescale by original norms
    output = output * norms
    return output.to(input.dtype)


# ============================================================================
# Kernel: Fused attention scores on RotorQuant-compressed keys
#
# Adapted from TurboQuant's Triton attention kernel.
# Computes Q@K^T by gathering centroids from quantized key indices.
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 32, "BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_S": 64, "BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_S": 128, "BLOCK_D": 64}, num_warps=8),
        triton.Config({"BLOCK_S": 64, "BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_S": 128, "BLOCK_D": 128}, num_warps=8),
    ],
    key=["kv_len", "head_dim"],
)
@triton.jit
def _fused_rotor_attention_kernel(
    Q_ptr, K_idx_ptr, K_norms_ptr, C_ptr, Out_ptr,
    kv_len, head_dim: tl.constexpr,
    n_q_heads, n_kv_heads, scale,
    stride_q_bh, stride_q_d,
    stride_ki_bh, stride_ki_s, stride_ki_d,
    stride_kn_bh, stride_kn_s,
    stride_o_bh, stride_o_s,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute attention scores from pre-rotated queries and quantized keys."""
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    batch_idx = pid_bh // n_q_heads
    q_head_idx = pid_bh % n_q_heads
    gqa_ratio = n_q_heads // n_kv_heads
    kv_head_idx = q_head_idx // gqa_ratio
    kv_bh = batch_idx * n_kv_heads + kv_head_idx

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < kv_len

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    for d_start in range(0, head_dim, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < head_dim

        q_ptrs = Q_ptr + pid_bh * stride_q_bh + d_offs * stride_q_d
        q_vals = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        ki_ptrs = (K_idx_ptr
                   + kv_bh * stride_ki_bh
                   + s_offs[:, None] * stride_ki_s
                   + d_offs[None, :] * stride_ki_d)
        combined_mask = s_mask[:, None] & d_mask[None, :]
        k_idx = tl.load(ki_ptrs, mask=combined_mask, other=0).to(tl.int32)

        k_vals = tl.load(C_ptr + k_idx, mask=combined_mask, other=0.0).to(tl.float32)

        acc += tl.sum(k_vals * q_vals[None, :], axis=1)

    kn_ptrs = K_norms_ptr + kv_bh * stride_kn_bh + s_offs * stride_kn_s
    norms = tl.load(kn_ptrs, mask=s_mask, other=0.0).to(tl.float32)

    scores = norms * acc * scale

    o_ptrs = Out_ptr + pid_bh * stride_o_bh + s_offs * stride_o_s
    tl.store(o_ptrs, scores, mask=s_mask)


def triton_fused_attention(
    q_rotated: torch.Tensor,
    key_indices: torch.Tensor,
    key_norms: torch.Tensor,
    centroids: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Fused attention scores on compressed keys via Triton.

    Args:
        q_rotated: Pre-rotated queries [batch, n_q_heads, q_len, head_dim]
        key_indices: Quantized key indices [batch, n_kv_heads, kv_len, head_dim]
        key_norms: Key norms [batch, n_kv_heads, kv_len]
        centroids: Centroid values [n_levels]
        scale: Attention scale (1/sqrt(head_dim))

    Returns:
        Attention scores [batch, n_q_heads, q_len, kv_len]
    """
    batch, n_q_heads, q_len, head_dim = q_rotated.shape
    _, n_kv_heads, kv_len, _ = key_indices.shape

    q_flat = q_rotated.reshape(batch * n_q_heads * q_len, head_dim).contiguous()
    ki_flat = key_indices.reshape(batch * n_kv_heads, kv_len, head_dim).contiguous()
    kn_flat = key_norms.reshape(batch * n_kv_heads, kv_len).contiguous()
    centroids = centroids.contiguous().float()

    out = torch.empty(batch * n_q_heads * q_len, kv_len,
                      device=q_rotated.device, dtype=torch.float32)

    effective_q_heads = n_q_heads * q_len

    # Use grid lambda so autotuned BLOCK_S is used in grid calculation
    grid = lambda meta: (batch * effective_q_heads,
                         triton.cdiv(kv_len, meta['BLOCK_S']))

    _fused_rotor_attention_kernel[grid](
        q_flat, ki_flat, kn_flat, centroids, out,
        kv_len, head_dim,
        effective_q_heads, n_kv_heads, scale,
        q_flat.stride(0), q_flat.stride(1),
        ki_flat.stride(0), ki_flat.stride(1), ki_flat.stride(2),
        kn_flat.stride(0), kn_flat.stride(1),
        out.stride(0), out.stride(1),
    )

    return out.reshape(batch, n_q_heads, q_len, kv_len)


# ============================================================================
# Kernel: Inverse rotor sandwich (dequantize path)
#   Input:  multivectors [batch, n_groups, 8]
#   Output: vectors [batch, emb_dim]
#
#   R̃ q R  =  (R̃ * q) * R
# ============================================================================

@triton.jit
def _rotor_inverse_sandwich_kernel(
    input_ptr, rotors_ptr, output_ptr,
    batch_size, emb_dim, n_groups: tl.constexpr,
    stride_in_b, stride_in_g, stride_in_c,
    stride_out_b, stride_out_d,
    BLOCK_G: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_groups

    r_s   = tl.load(rotors_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
    r_p12 = tl.load(rotors_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
    r_p13 = tl.load(rotors_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
    r_p23 = tl.load(rotors_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

    in_base = pid_b * stride_in_b + g_offs * stride_in_g
    x0 = tl.load(input_ptr + in_base + 0 * stride_in_c, mask=g_mask, other=0.0)
    x1 = tl.load(input_ptr + in_base + 1 * stride_in_c, mask=g_mask, other=0.0)
    x2 = tl.load(input_ptr + in_base + 2 * stride_in_c, mask=g_mask, other=0.0)
    x3 = tl.load(input_ptr + in_base + 3 * stride_in_c, mask=g_mask, other=0.0)
    x4 = tl.load(input_ptr + in_base + 4 * stride_in_c, mask=g_mask, other=0.0)
    x5 = tl.load(input_ptr + in_base + 5 * stride_in_c, mask=g_mask, other=0.0)
    x6 = tl.load(input_ptr + in_base + 6 * stride_in_c, mask=g_mask, other=0.0)
    x7 = tl.load(input_ptr + in_base + 7 * stride_in_c, mask=g_mask, other=0.0)

    # Inverse sandwich: temp = R̃ * x (LEFT), final = temp * R (RIGHT)
    t0, t1, t2, t3, t4, t5, t6, t7 = _gp_rotor_mv(
        r_s, -r_p12, -r_p13, -r_p23, x0, x1, x2, x3, x4, x5, x6, x7)
    f0, f1, f2, f3, f4, f5, f6, f7 = _gp_mv_rotor(
        t0, t1, t2, t3, t4, t5, t6, t7, r_s, r_p12, r_p13, r_p23)

    # Extract grade-1 to output
    d0 = g_offs * 3
    tl.store(output_ptr + pid_b * stride_out_b + d0 * stride_out_d,
             f1, mask=g_mask & (d0 < emb_dim))
    tl.store(output_ptr + pid_b * stride_out_b + (d0 + 1) * stride_out_d,
             f2, mask=g_mask & ((d0 + 1) < emb_dim))
    tl.store(output_ptr + pid_b * stride_out_b + (d0 + 2) * stride_out_d,
             f3, mask=g_mask & ((d0 + 2) < emb_dim))


def triton_rotor_inverse_sandwich(
    input_mv: torch.Tensor,
    rotors: torch.Tensor,
    emb_dim: int,
) -> torch.Tensor:
    """Inverse rotor sandwich R̃ x R using Triton."""
    batch_size, n_groups, _ = input_mv.shape

    input_f32 = input_mv.float().contiguous()
    rotors_f32 = rotors.float().contiguous()

    output = torch.empty(batch_size, emb_dim,
                         device=input_mv.device, dtype=torch.float32)

    BLOCK_G = min(triton.next_power_of_2(n_groups), 128)
    grid = (batch_size, triton.cdiv(n_groups, BLOCK_G))

    _rotor_inverse_sandwich_kernel[grid](
        input_f32, rotors_f32, output,
        batch_size, emb_dim, n_groups,
        input_f32.stride(0), input_f32.stride(1), input_f32.stride(2),
        output.stride(0), output.stride(1),
        BLOCK_G=BLOCK_G,
    )

    return output.to(input_mv.dtype)


# ============================================================================
# Helper: Pack RotorQuant rotors into [n_groups, 4] format for Triton
# ============================================================================

def pack_rotors_for_triton(rotors: torch.Tensor) -> torch.Tensor:
    """Convert RotorQuant rotors from [n_groups, 8] to [n_groups, 4] packed format.

    The 8-component Cl(3,0) rotor [s, e1, e2, e3, e12, e13, e23, e123]
    has non-zero components only at indices [0, 4, 5, 6] (scalar + bivector).
    We pack these as [s, e12, e13, e23] for the Triton kernels.
    """
    return torch.stack([
        rotors[..., 0],  # scalar
        rotors[..., 4],  # e12
        rotors[..., 5],  # e13
        rotors[..., 6],  # e23
    ], dim=-1)
