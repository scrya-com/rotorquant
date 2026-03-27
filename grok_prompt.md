# Task: Add QJL correction to RotorQuant's fused attention Triton kernel

## Problem

RotorQuant's fused attention kernel computes Q@K^T directly from compressed key indices (no decompression). But it uses **MSE-only** attention scores which are **biased** — the bias accumulates through 36 transformer layers, destroying perplexity:

- FP16 baseline: PPL = 7.07
- RotorQuant 3-bit fused (MSE-only): PPL = 167
- Google TurboQuant 3-bit fused (MSE + QJL): PPL ≈ 7.07 ← target

## The fix: QJL two-term unbiased estimator

```
<q, k> ≈ <q, k_mse> + ||residual|| * sqrt(π/2) / m * <S@q, sign(S@residual)>
         ─────────     ───────────────────────────────────────────────────────
          term1                              term2 (QJL correction)
```

- `k_mse` = MSE-reconstructed key (from quantized rotor centroid indices)
- `residual = k - k_mse`
- `S` = random Gaussian projection matrix (m × d), shared across all keys per layer
- `sign(S@residual)` = 1-bit per dim (stored per key, packed as uint8)
- `||residual||` = fp16 per key

## What needs to change

### Storage: add to compressed cache per key
- `qjl_signs`: packed bits [batch, n_kv_heads, kv_len, head_dim] as int8 {+1,-1}
- `residual_norms`: fp16 [batch, n_kv_heads, kv_len]
- `S`: random projection matrix [head_dim, head_dim] per layer (or shared)

### Key compression: compute QJL after MSE quantization
```python
k_mse = dequantize(indices, key_norms)
residual = k_original - k_mse
residual_norm = ||residual||
qjl_signs = sign(S @ residual.T)  # 1-bit per dim
```

### Query pre-processing: project through S
```python
query_sketch = Q @ S.T  # [batch, n_heads, q_len, m]
```

### Fused kernel: two-term score
```
term1 = norm * sum_d(Q_rot[d] * centroids[idx[s,d]])          # existing MSE
term2 = res_norm[s] * sqrt(π/2)/m * sum_d(q_sketch[d] * signs[s,d])  # QJL
score[s] = (term1 + term2) * scale
```

## Constraints
- Must work as a Triton kernel (not CUDA C++)
- Must patch into HuggingFace DynamicCache via monkey-patching
- GPU: RTX 5090 (Triton 3.6, PyTorch 2.11)
- Target: perplexity within 5% of FP16 at 3-bit

## Code files follow below

---
### FILE: turboquant/fused_attention.py
```python
"""
RotorQuant Fused Attention: compute Q@K^T directly from compressed keys.

Instead of: decompress keys → matmul (roundtrip, error compounds)
We do:      pre-rotate Q → gather centroids by index → dot product (no roundtrip)

This matches Google TurboQuant's approach but uses Clifford rotors.

Math:
  <Q, K_recon> = <Q, extract(R̃ @ C[idx] @ R)> * norm
               = <R @ embed(Q) @ R̃, C[idx]>_grade1 * norm  (sandwich preserves IP)
               ≈ sum_d( Q_rotated[d] * centroids[idx[d]] ) * norm * scale

So: pre-rotate Q once per query, then gather-dot against compressed keys.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional
from transformers import DynamicCache

from .rotorquant import RotorQuantMSE
from .triton_kernels import (
    triton_rotor_sandwich,
    triton_fused_attention,
    pack_rotors_for_triton,
)
from .clifford import MV_DIM, E1, E2, E3, E123


class RotorQuantCompressedCache(DynamicCache):
    """KV cache that stores compressed keys (uint8 indices + fp16 norms).

    Keys are quantized on insertion via rotor embedding + grade-aware quantization.
    During attention, the fused Triton kernel reads compressed keys directly.
    Values are stored in fp16 (standard).
    """

    def __init__(self, rq: RotorQuantMSE, device: str = "cuda"):
        super().__init__()
        self.rq = rq
        self.device = device
        self.packed_rotors = pack_rotors_for_triton(rq.rotors).to(device)
        self.centroids_vector = getattr(rq, 'centroids_vector').to(device)
        self.centroids_trivector = getattr(rq, 'centroids_trivector').to(device)

        # Build a single flat centroid array for the gather-dot kernel.
        # Vector and trivector use different codebooks, but we can use
        # vector centroids for all 4 components (3 vector + 1 trivector)
        # since they have similar scale after the codebook fix.
        # For maximum accuracy, use vector centroids for grade-1 and
        # trivector centroids for grade-3.
        self.n_groups = rq.n_groups

        # Per-layer compressed key storage
        self._compressed_keys: list[Optional[dict]] = []

    def _quantize_keys(self, key_states: torch.Tensor) -> dict:
        """Quantize keys: normalize → embed → rotor → nearest centroid → indices.

        Args:
            key_states: [batch, n_kv_heads, seq_len, head_dim] float

        Returns:
            dict with 'indices' (uint8) and 'norms' (fp16)
        """
        B, H, S, D = key_states.shape
        flat = key_states.reshape(-1, D).float()

        # Norm separation
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        flat_unit = flat / norms

        # Embed + rotor sandwich → [N, n_groups, 8]
        mv_rot = triton_rotor_sandwich(flat_unit, self.packed_rotors)

        # Extract non-zero grades: vector [1,2,3] + trivector [7]
        # Flatten to [N, n_groups * 4]
        v1 = mv_rot[:, :, E1]    # [N, n_groups]
        v2 = mv_rot[:, :, E2]
        v3 = mv_rot[:, :, E3]
        t7 = mv_rot[:, :, E123]

        # Quantize all components to nearest centroid using SINGLE codebook
        # (fused attention kernel uses one centroid array for gather-dot)
        c_v = self.centroids_vector
        b_v = (c_v[:-1] + c_v[1:]) / 2

        idx_v1 = torch.searchsorted(b_v, v1.contiguous()).to(torch.uint8)
        idx_v2 = torch.searchsorted(b_v, v2.contiguous()).to(torch.uint8)
        idx_v3 = torch.searchsorted(b_v, v3.contiguous()).to(torch.uint8)
        idx_t7 = torch.searchsorted(b_v, t7.contiguous()).to(torch.uint8)

        # Interleave: [v1_g0, v2_g0, v3_g0, t7_g0, v1_g1, ...] → [N, n_groups*4]
        indices = torch.stack([idx_v1, idx_v2, idx_v3, idx_t7], dim=-1)  # [N, n_groups, 4]
        indices = indices.reshape(-1, self.n_groups * 4)  # [N, n_groups*4]

        return {
            'indices': indices.reshape(B, H, S, self.n_groups * 4),
            'norms': norms.squeeze(-1).half().reshape(B, H, S),
        }

    def store_compressed_key(self, key_states: torch.Tensor, layer_idx: int):
        """Quantize and store key states."""
        compressed = self._quantize_keys(key_states)

        while len(self._compressed_keys) <= layer_idx:
            self._compressed_keys.append(None)

        if self._compressed_keys[layer_idx] is None:
            self._compressed_keys[layer_idx] = compressed
        else:
            prev = self._compressed_keys[layer_idx]
            self._compressed_keys[layer_idx] = {
                'indices': torch.cat([prev['indices'], compressed['indices']], dim=2),
                'norms': torch.cat([prev['norms'], compressed['norms']], dim=2),
            }

    def get_compressed_key(self, layer_idx: int) -> Optional[dict]:
        if layer_idx < len(self._compressed_keys):
            return self._compressed_keys[layer_idx]
        return None

    def get_kv_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self._compressed_keys) and self._compressed_keys[layer_idx] is not None:
            return self._compressed_keys[layer_idx]['indices'].shape[2]
        return 0


def pre_rotate_query(
    query: torch.Tensor,        # [batch, n_heads, q_len, head_dim]
    packed_rotors: torch.Tensor, # [n_groups, 4]
    n_groups: int,
) -> torch.Tensor:
    """Pre-rotate query into the rotor-compressed basis.

    Applies the forward rotor sandwich R @ embed(Q) @ R̃ and extracts
    the non-zero grade components (vector + trivector), flattened.

    Returns: [batch, n_heads, q_len, n_groups * 4]
    """
    B, H, Q, D = query.shape
    flat = query.reshape(-1, D)  # [B*H*Q, D]

    # Rotor sandwich → [B*H*Q, n_groups, 8]
    mv_rot = triton_rotor_sandwich(flat, packed_rotors)

    # Extract non-zero grades and flatten
    v1 = mv_rot[:, :, E1]
    v2 = mv_rot[:, :, E2]
    v3 = mv_rot[:, :, E3]
    t7 = mv_rot[:, :, E123]

    # Interleave to match key index layout: [v1_g0, v2_g0, v3_g0, t7_g0, ...]
    q_rot = torch.stack([v1, v2, v3, t7], dim=-1)  # [N, n_groups, 4]
    q_rot = q_rot.reshape(-1, n_groups * 4)  # [N, n_groups*4]

    return q_rot.reshape(B, H, Q, n_groups * 4)


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply RoPE."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA."""
    if n_rep == 1:
        return hidden_states
    batch, n_kv_heads, slen, head_dim = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(
        batch, n_kv_heads, n_rep, slen, head_dim
    ).reshape(batch, n_kv_heads * n_rep, slen, head_dim)


def make_fused_rotor_attention_forward(
    attn_module,
    cache: RotorQuantCompressedCache,
    layer_index: int,
):
    """Create a replacement forward for an attention layer using fused RotorQuant."""

    packed_rotors = cache.packed_rotors
    # Build flat centroid array: interleave vector and trivector centroids
    # Layout matches index layout: [v, v, v, t, v, v, v, t, ...]
    c_v = cache.centroids_vector
    c_t = cache.centroids_trivector
    # For the gather-dot kernel, we use vector centroids for all components
    # (they're very similar after calibration). A more precise version would
    # use separate centroids per component type.
    centroids = c_v  # single centroid array for gather-dot

    n_groups = cache.n_groups
    head_dim = cache.rq.d
    scale = 1.0 / math.sqrt(head_dim)
    n_heads = attn_module.num_heads
    n_kv_heads = getattr(attn_module, 'num_key_value_heads', None)
    if n_kv_heads is None:
        cfg = getattr(attn_module, 'config', None)
        n_kv_heads = getattr(cfg, 'num_key_value_heads', n_heads) if cfg else n_heads
    n_kv_groups = n_heads // n_kv_heads
    layer_idx = layer_index

    def fused_forward(
        hidden_states: torch.Tensor,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Step 1: Project Q, K, V
        query_states = attn_module.q_proj(hidden_states)
        key_states = attn_module.k_proj(hidden_states)
        value_states = attn_module.v_proj(hidden_states)

        # Step 2: Reshape
        query_states = query_states.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

        # Step 3: Apply RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = _apply_rotary_pos_emb(
                query_states, key_states, cos, sin)

        # Step 4: Compress and cache keys
        cache.store_compressed_key(key_states, layer_idx)

        # Step 5: Store values using DynamicCache.update (stores K+V normally)
        # We pass dummy keys (same as values) since we use compressed keys
        cache.update(value_states, value_states, layer_idx)

        # Step 6: Pre-rotate query into compressed basis
        q_rotated = pre_rotate_query(query_states, packed_rotors, n_groups)
        # [batch, n_heads, q_len, n_groups*4]

        # Step 7: Get compressed keys
        compressed = cache.get_compressed_key(layer_idx)
        key_indices = compressed['indices']  # [B, n_kv_heads, kv_len, n_groups*4]
        key_norms = compressed['norms']      # [B, n_kv_heads, kv_len]

        # Step 8: Fused attention via Triton gather-dot
        kv_len = key_indices.shape[2]
        attn_weights = triton_fused_attention(
            q_rotated, key_indices, key_norms, centroids, scale)

        # Step 9: Apply mask + softmax
        if attention_mask is not None:
            causal_mask = attention_mask
            if causal_mask.dim() == 4:
                attn_weights = attn_weights + causal_mask[:, :, :q_len, :kv_len]
            elif causal_mask.dim() == 2:
                attn_weights = attn_weights + causal_mask[:q_len, :kv_len]

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(query_states.dtype)

        # Step 10: Value projection (standard)
        full_values = cache.layers[layer_idx].values
        full_values_expanded = _repeat_kv(full_values, n_kv_groups)
        attn_output = torch.matmul(attn_weights, full_values_expanded)

        # Step 11: Reshape + output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        # Find output projection (different models use different names)
        if hasattr(attn_module, 'o_proj'):
            attn_output = attn_module.o_proj(attn_output)
        elif hasattr(attn_module, 'out_proj'):
            attn_output = attn_module.out_proj(attn_output)

        return attn_output, None

    return fused_forward


def install_fused_rotor_attention(model, bits: int = 3) -> RotorQuantCompressedCache:
    """Patch all attention layers to use fused RotorQuant.

    Returns a RotorQuantCompressedCache to pass as past_key_values.
    """
    config = model.config
    text_config = getattr(config, 'text_config', config)
    head_dim = getattr(text_config, 'head_dim',
                       text_config.hidden_size // text_config.num_attention_heads)

    rq = RotorQuantMSE(head_dim, bits, device="cuda")
    cache = RotorQuantCompressedCache(rq, device="cuda")

    patched = 0
    layer_idx = 0
    for name, module in model.named_modules():
        has_projs = all(hasattr(module, a) for a in ['q_proj', 'k_proj', 'v_proj'])
        has_out = hasattr(module, 'o_proj') or hasattr(module, 'out_proj')
        if has_projs and has_out:
            # Inject head counts if not on module (Qwen2 stores in config)
            if not hasattr(module, 'num_heads'):
                module.num_heads = text_config.num_attention_heads
            if not hasattr(module, 'num_key_value_heads'):
                module.num_key_value_heads = getattr(
                    text_config, 'num_key_value_heads', text_config.num_attention_heads)
            module.forward = make_fused_rotor_attention_forward(
                module, cache, layer_idx)
            patched += 1
            layer_idx += 1

    print(f"  Installed fused RotorQuant ({bits}-bit) on {patched} attention layers")
    return cache
```

---
### FILE: turboquant/triton_kernels.py
```python
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

    # Grade-aware quantization — only non-zero grades (vector + trivector)
    # Scalar (o0) and bivector (o4,o5,o6) are always zero after sandwich of grade-1 input
    q0 = z  # scalar: always zero
    q1 = _quantize_nearest(o1, c_vector_ptr, n_vector)
    q2 = _quantize_nearest(o2, c_vector_ptr, n_vector)
    q3 = _quantize_nearest(o3, c_vector_ptr, n_vector)
    q4 = z  # bivector: always zero
    q5 = z
    q6 = z
    q7 = _quantize_nearest(o7, c_trivector_ptr, n_trivector)

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
    c_t = c_trivector.float().contiguous()

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
```

---
### FILE: turboquant/rotorquant.py
```python
"""
RotorQuant: Reimagining TurboQuant with Clifford Algebra

Instead of TurboQuant's random orthogonal matrix Π (via QR decomposition),
RotorQuant uses Clifford rotors R = exp(B/2) for decorrelation.

Why this is better for geometric data:
1. Rotor sandwich R x R̃ preserves the FULL algebraic structure
   (inner products, outer products, grades) — not just norms
2. Rotors compose naturally: R₂(R₁ x R̃₁)R̃₂ = (R₂R₁) x (R₂R₁)~
3. Grade-aware quantization: different grades can use different bit budgets
4. The bivector structure of rotors means we only need 3 parameters
   (not d² for a full rotation matrix) — massive parameter savings

Algorithm:
  Stage 1 (MSE): Embed vectors as Cl(3,0) multivectors → rotor sandwich →
                  grade-aware Lloyd-Max quantization per component
  Stage 2 (QJL): 1-bit sign quantization on residual for unbiased inner products
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict

from .clifford import (
    MV_DIM, geometric_product, reverse, make_random_rotor,
    rotor_sandwich, embed_vectors_as_multivectors,
    extract_vectors_from_multivectors, multivector_norm_sq,
)
from .lloyd_max import LloydMaxCodebook


class RotorQuantMSE(nn.Module):
    """
    Stage 1: MSE-optimal quantizer using Clifford rotors.

    Instead of Π @ x (matrix multiply), we do R x R̃ (rotor sandwich).
    Then per-component Lloyd-Max quantization on the rotated multivector.
    """

    def __init__(self, d: int, bits: int, seed: int = 42,
                 grade_bits: Optional[Dict[str, int]] = None,
                 device: str = "cpu"):
        """
        Args:
            d: original vector dimension
            bits: default bits per component
            grade_bits: optional per-grade bit override, e.g.
                        {'scalar': 2, 'vector': 3, 'bivector': 2, 'trivector': 1}
            seed: random seed
            device: torch device
        """
        super().__init__()
        self.d = d
        self.bits = bits
        self.device = device

        # Compute how many multivector groups we need
        self.n_groups = (d + 2) // 3  # ceil(d/3)
        self.mv_dim = self.n_groups * MV_DIM  # total components

        # Grade-aware bit allocation
        # Only store non-zero grades: the rotor sandwich R v R̃ of a grade-1
        # vector produces ONLY odd grades (vector + trivector). Scalar and
        # bivector are mathematically guaranteed to be zero — don't waste
        # storage on them. This cuts stored indices from 8/group to 4/group.
        if grade_bits is None:
            grade_bits = {
                'vector': bits,
                'trivector': max(bits - 1, 1),
            }
        self.grade_bits = grade_bits

        # Create per-grade codebooks
        # d_eff determines the Lloyd-Max Gaussian σ = 1/√d_eff
        d_eff_vector = d       # vector grades: σ ≈ 1/√d
        d_eff_trivector = max(d // 2, 8)  # trivector: slightly wider distribution
        self.codebooks = nn.ModuleDict()
        for grade_name, gb in grade_bits.items():
            if grade_name == 'trivector':
                cb = LloydMaxCodebook(d_eff_trivector, gb)
            else:
                cb = LloydMaxCodebook(d_eff_vector, gb)
            self.register_buffer(f'centroids_{grade_name}',
                                 cb.centroids.to(device))

        # Only quantize non-zero grades (vector + trivector)
        # [scalar, e1, e2, e3, e12, e13, e23, e123]
        self.grade_map = {
            'vector':   [1, 2, 3],
            'trivector': [7],
        }

        # Pre-compute random rotors (one per group for decorrelation)
        rotors = []
        for i in range(self.n_groups):
            r = make_random_rotor((), device=device, seed=seed + i)
            rotors.append(r)
        self.register_buffer('rotors', torch.stack(rotors))  # (n_groups, 8)

    def _apply_rotors(self, mv: torch.Tensor) -> torch.Tensor:
        """Apply per-group rotor sandwich: R_i x_i R̃_i"""
        # mv: (..., n_groups, 8)
        # rotors: (n_groups, 8) → broadcast over batch dims
        return rotor_sandwich(self.rotors, mv)

    def _unapply_rotors(self, mv: torch.Tensor) -> torch.Tensor:
        """Inverse rotor sandwich: R̃_i x_i R_i"""
        rotor_rev = reverse(self.rotors)
        return rotor_sandwich(rotor_rev, mv)

    def _quantize_grade(self, x: torch.Tensor, grade_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize components of a specific grade."""
        centroids = getattr(self, f'centroids_{grade_name}')
        diffs = x.unsqueeze(-1) - centroids  # (..., n_components, n_levels)
        indices = diffs.abs().argmin(dim=-1)
        x_q = centroids[indices]
        return x_q, indices

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Quantize vectors via rotor + grade-aware Lloyd-Max.

        x: (..., d) input vectors
        Returns: (mv_q, indices_dict)
        """
        # Normalize to unit vectors (store norms separately)
        norms = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x / norms

        # Embed as multivectors
        mv = embed_vectors_as_multivectors(x_unit)  # (..., n_groups, 8)

        # Apply rotor decorrelation
        mv_rot = self._apply_rotors(mv)

        # Grade-aware quantization
        mv_q = torch.zeros_like(mv_rot)
        all_indices = {}

        for grade_name, component_indices in self.grade_map.items():
            grade_data = mv_rot[..., component_indices]  # (..., n_groups, n_components)
            flat = grade_data.reshape(*grade_data.shape[:-1], -1)
            q_flat, idx = self._quantize_grade(flat, grade_name)
            q_data = q_flat.reshape_as(grade_data)
            mv_q[..., component_indices] = q_data
            all_indices[grade_name] = idx

        # Store norms in indices for dequantize
        all_indices['_norms'] = norms.squeeze(-1)

        return mv_q, all_indices

    def dequantize(self, indices: dict) -> torch.Tensor:
        """Reconstruct vectors from quantized indices."""
        sample_centroids = getattr(self, 'centroids_vector')
        vector_idx = indices['vector']
        flat_batch = vector_idx.shape[0] if vector_idx.dim() >= 1 else 1

        mv_q = torch.zeros(flat_batch, self.n_groups, MV_DIM,
                           dtype=sample_centroids.dtype,
                           device=sample_centroids.device)

        for grade_name, component_indices in self.grade_map.items():
            if grade_name.startswith('_'):
                continue
            centroids = getattr(self, f'centroids_{grade_name}')
            idx = indices[grade_name]
            values = centroids[idx]
            n_components = len(component_indices)
            values = values.reshape(flat_batch, self.n_groups, n_components)
            mv_q[..., component_indices] = values

        # Undo rotor rotation
        mv_recon = self._unapply_rotors(mv_q)

        # Extract unit vectors and rescale by stored norms
        x_hat = extract_vectors_from_multivectors(mv_recon, self.d)
        if '_norms' in indices:
            norms = indices['_norms']
            if norms.dim() < x_hat.dim():
                norms = norms.unsqueeze(-1)
            x_hat = x_hat * norms

        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Full quantize-dequantize cycle."""
        mv_q, indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        return x_hat, indices


class RotorQuantProd(nn.Module):
    """
    Stage 1 + Stage 2: Unbiased inner product quantizer.

    Uses (b-1)-bit rotor MSE quantizer + 1-bit QJL on residuals.
    The QJL operates in the original vector space (not multivector space)
    since inner products are computed there.
    """

    def __init__(self, d: int, bits: int, qjl_dim: Optional[int] = None,
                 seed: int = 42, device: str = "cpu"):
        super().__init__()
        self.d = d
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.qjl_dim = qjl_dim or d
        self.device = device

        # Stage 1: Rotor MSE quantizer
        self.mse = RotorQuantMSE(d, self.mse_bits, seed=seed, device=device)

        # Stage 2: QJL projection matrix (same as TurboQuant)
        gen = torch.Generator(device='cpu')
        gen.manual_seed(seed + 1)
        S = torch.randn(self.qjl_dim, d, generator=gen)
        self.register_buffer("S", S.to(device))

    def quantize(self, x: torch.Tensor) -> dict:
        """Full RotorQuant quantization."""
        # Stage 1: Rotor MSE
        x_hat, mse_indices = self.mse(x)

        # Residual
        residual = x - x_hat
        residual_norm = torch.norm(residual, dim=-1, keepdim=True)

        # Stage 2: QJL sign quantization on residual
        projected = residual @ self.S.T
        qjl_signs = torch.sign(projected)
        qjl_signs[qjl_signs == 0] = 1.0

        return {
            'mse_indices': mse_indices,
            'qjl_signs': qjl_signs,
            'residual_norm': residual_norm.squeeze(-1),
        }

    def dequantize(self, compressed: dict) -> torch.Tensor:
        """Reconstruct from MSE component."""
        return self.mse.dequantize(compressed['mse_indices'])

    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        """
        Unbiased inner product estimate: <y, x>.

        Same QJL correction as TurboQuant — the math doesn't change
        because QJL operates in vector space.
        """
        x_mse = self.mse.dequantize(compressed['mse_indices'])
        term1 = (y * x_mse).sum(dim=-1)

        y_projected = y @ self.S.T
        qjl_ip = (y_projected * compressed['qjl_signs']).sum(dim=-1)

        m = self.qjl_dim
        correction_scale = math.sqrt(math.pi / 2) / m
        term2 = compressed['residual_norm'] * correction_scale * qjl_ip

        return term1 + term2

    def forward(self, x: torch.Tensor) -> dict:
        return self.quantize(x)


class RotorQuantKVCache:
    """
    KV cache using RotorQuant compression.
    Drop-in replacement for TurboQuantKVCache.
    """

    def __init__(self, d_key: int, d_value: int, bits: int = 3,
                 seed: int = 42, device: str = "cpu"):
        self.d_key = d_key
        self.d_value = d_value
        self.bits = bits
        self.device = device

        self.key_quantizer = RotorQuantProd(d_key, bits, seed=seed, device=device)
        self.value_quantizer = RotorQuantMSE(d_value, bits, seed=seed + 100, device=device)

        self.key_cache = []
        self.value_cache = []

    def append(self, keys: torch.Tensor, values: torch.Tensor):
        flat_keys = keys.reshape(-1, self.d_key)
        flat_values = values.reshape(-1, self.d_value)

        compressed_keys = self.key_quantizer.quantize(flat_keys)
        _, value_indices = self.value_quantizer(flat_values)

        self.key_cache.append(compressed_keys)
        self.value_cache.append(value_indices)

    def attention_scores(self, queries: torch.Tensor) -> torch.Tensor:
        scores = []
        for cached in self.key_cache:
            s = self.key_quantizer.inner_product(queries, cached)
            scores.append(s)
        return torch.cat(scores, dim=-1) if scores else torch.tensor([])

    def get_values(self) -> torch.Tensor:
        values = []
        for indices in self.value_cache:
            v = self.value_quantizer.dequantize(indices)
            values.append(v)
        return torch.cat(values, dim=0) if values else torch.tensor([])

    def __len__(self):
        return sum(
            c['qjl_signs'].shape[0] for c in self.key_cache
        ) if self.key_cache else 0
```

---
### FILE: turboquant/clifford.py
```python
"""
Clifford algebra Cl(3,0) for RotorQuant.

Multivector basis: [1, e1, e2, e3, e12, e13, e23, e123]
                    grade-0  grade-1     grade-2        grade-3

The geometric product table is hardcoded for GPU efficiency.
Rotors R = exp(B/2) where B is a bivector — they act via R x R̃
and naturally preserve inner products, norms, and algebraic structure.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


# Cl(3,0) basis element indices
S, E1, E2, E3, E12, E13, E23, E123 = range(8)
MV_DIM = 8  # 2^3 components for Cl(3,0)


def geometric_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Full Cl(3,0) geometric product: a * b

    Input:  a, b of shape (..., 8)
    Output: result of shape (..., 8)

    Multiplication table for Cl(3,0) with signature (+,+,+):
        e_i * e_i = +1  for i in {1,2,3}
        e_i * e_j = -e_j * e_i  for i != j
    """
    # Unbind components
    a0, a1, a2, a3, a12, a13, a23, a123 = a.unbind(dim=-1)
    b0, b1, b2, b3, b12, b13, b23, b123 = b.unbind(dim=-1)

    # Grade 0 (scalar)
    r0 = (a0*b0 + a1*b1 + a2*b2 + a3*b3
           - a12*b12 - a13*b13 - a23*b23 - a123*b123)

    # Grade 1 (vectors)
    r1 = (a0*b1 + a1*b0 - a2*b12 + a12*b2 - a3*b13 + a13*b3
           + a23*b123 + a123*b23)
    r2 = (a0*b2 + a2*b0 + a1*b12 - a12*b1 - a3*b23 + a23*b3
           - a13*b123 - a123*b13)
    r3 = (a0*b3 + a3*b0 + a1*b13 - a13*b1 + a2*b23 - a23*b2
           + a12*b123 + a123*b12)

    # Grade 2 (bivectors)
    r12 = (a0*b12 + a12*b0 + a1*b2 - a2*b1 + a13*b23 - a23*b13
            + a3*b123 - a123*b3)
    r13 = (a0*b13 + a13*b0 + a1*b3 - a3*b1 - a12*b23 + a23*b12
            - a2*b123 + a123*b2)
    r23 = (a0*b23 + a23*b0 + a2*b3 - a3*b2 + a12*b13 - a13*b12
            + a1*b123 - a123*b1)

    # Grade 3 (pseudoscalar)
    r123 = (a0*b123 + a123*b0 + a1*b23 - a23*b1 - a2*b13 + a13*b2
             + a3*b12 - a12*b3)

    return torch.stack([r0, r1, r2, r3, r12, r13, r23, r123], dim=-1)


def reverse(x: torch.Tensor) -> torch.Tensor:
    """
    Clifford reverse (reversion) x̃: reverses the order of basis vectors.

    Grade 0, 1: unchanged (sign = +1)
    Grade 2:    negated   (sign = -1)
    Grade 3:    negated   (sign = -1)

    This is used for rotor conjugation: R x R̃
    """
    signs = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1],
                         dtype=x.dtype, device=x.device)
    return x * signs


def multivector_norm_sq(x: torch.Tensor) -> torch.Tensor:
    """||x||² = <x x̃>_0  (scalar part of x * reverse(x))"""
    x_rev = reverse(x)
    product = geometric_product(x, x_rev)
    return product[..., 0]  # scalar part


def make_rotor(bivector: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Create a rotor R = cos(θ/2) + sin(θ/2) * B̂  where B̂ is a unit bivector.

    bivector: (..., 3) — coefficients for [e12, e13, e23]
    angle:    (...,)   — rotation angle in radians

    Returns: (..., 8) multivector rotor
    """
    # Normalize bivector
    bv_norm = bivector.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    bv_hat = bivector / bv_norm

    half_angle = angle.unsqueeze(-1) / 2
    cos_ha = torch.cos(half_angle)
    sin_ha = torch.sin(half_angle)

    # R = cos(θ/2) + sin(θ/2) * (b12*e12 + b13*e13 + b23*e23)
    rotor = torch.zeros(*bivector.shape[:-1], 8, dtype=bivector.dtype, device=bivector.device)
    rotor[..., S] = cos_ha.squeeze(-1)
    rotor[..., E12] = sin_ha.squeeze(-1) * bv_hat[..., 0]
    rotor[..., E13] = sin_ha.squeeze(-1) * bv_hat[..., 1]
    rotor[..., E23] = sin_ha.squeeze(-1) * bv_hat[..., 2]
    return rotor


def make_random_rotor(shape: Tuple[int, ...], device='cpu', seed=None) -> torch.Tensor:
    """
    Generate a random rotor via random bivector + random angle.
    Returns a normalized rotor R with R R̃ = 1.
    """
    gen = torch.Generator(device='cpu')
    if seed is not None:
        gen.manual_seed(seed)

    # Random bivector direction
    full_shape = list(shape) + [3]
    bv = torch.randn(full_shape, generator=gen).to(device)
    # Random angle in [0, 2π)
    angle_shape = list(shape) if shape else [1]
    angle = torch.rand(angle_shape, generator=gen).to(device) * 2 * math.pi

    rotor = make_rotor(bv, angle)
    # Normalize: R / sqrt(R R̃)
    norm = multivector_norm_sq(rotor).abs().sqrt().unsqueeze(-1).clamp(min=1e-8)
    return rotor / norm


def rotor_sandwich(rotor: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Apply rotor sandwich product: R x R̃
    This rotates x while preserving all algebraic structure.
    """
    rotor_rev = reverse(rotor)
    return geometric_product(geometric_product(rotor, x), rotor_rev)


def embed_vectors_as_multivectors(v: torch.Tensor) -> torch.Tensor:
    """
    Embed d-dimensional vectors into Cl(3,0) multivectors.

    For d divisible by 3: pack into grade-1 components (e1, e2, e3).
    For d not divisible by 3: also use scalar and bivector grades.

    v: (..., d) → (..., d_groups, 8)
    """
    d = v.shape[-1]
    # Pad to multiple of 3 if needed
    pad = (3 - d % 3) % 3
    if pad > 0:
        v = torch.nn.functional.pad(v, (0, pad))
    d_padded = v.shape[-1]
    n_groups = d_padded // 3

    # Reshape into groups of 3
    v_grouped = v.reshape(*v.shape[:-1], n_groups, 3)

    # Create multivectors with grade-1 components
    mv = torch.zeros(*v_grouped.shape[:-1], 8, dtype=v.dtype, device=v.device)
    mv[..., E1] = v_grouped[..., 0]
    mv[..., E2] = v_grouped[..., 1]
    mv[..., E3] = v_grouped[..., 2]

    return mv


def extract_vectors_from_multivectors(mv: torch.Tensor, orig_dim: int) -> torch.Tensor:
    """
    Extract d-dimensional vectors from Cl(3,0) multivectors.
    Inverse of embed_vectors_as_multivectors.

    mv: (..., n_groups, 8) → (..., d)
    """
    v = torch.stack([mv[..., E1], mv[..., E2], mv[..., E3]], dim=-1)
    v = v.reshape(*mv.shape[:-2], -1)
    return v[..., :orig_dim]
```

---
### FILE: turboquant/benchmark_perplexity.py
```python
"""
Perplexity benchmark: RotorQuant vs FP16 baseline on wikitext-2.

Measures the actual language modeling quality degradation from KV cache
quantization — the standard metric used in TurboQuant, KIVI, KVQuant, etc.

Method: run model forward pass on wikitext-2 test set, compute perplexity.
For RotorQuant: patch DynamicCache to quantize keys post-prefill (same
strategy as poc_high_context.py).

Usage:
    python -m turboquant.benchmark_perplexity
    python -m turboquant.benchmark_perplexity --model Qwen/Qwen2.5-7B-Instruct --bits 2 3 4
"""

import torch
import torch.nn.functional as F
import math
import time
import gc
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_perplexity(model, tokenizer, dataset_text, max_length=2048, stride=512, device="cuda"):
    """Compute perplexity on a text using sliding window.

    Uses the standard approach from Hugging Face perplexity docs:
    slide a window of max_length tokens with overlap of stride.
    """
    encodings = tokenizer(dataset_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.shape[1]

    nlls = []
    n_tokens = 0

    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - begin - 1  # tokens we score

        chunk_ids = input_ids[:, begin:end]

        with torch.no_grad():
            outputs = model(chunk_ids, use_cache=False)
            logits = outputs.logits

        # Only score the non-overlapping part (except first window)
        shift = max(0, max_length - stride) if begin > 0 else 0
        shift_logits = logits[:, shift:-1, :].contiguous()
        shift_labels = chunk_ids[:, shift + 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        n_scored = shift_labels.numel()
        nlls.append(loss.item())
        n_tokens += n_scored

        if end >= seq_len:
            break

    ppl = math.exp(sum(nlls) / n_tokens)
    return ppl, n_tokens


def compute_perplexity_with_rq(model, tokenizer, dataset_text, bits=3,
                                max_length=2048, stride=512, device="cuda"):
    """Compute perplexity with RotorQuant KV cache compression.

    Quantizes keys during the forward pass so attention sees quantized keys.
    This measures the actual quality impact of KV cache quantization.
    """
    from transformers import DynamicCache
    from turboquant.rotorquant import RotorQuantMSE
    from turboquant.triton_kernels import triton_rotor_full_fused, pack_rotors_for_triton

    compressors = {}

    def compress(ks, li):
        D = ks.shape[-1]
        if li not in compressors:
            rq = RotorQuantMSE(D, bits, seed=li * 1000, device=device)
            pk = pack_rotors_for_triton(rq.rotors).to(device)
            compressors[li] = (rq, pk)
        rq, pk = compressors[li]
        flat = ks.reshape(-1, D)
        kq = triton_rotor_full_fused(
            flat, pk, None,
            getattr(rq, 'centroids_vector'),
            None,
            getattr(rq, 'centroids_trivector'),
        )
        return kq.to(ks.dtype).reshape(ks.shape)

    _orig = DynamicCache.update

    def _patched(self, ks, vs, li, ck=None):
        # Quantize keys during forward pass — attention sees quantized keys
        kq = compress(ks, li)
        return _orig(self, kq, vs, li, ck)

    encodings = tokenizer(dataset_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.shape[1]

    nlls = []
    n_tokens = 0

    DynamicCache.update = _patched

    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        chunk_ids = input_ids[:, begin:end]

        with torch.no_grad():
            outputs = model(chunk_ids, use_cache=True)
            logits = outputs.logits

        shift = max(0, max_length - stride) if begin > 0 else 0
        shift_logits = logits[:, shift:-1, :].contiguous()
        shift_labels = chunk_ids[:, shift + 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        nlls.append(loss.item())
        n_tokens += shift_labels.numel()

        del outputs
        torch.cuda.empty_cache()

        if end >= seq_len:
            break

    DynamicCache.update = _orig

    ppl = math.exp(sum(nlls) / n_tokens)
    return ppl, n_tokens


def main():
    parser = argparse.ArgumentParser(description="RotorQuant Perplexity Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="Limit dataset tokens (0=full test set)")
    args = parser.parse_args()

    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    import logging; logging.disable(logging.WARNING)

    print()
    print("=" * 70)
    print("  RotorQuant Perplexity Benchmark (wikitext-2)")
    print(f"  Model: {args.model}")
    print(f"  Bits: {args.bits}")
    print(f"  Window: {args.max_length}, stride: {args.stride}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)

    # Load dataset
    print("\nLoading wikitext-2...", flush=True)
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    # Load model
    print("Loading model...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
        dtype=torch.float16,
    )
    model.eval()

    if args.max_tokens > 0:
        tokens = tokenizer(text, return_tensors="pt")
        text = tokenizer.decode(tokens.input_ids[0][:args.max_tokens])

    total_tokens = len(tokenizer(text).input_ids)
    print(f"Dataset: {total_tokens:,} tokens")
    print()

    # FP16 baseline
    print("Computing FP16 baseline perplexity...", flush=True)
    t0 = time.perf_counter()
    ppl_fp16, n_tok = compute_perplexity(
        model, tokenizer, text,
        max_length=args.max_length, stride=args.stride,
    )
    t_fp16 = time.perf_counter() - t0
    print(f"  FP16:     PPL = {ppl_fp16:.2f}  ({n_tok:,} tokens, {t_fp16:.1f}s)")
    print()

    # RotorQuant: roundtrip during forward pass (worst case — compounding)
    print("  Roundtrip quantization (keys quantized during forward pass):")
    print(f"  {'Method':>15s}  {'PPL':>8s}  {'Delta':>8s}  {'%change':>8s}  {'Time':>8s}")
    print(f"  {'─'*15}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    print(f"  {'FP16':>15s}  {ppl_fp16:>8.2f}  {'—':>8s}  {'—':>8s}  {t_fp16:>6.1f}s")

    for bits in args.bits:
        torch.cuda.empty_cache()
        gc.collect()

        t0 = time.perf_counter()
        ppl_rq, n_tok = compute_perplexity_with_rq(
            model, tokenizer, text, bits=bits,
            max_length=args.max_length, stride=args.stride,
        )
        t_rq = time.perf_counter() - t0

        delta = ppl_rq - ppl_fp16
        pct = (ppl_rq - ppl_fp16) / ppl_fp16 * 100

        print(f"  {'RQ ' + str(bits) + '-bit':>15s}  {ppl_rq:>8.2f}  {delta:>+8.2f}  {pct:>+7.1f}%  {t_rq:>6.1f}s")

    print()
    print("  NOTE: Roundtrip quantization degrades perplexity for ALL methods")
    print("  (TurboQuant roundtrip is even worse: PPL ~12,000).")
    print("  Google's 'zero accuracy loss' requires the fused attention kernel")
    print("  which computes Q@K^T directly from compressed keys without roundtrip.")
    print("  The post-prefill strategy (used for generation) avoids this by running")
    print("  prefill at full FP16 precision.")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

---
### REFERENCE: QJL score kernel (from amirzandieh/QJL — qjl_score_kernel.cu)
This CUDA kernel shows how the QJL two-term estimator is computed. We need the equivalent in Triton.
```cuda
#include <torch/extension.h>
#include <math_constants.h>
#include <cmath>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define EMB_DIM 128
#define FULL_MASK 0xffffffff


template <typename T>
__device__ float convert_to_float(T value) {
    // Return 0 by default, indicating misuse if not specialized correctly.
    return 0.0f;
}

template <>
__device__ float convert_to_float<c10::Half>(c10::Half value) {
    return __half2float(value);
}

template <>
__device__ float convert_to_float<float>(float value) {
    return value; 
}

template <>
__device__ float convert_to_float<at::BFloat16>(at::BFloat16 value) {
    return static_cast<float>(value); 
}

__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(FULL_MASK, val, offset);
  return val;
}

template<typename T, typename Tproj>
__global__ void calc_score_kernel(
    T* query_states,
    const uint8_t* key_quant,
    const uint8_t* key_outlier_quant,
    T* key_norm,
    T* key_outlier_norm,
    const uint8_t* outlier_indices,
    const float* query_sketch,
    const Tproj* rand_prj,
    float* scores,
    int batch_size, int head_size, int n_size, int group_size, int sketch_dim, int outlier_sketch_dim, int emb_dim,
    int outlier_counts) {

    size_t bh = blockIdx.x;
    size_t n = blockIdx.y;
    size_t threadLane = threadIdx.x;
    size_t wIdx = threadIdx.y;
    size_t gIdx = blockIdx.z * WARP_SIZE;

    int hash_dim = sketch_dim/8;
    int outlier_hash_dim = outlier_sketch_dim/8;

    int base_index_outlier_indices = (bh * n_size * outlier_counts) + (n * outlier_counts);
    const uint8_t* outlier_ind = outlier_indices + base_index_outlier_indices;

    int base_index_query_sketch = (bh * sketch_dim);
    const float* q_sketch = query_sketch + base_index_query_sketch;

    int base_index_key_quant = (bh * n_size * group_size * hash_dim) + (n * group_size * hash_dim) + (gIdx * hash_dim);
    const uint8_t* k_quant = key_quant + base_index_key_quant;

    int base_index_outlier_quant = (bh * n_size * group_size * outlier_hash_dim) + (n * group_size * outlier_hash_dim) + (gIdx * outlier_hash_dim);
    const uint8_t* outlier_quant = key_outlier_quant + base_index_outlier_quant;

    int base_index_key_norm = (bh * n_size * group_size) + (n * group_size) + gIdx;
    const T* k_norm = key_norm + base_index_key_norm;
    const T* outlier_norm = key_outlier_norm + base_index_key_norm;

    int base_index_query_states = (bh * emb_dim);
    const T* query = query_states + base_index_query_states;

    // load query states into shared memory
    __shared__ float shared_query[EMB_DIM];
    size_t tIdx = wIdx * WARP_SIZE + threadLane;
    for (size_t tile_idx{tIdx}; tile_idx < emb_dim; tile_idx += (WARP_SIZE * WARPS_PER_BLOCK)) {
        shared_query[tile_idx] = convert_to_float<T>(query[tile_idx]);
    }
    // load outlier indices into shared buffer
    __shared__ uint8_t shared_outlier_ind[WARP_SIZE];
    for (size_t tile_idx{tIdx}; tile_idx < outlier_counts; tile_idx += (WARP_SIZE * WARPS_PER_BLOCK)) {
        shared_outlier_ind[tile_idx] = outlier_ind[tile_idx];
    }
    // allocate shared memory to inner products of quantized keys or outliers with query_sketch
    __shared__ float shared_innprod[WARP_SIZE];
    __shared__ float shared_outlier_innprod[WARP_SIZE];
    if (wIdx == 0) {
        shared_innprod[threadLane] = 0.0;
        shared_outlier_innprod[threadLane] = 0.0;
    }
    __syncthreads();

    // reserve shared memory for a block of query sketch and query outlier sketch
    __shared__ float shared_q_sketch[WARP_SIZE][8];
    __shared__ float shared_q_outliers_sketch[WARP_SIZE][8];
    for (size_t chnl_tile{0}; chnl_tile < sketch_dim; chnl_tile += (8*WARP_SIZE)){
        // load a block of query sketch and compute query outlier sketch
        for (size_t q_idx{tIdx}; q_idx < (8*WARP_SIZE); q_idx += (WARP_SIZE * WARPS_PER_BLOCK)) {
            shared_q_sketch[q_idx/8][q_idx%8] = 0.0;
            shared_q_outliers_sketch[q_idx/8][q_idx%8] = 0.0;
            if (chnl_tile+q_idx < sketch_dim){
                shared_q_sketch[q_idx/8][q_idx%8] = q_sketch[chnl_tile+q_idx];
                for (size_t i{0}; i < outlier_counts; i++){
                    int otlr_idx = shared_outlier_ind[i];
                    shared_q_outliers_sketch[q_idx/8][q_idx%8] += shared_query[otlr_idx] * convert_to_float<Tproj>(rand_prj[(otlr_idx * sketch_dim) + chnl_tile+q_idx]); // convert_to_float(const_query[bh][otlr_idx])
                }
            }
        }

        for (size_t grp_tile{wIdx}; grp_tile < WARP_SIZE; grp_tile += WARPS_PER_BLOCK) {
            // load key quant and outlier quant
            uint8_t key_quant_buffer = k_quant[grp_tile*hash_dim + chnl_tile/8 + threadLane];
            uint8_t outlier_quant_buffer = 0;
            if (chnl_tile + 8*threadLane < outlier_sketch_dim){
                outlier_quant_buffer = outlier_quant[grp_tile*outlier_hash_dim + chnl_tile/8 + threadLane];
            }
            __syncthreads();

            float k_inner_prod = 0.0;
            float outlier_inner_prod = 0.0;
            for (int shift = 0; shift < 8; shift++) {
                float q_sketch_val = shared_q_sketch[threadLane][shift] - shared_q_outliers_sketch[threadLane][shift];
                k_inner_prod += (((key_quant_buffer >> shift)&1) ? q_sketch_val :-q_sketch_val);
                if (chnl_tile + 8*threadLane < outlier_sketch_dim) {
                    float q_otlr_sketch_val = shared_q_outliers_sketch[threadLane][shift];
                    outlier_inner_prod += (((outlier_quant_buffer >> shift)&1) ? q_otlr_sketch_val :-q_otlr_sketch_val);
                }
            }
            __syncthreads();

            k_inner_prod = warpReduceSum(k_inner_prod);
            outlier_inner_prod = warpReduceSum(outlier_inner_prod);
            __syncthreads();
            if (threadLane == 0) {
                shared_innprod[grp_tile] += k_inner_prod;
                shared_outlier_innprod[grp_tile] += outlier_inner_prod;
            }
        }
        __syncthreads();
    }
    __syncthreads();

    if (gIdx+threadLane >= group_size) return;
    if (wIdx == 0) {
        float scl = sqrtf(M_PI_2) / static_cast<float>(sketch_dim);
        float scl_otlr = sqrtf(M_PI_2) / static_cast<float>(outlier_sketch_dim);
        float norm_otlr = convert_to_float<T>(outlier_norm[threadLane]);
        float norm_k = sqrtf(pow(convert_to_float<T>(k_norm[threadLane]), 2) - pow(norm_otlr, 2));
        float score = scl * norm_k * shared_innprod[threadLane] + scl_otlr * norm_otlr * shared_outlier_innprod[threadLane];
        scores[(bh * n_size * group_size) + (n * group_size) + gIdx + threadLane] = score;
    }
}


template <typename T, typename Tproj>
torch::Tensor QJLScoreCudaTemplate(
    torch::Tensor key_quant,
    torch::Tensor key_outlier_quant,
    torch::Tensor key_norm,
    torch::Tensor key_outlier_norm,
    torch::Tensor outlier_indices,
    torch::Tensor query_sketch,
    torch::Tensor query_states,
    torch::Tensor rand_prj) {

    auto options = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);

    int batch = key_quant.size(0);
    int head = key_quant.size(1);
    int n = key_quant.size(2);
    int group_size = key_quant.size(3);
    int emb_dim = query_states.size(3);
    int sketch_dim = rand_prj.size(1);
    int outlier_sketch_dim = 8*key_outlier_quant.size(4);
    int outlier_counts = outlier_indices.size(3);

    auto scores = torch::zeros({batch, head, n * group_size, 1}, options).contiguous();
    
    auto query_states_ptr = query_states.data_ptr<T>();
    auto key_norm_ptr = key_norm.data_ptr<T>();
    auto key_outlier_norm_ptr = key_outlier_norm.data_ptr<T>();
    auto rand_prj_ptr = rand_prj.data_ptr<Tproj>();

    int blocksPerGroup = (group_size + WARP_SIZE - 1) / WARP_SIZE;
    dim3 numBlocks(batch * head, n, blocksPerGroup);
    dim3 threadsPerBlockDim(WARP_SIZE, WARPS_PER_BLOCK, 1);

    calc_score_kernel<<<numBlocks, threadsPerBlockDim>>>(
        query_states_ptr,
        key_quant.data_ptr<uint8_t>(),
        key_outlier_quant.data_ptr<uint8_t>(),
        key_norm_ptr,
        key_outlier_norm_ptr,
        outlier_indices.data_ptr<uint8_t>(),
        query_sketch.data_ptr<float>(),
        rand_prj_ptr,
        scores.data_ptr<float>(),
        batch, head, n, group_size, sketch_dim, outlier_sketch_dim, emb_dim, outlier_counts);

    return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qjl_score_cuda_half_half", &QJLScoreCudaTemplate<c10::Half, c10::Half>, "Cuda kernel to calculate scores fully parallel using Half precision",
          py::arg("key_quant"),
          py::arg("key_outlier_quant"),
          py::arg("key_norm"),
          py::arg("key_outlier_norm"),
          py::arg("outlier_indices"),
          py::arg("query_sketch"),
          py::arg("query_states"),
          py::arg("rand_prj"));

    m.def("qjl_score_cuda_half_float", &QJLScoreCudaTemplate<c10::Half, float>, "Cuda kernel to calculate scores fully parallel using Half to Float precision",
          py::arg("key_quant"),
          py::arg("key_outlier_quant"),
          py::arg("key_norm"),
          py::arg("key_outlier_norm"),
          py::arg("outlier_indices"),
          py::arg("query_sketch"),
          py::arg("query_states"),
          py::arg("rand_prj"));

    m.def("qjl_score_cuda_float_float", &QJLScoreCudaTemplate<float, float>, "Cuda kernel to calculate scores fully parallel using Float precision",
          py::arg("key_quant"),
          py::arg("key_outlier_quant"),
          py::arg("key_norm"),
          py::arg("key_outlier_norm"),
          py::arg("outlier_indices"),
          py::arg("query_sketch"),
          py::arg("query_states"),
          py::arg("rand_prj"));

    m.def("qjl_score_cuda_bf16_bf16", &QJLScoreCudaTemplate<at::BFloat16, at::BFloat16>, "Cuda kernel to calculate scores fully parallel using BF16 precision",
          py::arg("key_quant"),
          py::arg("key_outlier_quant"),
          py::arg("key_norm"),
          py::arg("key_outlier_norm"),
          py::arg("outlier_indices"),
          py::arg("query_sketch"),
          py::arg("query_states"),
          py::arg("rand_prj"));

    m.def("qjl_score_cuda_bf16_float", &QJLScoreCudaTemplate<at::BFloat16, float>, "Cuda kernel to calculate scores fully parallel using BF16 to Float precision",
          py::arg("key_quant"),
          py::arg("key_outlier_quant"),
          py::arg("key_norm"),
          py::arg("key_outlier_norm"),
          py::arg("outlier_indices"),
          py::arg("query_sketch"),
          py::arg("query_states"),
          py::arg("rand_prj"));
}
```

---
### REFERENCE: TurboQuant Triton attention (from dejan.ai turboquant — triton_attention.py)
This is the MSE-only fused attention kernel that TurboQuant uses. Our Triton kernel is based on this.
```python
"""
Triton kernel for fused quantized attention scores.

Instead of: dequantize keys to fp16 → Q @ K^T  (loads fp16 keys from HBM)
We do:      Q_rotated @ gather(centroids, key_indices) * norms   (loads uint8 indices)

Memory bandwidth savings: ~4x (uint8 + small centroid table vs fp16 keys)

Key math:
    <q, R^T @ centroids[idx]> = <R @ q, centroids[idx]>
    
    So pre-rotate query once (one matmul), then the per-KV-position work
    is just: score[s] = norm[s] * sum_d(q_rot[d] * centroids[idx[s,d]]) / sqrt(d)
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================================
# Triton kernel: fused gather-dot for quantized Q@K^T
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 32, "BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_S": 64, "BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_S": 128, "BLOCK_D": 64}, num_warps=8),
        triton.Config({"BLOCK_S": 64, "BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_S": 128, "BLOCK_D": 128}, num_warps=8),
    ],
    key=["seq_len", "head_dim"],
)
@triton.jit
def _fused_qk_scores_kernel(
    # Pre-rotated query: [BH_q, head_dim]  (BH_q = batch * n_q_heads)
    Q_ptr,
    # Compressed keys
    K_idx_ptr,    # [BH_kv, seq_len, head_dim] uint8
    K_norms_ptr,  # [BH_kv, seq_len] float16
    # Centroid table
    C_ptr,        # [n_levels] float32
    # Output scores
    Out_ptr,      # [BH_q, seq_len] float32
    # Dimensions
    seq_len,
    head_dim: tl.constexpr,
    n_q_heads,
    n_kv_heads,
    scale,        # 1/sqrt(head_dim)
    # Strides — Q: [BH_q, head_dim]
    stride_q_bh, stride_q_d,
    # Strides — K_idx: [BH_kv, seq_len, head_dim]
    stride_ki_bh, stride_ki_s, stride_ki_d,
    # Strides — K_norms: [BH_kv, seq_len]
    stride_kn_bh, stride_kn_s,
    # Strides — Out: [BH_q, seq_len]
    stride_o_bh, stride_o_s,
    # Block sizes (autotuned)
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute attention scores from pre-rotated queries and quantized keys.

    For each (query_head, kv_block):
        score[s] = key_norm[s] * sum_d(q_rot[d] * centroids[key_idx[s, d]]) * scale
    """
    pid_bh = tl.program_id(0)   # batch * query_head
    pid_s = tl.program_id(1)    # seq block

    # GQA: map query head → KV head
    batch_idx = pid_bh // n_q_heads
    q_head_idx = pid_bh % n_q_heads
    gqa_ratio = n_q_heads // n_kv_heads
    kv_head_idx = q_head_idx // gqa_ratio
    kv_bh = batch_idx * n_kv_heads + kv_head_idx

    # Sequence positions this program handles
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len

    # Accumulate dot product over head_dim in blocks
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    for d_start in range(0, head_dim, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < head_dim

        # Load query slice: Q[pid_bh, d_offs]
        q_ptrs = Q_ptr + pid_bh * stride_q_bh + d_offs * stride_q_d
        q_vals = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        # Load key indices: K_idx[kv_bh, s_offs, d_offs] → [BLOCK_S, BLOCK_D]
        ki_ptrs = (K_idx_ptr
                   + kv_bh * stride_ki_bh
                   + s_offs[:, None] * stride_ki_s
                   + d_offs[None, :] * stride_ki_d)
        combined_mask = s_mask[:, None] & d_mask[None, :]
        k_idx = tl.load(ki_ptrs, mask=combined_mask, other=0).to(tl.int32)

        # Gather centroids: C[k_idx] → [BLOCK_S, BLOCK_D]
        k_vals = tl.load(C_ptr + k_idx, mask=combined_mask, other=0.0).to(tl.float32)

        # Partial dot product: sum over D block
        acc += tl.sum(k_vals * q_vals[None, :], axis=1)

    # Load key norms: K_norms[kv_bh, s_offs]
    kn_ptrs = K_norms_ptr + kv_bh * stride_kn_bh + s_offs * stride_kn_s
    norms = tl.load(kn_ptrs, mask=s_mask, other=0.0).to(tl.float32)

    # Final score = norm * dot_product * scale
    scores = norms * acc * scale

    # Store
    o_ptrs = Out_ptr + pid_bh * stride_o_bh + s_offs * stride_o_s
    tl.store(o_ptrs, scores, mask=s_mask)


# ============================================================================
# Python wrapper
# ============================================================================

def fused_qk_scores(
    q_rotated: torch.Tensor,     # [batch, n_q_heads, q_len, head_dim] — pre-rotated
    key_indices: torch.Tensor,   # [batch, n_kv_heads, kv_len, head_dim] uint8
    key_norms: torch.Tensor,     # [batch, n_kv_heads, kv_len] float16
    centroids: torch.Tensor,     # [n_levels] float32
    scale: float,                # 1/sqrt(head_dim)
) -> torch.Tensor:
    """Compute attention scores Q @ K^T using compressed keys.

    Args:
        q_rotated: Query vectors pre-multiplied by rotation matrix Q^T.
                   Shape [batch, n_q_heads, q_len, head_dim]
        key_indices: Quantized key indices. [batch, n_kv_heads, kv_len, head_dim]
        key_norms: Key vector norms. [batch, n_kv_heads, kv_len]
        centroids: Lloyd-Max centroid values. [n_levels]
        scale: Attention scale factor (1/sqrt(head_dim))

    Returns:
        Attention scores [batch, n_q_heads, q_len, kv_len]
    """
    batch, n_q_heads, q_len, head_dim = q_rotated.shape
    _, n_kv_heads, kv_len, _ = key_indices.shape

    # For q_len > 1 (prefill), handle each query position
    # Reshape to [batch * n_q_heads * q_len, head_dim] for the kernel
    q_flat = q_rotated.reshape(batch * n_q_heads * q_len, head_dim).contiguous()
    ki_flat = key_indices.reshape(batch * n_kv_heads, kv_len, head_dim).contiguous()
    kn_flat = key_norms.reshape(batch * n_kv_heads, kv_len).contiguous()
    centroids = centroids.contiguous().float()

    # Output: [batch * n_q_heads * q_len, kv_len]
    out = torch.empty(batch * n_q_heads * q_len, kv_len,
                      device=q_rotated.device, dtype=torch.float32)

    # For the kernel, we treat each query position as a separate "head"
    # But GQA mapping needs to account for q_len grouping
    effective_q_heads = n_q_heads * q_len

    # Grid
    grid = (batch * effective_q_heads, triton.cdiv(kv_len, 64))  # 64 is a safe default

    _fused_qk_scores_kernel[grid](
        q_flat,
        ki_flat, kn_flat,
        centroids,
        out,
        kv_len,
        head_dim,
        effective_q_heads,
        n_kv_heads,
        scale,
        # Strides Q
        q_flat.stride(0), q_flat.stride(1),
        # Strides K_idx
        ki_flat.stride(0), ki_flat.stride(1), ki_flat.stride(2),
        # Strides K_norms
        kn_flat.stride(0), kn_flat.stride(1),
        # Strides Out
        out.stride(0), out.stride(1),
    )

    return out.reshape(batch, n_q_heads, q_len, kv_len)


# ============================================================================
# Self-test: verify Triton kernel matches PyTorch reference
# ============================================================================

def test_fused_kernel():
    """Compare fused Triton kernel against explicit dequantize + matmul."""
    import sys
    from turboquant_core import TurboQuantMSE

    torch.manual_seed(42)

    batch, n_q_heads, n_kv_heads = 1, 8, 4
    q_len, kv_len, head_dim = 1, 128, 256
    bits = 4

    # Create quantizer
    tq = TurboQuantMSE(d=head_dim, bits=bits, device="cuda")

    # Random Q and K
    q = torch.randn(batch, n_q_heads, q_len, head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, n_kv_heads, kv_len, head_dim, device="cuda", dtype=torch.float32)

    # Quantize K
    k_q = tq.quantize(k)
    k_indices = k_q["idx"]       # uint8
    k_norms = k_q["norms"]       # fp16

    # --- Reference: explicit dequantize + matmul ---
    k_deq = tq.dequantize(k_q)   # [batch, n_kv, kv_len, head_dim]
    # GQA expand
    gqa_ratio = n_q_heads // n_kv_heads
    k_expanded = k_deq.repeat_interleave(gqa_ratio, dim=1)
    scale = 1.0 / math.sqrt(head_dim)
    ref_scores = torch.matmul(q, k_expanded.transpose(2, 3)) * scale

    # --- Fused: pre-rotate query, then Triton kernel ---
    # Pre-rotate query: q_rot = q @ Q_T  (Q_T is the rotation matrix transpose)
    q_rot = q @ tq.Q_T.unsqueeze(0).unsqueeze(0)  # broadcast over batch, heads
    fused_scores = fused_qk_scores(q_rot, k_indices, k_norms, tq.centroids, scale)

    # Compare
    max_diff = (ref_scores - fused_scores).abs().max().item()
    mean_diff = (ref_scores - fused_scores).abs().mean().item()
    cos = torch.nn.functional.cosine_similarity(
        ref_scores.flatten().unsqueeze(0),
        fused_scores.flatten().unsqueeze(0)
    ).item()

    print(f"Fused kernel test (batch={batch}, q_heads={n_q_heads}, kv_heads={n_kv_heads}, "
          f"q_len={q_len}, kv_len={kv_len}, d={head_dim}, bits={bits}):")
    print(f"  Max diff:   {max_diff:.6f}")
    print(f"  Mean diff:  {mean_diff:.6f}")
    print(f"  Cosine sim: {cos:.6f}")
    print(f"  {'PASS' if cos > 0.999 else 'FAIL'}")
    print()
    return cos > 0.999


def benchmark_fused_vs_standard():
    """Benchmark fused kernel vs standard dequantize+matmul."""
    from turboquant_core import TurboQuantMSE

    torch.manual_seed(42)
    batch, n_q_heads, n_kv_heads = 1, 8, 4
    head_dim, bits = 256, 4
    scale = 1.0 / math.sqrt(head_dim)

    tq = TurboQuantMSE(d=head_dim, bits=bits, device="cuda")
    gqa_ratio = n_q_heads // n_kv_heads

    for kv_len in [128, 512, 1024, 2048, 4096]:
        q = torch.randn(batch, n_q_heads, 1, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, n_kv_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)

        # Quantize
        k_q = tq.quantize(k.float())
        k_deq = tq.dequantize(k_q).half()
        k_indices = k_q["idx"]
        k_norms = k_q["norms"]

        # Pre-rotate query
        q_rot = (q.float() @ tq.Q_T.unsqueeze(0).unsqueeze(0)).contiguous()

        # Warm up
        for _ in range(5):
            k_exp = k_deq.repeat_interleave(gqa_ratio, dim=1)
            _ = torch.matmul(q, k_exp.transpose(2, 3)) * scale
            _ = fused_qk_scores(q_rot, k_indices, k_norms, tq.centroids, scale)
        torch.cuda.synchronize()

        # Benchmark standard
        import time
        n_runs = 100
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            k_exp = k_deq.repeat_interleave(gqa_ratio, dim=1)
            _ = torch.matmul(q, k_exp.transpose(2, 3)) * scale
        torch.cuda.synchronize()
        t_std = (time.perf_counter() - t0) / n_runs * 1000

        # Benchmark fused
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = fused_qk_scores(q_rot, k_indices, k_norms, tq.centroids, scale)
        torch.cuda.synchronize()
        t_fused = (time.perf_counter() - t0) / n_runs * 1000

        speedup = t_std / t_fused
        print(f"  kv_len={kv_len:5d}  standard={t_std:.3f}ms  fused={t_fused:.3f}ms  "
              f"speedup={speedup:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("Triton fused attention kernel tests")
    print("=" * 60)

    ok = test_fused_kernel()
    if not ok:
        print("Kernel test FAILED — skipping benchmark")
        exit(1)

    print("Benchmarking fused vs standard attention scores:")
    benchmark_fused_vs_standard()
```

---
### REFERENCE: TurboQuant fused model integration (turboquant_fused.py)
```python
"""
Fused TurboQuant attention for Gemma 3.

Replaces standard Q@K^T with a Triton kernel that operates directly on
compressed uint8 key indices — never materializing fp16 keys.

Usage:
    from turboquant_fused import FusedTurboQuantRunner
    runner = FusedTurboQuantRunner(model, processor, bits=4)
    text = runner.generate("What is 2+2?", max_new_tokens=30)
"""

import torch
import math
from transformers import DynamicCache
from turboquant_core import TurboQuantMSE
from triton_attention import fused_qk_scores


class CompressedKVCache(DynamicCache):
    """KV cache that stores compressed keys (uint8 indices + norms).

    Keys are quantized on insertion. During attention, the fused Triton
    kernel reads compressed keys directly — no fp16 dequantization needed.

    Values are stored in fp16 (standard) since the softmax@V matmul
    benefits less from compression.
    """

    def __init__(self, quantizer: TurboQuantMSE):
        super().__init__()
        self.tq = quantizer
        # Per-layer compressed key storage
        self._compressed_keys: list[dict | None] = []

    def store_compressed_key(self, key_states: torch.Tensor, layer_idx: int):
        """Quantize and store key states. Called from patched attention."""
        while len(self._compressed_keys) <= layer_idx:
            self._compressed_keys.append(None)

        q = self.tq.quantize(key_states.float())

        if self._compressed_keys[layer_idx] is None:
            self._compressed_keys[layer_idx] = q
        else:
            prev = self._compressed_keys[layer_idx]
            self._compressed_keys[layer_idx] = {
                "idx": torch.cat([prev["idx"], q["idx"]], dim=2),
                "norms": torch.cat([prev["norms"], q["norms"]], dim=2),
            }

    def get_compressed_key(self, layer_idx: int) -> dict | None:
        if layer_idx < len(self._compressed_keys):
            return self._compressed_keys[layer_idx]
        return None

    def get_kv_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self._compressed_keys) and self._compressed_keys[layer_idx] is not None:
            return self._compressed_keys[layer_idx]["idx"].shape[2]
        return 0


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply RoPE — copied from transformers to avoid import issues."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA."""
    if n_rep == 1:
        return hidden_states
    batch, n_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, n_kv_heads * n_rep, slen, head_dim)


def make_fused_attention_forward(attn_module, cache: CompressedKVCache, quantizer: TurboQuantMSE, layer_index: int):
    """Create a replacement forward for a Gemma3 attention layer."""

    # Cache the rotation matrix for pre-rotating queries
    Q_T = quantizer.Q_T  # [head_dim, head_dim]
    centroids = quantizer.centroids
    head_dim = quantizer.d
    scale = 1.0 / math.sqrt(head_dim)
    n_heads = attn_module.num_heads
    # num_key_value_heads lives in config, not on the module
    cfg = attn_module.config
    n_kv_heads = getattr(cfg, 'num_key_value_heads', n_heads)
    n_kv_groups = n_heads // n_kv_heads
    layer_idx = layer_index  # passed in from enumeration

    # Check if this layer uses sliding window attention
    is_sliding = getattr(attn_module, 'is_sliding', False)
    sliding_window = getattr(attn_module, 'sliding_window', None)
    if is_sliding and sliding_window is None:
        # Try to get from config
        config = getattr(attn_module, 'config', None)
        if config:
            sliding_window = getattr(config, 'sliding_window', None)

    def fused_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Q/K/V projections
        query_states = attn_module.q_proj(hidden_states)
        key_states = attn_module.k_proj(hidden_states)
        value_states = attn_module.v_proj(hidden_states)

        # Reshape to [batch, n_heads, q_len, head_dim]
        query_states = query_states.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Store compressed keys
        cache.store_compressed_key(key_states, layer_idx)

        # Store values normally in the parent DynamicCache
        # We need to call the parent's update for values only
        # Hack: store values directly
        while len(cache.value_cache) <= layer_idx:
            cache.key_cache.append(torch.empty(0))  # placeholder
            cache.value_cache.append(torch.empty(0))
        if cache.value_cache[layer_idx].numel() == 0:
            cache.value_cache[layer_idx] = value_states
        else:
            cache.value_cache[layer_idx] = torch.cat(
                [cache.value_cache[layer_idx], value_states], dim=2
            )
        # Keep key_cache in sync for length tracking
        cache.key_cache[layer_idx] = cache.value_cache[layer_idx]  # dummy, same shape

        # Get full accumulated values
        full_values = cache.value_cache[layer_idx]
        kv_len = full_values.shape[2]

        # Get compressed keys
        compressed = cache.get_compressed_key(layer_idx)

        # --- Fused attention scores ---
        # Pre-rotate queries: q_rot = q @ Q_T
        q_rot = query_states.float() @ Q_T.unsqueeze(0).unsqueeze(0)

        # Use Triton kernel
        attn_weights = fused_qk_scores(
            q_rot, compressed["idx"], compressed["norms"],
            centroids, scale
        )

        # Apply attention mask (causal + sliding window if applicable)
        if attention_mask is not None:
            # attention_mask shape depends on transformers version
            # Typically [batch, 1, q_len, kv_len] or similar
            causal_mask = attention_mask
            if causal_mask.dim() == 4:
                attn_weights = attn_weights + causal_mask[:, :, :q_len, :kv_len]
            elif causal_mask.dim() == 2:
                attn_weights = attn_weights + causal_mask[:q_len, :kv_len]

        # Softmax
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(query_states.dtype)

        # Expand values for GQA and compute output
        full_values_expanded = _repeat_kv(full_values, n_kv_groups)
        attn_output = torch.matmul(attn_weights, full_values_expanded)

        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = attn_module.out_proj(attn_output)

        return attn_output, None  # (output, attn_weights)

    return fused_forward


def install_fused_attention(model, bits: int = 4) -> CompressedKVCache:
    """Patch all attention layers in a Gemma3 model to use fused TurboQuant.

    Returns a CompressedKVCache to pass as past_key_values to generate().
    """
    # Detect head_dim
    config = model.config
    if hasattr(config, 'text_config'):
        text_config = config.text_config
    else:
        text_config = config
    head_dim = getattr(text_config, 'head_dim', 256)

    # Create quantizer
    tq = TurboQuantMSE(d=head_dim, bits=bits, device="cuda")

    # Create compressed cache
    cache = CompressedKVCache(tq)

    # Find and patch text attention layers (not vision encoder)
    patched = 0
    layer_idx = 0
    for name, module in model.named_modules():
        if all(hasattr(module, attr) for attr in ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'num_heads']):
            module.forward = make_fused_attention_forward(module, cache, tq, layer_idx)
            patched += 1
            layer_idx += 1

    print(f"  Installed fused TurboQuant ({bits}-bit) on {patched} attention layers")
    return cache


class FusedTurboQuantRunner:
    """High-level runner: patches model, generates, unpatches.

    Usage:
        runner = FusedTurboQuantRunner(model, processor, bits=4)
        text = runner.generate("What is 2+2?", max_new_tokens=30)
    """

    def __init__(self, model, processor, bits: int = 4):
        self.model = model
        self.processor = processor
        self.bits = bits
        # Save original forwards for unpatching
        self._originals: dict[str, callable] = {}
        for name, module in model.named_modules():
            if all(hasattr(module, attr) for attr in ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'num_heads']):
                self._originals[name] = module.forward

    def generate(self, prompt: str, max_new_tokens: int = 200, system: str = "You are a helpful assistant."):
        # Install fused attention (creates fresh cache)
        cache = install_fused_attention(self.model, self.bits)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                past_key_values=cache,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        gen_ids = out[0][input_len:]
        text = self.processor.decode(gen_ids, skip_special_tokens=True)

        # Unpatch
        self._unpatch()

        return text

    def _unpatch(self):
        for name, module in self.model.named_modules():
            if name in self._originals:
                module.forward = self._originals[name]
```

---

## What I need you to produce

1. **Modified `fused_attention.py`** — add QJL sign storage, residual norm computation, query sketch pre-projection, and the two-term attention score in the patched forward.

2. **Modified `triton_kernels.py`** — add a new Triton kernel `triton_fused_attention_qjl` that computes both MSE term1 and QJL term2 in a single kernel, or modify the existing `_fused_rotor_attention_kernel`.

3. **A test script** that verifies:
   - QJL signs are computed correctly
   - The two-term estimator is unbiased (mean error ≈ 0)
   - Perplexity matches FP16 within 5%

Keep the same code style, use the existing Triton patterns, and make it work with the HuggingFace DynamicCache monkey-patching approach.
