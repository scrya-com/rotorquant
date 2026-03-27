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
