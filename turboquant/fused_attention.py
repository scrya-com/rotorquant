"""
RotorQuant Fused Attention + QJL two-term unbiased estimator.

Instead of roundtrip (quantize→dequantize→matmul, biased, PPL degrades):
  score = <Q, dequant(quant(K))>  ← biased, error compounds through layers

We compute the unbiased two-term estimator:
  score = <Q_rot, centroids[idx]> * norm              ← term1: MSE
        + ||residual|| * sqrt(π/2)/m * <S@Q, signs>   ← term2: QJL correction

Term2 makes the estimator unbiased (zero expected error), so errors cancel
across tokens/heads instead of accumulating through layers.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional
from transformers import DynamicCache

from .rotorquant import RotorQuantMSE
from .triton_kernels import (
    triton_rotor_sandwich,
    triton_rotor_full_fused,
    triton_rotor_inverse_sandwich,
    pack_rotors_for_triton,
)
from .clifford import MV_DIM, E1, E2, E3, E123


# ── Triton kernel: fused MSE + QJL attention ────────────────────────

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 32, "BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_S": 64, "BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_S": 128, "BLOCK_D": 64}, num_warps=8),
        triton.Config({"BLOCK_S": 64, "BLOCK_D": 128}, num_warps=4),
    ],
    key=["kv_len", "mse_dim"],
)
@triton.jit
def _fused_rotor_attention_qjl_kernel(
    # MSE inputs
    Q_rot_ptr, K_idx_ptr, K_norms_ptr, C_ptr,
    # QJL inputs
    Q_sketch_ptr, QJL_signs_ptr, Res_norms_ptr,
    # Output
    Out_ptr,
    # Dimensions
    kv_len,
    mse_dim: tl.constexpr,   # n_groups * 4 (rotated query dim)
    qjl_m: tl.constexpr,     # QJL projection dim
    n_q_heads, n_kv_heads,
    scale,
    qjl_scale,               # sqrt(pi/2) / m
    # Strides — Q_rot: [BH_q, mse_dim]
    stride_qr_bh, stride_qr_d,
    # Strides — K_idx: [BH_kv, kv_len, mse_dim]
    stride_ki_bh, stride_ki_s, stride_ki_d,
    # Strides — K_norms: [BH_kv, kv_len]
    stride_kn_bh, stride_kn_s,
    # Strides — Q_sketch: [BH_q, qjl_m]
    stride_qs_bh, stride_qs_d,
    # Strides — QJL_signs: [BH_kv, kv_len, qjl_m]
    stride_js_bh, stride_js_s, stride_js_d,
    # Strides — Res_norms: [BH_kv, kv_len]
    stride_rn_bh, stride_rn_s,
    # Strides — Out: [BH_q, kv_len]
    stride_o_bh, stride_o_s,
    # Block sizes
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Two-term attention: MSE gather-dot + QJL sign correction."""
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    # GQA mapping
    batch_idx = pid_bh // n_q_heads
    q_head_idx = pid_bh % n_q_heads
    gqa_ratio = n_q_heads // n_kv_heads
    kv_head_idx = q_head_idx // gqa_ratio
    kv_bh = batch_idx * n_kv_heads + kv_head_idx

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < kv_len

    # ── Term 1: MSE rotor gather-dot ──
    acc_mse = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for d_start in range(0, mse_dim, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < mse_dim

        q_ptrs = Q_rot_ptr + pid_bh * stride_qr_bh + d_offs * stride_qr_d
        q_vals = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        ki_ptrs = (K_idx_ptr + kv_bh * stride_ki_bh
                   + s_offs[:, None] * stride_ki_s
                   + d_offs[None, :] * stride_ki_d)
        mask_2d = s_mask[:, None] & d_mask[None, :]
        k_idx = tl.load(ki_ptrs, mask=mask_2d, other=0).to(tl.int32)
        k_vals = tl.load(C_ptr + k_idx, mask=mask_2d, other=0.0).to(tl.float32)

        acc_mse += tl.sum(k_vals * q_vals[None, :], axis=1)

    # ── Term 2: QJL correction ──
    acc_qjl = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for d_start in range(0, qjl_m, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask_q = d_offs < qjl_m

        qs_ptrs = Q_sketch_ptr + pid_bh * stride_qs_bh + d_offs * stride_qs_d
        qs_vals = tl.load(qs_ptrs, mask=d_mask_q, other=0.0).to(tl.float32)

        sign_ptrs = (QJL_signs_ptr + kv_bh * stride_js_bh
                     + s_offs[:, None] * stride_js_s
                     + d_offs[None, :] * stride_js_d)
        mask_2d_q = s_mask[:, None] & d_mask_q[None, :]
        signs = tl.load(sign_ptrs, mask=mask_2d_q, other=0).to(tl.float32)

        acc_qjl += tl.sum(qs_vals[None, :] * signs, axis=1)

    # ── Combine ──
    kn_ptrs = K_norms_ptr + kv_bh * stride_kn_bh + s_offs * stride_kn_s
    norms = tl.load(kn_ptrs, mask=s_mask, other=0.0).to(tl.float32)

    rn_ptrs = Res_norms_ptr + kv_bh * stride_rn_bh + s_offs * stride_rn_s
    res_norms = tl.load(rn_ptrs, mask=s_mask, other=0.0).to(tl.float32)

    term1 = norms * acc_mse
    term2 = res_norms * qjl_scale * acc_qjl
    scores = (term1 + term2) * scale

    o_ptrs = Out_ptr + pid_bh * stride_o_bh + s_offs * stride_o_s
    tl.store(o_ptrs, scores, mask=s_mask)


def triton_fused_attention_qjl(
    q_rotated: torch.Tensor,      # [batch, n_q_heads, q_len, mse_dim]
    q_sketch: torch.Tensor,       # [batch, n_q_heads, q_len, qjl_m]
    key_indices: torch.Tensor,    # [batch, n_kv_heads, kv_len, mse_dim] uint8
    key_norms: torch.Tensor,      # [batch, n_kv_heads, kv_len] fp16
    qjl_signs: torch.Tensor,     # [batch, n_kv_heads, kv_len, qjl_m] int8 {+1,-1}
    residual_norms: torch.Tensor, # [batch, n_kv_heads, kv_len] fp16
    centroids: torch.Tensor,     # [n_levels] float32
    scale: float,
) -> torch.Tensor:
    """Fused MSE + QJL attention scores via Triton."""
    batch, n_q_heads, q_len, mse_dim = q_rotated.shape
    _, n_kv_heads, kv_len, qjl_m = qjl_signs.shape

    q_flat = q_rotated.reshape(batch * n_q_heads * q_len, mse_dim).contiguous().float()
    qs_flat = q_sketch.reshape(batch * n_q_heads * q_len, qjl_m).contiguous().float()
    ki_flat = key_indices.reshape(batch * n_kv_heads, kv_len, mse_dim).contiguous()
    kn_flat = key_norms.reshape(batch * n_kv_heads, kv_len).contiguous().float()
    sign_flat = qjl_signs.reshape(batch * n_kv_heads, kv_len, qjl_m).contiguous().float()
    rn_flat = residual_norms.reshape(batch * n_kv_heads, kv_len).contiguous().float()
    centroids = centroids.contiguous().float()

    out = torch.empty(batch * n_q_heads * q_len, kv_len,
                      device=q_rotated.device, dtype=torch.float32)

    effective_q_heads = n_q_heads * q_len
    qjl_scale = math.sqrt(math.pi / 2) / qjl_m

    grid = lambda meta: (batch * effective_q_heads,
                         triton.cdiv(kv_len, meta['BLOCK_S']))

    _fused_rotor_attention_qjl_kernel[grid](
        q_flat, ki_flat, kn_flat, centroids,
        qs_flat, sign_flat, rn_flat,
        out,
        kv_len, mse_dim, qjl_m,
        effective_q_heads, n_kv_heads,
        scale, qjl_scale,
        q_flat.stride(0), q_flat.stride(1),
        ki_flat.stride(0), ki_flat.stride(1), ki_flat.stride(2),
        kn_flat.stride(0), kn_flat.stride(1),
        qs_flat.stride(0), qs_flat.stride(1),
        sign_flat.stride(0), sign_flat.stride(1), sign_flat.stride(2),
        rn_flat.stride(0), rn_flat.stride(1),
        out.stride(0), out.stride(1),
    )

    return out.reshape(batch, n_q_heads, q_len, kv_len)


# ── Compressed cache with QJL ───────────────────────────────────────

class RotorQuantCompressedCache(DynamicCache):
    """KV cache storing rotor MSE indices + QJL signs + norms.

    Per key vector stores:
      - mse_indices: uint8 [n_groups*4] — rotor centroid indices (4 non-zero grades)
      - mse_norms: fp16 — original ||key||
      - qjl_signs: int8 [head_dim] — sign(S @ residual), {+1, -1}
      - residual_norms: fp16 — ||key - key_mse||
    """

    def __init__(self, rq: RotorQuantMSE, device: str = "cuda"):
        super().__init__()
        self.rq = rq
        self.device = device
        self.packed_rotors = pack_rotors_for_triton(rq.rotors).to(device)
        self.centroids_vector = getattr(rq, 'centroids_vector').to(device)
        self.n_groups = rq.n_groups
        self.head_dim = rq.d

        # Per-layer QJL projection matrix S (head_dim × head_dim)
        self._S_matrices: list[Optional[torch.Tensor]] = []
        # Per-layer compressed storage
        self._compressed_keys: list[Optional[dict]] = []

    def _get_S(self, layer_idx: int) -> torch.Tensor:
        """Lazy-init random projection matrix for this layer."""
        while len(self._S_matrices) <= layer_idx:
            self._S_matrices.append(None)
        if self._S_matrices[layer_idx] is None:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(layer_idx * 7919 + 42)
            S = torch.randn(self.head_dim, self.head_dim, generator=gen)
            self._S_matrices[layer_idx] = S.to(self.device)
        return self._S_matrices[layer_idx]

    def _quantize_keys(self, key_states: torch.Tensor, layer_idx: int) -> dict:
        """Rotor MSE quantization → residual → QJL signs."""
        B, H, S_len, D = key_states.shape
        flat = key_states.reshape(-1, D).float()  # [N, D]

        # ── Stage 1: Rotor MSE ──
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        flat_unit = flat / norms

        # Rotor sandwich
        mv_rot = triton_rotor_sandwich(flat_unit, self.packed_rotors)  # [N, n_groups, 8]

        # Extract non-zero grades + quantize
        c_v = self.centroids_vector
        b_v = (c_v[:-1] + c_v[1:]) / 2

        v1 = mv_rot[:, :, E1]
        v2 = mv_rot[:, :, E2]
        v3 = mv_rot[:, :, E3]
        t7 = mv_rot[:, :, E123]

        idx_v1 = torch.searchsorted(b_v, v1.contiguous())
        idx_v2 = torch.searchsorted(b_v, v2.contiguous())
        idx_v3 = torch.searchsorted(b_v, v3.contiguous())
        idx_t7 = torch.searchsorted(b_v, t7.contiguous())

        # Reconstruct MSE keys for residual computation
        q_v1 = c_v[idx_v1]; q_v2 = c_v[idx_v2]; q_v3 = c_v[idx_v3]; q_t7 = c_v[idx_t7]

        # Build quantized MV and inverse sandwich to get k_mse
        mv_q = torch.zeros_like(mv_rot)
        mv_q[:, :, E1] = q_v1; mv_q[:, :, E2] = q_v2
        mv_q[:, :, E3] = q_v3; mv_q[:, :, E123] = q_t7

        k_mse_unit = triton_rotor_inverse_sandwich(mv_q, self.packed_rotors, D)
        k_mse = k_mse_unit * norms  # [N, D]

        # Pack indices
        indices = torch.stack([idx_v1, idx_v2, idx_v3, idx_t7], dim=-1)  # [N, n_groups, 4]
        indices = indices.reshape(-1, self.n_groups * 4).to(torch.uint8)

        # ── Stage 2: QJL on residual ──
        residual = flat - k_mse
        residual_norms = residual.norm(dim=-1)  # [N]

        S = self._get_S(layer_idx)
        projected = residual @ S.T  # [N, D]
        qjl_signs = torch.sign(projected).to(torch.int8)
        qjl_signs[qjl_signs == 0] = 1

        return {
            'mse_indices': indices.reshape(B, H, S_len, self.n_groups * 4),
            'mse_norms': norms.squeeze(-1).half().reshape(B, H, S_len),
            'qjl_signs': qjl_signs.reshape(B, H, S_len, D),
            'residual_norms': residual_norms.half().reshape(B, H, S_len),
        }

    def store_compressed_key(self, key_states: torch.Tensor, layer_idx: int):
        compressed = self._quantize_keys(key_states, layer_idx)
        while len(self._compressed_keys) <= layer_idx:
            self._compressed_keys.append(None)
        if self._compressed_keys[layer_idx] is None:
            self._compressed_keys[layer_idx] = compressed
        else:
            prev = self._compressed_keys[layer_idx]
            self._compressed_keys[layer_idx] = {
                k: torch.cat([prev[k], compressed[k]], dim=2) for k in compressed
            }

    def get_compressed_key(self, layer_idx: int) -> Optional[dict]:
        if layer_idx < len(self._compressed_keys):
            return self._compressed_keys[layer_idx]
        return None


# ── Helper functions ────────────────────────────────────────────────

def pre_rotate_query(query, packed_rotors, n_groups):
    """Pre-rotate Q into rotor-compressed basis, extract non-zero grades."""
    B, H, Q, D = query.shape
    flat = query.reshape(-1, D)
    mv_rot = triton_rotor_sandwich(flat, packed_rotors)
    v1 = mv_rot[:, :, E1]; v2 = mv_rot[:, :, E2]
    v3 = mv_rot[:, :, E3]; t7 = mv_rot[:, :, E123]
    q_rot = torch.stack([v1, v2, v3, t7], dim=-1).reshape(-1, n_groups * 4)
    return q_rot.reshape(B, H, Q, n_groups * 4)


def pre_sketch_query(query, S):
    """Pre-project Q through S for QJL term2."""
    B, H, Q, D = query.shape
    return (query.float().reshape(-1, D) @ S.T).reshape(B, H, Q, -1)


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def _repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    b, h, s, d = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


# ── Patched attention forward ───────────────────────────────────────

def make_fused_rotor_attention_forward(attn_module, cache, layer_index):
    """Create replacement forward with MSE + QJL two-term estimator."""
    packed_rotors = cache.packed_rotors
    centroids = cache.centroids_vector
    n_groups = cache.n_groups
    head_dim = cache.head_dim
    scale = 1.0 / math.sqrt(head_dim)
    n_heads = attn_module.num_heads
    n_kv_heads = getattr(attn_module, 'num_key_value_heads', n_heads)
    n_kv_groups = n_heads // n_kv_heads
    layer_idx = layer_index

    def fused_forward(hidden_states, position_embeddings=None, attention_mask=None,
                      past_key_values=None, cache_position=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()

        Q = attn_module.q_proj(hidden_states)
        K = attn_module.k_proj(hidden_states)
        V = attn_module.v_proj(hidden_states)

        Q = Q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
        K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            Q, K = _apply_rotary_pos_emb(Q, K, cos, sin)

        # Compress keys (MSE + QJL)
        cache.store_compressed_key(K, layer_idx)

        # Store values via DynamicCache
        cache.update(V, V, layer_idx)

        # Pre-rotate query for MSE term1
        q_rotated = pre_rotate_query(Q, packed_rotors, n_groups)

        # Pre-sketch query for QJL term2
        S = cache._get_S(layer_idx)
        q_sketch = pre_sketch_query(Q, S)

        # Get compressed keys
        compressed = cache.get_compressed_key(layer_idx)
        kv_len = compressed['mse_indices'].shape[2]

        # Fused two-term attention
        attn_weights = triton_fused_attention_qjl(
            q_rotated, q_sketch,
            compressed['mse_indices'], compressed['mse_norms'],
            compressed['qjl_signs'], compressed['residual_norms'],
            centroids, scale,
        )

        # Mask + softmax
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                attn_weights = attn_weights + attention_mask[:, :, :q_len, :kv_len]
            elif attention_mask.dim() == 2:
                attn_weights = attn_weights + attention_mask[:q_len, :kv_len]

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)

        # Value projection
        full_values = cache.layers[layer_idx].values
        attn_output = torch.matmul(attn_weights, _repeat_kv(full_values, n_kv_groups))

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        if hasattr(attn_module, 'o_proj'):
            attn_output = attn_module.o_proj(attn_output)
        elif hasattr(attn_module, 'out_proj'):
            attn_output = attn_module.out_proj(attn_output)

        return attn_output, None

    return fused_forward


def install_fused_rotor_attention(model, bits: int = 3) -> RotorQuantCompressedCache:
    """Patch all attention layers with fused RotorQuant + QJL."""
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
            if not hasattr(module, 'num_heads'):
                module.num_heads = text_config.num_attention_heads
            if not hasattr(module, 'num_key_value_heads'):
                module.num_key_value_heads = getattr(
                    text_config, 'num_key_value_heads', text_config.num_attention_heads)
            module.forward = make_fused_rotor_attention_forward(module, cache, layer_idx)
            patched += 1
            layer_idx += 1

    print(f"  Installed fused RotorQuant + QJL ({bits}-bit) on {patched} layers")
    return cache
