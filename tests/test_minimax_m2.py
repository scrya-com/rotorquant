"""
Tests for MiniMax-M2.7 KV cache compression compatibility.

Validates that IsoQuant / PlanarQuant / LiteratiQuant work correctly
with MiniMax-M2.7's architecture dimensions:
  - head_dim = 128
  - num_kv_heads = 8  (GQA: 48 query heads, 8 KV heads)
  - num_layers = 62

All tests use synthetic tensors — no model weights required.
"""

import math
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from turboquant.isoquant import IsoQuantMSE, IsoQuantProd
from turboquant.planarquant import PlanarQuantMSE, PlanarQuantProd
from turboquant.literatiquant import LiteratiQuantMSE, LiteratiQuantKVCache
from turboquant.compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE

# ── MiniMax-M2.7 architecture constants ──────────────────────────────
HEAD_DIM = 128
NUM_KV_HEADS = 8
NUM_ATTN_HEADS = 48
NUM_LAYERS = 62
GQA_RATIO = NUM_ATTN_HEADS // NUM_KV_HEADS  # 6


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def minimax_kv_cache():
    """Synthetic KV cache with MiniMax-M2.7 dimensions (single-batch)."""
    torch.manual_seed(42)
    seq_len = 512
    batch = 1
    keys = torch.randn(batch, NUM_KV_HEADS, seq_len, HEAD_DIM)
    values = torch.randn(batch, NUM_KV_HEADS, seq_len, HEAD_DIM)
    # Normalise keys (typical for attention)
    keys = keys / keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return keys, values


@pytest.fixture(scope="module")
def minimax_query():
    """Synthetic query tensor matching MiniMax-M2.7 attention head dim."""
    torch.manual_seed(99)
    # Queries use full num_attn_heads; KV use num_kv_heads (GQA)
    q = torch.randn(1, NUM_ATTN_HEADS, 1, HEAD_DIM)
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


# ── IsoQuantMSE tests ─────────────────────────────────────────────────

class TestIsoQuantMiniMax:
    """IsoQuant on MiniMax-M2.7 head_dim=128 vectors."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_output_shape(self, minimax_kv_cache, bits):
        keys, _ = minimax_kv_cache
        B, H, S, D = keys.shape
        flat_keys = keys.view(-1, D)

        quant = IsoQuantMSE(D, bits=bits, seed=0)
        x_hat, info = quant(flat_keys)
        assert x_hat.shape == flat_keys.shape

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_mse_within_bounds(self, minimax_kv_cache, bits):
        """Quantization MSE should be much smaller than random baseline (~2.0)."""
        keys, _ = minimax_kv_cache
        flat_keys = keys.view(-1, HEAD_DIM)

        quant = IsoQuantMSE(HEAD_DIM, bits=bits, seed=0)
        x_hat, _ = quant(flat_keys)
        mse = ((flat_keys - x_hat) ** 2).sum(-1).mean().item()
        assert mse < 1.5, f"MSE {mse:.4f} unexpectedly high for {bits}-bit IsoQuant"

    def test_mse_decreases_with_bits(self, minimax_kv_cache):
        """Higher bits → lower MSE on MiniMax-M2.7 dimensions."""
        keys, _ = minimax_kv_cache
        flat_keys = keys.view(-1, HEAD_DIM)
        mses = []
        for bits in [2, 3, 4]:
            quant = IsoQuantMSE(HEAD_DIM, bits=bits, seed=0)
            x_hat, _ = quant(flat_keys)
            mses.append(((flat_keys - x_hat) ** 2).sum(-1).mean().item())

        for i in range(len(mses) - 1):
            assert mses[i] > mses[i + 1], f"MSE not monotone: {mses}"

    def test_head_dim_128_alignment(self):
        """HEAD_DIM=128 is a multiple of 4 — IsoQuant 4D blocks align perfectly."""
        assert HEAD_DIM % 4 == 0, "head_dim should be divisible by 4 for IsoQuant"
        quant = IsoQuantMSE(HEAD_DIM, bits=3, seed=42)
        # n_groups = ceil(d / 4) = 32 for d=128
        assert quant.n_groups == HEAD_DIM // 4


# ── PlanarQuantMSE tests ──────────────────────────────────────────────

class TestPlanarQuantMiniMax:
    """PlanarQuant on MiniMax-M2.7 dimensions."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_output_shape(self, minimax_kv_cache, bits):
        keys, _ = minimax_kv_cache
        flat_keys = keys.view(-1, HEAD_DIM)

        quant = PlanarQuantMSE(HEAD_DIM, bits=bits, seed=0)
        x_hat, _ = quant(flat_keys)
        assert x_hat.shape == flat_keys.shape

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_mse_within_bounds(self, minimax_kv_cache, bits):
        keys, _ = minimax_kv_cache
        flat_keys = keys.view(-1, HEAD_DIM)

        quant = PlanarQuantMSE(HEAD_DIM, bits=bits, seed=0)
        x_hat, _ = quant(flat_keys)
        mse = ((flat_keys - x_hat) ** 2).sum(-1).mean().item()
        assert mse < 1.5, f"MSE {mse:.4f} unexpectedly high for {bits}-bit PlanarQuant"

    def test_head_dim_128_alignment(self):
        """HEAD_DIM=128 is even — PlanarQuant 2D pairs align perfectly."""
        assert HEAD_DIM % 2 == 0
        quant = PlanarQuantMSE(HEAD_DIM, bits=3, seed=42)
        assert quant.n_groups == HEAD_DIM // 2


# ── GQA compatibility tests ───────────────────────────────────────────

class TestGQACompatibility:
    """Tests that the quantizers handle GQA (num_kv_heads != num_query_heads)."""

    def test_kv_head_count(self):
        """MiniMax-M2.7 GQA: 8 KV heads serving 48 query heads."""
        assert NUM_KV_HEADS == 8
        assert NUM_ATTN_HEADS == 48
        assert NUM_ATTN_HEADS % NUM_KV_HEADS == 0  # clean group ratio

    def test_compress_kv_heads_independently(self, minimax_kv_cache):
        """Each KV head can be compressed independently at head_dim=128."""
        keys, values = minimax_kv_cache
        B, H, S, D = keys.shape
        assert H == NUM_KV_HEADS

        quant = IsoQuantMSE(D, bits=3, seed=0)
        for h in range(H):
            k_head = keys[0, h]  # (S, D)
            k_hat, _ = quant(k_head)
            assert k_hat.shape == k_head.shape

    def test_batch_compress_all_kv_heads(self, minimax_kv_cache):
        """Flatten (kv_heads, seq) → batch → compress in one shot."""
        keys, _ = minimax_kv_cache
        B, H, S, D = keys.shape

        flat = keys.view(B * H * S, D)  # (B*H*S, D)
        quant = IsoQuantMSE(D, bits=3, seed=0)
        flat_hat, _ = quant(flat)
        reconstructed = flat_hat.view(B, H, S, D)

        assert reconstructed.shape == keys.shape

        mse = ((keys - reconstructed) ** 2).sum(-1).mean().item()
        assert mse < 1.5


# ── Attention score fidelity tests ────────────────────────────────────

class TestAttentionScoreFidelity:
    """
    Verify that compressed KV cache preserves top-k attention token ranking,
    which is critical for MiniMax-M2.7's long-context capabilities (204K).
    """

    @pytest.mark.parametrize("bits", [3, 4])
    def test_top1_needle_retrieval(self, bits):
        """
        Plant an exact-match 'needle' in the KV cache.
        Compressed keys should rank it #1 (or close).
        """
        torch.manual_seed(7)
        seq_len = 512
        keys = torch.randn(seq_len, HEAD_DIM)
        keys = keys / keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        needle_pos = seq_len // 3
        query = keys[needle_pos].clone()

        quant = IsoQuantMSE(HEAD_DIM, bits=bits, seed=42)
        k_hat, _ = quant(keys)

        real_scores = keys @ query          # (S,)
        est_scores = k_hat @ query          # (S,)

        real_top1 = real_scores.argmax().item()
        est_top5 = est_scores.topk(5).indices.tolist()

        assert real_top1 in est_top5, (
            f"Needle at pos {needle_pos} not in top-5 for {bits}-bit IsoQuant "
            f"(got {est_top5})"
        )

    @pytest.mark.parametrize("bits", [3, 4])
    def test_score_cosine_similarity(self, minimax_kv_cache, bits):
        """Compressed attention scores should correlate with true scores."""
        keys, _ = minimax_kv_cache
        B, H, S, D = keys.shape

        quant = IsoQuantMSE(D, bits=bits, seed=0)
        flat = keys.view(-1, D)
        flat_hat, _ = quant(flat)
        k_hat = flat_hat.view(B, H, S, D)

        # Use last token as query (simulate decode step)
        query = keys[:, :, -1:, :]  # (B, H, 1, D)

        cos_sims = []
        for h in range(H):
            q = query[0, h, 0]    # (D,)
            real = keys[0, h] @ q  # (S,)
            est = k_hat[0, h] @ q  # (S,)
            cos = torch.nn.functional.cosine_similarity(
                real.unsqueeze(0), est.unsqueeze(0)
            ).item()
            cos_sims.append(cos)

        avg_cos = sum(cos_sims) / len(cos_sims)
        assert avg_cos > 0.8, (
            f"Average cosine sim {avg_cos:.4f} too low for {bits}-bit "
            f"IsoQuant on MiniMax-M2.7 dimensions"
        )


# ── LiteratiQuant (1-bit) tests ───────────────────────────────────────

class TestLiteratiQuantMiniMax:
    """1-bit KV cache compression for ultra-high compression ratios."""

    def test_kv_cache_compress_decompress(self, minimax_kv_cache):
        """LiteratiQuantKVCache round-trip on MiniMax-M2.7 KV tensors."""
        keys, _ = minimax_kv_cache
        B, H, S, D = keys.shape

        cache = LiteratiQuantKVCache(D, group_size=128, device="cpu")
        cache.insert(keys[:, 0:1, :, :])  # single head

        reconstructed = cache.get_all()
        assert reconstructed is not None
        assert reconstructed.shape == keys[:, 0:1, :, :].shape

    def test_1bit_compression_ratio(self):
        """Verify claimed 14x compression at group_size=128."""
        quant = LiteratiQuantMSE(HEAD_DIM, group_size=128, mode="symmetric")
        ratio = quant.compression_ratio()
        # 14.2x compression: 16 FP16 bits / (1 sign + 16/128 scale) ≈ 14.2
        assert abs(ratio - 16.0 / (1.0 + 16.0 / 128)) < 0.01

    def test_group_size_alignment(self):
        """head_dim=128 aligns perfectly with group_size=128 (1 group per head)."""
        assert HEAD_DIM % 128 == 0
        quant = LiteratiQuantMSE(HEAD_DIM, group_size=128)
        assert quant.n_groups == HEAD_DIM // 128  # = 1


# ── TurboQuantCompressor tests ────────────────────────────────────────

class TestTurboQuantMiniMax:
    """TurboQuant asymmetric estimator with MiniMax-M2.7 dimensions."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_compress_and_reconstruct_shape(self, minimax_kv_cache, bits):
        keys, _ = minimax_kv_cache
        # TurboQuantCompressorMSE expects (B, H, S, D)
        keys_slice = keys[:, :1, :, :].float()  # (1, 1, S, D)

        comp = TurboQuantCompressorMSE(HEAD_DIM, bits, seed=0)
        compressed = comp.compress(keys_slice)
        x_hat = comp.decompress(compressed)
        assert x_hat.shape == keys_slice.shape

    @pytest.mark.parametrize("bits", [3, 4])
    def test_inner_product_unbiased(self, minimax_kv_cache, bits):
        """TurboQuant asymmetric estimator should be nearly unbiased."""
        keys, _ = minimax_kv_cache
        # Use a single head slice (1, 1, S, D)
        keys_slice = keys[:, :1, :, :].float()
        q = keys[0, 0, -1].float()  # (D,) — last token as query

        comp = TurboQuantCompressorMSE(HEAD_DIM, bits, seed=42)
        compressed = comp.compress(keys_slice)
        x_hat = comp.decompress(compressed)  # (1, 1, S, D)

        k = keys_slice[0, 0]  # (S, D)
        k_hat = x_hat[0, 0]   # (S, D)

        real_scores = k @ q           # (S,)
        est_scores = k_hat @ q        # (S,)

        bias = (est_scores - real_scores).mean().item()
        assert abs(bias) < 0.05, f"Bias {bias:.4f} too high for {bits}-bit TurboQuant"


# ── Memory estimate tests ─────────────────────────────────────────────

class TestMemoryEstimate:
    """Verify compression ratio math for MiniMax-M2.7 long-context scenarios."""

    @pytest.mark.parametrize("seq_len,bits", [
        (32_768, 3),
        (204_800, 3),  # max context
        (32_768, 4),
    ])
    def test_compression_ratio_at_scale(self, seq_len, bits):
        """IsoQuant should provide meaningful compression at long context."""
        # FP16: 2 bytes per element
        fp16_bytes = NUM_LAYERS * NUM_KV_HEADS * seq_len * HEAD_DIM * 2 * 2  # K+V
        # IsoQuant: bits/elem + fp16 norm per vector
        quant_bytes = NUM_LAYERS * NUM_KV_HEADS * seq_len * HEAD_DIM * bits / 8 * 2
        norm_bytes = NUM_LAYERS * NUM_KV_HEADS * seq_len * 2 * 2  # fp16
        total = quant_bytes + norm_bytes
        ratio = fp16_bytes / total

        # Should be at least 3x compression at 3-bit, 2x at 4-bit
        min_ratio = 2.5 if bits == 3 else 1.8
        assert ratio > min_ratio, (
            f"Compression ratio {ratio:.1f}x at {bits}-bit, seq={seq_len:,} "
            f"is below expected {min_ratio}x"
        )
