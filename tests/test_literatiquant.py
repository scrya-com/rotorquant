"""
Tests for LiteratiQuant: 1-bit symmetric group quantization.

Validates:
  1. Sign quantization correctness (round-trip signs match)
  2. Scale computation (mean_abs matches expected)
  3. Pack/unpack sign bits (exact round-trip)
  4. Linear layer forward/backward (STE gradients flow)
  5. Embedding layer forward
  6. KV cache compress/decompress
  7. Model replacement utility
  8. Compression ratio math
"""

import torch
import torch.nn as nn
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from turboquant.literatiquant import (
    LiteratiQuantMSE, LiteratiQuantLinear, LiteratiQuantEmbedding,
    LiteratiQuantKVCache, literati_replace,
    quantize_literati, compute_scales_mean_abs,
    pack_signs, unpack_signs, sign_ste,
    export_literati_to_gguf_tensors,
)


class TestSignSTE:
    def test_forward_correctness(self):
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        signs = sign_ste(x)
        assert signs.tolist() == [-1.0, -1.0, 1.0, 1.0, 1.0]

    def test_gradient_passes_through(self):
        x = torch.randn(10, requires_grad=True)
        signs = sign_ste(x)
        loss = signs.sum()
        loss.backward()
        # STE: gradient should be 1.0 everywhere
        assert torch.allclose(x.grad, torch.ones_like(x))


class TestScales:
    def test_mean_abs_positive(self):
        w = torch.randn(4, 128)
        scales = compute_scales_mean_abs(w, group_size=128)
        assert (scales > 0).all()
        assert scales.shape == (4, 1)

    def test_mean_abs_correctness(self):
        w = torch.ones(1, 128) * 3.0
        scales = compute_scales_mean_abs(w, group_size=128)
        assert torch.allclose(scales, torch.tensor([[3.0]]))

    def test_padding(self):
        w = torch.randn(2, 100)  # not multiple of 128
        scales = compute_scales_mean_abs(w, group_size=128)
        assert scales.shape == (2, 1)


class TestQuantizeLiterati:
    def test_output_shape(self):
        w = torch.randn(8, 256)
        scales = compute_scales_mean_abs(w, group_size=128)
        w_q = quantize_literati(w, scales, group_size=128)
        assert w_q.shape == w.shape

    def test_values_are_sign_times_scale(self):
        w = torch.randn(1, 128)
        scales = compute_scales_mean_abs(w, group_size=128)
        w_q = quantize_literati(w, scales, group_size=128)
        # All values should be either +scale or -scale
        unique = w_q.abs().unique()
        assert len(unique) == 1
        assert torch.allclose(unique[0], scales.squeeze(), atol=1e-6)

    def test_signs_match_input(self):
        w = torch.randn(4, 128)
        scales = compute_scales_mean_abs(w, group_size=128)
        w_q = quantize_literati(w, scales, group_size=128)
        # Signs of quantized should match signs of original
        assert (torch.sign(w_q) == torch.sign(w)).all()

    def test_nonmultiple_dim(self):
        w = torch.randn(3, 100)
        scales = compute_scales_mean_abs(w, group_size=128)
        w_q = quantize_literati(w, scales, group_size=128)
        assert w_q.shape == (3, 100)


class TestPackUnpackSigns:
    def test_roundtrip_exact(self):
        w = torch.randn(16, 128)
        packed = pack_signs(w, group_size=128)
        unpacked = unpack_signs(packed, group_size=128)

        expected = torch.sign(w)
        expected[expected == 0] = 1.0
        expected = expected.reshape(16, 1, 128)

        assert torch.allclose(unpacked, expected)

    def test_packed_shape(self):
        w = torch.randn(4, 128)
        packed = pack_signs(w, group_size=128)
        # 128 bits / 8 = 16 bytes per group
        assert packed.shape == (4, 1, 16)
        assert packed.dtype == torch.uint8

    def test_all_positive(self):
        w = torch.ones(1, 128)
        packed = pack_signs(w, group_size=128)
        unpacked = unpack_signs(packed, group_size=128)
        assert (unpacked == 1.0).all()

    def test_all_negative(self):
        w = -torch.ones(1, 128)
        packed = pack_signs(w, group_size=128)
        unpacked = unpack_signs(packed, group_size=128)
        assert (unpacked == -1.0).all()


class TestLiteratiQuantMSE:
    def test_forward_shape(self):
        lq = LiteratiQuantMSE(128, group_size=128)
        x = torch.randn(32, 128)
        x_hat, info = lq(x)
        assert x_hat.shape == x.shape

    def test_compression_ratio(self):
        lq = LiteratiQuantMSE(128, group_size=128)
        ratio = lq.compression_ratio()
        # 16 / 1.125 ≈ 14.22
        assert abs(ratio - 14.22) < 0.1

    def test_mse_bounded(self):
        lq = LiteratiQuantMSE(128, group_size=128)
        x = torch.randn(1000, 128)
        x_hat, _ = lq(x)
        mse = (x - x_hat).pow(2).mean().item()
        # 1-bit quantization has significant MSE, but it should be bounded
        assert mse < 2.0  # loose bound

    def test_different_group_sizes(self):
        for gs in [32, 64, 128, 256]:
            lq = LiteratiQuantMSE(256, group_size=gs)
            x = torch.randn(10, 256)
            x_hat, _ = lq(x)
            assert x_hat.shape == x.shape


class TestLiteratiQuantLinear:
    def test_forward_shape(self):
        linear = LiteratiQuantLinear(128, 64, group_size=128)
        x = torch.randn(4, 128)
        y = linear(x)
        assert y.shape == (4, 64)

    def test_gradient_flows(self):
        linear = LiteratiQuantLinear(128, 64, group_size=128)
        x = torch.randn(4, 128, requires_grad=True)
        y = linear(x)
        loss = y.sum()
        loss.backward()
        # Weight and scale gradients should exist
        assert linear.weight.grad is not None
        assert linear.scales.grad is not None
        assert x.grad is not None

    def test_compute_scales_from_weights(self):
        linear = LiteratiQuantLinear(128, 64, group_size=128)
        linear.compute_scales_from_weights()
        # Scales should be positive
        assert (linear.scales.data > 0).all()

    def test_with_bias(self):
        linear = LiteratiQuantLinear(128, 64, group_size=128, bias=True)
        x = torch.randn(4, 128)
        y = linear(x)
        assert y.shape == (4, 64)
        assert linear.bias is not None


class TestLiteratiQuantEmbedding:
    def test_forward_shape(self):
        emb = LiteratiQuantEmbedding(1000, 128, group_size=128)
        ids = torch.tensor([0, 1, 5, 999])
        y = emb(ids)
        assert y.shape == (4, 128)

    def test_padding_idx(self):
        emb = LiteratiQuantEmbedding(1000, 128, padding_idx=0)
        ids = torch.tensor([0])
        y = emb(ids)
        # Padding index output should be zeros
        assert torch.allclose(y, torch.zeros(1, 128), atol=1e-6)


class TestLiteratiQuantKVCache:
    def test_compress_decompress(self):
        cache = LiteratiQuantKVCache(128, group_size=128)
        kv = torch.randn(1, 4, 8, 128)  # batch, heads, seq, dim
        cache.insert(kv)

        recon = cache.get_all()
        assert recon.shape == kv.shape

        # Cosine similarity should be high despite 1-bit
        cos_sim = nn.functional.cosine_similarity(
            kv.reshape(-1, 128), recon.reshape(-1, 128), dim=-1
        ).mean()
        assert cos_sim > 0.5  # 1-bit is lossy but direction-preserving

    def test_incremental_insert(self):
        cache = LiteratiQuantKVCache(128)
        kv1 = torch.randn(1, 2, 4, 128)
        kv2 = torch.randn(1, 2, 2, 128)

        cache.insert(kv1)
        assert cache.seq_len == 4

        cache.insert(kv2)
        assert cache.seq_len == 6

    def test_memory_bytes(self):
        cache = LiteratiQuantKVCache(128, group_size=128)
        kv = torch.randn(1, 32, 1024, 128)
        cache.insert(kv)

        fp16_bytes = 1 * 32 * 1024 * 128 * 2
        compressed = cache.memory_bytes()
        ratio = fp16_bytes / compressed
        assert ratio > 5  # should be well compressed

    def test_clear(self):
        cache = LiteratiQuantKVCache(128)
        cache.insert(torch.randn(1, 2, 4, 128))
        cache.clear()
        assert cache.seq_len == 0
        assert cache.get_all() is None


class TestLiteratiReplace:
    def test_replaces_linear(self):
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        literati_replace(model)
        assert isinstance(model[0], LiteratiQuantLinear)
        assert isinstance(model[2], LiteratiQuantLinear)
        assert isinstance(model[1], nn.ReLU)  # unchanged

    def test_replaces_embedding(self):
        model = nn.Sequential(nn.Embedding(1000, 128))
        literati_replace(model)
        assert isinstance(model[0], LiteratiQuantEmbedding)

    def test_skip_names(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.proj = nn.Linear(64, 32)
                self.lm_head = nn.Linear(32, 100)

            def forward(self, x):
                return self.lm_head(self.proj(self.embed(x)))

        model = TinyModel()
        literati_replace(model, skip_names={'lm_head'})
        assert isinstance(model.proj, LiteratiQuantLinear)
        assert isinstance(model.lm_head, nn.Linear)  # skipped

    def test_weights_preserved(self):
        linear = nn.Linear(128, 64, bias=False)
        original_weight = linear.weight.data.clone()
        model = nn.Sequential(linear)
        literati_replace(model)
        assert torch.allclose(model[0].weight.data, original_weight)


class TestGGUFExport:
    def test_export_structure(self):
        model = nn.Sequential(
            LiteratiQuantLinear(128, 64, group_size=128),
            LiteratiQuantEmbedding(100, 128, group_size=128),
        )
        tensors = export_literati_to_gguf_tensors(model)
        assert len(tensors) == 2
        for name, data in tensors.items():
            assert 'packed_signs' in data
            assert 'scales' in data
            assert 'shape' in data
            assert data['group_size'] == 128
            assert data['scales'].dtype == torch.float16


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
