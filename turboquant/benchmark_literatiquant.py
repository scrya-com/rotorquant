"""
LiteratiQuant benchmark: 1-bit group-128 vs rotation-based methods.

Compares:
  1. Reconstruction MSE (higher for 1-bit, but that's the tradeoff)
  2. Compression ratio (14x vs 4-5x)
  3. Quantize/dequantize speed
  4. KV cache memory savings
  5. QAT training simulation (linear layer forward/backward)

The key insight: LiteratiQuant trades MSE for extreme compression.
It's complementary to rotation-based methods, not a replacement.
"""

import torch
import time
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from turboquant.literatiquant import (
    LiteratiQuantMSE, LiteratiQuantRotated, LiteratiQuantLinear,
    LiteratiQuantEmbedding, LiteratiQuantKVCache, literati_replace,
    quantize_literati, compute_scales_mean_abs, pack_signs, unpack_signs,
)
from turboquant.isoquant import IsoQuantMSE
from turboquant.planarquant import PlanarQuantMSE


def benchmark_mse(d, n_vectors=8192, device='cuda'):
    """Compare reconstruction MSE: LiteratiQuant (1-bit) vs multi-bit methods."""
    torch.manual_seed(42)
    x = torch.randn(n_vectors, d, device=device)
    x = x / x.norm(dim=-1, keepdim=True)

    results = {}

    # LiteratiQuant symmetric (1-bit, group 128)
    lq = LiteratiQuantMSE(d, group_size=128, mode='symmetric', device=device)
    x_hat_lq, _ = lq(x)
    results['Literati-1bit-sym'] = {
        'mse': (x - x_hat_lq).pow(2).mean().item(),
        'bits': lq.bits_per_element,
        'compression': lq.compression_ratio(),
    }

    # LiteratiQuant asymmetric (1-bit + offset, group 128)
    lq_asym = LiteratiQuantMSE(d, group_size=128, mode='asymmetric', device=device)
    x_hat_asym, _ = lq_asym(x)
    results['Literati-1bit-asym'] = {
        'mse': (x - x_hat_asym).pow(2).mean().item(),
        'bits': lq_asym.bits_per_element,
        'compression': lq_asym.compression_ratio(),
    }

    # LiteratiQuant 2-bit (4-level, group 128)
    lq_2bit = LiteratiQuantMSE(d, group_size=128, mode='2bit', device=device)
    x_hat_2bit, _ = lq_2bit(x)
    results['Literati-2bit'] = {
        'mse': (x - x_hat_2bit).pow(2).mean().item(),
        'bits': lq_2bit.bits_per_element,
        'compression': lq_2bit.compression_ratio(),
    }

    # LiteratiQuant + IsoQuant rotation (1-bit, group 128)
    lq_rot = LiteratiQuantRotated(d, group_size=128, device=device)
    x_hat_rot, _ = lq_rot(x)
    results['Literati+IsoRot-1bit'] = {
        'mse': (x - x_hat_rot).pow(2).mean().item(),
        'bits': lq_rot.bits_per_element,
        'compression': lq_rot.compression_ratio(),
    }

    # IsoQuant 3-bit (rotation-based)
    for bits in [2, 3, 4]:
        iq = IsoQuantMSE(d, bits, seed=42, mode='fast', device=device)
        x_hat_iq, _ = iq(x)
        results[f'IsoQuant-{bits}bit'] = {
            'mse': (x - x_hat_iq).pow(2).mean().item(),
            'bits': bits,
            'compression': 16.0 / bits,
        }

    # PlanarQuant 3-bit
    pq = PlanarQuantMSE(d, 3, seed=42, device=device)
    x_hat_pq, _ = pq(x)
    results['PlanarQuant-3bit'] = {
        'mse': (x - x_hat_pq).pow(2).mean().item(),
        'bits': 3,
        'compression': 16.0 / 3,
    }

    return results


def benchmark_speed(d, n_vectors=8192, warmup=50, iters=200, device='cuda'):
    """Compare quantize+dequantize latency."""
    torch.manual_seed(42)
    x = torch.randn(n_vectors, d, device=device)
    x = x / x.norm(dim=-1, keepdim=True)

    methods = {
        'Literati-1bit-sym': LiteratiQuantMSE(d, group_size=128, mode='symmetric', device=device),
        'Literati-1bit-asym': LiteratiQuantMSE(d, group_size=128, mode='asymmetric', device=device),
        'Literati-2bit': LiteratiQuantMSE(d, group_size=128, mode='2bit', device=device),
        'Literati+IsoRot': LiteratiQuantRotated(d, group_size=128, device=device),
        'IsoQuant-3bit': IsoQuantMSE(d, 3, seed=42, mode='fast', device=device),
        'PlanarQuant-3bit': PlanarQuantMSE(d, 3, seed=42, device=device),
    }

    results = {}
    for name, model in methods.items():
        for _ in range(warmup):
            model(x)
        if device == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters * 1e6
        results[name] = dt

    return results


def benchmark_kv_cache(batch=1, n_heads=32, seq_len=4096, d=128, device='cuda'):
    """Compare KV cache memory: LiteratiQuant vs FP16."""
    torch.manual_seed(42)
    kv = torch.randn(batch, n_heads, seq_len, d, device=device)

    cache = LiteratiQuantKVCache(d, group_size=128, device=device)
    cache.insert(kv)

    compressed_bytes = cache.memory_bytes()
    fp16_bytes = kv.numel() * 2  # 2 bytes per FP16

    # Reconstruction quality
    kv_recon = cache.get_all()
    mse = (kv - kv_recon).pow(2).mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        kv.reshape(-1, d), kv_recon.reshape(-1, d), dim=-1
    ).mean().item()

    return {
        'fp16_mb': fp16_bytes / 1e6,
        'literati_mb': compressed_bytes / 1e6,
        'compression': fp16_bytes / compressed_bytes,
        'mse': mse,
        'cos_sim': cos_sim,
    }


def benchmark_qat_training(in_features=4096, out_features=4096,
                           batch=8, seq_len=256, device='cuda'):
    """Simulate QAT training: forward + backward through LiteratiQuantLinear."""
    torch.manual_seed(42)

    linear_q = LiteratiQuantLinear(in_features, out_features, group_size=128).to(device)
    linear_fp = torch.nn.Linear(in_features, out_features, bias=False).to(device)

    x = torch.randn(batch, seq_len, in_features, device=device, requires_grad=True)
    target = torch.randn(batch, seq_len, out_features, device=device)

    results = {}

    # LiteratiQuant forward+backward
    for _ in range(10):  # warmup
        y = linear_q(x)
        loss = (y - target).pow(2).mean()
        loss.backward()
        linear_q.zero_grad()
    if device == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(50):
        y = linear_q(x)
        loss = (y - target).pow(2).mean()
        loss.backward()
        linear_q.zero_grad()
    if device == 'cuda':
        torch.cuda.synchronize()
    results['LiteratiQuant'] = (time.perf_counter() - t0) / 50 * 1e3  # ms

    # FP16 baseline
    x2 = x.detach().requires_grad_(True)
    for _ in range(10):
        y = linear_fp(x2)
        loss = (y - target).pow(2).mean()
        loss.backward()
        linear_fp.zero_grad()
    if device == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(50):
        y = linear_fp(x2)
        loss = (y - target).pow(2).mean()
        loss.backward()
        linear_fp.zero_grad()
    if device == 'cuda':
        torch.cuda.synchronize()
    results['FP32-Linear'] = (time.perf_counter() - t0) / 50 * 1e3

    return results


def benchmark_packing(d=128, n_vectors=65536, device='cuda'):
    """Benchmark sign packing/unpacking throughput."""
    torch.manual_seed(42)
    w = torch.randn(n_vectors, d, device=device)

    # Pack
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        packed = pack_signs(w, group_size=128)
    if device == 'cuda':
        torch.cuda.synchronize()
    pack_time = (time.perf_counter() - t0) / 100 * 1e3

    # Unpack
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        signs = unpack_signs(packed, group_size=128)
    if device == 'cuda':
        torch.cuda.synchronize()
    unpack_time = (time.perf_counter() - t0) / 100 * 1e3

    # Verify roundtrip
    original_signs = torch.sign(w)
    original_signs[original_signs == 0] = 1.0
    unpacked = unpack_signs(pack_signs(w, 128), 128)
    # Reshape for comparison
    pad = (128 - d % 128) % 128
    if pad > 0:
        original_signs = torch.nn.functional.pad(original_signs, (0, pad))
    original_signs = original_signs.reshape(n_vectors, -1, 128)
    match = (unpacked == original_signs).float().mean().item()

    return {
        'pack_ms': pack_time,
        'unpack_ms': unpack_time,
        'roundtrip_accuracy': match,
    }


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    # ── MSE vs Compression ──
    print("=" * 80)
    print("  MSE vs COMPRESSION (d=128, 8192 unit vectors)")
    print("=" * 80)
    print(f"{'Method':>25} {'bits/elem':>10} {'Compression':>12} {'MSE':>12}")
    print("-" * 65)
    mses = benchmark_mse(128, device=device)
    for name, info in sorted(mses.items(), key=lambda kv: kv[1]['bits']):
        print(f"{name:>25} {info['bits']:>10.3f} {info['compression']:>11.1f}x {info['mse']:>12.6f}")

    # ── Speed ──
    if device == 'cuda':
        print()
        print("=" * 80)
        print("  LATENCY (microseconds, d=128, 8192 vectors)")
        print("=" * 80)
        speeds = benchmark_speed(128, device=device)
        for name, us in sorted(speeds.items(), key=lambda kv: kv[1]):
            print(f"  {name:>25}: {us:>8.0f} µs")

    # ── KV Cache ──
    print()
    print("=" * 80)
    print("  KV CACHE COMPRESSION (batch=1, heads=32, seq=4096, d=128)")
    print("=" * 80)
    kv = benchmark_kv_cache(device=device)
    print(f"  FP16 cache:     {kv['fp16_mb']:>8.2f} MB")
    print(f"  LiteratiQuant:  {kv['literati_mb']:>8.2f} MB")
    print(f"  Compression:    {kv['compression']:>8.1f}x")
    print(f"  Recon MSE:      {kv['mse']:>8.6f}")
    print(f"  Cosine sim:     {kv['cos_sim']:>8.6f}")

    # ── Packing ──
    print()
    print("=" * 80)
    print("  SIGN PACKING THROUGHPUT (65536 x 128)")
    print("=" * 80)
    packing = benchmark_packing(device=device)
    print(f"  Pack:             {packing['pack_ms']:>8.2f} ms")
    print(f"  Unpack:           {packing['unpack_ms']:>8.2f} ms")
    print(f"  Roundtrip match:  {packing['roundtrip_accuracy'] * 100:>8.2f}%")

    # ── QAT Training ──
    if device == 'cuda':
        print()
        print("=" * 80)
        print("  QAT TRAINING (4096→4096 linear, batch=8, seq=256)")
        print("=" * 80)
        qat = benchmark_qat_training(device=device)
        for name, ms in qat.items():
            print(f"  {name:>20}: {ms:>8.2f} ms/iter")
        overhead = qat['LiteratiQuant'] / qat['FP32-Linear']
        print(f"  QAT overhead:     {overhead:>8.2f}x")

    # ── Summary ──
    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print("  LiteratiQuant: 1-bit sign × learned scale, group-128 (1.125 bits/elem)")
    print("  - 14x compression vs FP16 (vs 4-5x for rotation methods)")
    print("  - Higher MSE than multi-bit methods (expected: 1-bit vs 3-4 bit)")
    print("  - Orthogonal to rotation: can combine rotation + 1-bit for best of both")
    print("  - QAT-friendly: STE gradients flow through sign/scale")
    print("  - KV cache: head_dim=128 = perfect group alignment, zero waste")
    print()
    print("  Use cases:")
    print("  - Extreme compression for on-device / edge inference")
    print("  - QAT training of 1-bit weight models")
    print("  - KV cache for ultra-long context (1M+ tokens)")
    print("  - Combined with IsoQuant rotation pre-processing")
