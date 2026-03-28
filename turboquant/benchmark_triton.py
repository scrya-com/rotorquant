"""
Triton vs PyTorch vs CUDA Benchmark for RotorQuant

Compares three backends:
  1. PyTorch (pure tensor ops — current rotorquant.py)
  2. CUDA C++ (rotor_fused_kernel.cu — requires compilation)
  3. Triton (triton_kernels.py — portable, auto-tuned)

Tests:
  A. Rotor sandwich (forward): embed + R x R̃
  B. Full fused pipeline: embed→rotor→quantize→unrotor→extract
  C. Fused attention scores on compressed keys
  D. Numerical correctness verification

Usage:
    python -m turboquant.benchmark_triton
"""

import torch
import torch.nn.functional as F
import time
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def time_fn(fn, n_warmup=10, n_iter=100, sync=True):
    """Time a function with warmup and averaging."""
    device = "cuda"
    for _ in range(n_warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    if sync:
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_iter * 1000  # ms
    return elapsed


def verify_correctness():
    """Verify Triton kernels match PyTorch reference."""
    from turboquant.rotorquant import RotorQuantMSE
    from turboquant.triton_kernels import (
        triton_rotor_sandwich, triton_rotor_full_fused,
        triton_rotor_inverse_sandwich, pack_rotors_for_triton,
    )
    from turboquant.clifford import (
        embed_vectors_as_multivectors, rotor_sandwich,
        extract_vectors_from_multivectors, reverse,
    )

    print("=" * 70)
    print("CORRECTNESS VERIFICATION: Triton vs PyTorch")
    print("=" * 70)

    d = 128
    n = 1024
    bits = 3
    device = "cuda"

    rq = RotorQuantMSE(d, bits, seed=42, device=device)
    x = torch.randn(n, d, device=device)
    x = x / x.norm(dim=-1, keepdim=True)

    # Pack rotors for Triton
    packed_rotors = pack_rotors_for_triton(rq.rotors).to(device)

    # --- Test 1: Rotor sandwich ---
    # PyTorch reference
    mv_ref = embed_vectors_as_multivectors(x)
    mv_rot_ref = rotor_sandwich(rq.rotors, mv_ref)

    # Triton
    mv_rot_triton = triton_rotor_sandwich(x, packed_rotors)

    # Compare
    max_diff = (mv_rot_ref - mv_rot_triton).abs().max().item()
    cos_sim = F.cosine_similarity(
        mv_rot_ref.reshape(n, -1), mv_rot_triton.reshape(n, -1), dim=-1
    ).mean().item()
    print(f"\n  Rotor sandwich (forward):")
    print(f"    Max diff:    {max_diff:.2e}")
    print(f"    Cosine sim:  {cos_sim:.8f}")
    print(f"    {'PASS' if cos_sim > 0.9999 else 'FAIL'}")

    # --- Test 2: Inverse rotor sandwich ---
    from turboquant.clifford import reverse as clifford_reverse

    # PyTorch reference
    rotor_rev = clifford_reverse(rq.rotors)
    mv_recon_ref = rotor_sandwich(rotor_rev, mv_rot_ref)
    v_recon_ref = extract_vectors_from_multivectors(mv_recon_ref, d)

    # Triton
    v_recon_triton = triton_rotor_inverse_sandwich(mv_rot_triton, packed_rotors, d)

    max_diff = (v_recon_ref - v_recon_triton).abs().max().item()
    cos_sim = F.cosine_similarity(v_recon_ref, v_recon_triton, dim=-1).mean().item()
    print(f"\n  Inverse sandwich (reconstruct):")
    print(f"    Max diff:    {max_diff:.2e}")
    print(f"    Cosine sim:  {cos_sim:.8f}")
    print(f"    {'PASS' if cos_sim > 0.9999 else 'FAIL'}")

    # --- Test 3: Full fused pipeline ---
    # PyTorch reference
    x_hat_ref, _ = rq(x)

    # Triton
    c_scalar = None
    c_vector = getattr(rq, 'centroids_vector')
    c_bivector = None
    c_trivector = None

    x_hat_triton = triton_rotor_full_fused(
        x, packed_rotors, c_scalar, c_vector, c_bivector, c_trivector)

    max_diff = (x_hat_ref - x_hat_triton).abs().max().item()
    cos_sim = F.cosine_similarity(x_hat_ref, x_hat_triton, dim=-1).mean().item()
    mse_ref = ((x - x_hat_ref) ** 2).sum(dim=-1).mean().item()
    mse_triton = ((x - x_hat_triton) ** 2).sum(dim=-1).mean().item()
    print(f"\n  Full fused pipeline (quantize→dequantize):")
    print(f"    Max diff:    {max_diff:.2e}")
    print(f"    Cosine sim:  {cos_sim:.8f}")
    print(f"    MSE (ref):   {mse_ref:.6f}")
    print(f"    MSE (triton):{mse_triton:.6f}")
    print(f"    {'PASS' if cos_sim > 0.999 else 'FAIL'}")

    print()
    return cos_sim > 0.999


def benchmark_rotor_sandwich():
    """Benchmark rotor sandwich: PyTorch vs Triton."""
    from turboquant.rotorquant import RotorQuantMSE
    from turboquant.triton_kernels import (
        triton_rotor_sandwich, pack_rotors_for_triton,
    )
    from turboquant.clifford import embed_vectors_as_multivectors, rotor_sandwich

    print("=" * 70)
    print("BENCHMARK A: Rotor Sandwich (embed + R x R̃)")
    print("=" * 70)

    d = 128
    bits = 3
    device = "cuda"

    rq = RotorQuantMSE(d, bits, seed=42, device=device)
    packed_rotors = pack_rotors_for_triton(rq.rotors).to(device)

    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  d={d}, bits={bits}\n")
    print(f"  {'n_vecs':>8s}  {'PyTorch (ms)':>14s}  {'Triton (ms)':>14s}  {'Speedup':>10s}")
    print(f"  {'─'*8}  {'─'*14}  {'─'*14}  {'─'*10}")

    for n in [256, 1024, 4096, 16384, 65536]:
        x = torch.randn(n, d, device=device)

        def pytorch_fn():
            mv = embed_vectors_as_multivectors(x)
            return rotor_sandwich(rq.rotors, mv)

        def triton_fn():
            return triton_rotor_sandwich(x, packed_rotors)

        t_pt = time_fn(pytorch_fn)
        t_tr = time_fn(triton_fn)
        speedup = t_pt / t_tr

        print(f"  {n:>8d}  {t_pt:>14.3f}  {t_tr:>14.3f}  {speedup:>9.1f}x")

    print()


def benchmark_full_fused():
    """Benchmark full pipeline: PyTorch vs Triton."""
    from turboquant.rotorquant import RotorQuantMSE
    from turboquant.triton_kernels import (
        triton_rotor_full_fused, pack_rotors_for_triton,
    )

    print("=" * 70)
    print("BENCHMARK B: Full Fused Pipeline (embed→rotor→quantize→unrotor→extract)")
    print("=" * 70)

    d = 128
    bits = 3
    device = "cuda"

    rq = RotorQuantMSE(d, bits, seed=42, device=device)
    packed_rotors = pack_rotors_for_triton(rq.rotors).to(device)
    c_s = None
    c_v = getattr(rq, 'centroids_vector')
    c_b = None
    c_t = None

    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  d={d}, bits={bits}\n")
    print(f"  {'n_vecs':>8s}  {'PyTorch (ms)':>14s}  {'Triton (ms)':>14s}  {'Speedup':>10s}")
    print(f"  {'─'*8}  {'─'*14}  {'─'*14}  {'─'*10}")

    for n in [256, 1024, 4096, 16384, 65536]:
        x = torch.randn(n, d, device=device)
        x = x / x.norm(dim=-1, keepdim=True)

        def pytorch_fn():
            return rq(x)

        def triton_fn():
            return triton_rotor_full_fused(x, packed_rotors, c_s, c_v, c_b, c_t)

        t_pt = time_fn(pytorch_fn)
        t_tr = time_fn(triton_fn)
        speedup = t_pt / t_tr

        print(f"  {n:>8d}  {t_pt:>14.3f}  {t_tr:>14.3f}  {speedup:>9.1f}x")

    print()


def benchmark_fused_attention():
    """Benchmark fused attention scores on compressed keys."""
    from turboquant.rotorquant import RotorQuantMSE
    from turboquant.triton_kernels import (
        triton_fused_attention, triton_rotor_sandwich,
        triton_rotor_inverse_sandwich, pack_rotors_for_triton,
    )

    print("=" * 70)
    print("BENCHMARK C: Fused Attention Scores (Q @ K^T on compressed keys)")
    print("=" * 70)

    d = 128
    bits = 3
    n_q_heads = 8
    n_kv_heads = 4
    device = "cuda"

    rq = RotorQuantMSE(d, bits, seed=42, device=device)
    packed_rotors = pack_rotors_for_triton(rq.rotors).to(device)
    scale = 1.0 / math.sqrt(d)

    # Use vector-grade centroids as the single centroid table
    centroids = getattr(rq, 'centroids_vector')

    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  d={d}, bits={bits}, q_heads={n_q_heads}, kv_heads={n_kv_heads}\n")
    print(f"  {'kv_len':>8s}  {'Standard (ms)':>14s}  {'Fused (ms)':>14s}  {'Speedup':>10s}  {'Cos sim':>10s}")
    print(f"  {'─'*8}  {'─'*14}  {'─'*14}  {'─'*10}  {'─'*10}")

    gqa_ratio = n_q_heads // n_kv_heads

    for kv_len in [128, 512, 1024, 2048, 4096]:
        batch = 1
        q = torch.randn(batch, n_q_heads, 1, d, device=device, dtype=torch.float32)
        k = torch.randn(batch, n_kv_heads, kv_len, d, device=device, dtype=torch.float32)
        k = k / k.norm(dim=-1, keepdim=True)

        # Quantize keys to uint8 indices
        k_flat = k.reshape(-1, d)
        k_norms = k_flat.norm(dim=-1)
        k_unit = k_flat / k_norms.unsqueeze(-1)
        k_rot = k_unit @ rq.mse.Pi.T if hasattr(rq, 'mse') else k_unit

        # Simple scalar quantization for indices
        diffs = k_flat.unsqueeze(-1) - centroids
        k_indices = diffs.abs().argmin(dim=-1).to(torch.uint8)
        k_indices = k_indices.reshape(batch, n_kv_heads, kv_len, d)
        k_norms_reshaped = k_norms.reshape(batch, n_kv_heads, kv_len).half()

        # Pre-rotate queries
        q_rot = q.contiguous()

        # Standard: dequantize + matmul
        def standard_fn():
            k_deq = centroids[k_indices.long()]
            k_exp = k_deq.repeat_interleave(gqa_ratio, dim=1)
            return torch.matmul(q, k_exp.transpose(2, 3)) * scale

        # Fused: Triton kernel
        def fused_fn():
            return triton_fused_attention(q_rot, k_indices, k_norms_reshaped,
                                          centroids, scale)

        # Verify correctness
        ref = standard_fn()
        fused = fused_fn()
        cos = F.cosine_similarity(ref.flatten().unsqueeze(0),
                                  fused.flatten().unsqueeze(0)).item()

        t_std = time_fn(standard_fn)
        t_fused = time_fn(fused_fn)
        speedup = t_std / t_fused

        print(f"  {kv_len:>8d}  {t_std:>14.3f}  {t_fused:>14.3f}  {speedup:>9.1f}x  {cos:>10.6f}")

    print()


def benchmark_varying_dimensions():
    """Benchmark across different embedding dimensions."""
    from turboquant.rotorquant import RotorQuantMSE
    from turboquant.triton_kernels import (
        triton_rotor_full_fused, pack_rotors_for_triton,
    )

    print("=" * 70)
    print("BENCHMARK D: Varying Embedding Dimensions")
    print("=" * 70)

    n = 4096
    bits = 3
    device = "cuda"

    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  n_vectors={n}, bits={bits}\n")
    print(f"  {'dim':>6s}  {'n_groups':>8s}  {'PyTorch (ms)':>14s}  {'Triton (ms)':>14s}  {'Speedup':>10s}  {'Cos sim':>10s}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*14}  {'─'*14}  {'─'*10}  {'─'*10}")

    for d in [64, 128, 256, 512]:
        rq = RotorQuantMSE(d, bits, seed=42, device=device)
        packed_rotors = pack_rotors_for_triton(rq.rotors).to(device)
        c_s = None
        c_v = getattr(rq, 'centroids_vector')
        c_b = None
        c_t = None
        n_groups = (d + 2) // 3

        x = torch.randn(n, d, device=device)
        x = x / x.norm(dim=-1, keepdim=True)

        # Verify
        x_hat_ref, _ = rq(x)
        x_hat_tr = triton_rotor_full_fused(x, packed_rotors, c_s, c_v, c_b, c_t)
        cos = F.cosine_similarity(x_hat_ref, x_hat_tr, dim=-1).mean().item()

        def pytorch_fn():
            return rq(x)

        def triton_fn():
            return triton_rotor_full_fused(x, packed_rotors, c_s, c_v, c_b, c_t)

        t_pt = time_fn(pytorch_fn)
        t_tr = time_fn(triton_fn)
        speedup = t_pt / t_tr

        print(f"  {d:>6d}  {n_groups:>8d}  {t_pt:>14.3f}  {t_tr:>14.3f}  {speedup:>9.1f}x  {cos:>10.6f}")

    print()


def benchmark_vs_turboquant():
    """Compare RotorQuant-Triton end-to-end against TurboQuant PyTorch."""
    from turboquant.turboquant import TurboQuantMSE
    from turboquant.rotorquant import RotorQuantMSE
    from turboquant.triton_kernels import (
        triton_rotor_full_fused, pack_rotors_for_triton,
    )

    print("=" * 70)
    print("BENCHMARK E: RotorQuant-Triton vs TurboQuant-PyTorch (end-to-end)")
    print("=" * 70)

    d = 128
    device = "cuda"

    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  d={d}\n")
    print(f"  {'bits':>4s}  {'n_vecs':>8s}  {'TQ-PyTorch':>14s}  {'RQ-PyTorch':>14s}  {'RQ-Triton':>14s}  {'TQ/Triton':>10s}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*14}  {'─'*14}  {'─'*14}  {'─'*10}")

    for bits in [2, 3, 4]:
        tq = TurboQuantMSE(d, bits, seed=42, device=device)
        rq = RotorQuantMSE(d, bits, seed=42, device=device)
        packed_rotors = pack_rotors_for_triton(rq.rotors).to(device)
        c_s = None
        c_v = getattr(rq, 'centroids_vector')
        c_b = None
        c_t = None

        for n in [1024, 4096, 16384]:
            x = torch.randn(n, d, device=device)
            x = x / x.norm(dim=-1, keepdim=True)

            def tq_fn():
                return tq(x)

            def rq_pt_fn():
                return rq(x)

            def rq_tr_fn():
                return triton_rotor_full_fused(x, packed_rotors, c_s, c_v, c_b, c_t)

            t_tq = time_fn(tq_fn)
            t_rq_pt = time_fn(rq_pt_fn)
            t_rq_tr = time_fn(rq_tr_fn)
            speedup = t_tq / t_rq_tr

            print(f"  {bits:>4d}  {n:>8d}  {t_tq:>12.3f}ms  {t_rq_pt:>12.3f}ms  "
                  f"{t_rq_tr:>12.3f}ms  {speedup:>9.1f}x")

    print()


def benchmark_bitwidth_sweep():
    """Benchmark across different bit widths."""
    from turboquant.rotorquant import RotorQuantMSE
    from turboquant.triton_kernels import (
        triton_rotor_full_fused, pack_rotors_for_triton,
    )

    print("=" * 70)
    print("BENCHMARK F: Bit-width Sweep (quality vs speed)")
    print("=" * 70)

    d = 128
    n = 4096
    device = "cuda"

    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  d={d}, n={n}\n")
    print(f"  {'bits':>4s}  {'n_levels':>8s}  {'MSE (PT)':>12s}  {'MSE (Tri)':>12s}  "
          f"{'PT (ms)':>10s}  {'Tri (ms)':>10s}  {'Speedup':>10s}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}")

    x = torch.randn(n, d, device=device)
    x = x / x.norm(dim=-1, keepdim=True)

    for bits in [1, 2, 3, 4]:
        rq = RotorQuantMSE(d, bits, seed=42, device=device)
        packed_rotors = pack_rotors_for_triton(rq.rotors).to(device)
        c_s = None
        c_v = getattr(rq, 'centroids_vector')
        c_b = None
        c_t = None

        # Quality
        x_hat_pt, _ = rq(x)
        mse_pt = ((x - x_hat_pt) ** 2).sum(dim=-1).mean().item()

        x_hat_tr = triton_rotor_full_fused(x, packed_rotors, c_s, c_v, c_b, c_t)
        mse_tr = ((x - x_hat_tr) ** 2).sum(dim=-1).mean().item()

        # Speed
        def pytorch_fn():
            return rq(x)

        def triton_fn():
            return triton_rotor_full_fused(x, packed_rotors, c_s, c_v, c_b, c_t)

        t_pt = time_fn(pytorch_fn)
        t_tr = time_fn(triton_fn)
        speedup = t_pt / t_tr

        print(f"  {bits:>4d}  {2**bits:>8d}  {mse_pt:>12.6f}  {mse_tr:>12.6f}  "
              f"{t_pt:>8.3f}ms  {t_tr:>8.3f}ms  {speedup:>9.1f}x")

    print()


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  RotorQuant Triton Kernel Benchmarks                               ║")
    print("║  Comparing: PyTorch (pure ops) vs Triton (fused GPU kernels)       ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for Triton benchmarks")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import triton
        print(f"Triton: {triton.__version__}")
    except ImportError:
        print("ERROR: Triton not installed. Run: pip install triton")
        sys.exit(1)
    print()

    # Step 1: Verify correctness
    ok = verify_correctness()
    if not ok:
        print("CORRECTNESS CHECK FAILED — aborting benchmarks")
        sys.exit(1)

    # Step 2: Run benchmarks
    benchmark_rotor_sandwich()
    benchmark_full_fused()
    benchmark_fused_attention()
    benchmark_varying_dimensions()
    benchmark_vs_turboquant()
    benchmark_bitwidth_sweep()

    print("=" * 70)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 70)
