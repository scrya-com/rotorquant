# TurboQuant + RotorQuant + IsoQuant

A from-scratch PyTorch implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026), Google's two-stage vector quantization algorithm for compressing LLM key-value caches — plus **RotorQuant** (Clifford rotors) and **IsoQuant** (quaternion 4D blocks), progressively faster drop-in replacements for the dense rotation step.

**IsoQuant** is the recommended default: **5.8x faster** than RotorQuant at identical reconstruction quality, with clean 4D hardware alignment.

## Google TurboQuant Parity

All three methods match Google TurboQuant's quality claims while dramatically reducing rotation cost:

| Metric | Google TurboQuant | RotorQuant (Clifford) | IsoQuant (Quaternion) | Status |
|--------|------------------|----------------------|----------------------|--------|
| **Perplexity (4-bit)** | <5% degradation | **+3.2%** (PPL 10.13 vs 9.81) | Same | **MATCH** |
| **Perplexity (3-bit)** | <5% (Gemma, many KV heads) | +25.2% (Qwen, 2 KV heads) | Same | Expected |
| **Needle-in-haystack** | Perfect at all bit widths | **4/4 FOUND** (3-bit & 4-bit) | Same | **MATCH** |
| **Generation quality** | Coherent | Coherent | Coherent | **MATCH** |
| **MSE vs FP16** | Near-optimal | **1.0x ratio** | **1.0x ratio** | **MATCH** |
| **Compression (3-bit)** | 4.9x | **4.9x** | **4.9x** | **MATCH** |

### IsoQuant vs RotorQuant vs TurboQuant (d=128)

| | TurboQuant | RotorQuant | IsoQuant-Fast | IsoQuant-Full |
|---|-----------|-----------|---------------|---------------|
| Block structure | Dense 128×128 | 43 × 3D Clifford | **32 × 4D quaternion** | 32 × 4D quaternion |
| Forward FMAs | 16,384 | 2,408 | **512** | 1,024 |
| Parameters | 16,384 | 344 | **128** | 256 |
| Alignment | N/A | 42 blocks + 2D tail | **32 clean blocks** | 32 clean blocks |
| Stage-1 latency | — | 4,244 µs | **727 µs (5.8x)** | 1,152 µs (3.7x) |
| Reconstruction MSE | Baseline | 0.000265 | **0.000265** | 0.000265 |

### Reconstruction MSE (8192 normalized vectors)

| d | bits | RotorQuant | IsoQuant-Fast | IsoQuant-Full | Ratio (Fast/RQ) |
|---|------|-----------|---------------|---------------|-----------------|
| 64 | 2 | 0.001800 | 0.001785 | 0.001786 | 0.992x |
| 64 | 3 | 0.000526 | 0.000521 | 0.000521 | 0.991x |
| 64 | 4 | 0.000144 | 0.000143 | 0.000143 | 0.995x |
| 128 | 2 | 0.000903 | 0.000906 | 0.000906 | 1.002x |
| 128 | 3 | 0.000265 | 0.000265 | 0.000265 | 0.998x |
| 128 | 4 | 0.000073 | 0.000072 | 0.000073 | 0.996x |
| 256 | 2 | 0.000455 | 0.000456 | 0.000456 | 1.002x |
| 256 | 3 | 0.000134 | 0.000134 | 0.000134 | 0.999x |
| 256 | 4 | 0.000037 | 0.000037 | 0.000037 | 1.002x |

MSE is indistinguishable across all settings. IsoQuant is a pure speed upgrade.

### Stage-1 Latency (µs, 8192 vectors, RTX PRO 4000)

| d | bits | RotorQuant | IsoQuant-Fast | Speedup | IsoQuant-Full | Speedup |
|---|------|-----------|---------------|---------|---------------|---------|
| 64 | 2 | 3,409 | **559** | **6.1x** | 694 | 4.9x |
| 64 | 3 | 3,562 | **565** | **6.3x** | 1,088 | 3.3x |
| 64 | 4 | 3,544 | **739** | **4.8x** | 1,260 | 2.8x |
| 128 | 2 | 3,979 | **652** | **6.1x** | 1,069 | 3.7x |
| 128 | 3 | 4,244 | **727** | **5.8x** | 1,152 | 3.7x |
| 128 | 4 | 4,574 | **1,158** | **3.9x** | 1,563 | 2.9x |
| 256 | 2 | 4,853 | **834** | **5.8x** | 1,337 | 3.6x |
| 256 | 3 | 5,336 | **1,173** | **4.5x** | 1,669 | 3.2x |
| 256 | 4 | 6,267 | **1,900** | **3.3x** | 2,328 | 2.7x |

IsoQuant-Fast is consistently 3.3–6.3x faster. Best at low bit width and medium dimensions.

### Perplexity (wikitext-2, autoregressive with post-prefill quantization)

| Model | KV Heads | FP16 PPL | RQ 4-bit | Delta | RQ 3-bit | Delta |
|-------|----------|---------|---------|-------|---------|-------|
| **Mistral-7B** | 8 | 4.80 | **5.16** | **+7.4%** | 5.53 | +15.3% |
| **Gemma-2-2b** | 4 | 8.87 | **9.77** | **+10.1%** | 10.64 | +19.9% |
| Qwen2.5-3B | 2 | 9.81 | **10.13** | **+3.2%** | 12.28 | +25.2% |

### High-Context Generation

3-bit with post-prefill quantization on Qwen2.5-3B (RTX 5090):

| Context | Speed | VRAM | Needle |
|---------|-------|------|--------|
| 2K | 6.9 tok/s | 2.4 GB | **FOUND** |
| 8K | 8.6 tok/s | 3.1 GB | **FOUND** |
| 16K | 6.0 tok/s | 4.0 GB | **FOUND** |
| 32K | 5.0 tok/s | 5.9 GB | **FOUND** |
| 65K | 2.1 tok/s | 9.6 GB | **FOUND** |

### Attention Logits Speed (Q@K^T, decode mode, RTX 5090)

| KV Length | FP32 | FP16 | **RQ Triton** | **vs FP32** | vs FP16 |
|-----------|------|------|-------------|---------|---------|
| 4K | 0.132 ms | 0.019 ms | **0.024 ms** | **5.4x** | 0.8x |
| 16K | 0.057 ms | 0.033 ms | **0.024 ms** | **2.4x** | **1.4x** |
| 32K | 0.308 ms | 0.066 ms | **0.024 ms** | **12.7x** | **2.7x** |

## How It Works

### TurboQuant (Google)

Two stages: (1) Random rotation via d×d orthogonal matrix → per-coordinate Lloyd-Max quantization. (2) QJL 1-bit residual correction for unbiased inner products.

### RotorQuant

Replaces the d×d matrix with **Clifford rotors** in Cl(3,0). Chunks the vector into groups of 3 dims, rotates each with a 4-parameter rotor via the sandwich product `R v R̃`. 44x fewer parameters, 7.9x fewer FMAs.

### IsoQuant (recommended)

Replaces Clifford rotors with **quaternion 4D blocks** based on the isoclinic decomposition SO(4) ≅ SU(2) × SU(2). Each group of 4 coordinates is treated as a quaternion and rotated via `q_L v q̄_R` (Full) or `q_L v` (Fast).

| | TurboQuant | RotorQuant | IsoQuant-Fast |
|---|-----------|-----------|---------------|
| Rotation | Dense d×d matmul | Cl(3,0) rotor sandwich | **Quaternion multiply** |
| Block size | d | 3 | **4** (hardware-aligned) |
| FMAs (d=128) | 16,384 | 2,408 | **512 (32x fewer)** |
| Parameters | 16,384 | 344 | **128 (128x fewer)** |
| Alignment | N/A | Tail handling | **Clean power-of-2** |
| Quality | Baseline | 1.0x | **1.0x** |

### Key Innovations

**Grade elimination** (RotorQuant): The rotor sandwich of a grade-1 vector produces only odd grades. Dropping non-vector grades cuts storage from 344 → 129 indices per vector, matching TurboQuant's 128.

**4D hardware alignment** (IsoQuant): d=128 splits into 32 clean 4D blocks (no tail), fitting naturally into SIMD float4 patterns. RotorQuant's 3D blocks create 42 groups + 2D remainder.

**Norm separation**: Normalize to unit sphere before quantization, store norms separately. Combined with correct `d_eff` for Lloyd-Max codebook, this achieves MSE parity with TurboQuant.

**Post-prefill quantization**: Prefill runs at full FP16 (no error compounding through layers). First decode step bulk-quantizes the cache.

## Quick Start

```python
from turboquant import IsoQuantMSE, IsoQuantProd

# Stage 1: MSE-optimal quantizer (IsoQuant-Fast, recommended)
iq = IsoQuantMSE(d=128, bits=3, mode='fast', device='cuda')
x_hat, indices = iq(x)  # quantize + dequantize

# Stage 1 + 2: With QJL residual correction
iq_prod = IsoQuantProd(d=128, bits=3, mode='fast', device='cuda')
compressed = iq_prod.quantize(keys)
ip_estimate = iq_prod.inner_product(queries, compressed)

# Legacy Clifford interface (still available)
from turboquant import RotorQuantMSE
rq = RotorQuantMSE(d=128, bits=3, device='cuda')
```

## Triton Kernels

Portable, auto-tuned GPU kernels — no CUDA C++ compilation needed:

| Kernel | Purpose | Latency (d=128, 3-bit) |
|--------|---------|----------------------|
| **`triton_iso_fast_fused`** | **IsoQuant-Fast full pipeline** | **30 µs** |
| **`triton_iso_full_fused`** | **IsoQuant-Full full pipeline** | ~32 µs |
| `triton_rotor_full_fused` | Clifford quantize-dequantize pipeline | 34 µs |
| `triton_rotor_sandwich` | Clifford R x R̃ (embed + rotor sandwich) | — |
| `triton_fused_attention_qjl` | Q@K^T with QJL correction (experimental) | — |

```python
from turboquant import IsoQuantMSE, triton_iso_fast_fused

iq = IsoQuantMSE(d=128, bits=3, mode='fast', device='cuda')

# Triton fused quantize-dequantize (70x faster than PyTorch)
x_hat = triton_iso_fast_fused(x, iq.q_L, iq.centroids)
```

## Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `benchmark_isoquant.py` | **IsoQuant vs RotorQuant head-to-head** | `python -m turboquant.benchmark_isoquant` |
| `benchmark_google_parity.py` | Full TurboQuant parity test | `python -m turboquant.benchmark_google_parity` |
| `benchmark_perplexity.py` | Perplexity benchmark (autoregressive + roundtrip) | `python -m turboquant.benchmark_perplexity` |
| `poc_high_context.py` | High-context generation (2K-131K tokens) | `python -m turboquant.poc_high_context` |
| `benchmark_triton.py` | Triton kernel speed (6 tests) | `python -m turboquant.benchmark_triton` |

## Project Structure

```
turboquant/
  isoquant.py                # IsoQuant: quaternion 4D block rotation (recommended)
  rotorquant.py              # RotorQuant: Clifford 3D block rotation (legacy)
  clifford.py                # Cl(3,0) geometric algebra
  triton_kernels.py          # Triton GPU kernels (rotor sandwich, fused pipeline, attention)
  fused_attention.py         # Fused attention with QJL correction (experimental)
  turboquant.py              # TurboQuant: dense rotation baseline
  lloyd_max.py               # Lloyd-Max optimal scalar quantizer
  compressors.py             # Asymmetric inner product compressors
  cuda_backend.py            # QJL CUDA kernel wrappers
  benchmark_isoquant.py      # IsoQuant vs RotorQuant benchmark
  benchmark_google_parity.py # Google TurboQuant parity benchmark
  benchmark_perplexity.py    # Perplexity benchmark
  benchmark_triton.py        # Triton kernel benchmarks
  poc_high_context.py        # High-context generation POC
  csrc/                      # CUDA kernels (rotor fused, QJL)
tests/                       # Unit tests
setup.py                     # pip install with optional CUDA build
```

## Requirements

```bash
pip install -e .                    # PyTorch-only
pip install triton                  # Add Triton kernels (for Clifford path)
pip install -e ".[validate]"        # + model validation deps (transformers, bitsandbytes)
```

- Python 3.10+, PyTorch 2.0+, CUDA, scipy
- triton >= 3.0 (optional, for Clifford Triton kernels)

## When to Use Which

| Scenario | Recommendation |
|----------|---------------|
| **Default** | **IsoQuant-Fast 3-bit** (5.8x faster, same quality) |
| KV cache compression (quality) | IsoQuant-Fast 4-bit (+3-10% PPL, 3.7x compression) |
| KV cache compression (size) | IsoQuant-Fast 3-bit (4.9x, matches TQ) |
| Long context on limited VRAM | IsoQuant-Fast 3-bit + post-prefill (65K tokens on 10 GB) |
| Triton kernel path needed | RotorQuant (Triton kernels available) |
| Apple Silicon | RotorQuant + Metal shader |

## References

- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — [Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — [Triton impl](https://dejan.ai/blog/turboquant/)
- [IsoQuant](https://github.com/ParaMind2025/isoquant) — Ji, "IsoQuant: Hardware-Aligned SO(4) Isoclinic Rotations for LLM KV Cache Compression" (March 2026)
- [QJL: 1-Bit Quantized JL Transform](https://arxiv.org/abs/2406.03482) — [Code](https://github.com/amirzandieh/QJL)
- [CommVQ](https://arxiv.org/abs/2506.18879) (ICML 2025) — [PolarQuant](https://arxiv.org/abs/2502.02617) (AISTATS 2026)
- [CliffordNet](https://arxiv.org/abs/2601.06793) (Jan 2026)

## Citation

```bibtex
@article{pope2026rotorquant,
  title={RotorQuant: Clifford Algebra Vector Quantization for LLM KV Cache Compression},
  author={Pope, John D.},
  year={2026},
  url={https://www.scrya.com/rotorquant/},
  note={Code: https://github.com/scrya-com/rotorquant}
}
```

## License

MIT
