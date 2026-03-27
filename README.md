# TurboQuant + RotorQuant

A from-scratch PyTorch implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026), Google's two-stage vector quantization algorithm for compressing LLM key-value caches — plus **RotorQuant**, a Clifford algebra reimagining with **44x fewer parameters** and **Triton GPU kernels** for the quantize/dequantize pipeline.

## Quick Results

### RotorQuant High-Context Generation (NEW)

3-bit RotorQuant with post-prefill quantization on Qwen2.5-3B-Instruct:

| Context | Speed | VRAM | Needle-in-Haystack |
|---------|-------|------|--------------------|
| 2K | 4.5 tok/s | 2.4 GB | **FOUND** — "AURORA-7749" |
| 8K | 4.9 tok/s | 3.1 GB | **FOUND** |
| 16K | 6.4 tok/s | 4.0 GB | **FOUND** |
| 32K | 1.9 tok/s | 5.9 GB | **FOUND** |
| 65K | 0.9 tok/s | 9.6 GB | **FOUND** |
| 93K | 2.2 tok/s | 12.8 GB | **FOUND** (RTX 5090) |

FP16 baseline: ~40-50 tok/s. RotorQuant trades speed for memory — it's a **memory optimization**, not a speed optimization. The quantize/dequantize overhead slows generation, but enables fitting much longer contexts in VRAM.

### MSE Quality (after codebook fix)

RotorQuant now **matches TurboQuant exactly** on reconstruction quality:

| Bits | TurboQuant MSE | RotorQuant MSE | Ratio | Cosine Sim |
|------|---------------|---------------|-------|------------|
| 2 | 0.116 | 0.116 | **1.0x** | 0.941 |
| 3 | 0.034 | 0.034 | **1.0x** | 0.983 |
| 4 | 0.009 | 0.009 | **1.0x** | 0.995 |

This was achieved by fixing two bugs: correct Lloyd-Max `d_eff` parameter and adding norm separation (see [Codebook Fix](#codebook-calibration-fix) below).

### Triton Kernel Speed (quantize/dequantize operation)

The Triton fused kernel is 100-650x faster than PyTorch for the quantize-dequantize roundtrip. **Note**: this measures the kernel itself, not end-to-end inference throughput.

| n_vectors | RQ PyTorch | **RQ Triton** | Speedup |
|-----------|------------|--------------|---------|
| 1,024 | 3.22 ms | 0.015 ms | **212x** |
| 4,096 | 3.46 ms | 0.015 ms | **229x** |
| 16,384 | 5.10 ms | 0.017 ms | **295x** |
| 65,536 | 28.13 ms | 0.043 ms | **652x** |

Tested on RTX 5090, d=128, 3-bit. The Triton kernel fuses the entire embed→rotor→quantize→unrotor→extract pipeline into a single GPU kernel launch.

### Parameter Efficiency

| Method | Parameters (d=128) | Breakdown |
|--------|-------------------|-----------|
| TurboQuant | 16,399 | 128x128 rotation matrix + codebook |
| **RotorQuant** | **~380** | 43 rotors x 8 + 4 grade codebooks |
| **Ratio** | **~44x fewer** | |

At d=4096 (typical LLM head dim): TQ needs 16.7M params, RQ needs ~11K.

## Background

When an LLM generates text, it stores a **key** and **value** vector for every token it has seen, in every layer. This is the KV cache — the model's working memory. At 8K tokens on a 36-layer model like Qwen2.5-3B, this cache is **289 MB** in FP16. On a 12GB GPU, the KV cache — not the model weights — becomes the bottleneck for long context.

TurboQuant compresses this cache by quantizing the vectors to 2-4 bits per coordinate, achieving 3-7x compression with minimal impact on attention accuracy.

## How TurboQuant Works

The algorithm has two stages:

### Stage 1: Random Rotation + Lloyd-Max Quantization

Each vector is multiplied by a random orthogonal matrix (generated via QR decomposition of a Gaussian matrix). This rotation makes every coordinate follow a predictable Beta distribution (well-approximated by Gaussian N(0, 1/d)), enabling optimal per-coordinate Lloyd-Max scalar quantization.

### Stage 2: QJL Residual Correction (1 bit)

The Quantized Johnson-Lindenstrauss transform fixes the inner product bias from Stage 1. It projects the quantization residual through a random Gaussian matrix and stores just the **sign** — exactly 1 bit per dimension — making the inner product estimate mathematically unbiased.

```
<q, k> ~ <q, k_mse> + ||residual|| * sqrt(pi/2) / m * <S @ q, sign(S @ residual)>
```

## How RotorQuant Works

RotorQuant replaces TurboQuant's d x d random orthogonal matrix with **Clifford rotors** in Cl(3,0):

### The Key Idea

Instead of `Pi @ x` (16,384 multiply-adds for d=128), RotorQuant does `R x R_tilde` (rotor sandwich product) — only ~100 multiply-adds per vector, exploiting the algebraic structure of geometric algebra.

**Cl(3,0) multivectors** have 8 components: `[1, e1, e2, e3, e12, e13, e23, e123]`

A **rotor** R has only 4 non-zero components: `R = [s, 0, 0, 0, b12, b13, b23, 0]` (scalar + bivectors). This sparsity eliminates ~50% of the geometric product's FMAs.

### Why Rotors?

| Property | TurboQuant (Pi matrix) | RotorQuant (Rotor R) |
|----------|----------------------|---------------------|
| Parameters | d^2 = 16,384 | 8 per group x ceil(d/3) = 344 |
| Operations | d^2 FMAs (matmul) | ~100 FMAs (sparse GP) |
| Preserves | Norms + inner products | Norms + inner products + outer products + grades |
| Composition | Pi2 Pi1 (matrix multiply) | R2 R1 (geometric product) |

### Codebook Calibration Fix

The original RotorQuant had 2-4x worse MSE than TurboQuant. Two fixes brought it to exact parity:

1. **`d_eff` parameter**: The Lloyd-Max codebook was built with `d_eff = n_groups * 8 = 344`, making the Gaussian approximation use σ = 1/√344 ≈ 0.054. But after the rotor sandwich, vector-grade components have σ ≈ 1/√128 ≈ 0.088. The codebook centroids only covered ±0.116 when values ranged to ±0.50, clipping 67% of outlier magnitude. Fix: use `d_eff = d` (the original vector dimension).

2. **Norm separation**: Vectors must be normalized to the unit sphere before quantization. The codebook is designed for unit-sphere coordinates. Store norms separately and rescale after dequantization. This matches TurboQuant's approach.

### Post-Prefill Quantization

Naively quantizing keys during the forward pass causes error to compound through 36 layers — each layer's quantization noise feeds into the next. Both TurboQuant and RotorQuant produce garbage output with this approach.

The fix is **post-prefill quantization**:
1. **Prefill** runs at full FP16 precision — no quantization, no error compounding
2. **First decode step**: bulk-quantize the entire prefill cache via Triton fused kernel
3. **Each decode step**: quantize the new key for storage, but return the full-precision key for the current token's attention

This gives perfect prefill quality with compressed cache for decode.

### The Fused CUDA Kernel

The entire pipeline runs in a single kernel launch:

```
embed (3 dims -> multivector) -> R x R_tilde -> Lloyd-Max quantize -> R_tilde x R -> extract
```

Each thread handles one (batch_item, group) pair. Rotors and codebooks are loaded into shared memory. The sparse geometric product uses only 28 FMAs instead of 64 for the full product.

## Real Model Validation

### TurboQuant on Qwen2.5-3B-Instruct (all 36 layers, 72 KV heads)

| Config | Context | Cache Size | Compression | Cosine Sim | Top-1 | Top-5 |
|--------|---------|-----------|-------------|-----------|-------|-------|
| FP16 | 2K | 72.6 MB | 1.0x | - | - | - |
| TQ-4bit | 2K | 19.0 MB | 3.8x | 0.9988 | 86.1% | 95.8% |
| TQ-3bit | 2K | 14.5 MB | 5.0x | 0.9961 | 84.7% | 94.4% |
| TQ-2bit | 2K | 9.9 MB | 7.3x | 0.9897 | 63.9% | 83.3% |
| FP16 | 4K | 143.8 MB | 1.0x | - | - | - |
| TQ-4bit | 4K | 37.6 MB | 3.8x | 0.9986 | 91.7% | 94.4% |
| TQ-3bit | 4K | 28.6 MB | 5.0x | 0.9955 | 72.2% | 90.3% |
| TQ-2bit | 4K | 19.7 MB | 7.3x | 0.9878 | 65.3% | 83.3% |
| FP16 | 8K | 289.0 MB | 1.0x | - | - | - |
| TQ-4bit | 8K | 75.6 MB | 3.8x | 0.9983 | 86.1% | 95.8% |
| TQ-3bit | 8K | 57.6 MB | 5.0x | 0.9945 | 84.7% | 93.1% |
| TQ-2bit | 8K | 39.5 MB | 7.3x | 0.9851 | 68.1% | 87.5% |

**3-bit is the sweet spot**: 5x compression with 99.5% attention fidelity. At 128K context, that's ~3.6 GB instead of ~18 GB — fitting on a single 24GB GPU.

## CUDA Kernels

### QJL Kernels (from [amirzandieh/QJL](https://github.com/amirzandieh/QJL))

Fused CUDA kernels for 1-bit quantization and attention score computation:

| Kernel | Purpose |
|--------|---------|
| `qjl_quant_kernel.cu` | Fused random projection + sign quantization + outlier separation |
| `qjl_score_kernel.cu` | Fused attention score from 1-bit quantized keys |
| `qjl_gqa_score_kernel.cu` | Grouped Query Attention variant |
| `quantization.cu` | Quantized batched matmul for value reconstruction |

### RotorQuant Fused Kernels (CUDA + Metal)

Single fused kernel for the full RotorQuant pipeline on both NVIDIA and Apple Silicon:

```
normalize -> embed -> rotor_sandwich -> quantize -> inverse_sandwich -> extract -> rescale
```

Exploits rotor sparsity (4 of 8 multivector components are zero) to cut FMAs by ~50%. Each thread handles one (batch, group) pair with rotors and codebooks in shared/threadgroup memory.

Build:
```bash
# NVIDIA: Build CUDA kernels
CUDA_HOME=/usr/local/cuda python setup.py build_ext --inplace

# Apple Silicon: Compile Metal shader
xcrun -sdk macosx metal -c turboquant/rotor_fused.metal -o /tmp/rotor_fused.air -std=metal3.0
xcrun -sdk macosx metallib /tmp/rotor_fused.air -o turboquant/rotor_fused.metallib
```

## Triton Kernels

Portable, auto-tuned GPU kernels using [Triton](https://github.com/triton-lang/triton) — no CUDA C++ compilation needed, works on both NVIDIA and AMD GPUs.

Inspired by [TurboQuant's Triton attention kernel](https://dejan.ai/blog/turboquant/) which fuses Q@K^T on quantized keys, we built Triton kernels for the entire RotorQuant pipeline:

| Kernel | Purpose | Speedup vs PyTorch |
|--------|---------|-------------------|
| `triton_rotor_sandwich` | R x R̃ (embed + rotor sandwich) | 80-166x |
| `triton_rotor_full_fused` | Full quantize-dequantize pipeline | **128-652x** |
| `triton_fused_attention` | Q@K^T on compressed keys (gather-dot) | 1.1-1.5x |
| `triton_rotor_inverse_sandwich` | R̃ x R (dequantize path) | 80-166x |

### Usage

```python
from turboquant import RotorQuantMSE, pack_rotors_for_triton, triton_rotor_full_fused

# Create quantizer (PyTorch)
rq = RotorQuantMSE(d=128, bits=3, device="cuda")

# Pack rotors for Triton (one-time, ~0 cost)
packed_rotors = pack_rotors_for_triton(rq.rotors)

# Get centroids
c_s = rq.centroids_scalar
c_v = rq.centroids_vector
c_b = rq.centroids_bivector
c_t = rq.centroids_trivector

# Triton fused quantize-dequantize (200-650x faster than PyTorch)
x_hat = triton_rotor_full_fused(x, packed_rotors, c_s, c_v, c_b, c_t)
```

### Non-Commutative Algebra Bug Fix

During Triton development, we discovered and fixed a bug in the CUDA kernel's sparse geometric product. The rotor sandwich `R x R̃ = (R * x) * R̃` requires two DIFFERENT products:

- **R * x** (rotor on LEFT) — `_gp_rotor_mv`
- **temp * R̃** (rotor on RIGHT) — `_gp_mv_rotor`

These differ because Clifford algebra is non-commutative. The original CUDA kernel used the left-product formula for both steps, computing `R̃ * (R * x)` instead of `(R * x) * R̃`. For grade-1 vector inputs, the scalar and bivector intermediate components happen to be zero, so the error was small but non-zero. Both the Triton and CUDA kernels now use the correct left+right product pair.

## Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `test_turboquant.py` | Synthetic validation (codebook, MSE, QJL, needle) | `python -m turboquant.test_turboquant` |
| `validate.py` | Real model validation (Qwen2.5-3B, all layers) | `python -m turboquant.validate` |
| `validate_rotorquant.py` | RotorQuant vs TurboQuant on real model | `python -m turboquant.validate_rotorquant` |
| `poc_high_context.py` | **High-context generation POC (2K-93K tokens)** | `python -m turboquant.poc_high_context` |
| `benchmark_triton.py` | Triton vs PyTorch benchmark (6 tests) | `python -m turboquant.benchmark_triton` |
| `benchmark_cuda.py` | PyTorch vs QJL CUDA kernel speed | `python -m turboquant.benchmark_cuda` |
| `benchmark_rotorquant.py` | Full 7-test RotorQuant vs TurboQuant comparison | `python -m turboquant.benchmark_rotorquant` |
| `benchmark_metal.py` | Metal shader benchmark (Apple Silicon) | `python -m turboquant.benchmark_metal` |

## Project Structure

```
turboquant/
  __init__.py                # Package exports
  lloyd_max.py               # Lloyd-Max optimal scalar quantizer solver
  turboquant.py              # TurboQuant: TurboQuantMSE, TurboQuantProd, TurboQuantKVCache
  rotorquant.py              # RotorQuant: RotorQuantMSE, RotorQuantProd, RotorQuantKVCache
  clifford.py                # Cl(3,0) geometric algebra (geometric product, rotors, sandwich)
  triton_kernels.py          # Triton GPU kernels (rotor sandwich, full fused, attention)
  compressors.py             # Asymmetric inner product compressors for validation
  cuda_backend.py            # QJL CUDA kernel wrappers with PyTorch fallback
  poc_high_context.py        # High-context generation POC
  csrc/
    rotor_fused_kernel.cu    # Fused RotorQuant CUDA kernel (NVIDIA)
    qjl_quant_kernel.cu      # QJL quantization kernel
    qjl_score_kernel.cu      # QJL attention score kernel
    qjl_gqa_score_kernel.cu  # QJL GQA score kernel
    quantization.cu          # Quantized batched matmul
  rotor_fused.metal          # Fused RotorQuant Metal shader (Apple Silicon)
  benchmark_triton.py        # Triton vs PyTorch benchmarks
  benchmark_cuda.py          # CUDA kernel benchmarks
  benchmark_rotorquant.py    # RotorQuant vs TurboQuant benchmarks
setup.py                     # pip install with optional CUDA build
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- scipy (for codebook computation)
- triton >= 3.0 (for Triton GPU kernels — optional but recommended)
- transformers, accelerate, bitsandbytes (for real model validation and POC)

```bash
pip install -e .                    # PyTorch-only
pip install triton                  # Add Triton kernels (100-650x faster quantize/dequantize)
pip install -e ".[validate]"        # + model validation deps
python setup.py build_ext --inplace # Build CUDA kernels (alternative to Triton)
```

## When to Use Which

| Scenario | Recommendation |
|----------|---------------|
| Standard KV cache compression | TurboQuant 3-bit (proven, well-understood) |
| Long context on limited VRAM | **RotorQuant 3-bit + post-prefill quantization** |
| Parameter-constrained (edge/mobile) | RotorQuant (44x fewer params) |
| Apple Silicon | RotorQuant + Metal shader |
| Geometric data (3D, physics, robotics) | RotorQuant (preserves algebraic structure) |

**Note**: RotorQuant is a **memory** optimization. It enables longer contexts by compressing the KV cache, but the quantize/dequantize overhead reduces generation speed (~2-7 tok/s vs ~40-50 tok/s FP16 baseline on Qwen2.5-3B). For speed parity, a fused attention kernel that computes Q@K^T directly from compressed indices (avoiding decompression) is needed — this is the approach TurboQuant uses.

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization](https://arxiv.org/abs/2406.03482)
- [CommVQ: Commutative Vector Quantization for KV Cache Compression](https://arxiv.org/abs/2506.18879) (ICML 2025)
- [PolarQuant: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617)
- [QJL Reference Implementation](https://github.com/amirzandieh/QJL)
- [CliffordNet: All You Need is Geometric Algebra](https://arxiv.org/abs/2601.06793) (Jan 2026)
- [TurboQuant: From Paper to Triton Kernel](https://dejan.ai/blog/turboquant/)

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
