# TurboQuant + RotorQuant

A from-scratch PyTorch implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026), Google's two-stage vector quantization algorithm for compressing LLM key-value caches — plus **RotorQuant**, a Clifford algebra reimagining that is **10-19x faster** with **44x fewer parameters**.

## Quick Results

### RotorQuant vs TurboQuant — CUDA Fused Kernel Speed

Tested on RTX PRO 4000 Blackwell, d=128, 3-bit quantization:

| n_vectors | TurboQuant | RQ PyTorch | **RQ CUDA** | CUDA vs TQ |
|-----------|-----------|------------|------------|------------|
| 1,024 | 69 us | 3.30 ms | **6 us** | **11x faster** |
| 4,096 | 132 us | 3.86 ms | **12 us** | **11x faster** |
| 8,192 | 285 us | 4.70 ms | **20 us** | **14x faster** |
| 16,384 | 740 us | 6.71 ms | **39 us** | **19x faster** |

### Real Model Validation (Qwen2.5-3B-Instruct)

On actual KV cache data, RotorQuant **matches TurboQuant's attention fidelity** despite using 44x fewer parameters:

| Context | Bits | Method | Cosine Sim | Top-1 | Top-5 |
|---------|------|--------|------------|-------|-------|
| 2K | 3-bit | TurboQuant | 0.9906 | 81.2% | 93.8% |
| 2K | 3-bit | **RotorQuant** | 0.9903 | 81.2% | 93.8% |
| 4K | 4-bit | TurboQuant | 0.9880 | 75.0% | 93.8% |
| 4K | 4-bit | **RotorQuant** | 0.9874 | **81.2%** | 93.8% |

RotorQuant beats TurboQuant on top-1 accuracy at 4K/4-bit — the Clifford rotor decorrelation better preserves directional structure of real KV cache vectors.

### Parameter Efficiency

| Method | Parameters (d=128) | Breakdown |
|--------|-------------------|-----------|
| TurboQuant | 16,399 | 128x128 rotation matrix + codebook |
| **RotorQuant** | **372** | 43 rotors x 8 + 4 grade codebooks |
| **Ratio** | **44x fewer** | |

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

### The Fused CUDA Kernel

The entire pipeline runs in a single kernel launch:

```
embed (3 dims -> multivector) -> R x R_tilde -> Lloyd-Max quantize -> R_tilde x R -> extract
```

Each thread handles one (batch_item, group) pair. Rotors and codebooks are loaded into shared memory. The sparse geometric product (`gp_rotor_mv`) uses only 28 FMAs instead of 64 for the full product.

## Synthetic Benchmark Results

### MSE Distortion (d=128, 2000 random unit vectors)

| Bits | TurboQuant | RotorQuant | Theory Bound | Winner |
|------|-----------|-----------|-------------|--------|
| 1 | **0.361** | 0.457 | 0.680 | TQ |
| 2 | **0.116** | 0.197 | 0.170 | TQ |
| 3 | **0.034** | 0.081 | 0.043 | TQ |
| 4 | **0.009** | 0.032 | 0.011 | TQ |

TurboQuant wins on raw MSE — its full d x d rotation exactly induces the Beta distribution Lloyd-Max was optimized for. RotorQuant's 3-at-a-time grouping changes the distribution.

### Inner Product Preservation (d=128, 3000 pairs)

| Bits | Method | Bias | RMSE | Correlation |
|------|--------|------|------|-------------|
| 2 | TQ | +0.001 | 0.063 | 0.818 |
| 2 | RQ | -0.001 | 0.075 | 0.767 |
| 3 | TQ | -0.000 | 0.037 | **0.918** |
| 3 | RQ | +0.001 | 0.048 | 0.878 |
| 4 | TQ | +0.001 | 0.020 | **0.974** |
| 4 | RQ | -0.001 | 0.031 | 0.943 |

Both are unbiased (near-zero bias) thanks to QJL correction. TQ has better correlation on random vectors, but the gap narrows on real model data.

### Needle-in-Haystack Retrieval

**Perfect 9/9** for both methods across all bit-widths (2, 3, 4) and context lengths (512, 2048, 8192).

### Rotation Equivariance

Testing `quantize(R@x)` vs `R@quantize(x)` — how well each method handles pre-rotated data:

| Bits | Method | Equivariance Error | Cosine Sim |
|------|--------|-------------------|------------|
| 3 | TQ | 0.258 | 0.966 |
| 3 | RQ | 0.306 | 0.935 |
| 4 | TQ | 0.136 | 0.991 |
| 4 | RQ | 0.217 | 0.972 |

### Profiling Breakdown (before CUDA kernel)

| Step | Time | % of Total |
|------|------|-----------|
| Rotor sandwich (forward) | 1620 us | 41% |
| Rotor sandwich (inverse) | 1534 us | 39% |
| Lloyd-Max quantize | 639 us | 16% |
| Embed/extract | 137 us | 4% |

The fused CUDA kernel eliminated this bottleneck entirely — the full pipeline now takes 6-39 us.

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

### RotorQuant vs TurboQuant on Real KV Cache (8/36 layers, 16 heads)

| Context | Bits | Method | Cosine Sim | Top-1 | Top-5 |
|---------|------|--------|------------|-------|-------|
| 2K | 3-bit | TQ | 0.9906 | 81.2% | 93.8% |
| 2K | 3-bit | **RQ** | 0.9903 | 81.2% | 93.8% |
| 2K | 4-bit | TQ | 0.9911 | 81.2% | 93.8% |
| 2K | 4-bit | **RQ** | 0.9906 | 81.2% | 93.8% |
| 4K | 3-bit | TQ | 0.9875 | 81.2% | 87.5% |
| 4K | 3-bit | **RQ** | 0.9870 | 81.2% | **93.8%** |
| 4K | 4-bit | TQ | 0.9880 | 75.0% | 93.8% |
| 4K | 4-bit | **RQ** | 0.9874 | **81.2%** | 93.8% |

RotorQuant matches or beats TurboQuant on real data despite worse synthetic MSE. The QJL residual correction compensates for a weaker Stage 1, and the Clifford rotor decorrelation better preserves the directional structure of real KV cache vectors.

## CUDA Kernels

### QJL Kernels (from [amirzandieh/QJL](https://github.com/amirzandieh/QJL))

Fused CUDA kernels for 1-bit quantization and attention score computation:

| Kernel | Purpose |
|--------|---------|
| `qjl_quant_kernel.cu` | Fused random projection + sign quantization + outlier separation |
| `qjl_score_kernel.cu` | Fused attention score from 1-bit quantized keys |
| `qjl_gqa_score_kernel.cu` | Grouped Query Attention variant |
| `quantization.cu` | Quantized batched matmul for value reconstruction |

### RotorQuant Fused Kernel

Single CUDA kernel for the full RotorQuant pipeline:

```
embed -> rotor_sandwich -> quantize -> inverse_sandwich -> extract
```

Exploits rotor sparsity (4 of 8 multivector components are zero) to cut FMAs by ~50%. Each thread handles one (batch, group) pair with rotors and codebooks in shared memory.

Build:
```bash
# Build all CUDA kernels
CUDA_HOME=/usr/local/cuda python setup.py build_ext --inplace
```

## Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `test_turboquant.py` | Synthetic validation (codebook, MSE, QJL, needle) | `python -m turboquant.test_turboquant` |
| `validate.py` | Real model validation (Qwen2.5-3B, all layers) | `python -m turboquant.validate` |
| `validate_rotorquant.py` | RotorQuant vs TurboQuant on real model | `python -m turboquant.validate_rotorquant` |
| `benchmark_cuda.py` | PyTorch vs QJL CUDA kernel speed | `python -m turboquant.benchmark_cuda` |
| `benchmark_rotorquant.py` | Full 7-test RotorQuant vs TurboQuant comparison | `python -m turboquant.benchmark_rotorquant` |

## Project Structure

```
turboquant/
  __init__.py                # Package exports
  lloyd_max.py               # Lloyd-Max optimal scalar quantizer solver
  turboquant.py              # TurboQuant: TurboQuantMSE, TurboQuantProd, TurboQuantKVCache
  rotorquant.py              # RotorQuant: RotorQuantMSE, RotorQuantProd, RotorQuantKVCache
  clifford.py                # Cl(3,0) geometric algebra (geometric product, rotors, sandwich)
  compressors.py             # Asymmetric inner product compressors for validation
  cuda_backend.py            # QJL CUDA kernel wrappers with PyTorch fallback
  csrc/
    rotor_fused_kernel.cu    # Fused RotorQuant CUDA kernel
    qjl_quant_kernel.cu      # QJL quantization kernel
    qjl_score_kernel.cu      # QJL attention score kernel
    qjl_gqa_score_kernel.cu  # QJL GQA score kernel
    quantization.cu          # Quantized batched matmul
  test_turboquant.py         # Synthetic tests
  validate.py                # Real model validation
  validate_rotorquant.py     # RotorQuant real model validation
  benchmark_cuda.py          # CUDA kernel benchmarks
  benchmark_rotorquant.py    # RotorQuant vs TurboQuant benchmarks
setup.py                     # pip install with optional CUDA build
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- scipy (for codebook computation)
- transformers, accelerate, bitsandbytes (for real model validation only)

```bash
pip install -e .                    # PyTorch-only
pip install -e ".[validate]"        # + model validation deps
python setup.py build_ext --inplace # Build CUDA kernels
```

## When to Use Which

| Scenario | Recommendation |
|----------|---------------|
| Standard KV cache compression | TurboQuant 3-bit (proven, well-understood) |
| Parameter-constrained (edge/mobile) | RotorQuant (44x fewer params) |
| Maximum throughput | RotorQuant + CUDA kernel (10-19x faster) |
| Geometric data (3D, physics, robotics) | RotorQuant (preserves algebraic structure) |

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization](https://arxiv.org/abs/2406.03482)
- [PolarQuant: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617)
- [QJL Reference Implementation](https://github.com/amirzandieh/QJL)
- [CliffordNet: All You Need is Geometric Algebra](https://arxiv.org/abs/2601.06793) (Jan 2026)

## License

MIT
