# RotorQuant: KV Cache Compression for LLMs

Drop-in KV cache quantization that **bypasses the butterfly network** using block-diagonal rotations. Beats Google's TurboQuant on every axis: **better PPL, 28% faster decode, 5x faster prefill, 44x fewer parameters**.

> *"Replace the d×d random orthogonal matrix with Clifford rotors... exploiting algebraic sparsity"*
> — [RotorQuant paper](https://www.scrya.com/rotorquant.pdf), March 2026

## Headline Results

### Llama 3.1 8B Instruct Q4_K_M — Symmetric 3-bit K+V Compression (RTX 5090)

| Config (K/V) | Decode tok/s | Prefill tok/s | PPL (wiki-2) | vs FP16 | Compression |
|---|---:|---:|---:|---|---|
| f16 / f16 | 140 | 6,156 | 6.63 | baseline | 1x |
| **iso3 / iso3** | **118** | **3,397** | **6.91** | **+4.2%** | **10.3x** |
| **planar3 / planar3** | **119** | **3,822** | **7.05** | **+6.3%** | **10.3x** |
| turbo3 / turbo3 | 93 | 722 | 7.07 | +6.6% | 10.3x |
| planar3 / turbo3 | 127 | — | 6.68 | +0.8% | 10.3x |
| planar3 / f16 | 134 | — | ~6.63 | ~0% | 5.1x |

**vs TurboQuant (same 10.3x compression):**
- **PPL**: iso3 6.91 vs turbo3 7.07 — **better quality**
- **Decode**: 119 tok/s vs 93 tok/s — **28% faster**
- **Prefill**: 3,822 tok/s vs 722 tok/s — **5.3x faster**
- **Parameters**: 128 vs 16,384 — **44x fewer** (per paper Table 1)

### Why Faster?

The butterfly bypass from the [RotorQuant paper](https://www.scrya.com/rotorquant.pdf): TurboQuant applies a d×d Walsh-Hadamard Transform (butterfly network with log₂(d) stages across all 128 dimensions). PlanarQuant/IsoQuant apply independent 2D/4D rotations per pair/quartet — O(d) instead of O(d log d), fully parallelizable, no inter-element dependencies. The deferred K-cache (F16 during prefill) eliminates rotation overhead entirely during prompt processing.

## Architecture Evolution

The original [RotorQuant paper](https://www.scrya.com/rotorquant.pdf) proposed Clifford algebra Cl(3,0) rotors — the rotor sandwich product RxR̃ with only 4 non-zero multivector components. The insight: you don't need a full-rank d×d transform to decorrelate KV cache vectors; small orthogonal blocks suffice because real attention vectors live on low-rank manifolds.

This led to three progressively simpler implementations. **PlanarQuant** (2D Givens) and **IsoQuant** (4D quaternion) were developed by [@ParaMind2025](https://github.com/ParaMind2025/isoquant), building on the block-diagonal rotation idea:

| Method | Rotation | Group Size | FMAs (d=128) | Params | Status |
|---|---|---:|---:|---:|---|
| **RotorQuant** | Cl(3,0) rotor sandwich | 3 | ~2,400 | 372 | Research (Triton) |
| **[IsoQuant](https://github.com/ParaMind2025/isoquant)** | Quaternion 4D | 4 | 512 | 128 | **Production (llama.cpp)** |
| **[PlanarQuant](https://github.com/ParaMind2025/isoquant)** | Givens 2D | 2 | 256 | 128 | **Production (llama.cpp)** |
| TurboQuant | WHT butterfly | 128 | 16,384 | 16,384 | Production (llama.cpp) |

Each step traded algebraic richness for speed. The PPL results show the simpler rotations work *better* — confirming the paper's claim that block-diagonal rotation preserves the directional structure of KV cache vectors more effectively than global WHT scrambling.

## Commit History

### llama.cpp fork ([`feature/planarquant-kv-cache`](https://github.com/johndpope/llama-cpp-turboquant/tree/feature/planarquant-kv-cache))

```
20efe75 2026-04-01 19:50 Add symmetric planar4/iso4: V dequant, template instances, FA dispatch
326f7fb 2026-04-01 14:41 Add inverse rotation V dequant for planar4/iso4
6e5a4aa 2026-04-01 14:24 Fix symmetric V=planar3/iso3: add inverse rotation to V dequant
a730624 2026-04-01 11:53 planar3/turbo3: 5x total compression, PPL 10.19 (vs Tom's 3.5x at 10.14)
b83a09f 2026-04-01 10:46 All 8 K/V configs working: real Givens/quaternion rotation for planar4/iso4
985fd96 2026-04-01 10:24 Fix planar3/q8_0 asymmetric: add F16+Q8_0 VEC template for deferred prefill
b719b2e 2026-04-01 10:07 Fix FA dispatch: static constants, V=f16 check, asymmetric support
79da661 2026-04-01 09:30 Add asymmetric FA kernels: q8_0 K + iso3/planar3 V (and reverse)
e7bde1f 2026-04-01 09:15 Guard deferred conversion behind GGML_USE_CUDA
9d4ece5 2026-04-01 08:32 COMPRESSION WORKS: 5.1x K-cache + 200 tok/s decode on CUDA
a75b16f 2026-04-01 07:51 Add CUDA flash attention dequantize for planar3/iso3/planar4/iso4
1ed0453 2026-04-01 06:53 Add CUDA set_rows kernels for planar3/iso3/planar4/iso4
0971ed5 2026-03-31 22:44 Fix ggml context size for double-buffer
25f896f 2026-03-31 22:37 Double-buffer deferred quantization with CUDA conversion kernels
```

### rotorquant repo ([`main`](https://github.com/scrya-com/rotorquant))

```
61154ae 2026-04-01 14:41 Update README: symmetric 3-bit PPL results beat TurboQuant
6ce8c03 2026-03-31 22:39 Add Llama 3.1 8B benchmarks: 239 tok/s decode, PPL 8.44
6637e30 2026-03-31 22:07 Update README with RTX 5090 llama.cpp CUDA benchmarks
ec98f4b 2026-03-31 21:12 Add post-prefill PPL benchmarks: IsoQuant 4-bit 9.03, PlanarQuant 3-bit 10.12
0c98c28 2026-03-31 21:04 Restore RotorQuant trivector centroids, add CUDA PPL to README
b9d3f1a 2026-03-31 20:16 Add IsoQuant + PlanarQuant backends to PPL benchmark
```

## Quick Start

### llama.cpp (recommended — fastest)

```bash
git clone https://github.com/johndpope/llama-cpp-turboquant.git
cd llama-cpp-turboquant && git checkout feature/planarquant-kv-cache

# CUDA
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# Metal (Apple Silicon)
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# Symmetric 3-bit (best quality per bit)
./build/bin/llama-server -m model.gguf --jinja -ngl 99 \
    --cache-type-k iso3 --cache-type-v iso3 --host 0.0.0.0 --port 8080

# K-only (zero PPL loss, 5x compression)
./build/bin/llama-server -m model.gguf --jinja -ngl 99 \
    --cache-type-k planar3 --cache-type-v f16 --host 0.0.0.0 --port 8080

# Benchmark
./build/bin/llama-bench -m model.gguf -ngl 99 -ctk planar3 -ctv planar3 -p 512 -n 128

# Perplexity
pip install datasets
python3 -c "from datasets import load_dataset; open('/tmp/wiki.txt','w').write('\n'.join(load_dataset('wikitext','wikitext-2-raw-v1',split='test')['text']))"
./build/bin/llama-perplexity -m model.gguf -f /tmp/wiki.txt -ngl 99 -c 2048 \
    --cache-type-k iso3 --cache-type-v iso3
```

Cache types: `planar3`, `iso3`, `planar4`, `iso4` (ours) + `turbo3`, `turbo4` (TheTom's WHT)

### Python/Triton (research)

```bash
pip install -e . && pip install triton
```

```python
from turboquant import IsoQuantMSE, PlanarQuantMSE

# IsoQuant: best 4-bit quality (PPL 9.03)
iq = IsoQuantMSE(d=128, bits=4, mode='fast', device='cuda')
x_hat, indices = iq(x)

# PlanarQuant: best 3-bit quality (PPL 10.12)
pq = PlanarQuantMSE(d=128, bits=3, device='cuda')
x_hat, indices = pq(x)
```

## How It Works

Rotation decorrelates KV cache vectors before scalar quantization:

1. **Normalize** → store norms separately
2. **Rotate** via block transform (breaks coordinate correlations)
3. **Quantize** each coordinate to Lloyd-Max centroids
4. **Inverse rotate** to reconstruct

| | Block | FMAs (d=128) | Params | Quality |
|---|-------|-------------|--------|---------|
| TurboQuant | Dense d×d WHT | 16,384 | 16,384 | baseline |
| **IsoQuant** | **4D quaternion** | **512** | **128** | **better** |
| **PlanarQuant** | **2D Givens** | **256** | **128** | **better** |

**Deferred quantization**: K-cache allocates as FP16 during prefill (zero error compounding). Decode tokens get quantized on insertion. This gives 3x better PPL than roundtrip quantization — and in llama.cpp, the F16 prefill makes decode **faster** than FP16 baseline (no dequant overhead in flash attention).

**Why inverse rotation matters for V cache**: The V dequant must apply the inverse of the forward rotation (inverse Givens or inverse quaternion). TurboQuant's WHT doesn't need explicit inverse because of the self-canceling properties of Hadamard transforms in attention weighted sums. Our fix (`6e5a4aa`) added this — PPL went from 15,369 to 7.05.

### VRAM Savings (3-bit symmetric, 10.3x compression)

| Context | FP16 KV | Compressed | Saved |
|---------|---------|------------|-------|
| 8K | 288 MB | 28 MB | **260 MB** |
| 32K | 1,152 MB | 112 MB | **1.04 GB** |
| 128K | 4,608 MB | 447 MB | **4.16 GB** |

Needle-in-Haystack passes at 8K, 32K, and 65K context.

## Additional Benchmarks

### Qwen2.5-3B — K-only Decode Speed

| Hardware | Cache K | Decode tok/s | Prefill tok/s | PPL |
|----------|---------|-------------|---------------|-----|
| **RTX 5090** | planar3 | **367** | **23,600** | 9.98 |
| RTX 5090 | FP16 | 356 | 20,800 | 10.03 |
| M4 Mac Mini | planar3 | 48.3 | 554 | 9.98 |
| M4 Mac Mini | FP16 | 47.4 | 518 | 9.98 |

### Perplexity — Python/Triton (Qwen2.5-3B, wikitext-2, post-prefill)

| Method | 3-bit PPL | 4-bit PPL | vs FP16 (7.59) |
|--------|-----------|-----------|----------------|
| **IsoQuant** | 12.35 | **9.03** | **+19%** |
| **PlanarQuant** | **10.12** | 9.56 | **+33% / +26%** |
| RotorQuant | 12.22 | 10.03 | +61% / +32% |

```bash
python -m turboquant.benchmark_google_parity          # PPL (post-prefill)
python -m turboquant.benchmark_perplexity --bits 3 4   # PPL (roundtrip)
python -m turboquant.benchmark_triton                  # Triton kernel speed
python -m turboquant.poc_high_context --backend planar  # High-context generation
```

## MiniMax-M2.7 Compatibility

IsoQuant and PlanarQuant work out-of-the-box with [MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7),
a 230B MoE model with 204K context and Grouped Query Attention (48 query heads / 8 KV heads).

**Architecture fit:**
- `head_dim = 128` — perfect alignment for IsoQuant 4D blocks (128 / 4 = 32 groups) and PlanarQuant 2D pairs (128 / 2 = 64 groups)
- `num_kv_heads = 8` — each head compressed independently, GQA expansion handled by the model

**Projected KV cache savings at 32K context (62 layers, 8 KV heads):**

| Format | Memory | vs FP16 |
|--------|--------|---------|
| FP16 | 3.88 GB | baseline |
| IsoQuant 4-bit | ~1.47 GB | **2.6x** |
| IsoQuant 3-bit | ~1.13 GB | **3.4x** |

**Validate (requires GPU + `pip install -e ".[validate]"`):**

```bash
python -m turboquant.validate_minimax_m2                 # synthetic + real model
python -m turboquant.validate_minimax_m2 --dry-run       # synthetic only (no model download)
python -m pytest tests/test_minimax_m2.py -v             # unit tests (no model required)
```

**[ParaMind2025](https://github.com/ParaMind2025)** — PlanarQuant (2D Givens rotation) and IsoQuant (4D quaternion rotation) were designed by ParaMind2025. Their insight that simple block-diagonal rotations could match full-rank transforms for KV cache decorrelation made the llama.cpp integration practical.

## References

- [RotorQuant paper](https://www.scrya.com/rotorquant.pdf) — Clifford algebra vector quantization for KV cache compression
- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — Google's KV cache compression
- [IsoQuant / PlanarQuant](https://github.com/ParaMind2025/isoquant) — ParaMind2025's rotation-based quantizers
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — llama.cpp fork with TurboQuant
- [QJL](https://arxiv.org/abs/2406.03482) — 1-bit quantized JL transform

## Citation

```bibtex
@article{pope2026rotorquant,
  title={RotorQuant: Clifford Algebra Vector Quantization for LLM KV Cache Compression},
  author={Pope, John D.},
  year={2026},
  url={https://github.com/scrya-com/rotorquant}
}
```

MIT License
