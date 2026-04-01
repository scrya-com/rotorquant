# RotorQuant Project

## Status (2026-04-01)

### llama.cpp CUDA Integration — WORKING
- Branch: `feature/planarquant-kv-cache` on `johndpope/llama-cpp-turboquant`
- All 3-bit symmetric configs working: planar3/planar3, iso3/iso3
- All asymmetric configs working: planar3/turbo3, planar3/q8_0, iso3/q8_0, etc.
- Deferred K-cache quantization: F16 during prefill, convert post-prefill
- 4-bit symmetric (planar4/planar4, iso4/iso4): V dequant code written, FA dispatch crash during prefill (missing F16+planar4 template linking issue)

### PPL Results — Llama 3.1 8B Instruct Q4_K_M, WikiText-2, ctx=2048
| Config (K/V) | PPL | vs FP16 (6.63) |
|---|---|---|
| f16 / f16 | 6.63 | baseline |
| **iso3 / iso3** | **6.91** | +4.2% |
| **planar3 / planar3** | **7.05** | +6.3% |
| turbo3 / turbo3 | 7.07 | +6.6% |
| planar3 / turbo3 | 6.68 | +0.8% |

**Both iso3 and planar3 beat TurboQuant** at the same 10.3x compression ratio.

### Key Fix (2026-04-01): V Dequant Inverse Rotation
- V dequant must apply inverse Givens (planar3) or inverse quaternion (iso3) rotation
- TurboQuant WHT doesn't need explicit inverse due to self-canceling Hadamard properties
- Fix commit: `6e5a4aa` — PPL went from 15,369 → 7.05
- Root cause: `extern __constant__` cross-TU references were fragile + missing inverse rotation

## Architecture
- `turboquant/isoquant.py` — **IsoQuantMSE** (recommended): quaternion 4D block rotation
- `turboquant/planarquant.py` — **PlanarQuantMSE**: Givens 2D rotation
- `turboquant/rotorquant.py` — RotorQuantMSE (legacy): Clifford algebra rotor sandwich
- `turboquant/triton_kernels.py` — 5 Triton kernels (rotor forward/inverse, fused pipeline, attention)
- `turboquant/triton_planarquant.py` — Triton fused kernels for PlanarQuant
- `turboquant/fused_attention.py` — Fused attention with QJL two-term estimator
- `turboquant/fused_planar_attention.py` — Fused quantize+attention Triton kernel

## llama.cpp Key Files (in /tmp/llama-cpp-cuda/)
- `ggml/src/ggml-cuda/planar-iso-constants.cuh` — Static __constant__ rotation params (Givens cos/sin, quaternion qw/qx/qy/qz, centroids). Per-TU copies, no cross-module extern needed.
- `ggml/src/ggml-cuda/set-rows-planar-iso.cuh` — Device quantize functions for set_rows (V cache path)
- `ggml/src/ggml-cuda/cpy-planar-iso.cu` — Bulk F16→quantized CUDA kernels for ggml_cpy (K cache deferred conversion path)
- `ggml/src/ggml-cuda/fattn-common.cuh` — FA vec_dot_KQ (K dequant + Q dot) and dequantize_V (V dequant with inverse rotation)
- `ggml/src/ggml-cuda/dequantize.cuh` — K dequantize for non-FA paths
- `ggml/src/ggml-cuda/fattn.cu` — FA kernel dispatch (FATTN_VEC_CASES_ALL_D macros)
- `ggml/src/ggml-cuda/CMakeLists.txt` — Template instance file list (NOT glob — must add new .cu files explicitly)
- `src/llama-kv-cache.cpp` — Double-buffer allocation, deferred conversion, V zero-padding

## Default Usage
IsoQuant symmetric (iso3/iso3) is the recommended default for max compression.
PlanarQuant K-only (planar3/f16) for zero PPL loss with 5.1x compression.

## TODOs
- [ ] Fix 4-bit symmetric dispatch crash (planar4/planar4, iso4/iso4) — FA template linking issue during deferred prefill (K=F16, V=planar4)
- [ ] Run decode speed benchmarks for symmetric 3-bit configs (llama-bench)
- [ ] Port symmetric V dequant fix to Metal backend (Mac M4)
- [ ] Reply to TheTom on PR #34 with CUDA benchmark results
- [ ] Add NIAH test for symmetric configs at 8K/32K/65K context
- [ ] Investigate why turbo3 V works without inverse WHT (mathematical analysis for paper)
