"""
MiniMax-M2.7 KV Cache Compression Validation.

Demonstrates IsoQuant / PlanarQuant KV cache compression on
MiniMax-M2.7 (MiniMaxAI/MiniMax-M2.7), a 230B MoE model with:
  - head_dim = 128  (4D quaternion blocks align perfectly)
  - num_kv_heads = 8  (GQA: 48 query heads, 8 KV heads)
  - num_layers = 62
  - max_context = 204800 tokens

Requirements:
    pip install -e ".[validate]"
    # MiniMax-M2.7 requires trust_remote_code=True and ~48 GB GPU VRAM
    # (with 4-bit bitsandbytes quantization).

Usage:
    python turboquant/validate_minimax_m2.py
    python turboquant/validate_minimax_m2.py --dry-run   # synthetic benchmark only
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant.isoquant import IsoQuantMSE
from turboquant.planarquant import PlanarQuantMSE
from turboquant.compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE

# ── MiniMax-M2.7 architecture constants ─────────────────────────────
MODEL_NAME = "MiniMaxAI/MiniMax-M2.7"
HEAD_DIM = 128
NUM_KV_HEADS = 8
NUM_ATTN_HEADS = 48
NUM_LAYERS = 62
MAX_CONTEXT = 204_800

FILLER = (
    "The quarterly financial review meeting covered several topics including "
    "budget allocations for the upcoming fiscal year, departmental spending "
    "reports, and projected revenue streams from various business units. "
    "The committee discussed infrastructure upgrades planned for the western "
    "regional offices and noted that maintenance schedules should be coordinated "
    "with the facilities management team.\n\n"
)


# ── Compressor helpers ────────────────────────────────────────────────

def _compress_and_score_iso(keys: torch.Tensor, query: torch.Tensor,
                             bits: int, layer_idx: int) -> dict:
    """Compress keys with IsoQuant and evaluate attention score fidelity."""
    B, H, S, D = keys.shape
    cosine_sims, top1, top5, n = [], 0, 0, 0

    quantizer = IsoQuantMSE(D, bits=bits, seed=layer_idx * 1000, device=str(keys.device))

    for h in range(H):
        k = keys[0, h].float()      # (S, D)
        q = query[0, h, 0].float()  # (D,)

        real_scores = k @ q  # (S,)

        k_norm = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        q_hat, _ = quantizer(k_norm)
        est_scores = q_hat @ q  # (S,) — dequant dot product

        cos = F.cosine_similarity(real_scores.unsqueeze(0),
                                   est_scores.unsqueeze(0)).item()
        cosine_sims.append(cos)

        real_top1 = real_scores.argmax().item()
        if real_top1 == est_scores.argmax().item():
            top1 += 1
        if real_top1 in est_scores.topk(5).indices.tolist():
            top5 += 1
        n += 1

    return {"cosine_sims": cosine_sims, "top1": top1, "top5": top5, "n": n}


def _compress_and_score_turbo(keys: torch.Tensor, query: torch.Tensor,
                               bits: int, layer_idx: int) -> dict:
    """Compress keys with TurboQuant asymmetric estimator."""
    B, H, S, D = keys.shape
    cosine_sims, top1, top5, n = [], 0, 0, 0

    key_comp = TurboQuantCompressorV2(D, bits, seed=layer_idx * 1000,
                                       device=str(keys.device))

    for h in range(H):
        k = keys[0, h].float()           # (S, D)
        q = query[0, h].float()          # (1, D)

        real_scores = (q @ k.T).squeeze(0)  # (S,)
        compressed_k = key_comp.compress(k.unsqueeze(0))  # add batch dim
        est_scores = key_comp.asymmetric_attention_scores(
            q.unsqueeze(0), compressed_k
        ).squeeze()  # (S,)

        cos = F.cosine_similarity(real_scores.unsqueeze(0),
                                   est_scores.unsqueeze(0)).item()
        cosine_sims.append(cos)

        real_top1 = real_scores.argmax().item()
        if real_top1 == est_scores.argmax().item():
            top1 += 1
        if real_top1 in est_scores.topk(5).indices.tolist():
            top5 += 1
        n += 1

    return {"cosine_sims": cosine_sims, "top1": top1, "top5": top5, "n": n}


# ── Dry-run: synthetic benchmark ──────────────────────────────────────

def run_synthetic_benchmark(seq_len: int = 2048, n_layers: int = 8):
    """
    Benchmark on synthetic KV tensors with MiniMax-M2.7 dimensions.

    No model loading required — useful for CI / quick sanity checks.
    """
    print(f"\n{'─' * 60}")
    print(f"Synthetic benchmark: seq={seq_len}, layers={n_layers}")
    print(f"Architecture: head_dim={HEAD_DIM}, kv_heads={NUM_KV_HEADS}")
    print(f"{'─' * 60}")

    torch.manual_seed(0)
    device = "cpu"

    for bits in [3, 4]:
        all_iso_cos, all_turbo_cos = [], []
        iso_top1 = turbo_top1 = iso_n = turbo_n = 0

        t0 = time.perf_counter()
        for layer_idx in range(n_layers):
            # Synthetic KV cache: (batch=1, kv_heads, seq, head_dim)
            keys = torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM, device=device)
            keys = keys / keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            query = keys[:, :, -1:, :]  # last token as query

            r_iso = _compress_and_score_iso(keys, query, bits, layer_idx)
            r_tq = _compress_and_score_turbo(keys, query, bits, layer_idx)

            all_iso_cos.extend(r_iso["cosine_sims"])
            all_turbo_cos.extend(r_tq["cosine_sims"])
            iso_top1 += r_iso["top1"];   iso_n += r_iso["n"]
            turbo_top1 += r_tq["top1"]; turbo_n += r_tq["n"]

        elapsed = time.perf_counter() - t0

        # Compression ratio: head_dim=128 @ FP32 vs quantized
        uncompressed_bits = 1 * NUM_KV_HEADS * seq_len * HEAD_DIM * 32  # fp32
        # IsoQuant: bits per dim + tiny norm overhead
        iso_bits = 1 * NUM_KV_HEADS * seq_len * HEAD_DIM * bits
        ratio = uncompressed_bits / iso_bits

        print(f"\n  {bits}-bit (vs fp32 {ratio:.1f}x compression)  [{elapsed:.2f}s]")
        print(f"    IsoQuant   cosine sim: {sum(all_iso_cos)/len(all_iso_cos):.4f}  "
              f"top1: {100*iso_top1/iso_n:.1f}%")
        print(f"    TurboQuant cosine sim: {sum(all_turbo_cos)/len(all_turbo_cos):.4f}  "
              f"top1: {100*turbo_top1/turbo_n:.1f}%")


# ── Full validation with real MiniMax-M2.7 model ──────────────────────

def run_model_validation(target_tokens: int = 2048):
    """
    Run KV cache compression validation on a real MiniMax-M2.7 forward pass.

    Loads the model with 4-bit quantization (bitsandbytes NF4) to fit in
    available GPU VRAM.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install -e '.[validate]'")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required for MiniMax-M2.7 model validation.")
        sys.exit(1)

    print(f"\nLoading {MODEL_NAME} with 4-bit NF4 quantization…")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    gpu_mb = torch.cuda.memory_allocated() // 1024 // 1024
    print(f"Loaded. GPU: {gpu_mb} MB\n")

    filler_tokens = len(tokenizer.encode(FILLER))
    n_reps = max(1, target_tokens // filler_tokens)
    prompt = FILLER * n_reps

    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=target_tokens + 256,
    ).to("cuda")
    seq_len = inputs["input_ids"].shape[1]
    print(f"Context: {seq_len} tokens\n")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)
    cache = outputs.past_key_values

    # Handle both DynamicCache and legacy tuple-of-tuples
    if hasattr(cache, "key_cache"):
        # transformers >= 4.44: DynamicCache
        cache_layers = [(cache.key_cache[i], cache.value_cache[i])
                        for i in range(len(cache.key_cache))]
    elif hasattr(cache, "layers"):
        cache_layers = [(cache.layers[i].keys, cache.layers[i].values)
                        for i in range(len(cache.layers))]
    else:
        # Legacy tuple-of-tuples: ((k0, v0), (k1, v1), …)
        cache_layers = list(cache)

    n_layers = len(cache_layers)
    print(f"Cache: {n_layers} layers  kv_heads={cache_layers[0][0].shape[1]}")
    print()

    for bits in [3, 4]:
        all_cos, top1_hits, top5_hits, n_checks = [], 0, 0, 0
        t0 = time.perf_counter()

        for layer_idx in range(min(n_layers, 16)):  # first 16 layers for speed
            keys, _ = cache_layers[layer_idx]
            keys = keys.float()
            B, H, S, D = keys.shape
            query = keys[:, :, -1:, :]

            r = _compress_and_score_iso(keys, query, bits, layer_idx)
            all_cos.extend(r["cosine_sims"])
            top1_hits += r["top1"]
            top5_hits += r["top5"]
            n_checks += r["n"]

        elapsed = time.perf_counter() - t0
        avg_cos = sum(all_cos) / len(all_cos)

        print(f"  IsoQuant {bits}-bit  [16/{n_layers} layers, {elapsed:.1f}s]")
        print(f"    Score cosine sim:  {avg_cos:.6f}")
        print(f"    Top-1 match:       {100*top1_hits/n_checks:.1f}%  ({top1_hits}/{n_checks})")
        print(f"    Top-5 match:       {100*top5_hits/n_checks:.1f}%  ({top5_hits}/{n_checks})")
        print()


# ── Memory estimate ───────────────────────────────────────────────────

def print_memory_estimate(seq_len: int = 32768):
    """Print KV cache memory breakdown for MiniMax-M2.7 at various compressions."""
    print(f"\n{'─' * 60}")
    print(f"KV Cache Memory: MiniMax-M2.7, seq={seq_len:,}")
    print(f"  (kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}, layers={NUM_LAYERS})")
    print(f"{'─' * 60}")

    n_vecs = NUM_LAYERS * NUM_KV_HEADS * seq_len  # K + V separately
    fp16_bytes = n_vecs * 2 * HEAD_DIM * 2  # K+V, fp16
    print(f"  FP16 baseline:   {fp16_bytes / 1024**3:.2f} GB")

    for bits in [1, 2, 3, 4]:
        # IsoQuant: bits/elem + norm (fp16 per vector)
        quant_bits = n_vecs * 2 * HEAD_DIM * bits
        norm_bits = n_vecs * 2 * 16  # fp16 norm per vector
        total_bytes = (quant_bits + norm_bits) / 8
        ratio = fp16_bytes / total_bytes
        print(f"  IsoQuant {bits}-bit:  {total_bytes / 1024**3:.2f} GB  ({ratio:.1f}x)")


# ── Entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate IsoQuant KV cache compression on MiniMax-M2.7"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run synthetic benchmark only (no model loading)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=2048,
        help="Target context length for model validation (default: 2048)"
    )
    parser.add_argument(
        "--layers", type=int, default=8,
        help="Number of layers for synthetic benchmark (default: 8)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RotorQuant × MiniMax-M2.7 KV Cache Compression")
    print("=" * 60)
    print(f"Model architecture: {HEAD_DIM}D head, {NUM_KV_HEADS} KV heads, "
          f"{NUM_LAYERS} layers, GQA")

    print_memory_estimate(seq_len=32768)

    run_synthetic_benchmark(seq_len=2048, n_layers=args.layers)

    if not args.dry_run:
        run_model_validation(target_tokens=args.seq_len)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
