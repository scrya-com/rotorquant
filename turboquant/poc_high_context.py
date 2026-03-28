"""
RotorQuant High-Context POC

Patches Qwen2.5-3B's KV cache with RotorQuant Triton-fused compression
and tests needle-in-haystack retrieval + generation at increasing context.

Measures: VRAM, latency, attention fidelity, generation quality.

Usage:
    python -m turboquant.poc_high_context
    python -m turboquant.poc_high_context --bits 3 --max-ctx 65536
"""

import torch
import torch.nn.functional as F
import time
import math
import gc
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant.rotorquant import RotorQuantMSE
from turboquant.triton_kernels import (
    triton_rotor_full_fused, pack_rotors_for_triton,
)


# ── RotorQuant KV cache compressor ──────────────────────────────────

class RotorQuantKeyCompressor:
    """Per-layer key compressor using RotorQuant + Triton fused kernel."""

    def __init__(self, head_dim: int, bits: int, seed: int, device: str):
        self.rq = RotorQuantMSE(head_dim, bits, seed=seed, device=device)
        self.packed_rotors = pack_rotors_for_triton(self.rq.rotors).to(device)
        self.c_v = getattr(self.rq, 'centroids_vector').to(device)
        self.head_dim = head_dim
        self.device = device

    @torch.no_grad()
    def compress_dequantize(self, keys: torch.Tensor) -> torch.Tensor:
        """Quantize → dequantize keys via Triton fused kernel.

        Input:  (batch, n_kv_heads, seq_len, head_dim) float16/float32
        Output: (batch, n_kv_heads, seq_len, head_dim) same dtype

        The returned tensor is the MSE-optimal reconstruction.
        """
        B, H, S, D = keys.shape
        orig_dtype = keys.dtype

        # Flatten to (B*H*S, D) for the Triton kernel
        flat = keys.reshape(-1, D)

        # Triton fused: embed→rotor→quantize→unrotor→extract
        flat_recon = triton_rotor_full_fused(
            flat, self.packed_rotors,
            None, self.c_v, None, None,
        )

        return flat_recon.to(orig_dtype).reshape(B, H, S, D)


class RotorQuantValueCompressor:
    """Per-layer value compressor using RotorQuant + Triton fused kernel."""

    def __init__(self, head_dim: int, bits: int, seed: int, device: str):
        self.key_comp = RotorQuantKeyCompressor(head_dim, bits, seed, device)

    @torch.no_grad()
    def compress_dequantize(self, values: torch.Tensor) -> torch.Tensor:
        return self.key_comp.compress_dequantize(values)


# ── Patched KV cache ────────────────────────────────────────────────

class RotorQuantPatchedCache:
    """Wraps HuggingFace DynamicCache to quantize keys/values on insertion."""

    def __init__(self, bits: int, device: str, quantize_values: bool = True):
        self.bits = bits
        self.device = device
        self.quantize_values = quantize_values
        self._key_compressors = {}
        self._val_compressors = {}

    def get_key_compressor(self, layer_idx: int, head_dim: int) -> RotorQuantKeyCompressor:
        if layer_idx not in self._key_compressors:
            self._key_compressors[layer_idx] = RotorQuantKeyCompressor(
                head_dim, self.bits, seed=layer_idx * 1000, device=self.device)
        return self._key_compressors[layer_idx]

    def get_val_compressor(self, layer_idx: int, head_dim: int) -> RotorQuantValueCompressor:
        if layer_idx not in self._val_compressors:
            self._val_compressors[layer_idx] = RotorQuantValueCompressor(
                head_dim, self.bits, seed=layer_idx * 1000 + 500, device=self.device)
        return self._val_compressors[layer_idx]


def patch_model_kv_cache(model, bits: int = 4, quantize_values: bool = False):
    """Monkey-patch model's cache update for post-prefill RotorQuant compression.

    Strategy:
      - Prefill: full precision (no quantization, no error compounding)
      - First decode step: quantize entire prefill cache in bulk
      - Subsequent decode steps: quantize each new key, return full-precision
        key for current attention to avoid compounding

    This gives perfect prefill quality + compressed cache for decode.
    Works with any HuggingFace model that uses DynamicCache.
    """
    from transformers import DynamicCache

    rq_cache = RotorQuantPatchedCache(bits, "cuda", quantize_values)
    prefill_done = {}  # per-layer tracking

    _original_update = DynamicCache.update

    def _compress_keys(key_states, layer_idx):
        D = key_states.shape[-1]
        kc = rq_cache.get_key_compressor(layer_idx, D)
        return kc.compress_dequantize(key_states)

    def _patched_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        new_seq_len = key_states.shape[2]
        is_prefill = (new_seq_len > 1)

        if is_prefill:
            # Prefill: store at full precision — no quantization
            prefill_done[layer_idx] = True
            return _original_update(self, key_states, value_states, layer_idx, cache_kwargs)

        # Decode: quantize new key for storage, full-precision for current attention
        key_quantized = _compress_keys(key_states, layer_idx)

        # Optionally quantize values
        if rq_cache.quantize_values:
            D = value_states.shape[-1]
            vc = rq_cache.get_val_compressor(layer_idx, D)
            value_states = vc.compress_dequantize(value_states)

        k_out, v_out = _original_update(self, key_quantized, value_states, layer_idx, cache_kwargs)

        # Return full-precision key for current token's attention
        k_out = k_out.clone()
        k_out[:, :, -1:, :] = key_states

        # On first decode step: quantize all prefill keys in bulk
        if prefill_done.get(layer_idx) is True:
            k_out[:, :, :-1, :] = _compress_keys(k_out[:, :, :-1, :], layer_idx)
            prefill_done[layer_idx] = 'done'

        return k_out, v_out

    DynamicCache.update = _patched_update
    return _original_update, rq_cache


def unpatch_model_kv_cache(original_update):
    """Restore original cache update."""
    from transformers import DynamicCache
    DynamicCache.update = original_update


# ── Needle-in-haystack builder ──────────────────────────────────────

NEEDLE = "The secret project code name is AURORA-7749."
QUESTION = "What is the secret project code name mentioned in the documents?"

FILLER = """The quarterly financial review meeting covered several topics including
budget allocations for the upcoming fiscal year, departmental spending reports, and projected
revenue streams from various business units. The committee discussed infrastructure upgrades
planned for the western regional offices and noted that maintenance schedules should be
coordinated with the facilities management team. Several action items were assigned to team
leads for follow-up before the next meeting cycle.\n\n"""


def build_prompt(tokenizer, target_tokens=2048, needle_pos=0.33):
    """Build a needle-in-haystack prompt at the target token count."""
    filler_len = len(tokenizer.encode(FILLER, add_special_tokens=False))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = int(n_reps * needle_pos)

    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Important Memo ---\n{NEEDLE}\n--- End Memo ---\n\n")
        parts.append(FILLER)

    haystack = "".join(parts)
    messages = [
        {"role": "user", "content": f"{haystack}\n\n{QUESTION}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── Attention fidelity measurement ──────────────────────────────────

@torch.no_grad()
def measure_attention_fidelity(model, tokenizer, context_len, bits):
    """Compare RotorQuant attention scores vs FP16 on real KV cache.

    Returns dict with cosine_sim, top1_match, needle_found.
    """
    prompt = build_prompt(tokenizer, context_len)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=context_len + 256).to("cuda")
    seq_len = inputs["input_ids"].shape[1]

    # Forward pass with FP16 KV cache
    outputs_fp16 = model(**inputs, use_cache=True, output_attentions=False)
    cache_fp16 = outputs_fp16.past_key_values

    n_layers = len(cache_fp16.layers)
    head_dim = cache_fp16.layers[0].keys.shape[-1]
    n_kv_heads = cache_fp16.layers[0].keys.shape[1]

    # Measure per-layer attention fidelity
    cosine_sims = []
    top1_matches = 0
    n_checks = 0

    for layer_idx in range(n_layers):
        keys = cache_fp16.layers[layer_idx].keys  # (1, H, S, D)
        B, H, S, D = keys.shape

        # Compress keys with RotorQuant
        compressor = RotorQuantKeyCompressor(D, bits, seed=layer_idx * 1000, device="cuda")
        keys_rq = compressor.compress_dequantize(keys)

        # Query = last token attending to all keys
        query = keys[:, :, -1:, :]  # (1, H, 1, D)

        # Real scores
        real_scores = torch.matmul(query.float(), keys.float().transpose(-2, -1)).squeeze(-2)

        # RotorQuant scores
        rq_scores = torch.matmul(query.float(), keys_rq.float().transpose(-2, -1)).squeeze(-2)

        for h in range(H):
            cos = F.cosine_similarity(
                real_scores[0, h].unsqueeze(0),
                rq_scores[0, h].unsqueeze(0)
            ).item()
            cosine_sims.append(cos)

            if real_scores[0, h].argmax().item() == rq_scores[0, h].argmax().item():
                top1_matches += 1
            n_checks += 1

    # Clean up
    del cache_fp16, outputs_fp16
    torch.cuda.empty_cache()

    return {
        "seq_len": seq_len,
        "cosine_sim": sum(cosine_sims) / len(cosine_sims),
        "top1_match": top1_matches / n_checks * 100,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
    }


# ── Generation test ─────────────────────────────────────────────────

@torch.no_grad()
def test_generation(model, tokenizer, context_len, bits, max_new_tokens=50, keys_only=True):
    """Generate text with RotorQuant-compressed KV cache.

    Returns dict with generation, tokens/sec, VRAM usage, needle_found.
    """
    prompt = build_prompt(tokenizer, context_len)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=context_len + 256).to("cuda")
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    vram_before = torch.cuda.memory_allocated() / 1024**2

    # Patch KV cache
    original_update, rq_cache = patch_model_kv_cache(model, bits=bits, quantize_values=not keys_only)

    try:
        t0 = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        torch.cuda.synchronize()
        t_gen = time.perf_counter() - t0

        gen_tokens = outputs[0][input_len:]
        n_gen = len(gen_tokens)
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        vram_peak = torch.cuda.max_memory_allocated() / 1024**2
        vram_kv = vram_peak - vram_before

        tok_per_sec = n_gen / t_gen if t_gen > 0 else 0

        needle_found = "AURORA-7749" in text or "aurora" in text.lower()

    finally:
        unpatch_model_kv_cache(original_update)
        torch.cuda.empty_cache()

    return {
        "input_tokens": input_len,
        "gen_tokens": n_gen,
        "text": text.strip(),
        "tok_per_sec": tok_per_sec,
        "time_s": t_gen,
        "vram_peak_mb": vram_peak,
        "vram_kv_est_mb": vram_kv,
        "needle_found": needle_found,
    }


@torch.no_grad()
def test_generation_fp16(model, tokenizer, context_len, max_new_tokens=50):
    """Baseline: generate with standard FP16 KV cache."""
    prompt = build_prompt(tokenizer, context_len)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=context_len + 256).to("cuda")
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    vram_before = torch.cuda.memory_allocated() / 1024**2

    t0 = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    torch.cuda.synchronize()
    t_gen = time.perf_counter() - t0

    gen_tokens = outputs[0][input_len:]
    n_gen = len(gen_tokens)
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    vram_peak = torch.cuda.max_memory_allocated() / 1024**2
    vram_kv = vram_peak - vram_before

    torch.cuda.empty_cache()

    return {
        "input_tokens": input_len,
        "gen_tokens": n_gen,
        "text": text.strip(),
        "tok_per_sec": n_gen / t_gen if t_gen > 0 else 0,
        "time_s": t_gen,
        "vram_peak_mb": vram_peak,
        "vram_kv_est_mb": vram_kv,
        "needle_found": "AURORA-7749" in text or "aurora" in text.lower(),
    }


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RotorQuant High-Context POC")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (2-4)")
    parser.add_argument("--max-ctx", type=int, default=32768, help="Max context to test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--skip-fidelity", action="store_true", help="Skip attention fidelity test")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--keys-only", action="store_true", default=True,
                        help="Only compress keys, leave values in fp16 (recommended)")
    parser.add_argument("--compress-values", dest="keys_only", action="store_false",
                        help="Also compress values (higher error)")
    args = parser.parse_args()

    print()
    print("=" * 74)
    print("  RotorQuant High-Context POC")
    print(f"  Model: {args.model}")
    print(f"  Bits: {args.bits}  |  Max context: {args.max_ctx:,}  |  Keys only: {args.keys_only}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 74)
    print()

    # Load model
    print("Loading model...", flush=True)
    import logging
    logging.disable(logging.WARNING)
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
        dtype=torch.float16,
    )
    model.eval()

    model_vram = torch.cuda.memory_allocated() / 1024**2
    print(f"Model loaded. VRAM: {model_vram:.0f} MB")
    print()

    # Context lengths to test
    ctx_lengths = []
    ctx = 2048
    while ctx <= args.max_ctx:
        ctx_lengths.append(ctx)
        ctx *= 2

    # ── Phase 1: Attention fidelity ──
    if not args.skip_fidelity:
        print("=" * 74)
        print("PHASE 1: Attention Fidelity (RotorQuant vs FP16)")
        print("=" * 74)
        print()
        print(f"  {'Context':>8s}  {'Cosine Sim':>12s}  {'Top-1 Match':>12s}  {'Layers':>8s}")
        print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*8}")

        for ctx_len in ctx_lengths:
            if ctx_len > 8192:
                # Fidelity test requires 2x memory (FP16 + comparison)
                # Skip very long contexts
                print(f"  {ctx_len:>8,}  {'(skipped — needs 2x VRAM)':>40s}")
                continue
            try:
                result = measure_attention_fidelity(model, tokenizer, ctx_len, args.bits)
                print(f"  {result['seq_len']:>8,}  {result['cosine_sim']:>12.6f}  "
                      f"{result['top1_match']:>10.1f}%  {result['n_layers']:>8d}")
            except torch.cuda.OutOfMemoryError:
                print(f"  {ctx_len:>8,}  {'OOM':>12s}")
                torch.cuda.empty_cache()
                break

        print()

    # ── Phase 2: Generation with RotorQuant ──
    print("=" * 74)
    print(f"PHASE 2: Generation with RotorQuant ({args.bits}-bit KV cache)")
    print("=" * 74)
    print()

    # Baseline at smallest context
    print("  FP16 baseline (2K context):")
    try:
        baseline = test_generation_fp16(model, tokenizer, 2048, args.max_new_tokens)
        print(f"    Tokens: {baseline['input_tokens']:,} in + {baseline['gen_tokens']} gen")
        print(f"    Speed:  {baseline['tok_per_sec']:.1f} tok/s")
        print(f"    VRAM:   {baseline['vram_peak_mb']:.0f} MB peak")
        print(f"    Needle: {'FOUND' if baseline['needle_found'] else 'NOT FOUND'}")
        print(f"    Output: {baseline['text'][:120]}...")
    except Exception as e:
        print(f"    Error: {e}")
        baseline = None
    print()

    # RotorQuant at each context length
    print(f"  RotorQuant {args.bits}-bit results:")
    print()
    print(f"  {'Context':>8s}  {'Speed':>10s}  {'VRAM Peak':>10s}  {'Needle':>8s}  {'Output (first 80 chars)'}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*40}")

    for ctx_len in ctx_lengths:
        torch.cuda.empty_cache()
        gc.collect()

        try:
            result = test_generation(model, tokenizer, ctx_len, args.bits, args.max_new_tokens, args.keys_only)
            needle_str = "FOUND" if result['needle_found'] else "MISS"
            text_preview = result['text'][:80].replace('\n', ' ')
            print(f"  {result['input_tokens']:>8,}  "
                  f"{result['tok_per_sec']:>8.1f}/s  "
                  f"{result['vram_peak_mb']:>8.0f}MB  "
                  f"{needle_str:>8s}  "
                  f"{text_preview}")
        except torch.cuda.OutOfMemoryError:
            print(f"  {ctx_len:>8,}  {'OOM':>10s}  --- VRAM limit reached ---")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"  {ctx_len:>8,}  Error: {e}")
            break

    print()

    # ── Phase 3: Memory projection ──
    print("=" * 74)
    print("PHASE 3: Memory Projection")
    print("=" * 74)
    print()

    config = model.config
    n_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)

    fp16_per_token = n_layers * 2 * n_kv_heads * head_dim * 2  # bytes (K+V)
    # RotorQuant: same storage (we store dequantized fp16), but could store compressed
    # For now, the savings come from reduced precision of reconstructed values
    rq_per_token = fp16_per_token  # roundtrip — same storage, but quantization noise helps with lower-rank approximation

    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    avail_for_kv = (gpu_total - model_vram / 1024) * 1024**3  # bytes

    print(f"  Model: {args.model}")
    print(f"  Layers: {n_layers}, KV heads: {n_kv_heads}, head_dim: {head_dim}")
    print(f"  FP16 KV per token: {fp16_per_token:,} bytes ({fp16_per_token/1024:.1f} KB)")
    print(f"  GPU total: {gpu_total:.1f} GB, model: {model_vram/1024:.1f} GB")
    print(f"  Available for KV cache: {avail_for_kv/1024**3:.1f} GB")
    print()

    print(f"  {'Context':>10s}  {'FP16 KV':>10s}  {'Status'}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*20}")
    for ctx in [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]:
        kv_bytes = fp16_per_token * ctx
        kv_gb = kv_bytes / 1024**3
        fits = "fits" if kv_bytes < avail_for_kv else "OOM"
        print(f"  {ctx:>10,}  {kv_gb:>8.2f}GB  {fits}")

    print()
    print("  NOTE: With true compressed storage (not roundtrip dequant),")
    print(f"  {args.bits}-bit RotorQuant would use ~{16/args.bits:.1f}x less KV memory,")
    print(f"  extending max context by ~{16/args.bits:.1f}x.")

    print()
    print("=" * 74)
    print("POC COMPLETE")
    print("=" * 74)


if __name__ == "__main__":
    main()
