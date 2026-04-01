"""
LiteratiQuant PPL + Tokens/Second Benchmark

Measures the actual impact of 1-bit KV cache quantization on:
  1. Perplexity (wikitext-2, autoregressive with post-prefill quantization)
  2. Decode speed (tokens/second)
  3. VRAM savings
  4. Needle-in-haystack (retrieval under extreme compression)

Compares LiteratiQuant (1-bit, 14x compression) against:
  - FP16 baseline
  - IsoQuant 3-bit (4.9x compression)
  - PlanarQuant 3-bit (4.9x compression)

Usage:
    python -m turboquant.benchmark_literatiquant_ppl
    python -m turboquant.benchmark_literatiquant_ppl --model Qwen/Qwen2.5-3B-Instruct
    python -m turboquant.benchmark_literatiquant_ppl --model meta-llama/Llama-3.1-8B-Instruct --group-sizes 64 128
"""

import torch
import torch.nn.functional as F
import math
import time
import gc
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Helpers ────────────────────────────────────────────────────────

def load_model(model_name, device="cuda"):
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    import logging; logging.disable(logging.WARNING)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"),
        device_map="auto", dtype=torch.float16)
    model.eval()
    return model, tokenizer


def make_literati_patcher(group_size=128, device="cuda"):
    """Create DynamicCache patcher for LiteratiQuant KV compression."""
    from transformers import DynamicCache
    from turboquant.literatiquant import quantize_literati, compute_scales_mean_abs

    prefill_done = {}

    def compress(ks):
        D = ks.shape[-1]
        flat = ks.reshape(-1, D).float()
        scales = compute_scales_mean_abs(flat, group_size=group_size)
        kq = quantize_literati(flat, scales, group_size=group_size)
        return kq.to(ks.dtype).reshape(ks.shape)

    _orig = DynamicCache.update

    def _patch(self, ks, vs, li, ck=None):
        ns = ks.shape[2]
        if ns > 1:
            # Prefill: keep FP16 keys (deferred quantization)
            prefill_done[li] = True
            return _orig(self, ks, vs, li, ck)
        # Decode: quantize incoming key
        kq = compress(ks)
        ko, vo = _orig(self, kq, vs, li, ck)
        # Restore the original key for the current token (FP16 for this step)
        ko = ko.clone()
        ko[:, :, -1:, :] = ks
        # First decode after prefill: quantize all prefill keys
        if prefill_done.get(li) is True:
            ko[:, :, :-1, :] = compress(ko[:, :, :-1, :])
            prefill_done[li] = 'done'
        return ko, vo

    return _patch, _orig, prefill_done


def make_rotation_patcher(backend, bits, device="cuda"):
    """Create patcher for rotation-based methods (IsoQuant/PlanarQuant)."""
    from transformers import DynamicCache

    compressors = {}
    prefill_done = {}

    if backend == 'isoquant':
        from turboquant.isoquant import IsoQuantMSE
        from turboquant.triton_isoquant import triton_iso_fast_fused

        def compress(ks, li):
            D = ks.shape[-1]
            if li not in compressors:
                iq = IsoQuantMSE(D, bits, seed=li * 1000, mode='fast', device=device)
                compressors[li] = iq
            iq = compressors[li]
            flat = ks.reshape(-1, D).float()
            kq = triton_iso_fast_fused(flat, iq.q_L, iq.centroids)
            return kq.to(ks.dtype).reshape(ks.shape)

    elif backend == 'planarquant':
        from turboquant.planarquant import PlanarQuantMSE
        from turboquant.triton_planarquant import triton_planar2_fused

        def compress(ks, li):
            D = ks.shape[-1]
            if li not in compressors:
                pq = PlanarQuantMSE(D, bits, seed=li * 1000, device=device)
                compressors[li] = pq
            pq = compressors[li]
            flat = ks.reshape(-1, D).float()
            kq = triton_planar2_fused(flat, pq.rot2, pq.centroids)
            return kq.to(ks.dtype).reshape(ks.shape)

    _orig = DynamicCache.update

    def _patch(self, ks, vs, li, ck=None):
        ns = ks.shape[2]
        if ns > 1:
            prefill_done[li] = True
            return _orig(self, ks, vs, li, ck)
        kq = compress(ks, li)
        ko, vo = _orig(self, kq, vs, li, ck)
        ko = ko.clone()
        ko[:, :, -1:, :] = ks
        if prefill_done.get(li) is True:
            ko[:, :, :-1, :] = compress(ko[:, :, :-1, :], li)
            prefill_done[li] = 'done'
        return ko, vo

    return _patch, _orig, compressors, prefill_done


# ── Test 1: Autoregressive Perplexity ──────────────────────────────

@torch.no_grad()
def test_perplexity(model, tokenizer, n_tokens=512, prefill_len=256,
                    patch_fn=None, orig_fn=None):
    """Autoregressive PPL with post-prefill KV quantization."""
    from transformers import DynamicCache
    from datasets import load_dataset

    text = '\n\n'.join(load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')['text'])
    input_ids = tokenizer(text, return_tensors='pt').input_ids[:, :n_tokens].to('cuda')

    if patch_fn:
        DynamicCache.update = patch_fn

    # Prefill
    context = input_ids[:, :prefill_len]
    out = model(context, use_cache=True)
    cache = out.past_key_values
    logits = out.logits[:, -1:, :]

    # Autoregressive decode
    nlls = []
    t0 = time.perf_counter()
    for i in range(input_ids.shape[1] - prefill_len):
        token = input_ids[:, prefill_len + i:prefill_len + i + 1]
        nll = -F.log_softmax(logits, dim=-1)[0, 0, token[0, 0]].item()
        nlls.append(nll)
        mask = torch.ones(1, prefill_len + i + 1, device='cuda', dtype=torch.long)
        out = model(token, attention_mask=mask, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[:, -1:, :]
    decode_time = time.perf_counter() - t0

    if orig_fn:
        DynamicCache.update = orig_fn

    n_decode = len(nlls)
    ppl = math.exp(sum(nlls) / n_decode)
    tok_s = n_decode / decode_time

    del cache
    torch.cuda.empty_cache()
    gc.collect()

    return ppl, tok_s, n_decode


# ── Test 2: Decode Speed (pure tok/s measurement) ─────────────────

@torch.no_grad()
def test_decode_speed(model, tokenizer, context_len=2048, gen_tokens=128,
                      patch_fn=None, orig_fn=None):
    """Measure pure decode speed with a fixed context."""
    from transformers import DynamicCache

    if patch_fn:
        DynamicCache.update = patch_fn

    # Generate a context prompt
    prompt = "The history of machine learning begins with" + " important" * (context_len // 2)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                       max_length=context_len).to('cuda')

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(
        **inputs, max_new_tokens=gen_tokens,
        do_sample=False, use_cache=True,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    n_gen = out.shape[1] - inputs['input_ids'].shape[1]
    tok_s = n_gen / elapsed

    if orig_fn:
        DynamicCache.update = orig_fn

    torch.cuda.empty_cache()
    gc.collect()

    return tok_s, vram_mb, n_gen


# ── Test 3: NIAH (Needle-in-a-Haystack) ──────────────────────────

NEEDLE = 'The secret project code name is AURORA-7749.'
FILLER = ('The quarterly financial review meeting covered several topics including '
          'budget allocations for the upcoming fiscal year, departmental spending reports, '
          'and projected revenue streams from various business units. The committee discussed '
          'infrastructure upgrades planned for the western regional offices and noted that '
          'maintenance schedules should be coordinated with the facilities management team. '
          'Several action items were assigned to team leads for follow-up before the next '
          'meeting cycle.\n\n')


@torch.no_grad()
def test_niah(model, tokenizer, contexts=[2048, 8192],
              patch_fn=None, orig_fn=None):
    """Needle-in-haystack test at given context lengths."""
    from transformers import DynamicCache

    results = []
    for ctx in contexts:
        if patch_fn:
            DynamicCache.update = patch_fn

        n_reps = max(1, ctx // 110)
        msgs = [{'role': 'user', 'content':
                 FILLER * (n_reps // 3) + '\n--- Memo ---\n' + NEEDLE + '\n--- End ---\n\n'
                 + FILLER * (n_reps - n_reps // 3) + '\nWhat is the secret project code name?'}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                           max_length=ctx + 500).to('cuda')
        input_len = inputs['input_ids'].shape[1]

        try:
            out = model.generate(**inputs, max_new_tokens=40, do_sample=False, use_cache=True)
            text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
            found = 'AURORA-7749' in text
            results.append((input_len, found, text[:80]))
        except torch.cuda.OutOfMemoryError:
            results.append((input_len, None, 'OOM'))

        if orig_fn:
            DynamicCache.update = orig_fn

        torch.cuda.empty_cache()
        gc.collect()

    return results


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LiteratiQuant PPL + Speed Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--ppl-tokens", type=int, default=512,
                        help="Total tokens for PPL eval (prefill + decode)")
    parser.add_argument("--prefill-len", type=int, default=256,
                        help="Prefill context length")
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[128],
                        help="LiteratiQuant group sizes to test")
    parser.add_argument("--context-len", type=int, default=2048,
                        help="Context length for speed test")
    parser.add_argument("--gen-tokens", type=int, default=128,
                        help="Tokens to generate for speed test")
    parser.add_argument("--skip-niah", action="store_true")
    parser.add_argument("--skip-rotation", action="store_true",
                        help="Skip IsoQuant/PlanarQuant comparisons")
    parser.add_argument("--niah-contexts", type=int, nargs="+", default=[2048, 8192])
    args = parser.parse_args()

    print()
    print("=" * 75)
    print("  LiteratiQuant PPL + Tokens/Second Benchmark")
    print(f"  Model: {args.model}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Group sizes: {args.group_sizes}")
    print(f"  PPL tokens: {args.ppl_tokens} (prefill: {args.prefill_len})")
    print(f"  Speed test: ctx={args.context_len}, gen={args.gen_tokens}")
    print("=" * 75)

    model, tokenizer = load_model(args.model)

    # ── PPL Benchmark ──
    print()
    print("─" * 75)
    print("  1. AUTOREGRESSIVE PERPLEXITY (post-prefill KV quantization)")
    print("─" * 75)
    print(f"  {'Method':>25}  {'PPL':>8}  {'Delta':>8}  {'%chg':>7}  {'tok/s':>8}  {'Compress':>10}")
    print(f"  {'─'*25}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*10}")

    # FP16 baseline
    ppl_fp16, tok_s_fp16, n_tok = test_perplexity(
        model, tokenizer, n_tokens=args.ppl_tokens, prefill_len=args.prefill_len)
    print(f"  {'FP16':>25}  {ppl_fp16:>8.2f}  {'—':>8}  {'—':>7}  {tok_s_fp16:>7.1f}  {'1.0x':>10}")

    # LiteratiQuant at various group sizes
    for gs in args.group_sizes:
        bits_per = 1.0 + 16.0 / gs
        compress_ratio = 16.0 / bits_per
        _patch, _orig, _pf = make_literati_patcher(group_size=gs)

        ppl, tok_s, _ = test_perplexity(
            model, tokenizer, n_tokens=args.ppl_tokens, prefill_len=args.prefill_len,
            patch_fn=_patch, orig_fn=_orig)
        _pf.clear()

        delta = ppl - ppl_fp16
        pct = delta / ppl_fp16 * 100
        name = f"LiteratiQuant g{gs}"
        print(f"  {name:>25}  {ppl:>8.2f}  {delta:>+8.2f}  {pct:>+6.1f}%  {tok_s:>7.1f}  {compress_ratio:>9.1f}x")

    # Rotation-based comparisons
    if not args.skip_rotation:
        for backend, bits, label in [('isoquant', 3, 'IsoQuant 3b'), ('planarquant', 3, 'PlanarQuant 3b')]:
            try:
                _patch, _orig, _comps, _pf = make_rotation_patcher(backend, bits)
                ppl, tok_s, _ = test_perplexity(
                    model, tokenizer, n_tokens=args.ppl_tokens, prefill_len=args.prefill_len,
                    patch_fn=_patch, orig_fn=_orig)
                _comps.clear(); _pf.clear()
                delta = ppl - ppl_fp16
                pct = delta / ppl_fp16 * 100
                compress = 16.0 / bits
                print(f"  {label:>25}  {ppl:>8.2f}  {delta:>+8.2f}  {pct:>+6.1f}%  {tok_s:>7.1f}  {compress:>9.1f}x")
            except Exception as e:
                print(f"  {label:>25}  {'ERROR':>8}  {str(e)[:40]}")

    # ── Speed Benchmark ──
    print()
    print("─" * 75)
    print(f"  2. DECODE SPEED (context={args.context_len}, generate={args.gen_tokens} tokens)")
    print("─" * 75)
    print(f"  {'Method':>25}  {'tok/s':>8}  {'VRAM MB':>10}  {'Generated':>10}")
    print(f"  {'─'*25}  {'─'*8}  {'─'*10}  {'─'*10}")

    # FP16
    tok_s, vram, n_gen = test_decode_speed(
        model, tokenizer, context_len=args.context_len, gen_tokens=args.gen_tokens)
    print(f"  {'FP16':>25}  {tok_s:>7.1f}  {vram:>9.0f}  {n_gen:>10}")
    fp16_tok_s = tok_s

    # LiteratiQuant
    for gs in args.group_sizes:
        _patch, _orig, _pf = make_literati_patcher(group_size=gs)
        tok_s, vram, n_gen = test_decode_speed(
            model, tokenizer, context_len=args.context_len, gen_tokens=args.gen_tokens,
            patch_fn=_patch, orig_fn=_orig)
        _pf.clear()
        name = f"LiteratiQuant g{gs}"
        speedup = tok_s / fp16_tok_s if fp16_tok_s > 0 else 0
        print(f"  {name:>25}  {tok_s:>7.1f}  {vram:>9.0f}  {n_gen:>10}  ({speedup:.2f}x)")

    # Rotation
    if not args.skip_rotation:
        for backend, bits, label in [('isoquant', 3, 'IsoQuant 3b'), ('planarquant', 3, 'PlanarQuant 3b')]:
            try:
                _patch, _orig, _comps, _pf = make_rotation_patcher(backend, bits)
                tok_s, vram, n_gen = test_decode_speed(
                    model, tokenizer, context_len=args.context_len, gen_tokens=args.gen_tokens,
                    patch_fn=_patch, orig_fn=_orig)
                _comps.clear(); _pf.clear()
                speedup = tok_s / fp16_tok_s if fp16_tok_s > 0 else 0
                print(f"  {label:>25}  {tok_s:>7.1f}  {vram:>9.0f}  {n_gen:>10}  ({speedup:.2f}x)")
            except Exception as e:
                print(f"  {label:>25}  {'ERROR':>8}  {str(e)[:40]}")

    # ── NIAH ──
    if not args.skip_niah:
        print()
        print("─" * 75)
        print("  3. NEEDLE-IN-A-HAYSTACK")
        print("─" * 75)

        for gs in args.group_sizes:
            _patch, _orig, _pf = make_literati_patcher(group_size=gs)
            name = f"LiteratiQuant g{gs}"
            results = test_niah(model, tokenizer, contexts=args.niah_contexts,
                                patch_fn=_patch, orig_fn=_orig)
            _pf.clear()

            for ctx_len, found, text in results:
                status = 'PASS' if found else ('FAIL' if found is not None else 'OOM')
                print(f"  {name:>25}  ctx={ctx_len:>6}  [{status}]  {text}")

        # FP16 baseline NIAH
        results = test_niah(model, tokenizer, contexts=args.niah_contexts)
        for ctx_len, found, text in results:
            status = 'PASS' if found else ('FAIL' if found is not None else 'OOM')
            print(f"  {'FP16':>25}  ctx={ctx_len:>6}  [{status}]  {text}")

    # ── Summary ──
    print()
    print("=" * 75)
    print("  SUMMARY")
    print("=" * 75)
    print("  LiteratiQuant: 1-bit sign × scale, group-128")
    print(f"  Compression: {16.0 / (1.0 + 16.0/128):.1f}x vs FP16")
    print("  Trade-off: higher PPL degradation vs extreme memory savings")
    print("  Best for: ultra-long context (100K+), edge/mobile deployment")
    print("  Combine with rotation pre-processing for better quality at 1-bit")
    print("=" * 75)


if __name__ == "__main__":
    main()
