"""
Perplexity benchmark: RotorQuant vs FP16 baseline on wikitext-2.

Measures the actual language modeling quality degradation from KV cache
quantization — the standard metric used in TurboQuant, KIVI, KVQuant, etc.

Method: run model forward pass on wikitext-2 test set, compute perplexity.
For RotorQuant: patch DynamicCache to quantize keys post-prefill (same
strategy as poc_high_context.py).

Usage:
    python -m turboquant.benchmark_perplexity
    python -m turboquant.benchmark_perplexity --model Qwen/Qwen2.5-7B-Instruct --bits 2 3 4
"""

import torch
import torch.nn.functional as F
import math
import time
import gc
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_perplexity(model, tokenizer, dataset_text, max_length=2048, stride=512, device="cuda"):
    """Compute perplexity on a text using sliding window.

    Uses the standard approach from Hugging Face perplexity docs:
    slide a window of max_length tokens with overlap of stride.
    """
    encodings = tokenizer(dataset_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.shape[1]

    nlls = []
    n_tokens = 0

    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - begin - 1  # tokens we score

        chunk_ids = input_ids[:, begin:end]

        with torch.no_grad():
            outputs = model(chunk_ids, use_cache=False)
            logits = outputs.logits

        # Only score the non-overlapping part (except first window)
        shift = max(0, max_length - stride) if begin > 0 else 0
        shift_logits = logits[:, shift:-1, :].contiguous()
        shift_labels = chunk_ids[:, shift + 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        n_scored = shift_labels.numel()
        nlls.append(loss.item())
        n_tokens += n_scored

        if end >= seq_len:
            break

    ppl = math.exp(sum(nlls) / n_tokens)
    return ppl, n_tokens


def _make_compressor(backend, bits, device):
    """Create a key compressor function for the given backend.

    Returns a function: compress(keys_tensor, layer_idx) -> quantized_keys
    """
    compressors = {}

    if backend == 'rotorquant':
        from turboquant.rotorquant import RotorQuantMSE
        from turboquant.triton_kernels import triton_rotor_full_fused, pack_rotors_for_triton

        def compress(ks, li):
            D = ks.shape[-1]
            if li not in compressors:
                rq = RotorQuantMSE(D, bits, seed=li * 1000, device=device)
                pk = pack_rotors_for_triton(rq.rotors).to(device)
                compressors[li] = (rq, pk)
            rq, pk = compressors[li]
            flat = ks.reshape(-1, D)
            c_v = getattr(rq, 'centroids_vector', None)
            c_t = getattr(rq, 'centroids_trivector', None)
            if c_t is None:
                c_t = c_v  # fallback: use same centroids for all grades
            kq = triton_rotor_full_fused(flat, pk, None, c_v, None, c_t)
            return kq.to(ks.dtype).reshape(ks.shape)

    elif backend == 'isoquant':
        from turboquant.isoquant import IsoQuantMSE
        from turboquant.triton_isoquant import triton_iso_fast_fused

        def compress(ks, li):
            D = ks.shape[-1]
            if li not in compressors:
                iq = IsoQuantMSE(D, bits, seed=li * 1000, mode='fast', device=device)
                compressors[li] = iq
            iq = compressors[li]
            flat = ks.reshape(-1, D)
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
            flat = ks.reshape(-1, D)
            kq = triton_planar2_fused(flat, pq.rot2, pq.centroids)
            return kq.to(ks.dtype).reshape(ks.shape)

    else:
        raise ValueError(f"Unknown backend: {backend}")

    return compress


def compute_perplexity_with_rq(model, tokenizer, dataset_text, bits=3,
                                max_length=2048, stride=512, device="cuda",
                                backend="rotorquant"):
    """Compute perplexity with quantized KV cache compression.

    Quantizes keys during the forward pass so attention sees quantized keys.
    This measures the actual quality impact of KV cache quantization.

    backend: 'rotorquant', 'isoquant', or 'planarquant'
    """
    from transformers import DynamicCache

    compress = _make_compressor(backend, bits, device)

    _orig = DynamicCache.update

    def _patched(self, ks, vs, li, ck=None):
        # Quantize keys during forward pass — attention sees quantized keys
        kq = compress(ks, li)
        return _orig(self, kq, vs, li, ck)

    encodings = tokenizer(dataset_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.shape[1]

    nlls = []
    n_tokens = 0

    DynamicCache.update = _patched

    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        chunk_ids = input_ids[:, begin:end]

        with torch.no_grad():
            outputs = model(chunk_ids, use_cache=True)
            logits = outputs.logits

        shift = max(0, max_length - stride) if begin > 0 else 0
        shift_logits = logits[:, shift:-1, :].contiguous()
        shift_labels = chunk_ids[:, shift + 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        nlls.append(loss.item())
        n_tokens += shift_labels.numel()

        del outputs
        torch.cuda.empty_cache()

        if end >= seq_len:
            break

    DynamicCache.update = _orig

    ppl = math.exp(sum(nlls) / n_tokens)
    return ppl, n_tokens


def main():
    parser = argparse.ArgumentParser(description="KV Cache Quantization Perplexity Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--backends", type=str, nargs="+",
                        default=["rotorquant", "isoquant", "planarquant"],
                        help="Backends to benchmark")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="Limit dataset tokens (0=full test set)")
    args = parser.parse_args()

    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    import logging; logging.disable(logging.WARNING)

    backend_names = {
        'rotorquant': 'RotorQuant',
        'isoquant': 'IsoQuant',
        'planarquant': 'PlanarQuant',
    }

    print()
    print("=" * 75)
    print("  KV Cache Quantization Perplexity Benchmark (wikitext-2)")
    print(f"  Model: {args.model}")
    print(f"  Bits: {args.bits}")
    print(f"  Backends: {[backend_names.get(b, b) for b in args.backends]}")
    print(f"  Window: {args.max_length}, stride: {args.stride}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print("=" * 75)

    # Load dataset
    print("\nLoading wikitext-2...", flush=True)
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    # Load model
    print("Loading model...", flush=True)
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

    if args.max_tokens > 0:
        tokens = tokenizer(text, return_tensors="pt")
        text = tokenizer.decode(tokens.input_ids[0][:args.max_tokens])

    total_tokens = len(tokenizer(text).input_ids)
    print(f"Dataset: {total_tokens:,} tokens")
    print()

    # FP16 baseline
    print("Computing FP16 baseline perplexity...", flush=True)
    t0 = time.perf_counter()
    ppl_fp16, n_tok = compute_perplexity(
        model, tokenizer, text,
        max_length=args.max_length, stride=args.stride,
    )
    t_fp16 = time.perf_counter() - t0
    print(f"  FP16:     PPL = {ppl_fp16:.2f}  ({n_tok:,} tokens, {t_fp16:.1f}s)")
    print()

    # All backends × all bit widths
    print("  Roundtrip quantization (keys quantized during forward pass):")
    print(f"  {'Method':>20s}  {'PPL':>8s}  {'Delta':>8s}  {'%change':>8s}  {'Time':>8s}")
    print(f"  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    print(f"  {'FP16':>20s}  {ppl_fp16:>8.2f}  {'—':>8s}  {'—':>8s}  {t_fp16:>6.1f}s")

    for bits in args.bits:
        for backend in args.backends:
            torch.cuda.empty_cache()
            gc.collect()

            name = f"{backend_names.get(backend, backend)} {bits}b"

            try:
                t0 = time.perf_counter()
                ppl, n_tok = compute_perplexity_with_rq(
                    model, tokenizer, text, bits=bits,
                    max_length=args.max_length, stride=args.stride,
                    backend=backend,
                )
                t_elapsed = time.perf_counter() - t0

                delta = ppl - ppl_fp16
                pct = (ppl - ppl_fp16) / ppl_fp16 * 100

                print(f"  {name:>20s}  {ppl:>8.2f}  {delta:>+8.2f}  {pct:>+7.1f}%  {t_elapsed:>6.1f}s")
            except Exception as e:
                print(f"  {name:>20s}  {'ERROR':>8s}  {str(e)[:40]}")

    print()
    print("=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == "__main__":
    main()
