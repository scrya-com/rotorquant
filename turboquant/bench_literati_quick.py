"""
Quick PPL comparison: LiteratiQuant modes vs IsoQuant baselines.
Tests 1-bit, 2-bit, and IsoQuant 2/3-bit on Qwen2.5-3B.
"""

import torch
import torch.nn.functional as F
import math
import time
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
import logging; logging.disable(logging.WARNING)


def load_model(model_name="Qwen/Qwen2.5-3B-Instruct"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"),
        device_map="auto", dtype=torch.float16)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def eval_ppl(model, tokenizer, input_ids, prefill_len, patch_fn=None, orig_fn=None):
    from transformers import DynamicCache
    if patch_fn:
        DynamicCache.update = patch_fn

    context = input_ids[:, :prefill_len]
    out = model(context, use_cache=True)
    cache = out.past_key_values
    logits = out.logits[:, -1:, :]

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
    elapsed = time.perf_counter() - t0

    if orig_fn:
        DynamicCache.update = orig_fn

    n = len(nlls)
    ppl = math.exp(sum(nlls) / n)
    tok_s = n / elapsed

    del cache; torch.cuda.empty_cache(); gc.collect()
    return ppl, tok_s, n


def make_patcher(mode, group_size=128, bits=3):
    """Create DynamicCache patcher for various quantization backends."""
    from transformers import DynamicCache

    prefill_done = {}
    compressors = {}

    if mode == 'literati-1bit':
        from turboquant.literatiquant import quantize_literati, compute_scales_mean_abs
        def compress(ks, li=None):
            D = ks.shape[-1]
            flat = ks.reshape(-1, D).float()
            scales = compute_scales_mean_abs(flat, group_size)
            kq = quantize_literati(flat, scales, group_size)
            return kq.to(ks.dtype).reshape(ks.shape)

    elif mode == 'literati-residual':
        # Residual binary: two 1-bit passes = 2 bits total
        # Pass 1: sign(x)*s1.  Pass 2: sign(residual)*s2.
        # Reconstruction: sign1*s1 + sign2*s2
        from turboquant.literatiquant import compute_scales_mean_abs, quantize_literati
        def compress(ks, li=None):
            D = ks.shape[-1]
            flat = ks.reshape(-1, D).float()
            # Pass 1
            s1 = compute_scales_mean_abs(flat, group_size)
            q1 = quantize_literati(flat, s1, group_size)
            # Pass 2 on residual
            residual = flat - q1
            s2 = compute_scales_mean_abs(residual, group_size)
            q2 = quantize_literati(residual, s2, group_size)
            return (q1 + q2).to(ks.dtype).reshape(ks.shape)

    elif mode == 'literati-perchannel':
        # Per-CHANNEL scale (across sequence dim) instead of per-group
        # Preserves relative channel magnitudes that attention needs
        def compress(ks, li=None):
            # ks: (batch, heads, seq, dim)
            # Scale per channel (dim axis), shared across seq positions
            ch_scale = ks.abs().mean(dim=2, keepdim=True).clamp(min=1e-8)  # (B,H,1,D)
            normalized = ks / ch_scale
            # Sign quantize the normalized values
            signs = torch.sign(normalized)
            signs[signs == 0] = 1.0
            # Reconstruct with per-channel scale preserved
            kq = signs * ch_scale
            return kq.to(ks.dtype)

    elif mode == 'literati-2bit':
        from turboquant.literatiquant import quantize_literati_2bit, compute_scales_for_2bit
        def compress(ks, li=None):
            D = ks.shape[-1]
            flat = ks.reshape(-1, D).float()
            scales = compute_scales_for_2bit(flat, group_size)
            kq = quantize_literati_2bit(flat, scales, group_size)
            return kq.to(ks.dtype).reshape(ks.shape)

    elif mode == 'literati-2bit-residual':
        # 2-bit residual: two 1-bit passes
        from turboquant.literatiquant import compute_scales_mean_abs, quantize_literati
        def compress(ks, li=None):
            D = ks.shape[-1]
            flat = ks.reshape(-1, D).float()
            s1 = compute_scales_mean_abs(flat, group_size)
            q1 = quantize_literati(flat, s1, group_size)
            residual = flat - q1
            s2 = compute_scales_mean_abs(residual, group_size)
            q2 = quantize_literati(residual, s2, group_size)
            return (q1 + q2).to(ks.dtype).reshape(ks.shape)

    elif mode == 'isoquant':
        from turboquant.isoquant import IsoQuantMSE
        def compress(ks, li):
            D = ks.shape[-1]
            dev = ks.device
            if li not in compressors:
                iq = IsoQuantMSE(D, bits, seed=li * 1000, mode='fast', device=str(dev))
                compressors[li] = iq
            iq = compressors[li]
            # Move buffers to same device as input
            iq_dev = next(iq.buffers()).device
            if iq_dev != dev:
                iq = iq.to(dev)
                compressors[li] = iq
            flat = ks.reshape(-1, D).float()
            x_hat, _ = iq(flat)
            return x_hat.to(ks.dtype).reshape(ks.shape)

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

    return _patch, _orig, prefill_done, compressors


if __name__ == '__main__':
    print("Loading model...", flush=True)
    model, tokenizer = load_model()

    from datasets import load_dataset
    text = '\n\n'.join(load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')['text'])
    input_ids = tokenizer(text, return_tensors='pt').input_ids[:, :512].to('cuda')

    prefill = 256
    print(f"\nQwen2.5-3B | RTX 5090 | {input_ids.shape[1]} tokens, prefill={prefill}")
    print(f"{'Method':>25}  {'PPL':>8}  {'Delta':>8}  {'%chg':>8}  {'tok/s':>7}  {'bits/e':>7}  {'ratio':>6}")
    print(f"{'─'*25}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}")

    # FP16
    ppl0, ts0, n = eval_ppl(model, tokenizer, input_ids, prefill)
    print(f"{'FP16':>25}  {ppl0:>8.2f}  {'—':>8}  {'—':>8}  {ts0:>6.1f}  {'16.0':>7}  {'1.0x':>6}")

    configs = [
        ('Literati 1-bit naive',    'literati-1bit',       128, 0, 1.125, 14.2),
        ('Literati 1-bit perchan',  'literati-perchannel', 128, 0, 1.125, 14.2),
        ('Literati 2-bit uniform',  'literati-2bit',       128, 0, 2.125, 7.5),
        ('Literati 2-bit residual', 'literati-residual',   128, 0, 2.25,  7.1),
        ('IsoQuant 2-bit',          'isoquant',            128, 2, 2.0,   8.0),
        ('IsoQuant 3-bit',          'isoquant',            128, 3, 3.0,   5.3),
        ('IsoQuant 4-bit',          'isoquant',            128, 4, 4.0,   4.0),
    ]

    for label, mode, gs, bits, bpe, ratio in configs:
        torch.cuda.empty_cache(); gc.collect()
        try:
            _patch, _orig, _pf, _c = make_patcher(mode, group_size=gs, bits=bits)
            ppl, ts, _ = eval_ppl(model, tokenizer, input_ids, prefill,
                                   patch_fn=_patch, orig_fn=_orig)
            _pf.clear(); _c.clear()
            delta = ppl - ppl0
            pct = delta / ppl0 * 100
            print(f"{label:>25}  {ppl:>8.2f}  {delta:>+8.2f}  {pct:>+7.1f}%  {ts:>6.1f}  {bpe:>7.3f}  {ratio:>5.1f}x")
        except Exception as e:
            print(f"{label:>25}  ERROR: {e}")

    print()
