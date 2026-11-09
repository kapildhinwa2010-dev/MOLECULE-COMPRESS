"""Ternary PTQ collapse experiment (Experiment A).

Post-training ternarization of a normally-trained LLM is expected to destroy
quality. This module measures that collapse honestly, with numbers:

    Qwen2.5-0.5B (bf16, public HF weights)
      -> perplexity on a fixed tiny corpus            (baseline)
      -> ternarize every nn.Linear weight (absmean)   -> perplexity again
      -> int8 dynamic PTQ (comparison point)
      -> per-layer sensitivity: ternarize one layer at a time, delta ppl
      -> size accounting: bf16 vs packed ternary vs int8
      -> results/ptq_report.json + printed summary

Everything uses public tooling only (torch, transformers). No private
research code is used or referenced anywhere in this repository.

Run the real experiment:

    python ternary_ptq.py --model Qwen/Qwen2.5-0.5B --out results/ptq_report.json

Synthetic smoke check (no download):

    python ternary_ptq.py --demo
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
from torch import nn

__version__ = "0.1.0"

# Fixed tiny corpus. Deliberately short so CPU runs stay bounded.
CORPUS = [
    "The capital of France is Paris.",
    "Water boils at one hundred degrees Celsius.",
    "The sun rises in the east every morning.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The Pyramids of Giza are in Egypt.",
    "A triangle has three sides and three angles.",
    "Shakespeare wrote Hamlet in the early seventeenth century.",
    "The Pacific Ocean is the largest ocean on Earth.",
]


# --------------------------------------------------------------------------
# core math
# --------------------------------------------------------------------------

def absmean_scale(w: torch.Tensor) -> torch.Tensor:
    """BitNet-style absmean scale, clamped away from zero."""
    return w.abs().mean().clamp(min=1e-8)


def ternarize_tensor(w: torch.Tensor) -> torch.Tensor:
    """Map weights to {-1, 0, +1} via absmean-scaled rounding."""
    scale = absmean_scale(w)
    return torch.clamp(torch.round(w / scale), -1, 1)


def ternarize_(module: nn.Module, names=None) -> int:
    """Ternarize nn.Linear weights in-place. Returns count of layers hit."""
    hit = 0
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear) and (names is None or name in names):
            with torch.no_grad():
                m.weight.copy_(ternarize_tensor(m.weight.float()).to(m.weight.dtype))
            hit += 1
    return hit


def ternary_bytes(numel: int) -> int:
    """Packed ternary storage: 5 trits per byte (3^5 = 243 <= 256)."""
    return math.ceil(numel / 5)


def size_report(model: nn.Module) -> dict:
    linear_params = 0
    total_params = 0
    for p in model.parameters():
        total_params += p.numel()
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear):
            linear_params += m.weight.numel()
    return {
        "total_params": total_params,
        "linear_params": linear_params,
        "bf16_bytes": total_params * 2,
        # linear weights int8 (1 byte), everything else stays bf16
        "int8_bytes": linear_params * 1 + (total_params - linear_params) * 2,
        # linear weights packed ternary (5 per byte), everything else bf16
        "ternary_bytes": ternary_bytes(linear_params) + (total_params - linear_params) * 2,
    }


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def perplexity(model, tokenizer, texts=CORPUS) -> float:
    """Mean NLL perplexity over a fixed corpus (single-sentence batches)."""
    model.eval()
    nll, ntok = 0.0, 0
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, return_tensors="pt")
            # Causal-LM loss shifts labels, so the first token is context only.
            n = enc["input_ids"].numel() - 1
            if n <= 0:
                continue
            out = model(**enc, labels=enc["input_ids"])
            nll += out.loss.item() * n
            ntok += n
    if ntok == 0:
        raise ValueError("perplexity requires at least one text with two or more tokens")
    return math.exp(nll / ntok)


def layer_sensitivity(model, tokenizer, texts=CORPUS, k: int = 12) -> list:
    """Ternarize one Linear at a time, measure delta ppl, restore."""
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    saved = {n: m.weight.detach().clone() for n, m in linears}
    if k >= len(linears):
        idx = list(range(len(linears)))
    else:
        idx = sorted(set(round(i * (len(linears) - 1) / max(1, k - 1)) for i in range(k)))
    rows = []
    for i in idx:
        name, m = linears[i]
        ternarize_(model, names={name})
        try:
            ppl = perplexity(model, tokenizer, texts)
        finally:
            with torch.no_grad():
                m.weight.copy_(saved[name])
        rows.append({"layer": name, "ppl_solo": round(ppl, 4)})
    return rows


def int8_ppl(model, tokenizer, texts=CORPUS):
    """Dynamic int8 PTQ as a sanity comparison. Returns None if unsupported."""
    try:
        q = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        return perplexity(q, tokenizer, texts)
    except Exception as exc:  # noqa: BLE001 - comparison is optional
        print("int8 comparison unavailable:", exc)
        return None


# --------------------------------------------------------------------------
# experiment runner
# --------------------------------------------------------------------------

def run_experiment(model_name="Qwen/Qwen2.5-0.5B", out_path="results/ptq_report.json",
                   sensitivity_k=12) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    print("loading", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    sizes = size_report(model)
    print("size report:", json.dumps(sizes))

    print("baseline ppl ...")
    base = perplexity(model, tokenizer)
    print("baseline ppl =", round(base, 4))

    print("int8 dynamic PTQ ppl ...")
    i8 = int8_ppl(model, tokenizer)

    print("full ternary PTQ ...")
    n = ternarize_(model)
    tern = perplexity(model, tokenizer)
    print("ternary ppl =", round(tern, 4), "after", n, "layers")

    sens = []
    # sensitivity must run on the ORIGINAL weights
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    if sensitivity_k > 0:
        print("layer sensitivity (k =", sensitivity_k, ") ...")
        sens = layer_sensitivity(model, tokenizer, k=sensitivity_k)

    report = {
        "experiment": "ternary PTQ collapse (A)",
        "model": model_name,
        "version": __version__,
        "baseline_ppl": round(base, 4),
        "int8_ppl": None if i8 is None else round(i8, 4),
        "ternary_ppl": round(tern, 4),
        "collapse_factor": round(tern / max(base, 1e-9), 2),
        "layers_ternarized": n,
        "sizes": sizes,
        "sensitivity": sens,
        "corpus_size": len(CORPUS),
        "seconds": round(time.time() - t0, 1),
    }
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print("report written to", out_path)
    return report


def demo_report() -> dict:
    """Synthetic smoke check with a tiny random model (no download)."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
    sizes = size_report(model)
    n = ternarize_(model)
    w = model[0].weight
    unique = sorted(set(w.unique().tolist()))
    return {"layers_ternarized": n, "sizes": sizes, "unique_values": unique}


def main() -> None:
    ap = argparse.ArgumentParser(description="Ternary PTQ collapse experiment")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", default="results/ptq_report.json")
    ap.add_argument("--sensitivity", type=int, default=12)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        print(json.dumps(demo_report(), indent=2))
        return
    rep = run_experiment(args.model, args.out, args.sensitivity)
    print(json.dumps({k: v for k, v in rep.items() if k != "sensitivity"}, indent=2))


if __name__ == "__main__":
    main()
