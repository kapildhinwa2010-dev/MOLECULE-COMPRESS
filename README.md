# 🧪 MOLECULE-COMPRESS

**Quality-gated LLM compression experiments.** This repository measures what
compression actually does to a model - with numbers, not marketing.

> ⚠️ **Honest framing up front:** "no quality loss at extreme compression" is
> not a blanket claim we make. Each tier below has its own measured quality
> gate, and this repo publishes the failures alongside the wins.

---

## 🎯 What is being tested here

**Experiment A (this release): Ternary PTQ collapse.**

Can a normally-trained LLM be converted to ternary weights ({-1, 0, +1},
~1.6 bits/weight) *after* training, without retraining?

Hypothesis (from compression theory): no - post-training ternarization should
collapse quality, and ternary only works when the model is *trained* that way
(BitNet-style QAT). This repo proves or refutes that with measured perplexity
on Qwen2.5-0.5B.

## 📊 What gets measured

| Measurement | Meaning |
|---|---|
| Baseline perplexity | bf16 model on a fixed 8-sentence corpus |
| Full ternary PTQ perplexity | every nn.Linear ternarized (absmean scale) |
| int8 dynamic PTQ perplexity | comparison point (known to be safe) |
| Per-layer sensitivity | ternarize one layer at a time, delta ppl |
| Size accounting | bf16 vs packed-ternary vs int8 bytes |

## 🚀 Quick start

Install (GPU default):

    pip install torch

<details>
<summary>💻 CPU-only install</summary>

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers

</details>

Synthetic smoke check (no model download):

    python ternary_ptq.py --demo

Run the real experiment on Qwen2.5-0.5B (CPU is enough, ~5-10 min):

    python ternary_ptq.py --model Qwen/Qwen2.5-0.5B --out results/ptq_report.json

<details>
<summary>▶️ One-click: run via GitHub Actions</summary>

Go to **Actions → Ternary PTQ experiment → Run workflow**. The run downloads
the public weights, measures everything, and commits `results/ptq_report.json`
plus `ci_logs/experiment.log` back to this branch.

</details>

## 📈 Expected results (to be replaced by measured numbers)

| Config | Expected | Why |
|---|---|---|
| bf16 baseline | ppl ~3-6 on this corpus | normal model |
| int8 dynamic PTQ | near baseline | known-safe tier |
| full ternary PTQ | **collapse (ppl >> 10x)** | 1.6 bits/weight cannot represent trained weights |
| single-layer ternary | layer-dependent delta | sensitivity map |

⚠️ Until `results/ptq_report.json` exists in this repo, every number above is
a hypothesis, not a result.

## 🗺️ Roadmap

- **Experiment A** - ternary PTQ collapse proof (this release)
- **Experiment B** - selective ternarization (least-sensitive layers only)
- **Experiment C** - QAT distillation (needs GPU; out of CI scope)
- **Tier integration** - knowledge atoms + tensor decomposition + implicit
  weight codecs as one quality-gated pipeline

## ✅ Tests

    python -m pytest tests/ -v

CI runs the suite on CPU torch and commits its log as `ci_logs/pytest.log`
(the log is the proof - it is committed even when tests fail).

## ⚠️ Honest limitations

- Perplexity on an 8-sentence corpus is a smoke metric, not a benchmark.
- absmean scaling is the simplest ternary scheme; smarter PTQ (GPTQ-style,
  calibration data) will do better - that is a future experiment, not a claim.
- If the collapse is smaller than expected, the honest conclusion changes
  with it. The report JSON is the source of truth.

## 📄 License

MIT - see [LICENSE](LICENSE).
