
# NEGATIVE. RESULTS

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

    pip install torch transformers

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

From the `main` branch, go to **Actions → Ternary PTQ experiment → Run workflow**.
The run downloads the public weights and measures everything. Its result and
log are published back to `main` by a separate, narrowly permissioned job.

</details>

## 📈 Measured results (Experiment A, Qwen2.5-0.5B, CI run)

| Config | Perplexity | Verdict |
|---|---|---|
| bf16 baseline | 14.15 | normal model |
| int8 dynamic PTQ | 139.90 | worse than expected (~10x) - naive dynamic quant, no calibration |
| **full ternary PTQ** | **~8.2e+301** | 💥 **total collapse - hypothesis proven** |

**Sensitivity map (one layer ternarized at a time):**

| Layer type | Solo ppl | Reading |
|---|---|---|
| early q/k_proj (layers 0-13) | 15-32 | attention keys/values tolerate ternarization |
| o_proj / v_proj (mid) | 1e5-4e5 | break immediately |
| mid MLP (gate/up/down) | 3e4-5e6 | catastrophic |
| **lm_head** | **~2e+248** | the single most sensitive layer - never ternarize |

**Size accounting:** 988 MB (bf16) → 494 MB (int8) → **98.9 MB (packed ternary, 10x)**
- but the 98.9 MB model is measured garbage: **compression ratio without a quality gate is meaningless.**

Source of truth: `results/ptq_report.json` (committed by the experiment run).
Takeaway: post-training ternarization of a normally-trained LLM is refuted
with numbers; ternary requires training-time QAT (Experiment C).

## 🗺️ Roadmap

- **Experiment A** - ternary PTQ collapse proof (this release)
- **Experiment B** - selective ternarization (least-sensitive layers only)
- **Experiment C** - QAT distillation (needs GPU; out of CI scope)
- **Tier integration** - knowledge atoms + tensor decomposition + implicit
  weight codecs as one quality-gated pipeline

## ✅ Tests

    python -m pytest tests/ -v

CI runs the suite on CPU torch and uploads `ci_logs/pytest.log` as a workflow
artifact (including on test failure); it does not write CI logs back to the
repository.

## ⚠️ Honest limitations

- Perplexity on an 8-sentence corpus is a smoke metric, not a benchmark.
- absmean scaling is the simplest ternary scheme; smarter PTQ (GPTQ-style,
  calibration data) will do better - that is a future experiment, not a claim.
- If the collapse is smaller than expected, the honest conclusion changes
  with it. The report JSON is the source of truth.

## 📄 License

MIT - see [LICENSE](LICENSE).
