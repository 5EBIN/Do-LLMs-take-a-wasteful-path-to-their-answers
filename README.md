# Do LLMs take a wasteful path to their answers?

A small interpretability study testing whether a transformer's internal
representation wanders unnecessarily on the way to an answer — and whether that
slack could be exploited to make inference cheaper.

**Result: the hypothesis is falsified.** Across five measurements (Qwen2.5-0.5B
and 1.5B, base + instruct), the representational "wandering" turns out to be
**necessary, content-driven work**, not removable slack. The negative result is
the finding — and a smaller positive finding survived (below).

---

## TL;DR findings

- **M1 — Entropy by depth:** the answer's logit-lens entropy stays high through
  early/middle layers and collapses only in the **final ~20%** of depth. The
  trajectory is **non-monotonic** (it zigzags), and there's little redundant
  middle to skip → early-exit has little to work with. (Replicates at both model
  sizes; base ≈ instruct.)
- **M2 — Detour factor:** the angular (direction-only) path is **~9× longer**
  than the straight-line distance. Heavy directional wandering exists.
- **M3 — Consistency by task type:** wandering is **task-structured** —
  arithmetic traces a consistent route across prompts (mean ~0.65, *structural*),
  while factual/commonsense route erratically (~0.25–0.29). All types converge to
  a shared "commit" direction (~0.95) at the final layer.
- **M4 — Paraphrase test (the decisive one):** holding the *answer* fixed and
  varying only wording, paraphrases route **more** consistently than unrelated
  facts (overall ~0.59 vs 0.29 baseline). → the earlier erratic-ness was sensible
  *content routing*, **not** dithering. Efficiency idea falsified.

**What survived:** *representational paths to factual answers are
phrasing-invariant in their destination and content-determined in their route,
converging to a shared commit direction only in the final layers; stable from
0.5B to 1.5B.*

---

## Plots
| File | Measurement |
|------|-------------|
| `plot1_entropy.png` | M1 — when the answer decides |
| `plot2_detour.png` | M2 — detour factor |
| `plot3_trajectories.png` | M2b — 2D PCA of the path (qualitative only) |
| `plot4_consistency.png` | M3 — consistency by task type |
| `plot5_paraphrase.png` | M4 — paraphrase test |
| `summary_numbers.csv` | all key numbers |

## Run it
Open the notebook on Kaggle/Colab (GPU + internet), set `MODEL_SIZE` to
`"0.5B"` or `"1.5B"`, Run All. ~10–25 min, mostly weight download.

## Method (one paragraph)
The forward pass is read at the **last token position** at every layer. M1 uses
the **logit lens** (project each hidden state through the model's own
unembedding) and measures next-token entropy. M2–M4 work on the residual-stream
**trajectory** through representation space: M2 is total path length over
straight-line distance (raw and unit-normalized "angular"); M3/M4 measure the
mean pairwise cosine of layer-to-layer *step directions* across prompts.

## Why no training experiment
A natural next step is training the model to take straighter (more monotonic)
trajectories so it could exit early. I didn't pursue it: the observations (M1's
non-monotonic path, M3/M4's content-necessary wandering) predict that forcing the
path straight would cost accuracy on content-heavy queries. That's a prediction
from these measurements, not a tested result — confirming or refuting it would
need the training run, which this study does not include.

## Honest limitations
Small models (≤1.5B). The **logit lens is biased in middle layers** — some M1/M3
wiggle is the probe, not the model; a tuned lens would harden this. Few facts in
M4; last-token representations are sensitive to phrasing **length**. And even
genuine route-divergence wouldn't prove routes can be *forced together* without
hurting accuracy — that needs a training experiment, not done here.

## Related work
Spectral Journey (2025, transformers learn a spectral shortest-path heuristic);
"Diminishing Returns of Early-Exit" (2026); Mixture-of-Depths (2024); logit lens
/ tuned lens. Early-exit via learned/forced layer convergence is an active 2026
area; this study is observational and in a different (generative, non-distilled)
regime, so it neither builds on nor tests those methods.
