# Mechanistic Interpretability: Two Empirical Audits

Two self-contained Google Colab notebooks that **test** claims in mechanistic interpretability rather than
demonstrate them. Both run end to end on a single GPU, report negative results where they occur, and
include the controls needed to tell a real effect from a measurement artefact.

| Notebook | Question | Scale |
|---|---|---|
| [`Mech_Interp_Method_Benchmark.ipynb`](Mech_Interp_Method_Benchmark.ipynb) | Which attribution method should I use, on which model, and what does it cost? | 5 models × 2 tasks × 7 methods |
| [`Superposition_to_Sparse_Codes.ipynb`](Superposition_to_Sparse_Codes.ipynb) | Does the superposition → sparse-codes pipeline hold when you test each step? | 3-step replication, synthetic + CLIP + GPT-2 |

Open either directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aslansd/Mechanistic-Interpretability-Two-Empirical-Audits/blob/main/Mech_Interp_Method_Benchmark.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aslansd/Mechanistic-Interpretability-Two-Empirical-Audits/blob/main/Superposition_to_Sparse_Codes.ipynb)

---

## 1. Benchmarking attribution methods across models

A harness that runs seven component-attribution methods over two tasks and five architectures, then scores
them on **three independent axes**: agreement with exhaustive activation patching, reference-free
faithfulness, and compute cost.

**Headline result.** Rank agreement with exhaustive activation patching, restricted to the 32 heads that
matter:

| method | passes | GPT-2 | Pythia-410M | Qwen3-0.6B | OLMo-2-1B | Gemma-3-1B |
|---|---|---|---|---|---|---|
| **attribution patching** | **4** | **0.990** | **0.945** | **0.876** | **0.923** | **0.908** |
| direct logit attribution | 1 | 0.760 | 0.686 | 0.732 | 0.643 | 0.688 |
| mean ablation | n·h+1 | 0.538 | 0.541 | 0.587 | 0.466 | 0.363 |
| zero ablation | n·h+1 | 0.283 | 0.377 | 0.391 | 0.499 | 0.365 |
| attention to answer | 1 | 0.310 | 0.163 | 0.040 | 0.191 | 0.179 |
| random | 0 | −0.029 | −0.094 | 0.044 | 0.001 | 0.052 |

Attribution patching reproduces exhaustive patching at **ρ = 0.88–0.99 for four model passes instead of
105–449** — a 26× to 112× saving — and holds across MHA, GQA (2:1 and 4:1), RMSNorm, post-norm and
sliding-window attention.

**The more interesting finding: the two evaluation axes disagree about which method is best.** Rank
agreement crowns attribution patching. Normalised faithfulness crowns *mean ablation* on three of five
models — above exhaustive activation patching itself on four. That is not a contradiction: patching
measures each head in isolation and so underrates redundant heads, while faithfulness asks whether a
method's top-k, kept *together*, carries the behaviour. **"Which method is best" has no answer without
saying what for**: screen with attribution patching, extract circuits with mean ablation.

**Validation.** On GPT-2 IOI the harness recovers the published circuit unprompted — name movers (L9.H9,
L10.H0) and S-inhibition heads (L8.H6, L8.H10, L7.H3, L7.H9) — and the largest rank disagreement between
attribution and exhaustive patching among important heads is *two positions*.

### What's inside
- **Method zoo** (`METHODS`): activation patching, zero/mean ablation, attribution patching, direct logit
  attribution, an attention-based correlational baseline, and a random control. Adding a method means
  writing `fn(model, task, counter) -> [n_layers, n_heads]`.
- **Task battery**: IOI and induction, each gated on whether the model actually performs the task —
  benchmarking on a model that can't do the task measures noise.
- **Three metrics**: top-k-restricted Spearman, random-normalised faithfulness AUC, and measured cost.
- **Family B**: cross-model logit lens with a splice sanity check.

### Notes and gotchas (learned the hard way)
- **All-head Spearman is mostly noise.** ~90% of heads have no effect and their ordering is arbitrary in
  every method; the un-restricted numbers put zero ablation at −0.084 where the restricted value is 0.283.
- **Faithfulness AUC has a high floor** (random scores 0.49–0.85) because ablating heads leaves MLPs intact.
  Normalise against the random control or the numbers are unreadable.
- **`fold_ln` is unsupported for OLMo-2**, which silently invalidates DLA and the logit lens for it.
  Preflight detects this and flags the affected rows.
- **Pythia needs an attribute alias**: `transformers` v5 renamed GPT-NeoX's `embed_out` to `lm_head`.
- **SmolLM3-3B still OOMs on an A100 (40 GB)** during load, most likely because
  `enable_compatibility_mode()` materialises extra full-precision copies. Force bf16, skip compat mode, or
  use a smaller model.

---

## 2. Superposition → sparse interpretable codes

A replication of the three-step framework of Klindt et al., *A unifying framework from neural superposition
to sparse interpretable codes*, Nature Machine Intelligence 8:1025–1037 (2026).

**The key design decision:** the framework's own premise is that true latents are unobservable — which is
why its third step (interpretability metrics) exists. Steps 1 and 2 therefore *cannot* be tested on a real
LLM, because there is no ground-truth `z`. The notebook builds synthetic worlds for those, and uses real
models (CLIP, GPT-2 + SAEs) only where claims are checkable. Conflating the two is the most common way this
literature overclaims.

### Results

| Claim | Outcome |
|---|---|
| Compressed sensing `M = O(K log N/K)` | **Held**, constant ≈ 1.0–1.5 |
| Sparse coding recovers the dictionary | **Held**, 0.984 |
| Theorem 1 linear identifiability | **Held conditionally**, 0.973 — see below |
| SAE features more interpretable than neurons | **Held**, 0.617 vs 0.436 |
| CLIP representations are additive | **Weak**, 0.894 vs 0.816 control |
| MLP encoders close the amortisation gap | **Not reproduced** under matched hyperparameters |
| Eq. (9): mixing reduces interpretability | **No detectable effect** |

**Easier tasks give worse identifiability.** With overlapping clusters, `h = f∘g` is essentially perfectly
linear (R² 0.943 against a nonlinear ceiling of 0.969). With separable clusters the classifier hits 100%
accuracy and linearity *collapses* to 0.34–0.47. Theorem 1 requires *globally minimising* cross-entropy;
once the task is trivially solvable, many nonlinear encoders achieve zero loss and the objective stops
constraining the geometry. Rising accuracy is not evidence of faithfulness to latent structure.

**Additivity needs a control.** Raw `cos(E(A and B), E(A)+E(B)) = 0.921` looks like strong confirmation —
until you sum two objects *not in the image* and get 0.863. The compositional signal is the 0.06–0.08
*gap*, not the 0.92. CLIP-style spaces are strongly anisotropic, so a high cosine floor is expected for any
such claim. The measurement was validated on a mock encoder built to be exactly additive (returns 1.000 vs
0.055 control), so the small CLIP gap is a property of CLIP, not the metric.

**Binding is not additive.** A scene and its colour-swapped counterpart have attribute sums at cosine
**0.975**, and the true embedding is nearly equidistant from the correct sum (0.904) and the wrong one
(0.887). Object-level additivity can hold while attribute-level additivity fails.

**Reconstruction quality is not evidence of feature recovery.** At λ=0.01 the SAE reconstructs excellently
and looks healthy, while recovering the wrong features (MCC 0.483, L0 = 35 against a true K = 3). We can
only detect this because we know `z`.

### What's inside
Part I additivity, analogy and binding in CLIP with controls · Part II linear identifiability with a
nonlinear-ceiling comparison · Part III compressed-sensing phase diagram, λ sweep, SAE vs ISTA vs
MLP-encoder SAE · Part IV automated word intrusion on SAE features, neurons and random directions · Part V
placing real models on the scaling law.

---

## Running

Both notebooks install their own dependencies. Open in Colab, set **Runtime → Change runtime type → GPU**,
and run top to bottom.

- **Benchmark**: ~15 min on an A100, fits on a T4. Set `QUICK = False` for 8 prompts/task before quoting
  numbers — the published run used `QUICK = True` (4 prompts).
- **Superposition**: ~25–40 min on a T4. Parts II–III are pure PyTorch; Parts I and IV download CLIP, GPT-2,
  an SAE and a sentence-transformer (~1.5 GB).

Key versions: `transformer_lens >= 3.0` (requires `transformers >= 5.4`; Colab will ask you to restart
after install), `sae_lens >= 6.0`. A Hugging Face token is needed only for gated models (Gemma 3, Llama 3.2).

## Limitations

Both notebooks are **single-seed** and small-scale. Treat the numbers as a working harness rather than a
measurement: the benchmark used 4 prompts/task, and the word-intrusion corpus is 24 base sentences, which
is too small to resolve the mixing effect in Part IV. Architecture is confounded with scale and training
data across the five benchmark models, so no result there attributes an effect to a single property.

## Citation

Cite the reference paper of the second notebook:


```bibtex
@article{klindt2026unifying,
  title   = {A unifying framework from neural superposition to sparse interpretable codes},
  author  = {Klindt, David and O'Neill, Charles and Reizinger, Patrik and Maurer, Harald and Miolane, Nina},
  journal = {Nature Machine Intelligence}, volume = {8}, pages = {1025--1037}, year = {2026}
}
```

Built on [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens),
[SAELens](https://github.com/decoderesearch/SAELens) and
[Neuronpedia](https://www.neuronpedia.org).

## License

MIT for the code in these notebooks. Model weights and datasets retain their own licenses.
