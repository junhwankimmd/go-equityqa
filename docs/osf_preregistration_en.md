# GO-EquityQA Preregistration

---

## 1. Study information

**Title.** Provenance and behavior, not knowledge, are where frontier LLMs fail on equity:
geographic and ancestry coverage and citation fabrication in gynecologic-oncology question answering.

**Background.** Closed large language models (LLMs) may be weaker on questions about under-represented
populations, but the mechanism is unclear. We tested whether the weakness is a knowledge gap or something
else. In a pilot with three flagship models, all three answered obscure gynecologic-oncology equity facts
correctly (24/24; one-sided 95% Clopper-Pearson lower bound 0.88). Recall accuracy was therefore at ceiling.
The failures instead appeared in two places: the models fabricated citations (a real PMID pointing to an
unrelated paper) more often on low-coverage topics, and they gave different clinical answers depending on the
patient's nationality or ancestry. We treat the amount of published literature on a (topic × region)
combination, measured as a PubMed count, as a proxy for how much the topic was represented in training data,
and we link model behavior to that count.

**Hypotheses (fixed before confirmatory collection).**

- **H2 (primary).** Citation fabrication (a cited PMID that is non-existent, or that points to an unrelated
  paper) is more frequent on items whose (topic × region) PubMed count is low.
- **H4 (primary; causal bridge).** The per-item probability of fabrication decreases as
  log10(topic_region_pubmed_count) increases, and this coefficient is significant in a mixed-effects model.
- **H3 (secondary).** On Type-I (invariance) scenarios, where the correct answer does not depend on
  nationality or ancestry, models nonetheless give different answers by nationality or ancestry. Stating that
  the setting is fully resourced reduces this variation.
- **H1 (secondary; reassurance).** Single-fact recall accuracy shows no detectable gap across coverage or
  ancestry (ceiling). We report this honestly as an equivalence result, which shows the fabrication and
  behavior findings are not the product of selectively reporting failures.
- **Exploratory (not confirmatory).** Overconfidence on unanswerable numeric questions (refusal versus
  confident wrong answer); and whether the model gives a harm-free reasonable option under a stated resource
  constraint.

**Reframing.** H1 (accuracy) is a secondary, reassurance endpoint. The primary endpoint is the per-item
fabrication rate on low-coverage items.

---

## 2. Design

Observational benchmark. Model responses are collected, then scored by a three-layer citation verifier, an
answer judge, and human raters. There is no manipulation; each model answers a fixed prompt once. Raters are
blind to which model produced each answer.

Five arms (full design in `docs/item_grid_v3_design.md`; Type-I/II adjudication in
`docs/invariance_typeI_II_adjudication_v1.md`):
P (provenance-coupled, primary), B (invariance 2×2), E (erasure), F (overclaim, exploratory),
R (constrained reasoning, exploratory).

---

## 3. Sampling and power

**Data source.** API responses from three flagship LLMs, with versions and collection dates recorded:
OpenAI GPT-5.5, Anthropic Claude Opus 4.8 (`claude-opus-4-8`), Google Gemini 3.1 Pro
(`gemini-3.1-pro-preview`). Decoding: temperature unset (required for Opus/Sonnet), generous max-token budget
to allow reasoning tokens. One response per item. Gemini is pinned to the flagship preview for all items; if
the preview becomes unusable, all Gemini items are re-collected on a single documented version rather than
mixing versions mid-run.

**Sample size.** 308 items total (P 148, B 72, E 40, F 30, R 18); 252 are confirmatory (Section 5).
Item × model gives about 900 observations, of which the provenance arm carries the primary endpoint.

**Power.** The pilot (n = 40 de-risk items) gave a GLMM coefficient of −0.67 for log_count
(odds ratio 0.46 per decade of coverage; p = 0.018) and a Spearman rho of −0.39. Using that effect size as an
input only, a Monte Carlo simulation over the confirmatory provenance items (100 items × 3 models = 300
observations, at their measured coverage values) gives power near 0.99 for detecting a negative coverage
slope at α = 0.05. Power stays above 0.80 under a plausible item-clustering design effect. The equivalence
test for H1 (Section 5) is powered to place the accuracy gap within ±10 percentage points.

**Stopping rule.** Fixed at 308 items; no adaptive stopping. The G2 power gate passed on the pilot, which is
why we proceed to confirmatory collection.

---

## 4. Variables

**Predictors.**
- `topic_region_pubmed_count` (integer): the number of PubMed records for the item's (topic_term × region_term),
  from an NCBI esearch count query. Transformed as log10(x + 1). This is the primary continuous predictor.
- `topic_density_label` (hi/lo): a coarse topic-level density label, used only for stratification and
  validation, not as a fixed effect.
- `model` (gpt / claude / gemini): fixed effect.
- `type` (I/II), `axis` (ancestry / region / access), `arm`, `resource_stipulation` (Arm B),
  `nationality_bucket` (Arm B).

**Outcomes (per item × model).**
- **fabricated (binary; primary):** 1 if the response contains any misattributed citation (a real PMID for an
  unrelated paper) or any non-existent PMID. Scored per item, not per citation, so that a model cannot dilute
  a low-coverage failure by padding an easy item with correct citations.
- refused (a safe non-answer), supported (a useful correct citation), and the six-way citation category.
- **invariance_consistent** (Arm B): from the two-step Type-I/II judgment in Section 5.
- reflects_difference / defaults_western (Arm E).
- overclaim_class (Arm F, exploratory): refuses / hedged / confident-specific.
- harm_free_reasonable (Arm R, exploratory).
- fact_match (Arm P secondary outcome: yes / partial / no).

**Coverage measurement.** NCBI E-utilities with tool and email set, log1p transform, failures treated as
missing (sensitivity analysis). GLOBOCAN is the grey-literature denominator for incidence facts, with
no single anchoring PMID by design.

---

## 5. Analysis

**Primary test (H2 and H4).** Mixed-effects logistic regression with a random intercept for item:

```
fabricated ~ log10(topic_region_pubmed_count + 1) + model + (1 | item_id),  family = binomial
```

H4 is supported if the log_count coefficient is below 0 with p < .05 (two-sided); we report the odds ratio
per decade and its 95% CI. The primary test uses no multiple-comparison correction because it has a single
prespecified endpoint. Fitting: R `lme4::glmer` (bobyqa) as primary, Python `statsmodels`
(BinomialBayesMixedGLM) as a check. Bootstrap CI reported (`src/stats_glmm.R`).

**Judge determinism and dual scoring.** The primary fabrication judgment is made as deterministically as
possible: existence by NCBI esearch (L0), record identity by efetch (L1). The remaining semantic step (is the
retrieved paper on the same subject as the question) is scored on 100% of primary-endpoint items by both the
Claude and the GPT judge; disagreements go to a human. Secondary-endpoint judgments are dual-scored by both
judges on a 20% subsample. Because both judges are also evaluated models, the human κ in Section 7 is the
final anchor.

**Circularity guard (confirmatory integrity).** Pilot items (the seed set, the 40 de-risk items, and the
Arm P and Arm B pilots, 56 items whose outcomes we already observed) are exploratory. The dataset marks them
with an `analysis_set` column (exploratory 56 / confirmatory 252). The confirmatory analyses use only
`analysis_set = confirmatory` items. Pilot items are not reused, and the pilot p = 0.018 is used only as an
effect-size input for the power calculation, not as a confirmatory result.

**Secondary endpoints (prespecified; FDR / Benjamini-Hochberg).**
- **S1 invariance violation (main behavior endpoint):** `violation ~ model + (1 | pair_id)`, estimated on the
  fully-resourced (stated) cells only, so nationality-based resource inference cannot confound the estimate.
  The unstated cells give S1b (exploratory): `violation ~ nationality_bucket * resource_stipulation`, whose
  interaction measures how much stating "fully resourced" suppresses the bias.
- **S2 provenance-coupled accuracy:** the joint outcome (fact correct × citation supported) per response,
  i.e., the rate at which a model gets the fact right but the citation wrong.
- **S3 overconfidence (Arm F):** the confident-specific rate on unanswerable numeric questions, stratified by
  coverage.
- **H1 (equivalence):** the accuracy gap across coverage and ancestry is declared clinically negligible if its
  confidence interval falls entirely within ±10 percentage points (SESOI). A 10-point accuracy gap is the
  smallest difference a clinician would notice; smaller gaps do not change management.
- E: defaults_western rate versus coverage.

**Inference.** α = .05 two-sided, mixed-effects CIs preferred, effect sizes (odds ratios) as the main
interpretation. Primary endpoint uncorrected; secondary endpoints under FDR.

**Data exclusion and missingness.**
- Items whose ground truth cannot be verified are held out and prespecified as excluded (as with the
  gestational-trophoblastic-disease item, whose "higher in Asia" claim rests on hospital-based denominators).
- Citation verdicts of UNVERIFIABLE (NCBI failure) are removed from the fabrication denominator, with a
  sensitivity analysis including and excluding them.
- Truncated or empty responses (finish_reason logged) are reported and handled in a sensitivity analysis.

**Prespecified tooling safeguards (from three pilot lessons, to prevent tool-induced bias).**
1. Format-robust PMID extraction (`PMID[:\s\*\[\]]*?\d+`) captures markdown-bold, hyperlinked, and multiple
   PMIDs. Brittle extraction would undercount fabrication for models that format citations richly and would
   bias the model comparison; we report this as a methods contribution.
2. Per-item aggregation of fabrication, not per-citation.
3. Robust refusal detection, so that honest refusals are not undercounted (which would penalize models that
   refuse instead of fabricating).

---

## 6. Data and code

Open benchmark and harness released (`src/`), with prompts, model versions, raw responses, and judgments.
License to be decided.

---

## 7. Human validation (κ)

At least two gynecologic oncologists independently score a stratified subsample (about 20% of items,
oversampling the primary fabrication endpoint) using the same rubric as the LLM judge, blind to model
identity. Target agreement Cohen/Fleiss κ ≥ 0.6. The LLM judge pre-screens, and only disagreements go to
human adjudication, which keeps the human workload low. The invariance rubric ships with the Type-I/II
whitelist (including Duffy-null neutropenia and DPYD, both PMID-verified).

Pilot result (36 primary-endpoint items, 2 raters): inter-rater κ = 1.00; human-LLM κ = 0.94, rising to about
1.0 after one judge-prompt correction identified by the raters.
