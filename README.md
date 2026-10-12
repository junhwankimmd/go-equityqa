# GO-EquityQA

Benchmark, model responses, citation judgments, verification harness, and analysis code for the study

> **Plausible citations, unsupported claims, and nationality-dependent recommendations: hidden failure modes of large language models in gynecologic oncology**

A preregistered 308-item gynecologic-oncology benchmark (252 confirmatory and 56 exploration-only items) administered to GPT-5.5, Claude Opus 4.8, and Gemini 3.1 Pro Preview. It tests whether cited papers support the claims they accompany, whether unsupported citation tracks a PubMed-count measure of literature coverage, and whether otherwise-invariant management recommendations change with patient nationality.

**Preregistration:** Open Science Framework, registration [osf.io/a379w](https://osf.io/a379w); project [osf.io/a39qk](https://osf.io/a39qk) (DOI [10.17605/OSF.IO/A39QK](https://doi.org/10.17605/OSF.IO/A39QK)). Registered before confirmatory data collection; confirmatory analyses use only items with `analysis_set = confirmatory`.

## Repository layout

```
data/
  items_v3_300.csv                       all 308 items (arm, axis, question, expected key point,
                                         ground-truth source, analysis_set, coverage terms and PubMed counts)
  items_v3_confirmatory.csv              the 252 confirmatory items
  adjudication_dualjudge_final.csv       expert adjudication of the 32 judge-discrepant citations
  review/                                blinded expert rating sheets (raters A and B) and unblinding key
                                         for the 36-response judge-validation sample
results/
  responses_20260712_075650.json         confirmatory model responses (252 items x 3 models)
  responses_derisk_combined.json         exploration-only provenance responses (40 items x 3 models)
  responses_20260708_203950.json         pilot-batch responses (includes the remaining 8 exploration items)
  scored_20260712_075650.json            primary-judge (Claude Opus 4.8) verdicts, all confirmatory arms
  dualjudge_gpt_primary_20260829.jsonl   second-judge (GPT-5.5) verdicts, all 300 confirmatory provenance responses
  scored_final_adjudicated_20260829.json final provenance dataset (primary judge with adjudicated discrepancies)
  merged_strict_final_20260829.csv       analysis table: strict fabrication ~ coverage
  merged_citefail_final_20260829.csv     analysis table: citation-support failure ~ coverage
  glmm_final_summaries.txt               mixed-effects model coefficients, standard errors, and 95% CIs
  scored_20260711_165334.json            exploration-set verdicts (final judge)
  scored_expl8_rejudge_20260826.json     final-judge rescoring of the 8 pilot-batch exploration items
  judge_sensitivity_gpt.json             second-judge verdicts on an exploration subsample
  factmatch_primary_20260829.jsonl       fact match (yes / partial / no) for confirmatory provenance responses
  scored_gemini_fr.json                  Gemini overconfidence and constrained-reasoning verdicts
  hard_pilot_raw.json, hard_probe_verdicts.csv   the 24 accuracy-probe responses and expert verdicts
  merged_*_confirmatory.csv, merged_coverage_20260711_170334.csv   coverage-merged tables
  supp_S1_category_counts.csv            counts underlying Supplementary Table S1
  fig1-4 (PNG as submitted; PDF vector versions)
src/
  runner.py              response collection, citation verification, answer judges, per-item aggregation
  citation_verifier.py   three-layer citation verification (existence, record identity, claim support)
  coverage_proxy.py      PubMed-count literature-coverage proxy
  run_coverage.py        coverage-proxy driver: queries counts and merges them with scored responses
  glmm_final.R           mixed-effects models reported in the manuscript
  stats_glmm.R           mixed-effects models on coverage-merged tables
  make_figs.py           Figures 2-3 and Supplementary Table S1 counts
  make_pretty_figs.py    Figures 1 and 4
docs/
  osf_preregistration_en.md
```

## Reproducing the analyses

Python 3.10+ (`pip install -r requirements.txt`) and R 4.3 with `lme4`.

```bash
Rscript src/glmm_final.R          # writes results/glmm_final_summaries.txt
python src/make_figs.py           # Figures 2-3 and Supplementary Table S1 counts
python src/make_pretty_figs.py    # Figures 1 and 4 (requires the Cairo graphics library)
```

Citation-support failure counts a response as failed when at least one citation is on-topic but unsupported, partially supported, misattributed, or non-existent and no citation supports the claim. Strict fabrication counts only misattributed or non-existent identifiers. Coverage is `log_count = log10(PubMed count + 1)`.

Re-scoring or collecting responses requires API keys: copy `.env.example` to `.env` and fill in the keys you have (providers without a key are skipped), then for example

```bash
RESPONSES_JSON=results/responses_20260712_075650.json python src/runner.py
```

Model outputs are not deterministic, so newly collected responses will differ from the archived ones. Responses were collected on 11-12 July 2026; the archived responses and verdicts are the data analyzed in the manuscript.

## License

Code: MIT (`LICENSE`). Data (items, responses, judgments): CC BY 4.0 (`DATA_LICENSE`).

## Contact

Junhwan Kim, MD, Soonchunhyang University Seoul Hospital (junhwankimmd@gmail.com)
