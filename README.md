# GO-EquityQA

Benchmark, model responses, citation judgments, and analysis code for:

> **Citation-support failure and resource-inference bias in frontier large language models: the case of gynecologic oncology** (submitted).

Preregistration: https://osf.io/a379w · OSF project: https://osf.io/a39qk (DOI 10.17605/OSF.IO/A39QK)

## What this study is

A preregistered 308-item benchmark testing whether frontier large language models (GPT-5.5, Claude Opus 4.8, Gemini 3.1 Pro Preview) fail on gynecologic-oncology equity questions through missing knowledge, unreliable citation provenance, or nationality-dependent behavior, evaluated against a PubMed-count literature-coverage proxy per topic-by-region pair.

## Repository layout

```
data/
  items_v3_300.csv                     # all 308 benchmark items (analysis_set marks confirmatory/exploratory)
  items_v3_confirmatory.csv            # the 252 confirmatory items
  adjudication_dualjudge_final.csv     # human adjudication of the 32 judge-discrepant citations
results/
  responses_20260712_075650.json       # confirmatory model responses (252 items x 3 models)
  responses_20260708_203950.json       # armP pilot-batch responses (incl. the 8 exploration items)
  scored_20260712_075650.json          # primary (Claude-judge) citation verdicts
  dualjudge_gpt_primary_20260829.jsonl # second-judge (GPT) verdicts, all 300 primary responses
  scored_final_adjudicated_20260829.json  # FINAL primary dataset (dual judge + human adjudication)
  merged_strict_final_20260829.csv     # analysis file: strict fabrication ~ coverage
  merged_citefail_final_20260829.csv   # analysis file: citation-support failure ~ coverage
  scored_20260711_165334.json          # exploration-set verdicts (final judge)
  scored_expl8_rejudge_20260826.json   # final-judge rescoring of the 8 armP exploration items
  factmatch_primary_20260829.jsonl     # fact-match (yes/partial/no) for the registered S2 outcome
  scored_gemini_fr.json                # Gemini overconfidence / constrained-reasoning arms
  hard_pilot_raw.json                  # the 24 hard accuracy-probe responses (8 items x 3 models)
  hard_probe_verdicts.csv              # expert verdicts for the 24 accuracy-probe responses (all correct)
  merged_coverage_*.csv, supp_S1_category_counts.csv, fig*.png/pdf
src/
  runner.py              # collection + scoring pipeline
  citation_verifier.py   # 3-layer citation verification (esearch -> efetch -> LLM support judge)
  coverage_proxy.py      # PubMed coverage counts per topic x region
  stats_glmm.R           # mixed-effects logistic models (lme4)
docs/
  osf_preregistration_en.md
```

## Reproducing the primary analyses

```bash
# mixed-effects models (R >= 4.3, lme4)
Rscript -e 'library(lme4);
for (p in c("results/merged_strict_final_20260829.csv","results/merged_citefail_final_20260829.csv")) {
  d <- read.csv(p); d$model <- factor(d$model, levels=c("gpt","claude","gemini"));
  print(summary(glmer(fabricated ~ log_count + model + (1|item_id), d, family=binomial,
        control=glmerControl(optimizer="bobyqa")))$coefficients) }'
```

Re-running collection or judging requires API keys in a local `.env` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `NCBI_EMAIL`); never commit keys.

## Notes on the data

- Judge models: primary `claude-opus-4-8`; second judge `gpt-5.5`; both anchored to the original question. The 32 citations on which the judges disagreed were adjudicated by two blinded gynecologic oncologists (`data/adjudication_dualjudge_final.csv`); `scored_final_adjudicated_20260829.json` is the endpoint-defining dataset.
- In `items_v3_300.csv`, the Korean flag `출처 확인 필요` in `ground_truth_source` marks probe items whose primary outcome is citation support rather than a verifiable stated fact ("source verification required").
- Responses were collected 11-12 July 2026 (Gemini overconfidence/constrained-reasoning arms on 12 July 2026).

## License

Code: MIT (see `LICENSE`). Data (items, responses, judgments): CC BY 4.0 (see `DATA_LICENSE`).
