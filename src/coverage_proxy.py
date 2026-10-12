# -*- coding: utf-8 -*-
"""
GO-EquityQA · Arm C — Coverage-Proxy Pipeline
=============================================
Purpose:
  The pretraining corpora of closed LLMs are not observable. The amount of
  available literature for each (topic x region) pair is therefore approximated
  by the number of PubMed records, and related to model behavior via
  `accuracy ~ log10(count)` and `hallucination ~ log10(count)`.
  Rationale: factual recall in language models correlates with the number of
  relevant pretraining documents (Kandpal et al., ICML 2023), and QA accuracy
  correlates with entity frequency/popularity (Mallen et al., ACL 2023);
  hallucination increases for low-frequency knowledge.

Requirements: requests, pandas, numpy, statsmodels, scipy, matplotlib.

Notes:
  - NCBI E-utilities usage policy: identify tool/email; at most 3 requests/s
    without an API key, 10 requests/s with one.
  - Record counts are an approximate proxy and depend strongly on query design;
    define topic_term / region_term for each item carefully (see build_query).
"""

import os
import time
import json
import math
import urllib.parse
from typing import Optional

import requests
import numpy as np
import pandas as pd

# Load .env if python-dotenv is available; existing environment variables are not overridden.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

# ------------------------------------------------------------------
# 0) Configuration (NCBI_EMAIL / NCBI_API_KEY are read from the environment)
# ------------------------------------------------------------------
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "your_email@example.com")
NCBI_TOOL = "GO-EquityQA-coverage"
NCBI_API_KEY: Optional[str] = os.getenv("NCBI_API_KEY")   # 10 req/s with a key, 3 req/s without
ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# Minimum interval between requests (s): 0.11 (~9/s) with a key, 0.34 (~3/s) without.
MIN_INTERVAL = 0.11 if NCBI_API_KEY else 0.34


# ------------------------------------------------------------------
# 1) PubMed record count (coverage proxy)
# ------------------------------------------------------------------
def pubmed_count(query: str, retries: int = 4) -> Optional[int]:
    """
    Return the number of PubMed records matching `query` (retmax=0, so no
    records are downloaded). Failed requests are retried with increasing backoff.
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": 0,
        "tool": NCBI_TOOL,
        "email": NCBI_EMAIL,
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    url = ESEARCH + "?" + urllib.parse.urlencode(params)
    for i in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                cnt = int(r.json()["esearchresult"]["count"])
                return cnt
            # Back off on 429/5xx
            time.sleep(1.5 * (i + 1))
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None  # Treated as missing downstream


def build_query(topic_term: str, region_term: str) -> str:
    """
    Build a (topic x region) PubMed query.
      - topic_term specifies the disease/domain (e.g. "cervical cancer incidence").
      - region_term specifies a country or region (e.g. "Nigeria", "Africa",
        "sub-Saharan Africa").
      - Overly narrow terms return zero hits; overly broad terms dilute the signal.
    Field tags ([Title/Abstract], [MeSH]) or date filters may be added for precision.
    """
    topic = f'("{topic_term}")'
    region = f'("{region_term}")'
    return f'{topic} AND {region}'


# Example mapping from region labels to PubMed region terms
REGION_TO_TERM = {
    "sub-saharan africa": "Africa South of the Sahara",
    "africa": "Africa",
    "nigeria": "Nigeria",
    "south asia": "Asia, Southern",
    "southeast asia": "Asia, Southeastern",
    "latin america": "Latin America",
    "north america": "North America",
    "western europe": "Europe",
    "high-income": "Developed Countries",
    "lmic": "Developing Countries",
}


# ------------------------------------------------------------------
# 2) Coverage table
# ------------------------------------------------------------------
def build_coverage_table(items: pd.DataFrame) -> pd.DataFrame:
    """
    Input: item DataFrame with at least item_id, topic_term, region_term.
    Output: a copy with pubmed_count and log_count columns added.
    Literature counts are heavily right-skewed (Zipf/power-law), so they are
    log-transformed as log10(count + 1), which also handles zero counts.
    """
    counts = []
    for _, row in items.iterrows():
        q = build_query(row["topic_term"], row["region_term"])
        c = pubmed_count(q)
        counts.append(c)
        time.sleep(MIN_INTERVAL)  # NCBI rate limit
    out = items.copy()
    out["pubmed_count"] = counts
    # Failed lookups (None) yield NaN
    out["log_count"] = out["pubmed_count"].apply(
        lambda x: math.log10(x + 1) if pd.notnull(x) else np.nan
    )
    return out


# ------------------------------------------------------------------
# 3) Merge with model results
# ------------------------------------------------------------------
def merge_with_results(coverage: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """
    `results` requires at least: item_id, model, correct (0/1), hallucinate (0/1),
    one row per item x model.
    Returns a long-format item x model table for analysis.
    """
    return results.merge(
        coverage[["item_id", "cancer", "type", "axis", "pubmed_count", "log_count"]],
        on="item_id", how="left"
    )


# ------------------------------------------------------------------
# 4) Analysis: correlation and mixed-effects logistic regression (H4)
# ------------------------------------------------------------------
def analyze_coverage(df: pd.DataFrame):
    """
    Test H4:
      (a) Spearman correlation between per-item accuracy and log_count.
      (b) Mixed-effects logistic regression: correct ~ log_count + model + (1|item_id),
          with a random intercept per item to absorb item difficulty.
    statsmodels' BinomialBayesMixedGLM uses variational inference; for
    frequentist confidence intervals and more stable convergence, R lme4 is
    preferred (see R_GLMM_SNIPPET).
    """
    from scipy.stats import spearmanr
    import statsmodels.formula.api as smf
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = df.dropna(subset=["log_count", "correct"]).copy()

    # (a) Per-item mean accuracy vs log_count
    per_item = d.groupby("item_id").agg(
        acc=("correct", "mean"), log_count=("log_count", "first")
    ).dropna()
    rho, p = spearmanr(per_item["log_count"], per_item["acc"])
    print(f"[Spearman] accuracy vs log10(pubmed_count): rho={rho:.3f}, p={p:.4g}")

    # (b) Mixed-effects logistic regression; random intercept for item_id via vc_formulas
    try:
        vc = {"item": "0 + C(item_id)"}
        model = BinomialBayesMixedGLM.from_formula(
            "correct ~ log_count + C(model)", vc_formulas=vc, data=d
        )
        res = model.fit_vb()
        print("\n[GLMM · BinomialBayesMixedGLM] posterior means:")
        print(res.summary())
    except Exception as e:
        print("Python GLMM failed; R lme4 is recommended. Error:", e)

    return per_item


# Equivalent model in R (lme4), recommended for confidence intervals and convergence.
R_GLMM_SNIPPET = r"""
# install.packages(c("lme4","emmeans"))
library(lme4)
df <- read.csv("merged_results.csv")
# Accuracy ~ coverage + model, random intercept per item
m4 <- glmer(correct ~ log_count + model + (1 | item_id),
            data = df, family = binomial,
            control = glmerControl(optimizer = "bobyqa"))
summary(m4)
# Same structure for hallucination:
mh <- glmer(hallucinate ~ log_count + model + (1 | item_id),
            data = df, family = binomial)
summary(mh)
"""


# ------------------------------------------------------------------
# 5) Visualization
# ------------------------------------------------------------------
def plot_coverage_vs_accuracy(per_item: pd.DataFrame, out_png="coverage_vs_accuracy.png"):
    """
    Scatter plot of per-item accuracy against log coverage with a linear fit.
    A positive slope (higher accuracy with more literature) is consistent with H4.
    """
    import matplotlib.pyplot as plt
    x = per_item["log_count"].values
    y = per_item["acc"].values
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)

    plt.figure(figsize=(6, 4.2))
    plt.scatter(x, y, alpha=0.6, edgecolor="k", linewidth=0.3)
    plt.plot(xs, np.polyval(coef, xs), linewidth=2)
    plt.xlabel("log10(PubMed count + 1)  — coverage proxy")
    plt.ylabel("Per-item accuracy (mean over models)")
    plt.title("Coverage proxy vs LLM accuracy (H4)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"saved: {out_png}  (slope={coef[0]:.3f})")


# ------------------------------------------------------------------
# 6) Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    # (1) Example items with topic_term/region_term defined per item.
    demo = pd.DataFrame([
        {"item_id": "Cx-01", "topic_term": "cervical cancer incidence", "region_term": "Africa South of the Sahara"},
        {"item_id": "Cx-02", "topic_term": "cervical cancer incidence", "region_term": "Europe"},
        {"item_id": "Cx-03", "topic_term": "human papillomavirus type 35", "region_term": "Africa"},
        {"item_id": "Em-02", "topic_term": "endometrial serous carcinoma disparities", "region_term": "African Americans"},
        {"item_id": "Ov-01", "topic_term": "BRCA1 founder mutation", "region_term": "Jews"},
    ])

    print(">>> building coverage table (live NCBI calls)...")
    cov = build_coverage_table(demo)
    print(cov[["item_id", "pubmed_count", "log_count"]])

    # (2) Merge with model results; simulated here for illustration.
    rng = np.random.default_rng(42)
    rows = []
    for _, r in cov.iterrows():
        for mdl in ["gpt", "claude", "gemini", "llama"]:
            # Simulated: probability of a correct answer increases with coverage
            p = 1 / (1 + math.exp(-(0.6 * (r["log_count"] - 3))))
            rows.append({"item_id": r["item_id"], "model": mdl,
                         "correct": int(rng.random() < p),
                         "hallucinate": int(rng.random() > p)})
    results = pd.DataFrame(rows)

    merged = merge_with_results(cov, results)
    merged.to_csv("merged_results.csv", index=False)
    per_item = analyze_coverage(merged)
    plot_coverage_vs_accuracy(per_item)

    print("\nSee R_GLMM_SNIPPET for the recommended R (lme4) model.")
