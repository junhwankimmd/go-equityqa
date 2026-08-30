# -*- coding: utf-8 -*-
"""
GO-EquityQA · Arm C — Coverage-Proxy Pipeline
=============================================
Purpose (WHY):
  The pretraining corpora of closed LLMs cannot be inspected. We therefore proxy
  "how much data the world holds about a (topic x region)" with the **PubMed
  record count**, and then examine `accuracy ~ log10(count)` and
  `hallucination ~ log10(count)`.
  Rationale: a language model's factual ability correlates strongly with the
  number of relevant documents in pretraining (Kandpal et al., ICML 2023), and
  QA accuracy tracks entity frequency/popularity (Mallen et al., ACL 2023);
  hallucination increases at low frequency.

Environment: Google Colab (paid) or local. Standard library + requests + pandas
+ statsmodels + matplotlib.
Install: !pip install requests pandas numpy statsmodels matplotlib

Care (SAFETY):
  - NCBI E-utilities etiquette: declare tool/email; 3 req/s without an API key,
    10 req/s with one.
  - A record *count* is an approximate proxy. Query design drives the result, so
    define each item's topic_term / region_term with clinical care (see
    build_query below).
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

# Auto-load .env if present (same convention as the sibling citation_verifier /
# runner modules). Existing environment variables are not overwritten.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

# ------------------------------------------------------------------
# 0) Config (NCBI etiquette: declare tool/email; reuses NCBI_EMAIL/NCBI_API_KEY from .env)
# ------------------------------------------------------------------
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "your_email@example.com")
NCBI_TOOL = "GO-EquityQA-coverage"
NCBI_API_KEY: Optional[str] = os.getenv("NCBI_API_KEY")   # 10 req/s with a key, 3 req/s without
ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# Minimum interval between requests (s): 0.11 (~9/s) with a key, 0.34 (~3/s) without.
MIN_INTERVAL = 0.11 if NCBI_API_KEY else 0.34


# ------------------------------------------------------------------
# 1) PubMed count lookup (proxy core)
# ------------------------------------------------------------------
def pubmed_count(query: str, retries: int = 4) -> Optional[int]:
    """
    Return the result *count* for a PubMed query (no documents fetched, retmax=0).
    WHY: only the count is needed, so this is the lightest possible call.
    Retries with exponential backoff on failure.
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
            # 429/5xx -> backoff
            time.sleep(1.5 * (i + 1))
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None  # failures return None -> treated as missing downstream


def build_query(topic_term: str, region_term: str) -> str:
    """
    Build the (topic x region) query.
    Design principles (WHY):
      - The topic term pins the disease/domain (e.g., "cervical cancer incidence").
      - The region term is a country/region (e.g., "Nigeria", "Africa",
        "sub-Saharan Africa").
      - Too narrow -> zero hits; too broad -> diluted signal. Tune clinically.
    Add [Title/Abstract], [MeSH], or year filters as needed for precision.
    """
    topic = f'("{topic_term}")'
    region = f'("{region_term}")'
    return f'{topic} AND {region}'


# Example mapping from region labels to PubMed terms (extend per project)
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
# 2) Build the coverage table
# ------------------------------------------------------------------
def build_coverage_table(items: pd.DataFrame) -> pd.DataFrame:
    """
    Input: item DataFrame with at least: item_id, topic_term, region_term.
      (The seed CSV lacks topic_term/region_term; derive/augment as in the demo.)
    Output: adds pubmed_count and log_count columns.
    WHY the log transform: literature volume is Zipf/power-law skewed; log10
    linearizes it so the regression is interpretable. log1p-style (+1) guards
    against count=0.
    """
    counts = []
    for _, row in items.iterrows():
        q = build_query(row["topic_term"], row["region_term"])
        c = pubmed_count(q)
        counts.append(c)
        time.sleep(MIN_INTERVAL)  # NCBI etiquette: rate limit
    out = items.copy()
    out["pubmed_count"] = counts
    # Missing values (None) are dropped or handled before analysis
    out["log_count"] = out["pubmed_count"].apply(
        lambda x: math.log10(x + 1) if pd.notnull(x) else np.nan
    )
    return out


# ------------------------------------------------------------------
# 3) Merge with performance data
# ------------------------------------------------------------------
def merge_with_results(coverage: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """
    results columns (minimum): item_id, model, correct(0/1), hallucinate(0/1)
      - the Arm A/B run records organized per item.
    Returns: long-format analysis table at item x model level.
    """
    return results.merge(
        coverage[["item_id", "cancer", "type", "axis", "pubmed_count", "log_count"]],
        on="item_id", how="left"
    )


# ------------------------------------------------------------------
# 4) Analysis: correlation + mixed-effects logistic (H4)
# ------------------------------------------------------------------
def analyze_coverage(df: pd.DataFrame):
    """
    H4 tests:
      (a) Spearman correlation (aggregated accuracy vs log_count) — quick check
          of monotonic association.
      (b) Mixed-effects logistic: correct ~ log_count + model + (1|item_id),
          absorbing item difficulty as a random effect.
    Python GLMM caveat (WHY R is provided as well):
      statsmodels' BinomialBayesMixedGLM uses variational inference — handy, but
      R lme4 has better frequentist CIs and convergence stability (see the R
      snippet below).
    """
    from scipy.stats import spearmanr
    import statsmodels.formula.api as smf
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = df.dropna(subset=["log_count", "correct"]).copy()

    # (a) per-item mean accuracy vs log_count
    per_item = d.groupby("item_id").agg(
        acc=("correct", "mean"), log_count=("log_count", "first")
    ).dropna()
    rho, p = spearmanr(per_item["log_count"], per_item["acc"])
    print(f"[Spearman] accuracy vs log10(pubmed_count): rho={rho:.3f}, p={p:.4g}")

    # (b) mixed-effects logistic (random intercept: item_id)
    #     random effects are specified via vc_formula
    try:
        vc = {"item": "0 + C(item_id)"}
        model = BinomialBayesMixedGLM.from_formula(
            "correct ~ log_count + C(model)", vc_formulas=vc, data=d
        )
        res = model.fit_vb()
        print("\n[GLMM · BinomialBayesMixedGLM] posterior means:")
        print(res.summary())
    except Exception as e:
        print("Python GLMM failed -> use R lme4 instead. err:", e)

    return per_item


# R alternative (recommended; stable convergence/CIs) — run via rpy2 in Colab or in R directly:
R_GLMM_SNIPPET = r"""
# install.packages(c("lme4","emmeans"))
library(lme4)
df <- read.csv("merged_results.csv")
# accuracy ~ coverage + model, random intercept per item
m4 <- glmer(correct ~ log_count + model + (1 | item_id),
            data = df, family = binomial,
            control = glmerControl(optimizer = "bobyqa"))
summary(m4)
# hallucination with the same structure:
mh <- glmer(hallucinate ~ log_count + model + (1 | item_id),
            data = df, family = binomial)
summary(mh)
"""


# ------------------------------------------------------------------
# 5) Plot
# ------------------------------------------------------------------
def plot_coverage_vs_accuracy(per_item: pd.DataFrame, out_png="coverage_vs_accuracy.png"):
    """
    Scatter + linear fit. Interpretation (WHY): an upward slope means "more data,
    more accurate" -> supports H4.
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
# 6) Usage example (DEMO)
# ------------------------------------------------------------------
if __name__ == "__main__":
    # (1) Example of augmenting the seed CSV with topic_term/region_term.
    #     In practice, define these clinically per item and store them as columns.
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

    # (2) Merge with performance records (Arm A/B) — dummy example here.
    #     Replace with real model-run results.
    rng = np.random.default_rng(42)
    rows = []
    for _, r in cov.iterrows():
        for mdl in ["gpt", "claude", "gemini", "llama"]:
            # demo only: higher coverage -> higher accuracy probability (replace with real data)
            p = 1 / (1 + math.exp(-(0.6 * (r["log_count"] - 3))))
            rows.append({"item_id": r["item_id"], "model": mdl,
                         "correct": int(rng.random() < p),
                         "hallucinate": int(rng.random() > p)})
    results = pd.DataFrame(rows)

    merged = merge_with_results(cov, results)
    merged.to_csv("merged_results.csv", index=False)
    per_item = analyze_coverage(merged)
    plot_coverage_vs_accuracy(per_item)

    print("\nSee the R_GLMM_SNIPPET variable for the recommended R (lme4) code.")
