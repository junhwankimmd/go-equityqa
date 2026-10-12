# -*- coding: utf-8 -*-
"""
run_coverage.py: GO-EquityQA Arm C driver for the literature-coverage proxy (H4).

Pipeline: item CSV with topic_term/region_term; PubMed counts (coverage_proxy);
log10 transform; numeric validation of the hi/lo density labels; merge with the
scored JSON at the item x model level; coverage-outcome association (Spearman)
and figure.

Rationale:
  The pretraining corpora of closed LLMs cannot be inspected, so the volume of
  PubMed literature for each (topic x region) pair is used as a proxy for data
  density. Outputs are the per-item pubmed_count and a numeric check of whether
  the hi/lo labels agree with literature volume. Spearman correlations here are
  descriptive; confirmatory inference uses mixed-effects models.

Usage:
  python src/run_coverage.py
  # Uses the most recent results/scored_*.json by default; override with SCORED_JSON=...
Reproducibility: NCBI tool/email set via .env, retmax=0 (count-only queries),
log10(count + 1) transform, all intermediate tables saved.
"""
from __future__ import annotations
import os, sys, csv, json, glob, math, datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make src/ importable
import coverage_proxy as cp   # loads .env; provides pubmed_count/build_query

ITEMS_CSV  = os.getenv("ITEMS_CSV", "data/items_v3_300.csv")
SCORED_JSON = os.getenv("SCORED_JSON")   # default: most recent results/scored_*.json
OUT_DIR    = os.getenv("OUT_DIR", "results")


# ------------------------------------------------------------------
# 1) Coverage table: pubmed_count and log_count per item
# ------------------------------------------------------------------
def build_coverage(items):
    print(f"[coverage] Querying NCBI (email={cp.NCBI_EMAIL}, key={'Y' if cp.NCBI_API_KEY else 'N'})")
    rows = []
    for it in items:
        q = cp.build_query(it["topic_term"], it["region_term"])
        c = cp.pubmed_count(q)
        lc = math.log10(c + 1) if c is not None else None
        rows.append({**it, "pubmed_query": q, "pubmed_count": c, "log_count": lc})
        print(f"  {it['item_id']:9} {it.get('topic_density_label','?'):3} n={str(c):>7}  {q}")
        cp.time.sleep(cp.MIN_INTERVAL)
    return rows


# ------------------------------------------------------------------
# 2) Numeric validation of hi/lo labels
# ------------------------------------------------------------------
def validate_density(cov):
    # topic_density_label (topic-level hi/lo) is an auxiliary stratification variable;
    # the continuous count is the H4 predictor. This function checks only agreement
    # between the label and continuous coverage. Low counts on hi-labelled invariance
    # items with an African region term reflect a genuine coverage gap, not label error.
    hi = [r["pubmed_count"] for r in cov if r.get("topic_density_label") == "hi" and r["pubmed_count"] is not None]
    lo = [r["pubmed_count"] for r in cov if r.get("topic_density_label") == "lo" and r["pubmed_count"] is not None]
    def med(x): return sorted(x)[len(x)//2] if x else float("nan")
    # Rank-based separation, AUC = P(hi > lo); equals 1.0 when labels fully agree with coverage.
    pairs = [(h, l) for h in hi for l in lo]
    auc = sum((h > l) + 0.5*(h == l) for h, l in pairs) / len(pairs) if pairs else float("nan")
    print("\n===== Numeric validation of hi/lo labels =====")
    print(f"  hi(n={len(hi)}) median={med(hi):,}  range=[{min(hi):,}, {max(hi):,}]")
    print(f"  lo(n={len(lo)}) median={med(lo):,}  range=[{min(lo):,}, {max(lo):,}]")
    print(f"  Separation AUC P(hi>lo) = {auc:.3f}   (1.0 = perfect separation, 0.5 = none)")
    return {"hi_median": med(hi), "lo_median": med(lo), "auc": auc, "n_hi": len(hi), "n_lo": len(lo)}


# ------------------------------------------------------------------
# 3) Scored JSON to long table of outcomes (item x model)
# ------------------------------------------------------------------
def load_outcomes(scored_path):
    d = json.load(open(scored_path, encoding="utf-8"))
    pi = d["scored"]["per_item"]
    out = []   # {item_id, model, fabricated, refused, supported, acc_correct, reflects}
    for r in pi:
        rec = {"item_id": r["item_id"], "model": r["model"], "axis": r.get("axis"),
               "fabricated": None, "refused": None, "supported": None,
               "acc_correct": None, "reflects": None}
        if r["axis"] == "citation":
            rec["fabricated"] = int(bool(r.get("fabricated")))
            rec["refused"]    = int(bool(r.get("refused")))
            rec["supported"]  = int(bool(r.get("supported")))
        elif r["axis"] == "accuracy":
            rec["acc_correct"] = int(r.get("match") == "yes")
        elif r["axis"] == "erasure":
            rec["reflects"] = int(bool(r.get("reflects")))
        out.append(rec)
    return out, d.get("aggregate", {})


# ------------------------------------------------------------------
# 4) Merge and correlate (Spearman, standard-library implementation)
# ------------------------------------------------------------------
def _spearman(x, y):
    n = len(x)
    if n < 3: return None, None
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0.0]*n; i = 0
        while i < n:                      # average ranks for ties
            j = i
            while j+1 < n and v[order[j+1]] == v[order[i]]: j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j+1): rk[order[k]] = avg
            i = j+1
        return rk
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx)/n, sum(ry)/n
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a-mx)**2 for a in rx) * sum((b-my)**2 for b in ry))
    return (num/den if den else None), n


def merge_and_correlate(cov, outcomes):
    logc = {r["item_id"]: r["log_count"] for r in cov}
    dens = {r["item_id"]: r.get("topic_density_label") for r in cov}
    merged = []
    for o in outcomes:
        merged.append({**o, "log_count": logc.get(o["item_id"]), "topic_density_label": dens.get(o["item_id"])})

    # (H4a) Citation items: per-item fabrication rate (mean across models) vs log_count
    fab = defaultdict(list)
    for m in merged:
        if m["fabricated"] is not None and m["log_count"] is not None:
            fab[m["item_id"]].append(m["fabricated"])
    fab_items = sorted(fab)
    fx = [logc[i] for i in fab_items]
    fy = [sum(fab[i])/len(fab[i]) for i in fab_items]
    rho_f, n_f = _spearman(fx, fy)

    # (H4b) Accuracy items: per-item accuracy vs log_count (None if there is no variance)
    acc = defaultdict(list)
    for m in merged:
        if m["acc_correct"] is not None and m["log_count"] is not None:
            acc[m["item_id"]].append(m["acc_correct"])
    acc_items = sorted(acc)
    ax = [logc[i] for i in acc_items]
    ay = [sum(acc[i])/len(acc[i]) for i in acc_items]
    rho_a, n_a = _spearman(ax, ay)

    print("\n===== H4 coverage-outcome association (Spearman) =====")
    print(f"  [H4a fabrication] {len(fab_items)} citation items: Spearman(log_count, fabrication_rate) rho={rho_f} (n={n_f})")
    print(f"            expected sign: negative (lower coverage, higher fabrication)")
    for i in fab_items:
        print(f"             {i:9} log={logc[i]:.2f} fab_rate={sum(fab[i])/len(fab[i]):.2f} ({dens[i]})")
    if rho_a is None:
        print(f"  [H4b accuracy] {len(acc_items)} accuracy items: correlation undefined: no variance in accuracy")
    else:
        print(f"  [H4b accuracy] {len(acc_items)} accuracy items: rho={rho_a} (n={n_a}), expected sign: positive")
    return merged, {"h4a_fab_rho": rho_f, "h4a_n": n_f, "h4b_acc_rho": rho_a, "h4b_n": n_a,
                    "fab_items": fab_items, "fab_x": fx, "fab_y": fy}


# ------------------------------------------------------------------
# 5) Figure: (left) coverage by hi/lo label; (right) coverage vs fabrication rate
# ------------------------------------------------------------------
def make_figure(cov, corr, out_png):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Left: PubMed counts by hi/lo label (log-scale strip plot)
    for j, dsty in enumerate(["lo", "hi"]):
        ys = [r["pubmed_count"] for r in cov if r.get("topic_density_label") == dsty and r["pubmed_count"]]
        xs = [j + (k % 5 - 2) * 0.03 for k in range(len(ys))]
        axL.scatter(xs, ys, s=42, alpha=0.75,
                    color=("#c62828" if dsty == "lo" else "#90a4ae"), edgecolor="k", linewidth=0.3)
    axL.set_yscale("log"); axL.set_xticks([0, 1]); axL.set_xticklabels(["lo-label", "hi-label"])
    axL.set_ylabel("PubMed count (log scale)")
    axL.set_title("Coverage proxy validates hi/lo labels")
    for s in ["top", "right"]: axL.spines[s].set_visible(False)

    # Right: coverage vs per-item fabrication rate (citation items)
    fx, fy = corr["fab_x"], corr["fab_y"]
    axR.scatter(fx, fy, s=60, color="#c62828", edgecolor="k", linewidth=0.4)
    for x, y, lab in zip(fx, fy, corr["fab_items"]):
        axR.annotate(lab, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    if len(fx) >= 2:
        import numpy as np
        b = np.polyfit(fx, fy, 1); xs = np.linspace(min(fx), max(fx), 50)
        axR.plot(xs, np.polyval(b, xs), color="#455a64", lw=1.5, ls="--")
    axR.set_xlabel("log10(PubMed count + 1)"); axR.set_ylabel("per-item fabrication rate")
    axR.set_ylim(-0.05, 1.05); axR.set_title("H4: coverage vs citation fabrication")
    for s in ["top", "right"]: axR.spines[s].set_visible(False)

    fig.tight_layout(); fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print("saved:", out_png)


# ------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(ITEMS_CSV, encoding="utf-8") as f:
        items = list(csv.DictReader(f))
    miss = [it["item_id"] for it in items if not it.get("topic_term") or not it.get("region_term")]
    if miss:
        raise SystemExit(f"Missing topic_term/region_term for items: {miss}")
    print(f"Arm C coverage: {len(items)} items")

    cov = build_coverage(items)
    dens_stats = validate_density(cov)

    # Save coverage table
    cov_csv = f"{OUT_DIR}/coverage_table_{ts}.csv"
    keys = ["item_id", "cancer", "type", "axis", "probe_type", "topic_density_label",
            "topic_term", "region_term", "pubmed_query", "pubmed_count", "log_count"]
    with open(cov_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(cov)
    print("saved:", cov_csv)

    # Merge with scored outcomes, if available
    sp = SCORED_JSON or (sorted(glob.glob(f"{OUT_DIR}/scored_*.json"))[-1]
                         if glob.glob(f"{OUT_DIR}/scored_*.json") else None)
    corr = None
    if sp:
        print(f"\nUsing scored file: {sp}")
        outcomes, _ = load_outcomes(sp)
        merged, corr = merge_and_correlate(cov, outcomes)
        mrg_csv = f"{OUT_DIR}/merged_coverage_{ts}.csv"
        with open(mrg_csv, "w", encoding="utf-8", newline="") as f:
            fields = ["item_id", "model", "axis", "topic_density_label", "log_count",
                      "fabricated", "refused", "supported", "acc_correct", "reflects"]
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(merged)
        print("saved:", mrg_csv, "(long format for mixed-effects models)")
        make_figure(cov, corr, f"{OUT_DIR}/fig_coverage_{ts}.png")
    else:
        print("No scored_*.json found; writing coverage table only (outcome merge skipped).")

    json.dump({"density_validation": dens_stats, "h4_directional": corr and {
                 k: corr[k] for k in ("h4a_fab_rho", "h4a_n", "h4b_acc_rho", "h4b_n")}},
              open(f"{OUT_DIR}/coverage_summary_{ts}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nDone. Outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
