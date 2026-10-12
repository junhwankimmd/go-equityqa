#!/usr/bin/env python
"""Figures 2 and 3 of the manuscript.
Figure 2 (provenance): a coverage-vs-support-failure scatter; b definition-sensitivity tertiles;
                       c model safety spectrum; d per-item citation-category composition by coverage.
Figure 3 (behavior):   nationality-driven invariance violation, resource cue absent vs stated; erasure.
Also writes results/supp_S1_category_counts.csv (raw counts for Supplementary Table S1).
Output: results/fig2_provenance.{png,pdf}, results/fig3_behavior.{png,pdf} (300 dpi + vector)."""
import csv, json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":"Arial","font.size":8,"axes.titlesize":8,"axes.labelsize":8,
    "xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":6.5,
    "axes.linewidth":0.8,"lines.linewidth":1.2,"figure.dpi":300,"savefig.dpi":300,
    "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42})
OI={"black":"#000000","orange":"#E69F00","sky":"#56B4E9","green":"#009E73",
    "yellow":"#F0E442","blue":"#0072B2","vermillion":"#D55E00","purple":"#CC79A7","grey":"#999999"}

# ---------- data: citation categories per item x model + coverage ----------
d=json.load(open("results/scored_final_adjudicated_20260829.json",encoding="utf-8"))  # dual-judge + human adjudication
cit=d["per_item"]
cov={(r["item_id"],r["model"]):float(r["log_count"])
     for r in csv.DictReader(open("results/merged_citefail_confirmatory.csv"))}

STRICT={"MISATTRIBUTED_WRONGPAPER","FABRICATED_NONEXISTENT"}
BROAD_ONLY={"UNSUPPORTED_ONTOPIC","PARTIAL"}
def item_class(cats):
    s=set(cats)
    if s & STRICT: return "Misattributed or fabricated"
    if s & BROAD_ONLY: return "On-topic but unsupported"
    if "SUPPORTED" in s: return "Supported"
    return "No citation or refusal"

rows=[]
for r in cit:
    cats=[v.get("category") for v in (r.get("detail") or [])]
    lc=cov.get((r["item_id"],r["model"]))
    if lc is None: continue
    rows.append((r["model"], lc, item_class(cats)))

lcs=sorted(set(x[1] for x in rows))
t1,t2=np.quantile([x[1] for x in rows],[1/3,2/3])
def tert(lc): return 0 if lc<=t1 else (1 if lc<=t2 else 2)
TLAB=["Low","Mid","High"]
ORDER=["Supported","On-topic but unsupported","Misattributed or fabricated","No citation or refusal"]
COL={"Supported":OI["green"],"On-topic but unsupported":OI["orange"],
     "Misattributed or fabricated":OI["vermillion"],"No citation or refusal":OI["blue"]}

comp=np.zeros((3,4)); tot=np.zeros(3)
for m,lc,cl in rows:
    ti=tert(lc); comp[ti][ORDER.index(cl)]+=1; tot[ti]+=1
frac=comp/tot[:,None]

# Supplementary S1: counts by model x tertile x category
S1=collections.Counter()
for m,lc,cl in rows: S1[(m,TLAB[tert(lc)],cl)]+=1
with open("results/supp_S1_category_counts.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["model","coverage_tertile","category","n_items"])
    for (m,t,cl),n in sorted(S1.items()): w.writerow([m,t,cl,n])
# per-tertile shares under the worst-category composition (Figure 2b, Table S1)
for ti in range(3):
    strict=frac[ti][ORDER.index("Misattributed or fabricated")]
    broad=strict+frac[ti][ORDER.index("On-topic but unsupported")]
    print(f"tertile {TLAB[ti]}: misattributed={strict:.2f} misattributed+on-topic-unsupported={broad:.2f} n={int(tot[ti])}")

# summary values, deterministic item-level coverage-quantile tertiles
# confirmatory: merged_{coverage,citefail}_confirmatory.csv; exploration: 48 items (40 @20260711_165334 + 8 @expl8_rejudge_20260826)
CONF_BROAD=[0.67,0.60,0.26]; HELD_BROAD=[0.74,0.71,0.29]; STRICT3=[0.36,0.34,0.29]
MODELS=["GPT-5.5","Claude\nOpus 4.8","Gemini\n3.1 Pro"]
FAB=[0.32,0.24,0.44]; REF=[0.00,0.48,0.00]; SUP=[0.32,0.25,0.17]  # adjudicated

# ================= FIGURE 2 (provenance) =================
# panels ordered by first citation in the text: a=tertile (strict grey line), b=composition, c=scatter, d=models
fig,axs=plt.subplots(2,2,figsize=(7.2,6.0)); (b,e),(a,c)=axs
# a: scatter
byitem=collections.defaultdict(list); icov={}
for r in cit:
    lc=cov.get((r["item_id"],r["model"]))
    if lc is None: continue
    byitem[r["item_id"]].append(1 if r.get("citefail") else 0)
    icov[r["item_id"]]=lc
xs=[icov[i] for i in byitem]; ys=[sum(v)/len(v) for i,v in byitem.items()]
a.scatter(xs,ys,s=18,color=OI["blue"],alpha=0.45,edgecolors="none")
m_,b_=np.polyfit(xs,ys,1); xl=np.array([min(xs),max(xs)]); a.plot(xl,m_*xl+b_,color=OI["black"],ls="--")
a.set_xlabel("log$_{10}$(PubMed coverage + 1)"); a.set_ylabel("Citation-support failure rate"); a.set_ylim(-.05,1.05)
a.text(.5,1.14,r"Mixed-effects $\beta$ = -1.12, $P$ = 2×10$^{-8}$; Spearman -0.54",transform=a.transAxes,ha="center",va="top",fontsize=7)
# b: definition sensitivity
x=np.arange(3); w=.36
b.bar(x-w/2,CONF_BROAD,w,color=OI["blue"],label="Citation-support failure (confirmatory)")
b.bar(x+w/2,HELD_BROAD,w,color=OI["orange"],label="Citation-support failure (exploration-only)")
b.plot(x,STRICT3,color=OI["grey"],ls="--",marker="o",ms=4,label="Strict fabrication (confirmatory)")
b.set_xticks(x); b.set_xticklabels(TLAB); b.set_xlabel("Coverage tertile"); b.set_ylabel("Rate"); b.set_ylim(0,1)
b.legend(frameon=False,loc="upper right",handlelength=1.4,fontsize=5.8)
# c: model spectrum
x=np.arange(3); w=.26
for bc in (c.bar(x-w,FAB,w,color=OI["vermillion"],label="Strict fabrication"), c.bar(x,REF,w,color=OI["blue"],label="Refusal (safe)"), c.bar(x+w,SUP,w,color=OI["green"],label="Support")):
    c.bar_label(bc,fmt="%.2f",fontsize=5.8,padding=1.5)  # exact values on bars
c.set_xticks(x); c.set_xticklabels(MODELS); c.set_ylabel("Per-item rate"); c.set_ylim(0,.6); c.legend(frameon=False,loc="upper left")
# d: category composition by coverage (stacked)
x=np.arange(3); bottom=np.zeros(3)
for cl in ORDER:
    vals=frac[:,ORDER.index(cl)]
    e.bar(x,vals,.55,bottom=bottom,color=COL[cl],label=cl); bottom+=vals
e.set_xticks(x); e.set_xticklabels(TLAB); e.set_xlabel("Coverage tertile"); e.set_ylabel("Fraction of items"); e.set_ylim(0,1)
e.legend(frameon=False,loc="lower center",bbox_to_anchor=(.5,1.03),ncol=2,fontsize=6,handlelength=1.2,columnspacing=1.0)
for ax,lab in zip([b,e,a,c],"abcd"):
    ax.text(-.14,1.05,lab,transform=ax.transAxes,fontsize=11,fontweight="bold",va="bottom",ha="right")
fig.tight_layout(w_pad=2.2,h_pad=3.2)
fig.savefig("results/fig2_provenance.png",bbox_inches="tight"); fig.savefig("results/fig2_provenance.pdf",bbox_inches="tight")

# ================= FIGURE 3 (behavior: invariance + erasure) =================
INV_NOCUE=[0.11,0.67,0.56]; INV_STATED=[0.00,0.13,0.00]   # GPT, Claude, Gemini
ERA_REFLECT=[0.975,1.00,1.00]; ERA_DEFAULT=[0.025,0.00,0.00] # GPT, Claude, Gemini (39/40, 1/40)
fig3,(axA,axB)=plt.subplots(1,2,figsize=(7.2,3.3))
x=np.arange(3); w=.36
for bc in (axA.bar(x-w/2,INV_NOCUE,w,color=OI["purple"],label="Resource setting unstated"),
           axA.bar(x+w/2,INV_STATED,w,color=OI["grey"],label="Fully-resourced setting stated")):
    axA.bar_label(bc,fmt="%.2f",fontsize=6,padding=1.5)
axA.set_xticks(x); axA.set_xticklabels(MODELS); axA.set_ylabel("Invariance-violation rate"); axA.set_ylim(0,.8)
axA.legend(frameon=False,loc="lower center",bbox_to_anchor=(0.5,-0.5),ncol=1,fontsize=6.5)
axA.set_title("Invariance items",fontsize=8,pad=4)
for bc,vals in ((axB.bar(x-w/2,ERA_REFLECT,w,color=OI["green"],label="Reflects legitimate difference"),ERA_REFLECT),
                (axB.bar(x+w/2,ERA_DEFAULT,w,color=OI["vermillion"],label="Defaults to Western answer"),ERA_DEFAULT)):
    axB.bar_label(bc,labels=[f"{round(v+1e-9,2):.2f}" for v in vals],fontsize=6,padding=1.5)  # half-up: 0.975 -> 0.98 as in text
axB.set_xticks(x); axB.set_xticklabels(MODELS); axB.set_ylabel("Rate"); axB.set_ylim(0,1.12)
axB.legend(frameon=False,loc="lower center",bbox_to_anchor=(0.5,-0.5),fontsize=6.5)
axB.set_title("Erasure items",fontsize=8,pad=4)
for ax,lab in zip([axA,axB],"ab"):
    ax.text(-.17,1.06,lab,transform=ax.transAxes,fontsize=11,fontweight="bold",va="bottom",ha="right")
fig3.tight_layout(w_pad=2.6)
fig3.savefig("results/fig3_behavior.png",bbox_inches="tight"); fig3.savefig("results/fig3_behavior.pdf",bbox_inches="tight")
print("saved fig2_provenance.{png,pdf} (4 panels), fig3_behavior.{png,pdf} (2 panels), supp_S1_category_counts.csv")
