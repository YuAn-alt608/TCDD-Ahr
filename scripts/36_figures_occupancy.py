"""36 - Figure for the ring-5 dry experiment (report section 10.11).

Three panels, because the negative only means something next to the two things
that make it credible:

  (a) the same statistic does fire inside the same track - AHRR and CYP1A1 climb
      to the 99th / 89th percentile of random TSS windows on the ligand delta,
      so the probe is not blind;
  (b) at the bins that open in mouse SMC, TCDD occupancy is indistinguishable
      from matched background and the paired ligand delta is, if anything,
      lower - and it gets more clearly lower once poorly covered bins are
      dropped (the zero-fill sensitivity check);
  (c) but the open bins are systematically better covered than background, which
      is why any positive readout would have needed this panel to rule out
      "mappability" - shown here so the reader can see the confound was looked
      for, not missed.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)

CA = os.path.join(RES, "chip_alt")
t = pd.read_csv(os.path.join(CA, "occupancy_tests.csv"))
pc = pd.read_csv(os.path.join(CA, "occupancy_positive_control.csv"))
b = pd.read_csv(os.path.join(CA, "occupancy_bins.csv"))

RED, GREY, BLUE = "#c0392b", "#7f8c8d", "#2c6fbb"

fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.3))

# ---------------------------------------------------------------- (a) control
g = pc.sort_values("pct_delta_summit")
y = np.arange(len(g))
ax[0].barh(y, g.pct_delta_summit, color=[RED if v >= 90 else GREY for v in g.pct_delta_summit],
           height=.6)
ax[0].axvline(95, ls="--", c="k", lw=1)
ax[0].set_yticks(y)
ax[0].set_yticklabels(g.gene, fontsize=9)
ax[0].set_xlim(0, 112)
ax[0].set_xticks([0, 25, 50, 75, 100])
ax[0].set_xlabel("percentile of TCDD−DMSO summit vs 1,000 random TSS")
ax[0].set_title("(a) positive control, HCASMC TCDD vs DMSO", fontsize=10)
ax[0].text(96, -0.9, "p95", fontsize=7)
for i, v in enumerate(g.pct_delta_summit):
    ax[0].text(v + 1, i, f"{v:.0f}", va="center", fontsize=7)

# ------------------------------------------------------- (b) the actual test
rows = [
    ("TCDD", "SMC_open", "HCASMC_TCDD_mean"),
    ("DMSO", "SMC_open", "HCASMC_DMSO_mean"),
    ("Δ TCDD−DMSO", "SMC_open", "ligand_delta_mean"),
    ("Δ mean\ncov≥.5", "SMC_open[cov>=.5]", "ligand_delta_mean"),
    ("Δ summit\ncov≥.5", "SMC_open[cov>=.5]", "ligand_delta_summit"),
]
labs, zs, ps = [], [], []
for lab, s, stat in rows:
    r = t[(t.set == s) & (t.statistic == stat)]
    if not len(r):
        continue
    labs.append(lab); zs.append(float(r.z.iloc[0])); ps.append(float(r.p.iloc[0]))
x = np.arange(len(labs))
ax[1].bar(x, zs, color=[RED if v > 0 else BLUE for v in zs], width=.6)
ax[1].axhline(0, c="k", lw=.8)
ax[1].axhline(1.96, ls=":", c=GREY)
ax[1].axhline(-1.96, ls=":", c=GREY)
for i, (v, p) in enumerate(zip(zs, ps)):
    ax[1].text(i, v + (0.12 if v >= 0 else -0.22), f"p={p:.2f}", ha="center", fontsize=7)
ax[1].set_xticks(x)
ax[1].set_xticklabels(labs, fontsize=7.5, rotation=18, ha='right')
ax[1].set_ylabel("z vs GC/TSS-matched background")
ax[1].set_title("(b) SMC-open bins, human SMC AHR ChIP", fontsize=10)
ax[1].set_ylim(min(-2.9, min(zs) - .6), max(2.9, max(zs) + .6))

# ------------------------------------------------------------- (c) confound
smc = b[b.sets.str.contains("SMC_open")]
bg = b[b.sets == "background"]
ax[2].hist(bg.HCASMC_TCDD_cov, bins=25, density=True, alpha=.5, color=GREY,
           label=f"background (n={len(bg)})")
ax[2].hist(smc.HCASMC_TCDD_cov, bins=25, density=True, alpha=.5, color=RED,
           label=f"SMC open (n={len(smc)})")
ax[2].set_xlabel("fraction of 5-kb window with any TCDD read")
ax[2].set_ylabel("density")
ax[2].legend(fontsize=7, frameon=False)
ax[2].set_title("(c) the confound that had to be checked", fontsize=10)
ax[2].text(.02, .95, f"median {smc.HCASMC_TCDD_cov.median():.2f} vs "
                     f"{bg.HCASMC_TCDD_cov.median():.2f}",
           transform=ax[2].transAxes, fontsize=7, va="top")

fig.tight_layout()
p = os.path.join(OUT, "fig18_occupancy.png")
fig.savefig(p, dpi=190)
print("wrote", p)
