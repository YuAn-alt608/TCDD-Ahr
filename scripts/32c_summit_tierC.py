"""32c - Tier C with a summit test instead of a window mean.

Why a second signal test
------------------------
32b asked whether the *mean* signal over TSS +-2 kb clears the 95th percentile of
random TSS windows. Two things can make that test unfair in opposite directions:

  * a real summit that sits 3-8 kb from the annotated TSS is averaged away, and
  * a broad elevated domain (low mappability, copy number, hyper-ChIPable region)
    can push the mean up without any summit at all.

Both happen here: in GSE276152 the highest bin near AHRR is at +22.9 kb with signal
109, yet the processed narrowPeak of the same experiment contains no peak anywhere
within +-35 kb. So the mean and the peak list disagree, and neither alone decides.

This script adds the missing view: for each target take the **maximum 500 bp bin**
within +-10 kb (a summit proxy, robust to where exactly the TSS is annotated) and
compare it with the distribution of the same statistic over N random TSS windows in
the same track. It also reports where that summit sits, which is what separates
"bound at the promoter" from "bound 20 kb away".

Outputs (results/chip_alt/):
  summit_tierC.csv     dataset x target: summit, offset, percentile, pass
  summit_summary.json
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L
from lib_bigwig import BigWig

OUT = os.path.join(L.RES, "chip_alt")
RAW = os.path.join(L.ROOT, "data", "external", "chip_alt")

TRACKS = [
    ("GSE304148_AHR_ChIP_SKMEL2_control", "SKMEL2_AHR_Control_1.bw"),
    ("GSE304148_AHR_ChIP_SKMEL2_AHROE", "SKMEL2_AHR_AHROE_1.bw"),
    ("GSE240495_AHR_CUTRUN_K562", "K562_AHR_CUTRUN_WTF.bw"),
    ("GSE276152_AHR_ChIP_HCASMC_TCDD", "HCASMC_AHR_TCDD.bw"),
    ("GSE276152_AHR_ChIP_HCASMC_DMSO", "HCASMC_AHR_DMSO.bw"),
]
CANON = ["CYP1A1", "CYP1B1", "AHRR", "TIPARP", "NQO1", "AHR"]
HALF = 10000
BIN = 500
NBG = 500


def main():
    rf = pd.read_csv(os.path.join(L.ROOT, "data", "external", "hg38_refFlat.txt.gz"),
                     sep="\t", header=None,
                     names=["gene", "name", "chrom", "strand", "txStart", "txEnd",
                            "a", "b", "c", "d", "e", "f", "g", "h", "i"])
    tss = rf.groupby("gene").agg(chrom=("chrom", "first"),
                                 tss=("txStart", "min")).reset_index()
    up = tss.gene.astype(str).str.upper()
    d = dict(zip(up, zip(tss.chrom, tss.tss)))
    targets = [(g, d[g][0], int(d[g][1])) for g in CANON if g in d]

    rng = np.random.default_rng(0)
    rows = []
    for label, fname in TRACKS:
        p = os.path.join(RAW, fname)
        if not os.path.exists(p):
            print(f"  {fname} missing, skipped", flush=True)
            continue
        print(f"  {fname}", flush=True)
        bw = BigWig(p)
        okchrom = {g: (c, t) for g, c, t in targets if c in bw.chroms}
        pooled = []
        for g, c, t in targets:
            if c not in bw.chroms:
                continue
            offs = np.arange(-HALF, HALF, BIN)
            v = np.array([bw.window_mean(c, t + o, t + o + BIN) for o in offs], float)
            if not np.isfinite(v).any():
                continue
            i = int(np.nanargmax(v))
            pooled.append((g, c, t, float(v[i]), int(offs[i]), v))
        # null: same statistic at random TSS windows
        cand = tss[tss.chrom.isin(bw.chroms)]
        pick = cand.iloc[rng.choice(len(cand), min(NBG, len(cand)), replace=False)]
        null = []
        for _, r in pick.iterrows():
            t0 = int(r.tss)
            v = np.array([bw.window_mean(r.chrom, t0 + o, t0 + o + BIN)
                          for o in range(-HALF, HALF, BIN)], float)
            if np.isfinite(v).any():
                null.append(float(np.nanmax(v)))
        null = np.array(null, float)
        if len(null) < 50:
            continue
        for g, c, t, s, off, _v in pooled:
            pct = float((null < s).mean())
            rows.append(dict(dataset=label, gene=g, chrom=c, tss=t,
                             summit=s, offset=off, null_p95=float(np.percentile(null, 95)),
                             pct=pct, pass_summit=bool(pct >= 0.95)))
        n = int(sum(r["pass_summit"] for r in rows if r["dataset"] == label))
        print(f"    {label}: {n}/6 targets with a summit above the null p95 "
              f"(null n={len(null)})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "summit_tierC.csv"), index=False)
    sm = {}
    for ds, g in df.groupby("dataset"):
        sm[ds] = {"n_pass": int(g.pass_summit.sum()),
                  "genes": g[g.pass_summit].gene.tolist()}
    json.dump({"test": f"max {BIN} bp bin within TSS +-{HALF} bp vs {NBG} random TSS windows",
               "gate": "PASS = >=2 of 6 targets with summit >= 95th percentile",
               "result": sm}, open(os.path.join(OUT, "summit_summary.json"), "w"), indent=2)
    print("\n=== summit Tier C ===")
    print(df.round(3).to_string(index=False))
    print("\n" + json.dumps(sm, indent=2))


if __name__ == "__main__":
    main()
