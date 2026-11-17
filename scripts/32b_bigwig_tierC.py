"""32b - Tier C gate applied to datasets that deposit only bigWig.

Two of the nine AHR series found by scripts/32_alt_chip.py publish signal tracks
instead of called peaks:

  GSE304148  AHR ChIP-seq, human melanoma SKMEL2, AHR-overexpression vs control
  GSE240495  AHR CUT&RUN, human K562

Neither can be turned into a peak set here (no peak caller, no pyBigWig), so the
gate is expressed as a signal question instead of a peak question:

  for each of the six canonical TCDD targets, take the mean signal over TSS +-2 kb
  and +-10 kb, and ask whether it exceeds the 95th percentile of 3,000 random TSS
  windows in the same track.

  PASS  >= 2 of 6 targets above that percentile at +-2 kb
  weak  1 of 6, or >= 2 of 6 only at +-10 kb
  FAIL  none

This is weaker than a peak call - it cannot distinguish a broad domain of moderate
signal from a real summit - and the report says so. It is still enough to answer the
question that matters: does *any* public AHR occupancy track show these six genes as
bound?

Outputs (results/chip_alt/):
  bigwig_tierC.csv      per dataset x target x window: signal and percentile
  bigwig_summary.json
"""
import os, sys, json, ssl, urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L
from lib_bigwig import BigWig

OUT = os.path.join(L.RES, "chip_alt")
RAW = os.path.join(L.ROOT, "data", "external", "chip_alt")
os.makedirs(RAW, exist_ok=True)
CTX = ssl.create_default_context()
UA = {"User-Agent": "Mozilla/5.0"}
CANON = ["CYP1A1", "CYP1B1", "AHRR", "TIPARP", "NQO1", "AHR"]
WINS = {"2kb": 2_000, "10kb": 10_000}
NRAND = 3000

SAMPLES = [
    ("GSE304148_AHR_ChIP_SKMEL2_control", "GSM9144359", "GSM9144359_ChIP_SKMEL2_Control_1.last.bw",
     "SKMEL2_AHR_Control_1.bw"),
    ("GSE304148_AHR_ChIP_SKMEL2_AHROE", "GSM9144362", "GSM9144362_ChIP_SKMEL2_AHROE_1.last.bw",
     "SKMEL2_AHR_AHROE_1.bw"),
    ("GSE240495_AHR_CUTRUN_K562", "GSM7698849", "GSM7698849_K562_WTF_raw.rmDup.bw",
     "K562_AHR_CUTRUN_WTF.bw"),
    # the vascular SMC dataset the paper's earlier negative came from, put through the
    # *same* signal test so the comparison is apples to apples
    ("GSE276152_AHR_ChIP_HCASMC_TCDD", "GSM8492045", "GSM8492045_AHR_pos.bw",
     "HCASMC_AHR_TCDD.bw"),
    ("GSE276152_AHR_ChIP_HCASMC_DMSO", "GSM8492044", "GSM8492044_AHR_neg.bw",
     "HCASMC_AHR_DMSO.bw"),
]


def sample_url(gsm, fname):
    n = gsm[3:]
    return f"https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM{n[:-3]}nnn/{gsm}/suppl/{fname}"


def fetch(gsm, fname, dest, cap_mb=700):
    p = os.path.join(RAW, dest)
    if os.path.exists(p) and os.path.getsize(p) > 1e6:
        return p
    url = sample_url(gsm, fname)
    try:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
            size = int(r.headers.get("Content-Length", 0))
        if size > cap_mb * 1024 * 1024:
            print(f"  skip {dest}: {size/1e6:.0f} MB", flush=True)
            return None
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600,
                                    context=CTX) as r, open(p, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        return p
    except Exception as e:
        print(f"  {dest}: {e}", flush=True)
        return None


def main():
    # hg38 TSS
    rf = pd.read_csv(os.path.join(L.ROOT, "data", "external", "hg38_refFlat.txt.gz"),
                     sep="\t", header=None)
    rf = rf[[0, 2, 3, 4]]
    rf.columns = ["gene", "chrom", "strand", "txStart"]
    rf = rf[rf.chrom.isin([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])]
    tss = rf.groupby("gene").agg(chrom=("chrom", "first"),
                                 tss=("txStart", "min")).reset_index()
    up = pd.Series(tss.gene.astype(str).str.upper().to_numpy(), index=tss.index)
    keep, seen = [], set()
    for g in CANON:
        hit = np.where(up.to_numpy() == g)[0]
        if len(hit):
            keep.append(int(hit[0]))
            seen.add(g)
    rng = np.random.default_rng(0)
    rand_idx = rng.choice(len(tss), min(NRAND, len(tss)), replace=False)

    rows = []
    for label, gsm, fname, dest in SAMPLES:
        p = fetch(gsm, fname, dest)
        if p is None:
            continue
        print(f"  reading {dest}", flush=True)
        bw = BigWig(p)
        for wname, win in WINS.items():
            def sig(i):
                return bw.window_mean(tss.chrom.iloc[i], int(tss.tss.iloc[i]) - win,
                                      int(tss.tss.iloc[i]) + win)
            bg = np.array([sig(i) for i in rand_idx], float)
            bg = bg[np.isfinite(bg)]
            if len(bg) < 100:
                continue
            q95 = float(np.percentile(bg, 95))
            for i in keep:
                v = sig(i)
                rows.append(dict(dataset=label, window=wname,
                                 gene=tss.gene.iloc[i], signal=None if not np.isfinite(v) else v,
                                 bg_median=float(np.median(bg)), bg_p95=q95,
                                 pct=float((bg < v).mean()) if np.isfinite(v) else np.nan,
                                 above_p95=bool(np.isfinite(v) and v > q95)))
            print(f"    {label} {wname}: bg median {np.median(bg):.3f} p95 {q95:.3f}",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "bigwig_tierC.csv"), index=False)
    print("\n=== Tier C on signal tracks ===", flush=True)
    if len(df):
        print(df.round(3).to_string(index=False), flush=True)

    verdicts = {}
    for (ds, w), sub in df.groupby(["dataset", "window"]):
        verdicts[f"{ds}|{w}"] = int(sub.above_p95.sum())
    sm = {
        "datasets": sorted(df.dataset.unique().tolist()) if len(df) else [],
        "n_targets_above_p95": verdicts,
        "gate": "PASS = >=2/6 canonical targets above the 95th percentile of 3,000 random "
                "TSS windows at +-2 kb",
        "n_random_windows": NRAND,
        "caveat": ("Signal-based, not peak-based: no peak caller and no pyBigWig wheel for "
                   "this Python, so a broad low plateau would score like a summit. Reported "
                   "as binding potential, never as a binding call."),
    }
    json.dump(sm, open(os.path.join(OUT, "bigwig_summary.json"), "w"), indent=2, default=str)
    print("\n" + json.dumps(sm, indent=2, default=str), flush=True)
    print("done")


if __name__ == "__main__":
    main()
