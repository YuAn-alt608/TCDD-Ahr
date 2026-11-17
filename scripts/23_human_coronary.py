"""23 - Do the mouse-defined programmes exist as coherent subsets in human coronary?

External data: GSE131778 (Wirka et al., Nat Med 2019), human coronary scRNA-seq,
11,755 cells from 8 patients, counts as a dense gene x cell text matrix. The
deposited file carries no disease annotation, only barcode suffixes -1..-8, so
this analysis is deliberately limited to what the file can support:

  * coarse cell-type calling from canonical markers (no reliance on the paper's
    labels, so the test is independent),
  * projection of the three mouse programmes (modulated SMC, LAM-like macrophage,
    endothelial barrier panel) into those compartments,
  * the two ligand-receptor pairs that carry the paracrine claim (TIMP1-CD63,
    FN1-SDC4) checked for the expected sender/receiver partition.

It cannot test lesion-vs-normal, because the file has no condition column. Any
claim beyond "the programmes map onto the right human compartment" would be
unsupported and is not made.

Outputs (results/human/):
  human_celltype_scores.csv     per-cell marker and programme scores
  human_programme_summary.csv   score by compartment, plus sender/receiver check
  summary.json
"""
import os, sys, gzip, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

F = os.path.join(L.ROOT, "data", "external", "human_coronary.txt.gz")
OUT = os.path.join(L.RES, "human")
os.makedirs(OUT, exist_ok=True)

MARKERS = {
    "SMC": ["MYH11", "ACTA2", "TAGLN", "CNN1", "MYOCD"],
    "EC": ["PECAM1", "VWF", "CDH5", "EMCN", "KDR"],
    "Mac": ["CD68", "CD14", "LYZ", "CSF1R", "AIF1"],
    "Fib": ["DCN", "LUM", "COL1A1", "COL1A2", "PDGFRA"],
    "T": ["CD3D", "CD3E", "CD2", "IL7R"],
    "ModSMC": ["LGALS3", "SPP1", "FN1", "COL8A1", "VCAM1"],
    "LAM": ["TREM2", "LGALS3", "CSF1R", "SPP1", "APOE", "CD9", "FABP5", "LPL", "GPNMB"],
    "ECpanel": ["KLF2", "SMAD2", "NR1H3", "HIF1A", "KLF4"],
    "Matrix": ["TIMP1", "FN1", "COL1A2", "COL3A1", "SPP1"],
}
PAIRS = [("TIMP1", "CD63"), ("FN1", "SDC4"), ("SPP1", "CD44")]


def main():
    # ---- pass 1: header + which rows to keep
    with gzip.open(F, "rt") as f:
        cells = f.readline().rstrip("\n").split("\t")
    print(f"cells: {len(cells)}", flush=True)
    want = set()
    for v in MARKERS.values():
        want |= set(v)
    for a, b in PAIRS:
        want |= {a, b}
    want |= {"PTPRC", "MYH11", "DCN", "CD68", "PECAM1", "COL1A1", "ACTA2"}

    # ---- pass 2: stream rows, keep the wanted genes
    rng = np.random.default_rng(0)
    want_rows, want_names, bg_rows, bg_names = [], [], [], []
    lib_all = np.zeros(len(cells), dtype=np.float64)
    with gzip.open(F, "rt") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            v = np.asarray(p[1:], dtype=np.float32)
            lib_all += v                     # true library size uses every gene
            g = p[0]
            if g in want:
                want_names.append(g)
                want_rows.append(v)
            elif len(bg_rows) < 3000 and rng.random() < 0.0012:
                bg_names.append(g)
                bg_rows.append(v)
    X = np.vstack(want_rows + bg_rows).astype(np.float32)
    keep_names = want_names + bg_names
    print(f"kept genes {X.shape} ({len(want_names)} panels + {len(bg_names)} background)", flush=True)

    df = pd.DataFrame(X, index=keep_names, columns=cells)
    df = df[~df.index.duplicated()]
    print(f"median counts/cell {np.median(lib_all):.0f}", flush=True)
    cpm = pd.DataFrame(np.log1p(df.to_numpy() / np.maximum(lib_all, 1)[None, :] * 1e4),
                       index=df.index, columns=df.columns)

    # z-score within the kept panel (mean/sd per gene across cells)
    z = (cpm.sub(cpm.mean(1), axis=0)).div(cpm.std(1) + 1e-6, axis=0)

    scores = {}
    for name, gl in MARKERS.items():
        gl = [g for g in gl if g in z.index]
        if not gl:
            continue
        scores[name] = z.loc[gl].mean(0).to_numpy()
    S = pd.DataFrame(scores, index=cells)
    ct_cols = ["SMC", "EC", "Mac", "Fib", "T"]
    S["compartment"] = S[ct_cols].idxmax(1).where(S[ct_cols].max(1) > 0.05, "other")
    S["sample"] = [c.split("-")[-1] for c in S.index]
    S.to_csv(os.path.join(OUT, "human_celltype_scores.csv"), float_format="%.4f")

    print("\ncompartment sizes:\n" + S.compartment.value_counts().to_string(), flush=True)

    rows = []
    for comp, prog in [("SMC", "ModSMC"), ("Mac", "LAM"), ("EC", "ECpanel"),
                       ("SMC", "Matrix"), ("Fib", "Matrix")]:
        sub = S[S.compartment == comp]
        if len(sub) == 0:
            continue
        rows.append(dict(compartment=comp, programme=prog, n_cells=len(sub),
                         mean=float(sub[prog].mean()), sd=float(sub[prog].std()),
                         frac_positive=float((sub[prog] > 0).mean()),
                         mean_other=float(S[S.compartment != comp][prog].mean()),
                         n_other=int((S.compartment != comp).sum())))
    summ = pd.DataFrame(rows)

    # sender / receiver partition for the paracrine pairs
    for lig, rec in PAIRS:
        if lig not in cpm.index or rec not in cpm.index:
            continue
        lv = cpm.loc[lig]
        rv = cpm.loc[rec]
        rec_exp = S["compartment"].map(
            lambda c: float(rv[S.compartment == c].mean()) if (S.compartment == c).sum() else np.nan)
        lig_exp = S["compartment"].map(
            lambda c: float(lv[S.compartment == c].mean()) if (S.compartment == c).sum() else np.nan)
        summ = pd.concat([summ, pd.DataFrame([dict(
            compartment="pair:" + lig + "-" + rec, programme="sender_receiver",
            n_cells=int(len(S)), mean=float(lv.mean()), sd=float(rv.mean()),
            frac_positive=float((lv > 0).mean()),
            mean_other=float(rec_exp[S.compartment == "Mac"].mean()),
            n_other=int((S.compartment == "Mac").sum()))])], ignore_index=True)
    summ.to_csv(os.path.join(OUT, "human_programme_summary.csv"), index=False)
    print("\n" + summ.round(3).to_string(index=False), flush=True)

    sm = {"n_cells": int(len(S)), "n_samples": int(S["sample"].nunique()),
          "compartment_sizes": S.compartment.value_counts().to_dict(),
          "module_mean_by_compartment": {
              f"{r.compartment}|{r.programme}": round(float(r.mean), 3)
              for r in summ.itertuples() if r.programme != "sender_receiver"},
          "provenance": "GSE131778 human coronary scRNA-seq (Wirka et al. 2019)",
          "no_condition_labels": True}
    json.dump(sm, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=str)
    print(json.dumps(sm, indent=2, default=str))


if __name__ == "__main__":
    main()
