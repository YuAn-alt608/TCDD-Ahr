"""42 - The stromal continuum: why the Matrix panel "loses" its SMC label in human.

Script 39 declared SMC as the target compartment for the Matrix panel *before*
looking, and the panel then scored best in fibroblasts (AUC 0.687 vs 0.162) in
human. Written up as a limitation that reads as "mouse and human disagree". The
data say something more specific and more useful, and this script tests it in
two steps:

  A. In the mouse data the same panel is already not SMC-specific: matrix_ligand
     is high in ModifiedSMC and in Fibroblast, low in contractile SMC, in the
     same animal. So the panel marks a de-differentiated matrix-producing state,
     not a lineage.

  B. In human, if that reading is right, then inside SMC-labelled cells the
     Matrix score must fall as contractile genes rise and must rise with
     fibromyocyte markers (TNFRSF11B, LUM). That is a within-lineage test and it
     cannot be produced by cell-type composition.

If B holds, the cross-species "discrepancy" is a labelling consequence of a
continuum that both species share, which is a result rather than a caveat.

Outputs (results/human/):
  stromal_continuum_human.csv     correlations inside SMC / Fib / stroma
  stromal_continuum_mouse.csv     module score by cell type, per animal
  stromal_continuum_summary.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "human")

CONTRACT = ["MYH11", "ACTA2", "TAGLN", "CNN1", "MYOCD"]
# COL1A2, COL3A1 and FN1 are *inside* the Matrix panel; using them as the
# "ECM end" of the gradient would make the gradient partly tautological, so the
# non-circular ECM composite keeps only genes the panel does not contain.
ECM = ["COL1A1", "POSTN"]
FIBRO = ["TNFRSF11B", "LUM"]
NBOOT = 1000
SEED = 13


def zscore(s):
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)


def auc(pos, neg):
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) /
                 (len(pos) * len(neg)))


def partial_r(x, y, cov):
    """Pearson r of x,y after regressing both on cov (donor dummies + depth)."""
    X = np.column_stack([np.ones(len(np.asarray(cov[0], float)))] +
                        [np.asarray(c, float) for c in cov])
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(stats.pearsonr(rx, ry).statistic)


def cluster_ci(df, col, target, nboot=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    donors = df["sample"].unique()
    idx = {dn: g.index.to_numpy() for dn, g in df.groupby("sample")}
    out = []
    for _ in range(nboot):
        pick = rng.choice(donors, len(donors), replace=True)
        sub = df.loc[np.concatenate([idx[dn] for dn in pick])]
        v = partial_r(sub[col].to_numpy(), sub[target].to_numpy(),
                      [sub["log_nCount"].to_numpy(), sub["pct_ribo"].to_numpy(),
                       sub["pct_mito"].to_numpy()])
        out.append(v)
    o = np.array(out)
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5)), float((o < 0).mean())


def main():
    d = pd.read_csv(os.path.join(OUT, "human_celltype_scores.csv"), index_col=0)
    qc = pd.read_csv(os.path.join(OUT, "human_cell_qc.csv"), index_col=0)
    ex = pd.read_csv(os.path.join(OUT, "human_marker_logcpm.csv"), index_col=0)
    d = d.join(qc).join(ex)
    d["log_nCount"] = np.log10(d["nCount"])
    genes = [g for g in CONTRACT + ECM + FIBRO if g in d.columns]
    z = pd.DataFrame({g: zscore(d[g]) for g in genes}, index=d.index)
    d["contractile"] = z[[g for g in CONTRACT if g in z]].mean(1)
    d["ecm"] = z[[g for g in ECM if g in z]].mean(1)
    d["dediff"] = d["ecm"] - d["contractile"]

    # ------------------------------------------------------------ A. mouse
    mm = pd.read_csv(os.path.join(RES, "pseudobulk", "rna_module_by_mouse.csv"))
    mm = mm[mm.module == "matrix_ligand"]
    piv = mm.pivot_table(index=["mouse", "group"], columns="celltype",
                         values="score", aggfunc="mean")
    rows = []
    for (mn, gp), r in piv.iterrows():
        if np.isfinite(r.get("SMC", np.nan)) and np.isfinite(r.get("ModifiedSMC", np.nan)):
            rows.append(dict(mouse=mn, group=gp, SMC=r["SMC"],
                             ModifiedSMC=r["ModifiedSMC"],
                             Fibroblast=r.get("Fibroblast", np.nan),
                             d_mod_vs_smc=r["ModifiedSMC"] - r["SMC"]))
    mrow = pd.DataFrame(rows)
    mrow.to_csv(os.path.join(OUT, "stromal_continuum_mouse.csv"), index=False)
    dd = mrow["d_mod_vs_smc"].dropna()
    d2 = (mrow["ModifiedSMC"] - mrow["Fibroblast"]).dropna()
    mouse_stats = dict(
        n_animals=int(len(dd)),
        median_mod_minus_smc=float(dd.median()),
        n_up=int((dd > 0).sum()),
        p_wilcoxon=float(stats.wilcoxon(dd).pvalue) if len(dd) >= 6 else np.nan,
        median_mod_minus_fib=float(d2.median()),
        p_mod_vs_fib=float(stats.wilcoxon(d2).pvalue) if len(d2) >= 6 else np.nan,
    )
    print("A. mouse matrix_ligand, per animal:", mouse_stats, flush=True)
    print(mrow.round(2).to_string(index=False), flush=True)

    # ------------------------------------------------------------ B. human
    st = d[d.compartment.isin(["SMC", "Fib"])].copy()
    print(f"\nB. human stromal cells {len(st)} (SMC {(st.compartment=='SMC').sum()}, "
          f"Fib {(st.compartment=='Fib').sum()})", flush=True)
    a_str = auc(st.loc[st.compartment == "Fib", "Matrix"],
                st.loc[st.compartment == "SMC", "Matrix"])
    print(f"   AUC Matrix Fib vs SMC within stroma = {a_str:.3f}", flush=True)

    targets = ["MYH11", "ACTA2", "TNFRSF11B", "LUM", "contractile", "ecm", "dediff"]
    rows = []
    for comp in ["SMC", "Fib", "stroma"]:
        sub = st if comp == "stroma" else st[st.compartment == comp]
        sub = sub.copy()
        for t in targets:
            if t not in sub.columns:
                continue
            r = partial_r(sub["Matrix"].to_numpy(), sub[t].to_numpy(),
                          [sub["log_nCount"].to_numpy(), sub["pct_ribo"].to_numpy(),
                           sub["pct_mito"].to_numpy()])
            lo, hi, frac = cluster_ci(sub, "Matrix", t)
            rows.append(dict(compartment=comp, n_cells=len(sub), target=t,
                             partial_r=r, ci_lo=lo, ci_hi=hi,
                             n_cells_neg_frac=frac,
                             spearman=float(stats.spearmanr(sub["Matrix"], sub[t]).statistic)))
    cc = pd.DataFrame(rows)
    cc.to_csv(os.path.join(OUT, "stromal_continuum_human.csv"), index=False)
    print("\n   Matrix vs marker, donor+depth partialled (CI resamples donors):", flush=True)
    print(cc.round(3).to_string(index=False), flush=True)

    # ---------------------------------------------- the gradient itself
    smc = st[st.compartment == "SMC"].copy()
    smc["bin"] = pd.qcut(smc["dediff"], 10, labels=False)
    grad = smc.groupby("bin").agg(dediff=("dediff", "median"),
                                  matrix=("Matrix", "mean"),
                                  myh11=("MYH11", "mean"),
                                  tnfrsf11b=("TNFRSF11B", "mean"),
                                  n=("Matrix", "size")).reset_index()
    fib_mean = float(st.loc[st.compartment == "Fib", "Matrix"].mean())
    print("\n   SMC deciles of de-differentiation index:", flush=True)
    print(grad.round(3).to_string(index=False), flush=True)
    print(f"   (Fibroblast mean Matrix = {fib_mean:.3f})", flush=True)
    rho_grad = float(stats.spearmanr(grad["dediff"], grad["matrix"]).statistic)

    # ------------------------------------------------------------ verdict
    g = cc[(cc.compartment == "SMC")].set_index("target")
    verdict = dict(
        myh11_negative=bool(g.loc["MYH11", "partial_r"] < -0.1 and
                            g.loc["MYH11", "ci_hi"] < 0),
        fibro_markers_positive=bool(g.loc["TNFRSF11B", "partial_r"] > 0.1 and
                                    g.loc["TNFRSF11B", "ci_lo"] > 0),
        gradient_monotone=bool(rho_grad > 0.9),
        continuum=bool(a_str > 0.6),
    )
    summary = dict(
        n_stromal=int(len(st)), auc_fib_vs_smc=a_str,
        mouse=mouse_stats, human_correlations=cc.round(4).to_dict("records"),
        gradient=grad.round(4).to_dict("records"),
        gradient_rho=rho_grad, fib_matrix_mean=fib_mean,
        verdict=verdict,
        note=("Human correlations are Pearson on residuals after removing donor "
              "and log nCount / pct_ribo / pct_mito; the CI resamples donors. "
              "The mouse comparison uses the same matrix gene set as script 19."),
    )
    with open(os.path.join(OUT, "stromal_continuum_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("\n   verdict:", verdict, flush=True)


if __name__ == "__main__":
    main()
