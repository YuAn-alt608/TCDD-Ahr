"""Epigenetic-switch reversibility evidence without a deep generative model.

Two independent, fully data-driven lines of evidence:
  (1) association: does the switch's expression / regulon activity correlate with the position of a
      cell on the exposure axis (control centroid -> TCDD centroid)?
  (2) real genetic perturbation: does SMC-specific Ahr knockout move cells back toward the control
      centroid or away from it?  This is a true in-vivo perturbation, not a simulation.
"""
import json, os, sys
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

SWITCHES = ["Sirt1", "Suv39h1", "Setdb1", "Ehmt2", "Trim28", "Hdac1", "Dnmt1", "Ezh2", "Ahr",
            "Sirt6", "Kdm6b"]


def main():
    a = sc.read_h5ad(os.path.join(L.RES, "D2", "processed.h5ad"))
    X = sp.csr_matrix(a.X)
    out = []
    for ct, n in a.obs.celltype.value_counts().items():
        if n < 300:
            continue
        m = (a.obs.celltype == ct).to_numpy()
        sub = a[m]
        grp = sub.obs.group.astype(str).to_numpy()
        is_c, is_t = grp == "Control", grp == "TCDD_16wk_WT"
        is_k = grp == "TCDD_16wk_AhrKO"
        if is_c.sum() < 50 or is_t.sum() < 50:
            continue
        s2 = sub.copy()
        sc.pp.highly_variable_genes(s2, n_top_genes=2000)
        sc.pp.scale(s2, max_value=10)
        sc.pp.pca(s2, n_comps=30, random_state=0)
        P = s2.obsm["X_pca"]
        axis = P[is_t].mean(0) - P[is_c].mean(0)
        axis /= np.linalg.norm(axis) + 1e-12
        pos = P @ axis                       # exposure-axis coordinate
        pos = (pos - pos[is_c].mean())        # 0 = control centroid

        for g in SWITCHES:
            if g not in a.var_names:
                continue
            v = np.asarray(X[m][:, a.var_names.get_loc(g)].todense()).ravel()
            rho, p = spearmanr(v[is_t | is_k], pos[is_t | is_k]) if (is_t | is_k).sum() > 30 else (np.nan, np.nan)
            d_expr = float(np.mean(v[is_t]) - np.mean(v[is_c]))
            d_pos_ko = float(pos[is_k].mean() - pos[is_t].mean()) if is_k.sum() > 20 else np.nan
            out.append({"celltype": ct, "gene": g, "n": int(m.sum()),
                        "expr_control": float(v[is_c].mean()),
                        "delta_expr_tcdd": d_expr,
                        "rho_expr_vs_axis": float(rho), "p_rho": float(p),
                        "axis_pos_control": 0.0,
                        "axis_pos_tcdd": float(pos[is_t].mean()),
                        "axis_pos_ahrko": float(pos[is_k].mean()) if is_k.sum() > 20 else np.nan,
                        "ahrko_shift_vs_tcdd": d_pos_ko})
        print(f"  {ct}: done", flush=True)

    d = pd.DataFrame(out)
    od = os.path.join(L.RES, "perturbation")
    os.makedirs(od, exist_ok=True)
    d.to_csv(os.path.join(od, "switch_evidence.csv"), index=False)

    # summary: candidate rescue targets = reduced by TCDD and negatively aligned with the axis
    s = (d.groupby("gene")
         .agg(mean_delta_expr=("delta_expr_tcdd", "mean"),
              mean_rho=("rho_expr_vs_axis", "mean"),
              n_celltypes=("celltype", "nunique"))
         .sort_values("mean_rho"))
    s.to_csv(os.path.join(od, "switch_summary.csv"))
    print(s.round(3).to_string(), flush=True)

    # AHR-KO rescue summary per cell type
    k = d.dropna(subset=["ahrko_shift_vs_tcdd"]).groupby("celltype").agg(
        axis_pos_tcdd=("axis_pos_tcdd", "mean"),
        axis_pos_ahrko=("axis_pos_ahrko", "mean"),
        shift=("ahrko_shift_vs_tcdd", "mean"))
    k["toward_control"] = k["shift"] < 0
    k.to_csv(os.path.join(od, "ahrko_rescue.csv"))
    print(k.round(3).to_string(), flush=True)

    # keep the old file name for the report so figures still work
    rep = d.copy()
    rep["perturbation"] = rep.gene
    rep["reversal_fraction"] = -rep.rho_expr_vs_axis
    json.dump({"note": "association + real Ahr-KO perturbation; no generative model"},
              open(os.path.join(od, "model_note.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
