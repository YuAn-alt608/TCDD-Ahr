"""Refine D2 cell-type labels using the genetic lineage label (Myh11-CreERT2;Rosa-tdT).

tdT+ barcodes are SMC-derived by construction, so any tdT+ cell called as fibroblast/epithelial
must be a phenotypically modulated SMC. tdT- libraries are SMC-depleted.
"""
import os, sys
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L


def main():
    p = os.path.join(L.RES, "D2", "processed.h5ad")
    a = sc.read_h5ad(p)
    ct = a.obs["celltype"].astype(str).to_numpy()
    srt = a.obs["sorted"].astype(str).to_numpy()

    def score(gene_list):
        gs = [g for g in gene_list if g in a.var_names]
        return np.asarray(a[:, gs].X.mean(axis=1)).ravel() if gs else np.zeros(a.n_obs)

    smc = score(["Myh11", "Acta2", "Tagln", "Cnn1"])
    mod = score(["Lgals3", "Spp1", "Fn1", "Col8a1", "Vcam1"])
    new = ct.copy()
    thr = np.median(smc[srt == "tdT+"])
    tdt = srt == "tdT+"
    # SMC-derived but called something else -> modulated SMC
    bad = tdt & ~np.isin(ct, ["SMC", "ModifiedSMC"])
    new[bad] = "ModifiedSMC"
    # within SMC-derived, split contractile vs modulated
    new[tdt & (smc >= thr)] = "SMC"
    new[tdt & (smc < thr)] = "ModifiedSMC"
    # non-SMC libraries cannot contain SMC
    new[(~tdt) & np.isin(new, ["SMC", "ModifiedSMC"])] = "Fibroblast"

    changed = int((new != ct).sum())
    a.obs["celltype_raw"] = a.obs["celltype"].astype(str)
    a.obs["celltype"] = pd.Categorical(new)
    a.obs["lineage"] = a.obs["celltype"].map(L.CLASS).fillna("Other").astype("category")
    a.write_h5ad(p, compression="gzip")
    print(f"relabelled {changed}/{a.n_obs} cells ({changed/a.n_obs:.1%})")
    print(pd.crosstab(a.obs.celltype_raw, a.obs.celltype).to_string())
    print(a.obs.groupby(["group", "lineage"], observed=True).size().to_string())


if __name__ == "__main__":
    main()
