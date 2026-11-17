"""Process one dataset: QC -> cluster -> annotate lineage -> compute 4 heterochromatin fingerprints.

Usage: python 04_process.py D1|D2|D3
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import scanpy as sc
import decoupler as dc
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

CAP_PER_SAMPLE = 4000
CAP = {"D1": 4000, "D2": 2500, "D3": 4000}
NET = pd.read_csv(os.path.join(L.ANN, "collectri_mouse.csv"))
NET = NET[["source", "target", "weight"]].dropna().drop_duplicates()


def subsample(a, cap=CAP_PER_SAMPLE, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for s, idx in a.obs.groupby("sample", observed=True).groups.items():
        idx = np.asarray(idx)
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    return a[np.isin(a.obs_names, keep)].copy()


def cluster_and_annotate(a, res=0.8):
    print("    HVG...", flush=True)
    sc.pp.highly_variable_genes(a, n_top_genes=3000, batch_key="sample")
    print("    PCA...", flush=True)
    sc.pp.pca(a, n_comps=30, mask_var="highly_variable", zero_center=True, chunked=False, random_state=0)
    print("    harmony...", flush=True)
    try:
        sc.external.pp.harmony_integrate(a, "sample", max_iter_harmony=8)
        rep = "X_pca_harmony"
    except Exception as e:
        print("  harmony failed:", e, flush=True)
        rep = "X_pca"
    print("    neighbors...", flush=True)
    sc.pp.neighbors(a, use_rep=rep, n_neighbors=15, random_state=0)
    print("    leiden...", flush=True)
    sc.tl.leiden(a, resolution=res, key_added="leiden", random_state=0, flavor="igraph", n_iterations=2)

    # per-cluster marker annotation
    present = {k: [g for g in v if g in a.var_names] for k, v in L.MARKERS.items()}
    present = {k: v for k, v in present.items() if len(v) >= 2}
    S = {}
    for k, gs in present.items():
        S[k] = np.asarray(a[:, gs].X.mean(axis=1)).ravel()
    S = pd.DataFrame(S, index=a.obs_names)
    S["leiden"] = a.obs["leiden"].to_numpy()
    cl_mean = S.groupby("leiden", observed=True).mean()
    cl_lab = cl_mean.idxmax(axis=1)
    print("  cluster -> label:\n", cl_lab.to_string(), flush=True)
    a.obs["celltype"] = a.obs["leiden"].map(cl_lab).astype(str)
    a.obs["celltype"] = a.obs["celltype"].astype("category")
    a.obs["lineage"] = a.obs["celltype"].map(L.CLASS).fillna("Other").astype("category")
    return a


def compute_metrics(a, ref_col="group", ref_value="Control", act=None):
    """All per-cell fingerprints. Index-safe: every table is built on a.obs_names."""
    idx = a.obs_names
    ctrl = (a.obs[ref_col].astype(str) == ref_value).to_numpy()
    strata = a.obs["celltype"].astype(str).to_numpy()
    M = pd.DataFrame(index=idx)

    print("  leakage indices...", flush=True)
    for st in ["ConstitutiveHet", "LADproxy", "H3K9me3", "TEprom_young", "TEprom_LTR_strict"]:
        gs = L.load_set(st)
        M[f"LI_{st}"] = L.leakage_index(a, gs, ctrl)
        M[f"n_{st}"] = len([g for g in gs if g in a.var_names])
        print(f"    {st}: {M[f'n_{st}'].iloc[0]} genes measured", flush=True)

    print("  module scores...", flush=True)
    for name, gs in [("IFN_dsRNA", L.DSRNA_IFN), ("Restriction", L.RESTRICTION),
                     ("AgingSen", L.AGING_SEN)]:
        gsm = [g for g in gs if g in a.var_names]
        M[f"mod_{name}"] = (np.asarray(a[:, gsm].X.mean(axis=1)).ravel()
                            if gsm else np.zeros(a.n_obs))
        print(f"    {name}: {len(gsm)} genes", flush=True)

    print("  entropy...", flush=True)
    H = L.entropy_per_cell(a)
    M["entropy_raw"] = H
    M["entropy_res"] = L.regress_entropy(H, a, ctrl)

    for c in [c for c in M.columns if not c.startswith("n_")]:
        M["z_" + c] = L.zscore_within(a, M[c].to_numpy(float), strata, ctrl)

    a.obs = a.obs.join(M)

    C = M[["z_LI_TEprom_young", "z_LI_TEprom_LTR_strict", "z_mod_IFN_dsRNA",
           "z_mod_Restriction"]].copy()
    C["z_mod_Restriction"] = -C["z_mod_Restriction"]
    a.obs["TES"] = C.mean(axis=1, skipna=True).to_numpy()
    a.obs["HGDI"] = M["z_LI_ConstitutiveHet"].to_numpy()
    a.obs["LADleak"] = M["z_LI_LADproxy"].to_numpy()

    if act is None:
        print("  regulon activity (CollecTRI / ULM)...", flush=True)
        t0 = time.time()
        dc.mt.ulm(data=a, net=NET, tmin=5, verbose=False)
        raw_act = a.obsm.get("score_ulm")
        if isinstance(raw_act, pd.DataFrame):
            act = raw_act.copy()
            act.index = idx
        else:
            cols = [c for c in sorted(set(NET.source)) if c in a.var_names]
            A = np.asarray(raw_act)
            if A.shape[0] != len(idx):
                A = A.T
            act = pd.DataFrame(A, index=idx, columns=cols)
        print(f"    TFs: {act.shape[1]} in {time.time()-t0:.0f}s", flush=True)
    else:
        act = act.copy()
        act.index = idx
        print(f"    reused TF activity: {act.shape[1]} TFs", flush=True)
    act.columns = [f"TF_{c}" if not str(c).startswith("TF_") else c for c in act.columns]
    a.obs = a.obs.join(act)

    A = np.clip(act.to_numpy(float), 0, None)
    s = A.sum(axis=1, keepdims=True); s[s == 0] = 1
    P = A / s
    with np.errstate(divide="ignore", invalid="ignore"):
        re = -(np.where(P > 0, P * np.log(P), 0.0)).sum(axis=1)
    a.obs["regulon_entropy"] = re
    a.obs["z_regulon_entropy"] = L.zscore_within(a, re, strata, ctrl)
    return a, act


def main(ds):
    t0 = time.time()
    print(f"[{ds}] loading", flush=True)
    if ds == "D2":
        a = L.load_d2(cap_per_sample=2200)
    else:
        a = {"D1": L.load_d1, "D3": L.load_d3}[ds]()
    a = L.standardize(a)
    a = subsample(a, cap=CAP.get(ds, CAP_PER_SAMPLE))
    print(f"[{ds}] analysis set: {a.n_obs} cells x {a.n_vars} genes", flush=True)
    ckpt = os.path.join(L.RES, ds, "clustered.h5ad")
    os.makedirs(os.path.join(L.RES, ds), exist_ok=True)
    if os.path.exists(ckpt):
        print(f"[{ds}] reusing clustering checkpoint", flush=True)
        a = sc.read_h5ad(ckpt)
    else:
        a = cluster_and_annotate(a)
        a.write_h5ad(ckpt, compression="gzip")
    print(a.obs.groupby(["group", "lineage"], observed=True).size().to_string(), flush=True)
    a, act = compute_metrics(a)
    out = os.path.join(L.RES, ds)
    os.makedirs(out, exist_ok=True)
    a.write_h5ad(os.path.join(out, "processed.h5ad"), compression="gzip")
    act.to_csv(os.path.join(out, "regulon_activity.csv.gz"), compression="gzip", float_format="%.5g")
    slim = a.obs.drop(columns=[c for c in a.obs.columns if c.startswith("TF_")], errors="ignore")
    slim.to_csv(os.path.join(out, "cell_metrics.csv.gz"), compression="gzip", float_format="%.5g")
    print(f"[{ds}] done in {(time.time()-t0)/60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
