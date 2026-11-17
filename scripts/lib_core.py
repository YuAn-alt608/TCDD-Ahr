"""Shared utilities: dataset loading, lineage annotation, heterochromatin metrics."""
import json
import os
import re
import warnings

import numpy as np
import pandas as pd

# The single-cell stack is only needed by the scripts that touch the h5ad files;
# everything that reports on already-written tables (the report builder, the
# occupancy/permutation scripts) should stay runnable in a plain numpy/pandas
# interpreter. Nothing outside this module reads `sc` off it, so the import is
# optional and the modules that need it import it themselves.
try:
    import anndata as ad
    import scanpy as sc
    import scipy.sparse as sp
except ModuleNotFoundError as _e:            # pragma: no cover
    _MISSING = _e
    ad = sc = sp = None
else:
    _MISSING = None

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
ANN = os.path.join(ROOT, "data", "annotation", "processed")
SETS = os.path.join(ANN, "sets")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

# ----------------------------------------------------------------------------- gene sets
def load_set(name):
    p = os.path.join(SETS, name + ".txt")
    if not os.path.exists(p):
        return set()
    return {l.strip() for l in open(p) if l.strip()}


# curated modules (mouse symbols)
RESTRICTION = """Trim28 Setdb1 Suv39h1 Suv39h2 Ehmt1 Ehmt2 Hdac1 Hdac2 Hdac3 Sirt1 Sirt6
Dnmt1 Dnmt3a Dnmt3b Uhrf1 Morc2 Morc3 Zfp809 Zfp708 Zfp57 Atf7ip Gapdh1 Hells
Piwil1 Piwil2 Piwil4 Mov10l1 Mael Ddx4 Tdrd1 Tdrd5 Tdrd9 Asz1 Fkbp6""".split()

DSRNA_IFN = """Ddx58 Ifih1 Dhx58 Zbp1 Eif2ak2 Oas1a Oas1b Oas2 Oas3 Oasl1 Oasl2 Isg15
Mx1 Mx2 Ifit1 Ifit2 Ifit3 Irf3 Irf7 Stat1 Stat2 Ifnb1 Ifna1 Ifna2 Ifna4 Ifna5
Cxcl10 Ccl5 Rsad2 Usp18 Bst2 Samhd1 Trex1 Aicda Adar""".split()

AGING_SEN = """Cdkn1a Cdkn2a Cdkn2b Trp53 Trp53bp1 Glb1 Serpine1 Igfbp3 Igfbp5 Igfbp7
Il6 Cxcl1 Cxcl2 Cxcl12 Mmp3 Mmp9 Mmp13 Timp1 Ccn2 Tgfbr1 B2m Gdf15 Jun Fos
E2f1 Rb1 E2f7 E2f8 Lamna Lmnb1 Lmnb2 Padi4 Hmgb1""".split()

EPIGEN_REGULATORS = """Sirt1 Sirt2 Sirt3 Sirt6 Sirt7 Suv39h1 Suv39h2 Setdb1 Ehmt1 Ehmt2
Trim28 Hdac1 Hdac2 Hdac3 Hdac9 Dnmt1 Dnmt3a Dnmt3b Tet1 Tet2 Uhrf1 Morc2 Ep300
Kat2a Kat5 Kdm6b Kdm5a Ezh2 Eed Suz12 Lsd1 Ahr Nfe2l2 Foxo1 Foxo3 Hif1a""".split()

# ----------------------------------------------------------------------------- markers
MARKERS = {
    "Epithelial": ["Epcam", "Cdh1", "Krt8", "Krt18", "Krt19"],
    "AT2": ["Sftpc", "Sftpb", "Abca3", "Lamp3"],
    "AT1": ["Ager", "Pdpn", "Hpgd", "Aqp5"],
    "Club": ["Scgb1a1", "Scgb3a1", "Cyp2f2"],
    "Ciliated": ["Foxj1", "Tppp3", "Ccdc153"],
    "Endothelial": ["Pecam1", "Cdh5", "Kdr", "Emcn", "Tie1"],
    "LymphaticEC": ["Lyve1", "Prox1", "Pdpn", "Ccl21a"],
    "Fibroblast": ["Col1a1", "Col1a2", "Pdgfra", "Dcn", "Lum", "Col3a1"],
    "SMC": ["Myh11", "Acta2", "Tagln", "Cnn1", "Myocd"],
    "ModifiedSMC": ["Lgals3", "Spp1", "Fn1", "Col8a1", "Vcam1"],
    "Pericyte": ["Rgs5", "Kcnj8", "Abcc9", "Des"],
    "Adipocyte": ["Adipoq", "Plin1", "Lep", "Fabp4", "Pparg"],
    "ASPC": ["Pdgfra", "Pdgfrb", "Lepr", "Dpp4", "Ly6a"],
    "Hepatocyte": ["Alb", "Apoa1", "Ttr", "Cyp3a11", "Serpina1a"],
    "Macrophage": ["Adgre1", "Csf1r", "Lyz2", "Fcgr1", "Mrc1"],
    "Monocyte": ["Ly6c2", "Ccr2", "Plac8", "Lyz2"],
    "Neutrophil": ["S100a8", "S100a9", "Retnlg", "Mmp8"],
    "DC": ["Cd209a", "Itgax", "Flt3", "Xcr1"],
    "Tcell": ["Cd3e", "Cd3d", "Cd3g", "Trbc2"],
    "Bcell": ["Cd79a", "Cd79b", "Ms4a1", "Cd19"],
    "NK": ["Nkg7", "Klrb1c", "Gzma", "Ncr1"],
    "Mast": ["Cpa3", "Kit", "Ms4a2"],
}

# standard 4-class lineage mapping
CLASS = {
    "Epithelial": "Parenchyma", "AT1": "Parenchyma", "AT2": "Parenchyma",
    "Club": "Parenchyma", "Ciliated": "Parenchyma", "Adjacent": "Parenchyma",
    "Adipocyte": "Parenchyma", "Hepatocyte": "Parenchyma",
    "SMC": "Parenchyma", "ModifiedSMC": "Parenchyma",
    "Endothelial": "Endothelial", "LymphaticEC": "Endothelial",
    "Fibroblast": "Mesenchymal", "Pericyte": "Mesenchymal", "ASPC": "Mesenchymal",
    "Mesothelial": "Mesenchymal",
    "Macrophage": "Immune", "Monocyte": "Immune", "Neutrophil": "Immune",
    "DC": "Immune", "Tcell": "Immune", "Bcell": "Immune", "NK": "Immune",
    "Mast": "Immune", "pDC": "Immune", "Basophil": "Immune",
}


def annotate_by_markers(adata, layer="lognorm", min_score=0.0):
    """Score cells against MARKERS and assign best label (max mean z-score)."""
    present = {k: [g for g in v if g in adata.var_names] for k, v in MARKERS.items()}
    present = {k: v for k, v in present.items() if len(v) >= 2}
    sc.tl.score_genes(adata, [], score_name="_dummy", random_state=0) if False else None
    scores = {}
    for k, gs in present.items():
        sc.tl.score_genes(adata, gs, score_name=f"sc_{k}", random_state=0, ctrl_size=50)
        scores[k] = adata.obs[f"sc_{k}"].to_numpy()
    S = pd.DataFrame(scores, index=adata.obs_names)
    lab = S.idxmax(axis=1)
    mx = S.max(axis=1)
    lab[mx < min_score] = "Unknown"
    adata.obs["celltype"] = lab.astype("category")
    adata.obs["celltype_score"] = mx.to_numpy()
    adata.obs["lineage"] = adata.obs["celltype"].map(CLASS).fillna("Other").astype("category")
    return adata


# ----------------------------------------------------------------------------- loaders
def _basic_qc(a, min_genes=200, max_pct_mt=20):
    a.var_names_make_unique()
    a.var["mt"] = a.var_names.str.startswith(("mt-", "MT-"))
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
    n0 = a.n_obs
    a = a[(a.obs.n_genes_by_counts >= min_genes) & (a.obs.pct_counts_mt <= max_pct_mt)].copy()
    print(f"    QC {n0} -> {a.n_obs} cells", flush=True)
    return a


def load_d1():
    """PM2.5 lung (GSE296297) - author-provided log-normalised h5ad."""
    a = ad.read_h5ad(os.path.join(RAW, "D1_PM25_lung.h5ad"))
    a.obs["sample"] = a.obs["sampleid"].astype(str)
    a.obs["group"] = a.obs["sample"].map({"CONTROL": "Control", "PM2_5": "PM2.5"}).astype("category")
    a.obs["dataset"] = "D1_PM2.5_lung"
    # X is log1p(CP10k); invert to pseudo-counts for a uniform pipeline
    X = a.X
    X = X.expm1() if sp.issparse(X) else np.expm1(X)
    a.X = sp.csr_matrix(X)
    a.layers["counts"] = a.X.copy()
    a.obs["celltype"] = a.obs["celltype1"].astype(str).replace({
        "Epithelium Cells": "Epithelial", "Fibroblast": "Fibroblast",
        "Endothelial Cells": "Endothelial", "Macrophages": "Macrophage",
        "Neutrophils": "Neutrophil", "B cells": "Bcell", "T cells": "Tcell",
        "NK": "NK", "Monocytes": "Monocyte", "DC": "DC", "ILC": "NK", "Unkwon": "Unknown",
    }).astype("category")
    a.obs["lineage"] = a.obs["celltype"].map(CLASS).fillna("Other").astype("category")
    return a


D2_META = {
    "D2_rna_CTR_TDT1.h5": ("Control", "tdT+", "0wk"), "D2_rna_CTR_TDT2.h5": ("Control", "tdT+", "0wk"),
    "D2_rna_CTR_TDT3.h5": ("Control", "tdT+", "0wk"), "D2_rna_CTR_NTDT1.h5": ("Control", "tdT-", "0wk"),
    "D2_rna_CTR_NTDT2.h5": ("Control", "tdT-", "0wk"), "D2_rna_CTR_NTDT3.h5": ("Control", "tdT-", "0wk"),
    "D2_rna_TCDD_TDT1.h5": ("TCDD_8wk", "tdT+", "8wk"), "D2_rna_TCDD_TDT2.h5": ("TCDD_8wk", "tdT+", "8wk"),
    "D2_rna_TCDD_TDT3.h5": ("TCDD_8wk", "tdT+", "8wk"), "D2_rna_TCDD_NTDT1.h5": ("TCDD_8wk", "tdT-", "8wk"),
    "D2_rna_TCDD_NTDT2.h5": ("TCDD_8wk", "tdT-", "8wk"), "D2_rna_TCDD_NTDT3.h5": ("TCDD_8wk", "tdT-", "8wk"),
    "D2_rna_WT16_TDT.h5": ("TCDD_16wk_WT", "tdT+", "16wk"),
    "D2_rna_WT16_NTDT.h5": ("TCDD_16wk_WT", "tdT-", "16wk"),
    "D2_rna_AKO16_TDT.h5": ("TCDD_16wk_AhrKO", "tdT+", "16wk"),
    "D2_rna_AKO16_NTDT.h5": ("TCDD_16wk_AhrKO", "tdT-", "16wk"),
}


def load_d2(cap_per_sample=None):
    """TCDD coronary vasculature (GSE276872) - per-sample CellRanger h5."""
    parts = []
    rng = np.random.default_rng(0)
    for f, (grp, lin, dur) in D2_META.items():
        p = os.path.join(RAW, f)
        if not os.path.exists(p):
            print("  missing", f); continue
        a = sc.read_10x_h5(p, gex_only=True)
        a.var_names_make_unique()
        if cap_per_sample and a.n_obs > cap_per_sample:
            a = a[rng.choice(a.n_obs, cap_per_sample, replace=False)].copy()
        a.obs_names = [f"{os.path.splitext(f)[0]}_{b}" for b in a.obs_names]
        a.obs["sample"] = os.path.splitext(f)[0]
        a.obs["group"] = grp
        a.obs["sorted"] = lin
        a.obs["duration"] = dur
        parts.append(a)
        print(f"    {f}: {a.n_obs} cells", flush=True)
    a = ad.concat(parts, join="outer", label="batch", index_unique=None, fill_value=0)
    a.obs["dataset"] = "D2_TCDD_coronary"
    a.layers["counts"] = a.X.copy()
    return a


D3_META = {
    "D3_FA1.tar.gz": ("Control", "rep1"), "D3_FA2.tar.gz": ("Control", "rep2"),
    "D3_FA3.tar.gz": ("Control", "rep3"), "D3_PM1.tar.gz": ("PM2.5", "rep1"),
    "D3_PM2.tar.gz": ("PM2.5", "rep2"), "D3_PM3.tar.gz": ("PM2.5", "rep3"),
}


def _read_tar_matrix(path, workdir):
    import shutil
    import tarfile
    os.makedirs(workdir, exist_ok=True)
    with tarfile.open(path) as t:
        t.extractall(workdir)
    mtx = None
    for r, _, fs in os.walk(workdir):
        if "matrix.mtx" in fs:
            mtx = os.path.join(r, "matrix.mtx")
    d = os.path.dirname(mtx)
    genes = pd.read_csv(os.path.join(d, "genes.tsv"), sep="\t", header=None)
    bcs = pd.read_csv(os.path.join(d, "barcodes.tsv"), sep="\t", header=None)
    M = sp.csr_matrix(scipy_mmread(mtx).T)
    a = ad.AnnData(M)
    a.obs_names = bcs[0].astype(str).to_numpy()
    a.var_names = genes[1].astype(str).to_numpy() if genes.shape[1] > 1 else genes[0].astype(str).to_numpy()
    shutil.rmtree(workdir, ignore_errors=True)
    return a


def scipy_mmread(p):
    from scipy.io import mmread
    return mmread(p)


def load_d3():
    """Maternal PM2.5, offspring iWAT (GSE307673) - per-sample mtx tar.gz."""
    tmp_root = os.path.join(ROOT, "data", "tmp")
    os.makedirs(tmp_root, exist_ok=True)
    parts = []
    for f, (grp, rep) in D3_META.items():
        p = os.path.join(RAW, f)
        if not os.path.exists(p):
            print("  missing", f); continue
        wd = os.path.join(tmp_root, f.replace(".tar.gz", ""))
        a = _read_tar_matrix(p, wd)
        a.obs["sample"] = f.replace(".tar.gz", "")
        a.obs["group"] = grp
        a.obs["rep"] = rep
        parts.append(a)
        print(f"    {f}: {a.n_obs} cells", flush=True)
    a = ad.concat(parts, join="outer", label="batch", index_unique="-", fill_value=0)
    a.obs["dataset"] = "D3_maternalPM2.5_iWAT"
    a.layers["counts"] = a.X.copy()
    return a


def standardize(a, min_cells=3):
    """counts -> qc -> normalize -> log1p. Keeps .layers['counts']."""
    a = _basic_qc(a)
    sc.pp.filter_genes(a, min_cells=min_cells)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


# ----------------------------------------------------------------------------- metrics
def _expr_percent(a, genes, per_cell_norm=True):
    """Mean expression over genes (log-normalised) as a dense vector."""
    gs = [g for g in genes if g in a.var_names]
    if len(gs) == 0:
        return np.zeros(a.n_obs), 0
    X = a[:, gs].X
    v = np.asarray(X.mean(axis=1)).ravel()
    return v, len(gs)


def matched_control_genes(a, genes, mask_cells, n_bins=20, seed=0):
    """Pick same-size gene set matched on mean expression within control cells."""
    rng = np.random.default_rng(seed)
    gs = [g for g in genes if g in a.var_names]
    if not gs:
        return []
    mu = np.asarray(a[mask_cells].X.mean(axis=0)).ravel()
    order = np.argsort(mu, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    bins = pd.qcut(ranks, n_bins, labels=False, duplicates="drop")
    target_bins = bins[[a.var_names.get_loc(g) for g in gs]]
    out = []
    for b in np.unique(target_bins):
        n_need = int((target_bins == b).sum())
        pool = np.where((bins == b) | (bins == np.clip(b, 0, n_bins - 1)))[0]
        pool = np.setdiff1d(pool, [a.var_names.get_loc(g) for g in gs])
        if len(pool) == 0:
            continue
        out.extend(rng.choice(pool, size=min(n_need, len(pool)), replace=False).tolist())
    return [a.var_names[i] for i in out]


def leakage_index(a, genes, ctrl_mask, seed=0):
    """Per-cell mean(set) - mean(expression-matched controls); robust to depth."""
    v_set, n = _expr_percent(a, genes, )
    if n == 0:
        return np.full(a.n_obs, np.nan)
    ctrl_genes = matched_control_genes(a, genes, ctrl_mask, seed=seed)
    v_ctl, _ = _expr_percent(a, ctrl_genes)
    return v_set - v_ctl


def entropy_per_cell(a, n_top=0):
    """Shannon entropy of the per-cell expression distribution (normalised counts)."""
    X = a.layers["counts"] if "counts" in a.layers else a.X
    X = sp.csr_matrix(X)
    H = np.zeros(a.n_obs)
    for i in range(a.n_obs):
        row = X.getrow(i).data.astype(np.float64)
        s = row.sum()
        if s <= 0 or len(row) < 2:
            H[i] = np.nan
            continue
        p = row / s
        H[i] = -(p * np.log(p)).sum()
    return H


def regress_entropy(H, a, ctrl_mask):
    """Residualise entropy on log(depth) and log(n_genes) using control cells."""
    X = a.layers["counts"] if "counts" in a.layers else a.X
    X = sp.csr_matrix(X)
    if "total_counts" in a.obs.columns:
        depth = np.log1p(a.obs["total_counts"].to_numpy(float))
    else:
        depth = np.log1p(np.asarray(X.sum(1)).ravel())
    if "n_genes_by_counts" in a.obs.columns:
        ng = np.log1p(a.obs["n_genes_by_counts"].to_numpy(float))
    else:
        ng = np.log1p(np.asarray((X > 0).sum(1)).ravel())
    ok = np.isfinite(H) & np.isfinite(depth) & np.isfinite(ng)
    Xd = np.c_[np.ones(ok.sum()), depth[ok], ng[ok]]
    use = ok & ctrl_mask
    Xd_fit = np.c_[np.ones(use.sum()), depth[use], ng[use]]
    beta, *_ = np.linalg.lstsq(Xd_fit, H[use], rcond=None)
    res = np.full(a.n_obs, np.nan)
    res[ok] = H[ok] - Xd @ beta
    return res


def zscore_within(a, v, strata, ref_mask):
    """z-score a per-cell vector using reference (control) cells within each stratum."""
    out = np.full_like(v, np.nan, dtype=float)
    for s in pd.unique(strata):
        m_all = (strata == s)
        m_ref = m_all & ref_mask
        if m_ref.sum() < 20:
            continue
        mu, sd = np.nanmean(v[m_ref]), np.nanstd(v[m_ref])
        if not np.isfinite(sd) or sd == 0:
            continue
        out[m_all] = (v[m_all] - mu) / sd
    return out


def cohens_d(x, y):
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) < 5 or len(y) < 5:
        return np.nan
    nx, ny = len(x), len(y)
    s = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if s == 0:
        return np.nan
    return (x.mean() - y.mean()) / s


def perm_p(x, y, n=2000, seed=0):
    """Two-sided permutation p-value on difference of means."""
    x = np.asarray(x)[np.isfinite(x)]; y = np.asarray(y)[np.isfinite(y)]
    if len(x) < 5 or len(y) < 5:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = x.mean() - y.mean()
    pool = np.r_[x, y]
    nx = len(x)
    cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[:nx].mean() - pool[nx:].mean()) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (n + 1)


def bh(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    o = np.argsort(pv)
    m = len(pv)
    q = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        v = pv[o[i]] * m / (i + 1)
        prev = min(prev, v)
        q[o[i]] = prev
    out[ok] = np.clip(q, 0, 1)
    return out
