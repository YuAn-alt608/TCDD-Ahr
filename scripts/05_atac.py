"""scATAC-seq layer (GSE276870, paired with GSE276872).

The four samples were processed independently, so their peak sets are sample specific (ArchR
iterative overlap; coordinates differ by tens of bp). Concatenating them on a union of peaks
gives ~594k nearly disjoint features and the embedding then clusters by *sample* rather than by
cell type. We therefore rebuild a shared feature space on fixed 5 kb genome bins - the usual
fix for this situation - and derive gene activity from bins within +/-5 kb of each TSS.
"""
import os, sys, time
import h5py
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

ATAC = {
    "D2_atac_CTRL1.h5": ("Control", "rep1"),
    "D2_atac_CTRL2.h5": ("Control", "rep2"),
    "D2_atac_WT_TCDD.h5": ("TCDD_16wk_WT", "rep1"),
    "D2_atac_AHRKO_TCDD.h5": ("TCDD_16wk_AhrKO", "rep1"),
}
CAP = 4000
BIN = 5000

KEY = {
    "SMC": ["Myh11", "Acta2", "Tagln", "Cnn1", "Myocd"],
    "Fibroblast": ["Pdgfra", "Col1a1", "Col1a2", "Dcn", "Postn", "Lum"],
    "Endothelial": ["Pecam1", "Cdh5", "Emcn", "Vwf", "Klf2"],
    "Macrophage": ["Csf1r", "Cd68", "Lyz2", "Adgre1", "C1qa"],
    "Tcell": ["Cd3e", "Cd3d", "Lck", "Cd2"],
    "Bcell": ["Cd79a", "Cd79b", "Ms4a1"],
    "Adipocyte": ["Adipoq", "Plin1", "Fabp4", "Lep"],
}


def read_atac_h5(path):
    with h5py.File(path, "r") as f:
        g = f["matrix"]
        shape = tuple(int(x) for x in g["shape"][:])
        data = g["data"][:]
        indices = g["indices"][:]
        indptr = g["indptr"][:]
        barcodes = np.array([b.decode() for b in g["barcodes"][:]])
        names = np.array([b.decode() for b in g["features"]["name"][:]])
    M = sp.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()
    return M, barcodes, names


def parse_peaks(names):
    ch = np.array([n.split(":")[0] for n in names])
    st = np.array([int(n.split(":")[1].split("-")[0]) for n in names])
    en = np.array([int(n.split(":")[1].split("-")[1]) for n in names])
    return ch, st, en


_CANON = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY", "chrM"]


def chrom_code(chroms):
    """Deterministic chromosome -> integer code, identical for peaks and for genes."""
    uniq = sorted(set(chroms))
    order = {c: i for i, c in enumerate(_CANON)}
    nxt = len(_CANON)
    for c in uniq:
        if c not in order:
            order[c] = nxt
            nxt += 1
    return np.array([order[c] for c in chroms]), order


def peaks_to_bins(names, bin_size=BIN):
    """For each peak, the genome-bin key(s) it touches (a peak may span two bins)."""
    ch, st, en = parse_peaks(names)
    cc, order = chrom_code(ch)
    b0 = (st // bin_size) * bin_size
    b1 = (en // bin_size) * bin_size
    k0 = cc.astype(np.int64) * 1_000_000_000 + b0
    k1 = cc.astype(np.int64) * 1_000_000_000 + b1
    multi = k1 != k0
    keys = np.concatenate([k0, k1[multi]])
    src = np.concatenate([np.arange(len(names)), np.where(multi)[0]])
    return keys, src


def build_bin_matrix(path, bin_size=BIN):
    """Sum fragment counts into fixed genome bins (a peak spanning two bins is split 1:1)."""
    M, barcodes, names = read_atac_h5(path)
    tot = np.asarray(M.sum(1)).ravel()
    M, barcodes = M[tot >= 1000], barcodes[tot >= 1000]
    if M.shape[0] > CAP:
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(M.shape[0], CAP, replace=False))
        M, barcodes = M[sel], barcodes[sel]
    keys, src = peaks_to_bins(names, bin_size)
    ukeys, inv = np.unique(keys, return_inverse=True)
    P = sp.csr_matrix((np.ones(len(src), dtype=np.float32), (src, inv)),
                      shape=(len(names), len(ukeys)))
    X = (M.tocsr() @ P).tocsr()
    X.sum_duplicates()
    X.data = np.minimum(X.data, 3)              # cap per-bin counts (PCR duplicates)
    return X, ukeys, barcodes


def main():
    t0 = time.time()
    parts, allkeys = [], []
    for f, (grp, rep) in ATAC.items():
        X, binkey, barcodes = build_bin_matrix(os.path.join(L.RAW, f))
        parts.append((grp, rep, f, X, binkey, barcodes))
        allkeys.append(binkey)
        print(f"  {f}: {len(barcodes)} cells, {len(binkey)} bins", flush=True)
    gkeys = np.unique(np.concatenate(allkeys))
    gidx = {k: i for i, k in enumerate(gkeys)}
    print("common bin space:", len(gkeys), flush=True)

    mats, obs = [], []
    for grp, rep, f, X, binkey, barcodes in parts:
        col = np.array([gidx[k] for k in binkey], dtype=np.int64)
        R = sp.csr_matrix((np.ones(len(col), dtype=np.float32), (np.arange(len(col)), col)),
                          shape=(len(col), len(gkeys)))
        mats.append((X @ R).tocsr())
        obs.append(pd.DataFrame({"sample": f.replace(".h5", ""), "group": grp, "rep": rep},
                                index=[f"{f.replace('.h5', '')}#{b}" for b in barcodes]))
    X = sp.vstack(mats).tocsr()
    X.data = (X.data > 0).astype(np.float32)
    A = ad.AnnData(X, obs=pd.concat(obs))

    # keep bins accessible in enough cells AND detected in >=2 samples
    per_sample = []
    for g in A.obs["group"].unique():
        m = (A.obs["group"] == g).to_numpy()
        per_sample.append(np.asarray((A.X[m] > 0).sum(0)).ravel() > 0)
    seen = np.sum(np.vstack(per_sample), axis=0)
    ncell = np.asarray((A.X > 0).sum(0)).ravel()
    keep = np.asarray((seen >= 2) & (ncell >= 20)).ravel()
    A = A[:, keep].copy()
    A.var_names = [str(k) for k in gkeys[keep]]
    print("after shared-bin filter:", A.shape, flush=True)

    tf = TfidfTransformer(norm="l2", sublinear_tf=True)
    Xt = tf.fit_transform(A.X)
    svd = TruncatedSVD(n_components=50, random_state=0)
    Lm = svd.fit_transform(Xt)
    A.obsm["X_lsi"] = Lm[:, 1:]
    print("LSI done", flush=True)
    sc.external.pp.harmony_integrate(A, "sample", basis="X_lsi",
                                     adjusted_basis="X_lsi_harmony", max_iter_harmony=8)
    sc.pp.neighbors(A, use_rep="X_lsi_harmony", n_neighbors=15, random_state=0)
    sc.tl.leiden(A, resolution=1.0, key_added="leiden", random_state=0,
                 flavor="igraph", n_iterations=2)
    print("clusters:", A.obs.leiden.nunique(), flush=True)
    print("cluster x group:\n" + pd.crosstab(A.obs.leiden, A.obs.group).to_string(), flush=True)

    # ---- gene activity from bins within +/- 5 kb of each TSS ----
    genes = pd.read_csv(os.path.join(L.ANN, "mm10_genes_tss.tsv"), sep="\t").reset_index(drop=True)
    var_key = np.array([int(v) for v in A.var_names], dtype=np.int64)
    kd = {int(k): i for i, k in enumerate(var_key)}
    gname = genes.gene.to_numpy()
    cc, order = chrom_code(genes.chrom.tolist())
    rows, cols = [], []
    for i, (cid, tss) in enumerate(zip(cc, genes.tss.to_numpy())):
        base = (int(tss) // BIN) * BIN
        for off in (-BIN, 0, BIN):
            j = kd.get(int(cid) * 1_000_000_000 + base + off)
            if j is not None:
                rows.append(i)
                cols.append(j)
    Agg = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                        shape=(len(genes), A.n_vars))
    act = np.asarray((A.X @ Agg.T).todense(), dtype=np.float32)
    tot = act.sum(1, keepdims=True)
    act = np.log1p(act / np.maximum(tot, 1) * 5000.0)
    gi = {g: i for i, g in enumerate(gname)}
    print("gene activity:", act.shape, flush=True)

    present = {k: [g for g in v if g in gi] for k, v in KEY.items()}
    present = {k: v for k, v in present.items() if len(v) >= 2}
    S = {}
    for k, gs in present.items():
        Mx = act[:, [gi[g] for g in gs]]
        S[k] = ((Mx - Mx.mean(0)) / (Mx.std(0) + 1e-9)).mean(1)
    S = pd.DataFrame(S, index=A.obs_names)
    S["leiden"] = A.obs.leiden.to_numpy()
    cm = S.groupby("leiden", observed=True).mean()
    lab = cm.idxmax(axis=1)
    top = cm.max(axis=1)
    second = cm.apply(lambda r: r.nlargest(2).iloc[-1], axis=1)
    lab[top < 0.25] = "Other"
    A.obs["celltype_atac"] = A.obs.leiden.map(lab).astype(str)
    A.obs["lineage"] = A.obs.celltype_atac.map(L.CLASS).fillna("Other").astype(str)
    rep = pd.DataFrame({"celltype": lab, "top_z": top.round(2), "second_z": second.round(2),
                        "n": A.obs.leiden.value_counts()}).sort_values("n", ascending=False)
    print("\ncluster assignment:\n" + rep.to_string(), flush=True)
    print("\ngroup x lineage:\n" + A.obs.groupby(["group", "lineage"], observed=True)
          .size().to_string(), flush=True)

    # ---- heterochromatin-domain accessibility ----
    def in_domains(keys, bed_path):
        key_chrom = (keys // 1_000_000_000).astype(int)
        key_start = keys % 1_000_000_000
        b = pd.read_csv(bed_path, sep="\t", header=None, names=["chrom", "start", "end"])
        b["cid"] = b.chrom.map(lambda c: order.get(c, -1)).astype(int)
        mask = np.zeros(len(keys), dtype=bool)
        for ci, sub in b.groupby("cid", observed=True):
            if ci < 0:
                continue
            idx = np.where(key_chrom == ci)[0]
            if len(idx) == 0:
                continue
            s = sub.start.to_numpy()
            e = sub.end.to_numpy()
            o2 = np.argsort(s)
            s = s[o2]
            e = np.maximum.accumulate(e[o2])
            en = key_start[idx] + BIN
            pos = np.searchsorted(s, en, side="right") - 1
            ok = pos >= 0
            hit = np.zeros(len(idx), dtype=bool)
            hit[ok] = e[pos[ok]] > key_start[idx][ok]
            mask[idx] = hit
        return mask

    m_het = in_domains(var_key, os.path.join(L.ANN, "domains_H3K9me3.bed"))
    m_lad = in_domains(var_key, os.path.join(L.ANN, "domains_LADproxy.bed"))
    tcell = np.asarray(A.X.sum(1)).ravel()
    A.obs["frac_het"] = np.asarray(A.X[:, m_het].sum(1)).ravel() / np.maximum(tcell, 1)
    A.obs["frac_lad"] = np.asarray(A.X[:, m_lad].sum(1)).ravel() / np.maximum(tcell, 1)
    het_genes = [g for g in L.load_set("ConstitutiveHet") if g in gi]
    A.obs["prom_het"] = act[:, [gi[g] for g in het_genes]].mean(1) if het_genes else 0.0
    A.obs["prom_all"] = act.mean(1)
    A.obs["prom_het_rel"] = A.obs["prom_het"] - A.obs["prom_all"]
    print(f"bins in H3K9me3: {m_het.sum()} | LADproxy: {m_lad.sum()} | het genes: {len(het_genes)}",
          flush=True)

    out = os.path.join(L.RES, "ATAC")
    os.makedirs(out, exist_ok=True)
    A.write_h5ad(os.path.join(out, "atac_processed.h5ad"), compression="gzip")
    A.obs.to_csv(os.path.join(out, "atac_cell_metrics.csv.gz"), compression="gzip",
                 float_format="%.5g")
    rep.to_csv(os.path.join(out, "atac_cluster_assignment.csv"))
    np.savez_compressed(os.path.join(out, "gene_activity.npz"), act=act, genes=gname)

    cols = ["frac_het", "frac_lad", "prom_het_rel"]
    summ = A.obs.groupby(["group", "lineage"], observed=True)[cols].agg(["mean", "count"])
    summ.to_csv(os.path.join(out, "het_accessibility_by_cluster.csv"))
    print("\n" + summ.round(4).to_string(), flush=True)

    rows2 = []
    for l in ["Endothelial", "Mesenchymal", "Parenchyma", "Immune", "Other"]:
        for col in cols:
            c = A.obs.loc[(A.obs.lineage == l) & (A.obs.group == "Control"), col].dropna()
            t = A.obs.loc[(A.obs.lineage == l) & (A.obs.group == "TCDD_16wk_WT"), col].dropna()
            k = A.obs.loc[(A.obs.lineage == l) & (A.obs.group == "TCDD_16wk_AhrKO"), col].dropna()
            if len(c) < 40 or len(t) < 40:
                continue
            rows2.append({"lineage": l, "metric": col, "n_control": len(c), "n_wt": len(t),
                          "n_ahrko": len(k), "mean_control": c.mean(), "mean_wt": t.mean(),
                          "mean_ahrko": k.mean() if len(k) >= 40 else np.nan,
                          "d_wt_vs_control": L.cohens_d(t, c),
                          "p_wt_vs_control": L.perm_p(t, c, n=2000)})
    e = pd.DataFrame(rows2)
    if not e.empty:
        e["fdr"] = L.bh(e.p_wt_vs_control.to_numpy())
        e.to_csv(os.path.join(out, "atac_lineage_effects.csv"), index=False)
        print("\n" + e.round(4).to_string(), flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
