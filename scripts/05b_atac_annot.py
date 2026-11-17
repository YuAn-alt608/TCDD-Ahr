"""Re-annotate the scATAC object from the cached gene-activity matrix.

Two fixes over the first pass:
  * peak->gene now returns the original gene row (an earlier version returned the rank inside
    the TSS-sorted chromosome block, which collapsed all signal onto the first few genes);
  * marker sets are scored by the mean of per-gene z-scores instead of raw mean activity, so a
    set with globally high accessibility no longer wins every cluster.
"""
import os, sys
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

RES = os.path.join(L.RES, "ATAC")
# marker sets checked by hand on this tissue; keeps the annotation interpretable
KEY = {
    "SMC": ["Myh11", "Acta2", "Tagln", "Cnn1", "Myocd"],
    "Fibroblast": ["Pdgfra", "Col1a1", "Col1a2", "Dcn", "Postn", "Lum"],
    "Endothelial": ["Pecam1", "Cdh5", "Emcn", "Vwf", "Klf2"],
    "Macrophage": ["Csf1r", "Cd68", "Lyz2", "Adgre1", "C1qa"],
    "Tcell": ["Cd3e", "Cd3d", "Lck", "Cd2"],
    "Bcell": ["Cd79a", "Cd79b", "Ms4a1"],
    "Adipocyte": ["Adipoq", "Plin1", "Fabp4", "Lep"],
}


def in_domains(names, bed_path):
    ch = np.array([n.split(":")[0] for n in names])
    st = np.array([int(n.split(":")[1].split("-")[0]) for n in names])
    en = np.array([int(n.split(":")[1].split("-")[1]) for n in names])
    mask = np.zeros(len(names), dtype=bool)
    if not os.path.exists(bed_path):
        return mask
    bed = pd.read_csv(bed_path, sep="\t", header=None, names=["chrom", "start", "end"])
    for chrom, sub in bed.groupby("chrom", observed=True):
        idx = np.where(ch == chrom)[0]
        if len(idx) == 0:
            continue
        s = sub.start.to_numpy()
        e = sub.end.to_numpy()
        o = np.argsort(s)
        s = s[o]
        e = np.maximum.accumulate(e[o])
        pos = np.searchsorted(s, en[idx], side="right") - 1
        ok = pos >= 0
        hit = np.zeros(len(idx), dtype=bool)
        hit[ok] = e[pos[ok]] > st[idx][ok]
        mask[idx] = hit
    return mask


def main():
    a = sc.read_h5ad(os.path.join(RES, "atac_processed.h5ad"))
    z = np.load(os.path.join(RES, "gene_activity.npz"), allow_pickle=True)
    act, gn = z["act"], list(z["genes"])
    print(a.shape, "activity", act.shape, flush=True)
    gi = {g: i for i, g in enumerate(gn)}

    present = {k: [g for g in v if g in gi] for k, v in KEY.items()}
    present = {k: v for k, v in present.items() if len(v) >= 2}
    for k, v in present.items():
        print(f"  {k}: {v}", flush=True)

    # per-gene z across cells, then average within the set
    S = {}
    for k, gs in present.items():
        M = act[:, [gi[g] for g in gs]]
        S[k] = ((M - M.mean(0)) / (M.std(0) + 1e-9)).mean(1)
    S = pd.DataFrame(S, index=a.obs_names)
    S["leiden"] = a.obs.leiden.to_numpy()
    cm = S.groupby("leiden", observed=True).mean()
    lab = cm.idxmax(axis=1)
    top = cm.max(axis=1)
    second = cm.apply(lambda r: r.nlargest(2).iloc[-1], axis=1)
    # a cluster is labelled only when the winning set is clearly enriched (top z >= 0.10 and a
    # margin of >= 0.08 over the runner-up); otherwise it stays unassigned and is excluded
    lab[(top < 0.10) | ((top - second) < 0.08)] = "Other"
    a.obs["celltype_atac"] = a.obs.leiden.map(lab).astype(str)
    a.obs["lineage"] = a.obs.celltype_atac.map(L.CLASS).fillna("Other").astype(str)

    rep = pd.DataFrame({"celltype": lab, "top_z": top.round(2),
                        "second_z": second.round(2),
                        "margin": (top - second).round(2),
                        "n": a.obs.leiden.value_counts()}).sort_values("n", ascending=False)
    print("\ncluster assignment:\n" + rep.to_string(), flush=True)
    print("\ngroup x lineage:\n" + a.obs.groupby(["group", "lineage"], observed=True)
          .size().to_string(), flush=True)

    # frac_het / frac_lad / prom_het_rel are already computed on the bin matrix in 05_atac.py
    need = {"frac_het", "frac_lad", "prom_het_rel"}
    missing = need - set(a.obs.columns)
    if missing:
        raise SystemExit(f"missing columns computed upstream: {missing}")

    a.write_h5ad(os.path.join(RES, "atac_processed.h5ad"), compression="gzip")
    a.obs.to_csv(os.path.join(RES, "atac_cell_metrics.csv.gz"), compression="gzip",
                 float_format="%.5g")
    rep.to_csv(os.path.join(RES, "atac_cluster_assignment.csv"))

    cols = ["frac_het", "frac_lad", "prom_het_rel"]
    summ = a.obs.groupby(["group", "lineage"], observed=True)[cols].agg(["mean", "count"])
    summ.to_csv(os.path.join(RES, "het_accessibility_by_cluster.csv"))
    print("\n" + summ.round(4).to_string(), flush=True)

    rows = []
    for l in ["SMC", "Fibroblast", "Endothelial", "Parenchyma", "Mesenchymal", "Immune"]:
        for col in cols:
            c = a.obs.loc[(a.obs.lineage == l) & (a.obs.group == "Control"), col].dropna()
            t = a.obs.loc[(a.obs.lineage == l) & (a.obs.group == "TCDD_16wk_WT"), col].dropna()
            k = a.obs.loc[(a.obs.lineage == l) & (a.obs.group == "TCDD_16wk_AhrKO"), col].dropna()
            if len(c) < 40 or len(t) < 40:
                continue
            rows.append({"lineage": l, "metric": col, "n_control": len(c), "n_wt": len(t),
                         "n_ahrko": len(k), "mean_control": c.mean(), "mean_wt": t.mean(),
                         "mean_ahrko": k.mean() if len(k) >= 40 else np.nan,
                         "d_wt_vs_control": L.cohens_d(t, c),
                         "p_wt_vs_control": L.perm_p(t, c, n=2000)})
    e = pd.DataFrame(rows)
    if not e.empty:
        e["fdr"] = L.bh(e.p_wt_vs_control.to_numpy())
        e.to_csv(os.path.join(RES, "atac_lineage_effects.csv"), index=False)
        print("\n" + e.round(4).to_string(), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
