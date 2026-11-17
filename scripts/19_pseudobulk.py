"""19 - Pseudobulk (animal-level) re-analysis of the two headline claims.

Cell-level tests on 30k cells inflate significance because cells are not
replicates. Here every metric is recomputed with the *mouse* as the unit:

  RNA  : counts summed per (mouse x celltype) -> CPM -> log1p; module scores are
         the mean log-CPM over the module's genes. Tests are only run where the
         design supports them (Control n=3 vs TCDD-8wk n=3, Welch t, BH).
  ATAC : cells averaged per (library x lineage). The 16-wk and Ahr-KO arms are a
         single animal each, so no test is possible; published contrasts are
         reported as per-animal values plus a z that uses the *between-control-
         mouse* SD (df = 1) - explicitly labelled as an estimate, not a test.

This script exists to make the replication structure explicit in the paper: the
8-week arm has n=3 mice, the 16-week WT and Ahr-KO arms have n=1 each. Anything
that needs a KO-vs-WT p-value cannot be computed at animal level and must be
either softened in the text or supported by a second cohort.

Outputs (results/pseudobulk/):
  rna_pseudobulk_cpm.csv        mouse x celltype, log CPM
  rna_module_by_mouse.csv       module scores per mouse x celltype
  rna_module_tests.csv          group means, d, Welch p (3v3 only)
  rna_gene_tests.csv            per-gene tests for the reported panel
  cell_vs_pseudobulk_p.csv      p-value inflation table
  atac_by_library.csv           per-library lineage means
  atac_animal_level.csv         per-lineage effects + within-control SD
  summary.json
"""
import os, sys, json, re
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

OUT = os.path.join(L.RES, "pseudobulk")
os.makedirs(OUT, exist_ok=True)
GROUPS = ["Control", "TCDD_8wk", "TCDD_16wk_WT", "TCDD_16wk_AhrKO"]

LAM = ["Trem2", "Lgals3", "Csf1r", "Spp1", "Apoe", "Cd9", "Fabp5", "Lpl", "Gpnmb", "Trem1"]
EC_READ = ["Klf2", "Smad2", "Nr1h3", "Hif1a"]
MATRIX = ["Timp1", "Fn1", "Col1a2", "Col3a1", "Spp1", "Cxcl12", "Il6", "Ccl2"]
SASP = ["Il6", "Cxcl1", "Cxcl2", "Mmp3", "Mmp9", "Mmp13", "Serpine1", "Igfbp3", "Igfbp5",
        "Tgfb1", "Ccl2", "Cxcl12"]


def mouse_of(sample):
    p = str(sample).split("_")
    # D2_rna_CTR_TDT1 -> CTR1 ; D2_rna_WT16_TDT -> WT16 ; D2_rna_AKO16_NTDT -> AKO16
    if len(p) >= 4 and p[2].endswith("16"):
        return p[2]
    if len(p) >= 4:
        return p[2] + re.sub(r"^(TDT|NTDT)", "", p[3])
    return str(sample)


def cohend(a, b):
    """Cohen's d with a pooled SD; allows n>=2 (L.cohens_d needs n>=5, i.e. cells)."""
    a = np.asarray(a, float)[np.isfinite(a)]; b = np.asarray(b, float)[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if sp2 <= 0:
        return np.nan
    return float((a.mean() - b.mean()) / np.sqrt(sp2))


def welch(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return stats.ttest_ind(a, b, equal_var=False).pvalue


def main():
    # ------------------------------------------------------------------ RNA
    a = ad.read_h5ad(os.path.join(L.RES, "D2", "processed.h5ad"))
    cnt = a.layers["counts"]
    cnt = sp.csr_matrix(cnt)
    genes = np.asarray(a.var_names)
    obs = a.obs.copy()
    obs["mouse"] = obs["sample"].map(mouse_of)
    obs["key"] = obs["mouse"].astype(str) + "|" + obs["celltype"].astype(str) + "|" + \
        obs["group"].astype(str)
    print(obs.groupby(["mouse", "celltype"]).size().unstack(fill_value=0).to_string(), flush=True)

    keys = obs["key"].to_numpy()
    ukeys, inv = np.unique(keys, return_inverse=True)
    ind = sp.csr_matrix((np.ones(len(inv), dtype=np.float32), (inv, np.arange(len(inv)))),
                        shape=(len(ukeys), len(inv)))
    pb = np.asarray((ind @ cnt).todense(), dtype=np.float64)   # (n_key, n_gene)
    ncell = np.asarray(ind.sum(1)).ravel()
    lib = pb.sum(1, keepdims=True)
    cpm = np.log1p(pb / np.maximum(lib, 1) * 1e6)
    cpmdf = pd.DataFrame(cpm, index=ukeys, columns=genes)
    meta = pd.DataFrame([k.split("|") for k in ukeys], columns=["mouse", "celltype", "group"])
    meta["n_cells"] = ncell.astype(int)
    meta.index = ukeys
    cpmdf.index.name = "key"
    out = pd.concat([meta, cpmdf], axis=1)
    out.to_csv(os.path.join(OUT, "rna_pseudobulk_cpm.csv"), float_format="%.4g")

    gidx = {g: i for i, g in enumerate(genes)}
    het = sorted(set(L.load_set("ConstitutiveHet")) & set(gidx))
    isg = sorted(set(L.load_set("IFN_dsRNA")) & set(gidx)) if L.load_set("IFN_dsRNA") else \
        [g for g in """Isg15 Ifit1 Ifit3 Irf7 Stat1 Bst2 Rsad2 Oasl1 Usp18 Cxcl10 Mx1""".split()
         if g in gidx]

    modules = {
        "het_derepression": [g for g in het if g in gidx],
        "ISG": isg,
        "LAM": [g for g in LAM if g in gidx],
        "matrix_ligand": [g for g in MATRIX if g in gidx],
        "SASP": [g for g in SASP if g in gidx],
        "EC_readout": [g for g in EC_READ if g in gidx],
    }
    mod_rows = []
    for name, gl in modules.items():
        v = cpmdf[gl].mean(1).to_numpy()
        for i, k in enumerate(ukeys):
            mod_rows.append(dict(mouse=meta.mouse.iloc[i], celltype=meta.celltype.iloc[i],
                                 group=meta.group.iloc[i], module=name, score=float(v[i]),
                                 n_cells=int(ncell[i])))
    moddf = pd.DataFrame(mod_rows)
    moddf.to_csv(os.path.join(OUT, "rna_module_by_mouse.csv"), index=False)

    # module tests (only where n>=3 per group)
    trows = []
    for (ct, mod), sub in moddf.groupby(["celltype", "module"]):
        m = sub.groupby("group").score.agg(["mean", "std", "count"])
        if "Control" not in m.index:
            continue
        c = sub[sub.group == "Control"].score
        w = sub[sub.group == "TCDD_8wk"].score
        w16 = sub[sub.group == "TCDD_16wk_WT"].score
        ko = sub[sub.group == "TCDD_16wk_AhrKO"].score
        trows.append(dict(celltype=ct, module=mod,
                          n_ctrl=len(c), n_8wk=len(w), n_16wk=len(w16), n_ako=len(ko),
                          mean_ctrl=c.mean(), mean_8wk=w.mean() if len(w) else np.nan,
                          mean_16wk=w16.mean() if len(w16) else np.nan,
                          mean_ako=ko.mean() if len(ko) else np.nan,
                          d_8wk=cohend(w.to_numpy(), c.to_numpy()),
                          p_8wk=welch(w, c),
                          delta_ako=float(ko.mean() - w16.mean()) if len(ko) and len(w16) else np.nan))
    tdf = pd.DataFrame(trows)
    tdf["fdr_8wk"] = np.nan
    for ct, sub in tdf.groupby("celltype"):
        tdf.loc[sub.index, "fdr_8wk"] = L.bh(sub.p_8wk.to_numpy())
    tdf.to_csv(os.path.join(OUT, "rna_module_tests.csv"), index=False)
    print("\nmodule tests (animal level):\n" + tdf.round(3).to_string(index=False), flush=True)

    # per-gene tests for the reported panel
    panel = ["Trem2", "Lgals3", "Csf1r", "Timp1", "Fn1", "Klf2", "Smad2", "Col1a2",
             "Cxcl12", "Ahr", "Cyp1a1", "Cyp1b1", "Spp1", "Lpl"]
    panel = [g for g in panel if g in gidx]
    grows = []
    for ct in ["SMC", "ModifiedSMC", "Macrophage", "Endothelial", "Fibroblast"]:
        sub = out[out.celltype == ct]
        if len(sub) == 0:
            continue
        for g in panel:
            c = sub[sub.group == "Control"][g].to_numpy()
            w = sub[sub.group == "TCDD_8wk"][g].to_numpy()
            w16 = sub[sub.group == "TCDD_16wk_WT"][g].to_numpy()
            ko = sub[sub.group == "TCDD_16wk_AhrKO"][g].to_numpy()
            grows.append(dict(celltype=ct, gene=g, n_ctrl=len(c), n_8wk=len(w),
                              mean_ctrl=c.mean() if len(c) else np.nan,
                              mean_8wk=w.mean() if len(w) else np.nan,
                              mean_16wk=w16.mean() if len(w16) else np.nan,
                              mean_ako=ko.mean() if len(ko) else np.nan,
                              d_8wk=cohend(w, c), p_8wk=welch(w, c),
                              delta_ako=float(ko.mean() - w16.mean()) if len(ko) and len(w16)
                              else np.nan))
    gdf = pd.DataFrame(grows)
    gdf["fdr_8wk"] = np.nan
    for ct, sub in gdf.groupby("celltype"):
        gdf.loc[sub.index, "fdr_8wk"] = L.bh(sub.p_8wk.to_numpy())
    gdf.to_csv(os.path.join(OUT, "rna_gene_tests.csv"), index=False)

    # ------------------------------------------------- p-value inflation table
    cm = pd.read_csv(os.path.join(L.RES, "D2", "cell_metrics.csv.gz"), index_col=0)
    cm = cm.join(obs[["mouse"]], how="left")
    irows = []
    for ct, met in [("Endothelial", "LI_ConstitutiveHet"), ("SMC", "LI_ConstitutiveHet"),
                    ("Macrophage", "mod_AgingSen"), ("Macrophage", "LI_H3K9me3")]:
        v = cm[cm.celltype == ct][["group", met, "mouse"]].dropna()
        if len(v) == 0:
            continue
        cell_p = stats.mannwhitneyu(v[v.group == "Control"][met], v[v.group == "TCDD_8wk"][met]
                                    ).pvalue if (v.group == "TCDD_8wk").sum() > 5 else np.nan
        pv = v.groupby(["mouse", "group"])[met].mean().reset_index()
        a1 = pv[pv.group == "Control"][met].to_numpy()
        b1 = pv[pv.group == "TCDD_8wk"][met].to_numpy()
        irows.append(dict(celltype=ct, metric=met, n_cells_8wk=int((v.group == "TCDD_8wk").sum()),
                          p_cell_level=cell_p, n_mice_ctrl=len(a1), n_mice_8wk=len(b1),
                          p_animal_level=welch(b1, a1),
                          d_animal_level=cohend(b1, a1)))
    idf = pd.DataFrame(irows)
    idf.to_csv(os.path.join(OUT, "cell_vs_pseudobulk_p.csv"), index=False)
    print("\ninflation check:\n" + idf.round(4).to_string(index=False), flush=True)

    # ------------------------------------------------------------------ ATAC
    am = pd.read_csv(os.path.join(L.RES, "ATAC", "atac_cell_metrics.csv.gz"), index_col=0)
    am["mouse"] = am["sample"].map(lambda s: {"D2_atac_CTRL1": "CTRL1", "D2_atac_CTRL2": "CTRL2",
                                             "D2_atac_WT_TCDD": "WT16",
                                             "D2_atac_AHRKO_TCDD": "AKO16"}.get(s, str(s)))
    dev = pd.read_csv(os.path.join(L.RES, "motif", "motif_deviation_per_cell.csv.gz"), index_col=0)
    keep = [c for c in ["Ahr::Arnt", "KLF2", "SMAD2", "ATF2", "FOSL1::JUND", "HIF1A",
                        "Nr1h3", "RELA", "Nfe2l2"] if c in dev.columns]
    am = am.join(dev[keep], how="left")

    metrics = ["frac_het", "frac_lad", "prom_het_rel"] + keep
    lrows = []
    for (smp, ln), sub in am.groupby(["mouse", "lineage"]):
        rec = dict(mouse=smp, lineage=ln, n_cells=int(len(sub)), group=sub.group.iloc[0])
        for m in metrics:
            rec[m] = float(sub[m].mean()) if m in sub else np.nan
        lrows.append(rec)
    ldf = pd.DataFrame(lrows)
    ldf.to_csv(os.path.join(OUT, "atac_by_library.csv"), index=False)
    print("\nATAC per library:\n" + ldf.round(4).to_string(index=False), flush=True)

    arows = []
    for ln, sub in ldf.groupby("lineage"):
        rec = dict(lineage=ln)
        for m in metrics:
            c = sub[sub.group == "Control"][m].to_numpy()
            w = sub[sub.group == "TCDD_16wk_WT"][m].to_numpy()
            k = sub[sub.group == "TCDD_16wk_AhrKO"][m].to_numpy()
            rec[f"{m}_ctrl"] = float(np.mean(c)) if len(c) else np.nan
            rec[f"{m}_wt"] = float(np.mean(w)) if len(w) else np.nan
            rec[f"{m}_ako"] = float(np.mean(k)) if len(k) else np.nan
            rec[f"{m}_sd_ctrl"] = float(np.std(c, ddof=1)) if len(c) > 1 else np.nan
            # z of the animal-level mean difference using the between-control-mouse SD.
            # df = 1; this is an effect-size estimate, NOT a test.
            if len(w) and len(c) > 1 and np.std(c, ddof=1) > 0:
                rec[f"{m}_z_wt_vs_ctrl"] = float((np.mean(w) - np.mean(c)) /
                                                 (np.std(c, ddof=1) * np.sqrt(1 / len(w) + 1 / len(c))))
            else:
                rec[f"{m}_z_wt_vs_ctrl"] = np.nan
            rec[f"{m}_delta_ako"] = float(np.mean(k) - np.mean(w)) if len(k) and len(w) else np.nan
        arows.append(rec)
    adf = pd.DataFrame(arows)
    adf.to_csv(os.path.join(OUT, "atac_animal_level.csv"), index=False)

    sm = {
        "design": {
            "rna": obs.groupby(["group", "mouse"]).size().reset_index()
                     .groupby("group").mouse.nunique().to_dict(),
            "rna_cells_per_group": obs.group.value_counts().to_dict(),
            "atac": am.groupby("group").mouse.nunique().to_dict(),
        },
        "note": ("16-week WT and Ahr-KO are a single animal each; animal-level tests are "
                 "reported only for Control vs 8-week (n=3 vs 3). KO contrasts are reported "
                 "as per-animal values with a df=1 z, and are labelled as estimates."),
        "atac_cells_per_library": am.groupby("mouse").size().to_dict(),
    }
    json.dump(sm, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=str)
    print(json.dumps(sm, indent=2, default=str))
    print("done")


if __name__ == "__main__":
    main()
