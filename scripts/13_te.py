"""Transposable-element layer.

What cannot be done here, stated up front: TE family-level transcript quantification
needs reads (FASTQ/BAM) and scTE/Telescope. GSE276872 deposits only filtered
barcode matrices (gene level), so no TE-multimapping read can be counted. Buying the
BAMs would mean ~1.2 TB and a STAR re-alignment; the spec is in scripts/TE_count_readme.md.

What *is* computable: genes whose promoter sits inside a given TE family are the
population of genes whose transcription is directly under that family's LTR control.
Their derepression is a family-resolved, promoter-level proxy for TE unleashing, and
it can be tested against the cGAS-STING / ISG module in the same cells.

Outputs: results/te/
"""
import os, sys, json
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

OUT = os.path.join(L.RES, "te")
os.makedirs(OUT, exist_ok=True)
RMSK = os.path.join(os.path.dirname(L.ANN), "mm10_rmsk.txt.gz")
TSS = os.path.join(L.ANN, "mm10_genes_tss.tsv")
W = 1000

FAMILIES = {
    "IAP":      lambda n, f: n.str.upper().str.startswith("IAP"),
    "ETn":      lambda n, f: n.str.upper().str.startswith("ETN") | n.str.upper().str.contains("MUSD"),
    "ERVK_oth": lambda n, f: (f == "ERVK") & ~n.str.upper().str.startswith("IAP") & ~n.str.upper().str.startswith("ETN"),
    "ERVL":     lambda n, f: f.isin(["ERVL", "ERVL-MaLR"]),
    "ERV1":     lambda n, f: f == "ERV1",
    "L1":       lambda n, f: f == "L1",
    "SINE_B":   lambda n, f: f.isin(["B1", "B2", "Alu", "MIR", "SINE"]),
}

ISG = ["Isg15", "Ifit1", "Ifit3", "Irf7", "Stat1", "Cxcl10", "Oasl2", "Mx1", "Ifi204", "Usp18"]
CGAS = ["Cgas", "Tmem173", "Sting1", "Tbk1", "Irf3", "Irf7", "Nlrp3", "Aim2", "Zbp1", "Ddx58"]


def load_rmsk():
    cols = ["bin", "swScore", "milliDiv", "milliDel", "milliIns", "chrom", "start", "end",
            "genoLeft", "strand", "repName", "repClass", "repFamily", "repStart",
            "repEnd", "repLeft", "id"]
    df = pd.read_csv(RMSK, sep="\t", header=None, usecols=range(17), names=cols,
                     dtype={"chrom": str, "repClass": str, "repFamily": str})
    df = df.dropna(subset=["start", "end"])
    df["start"] = df.start.astype("int64")
    df["end"] = df.end.astype("int64")
    keep = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"]
    return df[df.chrom.isin(keep) & df.repClass.isin(["LTR", "LINE", "SINE", "DNA"])]


def promoter_families():
    tes = load_rmsk()
    tes["repName"] = tes.repName.astype(str)
    tes["repFamily"] = tes.repFamily.astype(str)
    tss = pd.read_csv(TSS, sep="\t")
    tss["s"] = tss.tss - W
    tss["e"] = tss.tss + W
    out = {}
    for fam, fn in FAMILIES.items():
        sub = tes[fn(tes.repName, tes.repFamily)]
        genes = set()
        for chrom, g in tss.groupby("chrom"):
            t = sub[sub.chrom == chrom]
            if t.empty:
                continue
            ts = t.start.to_numpy()
            te = t.end.to_numpy()
            o = np.argsort(ts)
            ts, te = ts[o], te[o]
            te = np.maximum.accumulate(te)
            pos = np.searchsorted(ts, g.e.to_numpy(), side="right") - 1
            ok = pos >= 0
            hit = np.zeros(len(g), bool)
            hit[ok] = te[pos[ok]] > g.s.to_numpy()[ok]
            genes |= set(g.gene.to_numpy()[hit])
        out[fam] = {"n_te": int(len(sub)), "n_genes": len(genes), "genes": genes}
        print(f"  {fam}: {len(sub)} elements -> {len(genes)} promoter genes", flush=True)
    return out


def main():
    a = sc.read_h5ad(os.path.join(L.RES, "D2", "processed.h5ad"))
    genes = list(a.var_names)
    gi = {g: i for i, g in enumerate(genes)}
    ct = a.obs["celltype"].astype(str).to_numpy()
    grp = a.obs["group"].astype(str).to_numpy()
    X = a.X

    fams = promoter_families()
    pd.DataFrame([{"family": k, "n_elements": v["n_te"], "n_promoter_genes": v["n_genes"]}
                  for k, v in fams.items()]).to_csv(os.path.join(OUT, "te_family_sets.csv"), index=False)
    with open(os.path.join(OUT, "te_family_genes.tsv"), "w", encoding="utf-8") as f:
        f.write("family\tgene\n")
        for k, v in fams.items():
            for g in sorted(v["genes"]):
                f.write(f"{k}\t{g}\n")

    # ---- family proxy scores per cell
    scores = {}
    for fam, v in fams.items():
        idx = [gi[g] for g in v["genes"] if g in gi]
        if len(idx) < 5:
            continue
        scores[fam] = np.asarray(X[:, idx].mean(1)).ravel()
        print(f"  {fam}: scored on {len(idx)} detected genes", flush=True)
    sc_df = pd.DataFrame(scores, index=a.obs_names)
    sc_df["celltype"] = ct
    sc_df["group"] = grp

    # ---- effect sizes TCDD vs control, and Ahr-KO reversal
    rows = []
    for fam in scores:
        for c in pd.unique(ct):
            if c == "Other":
                continue
            m1 = (ct == c) & (grp == "Control")
            m2 = (ct == c) & (grp == "TCDD_16wk_WT")
            m3 = (ct == c) & (grp == "TCDD_16wk_AhrKO")
            v = sc_df[fam].to_numpy()
            rows.append({"family": fam, "celltype": c,
                         "mean_ctrl": float(v[m1].mean()) if m1.sum() else np.nan,
                         "mean_tcdd": float(v[m2].mean()) if m2.sum() else np.nan,
                         "mean_ako": float(v[m3].mean()) if m3.sum() else np.nan,
                         "d_tcdd_vs_ctrl": L.cohens_d(v[m2], v[m1]) if m1.sum() > 20 and m2.sum() > 20 else np.nan,
                         "p": L.perm_p(v[m2], v[m1], n=500) if m1.sum() > 20 and m2.sum() > 20 else np.nan,
                         "d_ako_vs_wt": L.cohens_d(v[m3], v[m2]) if m2.sum() > 20 and m3.sum() > 20 else np.nan})
    eff = pd.DataFrame(rows)
    eff["fdr"] = np.nan
    for fam, sub in eff.groupby("family"):
        eff.loc[sub.index, "fdr"] = L.bh(sub.p.fillna(1).to_numpy())
    eff.to_csv(os.path.join(OUT, "te_family_effects.csv"), index=False)
    print(eff.sort_values("d_tcdd_vs_ctrl", ascending=False).head(8).to_string(), flush=True)

    # ---- cGAS-STING pathway presence (is the sensor even expressed?)
    cg = [g for g in CGAS if g in gi]
    cg_rows = []
    for c in pd.unique(ct):
        m = ct == c
        if m.sum() < 30:
            continue
        for g in cg:
            v = np.asarray(X[m][:, gi[g]].todense()).ravel()
            cg_rows.append({"celltype": c, "gene": g, "mean": float(v.mean()),
                            "frac_detected": float((v > 0).mean())})
    cgd = pd.DataFrame(cg_rows)
    cgd.to_csv(os.path.join(OUT, "cgas_sting_expression.csv"), index=False)
    print(cgd.pivot_table(index="gene", columns="celltype", values="frac_detected").round(3).to_string(), flush=True)

    # ---- TE proxy vs ISG coupling, per cell type and group
    isg_idx = [gi[g] for g in ISG if g in gi]
    isg = np.asarray(X[:, isg_idx].mean(1)).ravel()
    corr = []
    for fam in scores:
        for c in pd.unique(ct):
            if c == "Other":
                continue
            for gname, m in (("Control", (ct == c) & (grp == "Control")),
                             ("TCDD_16wk_WT", (ct == c) & (grp == "TCDD_16wk_WT")),
                             ("TCDD_16wk_AhrKO", (ct == c) & (grp == "TCDD_16wk_AhrKO"))):
                if m.sum() < 40:
                    continue
                rho, p = stats.spearmanr(sc_df[fam].to_numpy()[m], isg[m])
                corr.append({"family": fam, "celltype": c, "group": gname,
                             "rho_vs_ISG": float(rho), "p": float(p), "n": int(m.sum())})
    corr = pd.DataFrame(corr)
    corr.to_csv(os.path.join(OUT, "te_proxy_vs_isg.csv"), index=False)
    top = corr[(corr.group == "TCDD_16wk_WT")].sort_values("rho_vs_ISG", ascending=False).head(6)
    print(top.to_string(), flush=True)

    summary = {
        "families": {k: {"n_elements": v["n_te"], "n_genes": v["n_genes"]} for k, v in fams.items()},
        "note": "TE family activity is a promoter-gene proxy, not TE transcript counting; "
                "no BAM/FASTQ in GEO, see scripts/TE_count_readme.md",
        "n_cells": int(a.shape[0]),
    }
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("done")


if __name__ == "__main__":
    main()
