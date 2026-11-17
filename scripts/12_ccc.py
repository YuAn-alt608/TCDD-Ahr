"""Cell-cell communication: does SMC-specific Ahr deletion pull the niche back via secreted factors?

Two structural constraints decide the design:

1. CellChat / NicheNet need R or a trained prior; here the ligand-receptor prior comes from
   OmniPath `ligrecextra` (mouse) and the score uses the CellChat Hill form.
2. The tdT+ and tdT- fractions are the two halves of the *same* heart, sorted apart. SMC are
   tdT+ and macrophages are tdT-, so the two cell types never co-occur in one sample. A
   per-sample sender->receiver score is therefore impossible, and pretending otherwise would be
   fabrication. Everything is computed at the **group** level (Control, TCDD 8wk, TCDD 16wk WT,
   TCDD 16wk AhrKO), pooling both fractions of the same animals, and reported as effect sizes
   without p-values (one pooled observation per group).

Outputs: results/ccc/
"""
import os, sys, io, json, ssl, urllib.request
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

OUT = os.path.join(L.RES, "ccc")
os.makedirs(OUT, exist_ok=True)
CTX = ssl.create_default_context()
GROUPS = ["Control", "TCDD_8wk", "TCDD_16wk_WT", "TCDD_16wk_AhrKO"]

SASP = ["Il6", "Il1a", "Il1b", "Ccl2", "Ccl5", "Ccl7", "Ccl8", "Cxcl1", "Cxcl2", "Cxcl10",
        "Cxcl12", "Mmp3", "Mmp9", "Mmp13", "Serpine1", "Igfbp3", "Igfbp7", "Tgfb1",
        "Vegfa", "Pdgfb", "Csf1", "Ccl3", "Ccl4", "Tnf", "Lif"]
INFLAM = ["Il1b", "Nlrp3", "Ccl2", "Ccl3", "Ccl4", "Cxcl2", "Tnf", "Il6", "Nfkbiz",
          "Nfkbia", "Icam1", "Vcam1", "Ptgs2", "Spp1", "Trem2", "Apoe", "Lgals3"]
ISG = ["Isg15", "Ifit1", "Ifit3", "Irf7", "Stat1", "Cxcl10", "Oasl2", "Mx1", "Ifi204", "Usp18"]
CHEMO = ["Ccl2", "Ccl3", "Ccl4", "Ccl5", "Ccl7", "Cxcl1", "Cxcl2", "Cxcl10", "Cxcl12",
         "Il6", "Tgfb1", "Csf1"]


def fetch_lr():
    p = os.path.join(OUT, "omnipath_ligrecextra_mouse.tsv")
    if os.path.exists(p):
        return pd.read_csv(p, sep="\t")
    url = ("https://omnipathdb.org/interactions?datasets=ligrecextra&genesymbols=1"
           "&organisms=10090&fields=genesymbols,organisms,sources,curation_effort&format=tsv")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                timeout=180, context=CTX) as r:
        raw = r.read().decode()
    open(p, "w", encoding="utf-8").write(raw)
    return pd.read_csv(io.StringIO(raw), sep="\t")


def main():
    a = sc.read_h5ad(os.path.join(L.RES, "D2", "processed.h5ad"))
    genes = list(a.var_names)
    gi = {g: i for i, g in enumerate(genes)}
    ct = a.obs["celltype"].astype(str).to_numpy()
    grp = a.obs["group"].astype(str).to_numpy()
    X = a.X

    frac = a.obs["sample"].astype(str).str.extract(r"(NTDT|TDT)")[0].to_numpy()
    print(pd.crosstab(pd.Series(ct, name="celltype"), pd.Series(frac, name="fraction")).to_string(),
          flush=True)

    celltypes = [c for c in pd.unique(ct) if c != "Other"]
    rows, idx = [], []
    for g in GROUPS:
        for c in celltypes:
            m = (grp == g) & (ct == c)
            if m.sum() < 10:
                continue
            rows.append(np.asarray(X[m].mean(0)).ravel())
            idx.append((g, c))
    pb = pd.DataFrame(np.vstack(rows),
                      index=pd.MultiIndex.from_tuples(idx, names=["group", "celltype"]),
                      columns=genes)
    pb.to_csv(os.path.join(OUT, "pseudobulk_group_celltype.csv.gz"), compression="gzip",
              float_format="%.4g")
    print("pseudobulk", pb.shape, flush=True)

    lr = fetch_lr().dropna(subset=["source_genesymbol", "target_genesymbol"])
    lr = lr[lr.source_genesymbol.isin(gi) & lr.target_genesymbol.isin(gi)]
    lr = lr.drop_duplicates(["source_genesymbol", "target_genesymbol"])
    lr.to_csv(os.path.join(OUT, "lr_pairs_used.csv"), index=False)
    print("LR pairs with both partners detected:", len(lr), flush=True)

    def hill(v, K=0.5):
        v = v / (v.max() + 1e-9) if v.max() > 0 else v
        return v ** 2 / (K ** 2 + v ** 2)

    recs = []
    for g in pb.index.get_level_values("group").unique():
        sub = pb.loc[g]
        cts = list(sub.index)
        Lg = hill(sub[lr.source_genesymbol.tolist()].to_numpy())
        Rg = hill(sub[lr.target_genesymbol.tolist()].to_numpy())
        for i, cs in enumerate(cts):
            for j, cr in enumerate(cts):
                if cs == cr:
                    continue
                s = Lg[i] * Rg[j]
                for k in np.argsort(s)[-10:][::-1]:
                    recs.append({"group": g, "sender": cs, "receiver": cr,
                                 "ligand": lr.source_genesymbol.iloc[k],
                                 "receptor": lr.target_genesymbol.iloc[k],
                                 "score": float(s[k])})
    comm = pd.DataFrame(recs)
    comm.to_csv(os.path.join(OUT, "communication_by_group.csv"), index=False)

    focus = comm[comm.sender.isin(["SMC", "ModifiedSMC"])]
    wide = (focus.pivot_table(index=["sender", "receiver", "ligand", "receptor"],
                              columns="group", values="score").reset_index())
    for g in GROUPS:
        if g not in wide.columns:
            wide[g] = np.nan
    wide["d_tcdd_vs_ctrl"] = wide["TCDD_16wk_WT"] - wide["Control"]
    wide["d_ako_vs_wt"] = wide["TCDD_16wk_AhrKO"] - wide["TCDD_16wk_WT"]
    wide["reversal"] = -wide["d_ako_vs_wt"] / wide["d_tcdd_vs_ctrl"].abs().replace(0, np.nan)
    wide.to_csv(os.path.join(OUT, "smc_outgoing_changes.csv"), index=False)

    smc_mac = wide[wide.receiver == "Macrophage"].sort_values("d_tcdd_vs_ctrl", ascending=False)
    smc_mac.to_csv(os.path.join(OUT, "smc_to_macrophage.csv"), index=False)
    print(smc_mac.head(12)[["sender", "ligand", "receptor", "Control", "TCDD_16wk_WT",
                            "TCDD_16wk_AhrKO", "d_tcdd_vs_ctrl", "d_ako_vs_wt"]]
          .round(4).to_string(), flush=True)

    def idx_of(gs):
        return [gi[g] for g in gs if g in gi]

    mod = pd.DataFrame({name: np.asarray(X[:, idx_of(gs)].mean(1)).ravel()
                        for name, gs in (("SASP", SASP), ("INFLAM", INFLAM), ("ISG", ISG))},
                       index=a.obs_names)
    mod["celltype"] = ct
    mod["group"] = grp
    per = mod.groupby(["group", "celltype"], observed=True).mean(numeric_only=True).reset_index()
    per.to_csv(os.path.join(OUT, "module_scores_by_group.csv"), index=False)
    print(per.pivot_table(index="celltype", columns="group", values="SASP").round(3).to_string(),
          flush=True)

    chem = smc_mac[smc_mac.ligand.isin(CHEMO)]
    chem.to_csv(os.path.join(OUT, "smc_to_macrophage_chemokines.csv"), index=False)

    de = []
    for c in ["Macrophage", "Endothelial", "SMC", "ModifiedSMC", "Fibroblast"]:
        m1 = (ct == c) & (grp == "TCDD_16wk_WT")
        m2 = (ct == c) & (grp == "TCDD_16wk_AhrKO")
        if m1.sum() < 30 or m2.sum() < 30:
            continue
        for g in ["Il1b", "Ccl2", "Cxcl10", "Isg15", "Ifit1", "Trem2", "Apoe", "Lgals3",
                  "Spp1", "Nlrp3", "Stat1", "Irf7", "Csf1r", "Ccr2", "Tgfb1", "Il6",
                  "Mrc1", "Cd163", "Ly6c2"]:
            if g not in gi:
                continue
            v1 = np.asarray(X[m1][:, gi[g]].todense()).ravel()
            v2 = np.asarray(X[m2][:, gi[g]].todense()).ravel()
            de.append({"celltype": c, "gene": g, "mean_wt": v1.mean(), "mean_ako": v2.mean(),
                       "d_ako_vs_wt": L.cohens_d(v2, v1)})
    pd.DataFrame(de).to_csv(os.path.join(OUT, "ako_vs_wt_marker_genes.csv"), index=False)

    summary = {
        "design": "group-level pseudobulk; tdT+ and tdT- are two halves of the same heart, "
                  "so SMC (tdT+) and macrophages (tdT-) never share a sample",
        "n_lr_pairs": int(len(lr)),
        "n_groups": len(GROUPS),
        "cells": int(a.shape[0]),
        "top_smc_to_macrophage": smc_mac.head(8)[
            ["sender", "ligand", "receptor", "d_tcdd_vs_ctrl", "d_ako_vs_wt"]].round(4).to_dict("records"),
        "n_pairs_tcdd_up_ako_down": int(((smc_mac.d_tcdd_vs_ctrl > 0) &
                                         (smc_mac.d_ako_vs_wt < 0)).sum()),
        "n_pairs_total": int(len(smc_mac)),
        "note": "no p-values: one pooled observation per group",
    }
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2)[:1200])


if __name__ == "__main__":
    main()
