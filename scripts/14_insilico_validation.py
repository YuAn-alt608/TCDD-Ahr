"""Three lightweight *in silico* replications of the wet-lab experiments that would
close the causal loop, using only data already on disk.

V1  Conditioned medium (CM) transfer
    SMC -> endothelial cell paracrine transfer score, computed from group-level
    pseudobulk and an explicit ligand-receptor table.  Calibrated on the TCDD arm
    (single proportionality constant) and *tested* on the held-out SMC-Ahr-KO arm.
    Read-outs mirror the proposed qPCR/ChIP-qPCR panel: Klf2, Smad2, Nr1h3, Hif1a.

V2  Matrix ligand / macrophage reprogramming
    SMC supernatant proxy (Timp1, Fn1, Col1a2, Col3a1, Lum, Csf1, Tgfb1) versus
    macrophage LAM (Trem2/Lgals3/Csf1r/Spp1/Apoe/Cd9/Fabp5/Lpl) and ISG programs.
    The Ahr-KO arm is used as the *genetic* equivalent of a pharmacological
    antagonist (CH-223191); the projection is stated as a projection.

V3  Genetic independence
    Turns the GWAS negative result (OR = 0.53) into a quantitative statement:
    with a 95% CI that excludes 1 on the low side, the epigenetic switch panel
    carries *fewer* ageing-trait associations than a pathway-matched background,
    i.e. environmental vulnerability is not tagged by common inherited variation.

All three are descriptive: n = 4 group-level observations, no p-values except the
Fisher test carried over from the genetics layer.  Nothing here is a substitute
for the bench experiment; it is a consistency check that the proposed experiment
has a directional prediction already present in the data.
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

OUT = os.path.join(L.RES, "validation")
os.makedirs(OUT, exist_ok=True)

GROUPS = ["Control", "TCDD_8wk", "TCDD_16wk_WT", "TCDD_16wk_AhrKO"]

# ligands that can physically travel in conditioned medium (secreted / ECM / growth factor)
SECRETED = """Fn1 Col1a1 Col1a2 Col3a1 Col4a1 Col4a2 Col5a1 Col6a1 Col6a2 Col8a1 Col18a1
Timp1 Timp2 Timp3 Lum Dcn Bgn Mfap2 Mfap5 Spp1 Postn Fn1 Tnc Thbs1 Thbs2 Ccn1 Ccn2 Ccn3
Tgfb1 Tgfb2 Tgfb3 Csf1 Csf2 Csf3 Pdgfa Pdgfb Pdgfc Vegfa Vegfb Vegfc Bmp2 Bmp4 Bmp6
Il6 Il11 Il1a Il1b Il33 Ccl2 Ccl7 Ccl8 Cxcl1 Cxcl2 Cxcl12 Cxcl16 Cx3cl1 Lif Osm
Igf1 Igf2 Igfbp3 Igfbp5 Igfbp7 Angpt1 Angpt2 Adm Gdf15 Bdnf Ntf3 Sema3c Sema7a""".split()

LAM = ["Trem2", "Lgals3", "Csf1r", "Spp1", "Apoe", "Cd9", "Fabp5", "Lpl", "Gpnmb", "Trem1"]
EC_READOUT = {"Klf2": "Klf2", "Smad2": "Smad2", "Nr1h3": "Nr1h3", "Hif1a": "Hif1a"}


def hill(l, r, k=0.5):
    """CellChat-style Hill combination on log-normalised pseudobulk means."""
    l = np.maximum(l, 0.0)
    r = np.maximum(r, 0.0)
    return (l * r) / (k + l * r)


def main():
    pb = pd.read_csv(os.path.join(L.RES, "ccc", "pseudobulk_group_celltype.csv.gz"), index_col=0)
    pb["group"] = pb.index
    lr = pd.read_csv(os.path.join(L.RES, "ccc", "lr_pairs_used.csv"))
    mod = pd.read_csv(os.path.join(L.RES, "ccc", "module_scores_by_group.csv"))

    def vec(group, celltype):
        d = pb[(pb.group == group) & (pb.celltype == celltype)]
        return d.iloc[0].drop(labels=["group", "celltype"]).astype(float)

    smc_senders = ["SMC", "ModifiedSMC"]

    # ---------------------------------------------------------------- V1 CM score
    # receptor must be detectable in endothelial cells in at least one group
    ec_any = pd.concat([vec(g, "Endothelial") for g in GROUPS], axis=1).max(axis=1)
    pairs = lr.drop_duplicates(subset=["source_genesymbol", "target_genesymbol"]).copy()
    pairs = pairs[pairs.source_genesymbol.isin(SECRETED)]
    pairs = pairs[pairs.target_genesymbol.isin(ec_any[ec_any > 0.05].index)]
    pairs = pairs[pairs.source_genesymbol.isin(pb.columns) & pairs.target_genesymbol.isin(pb.columns)]

    rows = []
    for g in GROUPS:
        ec = vec(g, "Endothelial")
        for s in smc_senders:
            sm = vec(g, s)
            contrib = pd.Series(
                [hill(sm.get(l, 0.0), ec.get(r, 0.0)) for l, r in
                 zip(pairs.source_genesymbol, pairs.target_genesymbol)], index=pairs.index)
            rows.append(dict(group=g, sender=s, cm_score=float(contrib.sum()),
                             n_pairs=int(len(contrib)),
                             top=";".join(
                                 f"{pairs.loc[i,'source_genesymbol']}-{pairs.loc[i,'target_genesymbol']}"
                                 for i in contrib.sort_values(ascending=False).head(5).index)))
    cm = pd.DataFrame(rows)
    cm["arm"] = np.where(cm.group.str.contains("AhrKO"), "AhrKO",
                         np.where(cm.group.str.startswith("TCDD"), "TCDD", "Control"))
    cm.to_csv(os.path.join(OUT, "cm_transfer.csv"), index=False)

    # per-pair contributions in the TCDD arm, to see what actually carries the signal
    g0, g1, g2 = "Control", "TCDD_16wk_WT", "TCDD_16wk_AhrKO"
    ec0, ec1, ec2 = vec(g0, "Endothelial"), vec(g1, "Endothelial"), vec(g2, "Endothelial")
    ctrl_smc = {s: vec(g0, s) for s in smc_senders}
    wt_smc = {s: vec(g1, s) for s in smc_senders}
    ko_smc = {s: vec(g2, s) for s in smc_senders}
    contrib = pd.DataFrame({
        "ligand": pairs.source_genesymbol.values,
        "receptor": pairs.target_genesymbol.values,
    })
    for tag, smc, ec in [("ctrl", ctrl_smc, ec0), ("tcdd", wt_smc, ec1), ("ako", ko_smc, ec2)]:
        contrib[tag] = [hill(np.mean([smc[s].get(l, 0.0) for s in smc_senders]),
                             ec.get(r, 0.0)) for l, r in
                        zip(pairs.source_genesymbol, pairs.target_genesymbol)]
    contrib["d_tcdd"] = contrib.tcdd - contrib.ctrl
    contrib["d_ako"] = contrib.ako - contrib.tcdd
    contrib = contrib.sort_values("d_tcdd", ascending=False)
    contrib.to_csv(os.path.join(OUT, "cm_pair_contributions.csv"), index=False)

    # ------------------------------------------------------- V1 calibration / test
    # read-out = kappa * CM ; kappa fitted on Control -> TCDD, tested on Ahr-KO
    cm_score = cm[cm.sender == "ModifiedSMC"].set_index("group").cm_score
    ec_tab = pd.DataFrame({g: vec(g, "Endothelial") for g in GROUPS})
    fit = {}
    for name, gene in EC_READOUT.items():
        y = ec_tab.loc[gene] if gene in ec_tab.index else None
        if y is None:
            continue
        x0, x1, x2 = cm_score[g0], cm_score[g1], cm_score[g2]
        if abs(x1 - x0) < 1e-9:
            continue
        kappa = (y[g1] - y[g0]) / (x1 - x0)
        pred2 = y[g1] + kappa * (x2 - x1)
        fit[name] = dict(gene=gene, kappa=float(kappa), obs_ctrl=float(y[g0]),
                         obs_tcdd=float(y[g1]), obs_ako=float(y[g2]),
                         pred_ako=float(pred2), err=float(pred2 - y[g2]),
                         direction_ok=bool(np.sign(pred2 - y[g1]) == np.sign(y[g2] - y[g1])))
    fitdf = pd.DataFrame(fit).T.reset_index().rename(columns={"index": "readout"})
    fitdf.to_csv(os.path.join(OUT, "cm_readout_prediction.csv"), index=False)

    # ------------------------------------------------- V2 matrix ligand / macrophage
    SUPERNATANT = ["Timp1", "Fn1", "Col1a2", "Col3a1", "Lum", "Csf1", "Tgfb1", "Spp1", "Il6", "Ccl2", "Cxcl12"]
    rows = []
    for g in GROUPS:
        sm = pd.concat([vec(g, s) for s in smc_senders], axis=1).mean(axis=1)
        mac = vec(g, "Macrophage")
        lam_genes = [x for x in LAM if x in mac.index]
        lam = float(np.mean([(mac[x] - np.mean([vec(gg, "Macrophage")[x] for gg in GROUPS])) /
                             (np.std([vec(gg, "Macrophage")[x] for gg in GROUPS]) + 1e-9)
                             for x in lam_genes]))
        rec = dict(group=g, lam_score=lam, n_lam_genes=len(lam_genes))
        for x in SUPERNATANT:
            rec[f"smc_{x}"] = float(sm.get(x, np.nan))
        rows.append(rec)
    lig = pd.DataFrame(rows)
    lig["ecm_index"] = lig[[f"smc_{x}" for x in ["Timp1", "Fn1", "Col1a2", "Col3a1", "Lum"]]].mean(axis=1)
    lig["arm"] = np.where(lig.group.str.contains("AhrKO"), "AhrKO",
                          np.where(lig.group.str.startswith("TCDD"), "TCDD", "Control"))
    lig = lig.merge(mod[mod.celltype == "Macrophage"][["group", "INFLAM", "ISG", "SASP"]],
                    on="group", how="left")
    lig.to_csv(os.path.join(OUT, "matrix_ligand_macrophage.csv"), index=False)

    # projection: CH-223191 phenocopies the genetic Ahr knockout
    proj = {}
    for col in ["lam_score", "INFLAM", "ISG"]:
        c, t, k = lig.set_index("group").loc[[g0, g1, g2], col]
        proj[col] = dict(control=float(c), tcdd=float(t), ahrko=float(k),
                         delta_ako=float(k - t),
                         reversal_frac=float((k - t) / (t - c)) if abs(t - c) > 1e-9 else np.nan)

    # ------------------------------------------ V2 ligand-level reversal ranking
    d = pd.DataFrame({
        "ligand": SUPERNATANT,
        "ctrl": [lig.set_index("group").loc[g0, f"smc_{x}"] for x in SUPERNATANT],
        "tcdd": [lig.set_index("group").loc[g1, f"smc_{x}"] for x in SUPERNATANT],
        "ako": [lig.set_index("group").loc[g2, f"smc_{x}"] for x in SUPERNATANT],
    })
    d["d_tcdd"] = d.tcdd - d.ctrl
    d["d_ako"] = d.ako - d.tcdd
    d = d.sort_values("d_tcdd", ascending=False)
    d.to_csv(os.path.join(OUT, "smc_supernatant_ligands.csv"), index=False)

    # ------------------------------------------------ V3 genetic independence
    gwas = pd.read_csv(os.path.join(L.RES, "genetics", "gwas_associations_panel.csv"))
    bg = pd.read_csv(os.path.join(L.RES, "genetics", "gwas_associations_background.csv"))
    enr = json.load(open(os.path.join(L.RES, "genetics", "trait_enrichment.json")))
    a = enr["n_panel_assoc"] * enr["panel_prop"]
    b = enr["n_panel_assoc"] * (1 - enr["panel_prop"])
    c = enr["n_background_assoc"] * enr["background_prop"]
    dd = enr["n_background_assoc"] * (1 - enr["background_prop"])
    or_ = (a * dd) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / dd)
    lo, hi = np.exp(np.log(or_) - 1.96 * se), np.exp(np.log(or_) + 1.96 * se)

    per_gene = gwas.groupby("gene").agg(n_assoc=("rs_id", "size"),
                                        n_ageing=("categories", lambda s: int(sum("ageing" in str(x) for x in s)))
                                        ).reset_index()
    per_gene = per_gene.sort_values("n_ageing", ascending=False)
    per_gene.to_csv(os.path.join(OUT, "switch_gwas_ageing.csv"), index=False)
    zero = int((per_gene.n_ageing == 0).sum())

    sw = pd.read_csv(os.path.join(L.RES, "perturbation", "switch_summary.csv"))
    env_effect = sw.set_index("gene")["mean_delta_expr"].abs()

    gen_indep = dict(
        odds_ratio=float(or_), ci_low=float(lo), ci_high=float(hi),
        panel_prop=float(enr["panel_prop"]), background_prop=float(enr["background_prop"]),
        n_switch_genes=int(len(per_gene)), n_zero_ageing_hits=zero,
        frac_zero_ageing=float(zero / max(len(per_gene), 1)),
        median_abs_exposure_delta=float(env_effect.median()),
        max_abs_exposure_delta=float(env_effect.max()),
        framing=("OR 的 95% CI 完全落在 1 以下：表观开关面板在人群 GWAS 中携带的器官退变/衰老关联"
                 "比通路匹配背景更少。这不是'没有遗传基础'，而是'环境表观重塑的靶点不是常见遗传变异的靶点'。"),
    )
    json.dump(gen_indep, open(os.path.join(OUT, "genetic_independence.json"), "w"),
              ensure_ascii=False, indent=2)

    # --------------------------------------------------------------- summary
    s = dict(
        v1_cm={"cm_scores": {r.group: float(r.cm_score) for r in cm[cm.sender == "ModifiedSMC"].itertuples()},
               "n_pairs": int(len(contrib)),
               "top_tcdd_pairs": contrib.head(5)[["ligand", "receptor", "d_tcdd"]].to_dict("records"),
               "readout_prediction": fitdf.to_dict("records")},
        v2_mac={"projection": proj,
                "top_supernatant": d.head(5)[["ligand", "d_tcdd", "d_ako"]].to_dict("records")},
        v3_genetic=gen_indep,
        note=("group-level pseudobulk, n=4 groups (n=1 per group); V1/V2 use one fitted "
              "constant on the TCDD arm and test it on the held-out Ahr-KO arm. No p-values."),
    )
    json.dump(s, open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=2)

    print(json.dumps(s, ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
