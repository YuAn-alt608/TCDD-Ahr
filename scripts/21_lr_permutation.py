"""21 - Does the ligand-receptor transfer model need the real network topology?

14_insilico_validation.py fits one parameter (kappa) on the Control -> TCDD arm and
predicts the held-out Ahr-KO arm for four endothelial read-outs; 4/4 directions
came out right. With four read-outs, 4/4 happens by chance 1/16 of the time, so
the claim "the model captured the real Timp1-Cd63 / Fn1-Sdc4 flow" needs a null.

Two degree-preserving rewires of the bipartite network, 1,000 draws each:

  ligand-shuffle : permute which ligand sits on each edge -> every receptor keeps
                   its in-degree exactly, ligand out-degrees are re-drawn.
  receptor-shuffle: permute which receptor sits on each edge -> every ligand keeps
                   its out-degree exactly, receptor in-degrees are re-drawn.

Each null network is pushed through the identical pipeline (Hill combination ->
CM score -> kappa fitted on Control->TCDD -> prediction of the Ahr-KO arm), so the
only thing that changes is the wiring.

Outputs (results/validation/):
  lr_permutation_null.csv   hits and error per draw x null type
  lr_permutation_test.csv   observed vs null, empirical p
  summary.json
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

OUT = os.path.join(L.RES, "validation")
os.makedirs(OUT, exist_ok=True)
GROUPS = ["Control", "TCDD_8wk", "TCDD_16wk_WT", "TCDD_16wk_AhrKO"]
EC_READOUT = {"Klf2": "Klf2", "Smad2": "Smad2", "Nr1h3": "Nr1h3", "Hif1a": "Hif1a"}
N_PERM = 1000
# identical list to 14_insilico_validation.py (secreted / ECM / growth factor)
SECRETED = """Fn1 Col1a1 Col1a2 Col3a1 Col4a1 Col4a2 Col5a1 Col6a1 Col6a2 Col8a1 Col18a1
Timp1 Timp2 Timp3 Lum Dcn Bgn Mfap2 Mfap5 Spp1 Postn Fn1 Tnc Thbs1 Thbs2 Ccn1 Ccn2 Ccn3
Tgfb1 Tgfb2 Tgfb3 Csf1 Csf2 Csf3 Pdgfa Pdgfb Pdgfc Vegfa Vegfb Vegfc Bmp2 Bmp4 Bmp6
Il6 Il11 Il1a Il1b Il33 Ccl2 Ccl7 Ccl8 Cxcl1 Cxcl2 Cxcl12 Cxcl16 Cx3cl1 Lif Osm
Igf1 Igf2 Igfbp3 Igfbp5 Igfbp7 Angpt1 Angpt2 Adm Gdf15 Bdnf Ntf3 Sema3c Sema7a""".split()


def hill(l, r, k=0.5):
    l = np.maximum(l, 0.0)
    r = np.maximum(r, 0.0)
    return (l * r) / (k + l * r)


def main():
    pb = pd.read_csv(os.path.join(L.RES, "ccc", "pseudobulk_group_celltype.csv.gz"), index_col=0)
    pb["group"] = pb.index
    lr = pd.read_csv(os.path.join(L.RES, "ccc", "lr_pairs_used.csv"))

    def vec(group, celltype):
        d = pb[(pb.group == group) & (pb.celltype == celltype)]
        return d.iloc[0].drop(labels=["group", "celltype"]).astype(float)

    # exactly the pair set used in 14: secreted ligand, receptor detected in EC
    ec_max = pd.concat([vec(g, "Endothelial") for g in GROUPS], axis=1).max(axis=1)
    pairs = lr.drop_duplicates(subset=["source_genesymbol", "target_genesymbol"]).copy()
    pairs = pairs[pairs.source_genesymbol.isin(SECRETED)]
    pairs = pairs[pairs.target_genesymbol.isin(ec_max[ec_max > 0.05].index)]
    pairs = pairs[pairs.source_genesymbol.isin(pb.columns) &
                  pairs.target_genesymbol.isin(pb.columns)].reset_index(drop=True)
    print(f"pairs used: {len(pairs)}", flush=True)

    smc_senders = ["SMC", "ModifiedSMC"]
    ec = {g: vec(g, "Endothelial") for g in GROUPS}
    smc = {g: np.mean([vec(g, s) for s in smc_senders], axis=0) for g in GROUPS}
    smc = {g: pd.Series(v, index=vec(g, "SMC").index) for g, v in smc.items()}

    def cm_score(lig, rec, group):
        l = smc[group].reindex(lig).fillna(0).to_numpy()
        r = ec[group].reindex(rec).fillna(0).to_numpy()
        return float(hill(l, r).sum())

    def evaluate(lig, rec):
        x = np.array([cm_score(lig, rec, g) for g in GROUPS])
        y = {k: np.array([ec[g].get(v, 0.0) for g in GROUPS]) for k, v in EC_READOUT.items()}
        hits, errs, kaps = 0, [], []
        for k, yy in y.items():
            if abs(x[2] - x[1]) < 1e-12:
                continue
            # kappa fitted on the Control -> 16wk-WT arm (indices 0 -> 2),
            # prediction made on the held-out Ahr-KO arm (3)
            kappa = (yy[2] - yy[0]) / (x[2] - x[0])
            pred = yy[2] + kappa * (x[3] - x[2])
            hits += int(np.sign(pred - yy[2]) == np.sign(yy[3] - yy[2]))
            errs.append(abs(pred - yy[3]))
            kaps.append(kappa)
        return hits, float(np.mean(errs)) if errs else np.nan, x, kaps

    lig0 = pairs.source_genesymbol.to_numpy()
    rec0 = pairs.target_genesymbol.to_numpy()
    hits0, err0, x0, k0 = evaluate(lig0, rec0)
    print(f"observed: {hits0}/4 hits, mean |err| {err0:.4f}, CM scores {np.round(x0, 2)}", flush=True)

    rng = np.random.default_rng(0)
    rows = []
    for kind in ["ligand_shuffle", "receptor_shuffle"]:
        for i in range(N_PERM):
            if kind == "ligand_shuffle":
                lig, rec = rng.permutation(lig0), rec0
            else:
                lig, rec = lig0, rng.permutation(rec0)
            h, e, _, _ = evaluate(lig, rec)
            rows.append(dict(null=kind, perm=i, hits=h, mean_abs_err=e))
        print(f"  {kind} done", flush=True)
    ndf = pd.DataFrame(rows)
    ndf.to_csv(os.path.join(OUT, "lr_permutation_null.csv"), index=False)

    trows = [dict(null="observed", hits=hits0, mean_abs_err=err0, p_hits=np.nan,
                  p_err=np.nan)]
    for kind, sub in ndf.groupby("null"):
        p_hits = (1 + (sub.hits >= hits0).sum()) / (1 + len(sub))
        p_err = (1 + (sub.mean_abs_err <= err0).sum()) / (1 + len(sub))
        trows.append(dict(null=kind, hits=float(sub.hits.mean()),
                          mean_abs_err=float(sub.mean_abs_err.mean()),
                          p_hits=float(p_hits), p_err=float(p_err),
                          frac_4of4=float((sub.hits == 4).mean()),
                          null_hits_dist={int(k): int(v) for k, v in
                                          sub.hits.value_counts().items()}))
    tdf = pd.DataFrame(trows)
    tdf.to_csv(os.path.join(OUT, "lr_permutation_test.csv"), index=False)
    print("\n" + tdf.to_string(index=False), flush=True)

    sm = {"n_pairs": int(len(pairs)), "n_perm": N_PERM, "observed_hits": int(hits0),
          "observed_mean_abs_err": err0, "cm_scores": dict(zip(GROUPS, np.round(x0, 3).tolist())),
          "chance_4of4": 1 / 16,
          "test": tdf.drop(columns=["null_hits_dist"]).round(4).to_dict(orient="records")}
    json.dump(sm, open(os.path.join(OUT, "lr_permutation_summary.json"), "w"), indent=2,
              default=float)
    print(json.dumps(sm, indent=2, default=float))


if __name__ == "__main__":
    main()
