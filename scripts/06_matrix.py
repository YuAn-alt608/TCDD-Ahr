"""Build the exposure x lineage interaction matrix with effect sizes, timing (T50) and meta-comparison."""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

METRICS = {
    "HGDI": "LI_ConstitutiveHet",
    "LADleak": "LI_LADproxy",
    "TEderep": "TES",
    "Entropy": "entropy_res",
    "RegulonEntropy": "regulon_entropy",
}
EXPOSURE = {
    "D1": ("PM2.5 (lung, 1 mo)", "1mo"),
    "D2": ("TCDD (coronary, 8/16 wk)", None),
    "D3": ("PM2.5 (maternal -> offspring iWAT)", "middle-age"),
}
# treat these groups as reference (control) per dataset
REF = {
    "D1": "Control",
    "D2": "Control",
    "D3": "Control",
}


def load(ds):
    p = os.path.join(L.RES, ds, "cell_metrics.csv.gz")
    df = pd.read_csv(p)
    return df


def effect_table(ds, df):
    rows = []
    ctrl = df[df.group == REF[ds]]
    for metric, col in METRICS.items():
        for (grp, ct), sub in df.groupby(["group", "celltype"], observed=True):
            if grp == REF[ds]:
                continue
            ref = ctrl[ctrl.celltype == ct]
            if len(ref) < 20 or len(sub) < 20:
                continue
            a = sub[col].to_numpy(float)
            b = ref[col].to_numpy(float)
            rows.append({
                "dataset": ds, "exposure": EXPOSURE[ds][0],
                "duration": sub["duration"].iloc[0] if "duration" in sub else EXPOSURE[ds][1],
                "group": grp, "celltype": ct, "lineage": sub["lineage"].iloc[0],
                "metric": metric, "n_case": len(a), "n_ref": len(b),
                "mean_case": float(np.nanmean(a)), "mean_ref": float(np.nanmean(b)),
                "delta": float(np.nanmean(a) - np.nanmean(b)),
                "cohens_d": L.cohens_d(a, b),
                "p_perm": L.perm_p(a, b, n=1000),
            })
    t = pd.DataFrame(rows)
    t["q_bh"] = np.nan
    for m in t.metric.unique():
        k = t.metric == m
        t.loc[k, "q_bh"] = L.bh(t.loc[k, "p_perm"].to_numpy())
    return t


def t50_table(t2):
    """Half-max duration (weeks) per lineage from the TCDD 0/8/16-week series."""
    out = []
    sub = t2[t2.group.isin(["TCDD_8wk", "TCDD_16wk_WT"])]
    for (ct, metric), s in sub.groupby(["celltype", "metric"], observed=True):
        d8 = s[s.group == "TCDD_8wk"].delta
        d16 = s[s.group == "TCDD_16wk_WT"].delta
        if len(d8) == 0 or len(d16) == 0:
            continue
        d8 = float(d8.iloc[0]); d16 = float(d16.iloc[0])
        half = d16 * 0.5
        # a half-max time is only meaningful when the effect is non-negligible, grows in the same
        # direction at 8 and 16 weeks, and is actually crossed inside the observed window
        if not np.isfinite(d16) or abs(d16) < 0.02 or d8 * d16 <= 0:
            t50 = np.nan
        elif abs(d8) >= abs(half):
            t50 = 8 * (half / d8) if d8 != 0 else np.nan
        else:
            frac = (half - d8) / (d16 - d8)
            t50 = 8 + 8 * frac
        if not np.isfinite(t50) or t50 <= 0 or t50 > 16:
            t50 = np.nan
        out.append({"celltype": ct, "lineage": s.lineage.iloc[0], "metric": metric,
                    "delta_8wk": d8, "delta_16wk": d16, "T50_weeks": t50,
                    "ratio_8_over_16": d8 / d16 if d16 else np.nan})
    return pd.DataFrame(out)


def main():
    out = os.path.join(L.RES, "matrix")
    os.makedirs(out, exist_ok=True)
    allt = []
    for ds in ["D1", "D2", "D3"]:
        p = os.path.join(L.RES, ds, "cell_metrics.csv.gz")
        if not os.path.exists(p):
            print("skip", ds); continue
        df = load(ds)
        # D2 carries a duration column; make group/duration explicit
        if ds == "D2":
            df["duration"] = df["duration"].astype(str) if "duration" in df else ""
        t = effect_table(ds, df)
        allt.append(t)
        print(f"{ds}: {len(t)} effects", flush=True)
    T = pd.concat(allt, ignore_index=True)
    T.to_csv(os.path.join(out, "exposure_lineage_effects.csv"), index=False)

    # wide matrix: exposure x celltype for each metric (Cohen's d)
    for metric in METRICS:
        w = T[T.metric == metric].pivot_table(index=["dataset", "group"],
                                              columns="celltype", values="cohens_d")
        w.to_csv(os.path.join(out, f"matrix_{metric}_cohensD.csv"))

    # niche vs parenchyma meta-comparison
    rows = []
    for (ds, grp, metric), s in T.groupby(["dataset", "group", "metric"], observed=True):
        niche = s[s.lineage.isin(["Endothelial", "Mesenchymal"])].cohens_d.dropna()
        par = s[s.lineage == "Parenchyma"].cohens_d.dropna()
        imm = s[s.lineage == "Immune"].cohens_d.dropna()
        if len(niche) and len(par):
            rows.append({"dataset": ds, "group": grp, "metric": metric,
                         "mean_d_niche": float(niche.mean()), "n_niche": len(niche),
                         "mean_d_parenchyma": float(par.mean()), "n_par": len(par),
                         "niche_minus_parenchyma": float(niche.mean() - par.mean()),
                         "mean_d_immune": float(imm.mean()) if len(imm) else np.nan})
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(out, "niche_vs_parenchyma.csv"), index=False)

    if (T.dataset == "D2").any():
        t50 = t50_table(T[T.dataset == "D2"])
        t50.to_csv(os.path.join(out, "T50_by_lineage.csv"), index=False)
        print(t50.sort_values(["metric", "T50_weeks"]).to_string(index=False), flush=True)

    print("\n=== top effects (|d|) ===", flush=True)
    print(T.reindex(T.cohens_d.abs().sort_values(ascending=False).index).head(15)
          [["dataset", "group", "celltype", "metric", "cohens_d", "q_bh"]].to_string(index=False))
    print("\n=== niche vs parenchyma ===", flush=True)
    print(M.sort_values("niche_minus_parenchyma", key=abs, ascending=False).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
