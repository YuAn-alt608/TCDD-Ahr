"""The frozen decision procedure for the "one more animal" question.

`25_power_n2.py` writes the pre-registration; this file is the part that runs *after* the
new animals exist. It implements exactly two things and refuses to do anything else:

  H1  Ahr-KO has NOT returned to control  ->  KO vs Control, pooled t, Holm over 4 endpoints
  H2  how far it returned                  ->  KO - WT with a bootstrap CI, NO p value

Until the samples land, every endpoint reports "insufficient" and the script exits 0 --
that is the correct behaviour, not a failure.

    python scripts/25b_n2_analysis.py             # real data (currently: insufficient)
    python scripts/25b_n2_analysis.py --selftest   # synthesise a 2nd KO + 2nd WT, prove the path
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

OUT = os.path.join(L.RES, "power")
ALPHA = 0.05

ENDPOINTS = [
    ("ATAC", "Parenchyma", "frac_het", "SMC H3K9me3 域开放"),
    ("ATAC", "Endothelial", "prom_het_rel", "内皮异染色质启动子相对开放"),
    ("RNA", "Endothelial", "het_derepression", "内皮异染色质去抑制模块"),
    ("RNA", "Macrophage", "LAM", "髓系 LAM 模块"),
]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def load_tables():
    rna = pd.read_csv(os.path.join(L.RES, "pseudobulk", "rna_module_by_mouse.csv"))
    atac = pd.read_csv(os.path.join(L.RES, "pseudobulk", "atac_by_library.csv"))
    return rna, atac


def values_for(assay, celltype, endpoint, rna, atac):
    if assay == "RNA":
        s = rna[(rna.celltype == celltype) & (rna.module == endpoint)]
        col = "score"
    else:
        s = atac[atac.lineage == celltype]
        col = endpoint
    out = {}
    for grp, sub in s.groupby("group"):
        out[grp] = sub[col].to_numpy(float)
    return out


def boot_delta(a, b, rng, nboot=4000):
    """Percentile CI for mean(a) - mean(b), resampling animals only."""
    if len(a) < 2 or len(b) < 1:
        return (np.nan, np.nan)
    d = np.empty(nboot)
    for i in range(nboot):
        d[i] = rng.choice(a, len(a), True).mean() - rng.choice(b, len(b), True).mean()
    return tuple(np.percentile(d, [2.5, 97.5]))


def run(rna, atac, verbose=True):
    rng = np.random.default_rng(7)
    rows = []
    for assay, ct, ep, label in ENDPOINTS:
        v = values_for(assay, ct, ep, rna, atac)
        ctrl = v.get("Control", np.array([]))
        wt = v.get("TCDD_16wk_WT", np.array([]))
        ko = v.get("TCDD_16wk_AhrKO", np.array([]))

        h1_p, status = np.nan, ""
        if len(ko) >= 2 and len(ctrl) >= 2:
            h1_p = float(stats.ttest_ind(ko, ctrl, equal_var=True).pvalue)
        else:
            status = f"证据不足：KO n={len(ko)}, Control n={len(ctrl)}（H1 需 KO≥2）"

        delta = float(ko.mean() - wt.mean()) if len(ko) and len(wt) else np.nan
        lo, hi = boot_delta(ko, wt, rng)
        dir_ko_ctrl = float(ko.mean() - ctrl.mean()) if len(ko) and len(ctrl) else np.nan

        rows.append({"assay": assay, "celltype": ct, "endpoint": ep, "claim": label,
                     "n_ko": len(ko), "n_wt": len(wt), "n_ctrl": len(ctrl),
                     "delta_ko_vs_ctrl": dir_ko_ctrl, "p_ko_vs_ctrl_raw": h1_p,
                     "delta_ko_vs_wt": delta, "ci_lo": lo, "ci_hi": hi, "status": status})

    df = pd.DataFrame(rows)
    testable = df.p_ko_vs_ctrl_raw.notna()
    df["p_holm"] = np.nan
    if testable.any():
        df.loc[testable, "p_holm"] = holm(df.loc[testable, "p_ko_vs_ctrl_raw"].to_numpy())

    def verdict(r):
        if pd.isna(r.p_holm):
            return "证据不足（不改结论）"
        if r.p_holm < ALPHA:
            return "H1 成立：KO 未复原" if abs(r.delta_ko_vs_ctrl) > 0 else "H1 成立（方向需核对）"
        return "未能排除复原"

    df["verdict"] = df.apply(verdict, axis=1)
    df["h2_note"] = ["仅描述（n=%d vs %d），不给 p" % (r.n_ko, r.n_wt) for r in df.itertuples()]
    if verbose:
        show = df[["assay", "celltype", "endpoint", "n_ko", "n_ctrl",
                   "delta_ko_vs_ctrl", "p_holm", "delta_ko_vs_wt", "ci_lo", "ci_hi", "verdict"]]
        print(show.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return df


def selftest(rna, atac):
    """Fabricate a 2nd KO and a 2nd WT from the control animals + the observed shift.

    This is *not* a result: it only proves that the frozen code path renders a verdict
    table and a Holm correction once n >= 2. Nothing is written to results/.
    """
    rng = np.random.default_rng(11)
    rna2 = rna.copy()
    atac2 = atac.copy()
    add_r = []
    for (ct, mod), sub in rna.groupby(["celltype", "module"]):
        base = sub[sub.group == "Control"].score.to_numpy(float)
        if len(base) == 0:
            continue
        shift = 0.0
        ko = sub[sub.group == "TCDD_16wk_AhrKO"].score.to_numpy(float)
        c0 = sub[sub.group == "Control"].score.to_numpy(float)
        if len(ko) and len(c0):
            shift = float(ko.mean() - c0.mean())
        add_r.append({"mouse": "SYNKO", "celltype": ct, "group": "TCDD_16wk_AhrKO",
                      "module": mod, "score": rng.choice(base) + shift, "n_cells": 300})
        add_r.append({"mouse": "SYNWT", "celltype": ct, "group": "TCDD_16wk_WT",
                      "module": mod, "score": rng.choice(base), "n_cells": 300})
    rna2 = pd.concat([rna2, pd.DataFrame(add_r)], ignore_index=True)

    add_a = []
    for lin, sub in atac.groupby("lineage"):
        base = sub[sub.group == "Control"]
        if len(base) < 1:
            continue
        row = base.iloc[0].to_dict()
        for metric in ["frac_het", "frac_lad", "prom_het_rel"]:
            row[metric] = float(base[metric].mean())      # 合成值 = 对照均值
        add_a.append({**row, "mouse": "SYNKO", "group": "TCDD_16wk_AhrKO"})
        add_a.append({**row, "mouse": "SYNWT", "group": "TCDD_16wk_WT"})
    atac2 = pd.concat([atac2, pd.DataFrame(add_a)], ignore_index=True)
    print("\n--- SELFTEST：用合成动物跑冻结流程（不是结果，不写盘）---", flush=True)
    return run(rna2, atac2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    rna, atac = load_tables()
    if args.selftest:
        selftest(rna, atac)
        return
    df = run(rna, atac)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "n2_verdict.csv"), index=False)
    json.dump({"n_testable": int(df.p_holm.notna().sum()), "n_total": int(len(df)),
               "endpoints": ENDPOINTS,
               "note": "H1 = KO vs Control (2 vs 3, pooled t, Holm); H2 = KO-WT bootstrap CI, no p."},
              open(os.path.join(OUT, "n2_verdict.json"), "w"), indent=1, ensure_ascii=False)
    print("written", os.path.join(OUT, "n2_verdict.csv"))


if __name__ == "__main__":
    main()
