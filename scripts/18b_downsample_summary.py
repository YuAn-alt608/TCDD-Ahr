"""18b - Turn the downsampling tables into the numbers quoted in the report.

Reads results/robust/downsample_{effects,modules,domains}.csv (written by
18_atac_downsample.py) and produces downsample_summary.json plus a small,
printable verdict table: for each claimed effect, the full-depth value, the
mean over the five seeds after thinning to the Ahr-KO depth, and the stress
condition at 60% of that depth.
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

OUT = os.path.join(L.RES, "robust")
TARGETS = ["match_ako", "stress_60pct"]
PANEL = ["Ahr::Arnt", "KLF2", "SMAD2", "ATF2", "FOSL1::JUND", "Nfe2l2", "RELA"]


def main():
    eff = pd.read_csv(os.path.join(OUT, "downsample_effects.csv"))
    dom = pd.read_csv(os.path.join(OUT, "downsample_domains.csv"))
    mods = pd.read_csv(os.path.join(OUT, "downsample_modules.csv"))

    def tf_summary(tf, lin, col):
        sub = eff[(eff.tf == tf) & (eff.lineage == lin)]
        if len(sub) == 0:
            return None
        base = float(sub[sub.target == "none"][col].iloc[0])
        out = {"full": round(base, 3)}
        for t in TARGETS:
            v = sub[sub.target == t][col]
            out[t] = [round(float(v.mean()), 3), round(float(v.min()), 3),
                      round(float(v.max()), 3)]
        return out

    verdict = {}
    for tf in PANEL:
        verdict[tf] = {
            "EC_d_ako_vs_wt": tf_summary(tf, "Endothelial", "d_ako_vs_wt"),
            "EC_p_ako_vs_wt": tf_summary(tf, "Endothelial", "p_ako_vs_wt"),
            "SMC_d_wt_vs_ctrl": tf_summary(tf, "Parenchyma", "d_wt_vs_ctrl"),
        }

    rows = []
    for met in ["frac_het", "frac_lad", "prom_het_rel"]:
        for lin in ["Parenchyma", "Endothelial", "Mesenchymal", "Immune"]:
            sub = dom[(dom.metric == met) & (dom.lineage == lin)]
            if len(sub) == 0:
                continue
            base = sub[sub.target == "none"].iloc[0]
            rec = dict(metric=met, lineage=lin,
                       full_d=round(float(base.d_wt_vs_ctrl), 3),
                       full_p=round(float(base.p_wt_vs_ctrl), 3))
            for t in TARGETS:
                v = sub[sub.target == t]
                rec[f"{t}_d"] = round(float(v.d_wt_vs_ctrl.mean()), 3)
                rec[f"{t}_p"] = round(float(v.p_wt_vs_ctrl.mean()), 3)
            rows.append(rec)
    doms = pd.DataFrame(rows)

    # a claim survives only if the sign is kept and the mean p stays below 0.05
    def verdict_of(r):
        kept = np.sign(r.match_ako_d) == np.sign(r.full_d) and r.match_ako_p < 0.05
        return "holds" if kept else ("direction only" if np.sign(r.match_ako_d) ==
                                     np.sign(r.full_d) else "lost")

    doms["verdict"] = doms.apply(verdict_of, axis=1)
    doms.to_csv(os.path.join(OUT, "downsample_domain_verdict.csv"), index=False)
    print(doms.to_string(index=False))

    sm = {"depth": {"AhrKO_median_counts": 3718, "WT_median_counts": 5114,
                    "CTRL_median_counts": [4682, 5378]},
          "design_note": "Ahr-KO ATAC library is 27% shallower than WT; thinning equalises depth",
          "n_seeds": 5, "n_perm": 500,
          "tf_verdict": verdict,
          "domain_verdict": doms.to_dict(orient="records")}
    json.dump(sm, open(os.path.join(OUT, "downsample_summary.json"), "w"), indent=2)
    print("\nwrote downsample_summary.json")


if __name__ == "__main__":
    main()
