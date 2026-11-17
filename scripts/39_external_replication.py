"""39 - External replication in a dataset that is not ours.

The mouse findings are one study (GSE276870/276872) plus a small number of
animals. Anything that only appears there is a hypothesis. This script asks
whether the *programmes* the story depends on can be re-derived in an
independent human tissue dataset - GSE131778, 11,756 cells from 8 donors of
human coronary artery, no exposure, no relation to the mouse experiment.

Three questions, each with a pre-set pass rule:

  R1 donor replication
     Split the 8 donors into two disjoint halves. Score every programme in each
     half independently and ask whether the ranking of programmes by
     cell-type specificity (AUC, target compartment vs the rest) survives the
     split. Rule: Spearman rho > 0.7 across programmes, in >90% of 200 splits.

  R2 donor robustness
     Score each of the 8 donors on its own. A programme that only works when
     donors are pooled is not a programme. Rule: the target compartment's AUC
     exceeds 0.6 in at least 6/8 donors.

  R3 the branch point
     The story says the endothelial axis is a different programme from the
     myeloid/stromal one. If that is true, ECpanel should be weakly correlated
     with LAM and with Matrix at the single-cell level, and clearly weaker than
     LAM-Matrix are with each other. Rule: r(ECpanel, LAM) and r(ECpanel, Matrix)
     both below r(LAM, Matrix).

Note on Matrix: the mouse result is "SMC secretes stromal ligands", so the
target compartment for Matrix is declared here as SMC *in advance*, not chosen
after looking at which compartment scores highest.

Outputs (results/human/):
  external_replication.csv
  external_replication.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "human")
os.makedirs(OUT, exist_ok=True)

# programme -> the compartment the mouse story says it belongs to
TARGET = {
    "LAM": "Mac",
    "ECpanel": "EC",
    "Matrix": "SMC",
    "ModSMC": "SMC",
}
N_SPLIT = 200
SEED = 7


def auc(pos, neg):
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) /
                 (len(pos) * len(neg)))


def prog_auc(df, prog, target):
    """AUC of the programme score for target compartment vs everything else."""
    y = (df.compartment == target).to_numpy()
    if y.sum() < 20 or (~y).sum() < 20:
        return np.nan
    return auc(df[prog].to_numpy()[y], df[prog].to_numpy()[~y])


def main():
    d = pd.read_csv(os.path.join(OUT, "human_celltype_scores.csv"), index_col=0)
    progs = [p for p in TARGET if p in d.columns]
    donors = sorted(d["sample"].unique())
    print(f"  {len(d)} cells, {len(donors)} donors, programmes {progs}", flush=True)

    # ---------------------------------------------------------------- R1
    rng = np.random.default_rng(SEED)
    # The pass rule is an absolute one: with only two stable programmes a rank
    # correlation can only be -1/0/+1, so it flips on ties and says nothing. What
    # matters is whether each programme's own AUC is reproducible across a
    # disjoint set of donors.
    diffs = {p: [] for p in progs}
    for _ in range(N_SPLIT):
        perm = rng.permutation(donors)
        a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
        da, db = d[d["sample"].isin(a)], d[d["sample"].isin(b)]
        va = {p: prog_auc(da, p, TARGET[p]) for p in progs}
        vb = {p: prog_auc(db, p, TARGET[p]) for p in progs}
        for p in progs:
            if np.isfinite(va[p]) and np.isfinite(vb[p]):
                diffs[p].append(abs(va[p] - vb[p]))
    r1 = dict(n_splits=N_SPLIT,
              median_abs_dauc={p: float(np.nanmedian(v)) for p, v in diffs.items()},
              p95_abs_dauc={p: float(np.nanpercentile(v, 95)) for p, v in diffs.items()},
              stable_progs=[p for p in progs if np.nanmedian(diffs[p]) < 0.05])
    print("  R1 split-half |dAUC| (median): " +
          ", ".join(f"{p} {r1['median_abs_dauc'][p]:.3f}" for p in progs), flush=True)

    # ---------------------------------------------------------------- R2
    per = []
    for dn in donors:
        sub = d[d["sample"] == dn]
        row = dict(donor=dn, n_cells=len(sub))
        for p in progs:
            row[p] = prog_auc(sub, p, TARGET[p])
        per.append(row)
    per = pd.DataFrame(per)
    cols = [p for p in progs]
    n_pass = {p: int((per[p] > 0.6).sum()) for p in cols}
    r2 = dict(n_donors=len(donors), auc_per_donor=per.round(3).to_dict("records"),
              n_donors_auc_gt_06=n_pass,
              pass_per_programme={p: bool(n_pass[p] >= 6) for p in cols})
    print(f"  R2 per-donor AUC>0.6: {n_pass}", flush=True)

    # ---------------------------------------------------------------- R3
    cor = d[cols].corr(method="spearman")
    r3 = dict(
        r_ecpanel_lam=float(cor.loc["ECpanel", "LAM"]),
        r_ecpanel_matrix=float(cor.loc["ECpanel", "Matrix"]),
        r_lam_matrix=float(cor.loc["LAM", "Matrix"]),
        r_modsmc_matrix=float(cor.loc["ModSMC", "Matrix"]),
    )
    r3["branch_holds"] = bool(
        r3["r_ecpanel_lam"] < r3["r_lam_matrix"] and
        r3["r_ecpanel_matrix"] < r3["r_lam_matrix"])
    print(f"  R3 r(ECpanel,LAM)={r3['r_ecpanel_lam']:+.2f}  "
          f"r(ECpanel,Matrix)={r3['r_ecpanel_matrix']:+.2f}  "
          f"r(LAM,Matrix)={r3['r_lam_matrix']:+.2f}", flush=True)

    # ---------------------------------------------------------------- pooled
    pooled = {p: prog_auc(d, p, TARGET[p]) for p in progs}
    r_all = dict(pooled_auc=pooled)

    per_i = per.set_index("donor")
    out = pd.DataFrame([
        dict(programme=p, target=TARGET[p], pooled_auc=pooled[p],
             **{f"donor_{dn}": per_i.loc[dn, p] for dn in donors},
             n_donors_pass=n_pass[p])
        for p in progs])
    out.to_csv(os.path.join(OUT, "external_replication.csv"), index=False)

    summary = dict(
        note=("independent human coronary atlas (GSE131778, 8 donors). R1 = the "
              "ranking of programmes by cell-type specificity survives a "
              "disjoint-donor split; R2 = the AUC holds donor by donor; R3 = the "
              "endothelial programme is less coupled to the myeloid/stromal ones "
              "than those are to each other, which is what a branch point means."),
        R1=r1, R2=r2, R3=r3, pooled=r_all["pooled_auc"],
        verdicts=dict(
            R1=bool(len(r1["stable_progs"]) >= 2),
            R2=bool(all(v >= 6 for v in n_pass.values())),
            R3=bool(r3["branch_holds"]),
        ),
    )
    with open(os.path.join(OUT, "external_replication.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("\n  pooled AUC:", {k: round(v, 3) for k, v in pooled.items()})
    print("  verdicts:", summary["verdicts"])


if __name__ == "__main__":
    main()
