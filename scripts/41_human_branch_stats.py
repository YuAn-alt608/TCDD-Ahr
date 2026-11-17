"""41 - Does the branch point survive donor stratification and technical covariates?

R3 in script 39 was a single pooled Spearman over 11,756 cells. That is the
weakest possible form of the claim and the easiest to attack: eight donors
contribute unequal numbers of cells and unequal depth, so a pooled correlation
can be produced by two donors with many cells rather than by the biology
(Simpson's paradox). This script re-asks the same three correlations four ways,
each answering a different objection:

  1. within donor            - 8 independent estimates, paired across pairs
                               (Wilcoxon signed-rank on the difference)
  2. donor fixed effects     - pooled estimate with every donor's mean removed,
                               which is what "within-donor" means at the cell
                               level; equivalent to a random-intercept LMM for
                               the slope when slopes are homogeneous
  3. partial correlation     - residuals after regressing each score on
                               log nCount, pct_ribo, pct_mito and donor
                               dummies, so depth/capture quality cannot create
                               the anti-correlation
  4. within cell type        - the same correlations computed inside a single
                               compartment, where composition cannot act

Inference unit is the donor, not the cell: the confidence interval on the
difference between correlations comes from a cluster bootstrap that resamples
donors, because cells within a donor are not independent.

Outputs (results/human/):
  r3_within_donor.csv   per-donor correlation for each pair
  r3_stratified.csv     per-compartment, donor-centred correlation
  r3_summary.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "human")

PAIRS = [("LAM", "Matrix"), ("ECpanel", "Matrix"), ("ECpanel", "LAM")]
NBOOT = 2000
SEED = 11


def corr(x, y, method="spearman"):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return np.nan
    if method == "spearman":
        return float(stats.spearmanr(x[m], y[m]).statistic)
    return float(stats.pearsonr(x[m], y[m]).statistic)


def demean(df, cols, by="sample"):
    """Remove each donor's mean (donor fixed effects, cell by cell)."""
    out = df.copy()
    for c in cols:
        out[c] = df[c] - df.groupby(by)[c].transform("mean")
    return out


def residualise(df, cols, covars):
    """OLS residuals of each score on covariates + donor dummies."""
    D = pd.get_dummies(df["sample"], drop_first=True).to_numpy(float)
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in covars] + [D])
    R = {}
    for c in cols:
        y = df[c].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        R[c] = y - X @ beta
    return pd.DataFrame(R, index=df.index)


def cluster_bootstrap(df, cols, nboot=NBOOT, seed=SEED):
    """Resample DONORS with replacement, keep donors centred, recompute r."""
    rng = np.random.default_rng(SEED)
    donors = df["sample"].unique()
    d = demean(df, cols)
    idx = {dn: g.index.to_numpy() for dn, g in d.groupby("sample")}
    vals = {p: [] for p in PAIRS}
    for _ in range(nboot):
        pick = rng.choice(donors, len(donors), replace=True)
        parts = [idx[dn] for dn in pick]
        # values were centred inside each donor before resampling, so a donor
        # drawn twice keeps its own centring and is not split into two means
        sub = d.loc[np.concatenate(parts)]
        for a, b in PAIRS:
            vals[(a, b)].append(corr(sub[a], sub[b]))
    return {p: np.array(v, float) for p, v in vals.items()}


def main():
    d = pd.read_csv(os.path.join(OUT, "human_celltype_scores.csv"), index_col=0)
    qc = pd.read_csv(os.path.join(OUT, "human_cell_qc.csv"), index_col=0)
    d = d.join(qc, how="left")
    d["log_nCount"] = np.log10(d["nCount"])
    cols = ["LAM", "ECpanel", "Matrix", "ModSMC"]
    don = sorted(d["sample"].unique())
    print(f"{len(d)} cells, {len(don)} donors", flush=True)

    # ---------------------------------------------------------------- 0. how much is donor
    icc = {}
    for c in cols:
        g = d.groupby("sample")[c]
        between = float(g.mean().var(ddof=1))
        total = float(d[c].var(ddof=1))
        icc[c] = round(between / total, 3) if total > 0 else np.nan
    print("  donor share of variance (between-donor var / total):", icc, flush=True)

    # ---------------------------------------------------------------- 1. within donor
    rows = []
    for dn in don:
        s = d[d["sample"] == dn]
        row = dict(donor=dn, n_cells=len(s))
        for a, b in PAIRS:
            row[f"{a}|{b}"] = corr(s[a], s[b])
        rows.append(row)
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(OUT, "r3_within_donor.csv"), index=False)

    w = {}
    for (a, b), ref in [(("LAM", "Matrix"), ("ECpanel", "Matrix")),
                        (("LAM", "Matrix"), ("ECpanel", "LAM"))]:
        x = per[f"{a}|{b}"].to_numpy()
        y = per[f"{ref[0]}|{ref[1]}"].to_numpy()
        dif = x - y
        ok = np.isfinite(dif)
        p = stats.wilcoxon(dif[ok], alternative="greater").pvalue if ok.sum() >= 6 else np.nan
        w[f"{a}|{b} > {ref[0]}|{ref[1]}"] = dict(
            median_diff=float(np.nanmedian(dif)), n_donors=int(ok.sum()),
            n_positive=int((dif[ok] > 0).sum()), p_wilcoxon=float(p))
    print("  within-donor deltas:", {k: (round(v["median_diff"], 3), v["p_wilcoxon"])
                                     for k, v in w.items()}, flush=True)

    # ---------------------------------------------------------------- 2. donor FE
    dm = demean(d, cols)
    fe = {f"{a}|{b}": corr(dm[a], dm[b]) for a, b in PAIRS}

    # ---------------------------------------------------------------- 3. partial
    covars = ["log_nCount", "pct_ribo", "pct_mito"]
    R = residualise(d, cols, covars)
    pc = {f"{a}|{b}": corr(R[a], R[b], "pearson") for a, b in PAIRS}
    print("  donor-FE r:", {k: round(v, 3) for k, v in fe.items()}, flush=True)
    print("  partial r (depth+ribo+mito+donor):",
          {k: round(v, 3) for k, v in pc.items()}, flush=True)

    # ---------------------------------------------------------------- 4. within cell type
    srows = []
    for comp, sub in dm.groupby("compartment"):
        if len(sub) < 200:
            continue
        row = dict(compartment=comp, n_cells=len(sub))
        for a, b in PAIRS:
            row[f"{a}|{b}"] = corr(sub[a], sub[b])
        srows.append(row)
    strat = pd.DataFrame(srows)
    strat.to_csv(os.path.join(OUT, "r3_stratified.csv"), index=False)
    print("\n  per-compartment (donor-centred):", flush=True)
    print(strat.round(3).to_string(index=False), flush=True)

    # the same, with depth partialled out inside each compartment: a positive
    # within-compartment correlation could still be "deep cells score high on
    # everything", which is exactly what the depth control is for
    prows = []
    for comp, sub in d.groupby("compartment"):
        if len(sub) < 200:
            continue
        Rs = residualise(sub, cols, covars)
        row = dict(compartment=comp, n_cells=len(sub))
        for a, b in PAIRS:
            row[f"{a}|{b}"] = corr(Rs[a], Rs[b], "pearson")
        prows.append(row)
    strat_p = pd.DataFrame(prows)
    strat_p.to_csv(os.path.join(OUT, "r3_stratified_partial.csv"), index=False)
    print("\n  per-compartment (donor + depth partialled):", flush=True)
    print(strat_p.round(3).to_string(index=False), flush=True)

    # ---------------------------------------------------------------- cluster bootstrap
    bs = cluster_bootstrap(d, cols)
    ci = {}
    for (a, b), v in bs.items():
        v = v[np.isfinite(v)]
        ci[f"{a}|{b}"] = dict(median=float(np.median(v)),
                              lo=float(np.percentile(v, 2.5)),
                              hi=float(np.percentile(v, 97.5)),
                              frac_negative=float((v < 0).mean()))
    dl = bs[("LAM", "Matrix")] - bs[("ECpanel", "Matrix")]
    de = bs[("LAM", "Matrix")] - bs[("ECpanel", "LAM")]
    dl, de = dl[np.isfinite(dl)], de[np.isfinite(de)]
    boot_delta = dict(
        delta_matrix=dict(median=float(np.median(dl)),
                          lo=float(np.percentile(dl, 2.5)),
                          hi=float(np.percentile(dl, 97.5)),
                          frac_positive=float((dl > 0).mean())),
        delta_lam=dict(median=float(np.median(de)),
                       lo=float(np.percentile(de, 2.5)),
                       hi=float(np.percentile(de, 97.5)),
                       frac_positive=float((de > 0).mean())))
    print("\n  cluster bootstrap r:", {k: (round(v["median"], 3),
                                          round(v["lo"], 3), round(v["hi"], 3))
                                       for k, v in ci.items()}, flush=True)
    print("  cluster bootstrap deltas:", boot_delta, flush=True)

    holds = dict(
        # the statement that survives everything: the endothelial programme is
        # not coupled to the stromal one
        ec_matrix_null=bool(abs(fe["ECpanel|Matrix"]) < 0.1 and
                            abs(pc["ECpanel|Matrix"]) < 0.1 and
                            abs(ci["ECpanel|Matrix"]["median"]) < 0.1 and
                            strat_p["ECpanel|Matrix"].abs().max() < 0.12),
        # the statement that survives donor stratification but weakens once
        # depth is partialled: myeloid-stromal coupling exceeds endothelial-stromal
        lam_gt_ec_matrix=bool(w["LAM|Matrix > ECpanel|Matrix"]["n_positive"] >= 7 and
                              boot_delta["delta_matrix"]["frac_positive"] > 0.975),
        # the statement we used to make and can no longer make: myeloid-stromal
        # is the *tightest* pair of the three
        lam_matrix_tightest=bool(pc["LAM|Matrix"] > pc["ECpanel|LAM"] and
                                 boot_delta["delta_lam"]["frac_positive"] > 0.975),
    )

    summary = dict(
        n_cells=int(len(d)), n_donors=len(don), donor_variance_share=icc,
        within_donor=per.round(4).to_dict("records"),
        within_donor_tests=w, donor_fixed_effects=fe,
        partial_depth_ribo_mito_donor=pc,
        stratified_donor_centred=strat.round(4).to_dict("records"),
        stratified_partial=strat_p.round(4).to_dict("records"),
        cluster_bootstrap=ci, cluster_bootstrap_delta=boot_delta,
        holds=holds,
        note=("Within-donor = 8 separate estimates; donor FE = each donor's mean "
              "removed before pooling (equivalent to a random-intercept LMM when "
              "slopes do not vary); partial = Pearson on residuals after "
              "regressing on log nCount, pct_ribo, pct_mito and donor dummies. "
              "All CIs resample donors, not cells."),
    )
    with open(os.path.join(OUT, "r3_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("\n  holds:", holds, flush=True)


if __name__ == "__main__":
    main()
