"""43 - What a p-value can and cannot mean at 3 vs 3.

Two very different tests are quoted next to each other in the report and they
have different ceilings:

  * the exact permutation test (composition, domain-level) has a hard floor set
    by the number of label splits, not by the biology. At 3 vs 3 there are
    C(6,3) = 20 splits and every split has its mirror among them, so the most
    extreme assignment still returns p = 3/21 = 0.143 under the same
    (count + 1)/(splits + 1) convention the rest of the paper uses. No effect
    size, however large, can get below that;
  * the Welch t-test (the four individual-level endpoints) is not bounded that
    way, so it *can* reach 0.0038 - which is exactly why the macrophage endpoint
    is significant and the other two are not.

This script makes both ceilings explicit and adds the things a reviewer asks
for next: Hedges' g with a confidence interval, the direction-concordance rate
(how many animals move the same way), the minimum detectable effect at the
designs actually used, and the n that would be needed at the observed effect.

Outputs (results/power/):
  exact_permutation_floor.csv
  endpoint_effect_table.csv
  n_required.csv
  floor_summary.json
"""
from __future__ import annotations

import itertools
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "power")
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.05
POWER = 0.80


# ----------------------------------------------------------------- statistics
def welch_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(stats.ttest_ind(a, b, equal_var=False).statistic)


def exact_perm_p(vals, lab):
    """Same convention as scripts/29 and 32: (count + 1) / (splits + 1)."""
    vals = np.asarray(vals, float)
    lab = np.asarray(lab).astype(bool)
    obs = welch_t(vals[lab], vals[~lab])
    n1 = int(lab.sum())
    tot = cnt = 0
    for comb in itertools.combinations(range(len(vals)), n1):
        m = np.zeros(len(vals), bool)
        m[list(comb)] = True
        s = welch_t(vals[m], vals[~m])
        tot += 1
        if np.isfinite(s) and abs(s) >= abs(obs) - 1e-12:
            cnt += 1
    return (cnt + 1) / (tot + 1), tot, cnt


def floor_of(vals, n1):
    """Smallest p any assignment of these six values could return."""
    vals = np.asarray(vals, float)
    ps = []
    seen = []
    for comb in itertools.combinations(range(len(vals)), n1):
        m = np.zeros(len(vals), bool)
        m[list(comb)] = True
        seen.append(welch_t(vals[m], vals[~m]))
    for t_obs in seen:
        cnt = sum(1 for s in seen if np.isfinite(s) and abs(s) >= abs(t_obs) - 1e-12)
        ps.append((cnt + 1) / (len(seen) + 1))
    return float(min(ps))


def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n0 = len(b), len(a)          # b = treated, a = control
    df = n1 + n0 - 2
    s_p = np.sqrt(((n1 - 1) * b.var(ddof=1) + (n0 - 1) * a.var(ddof=1)) / df)
    d = (b.mean() - a.mean()) / s_p if s_p > 0 else np.nan
    J = 1 - 3 / (4 * (n1 + n0) - 9)
    g = d * J
    # CI from the non-central t, then rescaled back to the d metric
    t = (b.mean() - a.mean()) / (s_p * np.sqrt(1 / n1 + 1 / n0))
    scale = np.sqrt(1 / n1 + 1 / n0)

    def f(delta, p):
        return float(stats.nct.cdf(t, df, delta)) - p

    def solve(p):
        # nct.cdf underflows to NaN for |delta| beyond ~30 at df=4, so the
        # bracket is searched on a finite grid rather than handed to brentq
        grid = np.linspace(-30, 30, 601)
        vals = np.array([f(x, p) for x in grid])
        ok = np.isfinite(vals)
        grid, vals = grid[ok], vals[ok]
        sgn = vals >= 0
        idx = np.where(sgn[:-1] & ~sgn[1:])[0]
        if len(idx) == 0:
            return np.nan
        i = idx[0]
        return optimize.brentq(f, grid[i], grid[i + 1], args=(p,))

    lo, hi = solve(0.975), solve(0.025)
    return float(g), float(lo * scale * J), float(hi * scale * J)


def power_at(d, n, alpha=ALPHA):
    df = 2 * n - 2
    delta = abs(d) / np.sqrt(2 / n)
    tc = stats.t.ppf(1 - alpha / 2, df)
    return float(1 - stats.nct.cdf(tc, df, delta) + stats.nct.cdf(-tc, df, delta))


def n_for_power(d, power=POWER, nmax=80):
    for n in range(2, nmax + 1):
        if power_at(d, n) >= power:
            return n
    return np.nan


def mde(n, power=POWER, alpha=ALPHA):
    lo, hi = 0.0, 20.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if power_at(mid, n, alpha) < power:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    # ---------------------------------------------------------- 1. the floor
    rows = []
    for n1, n2 in [(2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (5, 5), (6, 6)]:
        tot = len(list(itertools.combinations(range(n1 + n2), n1)))
        # the observed split and its mirror are always equally extreme, so the
        # floor is 3/(tot+1) whenever the statistic is sign-symmetric
        fl = 3 / (tot + 1)
        rows.append(dict(n_per_group_if_balanced=None if n1 != n2 else n1,
                         n_treated=n1, n_control=n2, n_splits=tot,
                         exact_floor_p=fl, can_reach_005=bool(fl < ALPHA)))
    fl = pd.DataFrame(rows)
    fl.to_csv(os.path.join(OUT, "exact_permutation_floor.csv"), index=False)
    print("exact permutation floors:\n" + fl.round(4).to_string(index=False), flush=True)

    # empirical check: for random 6-value vectors the smallest attainable p
    rng = np.random.default_rng(3)
    emp = [floor_of(rng.normal(size=6), 3) for _ in range(200)]
    print(f"  empirical floor over 200 random 3v3 vectors: "
          f"{min(emp):.4f}-{max(emp):.4f}", flush=True)

    # ---------------------------------------------------------- 2. endpoints
    cm = pd.read_csv(os.path.join(RES, "D2", "cell_metrics.csv.gz"), index_col=0)
    comp = pd.read_csv(os.path.join(RES, "audit", "composition_by_sample.csv"))
    m2m = (comp.dropna(subset=["sample"]).drop_duplicates("sample")
           .set_index("sample")["mouse"].to_dict())
    cm["mouse"] = cm["sample"].map(lambda s: m2m.get(s, np.nan))
    cm = cm[cm["mouse"].notna()]

    ENDPOINTS = [("Endothelial", "LI_ConstitutiveHet", "内皮 组成型异染色质指数"),
                 ("SMC", "LI_ConstitutiveHet", "SMC 组成型异染色质指数"),
                 ("Macrophage", "mod_AgingSen", "髓系 AgingSen 模块"),
                 ("Macrophage", "LI_H3K9me3", "髓系 H3K9me3 指数")]

    rows = []
    for ct, met, label in ENDPOINTS:
        v = cm[(cm.celltype == ct) & cm.group.isin(["Control", "TCDD_8wk"])]
        v = v[["mouse", "group", met]].dropna()
        per = v.groupby(["mouse", "group"])[met].mean().reset_index()
        a = per[per.group == "Control"][met].to_numpy()
        b = per[per.group == "TCDD_8wk"][met].to_numpy()
        if len(a) < 3 or len(b) < 3:
            continue
        g, glo, ghi = hedges_g(a, b)
        d = float((b.mean() - a.mean()) / np.sqrt(
            ((len(b) - 1) * b.var(ddof=1) + (len(a) - 1) * a.var(ddof=1)) /
            (len(a) + len(b) - 2)))
        p_ex, tot, cnt = exact_perm_p(np.concatenate([a, b]),
                                      np.r_[np.zeros(len(a), bool), np.ones(len(b), bool)])
        # direction concordance: share of treated animals that lie beyond the
        # control median on the side the effect actually goes
        side = np.sign(b.mean() - a.mean())
        concord = float(np.mean(side * (b - np.median(a)) > 0))
        # common-language effect size: P(random treated > random control)
        cles = float(np.mean([[1.0 if x > y else 0.5 if x == y else 0.0
                               for y in a] for x in b]))
        rows.append(dict(
            celltype=ct, metric=met, label=label,
            ctrl=", ".join(f"{x:.3f}" for x in a),
            tcdd=", ".join(f"{x:.3f}" for x in b),
            d_cohen=d,
            g=round(g, 3), g_lo=round(glo, 3), g_hi=round(ghi, 3),
            p_welch=float(stats.ttest_ind(b, a, equal_var=False).pvalue),
            p_exact=p_ex, n_splits=tot, n_as_extreme=cnt,
            at_floor=bool(abs(p_ex - 3 / (tot + 1)) < 1e-9),
            concordance=concord, cles=cles,
            power_at_3v3=power_at(d, 3), power_at_6v6=power_at(d, 6),
            n_required=n_for_power(d),
            all_treated_beyond_controls=bool(b.min() > a.max())))
    et = pd.DataFrame(rows)
    et.to_csv(os.path.join(OUT, "endpoint_effect_table.csv"), index=False)
    print("\nper-endpoint effects:\n" + et[[
        "label", "g", "g_lo", "g_hi", "p_welch", "p_exact", "concordance",
        "power_at_3v3", "n_required"]].round(3).to_string(index=False), flush=True)

    # ---------------------------------------------------------- 3. n required
    nrows = []
    for n in range(2, 12):
        nrows.append(dict(n_per_group=n, mde_80pct=round(mde(n), 3)))
    nr = pd.DataFrame(nrows)
    nr.to_csv(os.path.join(OUT, "n_required.csv"), index=False)
    print("\nMDE by design:\n" + nr.round(3).to_string(index=False), flush=True)

    summ = dict(
        exact_floor_3v3=3 / 21,
        exact_floor_note=("With C(6,3)=20 splits and the (count+1)/(splits+1) "
                          "convention, the observed split and its mirror are the "
                          "two most extreme assignments, so 3/21 is the smallest "
                          "p the test can return at 3 vs 3. It cannot reach 0.05 "
                          "whatever the effect size."),
        smallest_design_that_can_reject=fl[fl.can_reach_005].iloc[0].to_dict(),
        empirical_floor_range=[float(min(emp)), float(max(emp))],
        mde={int(r.n_per_group): float(r.mde_80pct) for r in nr.itertuples()},
        endpoints=et.round(4).to_dict("records"),
    )
    with open(os.path.join(OUT, "floor_summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)
    print("\n  summary written", flush=True)


if __name__ == "__main__":
    main()
