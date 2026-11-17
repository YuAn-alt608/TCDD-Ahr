"""29 - Composition audit: is the shift a change of cell state or a change of census?

The Reviewer Question
---------------------
"You show that Ahr-KO endothelium keeps a heterochromatin-derepressed programme.
 But your libraries are sorted tdT+ / tdT- fractions of a few mice, and the number
 of endothelial / macrophage / modified-SMC cells recovered differs between arms.
 Show me the shift survives when the *composition* is held fixed."

Three things this script insists on
-----------------------------------
1. The sort gate is design, not biology. tdT+ libraries physically cannot contain
   endothelial, fibroblast or macrophage cells; the tdT- libraries contain no SMC.
   So a "composition change" may only be evaluated *inside* a fraction. Anything
   computed across fractions is an artefact of the gate.

2. Composition is probed at a level that is independent of the annotation being
   defended. Two levels are reported:

     L1  celltype within fraction  (the census the FACS gate produced)
     L2  a de-novo substate partition inside each celltype: k-means on the
         harmonised embedding (X_pca_harmony / X_lsi_harmony), k fixed by cell
         number. This is deliberately NOT the leiden clusters used elsewhere -
         if the endpoints were composition-driven, a partition that never saw the
         endpoint labels should expose it.

3. Statistics are only computed where the design allows. Control vs TCDD-8wk has
   three mice per arm, each contributing one tdT+ and one tdT- library:
     * propeller-style test: CLR (>=3 parts) or logit (2 parts), Welch t, BH;
     * the same statistic permuted over all C(6,3)=20 label splits. Because there
       are only 20 splits, the smallest attainable two-sided p is 0.10 - this arm
       can never produce p<0.05 at the animal level, and that is stated rather than
       worked around.
   The single-mouse arms (16-wk WT, Ahr-KO) get a proportion, the min-max envelope
   of the three control mice, and a *cell-level* bootstrap CI. The two interval
   widths together are the quantitative reason n=1 composition is untestable:
   resampling cells narrows the interval, animal-to-animal spread does not.

4. Every frozen endpoint is decomposed with control as the reference:

       Delta_raw = SUM_c (w1c - w0c) mu0c            composition
                 + SUM_c  w0c (mu1c - mu0c)         within-state
                 + SUM_c (w1c - w0c)(mu1c - mu0c)   interaction

   and re-scored after post-stratifying every arm to the control composition. The
   effect size uses the *same* standardiser as scripts/25_power_n2.py (pooled
   between-animal SD from the two replicated arms for RNA; |c1-c2|/sqrt(2)*1.5 for
   ATAC with its df=1), so d_raw and d_adj are on one scale and only Delta moves.

Outputs (results/audit/):
  composition_by_sample.csv        L1 proportions per library
  composition_substate_by_sample.csv  L2 proportions per library
  composition_tests_8wk.csv        L1 tests
  substate_tests_8wk.csv           L2 tests
  composition_n1_envelope.csv      single-mouse arms vs control envelope
  endpoint_decomposition.csv       variance partition
  endpoint_adjusted_d.csv          Delta and d before/after matching composition
  summary.json
"""
import os, sys, json, itertools
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy import stats
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

OUT = os.path.join(L.RES, "audit")
os.makedirs(OUT, exist_ok=True)

GROUPS = ["Control", "TCDD_8wk", "TCDD_16wk_WT", "TCDD_16wk_AhrKO"]
LAM = ["Trem2", "Lgals3", "Csf1r", "Spp1", "Apoe", "Cd9", "Fabp5", "Lpl", "Gpnmb", "Trem1"]
MIN_STRATUM_CELLS = 15
NBOOT = 2000
ATAC_INFLATE = 1.5          # same as 25_power_n2.py: df=1 sigma is an underestimate
SEED = 0


# --------------------------------------------------------------------- statistics
def clr(P, delta):
    """Centred log-ratio with multiplicative zero replacement."""
    P = np.atleast_2d(np.asarray(P, float))
    if P.shape[1] < 2:
        return np.zeros((P.shape[0], 1))
    nz = (P == 0).sum(1)
    Q = np.clip((1.0 - nz * delta)[:, None] * P + delta[:, None] * (P == 0), 1e-12, None)
    lg = np.log(Q)
    return lg - lg.mean(1, keepdims=True)


def logit1(p, delta):
    p = np.clip(np.asarray(p, float), delta, 1 - delta)
    return np.log(p / (1 - p))[:, None]


def welch_t(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(stats.ttest_ind(a, b, equal_var=False).statistic)


def exact_perm_p(vals, lab, stat=welch_t):
    """Exact two-sided permutation p over every label split (3v3 -> 20 splits)."""
    vals = np.asarray(vals, float); lab = np.asarray(lab).astype(bool)
    obs = stat(vals[lab], vals[~lab])
    if not np.isfinite(obs):
        return np.nan, 0
    n1 = int(lab.sum())
    cnt = tot = 0
    for comb in itertools.combinations(np.arange(len(vals)), n1):
        m = np.zeros(len(vals), bool)
        m[list(comb)] = True
        s = stat(vals[m], vals[~m])
        tot += 1
        if np.isfinite(s) and abs(s) >= abs(obs) - 1e-12:
            cnt += 1
    return (cnt + 1) / (tot + 1), tot


def boot_ci_prop(k, n, nboot=NBOOT, seed=SEED):
    """Cell-level bootstrap: resample cells, keep the animal fixed."""
    if n <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    v = rng.binomial(int(n), float(k) / n, size=nboot) / n
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def kmeans_substate(E, k, seed=SEED):
    k = int(max(1, min(k, len(E) // 50)))
    if k < 2:
        return np.zeros(len(E), dtype=int).astype(str)
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(E)
    return lab.astype(str)


# --------------------------------------------------------------------- loading
def load_rna():
    a = ad.read_h5ad(os.path.join(L.RES, "D2", "processed.h5ad"))
    o = a.obs[["sample", "group", "sorted", "celltype", "leiden"]].copy()
    o["mouse"] = o["sample"].map(
        lambda s: str(s).split("_")[2] + ("" if str(s).split("_")[2].endswith("16")
                                          else str(s).split("_")[3].replace("TDT", "").replace("NTDT", "")))
    o["frac"] = o["sorted"].astype(str)
    o["celltype"] = o["celltype"].astype(str)
    # de-novo substate inside each celltype, on the harmonised PCs
    E = np.asarray(a.obsm["X_pca_harmony"])[:, :30]
    sub = np.full(len(o), "NA", dtype=object)
    for ct, pos in o.groupby("celltype").indices.items():
        pos = np.asarray(pos)
        k = 5 if len(pos) >= 1200 else (4 if len(pos) >= 500 else 3)
        sub[pos] = kmeans_substate(E[pos], k)
    o["sub"] = sub
    return a, o


def load_atac():
    a = ad.read_h5ad(os.path.join(L.RES, "ATAC", "atac_processed.h5ad"))
    o = a.obs[["sample", "group", "lineage", "celltype_atac"]].copy()
    o["mouse"] = o["sample"].map({"D2_atac_CTRL1": "CTRL1", "D2_atac_CTRL2": "CTRL2",
                                  "D2_atac_WT_TCDD": "WT16", "D2_atac_AHRKO_TCDD": "AKO16"})
    o["frac"] = "all"
    o["celltype"] = o["celltype_atac"].astype(str)
    o["leiden"] = a.obs["leiden"].astype(str)
    E = np.asarray(a.obsm["X_lsi_harmony"])[:, :30]
    sub = np.full(len(o), "NA", dtype=object)
    for ct, pos in o.groupby("celltype").indices.items():
        pos = np.asarray(pos)
        if len(pos) < 100:
            sub[pos] = "0"
            continue
        k = 5 if len(pos) >= 1200 else (4 if len(pos) >= 500 else 3)
        sub[pos] = kmeans_substate(E[pos], k)
    o["sub"] = sub
    for m in ["frac_het", "prom_het_rel", "frac_lad", "prom_het"]:
        o[m] = a.obs[m].to_numpy()
    return a, o


# --------------------------------------------------------------------- composition
def prop_table(df, keys, level, label):
    ct = pd.crosstab([df[k] for k in keys], df[level])
    ct = ct[ct.sum(1) > 0]
    out = ct.reset_index()
    out.columns = list(keys) + [f"n_{c}" for c in ct.columns]
    tot = out[[f"n_{c}" for c in ct.columns]].sum(1)
    for c in ct.columns:
        out[f"p_{c}"] = out[f"n_{c}"] / tot
    out["total_cells"] = tot
    out["level"] = label
    out["stratum"] = out[keys[:-2]].astype(str).agg("|".join, axis=1)
    return out


def test_family(tab, strat_keys, pcols, level):
    """Control vs TCDD-8wk, one family per stratum."""
    rows = []
    for stratum, sub in tab.groupby(strat_keys, observed=True):
        if not isinstance(stratum, tuple):
            stratum = (stratum,)
        c = sub[sub.group == "Control"]
        w = sub[sub.group == "TCDD_8wk"]
        if len(c) < 2 or len(w) < 2:
            continue
        # The 16-week libraries sit in the same table but are NOT part of this
        # contrast. Restricting here keeps the Welch test at 3 vs 3 and the exact
        # permutation at C(6,3)=20 splits; before this line both silently used
        # "Control vs everything else" (3 vs 5) and 56 label splits.
        sub = sub[sub.group.isin(["Control", "TCDD_8wk"])]
        c = sub[sub.group == "Control"]
        w = sub[sub.group == "TCDD_8wk"]
        use = [pc for pc in pcols
               if f"n_{pc[2:]}" in sub and (sub[f"n_{pc[2:]}"].to_numpy(float) > 0).any()]
        if len(use) < 2:
            continue
        delta = np.maximum(0.5 / sub["total_cells"].to_numpy(float), 1e-6)
        if len(use) >= 3:
            T = clr(sub[use].to_numpy(float), delta)
            comps = list(use)
        else:
            # two parts: logit of the first; the second is its complement, reported
            # for completeness without a second (redundant) test
            T = logit1(sub[use[0]].to_numpy(float), float(delta.mean()))
            comps = [use[0]]
        is_c = (sub.group == "Control").to_numpy()
        tv, pv = [], []
        for j in range(T.shape[1]):
            v = T[:, j]
            tv.append(welch_t(v[is_c], v[~is_c]))
            pv.append(stats.ttest_ind(v[is_c], v[~is_c], equal_var=False).pvalue
                      if np.isfinite(tv[-1]) else np.nan)
        fdr = L.bh(np.array(pv, float))
        for j, pc in enumerate(use):
            if j >= len(comps):
                pe, ns = exact_perm_p(sub[pc].to_numpy(float),
                                      (sub.group == "TCDD_8wk").to_numpy())
                rows.append(dict(level=level, stratum="|".join(map(str, stratum)),
                                 component=pc[2:], n_ctrl_lib=len(c), n_8wk_lib=len(w),
                                 p_ctrl=float(c[pc].mean()), p_8wk=float(w[pc].mean()),
                                 log2_ratio=float(np.log2(max(w[pc].mean(), 1e-9) /
                                                          max(c[pc].mean(), 1e-9))),
                                 welch_t=np.nan, p_welch=np.nan, fdr_welch=np.nan,
                                 p_exact_perm=pe, n_splits=ns))
                continue
            pe, ns = exact_perm_p(sub[pc].to_numpy(float),
                                  (sub.group == "TCDD_8wk").to_numpy())
            rows.append(dict(level=level, stratum="|".join(map(str, stratum)),
                             component=pc[2:], n_ctrl_lib=len(c), n_8wk_lib=len(w),
                             p_ctrl=float(c[pc].mean()), p_8wk=float(w[pc].mean()),
                             log2_ratio=float(np.log2(max(w[pc].mean(), 1e-9) /
                                                      max(c[pc].mean(), 1e-9))),
                             welch_t=tv[j],
                             p_welch=float(pv[j]) if np.isfinite(pv[j]) else np.nan,
                             fdr_welch=float(fdr[j]),
                             p_exact_perm=pe, n_splits=ns))
    return pd.DataFrame(rows)


def n1_envelope(tab, strat_keys, pcols, level):
    rows = []
    for stratum, sub in tab.groupby(strat_keys, observed=True):
        if not isinstance(stratum, tuple):
            stratum = (stratum,)
        c = sub[sub.group == "Control"]
        if len(c) < 2:
            continue
        use = [pc for pc in pcols
               if f"n_{pc[2:]}" in sub and (sub[f"n_{pc[2:]}"].to_numpy(float) > 0).any()]
        for pc in use:
            comp = pc[2:]
            lo, hi = float(c[pc].min()), float(c[pc].max())
            mu = float(c[pc].mean())
            for g in ["TCDD_16wk_WT", "TCDD_16wk_AhrKO"]:
                s = sub[sub.group == g]
                if len(s) == 0 or f"n_{comp}" not in s:
                    continue
                v = float(s[pc].iloc[0]); n = int(s[f"n_{comp}"].iloc[0])
                blo, bhi = boot_ci_prop(v * n, n)
                rows.append(dict(level=level, stratum="|".join(map(str, stratum)),
                                 component=comp, group=g, p=v,
                                 ctrl_mean=mu, ctrl_min=lo, ctrl_max=hi,
                                 ratio_to_ctrl=v / max(mu, 1e-9),
                                 inside_envelope=bool(lo <= v <= hi),
                                 boot_lo=blo, boot_hi=bhi, boot_width=bhi - blo,
                                 ctrl_width=hi - lo, n_cells=n))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- decomposition
def decompose(mu, w, ref, alt):
    strata = [s for s in mu[ref] if s in mu[alt] and s in w[ref] and s in w[alt]]
    if not strata:
        return None
    w0 = np.array([w[ref][s] for s in strata], float)
    w1 = np.array([w[alt][s] for s in strata], float)
    m0 = np.array([mu[ref][s] for s in strata], float)
    m1 = np.array([mu[alt][s] for s in strata], float)
    ok = np.isfinite(m0) & np.isfinite(m1) & (w0 > 0) & (w1 > 0)
    if ok.sum() == 0:
        return None
    w0, w1, m0, m1 = w0[ok] / w0[ok].sum(), w1[ok] / w1[ok].sum(), m0[ok], m1[ok]
    comp = float(((w1 - w0) * m0).sum())
    state = float((w0 * (m1 - m0)).sum())
    inter = float(((w1 - w0) * (m1 - m0)).sum())
    raw = float((w1 * m1).sum() - (w0 * m0).sum())
    return dict(delta_raw=raw, comp=comp, state=state, inter=inter,
                adj_delta=float((w0 * m1).sum() - (w0 * m0).sum()),
                residual=float(raw - comp - state - inter), n_strata=int(ok.sum()),
                w_ctrl=np.round(w0, 4).tolist(), w_alt=np.round(w1, 4).tolist())


def per_mouse_tables(pb, stratum_col, value_col, weight_col):
    """mouse -> (mu per stratum, w per stratum, group)."""
    out = {}
    for m, ms in pb.groupby("mouse"):
        st = ms.set_index(stratum_col)
        out[m] = (st[value_col].astype(float), st[weight_col].astype(float),
                  str(st["group"].iloc[0]))
    return out


def agg_tables(tab, mice):
    mu, w = {}, {}
    for m in mice:
        v, ww, g = tab[m]
        mu.setdefault(g, []).append(v)
        w.setdefault(g, []).append(ww)
    mu = {g: pd.concat(v, axis=1).mean(1).to_dict() for g, v in mu.items()}
    w = {g: pd.concat(v, axis=1).sum(1).to_dict() for g, v in w.items()}
    return mu, w


def animal_value(tab, m, fixed_w=None):
    v, ww, _ = tab[m]
    if fixed_w is not None:
        ww = pd.Series({k: fixed_w.get(k, 0.0) for k in v.index}, dtype=float)
    return float((v * ww).sum() / ww.sum()) if ww.sum() > 0 else np.nan


def analyse_endpoint(pb, assay, celltype, endpoint, contrasts, strat_name="sub",
                     value_col="score", tag="depth_adjusted"):
    """Per-stratum table + decomposition + composition-matched delta; bootstraps the
    3v3 contrast over animals (only where both arms have >=3 mice)."""
    tab = per_mouse_tables(pb, strat_name, value_col, "n_cells")
    mice_of = {}
    for m in tab:
        mice_of.setdefault(tab[m][2], []).append(m)
    mu, w = agg_tables(tab, list(tab))

    # control reference weights
    cw = pd.concat([tab[m][1] for m in mice_of.get("Control", [])], axis=1).sum(1)
    cw = (cw / cw.sum()).to_dict()

    strat_rows, dec_rows, adj_rows = [], [], []
    for alt in contrasts:
        if alt not in mu:
            continue
        d = decompose(mu, w, "Control", alt)
        if d is None:
            continue
        d.update(dict(assay=assay, celltype=celltype, endpoint=endpoint,
                      contrast=f"{alt} vs Control", estimator=tag))
        dec_rows.append(d)

        # per-stratum effect: does the shift exist inside every substate?
        for s in sorted(set(mu["Control"]) & set(mu[alt])):
            m0, m1 = mu["Control"].get(s, np.nan), mu[alt].get(s, np.nan)
            n0 = float(np.mean([tab[m][1].get(s, np.nan) for m in mice_of.get("Control", [])]))
            n1 = float(np.mean([tab[m][1].get(s, np.nan) for m in mice_of.get(alt, [])]))
            strat_rows.append(dict(assay=assay, celltype=celltype, endpoint=endpoint,
                                   contrast=f"{alt} vs Control", estimator=tag, stratum=s,
                                   mu_ctrl=m0, mu_alt=m1, delta=m1 - m0,
                                   n_cells_ctrl=n0, n_cells_alt=n1))
        neg = sum(1 for r in strat_rows
                  if r["contrast"] == f"{alt} vs Control" and r["delta"] < 0)
        tot = sum(1 for r in strat_rows if r["contrast"] == f"{alt} vs Control")
        dec_rows[-1]["n_strata_same_sign"] = max(neg, tot - neg)
        dec_rows[-1]["n_strata"] = tot

        # raw vs composition-matched, per mouse
        gvals = {}
        for mode, fwish in [("raw", None), ("composition_matched", cw)]:
            vals = {g: np.array([animal_value(tab, m, fwish) for m in mice_of[g]], float)
                    for g in mice_of}
            gvals[mode] = vals
            delta = float(np.nanmean(vals[alt]) - np.nanmean(vals["Control"]))
            adj_rows.append(dict(assay=assay, celltype=celltype, endpoint=endpoint,
                                 contrast=f"{alt} vs Control", estimator=tag, mode=mode,
                                 delta=delta,
                                 mean_ctrl=float(np.nanmean(vals["Control"])),
                                 mean_alt=float(np.nanmean(vals[alt])),
                                 n_ctrl=len(vals["Control"]), n_alt=len(vals[alt]),
                                 values_ctrl=np.round(vals["Control"], 6).tolist(),
                                 values_alt=np.round(vals[alt], 6).tolist()))
        rv = gvals["raw"]
        sg = (sigma_rna({"Control": rv["Control"], "TCDD_8wk": rv.get("TCDD_8wk", [])})
              if assay == "RNA" else sigma_atac({"Control": rv["Control"]}))
        for r in adj_rows:
            if (r["assay"] == assay and r["celltype"] == celltype and
                    r["endpoint"] == endpoint and r["contrast"] == f"{alt} vs Control" and
                    r["estimator"] == tag):
                r["sigma"] = sg

        # animal bootstrap, only for the replicated contrast
        if alt == "TCDD_8wk" and len(mice_of.get("Control", [])) >= 3 and len(mice_of[alt]) >= 3:
            rng = np.random.default_rng(SEED)
            B = 400
            draws = {"comp": [], "state": [], "delta_raw": [], "delta_adj": []}
            c0, c1 = mice_of["Control"], mice_of[alt]
            for _ in range(B):
                b0 = [c0[i] for i in rng.integers(0, len(c0), len(c0))]
                b1 = [c1[i] for i in rng.integers(0, len(c1), len(c1))]
                mub, wb = agg_tables(tab, b0 + b1)
                db = decompose(mub, wb, "Control", alt)
                if db is None:
                    continue
                draws["comp"].append(db["comp"])
                draws["state"].append(db["state"])
                draws["delta_raw"].append(db["delta_raw"])
                draws["delta_adj"].append(db["adj_delta"])
            for k, v in draws.items():
                if v:
                    dec_rows[-1][f"boot_{k}_lo"] = float(np.percentile(v, 2.5))
                    dec_rows[-1][f"boot_{k}_hi"] = float(np.percentile(v, 97.5))
    return strat_rows, dec_rows, adj_rows


def sigma_rna(v):
    """Pooled between-animal SD from the two replicated arms (df = 4), as in 25."""
    c0 = np.asarray(v.get("Control", []), float)
    c8 = np.asarray(v.get("TCDD_8wk", []), float)
    c0 = c0[np.isfinite(c0)]; c8 = c8[np.isfinite(c8)]
    if len(c0) < 2 or len(c8) < 2:
        return np.nan
    return float(np.sqrt(((len(c0) - 1) * c0.var(ddof=1) + (len(c8) - 1) * c8.var(ddof=1))
                         / (len(c0) + len(c8) - 2)))


def sigma_atac(v):
    """|c1-c2|/sqrt(2), df = 1, inflated 1.5x as in 25_power_n2.py."""
    c0 = np.asarray(v.get("Control", []), float)
    c0 = c0[np.isfinite(c0)]
    if len(c0) < 2:
        return np.nan
    return float(abs(c0[0] - c0[1]) / np.sqrt(2) * ATAC_INFLATE)


# --------------------------------------------------------------------- RNA module
def rna_cell_scores(a, obs, genes):
    """Per-cell module score = mean log1p(CPM) over the gene set, plus a depth-
    residualised version.

    Why per-cell: the mean of per-cell scores is *linear*, so
    mean(all cells) = SUM_c w_c mean(cells in c) holds exactly and the
    composition/state partition carries no algebraic residual. The frozen endpoint
    instead uses a per-mouse pseudobulk score, which is not additive across
    strata - it is reported alongside as the paper's own estimator, never used for
    the partition.

    Why residualise: for sparse gene sets (LAM: Trem2, Lgals3...) the per-cell
    score is mostly a detection rate and therefore tracks sequencing depth. One
    16-wk library sits at 1,274 UMI/cell against 6,000-10,000 everywhere else, so
    an unadjusted per-cell mean would report "depth" as "biology". The adjustment
    is a linear regression on log total UMI fitted on the *control* cells of the
    same celltype and applied to every cell - linear, so additivity across
    substates survives. Detected-gene count is deliberately NOT adjusted for: for
    a gene-set score it is a mediator of the biology, not a confounder.
    """
    gi = [a.var_names.get_loc(g) for g in genes if g in a.var_names]
    cnt = sp.csr_matrix(a.layers["counts"])
    tot = np.asarray(cnt.sum(1)).ravel()
    ng = np.asarray((cnt > 0).sum(1)).ravel()
    sc = np.log1p(np.asarray(cnt[:, gi].todense(), float) /
                  np.maximum(tot, 1)[:, None] * 1e6).mean(1)
    Xd = np.c_[np.ones(len(sc)), np.log1p(tot)]
    fit = (obs["group"].to_numpy() == "Control") & np.isfinite(sc)
    beta, *_ = np.linalg.lstsq(Xd[fit], sc[fit], rcond=None)
    adj = sc - Xd @ beta + float(np.mean(sc[fit]))
    d = pd.DataFrame({"mouse": obs["mouse"].to_numpy(), "group": obs["group"].to_numpy(),
                      "sub": obs["sub"].to_numpy(), "score": adj, "score_raw": sc})
    g = (d.groupby(["mouse", "group", "sub"])
         .agg(score=("score", "mean"), score_raw=("score_raw", "mean"),
              n_cells=("score", "size")).reset_index())

    # the paper's own estimator, one level down: pooled counts per (mouse, substate)
    # -> CPM -> log1p -> mean over genes. Not additive across substates, so it is
    # used for the per-substate table and reported with its additivity residual.
    key = (obs["mouse"].astype(str) + "|" + obs["sub"].astype(str)).to_numpy()
    uk, inv = np.unique(key, return_inverse=True)
    ind = sp.csr_matrix((np.ones(len(inv), np.float32), (inv, np.arange(len(inv)))),
                        shape=(len(uk), len(inv)))
    pb = np.asarray((ind @ cnt[:, gi]).todense(), float)
    lib = np.asarray((ind @ cnt).todense(), float).sum(1, keepdims=True)
    pbs = pd.DataFrame([k.split("|") for k in uk], columns=["mouse", "sub"])
    pbs["score_pb"] = np.log1p(pb / np.maximum(lib, 1) * 1e6).mean(1)
    g = g.merge(pbs, on=["mouse", "sub"], how="left")
    return g


def main():
    rna_a, rna = load_rna()
    atac_a, atac = load_atac()

    # ---------------- 0 the gate, stated once
    print("=== RNA: which celltype is observable in which sorted fraction ===", flush=True)
    gate = pd.crosstab(rna.celltype, rna.frac)
    print(gate.to_string(), flush=True)
    gate.to_csv(os.path.join(OUT, "sort_gate_check.csv"))

    cells = pd.concat([rna[["mouse", "group", "sample", "frac", "celltype", "sub"]],
                       atac[["mouse", "group", "sample", "frac", "celltype", "sub"]]],
                      ignore_index=True)
    cells["assay"] = ["RNA"] * len(rna) + ["ATAC"] * len(atac)

    # ---------------- 1 tables
    k1 = ["assay", "frac", "group", "mouse", "sample"]
    l1 = prop_table(cells, k1, "celltype", "L1 celltype/fraction")
    l1.to_csv(os.path.join(OUT, "composition_by_sample.csv"), index=False)

    k2 = ["assay", "frac", "celltype", "group", "mouse", "sample"]
    l2 = prop_table(cells, k2, "sub", "L2 substate/celltype")
    l2.to_csv(os.path.join(OUT, "composition_substate_by_sample.csv"), index=False)

    # ---------------- 2 L1 tests (only RNA: ATAC has 2 vs 1 libraries)
    pcols1 = [c for c in l1.columns if c.startswith("p_") and c != "p_NA"]
    t1 = test_family(l1[l1.assay == "RNA"], ["assay", "frac"], pcols1,
                     "L1 celltype/fraction")
    t1.to_csv(os.path.join(OUT, "composition_tests_8wk.csv"), index=False)
    print("\n=== L1 composition, Control vs TCDD-8wk (3 vs 3 mice) ===", flush=True)
    print((t1[["stratum", "component", "p_ctrl", "p_8wk", "log2_ratio", "welch_t",
               "p_welch", "fdr_welch", "p_exact_perm"]].round(4).to_string(index=False)
           if len(t1) else "  (no stratum had both arms with >=2 libraries)"), flush=True)

    # ---------------- 3 L2 tests
    keep_ct = ["Endothelial", "Macrophage", "SMC", "ModifiedSMC", "Fibroblast"]
    sub_rna = l2[(l2.assay == "RNA") & (l2.celltype.isin(keep_ct))]
    pcols2 = [c for c in sub_rna.columns if c.startswith("p_") and c != "p_NA"]
    t2 = test_family(sub_rna, ["assay", "frac", "celltype"], pcols2,
                     "L2 substate/celltype")
    t2.to_csv(os.path.join(OUT, "substate_tests_8wk.csv"), index=False)
    print("\n=== L2 de-novo substate composition inside celltypes ===", flush=True)
    print((t2[["stratum", "component", "p_ctrl", "p_8wk", "log2_ratio", "p_welch",
               "fdr_welch"]].round(4).to_string(index=False) if len(t2) else "  (none)"),
          flush=True)

    # ---------------- 4 n=1 arms
    e1 = n1_envelope(l1[l1.assay == "RNA"], ["assay", "frac"], pcols1,
                     "L1 celltype/fraction")
    e2 = n1_envelope(sub_rna, ["assay", "frac", "celltype"], pcols2,
                     "L2 substate/celltype")
    env = pd.concat([e1, e2], ignore_index=True)
    env.to_csv(os.path.join(OUT, "composition_n1_envelope.csv"), index=False)
    print("\n=== single-mouse arms against the 3-control envelope (L1) ===", flush=True)
    if len(e1):
        print(e1[["stratum", "component", "group", "p", "ctrl_min", "ctrl_max",
                  "inside_envelope", "boot_width", "ctrl_width"]]
              .round(4).to_string(index=False), flush=True)
        print(f"\ncells-resampling CI width / animal-spread width: median "
              f"{(e1.boot_width / e1.ctrl_width.replace(0, np.nan)).median():.3f}",
              flush=True)

    # ---------------- 5 endpoint decomposition
    dec_rows, adj_rows, strat_rows = [], [], []

    for ct, mod, genes in [("Endothelial", "het_derepression",
                            sorted(L.load_set("ConstitutiveHet"))),
                           ("Macrophage", "LAM", LAM)]:
        m = (rna.celltype == ct).to_numpy()
        obs = rna[m].copy()
        pb = rna_cell_scores(rna_a[m], obs, genes)
        for col, tag in [("score", "depth_adjusted"), ("score_raw", "per_cell_raw"),
                         ("score_pb", "substate_pseudobulk")]:
            sr, dr, ar = analyse_endpoint(pb, "RNA", ct, mod,
                                          ["TCDD_8wk", "TCDD_16wk_WT", "TCDD_16wk_AhrKO"],
                                          value_col=col, tag=tag)
            strat_rows += sr; dec_rows += dr; adj_rows += ar

    for ct, metric in [("Parenchyma", "frac_het"), ("Endothelial", "prom_het_rel")]:
        sub = atac[atac.lineage == ct].copy()
        if len(sub) < 100:
            continue
        g = (sub.groupby(["mouse", "group", "sub"])[metric]
             .agg(score="mean", n_cells="size").reset_index())
        sr, dr, ar = analyse_endpoint(g, "ATAC", ct, metric,
                                      ["TCDD_16wk_WT", "TCDD_16wk_AhrKO"],
                                      value_col="score", tag="per_cell")
        strat_rows += sr; dec_rows += dr; adj_rows += ar

    dec = pd.DataFrame(dec_rows)
    adj = pd.DataFrame(adj_rows)
    strat = pd.DataFrame(strat_rows)
    adj["d"] = adj.delta / adj.sigma.replace(0, np.nan)
    adj.drop(columns=["sigma"]).to_csv(os.path.join(OUT, "endpoint_adjusted_d.csv"), index=False)

    dec["pct_composition"] = 100 * dec.comp / dec.delta_raw.replace(0, np.nan)
    dec["pct_state"] = 100 * dec.state / dec.delta_raw.replace(0, np.nan)
    dec["delta_retained"] = dec.adj_delta / dec.delta_raw.replace(0, np.nan)
    dec.drop(columns=["w_ctrl", "w_alt"]).to_csv(
        os.path.join(OUT, "endpoint_decomposition.csv"), index=False)
    strat.to_csv(os.path.join(OUT, "endpoint_by_substate.csv"), index=False)

    print("\n=== per-substate effect (does the shift exist inside every substate?) ===",
          flush=True)
    if len(strat):
        print(strat.round(5).to_string(index=False), flush=True)

    print("\n=== variance partition (control as the reference) ===", flush=True)
    cols = ["assay", "celltype", "endpoint", "estimator", "contrast", "delta_raw", "comp",
            "state", "inter", "pct_composition", "delta_retained", "n_strata",
            "n_strata_same_sign"]
    print(dec[[c for c in cols if c in dec]].round(5).to_string(index=False), flush=True)

    print("\n=== Delta and d before / after matching the control composition ===", flush=True)
    keys = ["assay", "celltype", "endpoint", "estimator", "contrast"]

    def wide(val):
        t = adj.set_index(keys + ["mode"])[val].unstack("mode").reset_index()
        cols = []
        for c in t.columns:
            if isinstance(c, tuple):
                cols.append("_".join([x for x in c if x]))
            else:
                cols.append(c)
        t.columns = cols
        return t.rename(columns={k: f"{val}_{k}" for k in
                                 ["raw", "composition_matched"] if k in t.columns})

    pt = wide("delta").merge(wide("d"), on=keys, how="outer")
    pt["delta_retained"] = (pt.delta_composition_matched /
                            pt.delta_raw.replace(0, np.nan)).round(3)
    print(pt.round(4).to_string(index=False), flush=True)

    # ---------------- 5b verdict per endpoint (paper estimator where it exists)
    pref = {"RNA": "substate_pseudobulk", "ATAC": "per_cell"}
    dec["frac_strata_same_sign"] = dec.n_strata_same_sign / dec.n_strata
    dec["verdict"] = np.select(
        [(dec.delta_retained >= 0.7) & (dec.frac_strata_same_sign >= 0.75),
         (dec.delta_retained < 0.5)],
        ["cell-intrinsic (survives composition matching)", "composition-sensitive"], "mixed")
    vd = dec[dec.apply(lambda r: r.estimator == pref[r.assay], axis=1)].copy()
    vd = vd[["assay", "celltype", "endpoint", "contrast", "estimator", "delta_raw", "comp",
             "state", "pct_composition", "delta_retained", "frac_strata_same_sign", "verdict"]]
    vd.to_csv(os.path.join(OUT, "endpoint_verdict.csv"), index=False)
    print("\n=== verdict per frozen endpoint ===", flush=True)
    print(vd.round(4).to_string(index=False), flush=True)

    # ---------------- 6 summary
    sm = {
        "design": {
            "rna": "3 Control + 3 TCDD-8wk mice; each mouse contributes one tdT+ and one "
                   "tdT- library. 16-wk WT and Ahr-KO are one mouse each (both fractions).",
            "atac": "2 Control libraries, 1 WT, 1 Ahr-KO, 4000 cells/library.",
            "sort_gate": "tdT+ libraries contain only SMC/ModifiedSMC; tdT- libraries "
                         "contain no SMC. Composition is compared inside a fraction only.",
        },
        "substate_partition": "k-means (k=3-5 by cell number) on the harmonised embedding, "
                              "recomputed per celltype and independent of the endpoint labels",
        "min_attainable_p_3v3": "about 0.05 - there are only 20 animal-level label splits, so p<0.05 is reachable only in the single most extreme split",
        "L1": {"n_tests": int(len(t1)),
               "n_fdr05": int((t1.fdr_welch < 0.05).sum()) if len(t1) else 0},
        "L2": {"n_tests": int(len(t2)),
               "n_fdr05": int((t2.fdr_welch < 0.05).sum()) if len(t2) else 0},
        "n1_envelope": {
            "n_rows": int(len(env)),
            "frac_inside_control_envelope": round(float(env.inside_envelope.mean()), 3)
            if len(env) else None,
            "median_boot_over_ctrl_width": round(float(
                (env.boot_width / env.ctrl_width.replace(0, np.nan)).median()), 3)
            if len(env) else None,
        },
        "verdict": vd.verdict.value_counts().to_dict(),
        "endpoints": {f"{r.endpoint}|{r.contrast}": {
            "delta_raw": round(r.delta_raw, 6),
            "pct_composition": None if not np.isfinite(r.pct_composition)
            else round(float(r.pct_composition), 1),
            "delta_retained_after_matching": None if not np.isfinite(r.delta_raw) or r.delta_raw == 0
            else round(float(r.adj_delta / r.delta_raw), 3),
            "frac_strata_same_sign": round(float(r.frac_strata_same_sign), 2),
            "verdict": r.verdict}
            for r in dec.itertuples()},
        "note": ("Proportions come from libraries downsampled to a fixed per-sample cell cap, "
                 "so they are unbiased but noisier than the raw libraries. The KO/WT arms "
                 "have one mouse and are reported against the control envelope, never as "
                 "tests; their cell-level bootstrap CI covers sampling noise only."),
    }
    json.dump(sm, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=str)
    print("\n" + json.dumps(sm, indent=2, default=str), flush=True)
    print("done")


if __name__ == "__main__":
    main()
