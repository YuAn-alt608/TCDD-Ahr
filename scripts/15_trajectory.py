"""Per-lineage diffusion pseudotime of barrier collapse in D2 (0 / 8 / 16 weeks).

Why not RNA velocity: the processed matrices on GEO carry no spliced/unspliced
layers, and the raw reads would have to be re-quantified with velocyto/STARsolo.
What *can* be done without new data is a diffusion pseudotime per lineage with the
control extreme as root, which orders cells along their own collapse trajectory and
lets us ask: at what point of its trajectory does each lineage cross its barrier
threshold, and in what order do the lineages do so?

The trajectory is anchored on control cells by construction, so it cannot be used to
claim "treatment causes movement"; it is used only to (a) order cells within a
lineage, and (b) compare *where along that order* each lineage's fingerprints break.
Cross-lineage comparisons of absolute pseudotime are not made.
"""
import json, os, sys
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

OUT = os.path.join(L.RES, "trajectory")
os.makedirs(OUT, exist_ok=True)

METRICS = ["TES", "HGDI", "LADleak", "entropy_res", "regulon_entropy"]
TARGETS = ["Endothelial", "ModifiedSMC", "SMC", "Macrophage", "Monocyte", "Fibroblast"]


def main():
    ad = sc.read_h5ad(os.path.join(L.RES, "D2", "clustered.h5ad"))
    cm = pd.read_csv(os.path.join(L.RES, "D2", "cell_metrics.csv.gz"), index_col=0)
    common = ad.obs_names.intersection(cm.index)
    ad = ad[common].copy()
    cm = cm.loc[common]
    for m in METRICS:
        ad.obs[m] = cm[m].values

    rows, paga_rows = [], []
    for ct in TARGETS:
        sub = ad[ad.obs.celltype == ct].copy()
        if sub.n_obs < 200:
            continue
        sc.pp.neighbors(sub, use_rep="X_pca_harmony", n_neighbors=20)
        sc.tl.diffmap(sub, n_comps=15)
        # root: the control cell closest to the control centroid in diffusion space
        dc = sub.obsm["X_diffmap"]
        ctrl = (sub.obs.group == "Control").values
        if ctrl.sum() < 10:
            continue
        centroid = dc[ctrl].mean(0)
        dist = np.linalg.norm(dc - centroid, axis=1)
        dist[~ctrl] = np.inf
        sub.uns["iroot"] = int(np.argmin(dist))
        sc.tl.dpt(sub)
        pt = sub.obs["dpt_pseudotime"].values.astype(float)
        ok = np.isfinite(pt)
        if ok.sum() < 100:
            continue
        pct = np.full(len(pt), np.nan)
        pct[ok] = (pt[ok] - np.nanmin(pt[ok])) / (np.nanmax(pt[ok]) - np.nanmin(pt[ok]) + 1e-9)

        ctrl_mask = ctrl & ok
        for g in ["Control", "TCDD_8wk", "TCDD_16wk_WT", "TCDD_16wk_AhrKO"]:
            m = (sub.obs.group == g).values & ok
            if m.sum() < 20:
                continue
            rec = dict(celltype=ct, group=g, n=int(m.sum()),
                       pt_median=float(np.nanmedian(pct[m])),
                       pt_q25=float(np.nanpercentile(pct[m], 25)),
                       pt_q75=float(np.nanpercentile(pct[m], 75)))
            for met in METRICS:
                rec[f"{met}_median"] = float(np.nanmedian(sub.obs[met].values.astype(float)[m]))
            rows.append(rec)

        # lineage-level onset: first 10% bin where the profile passes the control
        # 90th percentile and stays above it in the next bin
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(pct, bins) - 1, 0, 9)
        onset, rho = {}, {}
        for met in METRICS:
            v = sub.obs[met].values.astype(float)
            thr = np.nanpercentile(v[ctrl_mask], 90)
            prof = np.array([np.nanmean(v[(idx == b) & ok]) if np.any((idx == b) & ok) else np.nan
                             for b in range(10)])
            o = np.nan
            for b in range(9):
                if np.isfinite(prof[b]) and prof[b] > thr and np.isfinite(prof[b + 1]) and prof[b + 1] > thr:
                    o = float(bins[b])
                    break
            onset[met] = o
            mm = (sub.obs.group == "TCDD_16wk_WT").values & ok
            if mm.sum() > 50:
                rho[met] = float(stats.spearmanr(pct[mm], v[mm]).statistic)
            else:
                rho[met] = np.nan
        # global PAGA for the same subset neighbourhood structure
        p_ctrl = float(np.nanmedian(pct[ctrl_mask]))
        p_wt = float(np.nanmedian(pct[(sub.obs.group == "TCDD_16wk_WT").values & ok]))
        p_ako = float(np.nanmedian(pct[(sub.obs.group == "TCDD_16wk_AhrKO").values & ok]))
        ret = (p_ako - p_wt) / (p_ctrl - p_wt) if abs(p_ctrl - p_wt) > 1e-6 else np.nan
        paga_rows.append(dict(celltype=ct, n=int(sub.n_obs),
                              pt_ctrl=p_ctrl, pt_tcdd16=p_wt, pt_ako=p_ako,
                              pt_shift=p_wt - p_ctrl, ako_return=ret,
                              **{f"onset_{k}": v for k, v in onset.items()},
                              **{f"rho_{k}": v for k, v in rho.items()}))

    tr = pd.DataFrame(rows)
    tr.to_csv(os.path.join(OUT, "dpt_by_group.csv"), index=False)
    pg = pd.DataFrame(paga_rows)
    pg.to_csv(os.path.join(OUT, "onset_by_lineage.csv"), index=False)

    # ---------------- cascade: merge measured T50 with trajectory onset
    t50 = pd.read_csv(os.path.join(L.RES, "matrix", "T50_by_lineage.csv"))
    t50 = t50.dropna(subset=["T50_weeks"])
    first = (t50.sort_values("T50_weeks").groupby("celltype", as_index=False).first()
             [["celltype", "metric", "T50_weeks"]])
    cascade = first.merge(pg, on="celltype", how="outer")
    if "T50_weeks" in cascade:
        cascade = cascade.sort_values("T50_weeks", na_position="last")
        cascade["rank"] = np.arange(1, len(cascade) + 1)
    cascade.to_csv(os.path.join(OUT, "cascade.csv"), index=False)

    summary = dict(
        n_lineages=int(len(pg)),
        exposure_aligned={r.celltype: bool(abs(r.pt_shift) > 0.05) for r in pg.itertuples()},
        pt_shift={r.celltype: float(r.pt_shift) for r in pg.itertuples()},
        ako_return={r.celltype: (None if pd.isna(r.ako_return) else float(r.ako_return)) for r in pg.itertuples()},
        order=[r.celltype for r in cascade.itertuples() if pd.notna(getattr(r, "T50_weeks", np.nan))],
        t50={r.celltype: float(r.T50_weeks) for r in cascade.itertuples()
             if hasattr(r, "T50_weeks") and pd.notna(r.T50_weeks)},
        metric={r.celltype: r.metric for r in cascade.itertuples()
                if hasattr(r, "metric") and isinstance(r.metric, str)},
        onset_median_pct=float(pg[[c for c in pg.columns if c.startswith("onset_")]].median().median()),
        note=("per-lineage DPT with control extreme as root; absolute pseudotime is not "
              "comparable across lineages; pt_shift = median pseudotime of TCDD-16wk minus "
              "control; ako_return = (AKO - WT) / (control - WT); onset = first 10%-bin above "
              "the control 90th percentile, confirmed in the next bin (mostly NaN, reported "
              "as a failed readout rather than dropped)"),
    )
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=2)
    print(pd.DataFrame(paga_rows).round(3).to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
