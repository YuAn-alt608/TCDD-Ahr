"""31 - Cell-subsampling and outlier robustness of the four frozen endpoints.

The Reviewer Question
---------------------
"Your KO arm is one mouse. Maybe three extreme cells drag the mean."

What is computed, per endpoint
------------------------------
1. Exact leave-one-cell-out (jackknife). For a mean, dropping cell i is
   (S - x_i)/(n - 1), so the whole influence distribution is available without
   refitting. Reported: the range of d over every single-cell deletion, i.e. the
   most one cell can possibly move the conclusion.

2. 1,000 x without-replacement 80% subsampling *within every animal*. For the RNA
   endpoints the subsample is pushed through the paper's own estimator (pooled
   counts -> CPM -> log1p -> module mean), not through a per-cell shortcut, so the
   distribution is the distribution of the published statistic. Reported: fraction
   of draws that keep the sign of the full-data d, and the 2.5-97.5% interval of d.

3. Trimming. The top and bottom 1% of cells *within each animal* are dropped and d
   is recomputed - the blunt version of "is this an outlier artefact".

The standardiser is held fixed at the frozen sigma (scripts/25_power_n2.py), so the
only thing that moves between draws is the numerator. Any spread in d is therefore
cell sampling, not sigma estimation.

Outputs (results/audit/):
  subsample_draws.csv      per draw d for each endpoint and contrast (long form)
  subsample_summary.csv    per endpoint: d_full, jackknife range, sign stability
  subsample_summary.json
"""
import os, sys, json
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

OUT = os.path.join(L.RES, "audit")
os.makedirs(OUT, exist_ok=True)

NDRAW = 1000
FRAC = 0.8
ATAC_INFLATE = 1.5
LAM = ["Trem2", "Lgals3", "Csf1r", "Spp1", "Apoe", "Cd9", "Fabp5", "Lpl", "Gpnmb", "Trem1"]
ATAC_LIB = {"D2_atac_CTRL1": "CTRL1", "D2_atac_CTRL2": "CTRL2",
            "D2_atac_WT_TCDD": "WT16", "D2_atac_AHRKO_TCDD": "AKO16"}
FROZEN = pd.read_csv(os.path.join(L.RES, "power", "endpoint_power.csv"))
FROZEN = FROZEN[FROZEN.primary].set_index(["assay", "celltype", "endpoint"])


def sigma_of(assay, celltype, endpoint):
    return float(FROZEN.loc[(assay, celltype, endpoint), "sigma_within"])


def d_of(delta, sig):
    return delta / sig if np.isfinite(sig) and sig > 0 else np.nan


def summarise(name, meta, d_full, d_draws, d_loo, d_trim, extra=None):
    d_draws = np.asarray(d_draws, float)
    same = float(np.mean(np.sign(d_draws) == np.sign(d_full)))
    row = dict(**meta, endpoint_name=name, d_full=d_full,
               n_draws=len(d_draws),
               draw_lo=float(np.percentile(d_draws, 2.5)),
               draw_hi=float(np.percentile(d_draws, 97.5)),
               draw_sd=float(np.nanstd(d_draws)),
               frac_same_sign=same,
               frac_within_50pct=float(np.mean(np.abs(d_draws - d_full) <= 0.5 * abs(d_full))),
               jack_lo=float(np.nanmin(d_loo)) if len(d_loo) else np.nan,
               jack_hi=float(np.nanmax(d_loo)) if len(d_loo) else np.nan,
               jack_same_sign=float(np.mean(np.sign(d_loo) == np.sign(d_full)))
               if len(d_loo) else np.nan,
               d_trim1pct=d_trim)
    if extra:
        row.update(extra)
    return row


# ------------------------------------------------------------------ ATAC
def atac_part(draw_rows, sum_rows):
    m = pd.read_csv(os.path.join(L.RES, "ATAC", "atac_cell_metrics.csv.gz"), index_col=0)
    m["mouse"] = m["sample"].map(ATAC_LIB)
    rng = np.random.default_rng(0)
    for lineage, metric in [("Parenchyma", "frac_het"), ("Endothelial", "prom_het_rel")]:
        sub = m[m.lineage == lineage]
        sig = sigma_of("ATAC", lineage, metric)
        per = {k: g[metric].to_numpy(float) for k, g in sub.groupby("mouse")}
        grp = sub.groupby("mouse").group.first().to_dict()
        full = {k: v.mean() for k, v in per.items()}

        def d_from(vals, alt):
            return d_of(vals[alt] - vals["Control"], sig)

        for alt in ["TCDD_16wk_AhrKO", "TCDD_16wk_WT"]:
            if alt not in full:
                continue
            d_full = d_from(full, alt)
            name = f"{lineage}|{metric}"
            meta = dict(assay="ATAC", celltype=lineage, endpoint=metric,
                        contrast=f"{alt} vs Control")

            # 80% subsampling, all animals independently
            draws = []
            for _ in range(NDRAW):
                vals = {k: float(rng.choice(v, max(int(FRAC * len(v)), 5), replace=False).mean())
                        for k, v in per.items()}
                draws.append(d_from(vals, alt))
                draw_rows.append(dict(**meta, draw=len(draws), d=draws[-1], kind="subsample80"))
            # jackknife over every cell of every animal
            loo = []
            for k, v in per.items():
                s, n = v.sum(), len(v)
                others = {kk: vv.mean() for kk, vv in per.items()}
                if n > 1:
                    others[k] = (s - v) / (n - 1)
                loo.append(d_from(others, alt))
            for x in loo:
                draw_rows.append(dict(**meta, draw=0, d=x, kind="jackknife_cell"))

            # 1% trimming within each animal
            trimmed = {}
            for k, v in per.items():
                q1, q99 = np.percentile(v, [1, 99])
                w = v[(v >= q1) & (v <= q99)]
                trimmed[k] = w.mean() if len(w) else v.mean()
            d_trim = d_from(trimmed, alt)
            sum_rows.append(summarise(name, meta, d_full, draws, loo, d_trim))

    return draw_rows, sum_rows


# ------------------------------------------------------------------ RNA
def rna_part(draw_rows, sum_rows):
    a = ad.read_h5ad(os.path.join(L.RES, "D2", "processed.h5ad"))
    cnt = sp.csr_matrix(a.layers["counts"])
    obs = a.obs.copy()
    obs["mouse"] = obs["sample"].map(
        lambda s: str(s).split("_")[2] + ("" if str(s).split("_")[2].endswith("16")
                                          else str(s).split("_")[3].replace("TDT", "").replace("NTDT", "")))
    genes = np.asarray(a.var_names)
    gidx = {g: i for i, g in enumerate(genes)}
    rng = np.random.default_rng(1)

    for ct, mod, glist in [("Endothelial", "het_derepression",
                            sorted(L.load_set("ConstitutiveHet"))),
                           ("Macrophage", "LAM", LAM)]:
        gi = [gidx[g] for g in glist if g in gidx]
        mask = (obs["celltype"].astype(str) == ct).to_numpy()
        X = np.asarray(cnt[mask][:, gi].todense(), float)          # cells x genes
        tot = np.asarray(cnt[mask].sum(1)).ravel()
        mouse = obs["mouse"].astype(str).to_numpy()[mask]
        group = obs["group"].astype(str).to_numpy()[mask]
        sig = sigma_of("RNA", ct, mod)

        idx = {m: np.where(mouse == m)[0] for m in np.unique(mouse)}
        gmap = {m: group[idx[m][0]] for m in idx}

        def score(rows):
            S = X[rows].sum(0)
            L = tot[rows].sum()
            return float(np.log1p(S / max(L, 1) * 1e6).mean())

        full = {m: score(r) for m, r in idx.items()}
        for alt in ["TCDD_16wk_AhrKO", "TCDD_16wk_WT"]:
            if alt not in gmap.values():
                continue
            m_alt = [m for m in idx if gmap[m] == alt]
            m_ctl = [m for m in idx if gmap[m] == "Control"]
            if not m_alt:
                continue
            d_full = d_of(np.mean([full[m] for m in m_alt]) -
                          np.mean([full[m] for m in m_ctl]), sig)
            name = f"{ct}|{mod}"
            meta = dict(assay="RNA", celltype=ct, endpoint=mod,
                        contrast=f"{alt} vs Control")

            draws = []
            for _ in range(NDRAW):
                vals = {}
                for m, r in idx.items():
                    k = max(int(FRAC * len(r)), 10)
                    vals[m] = score(rng.choice(r, k, replace=False))
                draws.append(d_of(np.mean([vals[m] for m in m_alt]) -
                                  np.mean([vals[m] for m in m_ctl]), sig))
                draw_rows.append(dict(**meta, draw=len(draws), d=draws[-1], kind="subsample80"))

            # jackknife: drop one cell from one animal (other animals at full value)
            loo = []
            for m, r in idx.items():
                if len(r) < 3:
                    continue
                for i in r:
                    keep = r[r != i]
                    others_alt = [full[x] if x != m else score(keep) for x in m_alt]
                    others_ctl = [full[x] if x != m else score(keep) for x in m_ctl]
                    if not others_alt or not others_ctl:
                        continue
                    loo.append(d_of(np.mean(others_alt) - np.mean(others_ctl), sig))
            for x in loo:
                draw_rows.append(dict(**meta, draw=0, d=x, kind="jackknife_cell"))

            trimmed = {}
            for m, r in idx.items():
                v = X[r].sum(1) / np.maximum(tot[r], 1)
                q1, q99 = np.percentile(full[m] * 0 + v, [1, 99])
                keep = r[(v >= q1) & (v <= q99)]
                trimmed[m] = score(keep) if len(keep) > 5 else full[m]
            d_trim = d_of(np.mean([trimmed[m] for m in m_alt]) -
                          np.mean([trimmed[m] for m in m_ctl]), sig)
            sum_rows.append(summarise(name, meta, d_full, draws, loo, d_trim))

    return draw_rows, sum_rows


def main():
    draw_rows, sum_rows = [], []
    draw_rows, sum_rows = atac_part(draw_rows, sum_rows)
    print("ATAC done", flush=True)
    draw_rows, sum_rows = rna_part(draw_rows, sum_rows)
    print("RNA done", flush=True)

    pd.DataFrame(draw_rows).to_csv(os.path.join(OUT, "subsample_draws.csv"), index=False)
    s = pd.DataFrame(sum_rows)
    s.to_csv(os.path.join(OUT, "subsample_summary.csv"), index=False)
    print("\n=== cell-subsampling robustness ===", flush=True)
    print(s[["assay", "celltype", "endpoint", "contrast", "d_full", "draw_lo", "draw_hi",
             "frac_same_sign", "frac_within_50pct", "jack_lo", "jack_hi", "jack_same_sign",
             "d_trim1pct"]].round(3).to_string(index=False), flush=True)

    sm = {
        "n_draws_per_endpoint": NDRAW, "subsample_fraction": FRAC,
        "min_frac_same_sign": round(float(s.frac_same_sign.min()), 3),
        "min_frac_within_50pct": round(float(s.frac_within_50pct.min()), 3),
        "all_keep_sign_95pct": bool((s.frac_same_sign >= 0.95).all()),
        "all_keep_magnitude_95pct": bool((s.frac_within_50pct >= 0.95).all()),
        "worst": s.loc[s.frac_same_sign.idxmin(),
                       ["assay", "celltype", "endpoint", "contrast", "frac_same_sign",
                        "frac_within_50pct", "jack_lo", "jack_hi"]].to_dict(),
        "note": ("The standardiser is the frozen sigma, so the spread of d is cell sampling "
                 "only. The jackknife columns are the range of d when a single cell is "
                 "deleted - the most one cell can possibly do."),
    }
    json.dump(sm, open(os.path.join(OUT, "subsample_summary.json"), "w"), indent=2, default=str)
    print("\n" + json.dumps(sm, indent=2, default=str), flush=True)
    print("done")


if __name__ == "__main__":
    main()
