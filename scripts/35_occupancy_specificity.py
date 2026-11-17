"""35 - Close ring 5 with a dry experiment: is AHR occupancy ligand- and
cell-type-specifically enriched in the bins that open in mouse SMC?

10.5 compared the study's own narrowPeak set to the ATAC bins and got a negative
result, but 10.9 showed that this peak set contradicts its own signal track (AHRR
TSS carries a 96th-percentile summit that was never called). So the peak set is
unusable, not the biology. This script throws the peak set away and asks the
quantitative question directly on the raw signal:

  for every mm10 5-kb bin, lift it to hg38, then read the HCASMC AHR ChIP bigWig
  (TCDD and DMSO, the same experiment) over the orthologous interval.

Four specificities must hold together, which is what makes this hard to explain
away by "this region is just easy to ChIP":

  1. ligand    - TCDD signal is higher in SMC-open bins than in matched closed
                 bins, while DMSO signal is not (tested directly as the paired
                 difference TCDD - DMSO inside each bin);
  2. cell type - bins that open in endothelial cells (same animals, same ligand)
                 are not enriched, so the effect tracks the lineage where the
                 chromatin change happens;
  3. region    - the enrichment survives matching on GC decile and TSS presence
                 (stratified resampling, 5,000 draws);
  4. depth     - coverage is not itself imbalanced between the sets, otherwise a
                 difference in mappability could masquerade as occupancy.

Signal is summarised two ways: the mean over the 5-kb bin, and the maximum of
500-bp tiles (a summit proxy - a real AHR footprint is ~300-600 bp and a 5-kb
mean dilutes it, which is exactly what made the deposited narrowPeak set look
empty at AHRR).

Zero handling: these are coverage tracks, so a window with no data block means
"no reads here", i.e. zero, not missing. Zero is therefore used for the primary
statistic, and `cov` (fraction of the window covered by any block) is carried
along so that test 4 can check the zero-filling is not doing the work.

Outputs (results/chip_alt/):
  occupancy_bins.csv    per-bin signal for every lifted candidate
  occupancy_tests.csv   the specificities, mean and summit statistics
  occupancy_summary.json

Caching: the lifted table and one .npz per track are written as soon as they are
ready, so a killed run resumes instead of re-reading 600 MB of bigWig.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_bigwig import BigWig

# lib_core pulls in anndata/scanpy for the single-cell side of the project; this
# script only needs paths, so it computes them itself and stays runnable in a
# plain numpy/pandas interpreter.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "chip_alt")
os.makedirs(OUT, exist_ok=True)

EXT = os.path.join(ROOT, "data", "external", "chip_alt")
BW = {
    "HCASMC_TCDD": os.path.join(EXT, "HCASMC_AHR_TCDD.bw"),
    "HCASMC_DMSO": os.path.join(EXT, "HCASMC_AHR_DMSO.bw"),
    "SKMEL2_AHROE": os.path.join(EXT, "SKMEL2_AHR_AHROE_1.bw"),
    "SKMEL2_ctrl": os.path.join(EXT, "SKMEL2_AHR_Control_1.bw"),
    "K562_CUTRUN": os.path.join(EXT, "K562_AHR_CUTRUN_WTF.bw"),
}
N_PERM = 5000
POOL_N = 8000
CHAIN = os.path.expanduser("~/.pyliftover/mm10ToHg38.over.chain.gz")
TILE = 500
BIN = 5000
RNG = np.random.default_rng(20261117)
LIFT_CACHE = os.path.join(OUT, "occ_lifted.csv")


def liftover_bins(bins):
    """mm10 5-kb bins -> hg38 windows, by lifting the bin centre.

    Requiring BOTH bin ends to fall inside aligned blocks drops the yield to ~20%
    because mouse-human intergenic sequence is poorly aligned at 5-kb scale. So the
    centre is lifted and the window is rebuilt as centre +/- 2.5 kb. Any edge error
    is a few hundred bp and is irrelevant to a 5-kb mean; crucially the same rule
    is applied to open and background bins, so the residual selection is shared.
    """
    import lib_chain
    pts = [(r.chrom, int(r.start) + BIN // 2) for r in bins.itertuples()]
    t0 = time.time()
    res = lib_chain.liftover_points(CHAIN, pts)
    print(f"    chain pass {time.time()-t0:.0f}s", flush=True)
    rows = []
    for r in res:
        if r is None:
            rows.append((np.nan, np.nan, np.nan))
        else:
            rows.append((r[0], max(0, r[1] - BIN // 2), r[1] + BIN // 2))
    return pd.DataFrame(rows, columns=["hg_chrom", "hg_start", "hg_end"],
                        index=bins.index)


def strata(gc, tss):
    q = np.clip((gc * 10).astype(int), 0, 9)
    return q * 2 + (tss > 0).astype(int)


def stratified_null(values, lab, strat, pool_mask, rng, n=N_PERM):
    """Null for the mean of `values` over a set with the same stratum make-up."""
    obs = np.nanmean(values[lab])
    s_lab = strat[lab]
    pool_idx = np.where(pool_mask)[0]
    by_s = {s: pool_idx[strat[pool_idx] == s] for s in np.unique(s_lab)}
    need = {s: int((s_lab == s).sum()) for s in np.unique(s_lab)}
    null = np.empty(n)
    for j in range(n):
        pick = []
        for s, kk in need.items():
            cand = by_s.get(s)
            if cand is None or len(cand) == 0:
                continue
            pick.append(rng.choice(cand, size=kk, replace=len(cand) < kk))
        if not pick:
            null[j] = np.nan
            continue
        null[j] = np.nanmean(values[np.concatenate(pick)])
    null = null[np.isfinite(null)]
    if not len(null):
        return obs, np.nan, np.nan, np.nan, np.nan
    p = (1 + (null >= obs).sum()) / (1 + len(null))
    z = (obs - null.mean()) / null.std(ddof=1) if null.std(ddof=1) > 0 else np.nan
    return obs, null.mean(), null.std(ddof=1), z, p


def auc(pos, neg):
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) /
                 (len(pos) * len(neg)))


def track_signal(path, sub, tag):
    """mean / summit / coverage for every lifted window, cached per track."""
    cache = os.path.join(OUT, f"occ_{tag}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        if len(z["mean"]) == len(sub):
            print(f"  {tag}: cache hit", flush=True)
            return z["mean"], z["summit"], z["cov"]
    bw = BigWig(path)
    t0 = time.time()
    mean = np.full(len(sub), np.nan)
    summit = np.full(len(sub), np.nan)
    cov = np.zeros(len(sub))
    hc = sub.hg_chrom.to_numpy()
    hs = sub.hg_start.to_numpy(float)
    he = sub.hg_end.to_numpy(float)
    for i in range(len(sub)):
        if not isinstance(hc[i], str) or not np.isfinite(hs[i]):
            continue
        m, s, c = bw.window_stats(hc[i], int(hs[i]), int(he[i]), TILE)
        mean[i], summit[i], cov[i] = m, s, c
    np.savez(cache, mean=mean, summit=summit, cov=cov)
    print(f"  {tag}: {np.isfinite(mean).sum()} covered, "
          f"median cov {np.median(cov):.2f}, {time.time()-t0:.0f}s", flush=True)
    return mean, summit, cov


def positive_control(n_random=1000, seed=11):
    """Does this statistic see the ligand at all, in the same two tracks?

    A null at SMC-open bins only means something if the identical statistic fires
    at canonical AHR targets. So the six textbook targets are read with the same
    window_mean / window_stats call, and their values are placed against the
    distribution of the same statistic over random refFlat TSS windows.
    """
    tcdd_p = os.path.join(EXT, "HCASMC_AHR_TCDD.bw")
    dmso_p = os.path.join(EXT, "HCASMC_AHR_DMSO.bw")
    if not (os.path.exists(tcdd_p) and os.path.exists(dmso_p)):
        return None
    rf = pd.read_csv(os.path.join(ROOT, "data", "external", "hg38_refFlat.txt.gz"),
                     sep="\t", header=None,
                     names=["gene", "name", "chrom", "strand", "txStart", "txEnd"] +
                           [f"x{i}" for i in range(8)])
    tss = rf.groupby("gene").agg(chrom=("chrom", "first"),
                                 pos=("txStart", "min")).reset_index()
    tss["chrom"] = tss.chrom.astype(str)
    tss = tss[tss.chrom.str.match(r"^chr(\d+|X)$")]
    tss["gene_up"] = tss.gene.astype(str).str.upper()

    canon = ["CYP1A1", "CYP1B1", "AHRR", "TIPARP", "NQO1", "AHR"]
    tgt = tss[tss.gene_up.isin(canon)].drop_duplicates("gene_up")
    rng = np.random.default_rng(seed)
    rand = tss.sample(min(n_random, len(tss)), random_state=seed)

    tw = BigWig(tcdd_p)
    dw = BigWig(dmso_p)

    def read(frame):
        out = []
        for r in frame.itertuples():
            s = int(r.pos) - BIN // 2
            tm, ts, _ = tw.window_stats(r.chrom, s, s + BIN, TILE)
            dm, ds, _ = dw.window_stats(r.chrom, s, s + BIN, TILE)
            out.append((np.nan_to_num(tm), np.nan_to_num(ts),
                        np.nan_to_num(dm), np.nan_to_num(ds)))
        a = np.array(out, float)
        return pd.DataFrame(dict(tcdd_mean=a[:, 0], tcdd_summit=a[:, 1],
                                 dmso_mean=a[:, 2], dmso_summit=a[:, 3]),
                            index=frame.index)

    null = read(rand)
    null["delta_mean"] = null.tcdd_mean - null.dmso_mean
    null["delta_summit"] = null.tcdd_summit - null.dmso_summit
    hits = read(tgt)
    hits["delta_mean"] = hits.tcdd_mean - hits.dmso_mean
    hits["delta_summit"] = hits.tcdd_summit - hits.dmso_summit

    rows = []
    for g, h in zip(tgt.gene_up.to_numpy(), hits.itertuples()):
        rec = dict(gene=g, delta_mean=h.delta_mean, delta_summit=h.delta_summit,
                   tcdd_summit=h.tcdd_summit)
        for col, v in [("tcdd_mean", h.tcdd_mean), ("tcdd_summit", h.tcdd_summit),
                       ("delta_mean", h.delta_mean), ("delta_summit", h.delta_summit)]:
            rec[f"pct_{col}"] = float((null[col] < v).mean() * 100)
        rows.append(rec)
    pc = pd.DataFrame(rows)
    pc.to_csv(os.path.join(OUT, "occupancy_positive_control.csv"), index=False)
    print("\n  positive control (percentile vs {} random TSS):".format(len(null)))
    print(pc.round(1).to_string(index=False), flush=True)
    return dict(n_random=int(len(null)), targets=pc.to_dict(orient="records"))


def main():
    bins = pd.read_csv(os.path.join(RES, "chip", "bin_ahr_overlap.csv"))
    gc = np.load(os.path.join(RES, "permutation", "bin_gc.npy"))
    tss = np.load(os.path.join(RES, "permutation", "bin_tss.npy"))
    assert len(bins) == len(gc) == len(tss)

    # which bins were testable at all in the ATAC contrast
    smc = pd.read_csv(os.path.join(RES, "motif", "da_bins_Parenchyma.csv"))
    ec = pd.read_csv(os.path.join(RES, "motif", "da_bins_Endothelial.csv"))
    # da_bins encodes the bin as chrom_index*1e9+start, so recover the start from
    # `end` instead of trusting bin_key
    key = bins.chrom + ":" + bins.start.astype("int64").astype(str)
    smc["start"] = (smc.end - BIN).astype("int64")
    ec["start"] = (ec.end - BIN).astype("int64")
    smc_key = set(smc.chrom + ":" + smc.start.astype(str))
    ec_key = set(ec.chrom + ":" + ec.start.astype(str))
    print(f"  keys: bins={len(key)} smc={len(smc_key)} ec={len(ec_key)} "
          f"overlap={len(set(key) & smc_key)}", flush=True)

    bins["testable"] = key.isin(smc_key)
    bins["smc_open"] = bins.testable & key.isin(
        set(smc[smc.sig_up].chrom + ":" + smc[smc.sig_up].start.astype(str)))
    bins["smc_closed"] = bins.testable & key.isin(
        set(smc[smc.sig_dn].chrom + ":" + smc[smc.sig_dn].start.astype(str)))
    bins["ec_open"] = bins.testable & key.isin(
        set(ec[ec.sig_up].chrom + ":" + ec[ec.sig_up].start.astype(str)))
    bins["gc"] = gc
    bins["tss"] = tss.astype(int)
    bins["strat"] = strata(gc, bins.tss.to_numpy())

    # every candidate bin is queried once; the sets are then tested independently
    # (lineage-open sets overlap, and de-duplicating them would silently shrink the
    # controls, which is exactly what happened in the first draft)
    ok = bins.testable & bins.gc.notna() & bins.chrom.notna() & bins.start.notna()
    bg_all = bins[ok & ~bins.smc_open & ~bins.ec_open & ~bins.smc_closed]
    pool = bg_all.sample(min(POOL_N, len(bg_all)), random_state=7)

    SETS = {
        "SMC_open": bins[bins.smc_open],
        "SMC_closed": bins[bins.smc_closed],
        "EC_open": bins[bins.ec_open],
        "background": pool,
    }
    cand = pd.concat([v.assign(_set=k) for k, v in SETS.items()])
    cand["bkey"] = cand.chrom + ":" + cand.start.astype("int64").astype(str)
    uniq = cand.drop_duplicates("bkey").reset_index(drop=True)
    print(f"  candidates: {len(cand)} rows -> {len(uniq)} unique bins", flush=True)
    mem = cand.groupby("bkey")._set.apply(frozenset)

    uniq = uniq[uniq.chrom.notna() & uniq.start.notna()].copy()
    uniq["chrom"] = uniq.chrom.astype(str)
    uniq["start"] = uniq.start.astype("int64")
    uniq = uniq.reset_index(drop=True)
    uniq["sets"] = uniq.bkey.map(mem)
    uniq["is_bg"] = uniq.sets.apply(lambda s: s == {"background"})
    uniq["is_smc_open"] = uniq.sets.apply(lambda s: "SMC_open" in s)

    if os.path.exists(LIFT_CACHE):
        hg = pd.read_csv(LIFT_CACHE)
        assert len(hg) == len(uniq), "lift cache does not match candidate table"
        print("  lift: cache hit", flush=True)
    else:
        print("  lifting to hg38 ...", flush=True)
        hg = liftover_bins(uniq[["chrom", "start"]])
        hg.to_csv(LIFT_CACHE, index=False)
    uniq = pd.concat([uniq.drop(columns=[c for c in hg.columns if c in uniq]),
                      hg], axis=1)
    print(f"  lifted {uniq.hg_chrom.notna().sum()}/{len(uniq)} "
          f"({uniq.hg_chrom.notna().mean():.1%})", flush=True)

    for tag, path in BW.items():
        if not os.path.exists(path):
            print(f"  SKIP {tag}: missing", flush=True)
            continue
        m, s, c = track_signal(path, uniq, tag)
        uniq[f"{tag}_mean"] = np.where(np.isfinite(m), m, 0.0)
        uniq[f"{tag}_summit"] = np.where(np.isfinite(s), s, 0.0)
        uniq[f"{tag}_cov"] = c
        uniq[f"{tag}_any"] = (c > 0).astype(int)

    if "HCASMC_TCDD_mean" in uniq and "HCASMC_DMSO_mean" in uniq:
        for stat in ["mean", "summit"]:
            a = uniq[f"HCASMC_TCDD_{stat}"].to_numpy(float)
            b = uniq[f"HCASMC_DMSO_{stat}"].to_numpy(float)
            uniq[f"ligand_delta_{stat}"] = a - b
            uniq[f"ligand_log2_{stat}"] = np.log2((a + 0.5) / (b + 0.5))

    keep = ["chrom", "start", "gc", "tss", "strat", "sets",
            "hg_chrom", "hg_start", "hg_end"] + \
           [c for c in uniq.columns if c.endswith(("_mean", "_summit", "_cov",
                                                   "_any", "_log2", "_delta"))]
    keep = list(dict.fromkeys(keep))
    out_bins = uniq[keep].copy()
    out_bins["sets"] = uniq.sets.apply(lambda s: "|".join(sorted(s)))
    out_bins.to_csv(os.path.join(OUT, "occupancy_bins.csv"), index=False)

    STATS = [f"HCASMC_{l}_{s}" for l in ("TCDD", "DMSO") for s in ("mean", "summit")] + \
            ["ligand_delta_mean", "ligand_log2_mean",
             "ligand_delta_summit", "ligand_log2_summit"] + \
            [f"{t}_{s}" for t in ("SKMEL2_AHROE", "SKMEL2_ctrl", "K562_CUTRUN")
             for s in ("mean", "summit")]
    STATS = [s for s in STATS if s in uniq.columns]

    rows = []
    for s in ["SMC_open", "SMC_closed", "EC_open"]:
        lab = uniq.sets.apply(lambda x, s=s: s in x).to_numpy()
        if lab.sum() < 10:
            print(f"  {s}: n={lab.sum()} too small, skipped", flush=True)
            continue
        pm = uniq.is_bg.to_numpy()
        for stat in STATS:
            v = uniq[stat].to_numpy(float)
            obs, nm, ns, z, p = stratified_null(v, lab, uniq.strat.to_numpy(), pm, RNG)
            rows.append(dict(set=s, statistic=stat, n=int(lab.sum()),
                             obs_mean=obs, null_mean=nm, null_sd=ns, z=z, p=p,
                             auc_vs_bg=auc(v[lab], v[pm])))
        # coverage balance: if the open bins map to better-mappable sequence the
        # zero-filled means are not comparable, so this has to be checked
        for stat in ["HCASMC_TCDD_cov", "HCASMC_DMSO_cov", "HCASMC_TCDD_any"]:
            if stat not in uniq.columns:
                continue
            v = uniq[stat].to_numpy(float)
            obs, nm, ns, z, p = stratified_null(v, lab, uniq.strat.to_numpy(), pm, RNG)
            rows.append(dict(set=s, statistic=stat + " [balance]", n=int(lab.sum()),
                             obs_mean=obs, null_mean=nm, null_sd=ns, z=z, p=p,
                             auc_vs_bg=auc(v[lab], v[pm])))
        print(f"  {s}: {int(lab.sum())} bins done", flush=True)

    # sensitivity: keep only bins where both HCASMC tracks actually cover >= 50%
    # of the window, so the zero-filling cannot be carrying the difference
    if {"HCASMC_TCDD_cov", "HCASMC_DMSO_cov"} <= set(uniq.columns):
        okc = ((uniq.HCASMC_TCDD_cov >= 0.5) &
               (uniq.HCASMC_DMSO_cov >= 0.5)).to_numpy()
        lab0 = uniq.sets.apply(lambda x: "SMC_open" in x).to_numpy()
        lab, pm = lab0 & okc, uniq.is_bg.to_numpy() & okc
        for stat in ["HCASMC_TCDD_summit", "HCASMC_DMSO_summit",
                     "ligand_delta_mean", "ligand_delta_summit"]:
            v = uniq[stat].to_numpy(float)
            obs, nm, ns, z, p = stratified_null(v, lab, uniq.strat.to_numpy(), pm, RNG)
            rows.append(dict(set="SMC_open[cov>=.5]", statistic=stat,
                             n=int(lab.sum()), obs_mean=obs, null_mean=nm,
                             null_sd=ns, z=z, p=p, auc_vs_bg=auc(v[lab], v[pm])))
        print(f"  SMC_open[cov>=.5]: {int(lab.sum())} bins done", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "occupancy_tests.csv"), index=False)

    # head-to-head on the same bins, two ligands -> paired ligand effect
    lab = uniq.is_smc_open.to_numpy()
    lig = dict(
        n=int(lab.sum()),
        mean_tcdd_gt_dmso=float(np.nanmean(uniq.loc[lab, "ligand_delta_mean"] > 0)),
        median_log2_mean=float(np.nanmedian(uniq.loc[lab, "ligand_log2_mean"])),
        summit_tcdd_gt_dmso=float(np.nanmean(uniq.loc[lab, "ligand_delta_summit"] > 0)),
        median_log2_summit=float(np.nanmedian(uniq.loc[lab, "ligand_log2_summit"])),
        summit_tcdd_median=float(np.nanmedian(uniq.loc[lab, "HCASMC_TCDD_summit"])),
        summit_dmso_median=float(np.nanmedian(uniq.loc[lab, "HCASMC_DMSO_summit"])),
    )
    summary = dict(
        note=("peak-set-free test of ring 5: HCASMC AHR ChIP signal read directly "
              "on the hg38 orthologues of mm10 5-kb bins; TCDD is the ligand arm, "
              "DMSO the internal control. Zero-filling is used for windows with no "
              "coverage; the *_cov / *_any balance rows show whether that choice "
              "could carry the result."),
        n_perm=N_PERM, pool=int(uniq.is_bg.sum()),
        lift_rate=float(uniq.hg_chrom.notna().mean()),
        sets={k: int(len(v)) for k, v in SETS.items()},
        ligand_paired=lig,
        tests=t.to_dict(orient="records"),
    )
    pc = positive_control()
    if pc:
        summary["positive_control"] = pc
    with open(os.path.join(OUT, "occupancy_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("\n" + t.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
