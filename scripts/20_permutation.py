"""20 - Is the heterochromatin-domain effect an artefact of how domains are defined?

Three perturbations, all on the existing 16,000-cell ATAC matrix:

  (1) Annotated-domain permutation (1000x). A random bin set cannot be compared
      to H3K9me3 bins directly: H3K9me3 bins are GC-poor and TSS-poor, and both
      properties shift accessibility on their own. So each permutation draws a
      same-size alternative set that is matched on GC (within 1 pp) and on local
      TSS density (same quartile of genes within +-50 kb), excluding every
      annotated heterochromatin / LAD / pericentromeric bin.
  (2) Threshold sensitivity. H3K9me3 domains are rebuilt at three stringencies:
      top 10% / top 20% / all peaks by ENCODE signal, plus the LAD proxy. The
      per-lineage effect has to survive all of them to be believable.
  (3) Euchromatin negative control. The same statistic computed on the matched
      euchromatic sets must be ~0: a domain-specific effect that also appears in
      GC/TSS-matched euchromatin is a composition artefact.

Outputs (results/permutation/):
  bin_gc.npy, bin_tss.npy        per-bin covariates (cached)
  domain_definitions.csv         bins and bp per definition
  threshold_sensitivity.csv      d per lineage x definition
  permutation_null.csv.gz        null d for every permutation x lineage
  permutation_test.csv           observed vs null, empirical p, z
  summary.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L
from lib_seq import TwoBit

OUT = os.path.join(L.RES, "permutation")
os.makedirs(OUT, exist_ok=True)
GENOME = os.path.join(L.ROOT, "data", "genome", "mm10.2bit")
CANON = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"]
BIN = 5000
N_PERM = 1000
DEBUG = os.environ.get("PERM_DEBUG") == "1"
LINEAGES = ["Parenchyma", "Endothelial", "Mesenchymal", "Immune"]


# --------------------------------------------------------------- covariates
def bin_gc(tb, keys):
    """GC fraction per 5 kb bin, decoded straight from the packed 2bit bytes."""
    code = (keys // 1_000_000_000).astype(int)
    start = keys % 1_000_000_000
    gc = np.full(len(keys), np.nan)
    for ci, chrom in enumerate(CANON):
        idx = np.where(code == ci)[0]
        if len(idx) == 0 or chrom not in tb.idx:
            continue
        dna, (nst, nsz), off = tb._record(chrom)
        tb.f.seek(off)
        buf = np.frombuffer(tb.f.read(dna // 4 + 1), dtype=np.uint8)
        b = np.empty(len(buf) * 4, dtype=np.uint8)
        b[0::4] = (buf >> 6) & 3
        b[1::4] = (buf >> 4) & 3
        b[2::4] = (buf >> 2) & 3
        b[3::4] = buf & 3
        isgc = ((b == 1) | (b == 3)).astype(np.int32)   # 1=C, 3=G in UCSC 2bit
        cs = np.concatenate([[0], np.cumsum(isgc, dtype=np.int64)])
        st = start[idx]
        en = np.minimum(st + BIN, dna)
        gc[idx] = (cs[en] - cs[st]) / np.maximum(en - st, 1)
        del buf, b, isgc, cs
    return gc


def bin_tss(keys):
    tss = pd.read_csv(os.path.join(L.ANN, "mm10_genes_tss.tsv"), sep="\t")
    code = (keys // 1_000_000_000).astype(int)
    start = keys % 1_000_000_000
    out = np.zeros(len(keys), dtype=np.int32)
    for ci, chrom in enumerate(CANON):
        idx = np.where(code == ci)[0]
        t = tss[tss.chrom == chrom].tss.to_numpy()
        if len(idx) == 0 or len(t) == 0:
            continue
        t = np.sort(t)
        lo = np.searchsorted(t, start[idx] - 50_000, "left")
        hi = np.searchsorted(t, start[idx] + 50_000, "right")
        out[idx] = hi - lo
    return out


def load_mask(keys, bed):
    code = (keys // 1_000_000_000).astype(int)
    start = keys % 1_000_000_000
    b = pd.read_csv(bed, sep="\t", header=None, names=["chrom", "start", "end"])
    order = {c: i for i, c in enumerate(CANON)}
    b["cid"] = b.chrom.map(lambda c: order.get(c, -1))
    mask = np.zeros(len(keys), dtype=bool)
    for ci, sub in b.groupby("cid", observed=True):
        if ci < 0:
            continue
        idx = np.where(code == ci)[0]
        if len(idx) == 0:
            continue
        s = sub.start.to_numpy(); e = sub.end.to_numpy()
        o = np.argsort(s); s = s[o]
        e = np.maximum.accumulate(e[o])
        en = start[idx] + BIN
        pos = np.searchsorted(s, en, "right") - 1
        ok = pos >= 0
        hit = np.zeros(len(idx), dtype=bool)
        hit[ok] = e[pos[ok]] > start[idx][ok]
        mask[idx] = hit
    return mask


def peaks_to_mask(keys, peaks):
    """peaks: DataFrame chrom/start/end (already filtered by stringency)."""
    tmp = os.path.join(OUT, "_tmp_peaks.bed")
    peaks[["chrom", "start", "end"]].to_csv(tmp, sep="\t", index=False, header=False)
    m = load_mask(keys, tmp)
    os.remove(tmp)
    return m


def main():
    t0 = time.time()
    a = ad.read_h5ad(os.path.join(L.RES, "ATAC", "atac_processed.h5ad"))
    keys = np.asarray(a.var_names, dtype=np.int64)
    X = a.X.tocsc()
    tot = np.asarray(X.sum(1)).ravel()
    grp = a.obs["group"].astype(str).to_numpy()
    lin = a.obs["lineage"].astype(str).to_numpy()
    print(f"matrix {a.shape} nnz={X.nnz}", flush=True)

    gcf = os.path.join(OUT, "bin_gc.npy")
    tsf = os.path.join(OUT, "bin_tss.npy")
    if os.path.exists(gcf) and os.path.exists(tsf):
        gc = np.load(gcf); tss = np.load(tsf)
    else:
        tb = TwoBit(GENOME)
        gc = bin_gc(tb, keys)
        print(f"GC done ({np.isfinite(gc).mean()*100:.1f}% of bins) {time.time()-t0:.0f}s", flush=True)
        tss = bin_tss(keys)
        np.save(gcf, gc); np.save(tsf, tss)
        print("covariates cached", flush=True)

    # ---------------------------------------------------------- definitions
    m_het = load_mask(keys, os.path.join(L.ANN, "domains_H3K9me3.bed"))
    m_lad = load_mask(keys, os.path.join(L.ANN, "domains_LADproxy.bed"))
    m_pc = load_mask(keys, os.path.join(L.ANN, "domains_pericentromeric.bed"))

    pk = []
    for f in ["H3K9me3_liver.bed.gz", "H3K9me3_heart.bed.gz"]:
        p = os.path.join(os.path.dirname(L.ANN), f)
        d = pd.read_csv(p, sep="\t", header=None, usecols=[0, 1, 2, 4],
                        names=["chrom", "start", "end", "score"], dtype={"chrom": str})
        d = d[d.chrom.isin(CANON)]
        pk.append(d)
    pk = pd.concat(pk, ignore_index=True)
    defs = {"H3K9me3_all": m_het}
    for q, tag in [(0.90, "top10"), (0.80, "top20")]:
        thr = pk.score.quantile(q)
        defs[f"H3K9me3_{tag}"] = peaks_to_mask(keys, pk[pk.score >= thr])
    defs["LAD_proxy"] = m_lad

    # ---------------------------------------------------------- cell fractions
    def frac(mask):
        return np.asarray(X[:, np.where(mask)[0]].sum(1)).ravel() / np.maximum(tot, 1)

    obs = {k: frac(v) for k, v in defs.items()}
    drow = []
    for k, v in defs.items():
        drow.append(dict(definition=k, n_bins=int(v.sum()), bp=int(v.sum() * BIN),
                         mean_ctrl=float(obs[k][grp == "Control"].mean()),
                         mean_wt=float(obs[k][grp == "TCDD_16wk_WT"].mean()),
                         mean_ako=float(obs[k][grp == "TCDD_16wk_AhrKO"].mean())))
    pd.DataFrame(drow).to_csv(os.path.join(OUT, "domain_definitions.csv"), index=False)

    # ---------------------------------------------------------- threshold table
    trows = []
    for k in defs:
        for ln in LINEAGES:
            m = lin == ln
            c = obs[k][m & (grp == "Control")]
            w = obs[k][m & (grp == "TCDD_16wk_WT")]
            ko = obs[k][m & (grp == "TCDD_16wk_AhrKO")]
            trows.append(dict(definition=k, lineage=ln, n_ctrl=len(c), n_wt=len(w),
                              d_wt_vs_ctrl=L.cohens_d(w, c),
                              p_wt_vs_ctrl=L.perm_p(w, c, n=1000, seed=7),
                              d_ako_vs_wt=L.cohens_d(ko, w),
                              delta_ako=float(ko.mean() - w.mean())))
    tdf = pd.DataFrame(trows)
    tdf.to_csv(os.path.join(OUT, "threshold_sensitivity.csv"), index=False)
    print("\nthreshold sensitivity:\n" + tdf.round(3).to_string(index=False), flush=True)

    # ---------------------------------------------------------- matched null pool
    domain_any = m_het | m_lad | m_pc
    pool = np.where(~domain_any & np.isfinite(gc))[0]
    gbin = np.round(gc * 100).astype(int)                     # 1 pp bins
    # TSS-density quartiles are defined on the pool but applied to every bin, so a
    # target bin can be matched even though it is excluded from the pool itself
    edges = np.quantile(tss[pool], [0.25, 0.50, 0.75])
    pool_q = np.digitize(tss, edges)      # 0..3 for all bins
    print(f"null pool: {len(pool)} bins", flush=True)

    # frequency-matched sampling: one draw per (GC bin x TSS quartile) stratum
    pool_key = np.array([f"{g}_{q}" for g, q in zip(gbin[pool], pool_q[pool])])
    pool_by = pd.Series(np.arange(len(pool)), index=pool_key).groupby(level=0).apply(
        lambda x: x.to_numpy())

    def matched_set(target_idx, rng):
        tk = np.array([f"{g}_{q}" for g, q in zip(gbin[target_idx], pool_q[target_idx])])
        want = pd.Series(tk).value_counts()
        out = []
        for k, n in want.items():
            cand = pool_by.get(k)
            if cand is None or len(cand) == 0:
                continue
            take = min(int(n), len(cand))
            out.append(rng.choice(cand, size=take, replace=False))
        if not out:
            return np.array([], dtype=int)
        return pool[np.unique(np.concatenate(out))]

    het_idx = np.where(m_het)[0]
    rng = np.random.default_rng(0)
    nulls, prow = [], []
    for i in range(N_PERM):
        idx = matched_set(het_idx, rng)
        if i == 0:
            print(f"    first draw: {len(idx)} bins, mean GC {np.nanmean(gc[idx]):.3f}")
        v = np.asarray(X[:, idx].sum(1)).ravel() / np.maximum(tot, 1)
        rec = {"perm": i, "n_bins": len(idx), "gc_mean": float(np.nanmean(gc[idx])),
               "tss_mean": float(np.mean(tss[idx]))}
        for ln in LINEAGES:
            m = lin == ln
            c = v[m & (grp == "Control")]
            w = v[m & (grp == "TCDD_16wk_WT")]
            rec[ln] = L.cohens_d(w, c)
        nulls.append(rec)
        if i % 200 == 0:
            print(f"  perm {i}/{N_PERM} ({time.time()-t0:.0f}s)", flush=True)
    ndf = pd.DataFrame(nulls)
    ndf.to_csv(os.path.join(OUT, "permutation_null.csv.gz"), index=False, compression="gzip")

    for ln in LINEAGES:
        m = lin == ln
        c = obs["H3K9me3_all"][m & (grp == "Control")]
        w = obs["H3K9me3_all"][m & (grp == "TCDD_16wk_WT")]
        d_obs = L.cohens_d(w, c)
        nl = ndf[ln].dropna()
        p_emp = (1 + (nl >= d_obs).sum()) / (1 + len(nl))
        prow.append(dict(lineage=ln, d_observed=d_obs,
                         null_mean=float(nl.mean()), null_sd=float(nl.std(ddof=1)),
                         z=float((d_obs - nl.mean()) / (nl.std(ddof=1) + 1e-9)),
                         p_one_sided=p_emp, n_perm=len(nl),
                         null_q95=float(nl.quantile(0.95)),
                         null_q05=float(nl.quantile(0.05)),
                         gc_target=float(np.nanmean(gc[het_idx])),
                         gc_null=float(np.nanmean(ndf.gc_mean))))
    pdf = pd.DataFrame(prow)
    pdf.to_csv(os.path.join(OUT, "permutation_test.csv"), index=False)
    print("\npermutation test (annotated H3K9me3 vs GC/TSS-matched euchromatin):\n"
          + pdf.round(3).to_string(index=False), flush=True)

    sm = {"n_perm": N_PERM, "n_het_bins": int(m_het.sum()),
          "null_pool_bins": int(len(pool)),
          "definitions": {k: int(v.sum()) for k, v in defs.items()},
          "null_gc_match": {"target": float(np.nanmean(gc[het_idx])),
                            "null_mean": float(ndf.gc_mean.mean())},
          "elapsed_s": round(time.time() - t0, 1)}
    json.dump(sm, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps(sm, indent=2))
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
