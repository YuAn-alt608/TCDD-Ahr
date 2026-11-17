"""30 - Matched-depth downsampling: do the four frozen endpoints survive when every
       library is cut to the Ahr-KO depth?

The Reviewer Question
---------------------
"The Ahr-KO ATAC library is 25% shallower than the WT libraries and one 16-wk
 transcriptome library has 8x fewer UMIs per cell. Your 'opening' and your
 'derepression' could be read depth."

Why not fragment files
----------------------
GSE276870 / GSE276872 release count matrices, not fragments or BAMs. Fragment-level
downsampling is therefore impossible without re-running CellRanger on 130 GB of
FASTQ, and the FASTQ route was already abandoned for throughput reasons
(scripts/TE_count_readme.md). What is done instead is *count-level* thinning:

    for each cell, each bin/UMI is kept with p = T*/n_i (a binomial draw)

which is the exact sampling model of multinomial read downsampling - the same
statement as "throw away reads until every cell has T*", minus the multi-mapping
detail that no count matrix can preserve. This is stated in the report as such; it
is not called fragment downsampling.

What is measured
----------------
Each library is thinned to a common budget, then the *same code path* that produced
the published numbers recomputes:

  ATAC  frac_het       sum(counts in H3K9me3 domains) / total counts
        prom_het_rel   mean log1p(gene activity of constitutive-het genes)
                       - mean log1p(gene activity of all genes)
  RNA   het_derepression / LAM   per-mouse pseudobulk CPM -> log1p -> module mean

and the animal-level effect is rebuilt with the frozen standardiser from
scripts/25_power_n2.py (RNA: pooled between-animal SD of the two replicated arms;
ATAC: |c1-c2|/sqrt(2) x 1.5 because the control SD has df = 1).

Two targets are used: T* = the Ahr-KO median (the question as asked) and
T* = 0.6 x that (a stress test, in case the KO library is itself the artefact).
Three seeds per target, so "no change" cannot be one lucky draw.

Outputs (results/audit/):
  depth_endpoints.csv      per seed x target x endpoint: delta, d, direction
  depth_summary.csv        per endpoint: d at full depth, mean d at T*, attenuation
  depth_summary.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

OUT = os.path.join(L.RES, "audit")
os.makedirs(OUT, exist_ok=True)
GENOME = os.path.join(L.ROOT, "data", "genome", "mm10.2bit")
PFM = os.path.join(os.path.dirname(L.ANN), "JASPAR2024_vertebrates.pfm")
CANON = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY", "chrM"]
BIN = 5000
ATAC_INFLATE = 1.5
SEEDS = [0, 1, 2]
LAM = ["Trem2", "Lgals3", "Csf1r", "Spp1", "Apoe", "Cd9", "Fabp5", "Lpl", "Gpnmb", "Trem1"]
ATAC_LIB = {"D2_atac_CTRL1": "CTRL1", "D2_atac_CTRL2": "CTRL2",
            "D2_atac_WT_TCDD": "WT16", "D2_atac_AHRKO_TCDD": "AKO16"}


# ------------------------------------------------------------------ primitives
def thin(X, target, rng):
    """Binomial thinning of every entry: the count-level equivalent of dropping
    reads until each cell holds `target` fragments."""
    tot = np.asarray(X.sum(1)).ravel()
    p = np.minimum(1.0, target / np.maximum(tot, 1))
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    Y = X.copy()
    Y.data = rng.binomial(Y.data.astype(np.int64), p[rows]).astype(np.float32)
    Y.eliminate_zeros()
    return Y


def in_domains(keys, bed_path):
    key_chrom = (keys // 1_000_000_000).astype(int)
    key_start = keys % 1_000_000_000
    b = pd.read_csv(bed_path, sep="\t", header=None, names=["chrom", "start", "end"])
    cid_map = {c: i for i, c in enumerate(CANON)}
    b["cid"] = b.chrom.map(lambda c: cid_map.get(c, -1))
    mask = np.zeros(len(keys), dtype=bool)
    for ci, sub in b.groupby("cid", observed=True):
        if ci < 0:
            continue
        idx = np.where(key_chrom == ci)[0]
        if len(idx) == 0:
            continue
        s, e = sub.start.to_numpy(), sub.end.to_numpy()
        o = np.argsort(s)
        s, e = s[o], np.maximum.accumulate(e[o])
        en = key_start[idx] + BIN
        pos = np.searchsorted(s, en, side="right") - 1
        ok = pos >= 0
        hit = np.zeros(len(idx), dtype=bool)
        hit[ok] = e[pos[ok]] > key_start[idx][ok]
        mask[idx] = hit
    return mask


def cohend_sigma(vals, assay):
    """Frozen standardiser from 25_power_n2.py."""
    c = np.asarray(vals.get("Control", []), float)
    if assay == "ATAC":
        if len(c) < 2:
            return np.nan
        return float(abs(c[0] - c[1]) / np.sqrt(2) * ATAC_INFLATE)
    w = np.asarray(vals.get("TCDD_8wk", []), float)
    if len(c) < 2 or len(w) < 2:
        return np.nan
    return float(np.sqrt(((len(c) - 1) * c.var(ddof=1) + (len(w) - 1) * w.var(ddof=1))
                         / (len(c) + len(w) - 2)))


# ------------------------------------------------------------------ ATAC
def atac_block():
    a = ad.read_h5ad(os.path.join(L.RES, "ATAC", "atac_processed.h5ad"))
    keys = np.asarray(a.var_names, dtype=np.int64)
    X = a.X.tocsr().astype(np.float32)
    lib = a.obs["sample"].astype(str).to_numpy()
    mouse = np.array([ATAC_LIB.get(s, s) for s in lib])
    lineage = a.obs["lineage"].astype(str).to_numpy()
    depth = pd.Series(np.asarray(X.sum(1)).ravel()).groupby(lib).median()
    print("ATAC median counts/cell per library:\n" + depth.round(0).to_string(), flush=True)

    m_het = in_domains(keys, os.path.join(L.ANN, "domains_H3K9me3.bed"))

    # gene activity map, reused from 18_atac_downsample.py
    cache = os.path.join(L.RES, "robust", "act_map.npz")
    z = np.load(cache, allow_pickle=True)
    p2g = z["p2g"]
    gnames = list(z["genes"])
    het_genes = sorted(set(L.load_set("ConstitutiveHet")) & set(gnames))
    hidx = [gnames.index(g) for g in het_genes]
    n_keep = (p2g >= 0).sum()
    G = sp.csr_matrix((np.ones(n_keep, dtype=np.float32),
                       (p2g[p2g >= 0], np.where(p2g >= 0)[0])),
                      shape=(len(gnames), len(keys)))
    Gh = G[hidx]

    def prom_metrics(Xd):
        """Exactly 05_atac.py's definition.

        act  = X @ (bins within +-5 kb of each TSS)
        tot  = act.sum(1)                      <- normalise by gene activity, NOT by
        act  = log1p(act / tot * 5000)            raw counts; this is the detail that
        prom_het_rel = mean(het genes) - mean(all genes)
        Two chunked passes so the 16000 x 22000 activity matrix is never resident.
        """
        step = 2000
        tot = np.zeros(Xd.shape[0], dtype=np.float64)
        for s0 in range(0, len(gnames), step):
            blk = np.asarray((Xd @ G[s0:s0 + step].T).todense(), dtype=np.float32)
            tot += blk.sum(1)
        act_h = np.asarray((Xd @ Gh.T).todense(), dtype=np.float32)
        prom_het = np.log1p(act_h / np.maximum(tot, 1)[:, None] * 5000.0).mean(1)
        acc = np.zeros(Xd.shape[0], dtype=np.float64)
        for s0 in range(0, len(gnames), step):
            blk = np.asarray((Xd @ G[s0:s0 + step].T).todense(), dtype=np.float32)
            acc += np.log1p(blk / np.maximum(tot, 1)[:, None] * 5000.0).sum(1)
        return prom_het - acc / len(gnames)

    def metrics(Xd):
        tot = np.asarray(Xd.sum(1)).ravel()
        return {"frac_het": np.asarray(Xd[:, m_het].sum(1)).ravel() / np.maximum(tot, 1),
                "prom_het_rel": prom_metrics(Xd)}

    return a, X, mouse, lineage, lib, depth, metrics


# ------------------------------------------------------------------ RNA
def rna_block():
    a = ad.read_h5ad(os.path.join(L.RES, "D2", "processed.h5ad"))
    cnt = sp.csr_matrix(a.layers["counts"]).astype(np.float32)
    obs = a.obs.copy()
    obs["mouse"] = obs["sample"].map(
        lambda s: str(s).split("_")[2] + ("" if str(s).split("_")[2].endswith("16")
                                          else str(s).split("_")[3].replace("TDT", "").replace("NTDT", "")))
    depth = pd.Series(np.asarray(cnt.sum(1)).ravel()).groupby(
        obs["sample"].astype(str)).median()
    print("\nRNA median UMI/cell per library:\n" + depth.round(0).to_string(), flush=True)
    return a, cnt, obs, depth


def module_pseudobulk(cnt, obs, gi, mask):
    """Per-mouse module score, two ways.

    log1p   : mean over genes of log1p(CPM) - the estimator behind every published
              number. Concave, so Poisson noise from a shallow library pushes it
              DOWN (Jensen). That is a real bias, not a bug, and it is the reason
              this script also reports:
    share   : sum(CPM over genes) / n_genes, i.e. the module's share of all counts.
              Linear in the counts, so multinomial thinning leaves it unbiased -
              depth-invariant by construction. If the direction survives here, the
              claim cannot be a depth artefact.
    """
    key = obs["mouse"].astype(str).to_numpy()[mask]
    uk, inv = np.unique(key, return_inverse=True)
    ind = sp.csr_matrix((np.ones(len(inv), np.float32), (inv, np.arange(len(inv)))),
                        shape=(len(uk), len(inv)))
    sub = cnt[mask]
    pb = np.asarray((ind @ sub[:, gi]).todense(), float)
    lib = np.asarray((ind @ sub).todense(), float).sum(1, keepdims=True)
    cpm = pb / np.maximum(lib, 1) * 1e6
    return pd.DataFrame({"log1p": np.log1p(cpm).mean(1),
                         "share": cpm.sum(1) / cpm.shape[1]}, index=uk)


def main():
    rows = []
    t0 = time.time()

    # ============================================================ ATAC
    a, X, mouse, lineage, lib, depth, metrics = atac_block()
    ko_t = float(depth.get("D2_atac_AHRKO_TCDD", np.nan))
    targets = {"full": None, "match_ako": ko_t, "stress_60pct": 0.6 * ko_t}
    print(f"\nATAC targets: {targets}", flush=True)

    a_obs = a.obs
    SIG = {}          # full-depth sigma, reused for every thinned target
    for tname, tval in targets.items():
        for seed in (SEEDS if tval is not None else [None]):
            Xd = X if tval is None else thin(X, tval, np.random.default_rng(seed))
            met = metrics(Xd)
            for metric, vec in met.items():
                df = pd.DataFrame({"mouse": mouse, "group": a_obs["group"].astype(str),
                                   "lineage": lineage, "v": vec})
                per = df.groupby(["mouse", "group", "lineage"]).v.mean().reset_index()
                for ln, sub in per.groupby("lineage"):
                    for contrast in [("TCDD_16wk_AhrKO", "Control"),
                                     ("TCDD_16wk_WT", "Control")]:
                        alt, ref = contrast
                        va = sub[sub.group == alt].v.to_numpy()
                        vr = sub[sub.group == ref].v.to_numpy()
                        if len(va) == 0 or len(vr) == 0:
                            continue
                        vals = {"Control": vr, "TCDD_8wk": np.array([])}
                        sig = cohend_sigma(vals, "ATAC")
                        # hold sigma at its full-depth value so the attenuation
                        # column isolates the change in delta, not a re-estimate
                        key = ("ATAC", ln, metric, contrast)
                        if tname == "full":
                            SIG[key] = sig
                        else:
                            sig = SIG.get(key, sig)
                        delta = float(va.mean() - vr.mean())
                        rows.append(dict(
                            assay="ATAC", endpoint=metric, statistic="per_cell",
                            celltype=ln, contrast=f"{alt} vs {ref}",
                            target=tname, target_depth=(None if tval is None else round(tval, 1)),
                            seed=(-1 if seed is None else seed), delta=delta, sigma=sig,
                            d=delta / sig if np.isfinite(sig) and sig > 0 else np.nan,
                            median_depth=float(np.median(np.asarray(Xd.sum(1)).ravel()
                                                         [lib == "D2_atac_AHRKO_TCDD"]))))
            print(f"  ATAC {tname} seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    # ============================================================ RNA
    rna_a, cnt, obs, rdepth = rna_block()
    SIG = {}          # same cache for the RNA half
    genes = np.asarray(rna_a.var_names)
    gidx = {g: i for i, g in enumerate(genes)}
    sets = {"het_derepression": [gidx[g] for g in sorted(L.load_set("ConstitutiveHet"))
                                 if g in gidx],
            "LAM": [gidx[g] for g in LAM if g in gidx]}
    for ct in ["Endothelial", "Macrophage"]:
        m = (obs["celltype"].astype(str) == ct).to_numpy()
        sub_obs = obs[m]
        tot = np.asarray(cnt[m].sum(1)).ravel()
        ko_m = (sub_obs["group"] == "TCDD_16wk_AhrKO").to_numpy()
        ko_t = float(np.median(tot[ko_m]))
        tlist = {"full": None, "match_ako": ko_t, "stress_min": float(np.min(tot))}
        for mod, gi in sets.items():
            if not gi:
                continue
            for tname, tval in tlist.items():
                for seed in (SEEDS if tval is not None else [None]):
                    if tval is None:
                        cc = cnt[m]
                    else:
                        rng = np.random.default_rng(1000 + seed)
                        cc = thin(cnt[m], tval, rng)
                    tab = module_pseudobulk(cc, sub_obs, gi, np.ones(int(m.sum()), bool))
                    gmap = {k2: sub_obs.group[sub_obs.mouse == k2].iloc[0] for k2 in tab.index}

                    def vals_of(g, stat):
                        return np.array([tab[stat][k2] for k2 in tab.index
                                         if gmap[k2] == g], float)
                    for stat in ["log1p", "share"]:
                        for alt in ["TCDD_16wk_AhrKO", "TCDD_16wk_WT"]:
                            va = vals_of(alt, stat); vr = vals_of("Control", stat)
                            if len(va) == 0:
                                continue
                            sig = cohend_sigma({"Control": vr,
                                                "TCDD_8wk": vals_of("TCDD_8wk", stat)}, "RNA")
                            key = ("RNA", ct, mod, stat, alt)
                            if tname == "full":
                                SIG[key] = sig
                            else:
                                sig = SIG.get(key, sig)
                            delta = float(va.mean() - vr.mean())
                            rows.append(dict(
                                assay="RNA", endpoint=mod, statistic=stat, celltype=ct,
                                contrast=f"{alt} vs Control", target=tname,
                                target_depth=(None if tval is None else round(tval, 1)),
                                seed=(-1 if seed is None else seed), delta=delta, sigma=sig,
                                d=delta / sig if np.isfinite(sig) and sig > 0 else np.nan,
                                median_depth=float(np.median(np.asarray(cc.sum(1)).ravel()))))
        print(f"  RNA {ct} done ({time.time()-t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "depth_endpoints.csv"), index=False)

    # ------------------------------------------------------------- summary
    summ = []
    for (assay, ct, ep, stat, contrast), sub in res.groupby(
            ["assay", "celltype", "endpoint", "statistic", "contrast"]):
        full = sub[sub.target == "full"]
        if len(full) == 0:
            continue
        d0 = float(full.d.iloc[0]); dl0 = float(full.delta.iloc[0])
        for tname in ["match_ako", "stress_60pct", "stress_min"]:
            s = sub[sub.target == tname]
            if len(s) == 0:
                continue
            same = int((np.sign(s.d) == np.sign(d0)).sum())
            summ.append(dict(assay=assay, celltype=ct, endpoint=ep, statistic=stat,
                             contrast=contrast,
                             target=tname, d_full=round(d0, 4), delta_full=round(dl0, 6),
                             d_mean=round(float(s.d.mean()), 4),
                             d_sd=round(float(s.d.std(ddof=1)), 4) if len(s) > 1 else np.nan,
                             delta_mean=round(float(s.delta.mean()), 6),
                             sign_match=f"{same}/{len(s)}",
                             attenuation_pct=round(100 * (1 - abs(float(s.delta.mean()) /
                                                                dl0)) if dl0 else np.nan, 1),
                             depth_of_target=float(s.target_depth.iloc[0])))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(OUT, "depth_summary.csv"), index=False)
    print("\n=== depth summary ===", flush=True)
    print(summ.to_string(index=False), flush=True)

    prim = {("ATAC", "Parenchyma", "frac_het"), ("ATAC", "Endothelial", "prom_het_rel"),
            ("RNA", "Endothelial", "het_derepression"), ("RNA", "Macrophage", "LAM")}
    ps = summ[[(r.assay, r.celltype, r.endpoint) in prim for r in summ.itertuples()]]
    ok = int(sum(int(r.sign_match.split("/")[0]) == int(r.sign_match.split("/")[1])
                 for r in ps[ps.target == "match_ako"].itertuples()))
    sm = {
        "n_frozen_endpoints": len(ps[ps.target == "match_ako"]),
        "direction_preserved_at_ako_depth": f"{ok}/{len(ps[ps.target == 'match_ako'])}",
        "max_abs_attenuation_pct_match_ako": (None if not len(ps[ps.target == "match_ako"])
                                              else float(ps[ps.target == "match_ako"]
                                                         .attenuation_pct.abs().max())),
        "method": ("binomial thinning of the count matrix per cell; equivalent to multinomial "
                   "read downsampling. Fragments/BAMs are not released for GSE276870/276872, so "
                   "this is NOT fragment-level downsampling and the report says so."),
        "note": ("ATAC sigma is |c1-c2|/sqrt(2)*1.5 (df=1) and is held fixed across targets, so "
                 "the attenuation column reflects delta only. RNA sigma is the pooled "
                 "between-animal SD of the Control and 8-week arms, also held fixed."),
    }
    json.dump(sm, open(os.path.join(OUT, "depth_summary.json"), "w"), indent=2, default=str)
    print("\n" + json.dumps(sm, indent=2, default=str), flush=True)
    print("done")


if __name__ == "__main__":
    main()
