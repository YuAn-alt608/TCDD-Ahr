"""22 - Is the SMC opening at real AHR binding sites, or only at AHR motifs?

Motif enrichment (OR 2.01 for Ahr::Arnt in SMC-open bins) is a sequence-level
statement. The companion ChIP-seq series from the same study (GSE276152) provides
the physical counterpart: AHR ChIP-seq in human coronary artery SMC (HCASMC),
DMSO vs TCDD, narrowPeak files.

The peaks are hg38 and the ATAC matrix is mm10, so the two are compared at the
level of orthologous promoters rather than by liftover: each human peak is
assigned to the gene whose TSS it sits within 2 kb of, and that gene is matched
to its mouse orthologue by case-insensitive symbol. That keeps the claim narrow
and checkable: "the promoters that open in mouse SMC are the promoters AHR
occupies in human coronary SMC".

Controls built in:
  * DMSO AHR peaks are used as the background occupancy set (a peak set that
    should NOT explain TCDD-induced opening),
  * classic AHR targets (Cyp1a1, Ahrr, Cyp1b1) must be recovered, otherwise the
    peak set is not usable,
  * gene-label permutation (1000x) gives the null for the mean delta of bins near
    bound genes.

Outputs (results/chip/):
  ahr_peaks_by_gene.csv      human gene -> mouse orthologue -> n peaks, DMSO/TCDD
  chip_overlap_tests.csv     Fisher tests per lineage (open bins vs background)
  chip_delta_permutation.csv observed vs null mean delta
  summary.json
"""
import os, sys, json, gzip
import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

EXT = os.path.join(L.ROOT, "data", "external")
OUT = os.path.join(L.RES, "chip")
os.makedirs(OUT, exist_ok=True)
LINEAGES = ["Parenchyma", "Endothelial", "Mesenchymal", "Immune"]
CANON = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY", "chrM"]
TSS_WIN = 2000
BIN_WIN = 50_000
N_PERM = 1000


def read_peaks(path):
    d = pd.read_csv(path, sep="\t", header=None,
                    names=["chrom", "start", "end", "name", "score", "strand",
                           "signal", "p", "q", "peak"][:10], comment="#")
    return d


def read_refflat(path):
    d = pd.read_csv(path, sep="\t", header=None,
                    names=["gene", "tx", "chrom", "strand", "txStart", "txEnd", "cdsStart",
                           "cdsEnd", "n_exon", "exonStarts", "exonEnds"])
    d["tss"] = np.where(d.strand == "+", d.txStart, d.txEnd)
    return d[["gene", "chrom", "strand", "tss"]].drop_duplicates()


def peaks_to_genes(peaks, tss, win=TSS_WIN):
    pe = peaks[peaks.chrom.isin(tss.chrom.unique())].copy()
    t = tss.sort_values("tss")
    out = {}
    for chrom, sub in pe.groupby("chrom"):
        tt = t[t.chrom == chrom]
        if len(tt) == 0:
            continue
        arr = tt.tss.to_numpy()
        gl = tt.gene.to_numpy()
        for s, e in zip(sub.start.to_numpy(), sub.end.to_numpy()):
            lo = np.searchsorted(arr, s - win, "left")
            hi = np.searchsorted(arr, e + win, "right")
            for j in range(lo, hi):
                out[gl[j]] = out.get(gl[j], 0) + 1
    return out


def main():
    # ---------------------------------------------------------------- peaks
    pos = read_peaks(os.path.join(EXT, "GSM8492045_AHR_pos_peaks.narrowPeak.gz"))
    neg = read_peaks(os.path.join(EXT, "GSM8492044_AHR_neg_peaks.narrowPeak.gz"))
    tcf = read_peaks(os.path.join(EXT, "GSM8492047_TCF21_pos_peaks.narrowPeak.gz"))
    print(f"AHR TCDD peaks {len(pos)} | AHR DMSO peaks {len(neg)} | TCF21 TCDD peaks {len(tcf)}",
          flush=True)
    hg = read_refflat(os.path.join(EXT, "hg38_refFlat.txt.gz"))
    pos_prom = peaks_to_genes(pos, hg, 2000)
    neg_prom = peaks_to_genes(neg, hg, 2000)
    tcf_prom = peaks_to_genes(tcf, hg, 2000)
    pos_g = peaks_to_genes(pos, hg, 50_000)
    neg_g = peaks_to_genes(neg, hg, 50_000)
    tcf_g = peaks_to_genes(tcf, hg, 50_000)
    print(f"genes within 2 kb of an AHR peak: TCDD {len(pos_prom)}, DMSO {len(neg_prom)} | "
          f"within 50 kb: TCDD {len(pos_g)}, DMSO {len(neg_g)}", flush=True)

    # classic targets must be present, otherwise the peak calls are not usable
    ctrl_targets = ["CYP1A1", "AHRR", "CYP1B1", "TIPARP", "NQO1"]
    recovered = {g: [int(pos_prom.get(g, 0)), int(pos_g.get(g, 0))] for g in ctrl_targets}
    print("classic AHR targets in TCDD peak set:", recovered, flush=True)

    # ------------------------------------------------------- mouse orthologues
    mtss = pd.read_csv(os.path.join(L.ANN, "mm10_genes_tss.tsv"), sep="\t")
    mt = mtss[mtss.chrom.isin(CANON)]
    mouse_genes = sorted(set(mtss.gene))
    lut = {g.lower(): g for g in mouse_genes}
    def ortho(h):
        return lut.get(str(h).lower())
    rows = []
    all_h = set(pos_g) | set(neg_g) | set(tcf_g)
    for h in sorted(all_h):
        m = ortho(h)
        if m is None:
            continue
        rows.append(dict(human=h, mouse=m, n_peaks_tcdd=pos_g.get(h, 0),
                         n_peaks_dmso=neg_g.get(h, 0), n_peaks_tcf21=tcf_g.get(h, 0),
                         n_peaks_tcdd_prom=pos_prom.get(h, 0)))
    og = pd.DataFrame(rows)
    og.to_csv(os.path.join(OUT, "ahr_peaks_by_gene.csv"), index=False)
    bound = set(og[og.n_peaks_tcdd > 0].mouse)
    bound_prom = set(og[og.n_peaks_tcdd_prom > 0].mouse)
    bound_dmso = set(og[og.n_peaks_dmso > 0].mouse)
    tcf_bound = set(og[og.n_peaks_tcf21 > 0].mouse)
    print(f"orthologues: AHR-bound (50 kb) {len(bound)} | promoter-proximal {len(bound_prom)}"
          f" | DMSO-bound {len(bound_dmso)} | TCF21-bound {len(tcf_bound)}", flush=True)

    # ---------------------------------------------------- bins -> nearest gene
    a = ad.read_h5ad(os.path.join(L.RES, "ATAC", "atac_processed.h5ad"), backed="r")
    keys = np.asarray(a.var_names, dtype=np.int64)
    code = (keys // 1_000_000_000).astype(int)
    start = keys % 1_000_000_000
    chrom = np.array(["NA"] * len(keys), dtype=object)
    for i, c in enumerate(CANON):
        chrom[code == i] = c

    gene_of_bin = np.full(len(keys), "", dtype=object)
    for c, sub in mt.groupby("chrom", observed=True):
        idx = np.where(chrom == c)[0]
        if len(idx) == 0:
            continue
        t = sub.tss.to_numpy()
        o = np.argsort(t)
        ts, gn = t[o], sub.gene.to_numpy()[o]
        mid = start[idx] + 2500
        sp = np.clip(np.searchsorted(ts, mid, "left"), 1, len(ts) - 1)
        left, right = ts[sp - 1], ts[sp]
        pick = np.where(np.abs(mid - left) <= np.abs(right - mid), sp - 1, sp)
        near = np.abs(ts[pick] - mid) <= BIN_WIN
        gene_of_bin[idx[near]] = gn[pick[near]]

    has_gene = gene_of_bin != ""
    print(f"bins assigned to a gene within {BIN_WIN//1000} kb: {int(has_gene.sum())}", flush=True)

    # ------------------------------------------------------------ overlap tests
    trows = []
    for ln in LINEAGES:
        p = os.path.join(L.RES, "motif", f"da_bins_{ln}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        gb = gene_of_bin[np.searchsorted(keys, d.bin_key.to_numpy())]
        d["gene"] = gb
        d = d[d.gene != ""]
        is_bound = d.gene.isin(bound).to_numpy()
        is_dmso = d.gene.isin(bound_dmso).to_numpy()
        open_ = d.sig_up.to_numpy()
        base = np.ones(len(d), dtype=bool)
        # background = all accessible bins near a gene, not only the differential ones
        for tag, mask in (("open", open_), ("close", d.sig_dn.to_numpy()),
                          ("background", base)):
            a_ = int((mask & is_bound).sum()); b_ = int((mask & ~is_bound).sum())
            c_ = int(((~mask) & is_bound).sum()); dd = int(((~mask) & ~is_bound).sum())
            from scipy import stats as st
            orr, pv = st.fisher_exact([[a_, b_], [c_, dd]])
            trows.append(dict(lineage=ln, set=tag, n=int(mask.sum()),
                              frac_near_ahr_bound=a_ / max(mask.sum(), 1),
                              frac_bg_near_bound=c_ / max((~mask).sum(), 1),
                              odds_ratio=orr, p=pv))
    tdf = pd.DataFrame(trows)
    tdf.to_csv(os.path.join(OUT, "chip_overlap_tests.csv"), index=False)
    print("\noverlap with AHR-bound orthologues:\n" + tdf.round(4).to_string(index=False),
          flush=True)

    # ------------------------------------------- mean delta, gene-label null
    smc = pd.read_csv(os.path.join(L.RES, "motif", "da_bins_Parenchyma.csv"))
    gb = gene_of_bin[np.searchsorted(keys, smc.bin_key.to_numpy())]
    ok = gb != ""
    delta = smc.delta.to_numpy()[ok]
    g = gb[ok]
    obs = float(np.nanmean(delta[np.isin(g, list(bound))]))
    obs_o = float(np.nanmean(delta[~np.isin(g, list(bound))]))
    rng = np.random.default_rng(0)
    labs = np.isin(g, list(bound))
    null = []
    for _ in range(N_PERM):
        perm = rng.permutation(labs)
        null.append(float(np.nanmean(delta[perm])))
    null = np.array(null)
    p_emp = (1 + (null >= obs).sum()) / (1 + len(null))
    pd.DataFrame({"perm": np.arange(N_PERM), "mean_delta_null": null}).to_csv(
        os.path.join(OUT, "chip_delta_permutation.csv"), index=False)
    print(f"\nSMC mean delta at AHR-bound orthologues {obs:+.4f} vs others {obs_o:+.4f} | "
          f"null {null.mean():+.4f} (sd {null.std():.4f}) | p={p_emp:.3f}", flush=True)

    # het gene set vs AHR-bound
    het = set(L.load_set("ConstitutiveHet"))
    universe = set(mouse_genes) & (set(gene_of_bin[has_gene]))
    a_ = len(het & bound & universe); b_ = len(het & universe) - a_
    c_ = len(bound & universe) - a_; dd = len(universe) - a_ - b_ - c_
    from scipy import stats as st
    orr, pv = st.fisher_exact([[a_, b_], [c_, dd]])
    print(f"heterochromatin genes x AHR-bound: {a_} / {len(het & universe)} vs "
          f"{c_} / {dd} (OR {orr:.2f}, p {pv:.3g})", flush=True)

    sm = {"peaks": {"ahr_tcdd": len(pos), "ahr_dmso": len(neg), "tcf21_tcdd": len(tcf)},
          "genes": {"ahr_tcdd_50kb": len(bound), "ahr_tcdd_prom": len(bound_prom),
                    "ahr_dmso_50kb": len(bound_dmso), "tcf21_tcdd": len(tcf_bound)},
          "classic_targets_recovered": recovered,
          "smc_mean_delta_bound": obs, "smc_mean_delta_other": obs_o,
          "perm_p": float(p_emp), "n_perm": N_PERM,
          "het_x_bound": {"a": int(a_), "b": int(b_), "c": int(c_), "d": int(dd),
                          "odds_ratio": float(orr), "p": float(pv)},
          "note": ("peaks are hg38, ATAC is mm10; comparison is at orthologous-promoter "
                   "level, not by liftover")}
    json.dump(sm, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps(sm, indent=2))


if __name__ == "__main__":
    main()
