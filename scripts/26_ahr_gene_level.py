"""Item 2 of the pre-submission list, done at gene resolution instead of 5 kb bins.

The bin-level test (22b) said: of 270 SMC bins that open, only 14 land inside a human
AHR peak (OR 1.29, p = 0.35). That test is fragile for three reasons -- 5 kb bins average
over the peak, liftOver keeps only 47% of them, and a bin can miss a promoter entirely.

So ask the same question the ChIP-qPCR would ask, but in silico and gene by gene:

    are the heterochromatin-gene promoters we claim get opened actually occupied by
    AHR in human coronary SMC?

Design, including the controls that make a negative interpretable:

  * het set    = TSS of ConstitutiveHet ∪ H3K9me3 genes (the same 1,987 promoters as 11b)
  * reference  = 7,817 genes with a human ortholog and a peak count (ahr_peaks_by_gene)
  * outcome    = AHR TCDD peak within ±2 kb of the human TSS
  * specificity= the same test with DMSO peaks (262) and with TCF21 peaks (131 genes)
                 -- if het genes are equally "enriched" for a control TF, the signal is
                 promoter-general, not AHR-specific
  * confounder = expression. het genes are largely silent in HCASMC; an AHR peak cannot
                 sit on a promoter that is never open. So detection in human SMC is
                 measured for every gene and the test is repeated within expressed genes.

Outputs (results/chip/)
    gene_level_promoters.csv   per-gene table: motif, human detection, mouse opening, peaks
    gene_level_tests.csv       Fisher tests (overall / motif-stratified / expressed-only / controls)
    qpcr_candidates.csv        Tier A (bound, opened), Tier B (opened, not bound), Tier C (canonical)
    gene_level_summary.json
"""
import os, sys, gzip, json, time
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L
import importlib.util

spec = importlib.util.spec_from_file_location("m11", os.path.join(os.path.dirname(__file__), "11_motif.py"))
M11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M11)

OUT = os.path.join(L.RES, "chip")
ANN = L.ANN
WIN = 1000          # same window as 11b, so the promoter sets reconcile
THR_FRAC = 0.85
CANON = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"]
AHR_MATRICES = ["MA0006.2"]          # Ahr::Arnt, the only family member in JASPAR2024
HUMAN = os.path.join(L.ROOT, "data", "external", "human_coronary.txt.gz")
REFFLAT = os.path.join(L.ROOT, "data", "external", "hg38_refFlat.txt.gz")
N_BG = 3000
CANONICAL_TARGETS = ["CYP1A1", "CYP1B1", "AHRR", "TIPARP", "NQO1", "AHR"]


def load_het_genes():
    genes = set()
    for s in ["ConstitutiveHet", "H3K9me3"]:
        genes |= set(L.load_set(s))
    return genes


def human_tss():
    rows = []
    with gzip.open(REFFLAT, "rt") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            # refFlat: geneName, name, chrom, strand, txStart, txEnd, ...
            name, chrom, strand = p[0], p[2], p[3]
            tx_start, tx_end = int(p[4]), int(p[5])
            if chrom not in CANON:
                continue
            rows.append((name, chrom, tx_end if strand == "+" else tx_start, strand))
    d = pd.DataFrame(rows, columns=["human", "chrom", "tss", "strand"])
    return d.drop_duplicates("human").set_index("human")


def smc_detection(wanted):
    """Detection fraction inside human SMC cells, streaming the dense text matrix once."""
    with gzip.open(HUMAN, "rt") as f:
        cells = f.readline().rstrip("\n").split("\t")
    lab = pd.read_csv(os.path.join(L.RES, "human", "human_celltype_scores.csv"),
                      index_col=0, usecols=["Unnamed: 0", "compartment"])
    assert len(lab) == len(cells)
    is_smc = (lab.compartment == "SMC").to_numpy()
    det, tot = {}, {}
    t0 = time.time()
    with gzip.open(HUMAN, "rt") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2 or p[0] not in wanted:
                continue
            v = np.fromiter((float(x) for x in p[1:]), dtype=np.float32, count=len(p) - 1)
            det[p[0]] = float((v[is_smc] > 0).mean())
            tot[p[0]] = float(v[is_smc].mean())
    print(f"  human detection for {len(det)}/{len(wanted)} genes in {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame({"human": list(det), "smc_detect": list(det.values()),
                         "smc_mean": [tot[k] for k in det]}).set_index("human")


def main():
    t0 = time.time()
    het = load_het_genes()
    tss = pd.read_csv(os.path.join(ANN, "mm10_genes_tss.tsv"), sep="\t")
    tss = tss[tss.chrom.isin(CANON)].copy()

    # ---------------------------------------------------------------- accessibility
    import scanpy as sc
    a = sc.read_h5ad(os.path.join(L.RES, "ATAC", "atac_processed.h5ad"))
    keys = np.asarray(a.var_names, dtype=np.int64)
    chrom, start = M11.decode_keys(keys)
    lin = a.obs["lineage"].astype(str).to_numpy()
    grp = a.obs["group"].astype(str).to_numpy()
    acc = (a.X > 0).astype(np.float32)
    print(f"  ATAC {acc.shape}, {time.time()-t0:.0f}s", flush=True)

    tss["bin_key"] = tss.chrom.map({c: i for i, c in enumerate(CANON)}) * 1_000_000_000 + (tss.tss // 5000) * 5000
    pos = pd.Series(np.arange(len(keys)), index=keys)
    tss["col"] = pos.reindex(tss.bin_key.to_numpy()).to_numpy()
    tss = tss[tss.col.notna()].copy()
    tss["col"] = tss.col.astype(int)
    print(f"  promoters mapped to bins: {len(tss)}, {time.time()-t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- Ahr motif
    motifs = M11.parse_pfm(M11.PFM)
    sub = {k: motifs[k] for k in AHR_MATRICES if k in motifs}
    assert sub, "AHR matrix not found in JASPAR file"
    seqs, keep = [], []
    for i, r in tss.reset_index(drop=True).iterrows():
        s = max(int(r.tss) - WIN // 2, 0)
        sq = M11.seq_of(r.chrom, s, s + WIN)
        if len(sq) == WIN:
            seqs.append(sq)
            keep.append(i)
    tss = tss.reset_index(drop=True).loc[keep].reset_index(drop=True)
    enc = M11.encode(seqs)
    # score_motifs already applies THR_FRAC and returns a bool hit matrix; keep the raw
    # best-window score as a second column so the threshold can be revisited without rerunning
    hits = M11.score_motifs(enc, sub, sorted(sub))
    pwm = sub[sorted(sub)[0]][1]
    m = pwm.shape[1]
    look = np.vstack([pwm, np.full((1, m), -6.0)]).T.astype(np.float32)
    S = np.zeros((enc.shape[0], enc.shape[1] - m + 1), dtype=np.float32)
    for j in range(m):
        S += look[j][enc[:, j:j + enc.shape[1] - m + 1]]
    tss["ahr_motif_score"] = S.max(1)
    tss["ahr_motif"] = hits.any(0)
    print(f"  AHR motif scored: {int(tss.ahr_motif.sum())}/{len(tss)} promoters positive, "
          f"{time.time()-t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- per-group opening
    rows = []
    for lineage in ["Parenchyma", "Endothelial"]:
        m_ct = lin == lineage
        g_ctrl = acc[m_ct & (grp == "Control")][:, tss.col.to_numpy()].mean(0)
        g_wt = acc[m_ct & (grp == "TCDD_16wk_WT")][:, tss.col.to_numpy()].mean(0)
        g_ko = acc[m_ct & (grp == "TCDD_16wk_AhrKO")][:, tss.col.to_numpy()].mean(0)
        g_ctrl = np.asarray(g_ctrl).ravel()
        g_wt = np.asarray(g_wt).ravel()
        g_ko = np.asarray(g_ko).ravel()
        d = tss[["gene", "chrom", "tss", "ahr_motif", "ahr_motif_score"]].copy()
        d["lineage"] = lineage
        d["open_ctrl"] = g_ctrl
        d["open_wt"] = g_wt
        d["open_ako"] = g_ko
        d["delta_wt"] = g_wt - g_ctrl
        d["delta_ako"] = g_ko - g_wt
        rows.append(d)
    per = pd.concat(rows, ignore_index=True)
    per["is_het_promoter"] = per.gene.isin(het)
    print(f"  accessibility done: {time.time()-t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- merge peaks
    ch = pd.read_csv(os.path.join(OUT, "ahr_peaks_by_gene.csv"))
    ch["key"] = ch.mouse.str.lower()
    per["key"] = per.gene.str.lower()
    per = per.merge(ch.drop(columns=["mouse"]), on="key", how="left")
    # genes absent from the ChIP table have no human ortholog / no annotated promoter
    per["in_chip_table"] = per.human.notna()
    print(f"  merged: {int(per.in_chip_table.sum())} of {len(per)} promoter-rows in ChIP table",
          flush=True)

    # ---------------------------------------------------------------- human expression
    het_h = set(per[per.is_het_promoter & per.in_chip_table].human)
    rng = np.random.default_rng(3)
    bg_pool = ch[~ch.key.isin({g.lower() for g in het})].human.dropna().unique()
    bg_h = set(rng.choice(bg_pool, size=min(N_BG, len(bg_pool)), replace=False))
    det = smc_detection(het_h | bg_h | set(CANONICAL_TARGETS))
    per = per.merge(det, left_on="human", right_index=True, how="left")

    # ---------------------------------------------------------------- Fisher tests
    def fisher(df, outcome, mask_a, mask_b, label):
        a_ = df[mask_a]
        b_ = df[mask_b]
        if len(a_) < 5 or len(b_) < 5:
            return None
        x1 = int(a_[outcome].sum()); n1 = len(a_)
        x2 = int(b_[outcome].sum()); n2 = len(b_)
        orr, p = stats.fisher_exact([[x1, n1 - x1], [x2, n2 - x2]])
        return {"test": label, "outcome": outcome, "group_a": "het", "n_a": n1, "hits_a": x1,
                "rate_a": x1 / n1, "group_b": "background", "n_b": n2, "hits_b": x2,
                "rate_b": x2 / n2, "odds_ratio": float(orr), "p": float(p)}

    base = per[(per.lineage == "Parenchyma") & per.in_chip_table].drop_duplicates("gene")
    base["ahr_prom"] = base.n_peaks_tcdd_prom.fillna(0) > 0
    base["ahr_any"] = base.n_peaks_tcdd.fillna(0) > 0
    base["tcf21_prom"] = base.n_peaks_tcf21.fillna(0) > 0
    base["dmso_prom"] = base.n_peaks_dmso.fillna(0) > 0
    base = base[base.human.isin(set(det.index))]           # only genes we could measure
    base["expressed"] = base.smc_detect.fillna(0) >= 0.10

    bg_mask = (~base.is_het_promoter) & base.human.isin(bg_h)
    tests = []
    tests.append(fisher(base, "ahr_prom", base.is_het_promoter, bg_mask, "het vs background｜AHR 启动子峰"))
    tests.append(fisher(base, "ahr_prom", base.is_het_promoter & base.ahr_motif,
                        base.is_het_promoter & ~base.ahr_motif, "het 内部｜有 vs 无 AhR motif"))
    tests.append(fisher(base, "ahr_prom", base.is_het_promoter & base.expressed,
                        bg_mask & base.expressed, "仅在人类 SMC 可检出基因中｜het vs background"))
    tests.append(fisher(base, "ahr_prom", base.is_het_promoter & base.ahr_motif & base.expressed,
                        bg_mask & base.expressed, "可检出且带 motif｜het vs background"))
    tests.append(fisher(base, "ahr_any", base.is_het_promoter, bg_mask, "特异性对照｜AHR 任意峰（±50 kb）"))
    tests.append(fisher(base, "tcf21_prom", base.is_het_promoter, bg_mask, "特异性对照｜TCF21 启动子峰"))
    tests.append(fisher(base, "dmso_prom", base.is_het_promoter, bg_mask, "特异性对照｜DMSO 启动子峰"))
    tests = pd.DataFrame([t for t in tests if t])
    tests["fdr"] = L.bh(tests.p.to_numpy())
    tests.to_csv(os.path.join(OUT, "gene_level_tests.csv"), index=False)
    print(tests.to_string(index=False), flush=True)

    # ---------------------------------------------------------------- positive control
    # A ChIP can only be used as evidence if it recovers its own canonical targets.
    # CYP1A1 / AHRR / TIPARP / NQO1 are the textbook TCDD-induced AHR loci; if the peak
    # set has nothing there, a negative elsewhere says nothing about biology.
    def peaks_of(fn):
        out = []
        with gzip.open(fn, "rt") as f:
            for line in f:
                q = line.split("\t")
                out.append((q[0], (int(q[1]) + int(q[2])) // 2))
        pk = pd.DataFrame(out, columns=["chrom", "mid"])
        return {c: np.sort(g.mid.to_numpy()) for c, g in pk.groupby("chrom")}

    def n_in_window(by, chrom, tss, w):
        arr = by.get(chrom)
        if arr is None:
            return 0
        lo, hi = np.searchsorted(arr, [tss - w, tss + w])
        return int(hi - lo)

    ht_all = human_tss()
    RANDOM = tss.sample(600, random_state=1)
    pos_rows = []
    for tag, fn in [("AHR_TCDD", "GSM8492045_AHR_pos_peaks.narrowPeak.gz"),
                    ("AHR_DMSO", "GSM8492044_AHR_neg_peaks.narrowPeak.gz")]:
        by = peaks_of(os.path.join(L.ROOT, "data", "external", fn))
        for w in [10_000, 50_000]:
            tgt = [n_in_window(by, r.chrom, int(r.tss), w)
                   for g in CANONICAL_TARGETS if g in ht_all.index
                   for r in [ht_all.loc[g]]]
            bgv = [n_in_window(by, r.chrom, int(r.tss), w) for _, r in ht_all.sample(600, random_state=1).iterrows()]
            pos_rows.append({"peak_set": tag, "window_kb": w // 1000,
                             "targets": ",".join(CANONICAL_TARGETS), "n_targets": len(tgt),
                             "target_mean_peaks": float(np.mean(tgt)),
                             "target_zero_frac": float(np.mean(np.asarray(tgt) == 0)),
                             "random_mean_peaks": float(np.mean(bgv)),
                             "enrichment": float(np.mean(tgt) / max(np.mean(bgv), 1e-9))})
    pos = pd.DataFrame(pos_rows)
    pos.to_csv(os.path.join(OUT, "chip_positive_control.csv"), index=False)
    print(pos.to_string(index=False), flush=True)

    # does carrying a human AHR peak associate with mouse opening?
    hh = base[base.is_het_promoter]
    mw = stats.mannwhitneyu(hh[hh.ahr_prom].delta_wt.dropna(), hh[~hh.ahr_prom].delta_wt.dropna(),
                            alternative="two-sided") if hh.ahr_prom.any() else None

    per.drop(columns=["key"]).to_csv(os.path.join(OUT, "gene_level_promoters.csv"), index=False)

    # ---------------------------------------------------------------- qPCR panels
    # Tiers no longer depend on the public ChIP (it failed its own positive control):
    #   C  assay QC      -- canonical AHR targets, MUST show induction/binding
    #   A  hypothesis    -- het promoters, AhR motif, opened by TCDD and reverted by KO
    #   B  specificity   -- het promoters, AhR motif, NOT opened
    ht = human_tss().rename(columns={"chrom": "hg38_chrom", "tss": "hg38_tss",
                                     "strand": "hg38_strand"})
    cand = base[base.is_het_promoter].copy()
    cand = cand.merge(ht, left_on="human", right_index=True, how="left")
    cand["mouse"] = cand.gene
    a_mask = cand.ahr_motif & (cand.delta_wt > 0) & (cand.delta_ako < 0)
    b_mask = cand.ahr_motif & (cand.delta_wt <= 0)
    cand["tier"] = ""
    cand.loc[a_mask, "tier"] = "A"
    cand.loc[b_mask, "tier"] = "B"
    pick = pd.concat([cand[a_mask].nlargest(10, "delta_wt"),
                      cand[b_mask].nsmallest(6, "delta_wt")])

    # the QC panel must exist even if the public peak set never mentions the gene
    can = pd.DataFrame({"human": CANONICAL_TARGETS})
    can = can.merge(ch, on="human", how="left")
    can = can.merge(ht, left_on="human", right_index=True, how="left")
    can["mouse"] = can["mouse"] if "mouse" in can else np.nan
    can["tier"] = "C"
    for c in ["ahr_motif", "ahr_motif_score", "delta_wt", "delta_ako",
              "open_ctrl", "open_wt", "open_ako"]:
        can[c] = np.nan
    can["ahr_prom"] = can.n_peaks_tcdd_prom.fillna(0) > 0
    can["smc_detect"] = [float(det.smc_detect[g]) if g in det.index else np.nan
                         for g in can.human]

    cols = ["tier", "human", "mouse", "hg38_chrom", "hg38_tss", "hg38_strand",
            "ahr_motif", "ahr_motif_score", "ahr_prom", "n_peaks_tcdd", "n_peaks_dmso",
            "n_peaks_tcf21", "smc_detect", "delta_wt", "delta_ako",
            "open_ctrl", "open_wt", "open_ako"]
    out = pd.concat([pick[cols], can[cols]], ignore_index=True).drop_duplicates("human")
    out = out.sort_values(["tier", "delta_wt"], ascending=[True, False])

    # mouse coordinates for the decisive experiment (MOVAS + TCDD, ChIP-qPCR / ATAC-qPCR)
    mt = tss[["gene", "chrom", "tss", "strand"]].drop_duplicates("gene").set_index("gene")
    out["mm10_chrom"] = [mt.chrom.get(g, np.nan) for g in out.mouse]
    out["mm10_tss"] = [mt.tss.get(g, np.nan) for g in out.mouse]
    out["mm10_strand"] = [mt.strand.get(g, np.nan) for g in out.mouse]
    out = out[["tier"] + [c for c in out.columns if c != "tier"]]
    out.to_csv(os.path.join(OUT, "qpcr_candidates.csv"), index=False)
    print(out[["tier", "human", "mouse", "delta_wt", "delta_ako", "smc_detect"]]
          .to_string(index=False), flush=True)

    summary = {
        "het_promoters": int(per[per.is_het_promoter].gene.nunique()),
        "het_with_human_ortholog_and_peak_table": int(len(het_h)),
        "het_measured_in_human": int(len(base[base.is_het_promoter])),
        "background_genes_sampled": int(len(bg_h)),
        "ahr_motif_positive": int(per[per.is_het_promoter].drop_duplicates("gene").ahr_motif.sum()),
        "base_rate_ahr_prom_background": float(base[bg_mask].ahr_prom.mean()),
        "base_rate_ahr_prom_het": float(base[base.is_het_promoter].ahr_prom.mean()),
        "mwu_peak_vs_no_peak_delta_wt_p": (float(mw.pvalue) if mw is not None else None),
        "candidates": {k: int((out.tier == k).sum()) for k in ["A", "B", "C"]},
        "positive_control": {
            "canonical_targets": CANONICAL_TARGETS,
            "ahr_tcdd_enrichment_10kb": float(pos[(pos.peak_set == "AHR_TCDD") & (pos.window_kb == 10)].enrichment.iloc[0]),
            "ahr_tcdd_enrichment_50kb": float(pos[(pos.peak_set == "AHR_TCDD") & (pos.window_kb == 50)].enrichment.iloc[0]),
            "verdict": ("public AHR ChIP fails its positive control: canonical TCDD targets carry no "
                        "more peaks than random genes, so the ahr_prom outcome above is uninformative "
                        "about binding and is reported only to show that resolution/liftOver was not the reason"),
        },
        "note": ("Gene-level re-test of the bin-level negative (22b): outcome is an AHR TCDD "
                 "peak within 2 kb of the human TSS; controls are DMSO peaks, TCF21 peaks and "
                 "expression stratification in human SMC."),
    }
    json.dump(summary, open(os.path.join(OUT, "gene_level_summary.json"), "w"), indent=1,
              ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
