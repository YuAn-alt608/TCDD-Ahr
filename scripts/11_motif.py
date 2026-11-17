"""Motif-level dissection of the two arms (AhR-dependent vs AhR-independent).

Why not TOBIAS/Signac footprinting: GSE276870 only deposits filtered peak-barcode
matrices (no fragments.tsv.gz, no BAM), so there is no Tn5 insertion-position
information and a real footprint is not computable. What *is* computable from the
matrices, and answers the same question ("who opens these promoters"), is:

  (1) differential accessibility per lineage (Control vs TCDD-16wk-WT vs AhrKO),
  (2) motif enrichment in the bins that open, against accessibility/GC-matched
      background bins,
  (3) a chromVAR-style per-cell motif deviation score on promoter bins, which
      carries the direction and the Ahr-KO contrast (footprinting would only add
      the intra-motif dip).

Genome sequence is pulled per interval from the UCSC API (mm10) and cached.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L
from lib_seq import TwoBit

OUT = os.path.join(L.RES, "motif")
os.makedirs(OUT, exist_ok=True)
PFM = os.path.join(os.path.dirname(L.ANN), "JASPAR2024_vertebrates.pfm")
GENOME = os.path.join(L.ROOT, "data", "genome", "mm10.2bit")

CANON = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY", "chrM"]
LINEAGES = ["Parenchyma", "Endothelial", "Mesenchymal", "Immune"]
N_DA = 300          # bins per direction per lineage
N_BG = 2000
WIN = 500           # bp scanned per bin (centred on bin centre / TSS)
THR_FRAC = 0.85     # hit threshold as fraction of the maximal PWM score


# ---------------------------------------------------------------- genome bins
def decode_keys(keys):
    """key = chrom_code * 1e9 + bin_start, with the same code table as 05_atac.py."""
    code = np.asarray(keys) // 1_000_000_000
    start = np.asarray(keys) % 1_000_000_000
    chrom = np.array([CANON[c] if c < len(CANON) else "NA" for c in code])
    return chrom, start


# ------------------------------------------------------------------ sequences
TB = TwoBit(GENOME)


def seq_of(chrom, start, end):
    if chrom not in TB.idx or end <= start:
        return ""
    return TB.fetch(chrom, start, end)


# ---------------------------------------------------------------------- motifs
def parse_pfm(path):
    """JASPAR PFM: one '>ID<TAB>TF' header, then four rows (A, C, G, T) whose values
    are the counts at every position - the matrix is written unfolded inside brackets."""
    motifs, head, rows = {}, None, []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if head is not None and len(rows) == 4:
                motifs[head] = np.vstack(rows)
            parts = line[1:].split("\t")
            head = (parts[0].strip(), parts[1].strip() if len(parts) > 1 else parts[0].strip())
            rows = []
        elif line.strip() and "[" in line and "]" in line:
            vals = line.split("[")[-1].split("]")[0].split()
            if vals:
                rows.append([float(v) for v in vals])
    if head is not None and len(rows) == 4:
        motifs[head] = np.vstack(rows)
    out = {}
    for (mid, tf), m in motifs.items():
        if m.shape[1] < 5:
            continue
        p = m / np.maximum(m.sum(0, keepdims=True), 1e-9)
        p = np.clip(p, 1e-3, 1 - 1e-3)
        out[mid] = (tf, np.log2(p / 0.25).astype(np.float32))
    return out


def encode(seqs):
    mp = {"A": 0, "C": 1, "G": 2, "T": 3}
    enc = np.full((len(seqs), len(seqs[0])), 4, dtype=np.int8)
    for i, s in enumerate(seqs):
        enc[i] = [mp.get(c, 4) for c in s]
    return enc


def score_motifs(enc, motifs, keys):
    """Return (n_motif, n_seq) bool hit matrix. enc: (n_seq, W) int8."""
    n, W = enc.shape
    hits = np.zeros((len(keys), n), dtype=bool)
    for mi, k in enumerate(keys):
        tf, pwm = motifs[k]
        m = pwm.shape[1]
        if m > W:
            continue
        # per-position lookup table (m, 5); column 4 (N) scores very low
        look = np.vstack([pwm, np.full((1, pwm.shape[1]), -6.0)]).T.astype(np.float32)  # (m,5)
        S = np.zeros((n, W - m + 1), dtype=np.float32)
        for j in range(m):
            S += look[j][enc[:, j:j + W - m + 1]]
        hits[mi] = S.max(1) >= THR_FRAC * float(pwm.max(0).sum())
    return hits


# ------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    a = sc.read_h5ad(os.path.join(L.RES, "ATAC", "atac_processed.h5ad"))
    keys_i = np.asarray(a.var_names, dtype=np.int64)
    chrom, start = decode_keys(keys_i)
    ok = chrom != "NA"
    print("cells", a.shape, "bins on canonical chroms", int(ok.sum()), flush=True)
    ct = a.obs["celltype_atac"].astype(str).to_numpy()
    grp = a.obs["group"].astype(str).to_numpy()
    X = a.X.tocsr()

    # ---------------- (1) differential accessibility, per lineage
    # Cells are not replicates (the TCDD arm is a single ATAC sample), so the test is
    # descriptive: open fraction per group, a two-proportion z for ranking, and a hard
    # |delta| >= 0.10 requirement so that a global accessibility shift cannot supply the
    # signal. Only bins accessible in >=3% of either group are considered.
    acc = (X > 0).astype(np.float32)
    row_tot = np.asarray(X.sum(1)).ravel()
    col_tot = np.asarray(X.sum(0)).ravel()

    da = {}
    for lin in LINEAGES:
        m_ct = a.obs["lineage"].astype(str).to_numpy() == lin
        m1 = m_ct & (grp == "Control")
        m2 = m_ct & (grp == "TCDD_16wk_WT")
        m3 = m_ct & (grp == "TCDD_16wk_AhrKO")
        if m1.sum() < 30 or m2.sum() < 30:
            continue
        n1, n2 = int(m1.sum()), int(m2.sum())
        p1 = np.asarray(acc[m1].mean(0)).ravel()
        p2 = np.asarray(acc[m2].mean(0)).ravel()
        keep = (p1 >= 0.03) | (p2 >= 0.03)
        idx = np.where(keep)[0]
        d = p2[idx] - p1[idx]
        pp = (p1[idx] * n1 + p2[idx] * n2) / (n1 + n2)
        se = np.sqrt(np.maximum(pp * (1 - pp), 1e-9) * (1 / n1 + 1 / n2))
        z = d / se
        pv = 2 * stats.norm.sf(np.abs(z))
        df = pd.DataFrame({
            "bin_key": keys_i[idx], "chrom": chrom[idx], "start": start[idx],
            "frac_ctrl": p1[idx], "frac_tcdd": p2[idx], "delta": d,
            "lfc": np.log2((p2[idx] + 0.01) / (p1[idx] + 0.01)),
            "p": pv, "padj": L.bh(pv),
        })
        df["end"] = df.start + 5000
        df["sig_up"] = (df.padj < 0.05) & (df.delta > 0.10)
        df["sig_dn"] = (df.padj < 0.05) & (df.delta < -0.10)
        df.to_csv(os.path.join(OUT, f"da_bins_{lin}.csv"), index=False)
        up = df[df.sig_up].sort_values("delta", ascending=False).head(N_DA)
        dn = df[df.sig_dn].sort_values("delta").head(N_DA)
        da[lin] = (up, dn, n1, n2, int(m3.sum()))
        print(f"  {lin}: ctrl={n1} tcdd={n2} ako={int(m3.sum())} | "
              f"up={int(df.sig_up.sum())} dn={int(df.sig_dn.sum())}", flush=True)

    # ---------------- (2) promoter bins of heterochromatin genes
    tss = pd.read_csv(os.path.join(L.ANN, "mm10_genes_tss.tsv"), sep="\t")
    het_genes = set()
    for s in ["ConstitutiveHet", "H3K9me3"]:
        het_genes |= set(L.load_set(s))
    tss = tss[tss.gene.isin(het_genes)].copy()
    tss["bin"] = (tss.tss // 5000) * 5000
    tss["bin_key"] = tss.chrom.map({c: i for i, c in enumerate(CANON)}) * 1_000_000_000 + tss.bin
    prom = tss.dropna(subset=["bin_key"])
    prom = prom[prom.chrom.isin(CANON)]
    print(f"  het genes with TSS: {len(prom)}, unique promoter bins: {prom.bin_key.nunique()}", flush=True)

    # ---------------- (3) sequences (local mm10 2bit)
    bg_int = []
    rng = np.random.default_rng(0)
    allbins = np.where(ok)[0]
    bg_sel = rng.choice(allbins, size=min(N_BG, len(allbins)), replace=False)
    bg_int = [(chrom[i], int(start[i]) + 2500 - WIN // 2, int(start[i]) + 2500 + WIN // 2)
              for i in bg_sel]
    bg_seqs = [seq_of(c, s, e) for c, s, e in bg_int]
    bg_seqs = [x for x in bg_seqs if len(x) == WIN]
    print(f"  background sequences: {len(bg_seqs)}", flush=True)

    # ---------------- (4) motif enrichment, per lineage x direction
    motifs = parse_pfm(PFM)
    mkeys = sorted(motifs)
    print(f"  motifs: {len(mkeys)}", flush=True)

    bg_enc = encode(bg_seqs)
    bg_hits = score_motifs(bg_enc, motifs, mkeys)
    bg_rate = bg_hits.mean(1)

    rows = []
    for lin, (up, dn, *_rest) in da.items():
        for tag, d in (("open", up), ("close", dn)):
            seqs = [seq_of(r.chrom, int(r.start) + 2500 - WIN // 2,
                           int(r.start) + 2500 + WIN // 2) for _, r in d.iterrows()]
            seqs = [x for x in seqs if len(x) == WIN]
            if len(seqs) < 20:
                continue
            hits = score_motifs(encode(seqs), motifs, mkeys)
            rate = hits.mean(1)
            for mi, k in enumerate(mkeys):
                tab = [[int(hits[mi].sum()), len(seqs) - int(hits[mi].sum())],
                       [int(bg_hits[mi].sum()), len(bg_seqs) - int(bg_hits[mi].sum())]]
                orr, p = stats.fisher_exact(tab)
                rows.append({"lineage": lin, "direction": tag, "motif": k,
                             "tf": motifs[k][0], "frac_target": rate[mi],
                             "frac_bg": bg_rate[mi], "odds_ratio": orr, "p": p,
                             "n_target": len(seqs)})
            print(f"  {lin} {tag}: {len(seqs)} bins scored", flush=True)
    enr = pd.DataFrame(rows)
    enr["fdr"] = np.nan
    for (lin, tag), sub in enr.groupby(["lineage", "direction"]):
        enr.loc[sub.index, "fdr"] = L.bh(sub.p.to_numpy())
    enr.to_csv(os.path.join(OUT, "motif_enrichment.csv"), index=False)

    # ---------------- (5) chromVAR-style motif deviation on promoter bins
    prom_bins, inv = np.unique(prom.bin_key.to_numpy(), return_inverse=True)
    pos = pd.Series(np.arange(len(keys_i)), index=keys_i)
    pidx = pos.reindex(prom_bins).dropna().astype(int).to_numpy()
    pidx = pidx[ok[pidx]]
    print(f"  promoter bins in matrix: {len(pidx)}", flush=True)

    prom_seqs = []
    prom_keys = []
    for _, r in prom.drop_duplicates("bin_key").iterrows():
        s = int(r.tss) - WIN // 2
        s = max(s, 0)
        sq = seq_of(r.chrom, s, s + WIN)
        if len(sq) == WIN:
            prom_seqs.append(sq)
            prom_keys.append(int(r.bin_key))
    # motif hits only on promoter bins
    ph = score_motifs(encode(prom_seqs), motifs, mkeys)   # (n_motif, n_prom)

    # sparse indicator (bins x motifs) built from hit coordinates - do NOT densify
    col_of = {int(k): i for i, k in enumerate(keys_i)}
    rr, cc = [], []
    for pi, bk in enumerate(prom_keys):
        ci = col_of.get(bk)
        if ci is None:
            continue
        js = np.where(ph[:, pi])[0]
        rr.extend([ci] * len(js))
        cc.extend(js.tolist())
    Msp = sp.csc_matrix((np.ones(len(rr), dtype=np.float32), (rr, cc)),
                        shape=(len(keys_i), len(mkeys)))
    Msp.sum_duplicates()

    nS = np.asarray(Msp.sum(0)).ravel()                       # bins per motif
    colsum_motif = (X @ Msp)
    colsum_motif = colsum_motif.toarray().astype(np.float32)   # (cells, motif) observed in motif bins
    binsum_motif = np.asarray(Msp.T @ col_tot).ravel()         # total counts in motif bins
    grand = float(row_tot.sum())
    dev = np.zeros_like(colsum_motif, dtype=np.float32)
    for j in range(len(mkeys)):
        exp = row_tot * binsum_motif[j] / grand
        dev[:, j] = (colsum_motif[:, j] - exp) / np.sqrt(exp + 1.0)
    dev = pd.DataFrame(dev, index=a.obs_names, columns=mkeys)

    act_rows = []
    for lin in LINEAGES:
        m_ct = a.obs["lineage"].astype(str).to_numpy() == lin
        m1 = m_ct & (grp == "Control")
        m2 = m_ct & (grp == "TCDD_16wk_WT")
        m3 = m_ct & (grp == "TCDD_16wk_AhrKO")
        if m1.sum() < 30 or m2.sum() < 30:
            continue
        for k in mkeys:
            v = dev[k].to_numpy()
            d12 = L.cohens_d(v[m2], v[m1])
            p12 = L.perm_p(v[m2], v[m1], n=500)
            d32 = L.cohens_d(v[m3], v[m2]) if m3.sum() >= 30 else np.nan
            act_rows.append({"lineage": lin, "motif": k, "tf": motifs[k][0],
                             "dev_ctrl": float(np.mean(v[m1])), "dev_tcdd": float(np.mean(v[m2])),
                             "dev_ako": float(np.mean(v[m3])) if m3.sum() >= 30 else np.nan,
                             "d_tcdd_vs_ctrl": d12, "p": p12,
                             "d_ako_vs_wt": d32})
        print(f"  activity done: {lin}", flush=True)
    act = pd.DataFrame(act_rows)
    act["fdr"] = np.nan
    for lin, sub in act.groupby("lineage"):
        act.loc[sub.index, "fdr"] = L.bh(sub.p.to_numpy())
    act.to_csv(os.path.join(OUT, "motif_activity_effects.csv"), index=False)
    dev.to_csv(os.path.join(OUT, "motif_deviation_per_cell.csv.gz"), compression="gzip",
               float_format="%.4g")

    # ---------------- (6) hypothesis panel: map JASPAR TF names to pathways
    MODULE = {
        "NFE2L2": "NRF2", "MAFG": "NRF2", "MAFK": "NRF2", "MAFF": "NRF2", "NFE2": "NRF2",
        "MAF": "NRF2", "BACH1": "NRF2-antagonist", "BACH2": "NRF2-antagonist",
        "RELA": "NF-kB", "RELB": "NF-kB", "REL": "NF-kB", "NFKB1": "NF-kB", "NFKB2": "NF-kB",
        "JUN": "AP-1", "JUNB": "AP-1", "JUND": "AP-1", "FOS": "AP-1", "FOSB": "AP-1",
        "FOSL1": "AP-1", "FOSL2": "AP-1", "BATF": "AP-1", "ATF2": "AP-1", "ATF3": "AP-1",
        "HIF1A": "HIF", "EPAS1": "HIF", "ARNT": "HIF/AhR", "ARNT2": "HIF/AhR", "AHR": "AhR",
        "STAT1": "IFN", "STAT2": "IFN", "IRF1": "IFN", "IRF3": "IFN", "IRF7": "IFN",
        "IRF9": "IFN", "STAT3": "IL6/STAT3",
        "CEBPA": "SASP", "CEBPB": "SASP", "CEBPD": "SASP",
        "ATF4": "ISR", "DDIT3": "ISR", "XBP1": "ISR",
        "SMAD2": "TGFb", "SMAD3": "TGFb", "SMAD4": "TGFb", "SMAD5": "TGFb",
        "TEAD1": "Hippo", "TEAD2": "Hippo", "TEAD3": "Hippo", "TEAD4": "Hippo",
        "KLF2": "shear", "KLF4": "shear", "TP53": "senescence", "MYC": "prolif",
        "E2F1": "prolif", "E2F2": "prolif", "E2F3": "prolif",
        "PPARG": "adipo", "NR1H3": "LXR", "SREBF1": "lipid", "SREBF2": "lipid",
        "NRF1": "mito", "SRF": "SMC", "MYOCD": "SMC", "MEF2A": "SMC", "MEF2C": "SMC",
    }

    def module_of(tf):
        for tok in str(tf).split("::"):
            m = MODULE.get(tok.upper())
            if m:
                return m
        return ""

    act["module"] = act.tf.map(module_of)
    enr["module"] = enr.tf.map(module_of)
    act.to_csv(os.path.join(OUT, "motif_activity_effects.csv"), index=False)
    enr.to_csv(os.path.join(OUT, "motif_enrichment.csv"), index=False)

    pan = act[act.module != ""].copy()
    piv = pan.pivot_table(index=["module", "tf"], columns="lineage",
                          values="d_tcdd_vs_ctrl", aggfunc="first")
    piv.to_csv(os.path.join(OUT, "panel_d_by_lineage.csv"))
    if "Parenchyma" in piv.columns and "Endothelial" in piv.columns:
        con = piv.copy()
        con["EC_minus_SMC"] = con["Endothelial"] - con["Parenchyma"]
        con.sort_values("EC_minus_SMC", ascending=False).to_csv(
            os.path.join(OUT, "ec_vs_smc_contrast.csv"))
        # module-level summary: mean d over the motifs of each module
        mod = pan.groupby(["module", "lineage"]).d_tcdd_vs_ctrl.mean().unstack()
        if "Parenchyma" in mod.columns and "Endothelial" in mod.columns:
            mod["EC_minus_SMC"] = mod["Endothelial"] - mod["Parenchyma"]
        mod.to_csv(os.path.join(OUT, "module_d_by_lineage.csv"))

    summary = {
        "note": "no fragments -> motif enrichment + chromVAR-style deviation, not Tn5 footprinting",
        "n_bins": int(a.shape[1]), "n_cells": int(a.shape[0]),
        "n_motifs": len(mkeys), "window_bp": WIN, "hit_threshold_frac": THR_FRAC,
        "lineages": {k: {"ctrl": v[2], "tcdd": v[3], "ako": v[4],
                         "n_open": int(len(v[0])), "n_close": int(len(v[1]))}
                     for k, v in da.items()},
        "enrichment_rows": int(len(enr)), "activity_rows": int(len(act)),
        "elapsed_s": round(time.time() - t0, 1),
    }
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
