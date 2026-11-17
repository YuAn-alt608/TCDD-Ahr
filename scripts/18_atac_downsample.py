"""18 - Depth-sensitivity of the scATAC claims (Supplementary Fig. S1a-S1c).

The Ahr-KO ATAC library is ~25% shallower than the two WT libraries (median bin
counts per cell 3,718 vs 5,114), and the Ahr-KO arm is a single animal. Any
Ahr-KO-vs-WT difference could therefore be a depth artefact. This script removes
depth as an explanation by construction:

  * every cell is thinned by a binomial draw with p = T*/n_i, i.e. residual
    fragments are *sampled down* to a common budget T* (Poisson thinning, the
    standard sequencing model; exact for multinomial downsampling of counts),
  * T* is set to the Ahr-KO median (the shallowest arm) and, as a stress test,
    to 60% of it,
  * the whole chromVAR deviation statistic and the domain-accessibility
    fractions are then recomputed from the thinned matrix with the *same* code
    path as 11_motif.py / 05_atac.py,
  * five independent seeds give a spread, so "no change" cannot come from one
    lucky draw.

Outputs (results/robust/):
  downsample_effects.csv      d and perm-p for the TF panel, per seed x contrast
  downsample_modules.csv      module-level mean d, per seed x contrast
  downsample_domains.csv      frac_het / frac_lad / prom_het_rel per seed x lineage
  downsample_summary.json     depth targets, per-arm median depth, verdicts
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L
from lib_seq import TwoBit

OUT = os.path.join(L.RES, "robust")
os.makedirs(OUT, exist_ok=True)
PFM = os.path.join(os.path.dirname(L.ANN), "JASPAR2024_vertebrates.pfm")
GENOME = os.path.join(L.ROOT, "data", "genome", "mm10.2bit")
CANON = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY", "chrM"]
LINEAGES = ["Parenchyma", "Endothelial", "Mesenchymal", "Immune"]
WIN = 500
THR_FRAC = 0.85
BIN = 5000
SEEDS = [0, 1, 2, 3, 4]
N_PERM = 500

PANEL = ["Ahr::Arnt", "ARNT::HIF1A", "KLF2", "KLF4", "Nr1h3", "SMAD2", "SMAD3", "HIF1A",
         "EPAS1", "ATF2", "ATF3", "FOSL1::JUND", "JUN::JUNB", "RELA", "NFKB1", "Nfe2l2",
         "MAFK", "STAT1", "IRF1", "CEBPB", "TEAD1", "SRF", "MYOCD", "TEAD4", "TP53"]

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


# ----------------------------------------------------------------- sequence + PFM
def decode_keys(keys):
    code = np.asarray(keys) // 1_000_000_000
    start = np.asarray(keys) % 1_000_000_000
    chrom = np.array(["NA"] * len(keys), dtype=object)
    for i, c in enumerate(CANON):
        chrom[code == i] = c
    return chrom, start.astype(np.int64)


def parse_pfm(path):
    """JASPAR PFM -> log2-odds PWM, exactly as in 11_motif.py (unfolded rows)."""
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


CODE = np.full(256, 4, dtype=np.int8)
for i, ch in enumerate("ACGT"):
    CODE[ord(ch)] = i


def encode(seqs):
    """(n_seq, W) int8 index matrix (4 = N)."""
    if not seqs:
        return np.zeros((0, 0), dtype=np.int8)
    W = max(len(s) for s in seqs)
    out = np.full((len(seqs), W), 4, dtype=np.int8)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = CODE[np.frombuffer(s.encode(), dtype=np.uint8)]
    return out


def score_motifs(enc, motifs, keys, verbose=False, chunk=400):
    """(n_motif, n_seq) bool hits, same threshold as 11_motif.py.
    Scored in row chunks with a per-position lookup table: the windowed einsum
    version needs ~0.4 GB per wide motif and pages on 16 GB machines."""
    n, W = enc.shape
    hits = np.zeros((len(keys), n), dtype=bool)
    for mi, k in enumerate(keys):
        pwm = motifs[k][1]
        m = pwm.shape[1]
        if m > W:
            continue
        look = np.vstack([pwm, np.full((1, m), -6.0)]).T.astype(np.float32)
        thr = THR_FRAC * float(pwm.max(0).sum())
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            S = np.zeros((e - s, W - m + 1), dtype=np.float32)
            for j in range(m):
                S += look[j][enc[s:e, j:j + W - m + 1]]
            hits[mi, s:e] = S.max(1) >= thr
        if verbose and mi % 200 == 0:
            print(f"    motif {mi}/{len(keys)}", flush=True)
    return hits


def in_domains(keys, bed_path):
    key_chrom = (keys // 1_000_000_000).astype(int)
    key_start = keys % 1_000_000_000
    b = pd.read_csv(bed_path, sep="\t", header=None, names=["chrom", "start", "end"])
    b["cid"] = b.chrom.map(lambda c: {c2: i for i, c2 in enumerate(CANON)}.get(c, -1))
    mask = np.zeros(len(keys), dtype=bool)
    for ci, sub in b.groupby("cid", observed=True):
        if ci < 0:
            continue
        idx = np.where(key_chrom == ci)[0]
        if len(idx) == 0:
            continue
        s, e = sub.start.to_numpy(), sub.end.to_numpy()
        o2 = np.argsort(s)
        s, e = s[o2], np.maximum.accumulate(e[o2])
        en = key_start[idx] + BIN
        pos = np.searchsorted(s, en, side="right") - 1
        ok = pos >= 0
        hit = np.zeros(len(idx), dtype=bool)
        hit[ok] = e[pos[ok]] > key_start[idx][ok]
        mask[idx] = hit
    return mask


# ---------------------------------------------------------------------- main
def main():
    t0 = time.time()
    a = ad.read_h5ad(os.path.join(L.RES, "ATAC", "atac_processed.h5ad"))
    keys = np.asarray(a.var_names, dtype=np.int64)
    chrom, start = decode_keys(keys)
    ok = chrom != "NA"
    X = a.X.tocsr().astype(np.float32)
    grp = a.obs["group"].astype(str).to_numpy()
    lin = a.obs["lineage"].astype(str).to_numpy()
    row_tot = np.asarray(X.sum(1)).ravel()
    # depth per arm, on the same quantity the ATAC pipeline uses
    depth = pd.Series(row_tot, index=a.obs["sample"].values).groupby(level=0).median()
    print("median counts/cell per library:\n" + depth.round(0).to_string(), flush=True)

    # ---------------- domain masks (identical definition to 05_atac.py)
    m_het = in_domains(keys, os.path.join(L.ANN, "domains_H3K9me3.bed"))
    m_lad = in_domains(keys, os.path.join(L.ANN, "domains_LADproxy.bed"))
    print(f"het bins {int(m_het.sum())} | lad bins {int(m_lad.sum())}", flush=True)

    # ---------------- gene activity map (identical definition to 05a_atac_activity.py)
    cache = os.path.join(OUT, "act_map.npz")
    genes = pd.read_csv(os.path.join(L.ANN, "mm10_genes_tss.tsv"), sep="\t").reset_index(drop=True)
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        p2g = z["p2g"]
        gnames = list(z["genes"])
    else:
        ch = chrom
        st = start
        en = start + BIN
        p2g = np.full(len(keys), -1, dtype=np.int32)
        for c, sub in genes.groupby("chrom", observed=True):
            idx = np.where(ch == c)[0]
            if len(idx) == 0:
                continue
            tss = sub.tss.to_numpy()
            o = np.argsort(tss)
            tss_s = tss[o]
            gi = sub.index.to_numpy()[o]
            lo = np.searchsorted(tss_s, st[idx] - 2000, side="left")
            hi = np.searchsorted(tss_s, en[idx] + 2000, side="right")
            for k in np.where(hi > lo)[0]:
                a1, b1 = lo[k], hi[k]
                mid = (st[idx][k] + en[idx][k]) // 2
                j = a1 + int(np.argmin(np.abs(tss_s[a1:b1] - mid)))
                p2g[idx[k]] = gi[j]
        gnames = genes.gene.tolist()
        np.savez_compressed(cache, p2g=p2g, genes=np.array(gnames))
    print(f"gene activity map: {int((p2g>=0).sum())} bins -> genes", flush=True)

    het_genes = sorted(set(L.load_set("ConstitutiveHet")) & set(gnames))
    hidx = [gnames.index(g) for g in het_genes]
    G = sp.csr_matrix((np.ones((p2g >= 0).sum(), dtype=np.float32),
                       (p2g[p2g >= 0], np.where(p2g >= 0)[0])),
                      shape=(len(gnames), len(keys)))
    # split: het genes vs rest, so prom_het and prom_all can be rebuilt in chunks
    Gh = G[hidx]
    n_het_g = len(hidx)

    def prom_metrics(Xd, tot_d):
        act_h = np.asarray((Xd @ Gh.T).todense(), dtype=np.float32)
        act_h = np.log1p(act_h / np.maximum(tot_d, 1)[:, None] * 5000.0)
        prom_het = act_h.mean(1)
        # mean over all genes, chunked to avoid a 1.3 GB dense block
        acc = np.zeros(Xd.shape[0], dtype=np.float64)
        step = 2000
        for s in range(0, len(gnames), step):
            blk = np.asarray((Xd @ G[s:s + step].T).todense(), dtype=np.float32)
            blk = np.log1p(blk / np.maximum(tot_d, 1)[:, None] * 5000.0)
            acc += blk.sum(1)
        prom_all = acc / len(gnames)
        return prom_het, prom_all, prom_het - prom_all

    # ---------------- motif x promoter-bin matrix (same construction as 11_motif.py)
    mc = os.path.join(OUT, "msp.npz")
    tss = pd.read_csv(os.path.join(L.ANN, "mm10_genes_tss.tsv"), sep="\t")
    tss = tss[tss.gene.isin(set(het_genes))].copy()
    tss["bin"] = (tss.tss // BIN) * BIN
    tss["bin_key"] = tss.chrom.map({c: i for i, c in enumerate(CANON)}) * 1_000_000_000 + tss.bin
    prom = tss.dropna(subset=["bin_key"])
    prom = prom[prom.chrom.isin(CANON)]
    if os.path.exists(mc):
        z = np.load(mc, allow_pickle=True)
        Msp = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        mkeys = list(z["mkeys"])
        motifs = parse_pfm(PFM)
        print(f"cached Msp {Msp.shape}", flush=True)
    else:
        tb = TwoBit(GENOME)
        motifs = parse_pfm(PFM)
        mkeys = sorted(motifs)
        col_of = {int(k): i for i, k in enumerate(keys)}
        prom_seqs, prom_keys = [], []
        for _, r in prom.drop_duplicates("bin_key").iterrows():
            s = max(int(r.tss) - WIN // 2, 0)
            sq = tb.fetch(r.chrom, s, s + WIN)
            if len(sq) == WIN:
                prom_seqs.append(sq)
                prom_keys.append(int(r.bin_key))
        print(f"promoter bins scanned: {len(prom_seqs)}", flush=True)
        ph = score_motifs(encode(prom_seqs), motifs, mkeys, verbose=True)
        rr, cc = [], []
        for pi, bk in enumerate(prom_keys):
            ci = col_of.get(bk)
            if ci is None:
                continue
            js = np.where(ph[:, pi])[0]
            rr.extend([ci] * len(js))
            cc.extend(js.tolist())
        Msp = sp.csc_matrix((np.ones(len(rr), dtype=np.float32), (rr, cc)),
                            shape=(len(keys), len(mkeys)))
        Msp.sum_duplicates()
        np.savez_compressed(mc, data=Msp.data, indices=Msp.indices, indptr=Msp.indptr,
                            shape=np.array(Msp.shape), mkeys=np.array(mkeys))
        print(f"built Msp {Msp.shape}", flush=True)

    # JASPAR matrix IDs are not TF names and one TF can have several matrices,
    # so collapse to TF level (mean over its matrices) before any contrast.
    tf_of = {k: motifs[k][0] for k in mkeys}
    UNIQ = sorted(set(tf_of.values()))
    IDX = {u: [i for i, k in enumerate(mkeys) if tf_of[k] == u] for u in UNIQ}

    def tf_level(d):
        out = np.zeros((d.shape[0], len(UNIQ)), dtype=np.float32)
        for j, u in enumerate(UNIQ):
            out[:, j] = d[:, IDX[u]].mean(1)
        return out

    nS = np.asarray(Msp.sum(0)).ravel()

    def deviation(Xd, tot_d, col_tot_d):
        obs = (Xd @ Msp).toarray().astype(np.float32)
        binsum = np.asarray(Msp.T @ col_tot_d).ravel()
        grand = float(tot_d.sum())
        exp = np.outer(tot_d, binsum / grand)
        return (obs - exp) / np.sqrt(exp + 1.0)

    def dev_tf(Xd, tot_d, col_d):
        return tf_level(deviation(Xd, tot_d, col_d))

    # ---------------- baseline (no thinning) as a reference point
    col_tot = np.asarray(X.sum(0)).ravel()
    dev0 = dev_tf(X, row_tot, col_tot)
    mkeys = UNIQ
    print(f"baseline deviation {dev0.shape} in {time.time()-t0:.0f}s", flush=True)

    # sanity: reproduce the published per-lineage effects for two TFs
    def contrast(dev, tf):
        j = mkeys.index(tf)
        v = dev[:, j]
        out = {}
        for ln in LINEAGES:
            m = lin == ln
            mc_ = m & (grp == "Control")
            mw = m & (grp == "TCDD_16wk_WT")
            mk_ = m & (grp == "TCDD_16wk_AhrKO")
            out[ln] = (L.cohens_d(v[mw], v[mc_]), L.cohens_d(v[mk_], v[mw]),
                       float(v[mc_].mean()), float(v[mw].mean()), float(v[mk_].mean()))
        return out
    print("baseline Ahr::Arnt / KLF2 (d_wt_vs_ctrl, d_ako_vs_wt):", flush=True)
    for tf in ["Ahr::Arnt", "KLF2", "SMAD2", "ATF2"]:
        c = contrast(dev0, tf)
        print(f"  {tf:14s} EC {c['Endothelial'][0]:+.3f} {c['Endothelial'][1]:+.3f} "
              f"| SMC {c['Parenchyma'][0]:+.3f} {c['Parenchyma'][1]:+.3f}", flush=True)

    targets = {"match_ako": float(np.median(row_tot[grp == "TCDD_16wk_AhrKO"])),
               "stress_60pct": 0.6 * float(np.median(row_tot[grp == "TCDD_16wk_AhrKO"]))}

    rows, mrows, drows = [], [], []

    def add_effects(dev, tag, seed, Xd=None, tot_d=None):
        pm = None
        if Xd is not None:
            ph, pa, pr = prom_metrics(Xd, tot_d)
            pm = {"frac_het": np.asarray(Xd[:, m_het].sum(1)).ravel() / np.maximum(tot_d, 1),
                  "frac_lad": np.asarray(Xd[:, m_lad].sum(1)).ravel() / np.maximum(tot_d, 1),
                  "prom_het_rel": pr}
        for ln in LINEAGES:
            m = lin == ln
            mc_ = m & (grp == "Control")
            mw = m & (grp == "TCDD_16wk_WT")
            mk_ = m & (grp == "TCDD_16wk_AhrKO")
            if mw.sum() < 30 or mc_.sum() < 30:
                continue
            for tf in PANEL:
                if tf not in mkeys:
                    continue
                v = dev[:, mkeys.index(tf)]
                d12 = L.cohens_d(v[mw], v[mc_])
                d32 = L.cohens_d(v[mk_], v[mw])
                p12 = L.perm_p(v[mw], v[mc_], n=N_PERM, seed=seed + 1)
                p32 = L.perm_p(v[mk_], v[mw], n=N_PERM, seed=seed + 2)
                rows.append(dict(target=tag, seed=seed, lineage=ln, tf=tf,
                                 module=module_of(tf), d_wt_vs_ctrl=d12, p_wt_vs_ctrl=p12,
                                 d_ako_vs_wt=d32, p_ako_vs_wt=p32,
                                 mean_ctrl=float(v[mc_].mean()),
                                 mean_wt=float(v[mw].mean()),
                                 mean_ako=float(v[mk_].mean())))
            # modules: mean d over member motifs
            for mod in sorted(set(module_of(t) for t in mkeys) - {""}):
                js = [mkeys.index(t) for t in mkeys if module_of(t) == mod]
                vv = dev[:, js].mean(1)
                mrows.append(dict(target=tag, seed=seed, lineage=ln, module=mod,
                                  d_wt_vs_ctrl=L.cohens_d(vv[mw], vv[mc_]),
                                  d_ako_vs_wt=L.cohens_d(vv[mk_], vv[mw])))
            if pm is not None:
                for name, vec in pm.items():
                    drows.append(dict(target=tag, seed=seed, lineage=ln, metric=name,
                                      mean_ctrl=float(vec[mc_].mean()),
                                      mean_wt=float(vec[mw].mean()),
                                      mean_ako=float(vec[mk_].mean()),
                                      d_wt_vs_ctrl=L.cohens_d(vec[mw], vec[mc_]),
                                      d_ako_vs_wt=L.cohens_d(vec[mk_], vec[mw]),
                                      p_wt_vs_ctrl=L.perm_p(vec[mw], vec[mc_], n=N_PERM,
                                                            seed=seed + 3),
                                      p_ako_vs_wt=L.perm_p(vec[mk_], vec[mw], n=N_PERM,
                                                           seed=seed + 4)))
        print(f"  effects done: {tag} seed {seed} ({time.time()-t0:.0f}s)", flush=True)

    add_effects(dev0, "none", -1)
    _, _, pr0 = prom_metrics(X, row_tot)
    for ln in LINEAGES:
        m = lin == ln
        mc_ = m & (grp == "Control")
        mw = m & (grp == "TCDD_16wk_WT")
        mk_ = m & (grp == "TCDD_16wk_AhrKO")
        if mw.sum() < 30:
            continue
        for name, vec in (("frac_het", np.asarray(X[:, m_het].sum(1)).ravel() / row_tot),
                          ("frac_lad", np.asarray(X[:, m_lad].sum(1)).ravel() / row_tot),
                          ("prom_het_rel", pr0)):
            drows.append(dict(target="none", seed=-1, lineage=ln, metric=name,
                              mean_ctrl=float(vec[mc_].mean()), mean_wt=float(vec[mw].mean()),
                              mean_ako=float(vec[mk_].mean()),
                              d_wt_vs_ctrl=L.cohens_d(vec[mw], vec[mc_]),
                              d_ako_vs_wt=L.cohens_d(vec[mk_], vec[mw]),
                              p_wt_vs_ctrl=L.perm_p(vec[mw], vec[mc_], n=N_PERM, seed=5),
                              p_ako_vs_wt=L.perm_p(vec[mk_], vec[mw], n=N_PERM, seed=6)))

    for tag, T in targets.items():
        for seed in SEEDS:
            rng = np.random.default_rng(1000 + seed)
            p = np.clip(T / np.maximum(row_tot, 1), 0, 1)
            Xd = X.copy()
            Xd.data = rng.binomial(X.data.astype(np.int64), p.repeat(
                np.diff(X.indptr))).astype(np.float32)
            Xd.eliminate_zeros()
            tot_d = np.asarray(Xd.sum(1)).ravel()
            col_d = np.asarray(Xd.sum(0)).ravel()
            dev = dev_tf(Xd, tot_d, col_d)
            add_effects(dev, tag, seed, Xd=Xd, tot_d=tot_d)
            del Xd, dev

    eff = pd.DataFrame(rows)
    mods = pd.DataFrame(mrows)
    dom = pd.DataFrame(drows)
    eff.to_csv(os.path.join(OUT, "downsample_effects.csv"), index=False)
    mods.to_csv(os.path.join(OUT, "downsample_modules.csv"), index=False)
    dom.to_csv(os.path.join(OUT, "downsample_domains.csv"), index=False)

    # ---------------- verdicts
    def summary_tf(tf, ln, col):
        sub = eff[(eff.tf == tf) & (eff.lineage == ln)]
        base = sub[sub.target == "none"][col].iloc[0]
        return {t: dict(base=float(base),
                        mean=float(sub[sub.target == t][col].mean()),
                        min=float(sub[sub.target == t][col].min()),
                        max=float(sub[sub.target == t][col].max()))
                for t in targets}

    verdict = {}
    for tf in ["Ahr::Arnt", "KLF2", "SMAD2", "ATF2", "FOSL1::JUND", "Nfe2l2", "RELA"]:
        verdict[tf] = {"Endothelial": {t: summary_tf(tf, "Endothelial", t)[t] for t in targets}
                       for t in ["d_ako_vs_wt"]}
        verdict[tf] = {"EC_d_ako_vs_wt": summary_tf(tf, "Endothelial", "d_ako_vs_wt"),
                       "SMC_d_wt_vs_ctrl": summary_tf(tf, "Parenchyma", "d_wt_vs_ctrl")}

    sm = {"depth_targets": targets,
          "median_counts_per_library": depth.round(1).to_dict(),
          "median_counts_per_group": {g: float(np.median(row_tot[grp == g]))
                                      for g in sorted(set(grp))},
          "n_cells": {g: int((grp == g).sum()) for g in sorted(set(grp))},
          "seeds": SEEDS, "n_perm": N_PERM,
          "promoter_bins": int(len(prom_keys) if 'prom_keys' in dir() else 0),
          "n_motifs": len(mkeys),
          "verdict": verdict,
          "elapsed_s": round(time.time() - t0, 1)}
    json.dump(sm, open(os.path.join(OUT, "downsample_summary.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in sm.items() if k != "verdict"}, indent=2))
    for tf, v in verdict.items():
        print(tf, {k: {kk: round(vv["mean"], 3) for kk, vv in v["EC_d_ako_vs_wt"].items()}})
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
