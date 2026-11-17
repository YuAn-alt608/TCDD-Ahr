"""Which motifs sit on the heterochromatin-gene promoters that actually open?

The genome-wide DA+enrichment pass (11_motif.py) answers "what is enriched in the bins that
open". This narrows it to the promoters that matter for the story: the 2,816 bins that carry
the TSS of a constitutive-heterochromatin / H3K9me3 gene. For every motif we ask whether the
promoters *carrying that motif* gained accessibility, in each lineage and each arm (TCDD vs
control, AhrKO vs WT). A random-bin-set null of the same size absorbs the global accessibility
shift between ATAC samples, which is the main artefact here (one TCDD library).

Outputs: results/motif/het_promoter_motifs.csv, het_promoter_motif_summary.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L
import importlib.util as _u

_spec = _u.spec_from_file_location("m11", os.path.join(os.path.dirname(os.path.abspath(__file__)), "11_motif.py"))
M11 = _u.module_from_spec(_spec)
_spec.loader.exec_module(M11)

OUT = M11.OUT
CANON = M11.CANON
WIN = 1000          # wider window for promoters, centred on the TSS
N_NULL = 1000

MODULE = {
    "NFE2L2": "NRF2", "MAFG": "NRF2", "MAFK": "NRF2", "MAFF": "NRF2", "NFE2": "NRF2",
    "MAF": "NRF2", "BACH1": "NRF2-antagonist", "BACH2": "NRF2-antagonist",
    "RELA": "NF-kB", "RELB": "NF-kB", "REL": "NF-kB", "NFKB1": "NF-kB", "NFKB2": "NF-kB",
    "JUN": "AP-1", "JUNB": "AP-1", "JUND": "AP-1", "FOS": "AP-1", "FOSB": "AP-1",
    "FOSL1": "AP-1", "FOSL2": "AP-1", "BATF": "AP-1", "ATF2": "AP-1", "ATF3": "AP-1",
    "HIF1A": "HIF", "EPAS1": "HIF", "ARNT": "HIF/AhR", "ARNT2": "HIF/AhR", "AHR": "AhR",
    "STAT1": "IFN", "STAT2": "IFN", "IRF1": "IFN", "IRF3": "IFN", "IRF7": "IFN", "IRF9": "IFN",
    "CEBPA": "SASP", "CEBPB": "SASP", "CEBPD": "SASP", "ATF4": "ISR", "DDIT3": "ISR", "XBP1": "ISR",
    "SMAD2": "TGFb", "SMAD3": "TGFb", "SMAD4": "TGFb", "SMAD5": "TGFb",
    "TEAD1": "Hippo", "TEAD2": "Hippo", "TEAD3": "Hippo", "TEAD4": "Hippo",
    "KLF2": "shear", "KLF4": "shear", "TP53": "senescence", "MYC": "prolif",
    "E2F1": "prolif", "PPARG": "adipo", "NR1H3": "LXR", "SRF": "SMC", "MYOCD": "SMC",
}


def module_of(tf):
    for tok in str(tf).split("::"):
        m = MODULE.get(tok.upper())
        if m:
            return m
    return ""


def main():
    t0 = time.time()
    a = sc.read_h5ad(os.path.join(L.RES, "ATAC", "atac_processed.h5ad"))
    keys_i = np.asarray(a.var_names, dtype=np.int64)
    chrom, start = M11.decode_keys(keys_i)
    ct = a.obs["celltype_atac"].astype(str).to_numpy()
    lin = a.obs["lineage"].astype(str).to_numpy()
    grp = a.obs["group"].astype(str).to_numpy()
    acc = (a.X > 0).astype(np.float32)

    genes = set()
    for s in ["ConstitutiveHet", "H3K9me3"]:
        genes |= set(L.load_set(s))
    tss = pd.read_csv(os.path.join(L.ANN, "mm10_genes_tss.tsv"), sep="\t")
    tss = tss[tss.gene.isin(genes) & tss.chrom.isin(CANON)].copy()
    tss["bin_key"] = tss.chrom.map({c: i for i, c in enumerate(CANON)}) * 1_000_000_000 + (tss.tss // 5000) * 5000
    tss = tss.drop_duplicates("bin_key")
    print(f"het promoter bins: {len(tss)}", flush=True)

    pos = pd.Series(np.arange(len(keys_i)), index=keys_i)
    pidx = pos.reindex(tss.bin_key.to_numpy()).dropna().astype(int).to_numpy()
    tss = tss.loc[pos.reindex(tss.bin_key.to_numpy()).notna().to_numpy()].reset_index(drop=True)
    print(f"  present in matrix: {len(pidx)}", flush=True)

    motifs = M11.parse_pfm(M11.PFM)
    mkeys = sorted(motifs)
    seqs, keep_rows = [], []
    for i, r in tss.iterrows():
        s = max(int(r.tss) - WIN // 2, 0)
        sq = M11.seq_of(r.chrom, s, s + WIN)
        if len(sq) == WIN:
            seqs.append(sq)
            keep_rows.append(i)
    tss = tss.loc[keep_rows].reset_index(drop=True)
    pidx = pidx[np.isin(np.arange(len(pidx)), keep_rows)]
    hits = M11.score_motifs(M11.encode(seqs), motifs, mkeys).astype(np.float32)  # (motif, bin)
    print(f"  scored {len(seqs)} promoters x {len(mkeys)} motifs in {time.time()-t0:.0f}s", flush=True)

    F = acc[:, pidx]                       # (cells, n_prom)
    rng = np.random.default_rng(0)
    rows = []
    for lineage in ["Parenchyma", "Endothelial", "Mesenchymal", "Immune"]:
        m_ct = lin == lineage
        masks = {"TCDD_vs_ctrl": (m_ct & (grp == "Control"), m_ct & (grp == "TCDD_16wk_WT")),
                 "AhrKO_vs_WT": (m_ct & (grp == "TCDD_16wk_WT"), m_ct & (grp == "TCDD_16wk_AhrKO"))}
        for contrast, (mA, mB) in masks.items():
            if mA.sum() < 30 or mB.sum() < 30:
                continue
            fa = np.asarray(F[mA].mean(0)).ravel()
            fb = np.asarray(F[mB].mean(0)).ravel()
            delta = fb - fa
            for mi, k in enumerate(mkeys):
                sel = hits[mi] > 0
                n = int(sel.sum())
                if n < 10:
                    continue
                obs = float(delta[sel].mean())
                null = np.array([delta[rng.choice(len(delta), n, replace=False)].mean()
                                 for _ in range(N_NULL)])
                p = float((np.abs(null) >= abs(obs)).mean())
                rows.append({"lineage": lineage, "contrast": contrast, "motif": k,
                             "tf": motifs[k][0], "module": module_of(motifs[k][0]),
                             "n_promoters": n, "delta_open_frac": obs,
                             "null_sd": float(null.std()), "p_perm": p})
        print(f"  {lineage} done ({time.time()-t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    res["fdr"] = np.nan
    for (l, c), sub in res.groupby(["lineage", "contrast"]):
        res.loc[sub.index, "fdr"] = L.bh(sub.p_perm.to_numpy())
    res.to_csv(os.path.join(OUT, "het_promoter_motifs.csv"), index=False)

    # module-level summary
    if len(res):
        mod = (res[res.contrast == "TCDD_vs_ctrl"]
               .groupby(["module", "lineage"]).delta_open_frac.mean().unstack())
        mod.to_csv(os.path.join(OUT, "het_promoter_module_delta.csv"))

    summary = {
        "n_promoters": int(len(seqs)),
        "window_bp": WIN,
        "n_motifs": len(mkeys),
        "n_null": N_NULL,
        "elapsed_s": round(time.time() - t0, 1),
    }
    json.dump(summary, open(os.path.join(OUT, "het_promoter_motif_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    for l in ["Parenchyma", "Endothelial"]:
        s = res[(res.lineage == l) & (res.contrast == "TCDD_vs_ctrl")].nsmallest(10, "p_perm")
        print(f"\n{l} TCDD vs ctrl (het promoters):")
        print(s[["tf", "module", "n_promoters", "delta_open_frac", "p_perm", "fdr"]].round(4).to_string())


if __name__ == "__main__":
    main()
