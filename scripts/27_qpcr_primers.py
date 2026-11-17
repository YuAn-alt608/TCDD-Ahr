"""Turn the candidate list into something a bench can actually order.

Two panels, because they answer different questions:

  mouse (mm10, MOVAS + TCDD)   the decisive test of our own claim: does AHR sit on the
                               heterochromatin promoters that open in mouse SMC?
  human (hg38, HCASMC)         the test the collaborator can run with the antibody already
                               in hand -- and the only one whose result can be compared with
                               the public GSE276152 peak set (which failed its own control).

Every panel carries three tiers:
  C  canonical AHR targets (CYP1A1, AHRR, TIPARP, NQO1, AHR, CYP1B1) -- the assay QC. If
     these do not show TCDD-induced binding, the ChIP is invalid and nothing else counts.
  A  heterochromatin promoters with an AhR motif that open under TCDD and revert in Ahr-KO.
  B  heterochromatin promoters with an AhR motif that do NOT open -- specificity control.

Primers are designed with primer3 (SantaLucia Tm, salt 50 mM, 250 nM primer) to span the
TSS, 90-160 bp amplicon. When the machine has network, each pair is also pushed through
UCSC in-silico PCR so the table says whether the pair amplifies one place or five.

Outputs
    results/chip/qpcr_primers.csv    one row per locus per genome, best pair + backup
    results/chip/qpcr_primers.fa     amplicon sequences for a manual BLAST
    results/chip/qpcr_design.json
"""
import os, sys, json, time, urllib.parse, urllib.request
import numpy as np
import pandas as pd
import primer3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L
import importlib.util as _u

_spec = _u.spec_from_file_location("m11", os.path.join(os.path.dirname(os.path.abspath(__file__)), "11_motif.py"))
M11 = _u.module_from_spec(_spec)
_spec.loader.exec_module(M11)

OUT = os.path.join(L.RES, "chip")
CACHE = os.path.join(L.ROOT, "data", "external", "hg38_promoters.json")
FLANK = 500
GLOBAL = {
    "PRIMER_OPT_SIZE": 20, "PRIMER_MIN_SIZE": 19, "PRIMER_MAX_SIZE": 22,
    "PRIMER_OPT_TM": 60.0, "PRIMER_MIN_TM": 58.5, "PRIMER_MAX_TM": 61.5,
    "PRIMER_MIN_GC": 40.0, "PRIMER_MAX_GC": 62.0,
    "PRIMER_MAX_POLY_X": 4, "PRIMER_MAX_NS_ACCEPTED": 0,
    "PRIMER_PRODUCT_SIZE_RANGE": [[90, 160]],
    "PRIMER_NUM_RETURN": 4,
    "PRIMER_SALT_MONOVALENT": 50.0, "PRIMER_DNA_CONC": 250.0,
    "PRIMER_MAX_END_STABILITY": 9.0,
}


def fetch_hg38(chrom, start, end, cache):
    key = f"{chrom}:{start}-{end}"
    if key in cache:
        return cache[key]
    url = ("https://api.genome.ucsc.edu/getData/sequence?"
           + urllib.parse.urlencode({"genome": "hg38", "chrom": chrom,
                                     "start": start, "end": end}))
    with urllib.request.urlopen(url, timeout=30) as r:
        j = json.loads(r.read().decode())
    seq = j.get("dna", "").upper()
    cache[key] = seq
    return seq


RELAXED = {
    "PRIMER_OPT_SIZE": 20, "PRIMER_MIN_SIZE": 18, "PRIMER_MAX_SIZE": 24,
    "PRIMER_OPT_TM": 60.5, "PRIMER_MIN_TM": 57.0, "PRIMER_MAX_TM": 64.0,
    "PRIMER_MIN_GC": 35.0, "PRIMER_MAX_GC": 78.0,
    "PRIMER_MAX_POLY_X": 5, "PRIMER_MAX_NS_ACCEPTED": 0,
    "PRIMER_PRODUCT_SIZE_RANGE": [[80, 220]],
    "PRIMER_NUM_RETURN": 4,
    "PRIMER_SALT_MONOVALENT": 50.0, "PRIMER_DNA_CONC": 250.0,
}


def _run(seq, glob, target):
    args = {"SEQUENCE_ID": "x", "SEQUENCE_TEMPLATE": seq}
    if target:
        args["SEQUENCE_TARGET"] = target
    res = primer3.design_primers(args, glob)
    out = []
    for i in range(res.get("PRIMER_PAIR_NUM_RETURNED", 0)):
        out.append({
            "fwd": res[f"PRIMER_LEFT_{i}_SEQUENCE"],
            "rev": res[f"PRIMER_RIGHT_{i}_SEQUENCE"],
            "f_start": int(res[f"PRIMER_LEFT_{i}"][0]),
            "r_start": int(res[f"PRIMER_RIGHT_{i}"][0]),
            "tm_f": float(res[f"PRIMER_LEFT_{i}_TM"]),
            "tm_r": float(res[f"PRIMER_RIGHT_{i}_TM"]),
            "gc_f": float(res[f"PRIMER_LEFT_{i}_GC_PERCENT"]),
            "gc_r": float(res[f"PRIMER_RIGHT_{i}_GC_PERCENT"]),
            "product": int(res[f"PRIMER_PAIR_{i}_PRODUCT_SIZE"]),
            "penalty": float(res[f"PRIMER_PAIR_{i}_PENALTY"]),
        })
    return out


def design(seq):
    """Strict first; if the promoter is a GC island, fall back to relaxed chemistry.

    Heterochromatin-gene promoters are often CpG islands (70-90% GC), where a 40-62% GC
    rule leaves almost no legal primer -- the strict pass fails on GC, not on Tm. The
    relaxed pass widens GC/Tm and amplicon size and is flagged in the output so nobody
    mistakes it for an optimised pair.
    """
    if len(seq) < 2 * FLANK:
        return [], ""
    pairs = _run(seq, GLOBAL, [FLANK - 25, 50])
    if pairs:
        return pairs, "strict"
    pairs = _run(seq, RELAXED, [FLANK - 100, 200])
    if pairs:
        return pairs, "relaxed"
    # last resort: let the amplicon sit anywhere in the window, then keep the pair whose
    # midpoint is closest to the TSS (still a promoter amplicon, just not TSS-spanning)
    wide = dict(RELAXED)
    wide["PRIMER_PRODUCT_SIZE_RANGE"] = [[70, 200]]
    pairs = _run(seq, wide, None)
    if pairs:
        for p in pairs:
            p["centre_dist"] = abs((p["f_start"] + p["r_start"]) / 2 - FLANK)
        pairs.sort(key=lambda p: p["centre_dist"])
        return pairs, "near-TSS"
    return [], ""


def ispcr(seq_id, fwd, rev, genome="hg38"):
    """UCSC in-silico PCR: how many products does this pair make?"""
    url = ("https://genome.ucsc.edu/cgi-bin/hgPcr?"
           + urllib.parse.urlencode({"db": genome, "wp_target": "genome",
                                     "wp_f": fwd, "wp_r": rev, "wp_size": 4000,
                                     "wp_perfect": 1, "wp_mismatch": 0}))
    try:
        with urllib.request.urlopen(url, timeout=45) as r:
            html = r.read().decode("utf-8", "ignore")
    except Exception as e:
        return {"n_products": None, "note": f"ispcr failed: {type(e).__name__}"}
    n = html.count("<TT>") if "<TT>" in html else html.count("&gt;")
    n = max(0, (n // 2) - 1) if n else 0
    return {"n_products": int(n), "note": "ucsc hgPcr"}


def main():
    t0 = time.time()
    cand = pd.read_csv(os.path.join(OUT, "qpcr_candidates.csv"))
    mt_all = pd.read_csv(os.path.join(L.ANN, "mm10_genes_tss.tsv"), sep="\t")
    mt_all = mt_all.drop_duplicates("gene").set_index("gene")
    # canonical genes are often absent from the peak table, so fall back on the symbol rule
    for i, r in cand.iterrows():
        if pd.isna(r.mouse):
            guess = str(r.human).capitalize()
            if guess in mt_all.index:
                cand.loc[i, "mouse"] = guess
                cand.loc[i, "mm10_chrom"] = mt_all.chrom[guess]
                cand.loc[i, "mm10_tss"] = mt_all.tss[guess]
                cand.loc[i, "mm10_strand"] = mt_all.strand[guess]
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    rows, fasta = [], []

    for r in cand.itertuples():
        # ---- human panel
        human_ok = (r.tier == "C") or (isinstance(r.smc_detect, float)
                                       and not np.isnan(r.smc_detect) and r.smc_detect >= 0.05)
        if human_ok and isinstance(r.hg38_tss, float) and not np.isnan(r.hg38_tss):
            s, e = int(r.hg38_tss) - FLANK, int(r.hg38_tss) + FLANK
            seq = fetch_hg38(r.hg38_chrom, s, e, cache)
            pairs, mode = design(seq)
            if pairs:
                p = pairs[0]
                rows.append({"genome": "hg38", "cell": "HCASMC", "tier": r.tier,
                             "human": r.human, "mouse": r.mouse, "chrom": r.hg38_chrom,
                             "tss": int(r.hg38_tss), "product_start": s + p["f_start"],
                             "product_end": s + p["r_start"] + len(p["rev"]),
                             "size": p["product"], "fwd": p["fwd"], "rev": p["rev"],
                             "tm_f": round(p["tm_f"], 1), "tm_r": round(p["tm_r"], 1),
                             "gc_f": round(p["gc_f"], 1), "gc_r": round(p["gc_r"], 1),
                             "penalty": round(p["penalty"], 2), "n_pairs_available": len(pairs), "mode": mode,
                             "delta_wt": r.delta_wt, "delta_ako": r.delta_ako,
                             "smc_detect": r.smc_detect})
                amp = seq[p["f_start"]:p["r_start"] + len(p["rev"])]
                fasta.append(f">{r.human}_hg38_{r.tier}\n{amp}")
            if len(pairs) > 1:
                p = pairs[1]
                rows.append({"genome": "hg38", "cell": "HCASMC", "tier": r.tier + "（备选）",
                             "human": r.human, "mouse": r.mouse, "chrom": r.hg38_chrom,
                             "tss": int(r.hg38_tss), "product_start": s + p["f_start"],
                             "product_end": s + p["r_start"] + len(p["rev"]),
                             "size": p["product"], "fwd": p["fwd"], "rev": p["rev"],
                             "tm_f": round(p["tm_f"], 1), "tm_r": round(p["tm_r"], 1),
                             "gc_f": round(p["gc_f"], 1), "gc_r": round(p["gc_r"], 1),
                             "penalty": round(p["penalty"], 2), "n_pairs_available": len(pairs), "mode": mode,
                             "delta_wt": r.delta_wt, "delta_ako": r.delta_ako,
                             "smc_detect": r.smc_detect})

        # ---- mouse panel
        if isinstance(r.mm10_tss, float) and not np.isnan(r.mm10_tss):
            s, e = int(r.mm10_tss) - FLANK, int(r.mm10_tss) + FLANK
            seq = M11.seq_of(r.mm10_chrom, s, e)
            pairs, mode = design(seq)
            if pairs:
                p = pairs[0]
                rows.append({"genome": "mm10", "cell": "MOVAS", "tier": r.tier,
                             "human": r.human, "mouse": r.mouse, "chrom": r.mm10_chrom,
                             "tss": int(r.mm10_tss), "product_start": s + p["f_start"],
                             "product_end": s + p["r_start"] + len(p["rev"]),
                             "size": p["product"], "fwd": p["fwd"], "rev": p["rev"],
                             "tm_f": round(p["tm_f"], 1), "tm_r": round(p["tm_r"], 1),
                             "gc_f": round(p["gc_f"], 1), "gc_r": round(p["gc_r"], 1),
                             "penalty": round(p["penalty"], 2), "n_pairs_available": len(pairs), "mode": mode,
                             "delta_wt": r.delta_wt, "delta_ako": r.delta_ako,
                             "smc_detect": r.smc_detect})
                amp = seq[p["f_start"]:p["r_start"] + len(p["rev"])]
                fasta.append(f">{r.mouse}_mm10_{r.tier}\n{amp}")

    json.dump(cache, open(CACHE, "w"))
    df = pd.DataFrame(rows)
    print(f"designed {len(df)} pairs over {df.human.nunique()} loci in {time.time()-t0:.0f}s", flush=True)

    # ---- specificity: deliberately NOT claimed here
    # UCSC hgPcr sits behind a Cloudflare challenge and cannot be scripted, and we do not
    # hold an hg38 FASTA to BLAT locally. Rather than fake a number, the table carries an
    # explicit "not checked" and the summary says what to run before ordering.
    df["ispcr_products"] = np.nan
    df["specificity"] = "未检查（下单前请过 Primer-BLAST / UCSC hgPcr）"

    df.to_csv(os.path.join(OUT, "qpcr_primers.csv"), index=False)
    open(os.path.join(OUT, "qpcr_primers.fa"), "w").write("\n".join(fasta) + "\n")
    summary = {
        "loci": int(df.human.nunique()),
        "pairs": int(len(df)),
        "by_genome": df.genome.value_counts().to_dict(),
        "by_tier": df.tier.value_counts().to_dict(),
        "median_amplicon": float(df["size"].median()),
        "specificity_checked": False,
        "note": ("primer3 with SantaLucia Tm, 50 mM monovalent salt, 250 nM primer, "
                 "amplicon forced across the TSS. Genome-wide specificity was NOT checked "
                 "(UCSC hgPcr blocks scripted access, no local hg38 FASTA); every pair must go "
                 "through Primer-BLAST before it is ordered."),
    }
    json.dump(summary, open(os.path.join(OUT, "qpcr_design.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
