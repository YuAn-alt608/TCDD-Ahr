"""33b - Replace the primer pairs that came out multi-hit.

33_primer_specificity.py found 2 of 42 pairs amplifying many genomic products:

  mm10  Iffo2 (tier B)   3,363 products, sizes from 51 bp up  -> a low-complexity
                         forward primer sitting in a repeat family
  hg38  Tiparp (tier C,  21 products of 87-89 bp               -> an Alu-like forward
        backup)                                                 primer

Shipping those would be a silent failure mode (a melt curve with three peaks and a Ct
that means nothing), so they are redesigned here and re-checked with the *same*
whole-genome in-silico PCR that caught them. The replacement is accepted only if it
yields exactly one product within 30 bp of the intended amplicon size; otherwise the
script walks to the next candidate and finally reports that the locus needs a manual
redesign.

How the new pair is generated
  * the template is the original amplicon +/-300 bp, taken from the local 2bit,
  * primer3 is asked to span the original amplicon (SEQUENCE_TARGET), strict chemistry
    first, then the relaxed set used by 27 for GC-island promoters,
  * candidates are tested one by one in order of primer3 penalty; in-silico PCR is the
    deciding vote, primer3 only proposes.

Outputs (results/chip/):
  qpcr_primers.csv          the failing rows are replaced in place, with
                            replaced=1 / old_fwd / old_rev / old_n_products kept
  qpcr_replacements.csv     one row per swap, old vs new, plus the re-check result
  ispcr_summary.json        updated ok / redesign counts
"""
import os, sys, json
import numpy as np
import pandas as pd
import primer3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_seq
import lib_core as L
import importlib.util as _u

_s = _u.spec_from_file_location("s33", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "33_primer_specificity.py"))
S33 = _u.module_from_spec(_s)
_s.loader.exec_module(S33)

OUT = os.path.join(L.RES, "chip")
GEN = {"hg38": os.path.join(L.ROOT, "data", "genome", "hg38.2bit"),
       "mm10": os.path.join(L.ROOT, "data", "genome", "mm10.2bit")}
FLANK = 300
STRICT = {
    "PRIMER_OPT_SIZE": 20, "PRIMER_MIN_SIZE": 19, "PRIMER_MAX_SIZE": 22,
    "PRIMER_OPT_TM": 60.0, "PRIMER_MIN_TM": 58.5, "PRIMER_MAX_TM": 61.5,
    "PRIMER_MIN_GC": 40.0, "PRIMER_MAX_GC": 62.0,
    "PRIMER_MAX_POLY_X": 4, "PRIMER_MAX_NS_ACCEPTED": 0,
    "PRIMER_PRODUCT_SIZE_RANGE": [[90, 160]],
    "PRIMER_NUM_RETURN": 6,
    "PRIMER_SALT_MONOVALENT": 50.0, "PRIMER_DNA_CONC": 250.0,
}
RELAX = dict(STRICT, **{"PRIMER_MIN_GC": 35.0, "PRIMER_MAX_GC": 70.0,
                        "PRIMER_MIN_TM": 57.0, "PRIMER_MAX_TM": 63.0,
                        "PRIMER_PRODUCT_SIZE_RANGE": [[90, 200]]})


def chroms_of(genome, tb):
    if genome == "hg38":
        want = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    else:
        want = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"]
    return [c for c in tb.idx if c in want]


def genomic_ispcr(tb, genome, fwd, rev):
    hits = []
    for chrom in chroms_of(genome, tb):
        seq = S33.chrom_bytes(tb, chrom)
        if len(seq) < 1000:
            continue
        hits += S33.ispcr_pair("x", {chrom: seq}, fwd, rev)
    return hits


def candidates(tb, genome, chrom, p_start, p_end, orig_size, tss):
    """Try progressively wider templates until primer3 returns pairs.

    A repeat-embedded promoter often has no unique 15-mer within the original
    amplicon, so the first pass keeps the amplicon (+-300 bp) and later passes are
    allowed to slide anywhere within +-800 / +-1500 bp of the TSS, still spanning
    the TSS itself. primer3 never decides specificity - it only proposes.
    """
    out = []
    tss = int(tss)
    plans = [(300, [p_start, p_end - p_start], STRICT),
             (300, [p_start, p_end - p_start], RELAX),
             (800, [tss - 50, 100], RELAX),
             (1500, [tss - 100, 200], RELAX)]
    for flank, tgt0, chem in plans:
        lo = max(0, min(p_start, tss) - flank)
        hi = max(p_end, tss) + flank
        dna = tb.fetch(chrom, lo, hi).upper()
        tgt = [tgt0[0] - lo, tgt0[1]]
        if tgt[0] < 0 or tgt[0] + tgt[1] > len(dna):
            continue
        res = primer3.design_primers(
            {"SEQUENCE_ID": "x", "SEQUENCE_TEMPLATE": dna, "SEQUENCE_TARGET": tgt}, chem)
        n = res.get("PRIMER_PAIR_NUM_RETURNED", 0)
        if not n:
            print(f"    flank {flank} tgt {tgt0}: {res.get('PRIMER_PAIR_EXPLAIN','')}",
                  flush=True)
            continue
        for i in range(n):
            out.append({"fwd": res[f"PRIMER_LEFT_{i}_SEQUENCE"],
                        "rev": res[f"PRIMER_RIGHT_{i}_SEQUENCE"],
                        "size": int(res[f"PRIMER_PAIR_{i}_PRODUCT_SIZE"]),
                        "tm_f": float(res[f"PRIMER_LEFT_{i}_TM"]),
                        "tm_r": float(res[f"PRIMER_RIGHT_{i}_TM"]),
                        "gc_f": float(res[f"PRIMER_LEFT_{i}_GC_PERCENT"]),
                        "gc_r": float(res[f"PRIMER_RIGHT_{i}_GC_PERCENT"]),
                        "penalty": float(res[f"PRIMER_PAIR_{i}_PENALTY"]),
                        "chem": "strict" if chem is STRICT else "relaxed",
                        "flank": flank,
                        "start": lo + int(res[f"PRIMER_LEFT_{i}"][0])})
        if len(out) >= 4:
            break
    out.sort(key=lambda d: d["penalty"])
    seen, uniq = set(), []
    for d in out:
        if d["fwd"] in seen:
            continue
        seen.add(d["fwd"])
        uniq.append(d)
    return uniq


def main():
    qp = pd.read_csv(os.path.join(OUT, "qpcr_primers.csv"))
    bad = qp[qp.verdict == "redesign"]
    print(f"{len(bad)} pair(s) to replace", flush=True)
    rows = []
    tbs = {}
    for _, r in bad.iterrows():
        g = r.genome
        if g not in tbs:
            tbs[g] = lib_seq.TwoBit(GEN[g])
        tb = tbs[g]
        cand = candidates(tb, g, r.chrom, int(r.product_start), int(r.product_end),
                          int(r["size"]), int(r.tss))
        print(f"  {g} {r.mouse if g=='mm10' else r.human} ({r.tier}): "
              f"{len(cand)} candidates, old {int(r.n_products)} products", flush=True)
        picked = None
        for k, c in enumerate(cand[:5]):
            hits = genomic_ispcr(tb, g, c["fwd"], c["rev"])
            ok = [h for h in hits if S33.MIN_PRODUCT <= h["size"] <= S33.MAX_PRODUCT]
            print(f"    cand{k}: {len(ok)} product(s) {[h['size'] for h in ok][:4]}",
                  flush=True)
            if len(ok) == 1 and abs(ok[0]["size"] - int(r["size"])) <= 30:
                picked = dict(c, n_products=1, size=ok[0]["size"],
                              product_start=int(ok[0]["start"]),
                              product_end=int(ok[0]["end"]))
                break
        if picked is None:
            print("    no candidate passed - locus needs manual redesign", flush=True)
            continue
        rows.append({"genome": g,
                     "gene": r.mouse if g == "mm10" else r.human,
                     "tier": r.tier, "chrom": r.chrom,
                     "old_fwd": r.fwd, "old_rev": r.rev,
                     "old_size": int(r["size"]), "old_n_products": int(r.n_products),
                     "new_fwd": picked["fwd"], "new_rev": picked["rev"],
                     "new_size": picked["size"], "new_n_products": 1,
                     "tm_f": picked["tm_f"], "tm_r": picked["tm_r"],
                     "gc_f": picked["gc_f"], "gc_r": picked["gc_r"],
                     "chem": picked["chem"]})
        m = (qp.genome == g) & (qp.fwd == r.fwd) & (qp.rev == r.rev)
        qp.loc[m, ["fwd", "rev", "size", "product_start", "product_end"]] = [
            picked["fwd"], picked["rev"], picked["size"],
            picked["product_start"], picked["product_end"]]
        qp.loc[m, "tm_f"] = picked["tm_f"]
        qp.loc[m, "tm_r"] = picked["tm_r"]
        qp.loc[m, "gc_f"] = picked["gc_f"]
        qp.loc[m, "gc_r"] = picked["gc_r"]
        qp.loc[m, "n_products"] = 1
        qp.loc[m, "sizes"] = str(picked["size"])
        qp.loc[m, "specificity"] = ("unique genomic product (re-checked whole genome; "
                                    "replaced after multi-hit)")
        qp.loc[m, "verdict"] = "ok"
        qp.loc[m, "replaced"] = 1
        qp.loc[m, "old_fwd"] = r.fwd
        qp.loc[m, "old_rev"] = r.rev
        qp.loc[m, "old_n_products"] = int(r.n_products)
        print(f"    -> replaced, single product {picked['size']} bp", flush=True)

    rp = pd.DataFrame(rows)
    rp.to_csv(os.path.join(OUT, "qpcr_replacements.csv"), index=False)
    qp.to_csv(os.path.join(OUT, "qpcr_primers.csv"), index=False)
    sm = json.load(open(os.path.join(OUT, "ispcr_summary.json")))
    vc = qp.verdict.value_counts()
    sm.update({"ok": int(vc.get("ok", 0)), "review": int(vc.get("review", 0)),
               "redesign": int(vc.get("redesign", 0)),
               "replaced_after_multi_hit": int(len(rp))})
    json.dump(sm, open(os.path.join(OUT, "ispcr_summary.json"), "w"), indent=2)
    print("\n=== final verdicts ===")
    print(qp.groupby(["genome", "verdict"]).size().to_string())
    print(json.dumps(sm, indent=2))


if __name__ == "__main__":
    main()
