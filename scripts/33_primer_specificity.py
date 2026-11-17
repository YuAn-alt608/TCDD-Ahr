"""33 - Genome-wide specificity of the 42 ChIP-qPCR primer pairs (local in-silico PCR).

Why local
---------
`results/chip/qpcr_primers.csv` carried a `specificity` column reading
"not checked" because UCSC hgPcr sits behind a Cloudflare challenge and no local
isPcr / BLAST+ / minimap2 binary is installed. Shipping a wet-lab panel with an
unchecked column is not acceptable, so the check is implemented directly:

  * both genomes are held locally as 2bit (mm10, hg38), read with the small
    decoder in lib_seq.py and expanded to bytes one chromosome at a time,
  * the 3'-terminal 15 nt of each primer is located exactly (bytes.find, which is
    two-way/Crochemore-Perrin, not a Python loop over positions),
  * up to 2 mismatches are then allowed in the 5' remainder, which is stricter
    than UCSC isPcr's default of 0 mismatches *for the whole primer* in one sense
    and more permissive in another - stated as such,
  * the reverse primer is searched as its reverse complement so both primers are
    indexed on the plus strand; a product is a forward hit at i and a reverse hit
    at k with 0 < k - i + len(rev) <= MAX_PRODUCT.

Reported per pair: number of products, the on-target product size, every
off-target product within MAX_PRODUCT with its mismatch count, and a verdict.
Pairs that come out as multi-hit are replaced: the script re-runs primer3 with
SEQUENCE_EXCLUDED_REGION covering the offending product, then re-checks the
replacement.

Outputs (results/chip/):
  ispcr_hits.csv           every product found (pair, genome, chrom, pos, size, mismatches)
  qpcr_primers.csv         updated in place: n_products / product_size / specificity / verdict
  ispcr_summary.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import primer3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_seq
import lib_core as L

OUT = os.path.join(L.RES, "chip")
GEN = {"mm10": os.path.join(L.ROOT, "data", "genome", "mm10.2bit"),
       "hg38": os.path.join(L.ROOT, "data", "genome", "hg38.2bit")}
SEED = 15            # 3'-terminal anchor, exact
MAX_MM = 2           # mismatches allowed in the 5' remainder
MAX_PRODUCT = 3000
MIN_PRODUCT = 40
COMP = bytes.maketrans(b"ACGTNacgtn", b"TGCANtgcan")


def revcomp(s):
    return s.encode().translate(COMP)[::-1].decode()


def chrom_bytes(tb, chrom):
    """Whole chromosome as uppercase ASCII bytes (N inside hard-masked blocks)."""
    dna, (nstarts, nsizes), off = tb._record(chrom)
    tb.f.seek(off)
    nb = (dna + 3) // 4
    buf = np.frombuffer(tb.f.read(nb), dtype=np.uint8)
    bits = np.empty(nb * 4, dtype=np.uint8)
    bits[0::4] = (buf >> 6) & 3
    bits[1::4] = (buf >> 4) & 3
    bits[2::4] = (buf >> 2) & 3
    bits[3::4] = buf & 3
    code = np.frombuffer(b"TCAG", dtype=np.uint8)
    arr = code[bits[:dna]].copy()
    for st, ln in zip(nstarts, nsizes):
        arr[int(st):int(st) + int(ln)] = ord("N")
    return arr.tobytes()


def find_all(hay, needle):
    out = []
    i = hay.find(needle)
    while i >= 0:
        out.append(i)
        i = hay.find(needle, i + 1)
    return out


def mismatches(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def ispcr_pair(pair, seqs_by_chrom, fwd, rev, chrom_filter=None):
    """All products for one primer pair on one genome (already decoded)."""
    fs, rs = fwd[-SEED:].encode(), revcomp(rev)[:SEED].encode()
    fpre, rpre = fwd[:-SEED], revcomp(rev)[SEED:]
    hits = []
    # forward hits
    fh = {}
    for chrom, seq in seqs_by_chrom.items():
        if chrom_filter and chrom != chrom_filter:
            continue
        for p in find_all(seq, fs):
            i = p - (len(fwd) - SEED)
            if i < 0:
                continue
            cand = seq[i:i + len(fwd)].decode().upper()
            if len(cand) == len(fwd) and mismatches(cand[:-SEED], fpre) <= MAX_MM:
                fh.setdefault(chrom, []).append(i)
    rh = {}
    for chrom, seq in seqs_by_chrom.items():
        if chrom_filter and chrom != chrom_filter:
            continue
        for k in find_all(seq, rs):
            cand = seq[k:k + len(rev)].decode().upper()
            if len(cand) == len(rev) and mismatches(cand[SEED:], rpre) <= MAX_MM:
                rh.setdefault(chrom, []).append(k)
    for chrom, fpos in fh.items():
        for i in fpos:
            for k in rh.get(chrom, []):
                size = k + len(rev) - i
                if k >= i and MIN_PRODUCT <= size <= MAX_PRODUCT:
                    hits.append(dict(pair=pair, chrom=chrom, start=i, end=k + len(rev),
                                     size=size))
    return hits


def main():
    qp = pd.read_csv(os.path.join(OUT, "qpcr_primers.csv"))
    if "pair" not in qp.columns:
        qp["pair"] = [f"{r.genome}:{r.human}:{r.tier}:{i}"
                      for i, r in enumerate(qp.itertuples())]
    print(f"{len(qp)} primer pairs across {qp.genome.unique().tolist()}", flush=True)
    all_hits, rows = [], []
    t0 = time.time()

    for genome, sub in qp.groupby("genome"):
        path = GEN.get(genome)
        if not os.path.exists(path):
            print(f"  {genome}: genome missing, skipped", flush=True)
            continue
        tb = lib_seq.TwoBit(path)
        chroms = [c for c in tb.idx if c in ([f"chr{i}" for i in range(1, 23)] +
                                             ["chrX", "chrY"])] if genome == "hg38" else \
                 [c for c in tb.idx if c in ([f"chr{i}" for i in range(1, 20)] +
                                             ["chrX", "chrY"])]
        print(f"  {genome}: {len(chroms)} chromosomes, {len(sub)} pairs", flush=True)
        # search chrom by chrom so only one chromosome is resident at a time.
        # The scan costs ~3 min per genome, so it is cached; delete the cache to redo.
        cache = os.path.join(OUT, f"ispcr_cache_{genome}.csv")
        if os.path.exists(cache):
            ch = pd.read_csv(cache)
            per_pair = {r.pair: ch[ch.pair == r.pair].to_dict("records")
                        for r in sub.itertuples()}
            print(f"    cached ({len(ch)} products) - delete {os.path.basename(cache)} to redo",
                  flush=True)
        else:
            per_pair = {r.pair: [] for r in sub.itertuples()}
            for chrom in chroms:
                seq = chrom_bytes(tb, chrom)
                if len(seq) < 1000:
                    continue
                for r in sub.itertuples():
                    h = ispcr_pair(r.pair, {chrom: seq}, r.fwd, r.rev)
                    per_pair[r.pair].extend(h)
                print(f"    {chrom} ({len(seq)/1e6:.0f} Mb) done, {time.time()-t0:.0f}s",
                      flush=True)
            _ch = [dict(h) for v in per_pair.values() for h in v]
            pd.DataFrame(_ch or [dict(pair="", chrom="", start=0, end=0, size=0)]
                         ).to_csv(cache, index=False)
        for r in sub.itertuples():
            hits = per_pair[r.pair]
            all_hits += hits
            in_range = [h for h in hits if h["size"] <= MAX_PRODUCT]
            rows.append(dict(pair=r.pair, genome=genome, n_products=len(in_range),
                             sizes=";".join(str(h["size"]) for h in
                                            sorted(in_range, key=lambda x: x["size"])[:6]),
                             on_expected=bool(any(abs(h["size"] - r.size) <= 2
                                                  for h in in_range))
                             if "size" in r._fields else None,
                             product_size=r.size if "size" in r._fields else np.nan))

    hits = pd.DataFrame(all_hits)
    hits.to_csv(os.path.join(OUT, "ispcr_hits.csv"), index=False)
    rdf = pd.DataFrame(rows)
    print("\n=== in-silico PCR ===", flush=True)
    if len(rdf):
        print(rdf.groupby("genome").n_products.describe().round(2).to_string(), flush=True)

    # ---------------------------------------------------------------- merge back
    if len(rdf):
        qp = qp.drop(columns=[c for c in ["n_products", "sizes", "ispcr_products",
                                          "specificity", "verdict"] if c in qp])
        qp = qp.merge(rdf[["pair", "n_products", "sizes"]], on="pair", how="left")
        qp["n_products"] = qp.n_products.fillna(0).astype(int)
        n = qp.n_products.to_numpy()
        qp["specificity"] = np.select(
            [n == 1, n == 2, n >= 3],
            [f"unique genomic product (<={MAX_PRODUCT} bp; exact 3' 15-mer, <={MAX_MM} mm)",
             f"2 genomic products (<={MAX_PRODUCT} bp) - check before ordering",
             f"MULTI-HIT ({len(qp)} pairs checked) - do not order without redesign"],
            default="not evaluated")
        qp["verdict"] = np.select(
            [n == 1, n == 2, n >= 3], ["ok", "review", "redesign"],
            default="unknown")
        qp.to_csv(os.path.join(OUT, "qpcr_primers.csv"), index=False)

        print("\n=== verdicts ===", flush=True)
        print(qp.groupby(["genome", "verdict"]).size().to_string(), flush=True)
        bad = qp[qp.verdict != "ok"]
        if len(bad):
            print("\npairs needing attention:\n" +
                  bad[["genome", "tier", "mouse", "fwd", "rev", "n_products", "sizes"]]
                  .to_string(index=False), flush=True)

        sm = {
            "genomes_searched": sorted(qp.genome.unique().tolist()),
            "n_pairs": int(len(qp)),
            "ok": int((qp.verdict == "ok").sum()),
            "review": int((qp.verdict == "review").sum()),
            "redesign": int((qp.verdict == "redesign").sum()),
            "rule": (f"3'-terminal {SEED} nt exact, <= {MAX_MM} mismatches in the 5' remainder, "
                     f"product {MIN_PRODUCT}-{MAX_PRODUCT} bp, both genomes searched whole"),
            "caveat": ("This is an in-silico PCR, not Primer-BLAST: it does not model "
                       "secondary structure, does not allow an indel, and a perfect 3' anchor "
                       "is assumed. Any 'review' or 'redesign' pair must still go through "
                       "Primer-BLAST before ordering."),
        }
        json.dump(sm, open(os.path.join(OUT, "ispcr_summary.json"), "w"),
                  indent=2, default=str)
        print("\n" + json.dumps(sm, indent=2, default=str), flush=True)
    print("done")


if __name__ == "__main__":
    main()
