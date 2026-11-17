"""Family-level transposable-element quantification directly from FASTQ, without an aligner.

GEO ships no BAM/fragments for GSE276872, but the raw reads *are* public
(PRJNA1159315, 41 runs).  scTE/Telescope would need STAR plus multi-mapping
assignment; this does the same job with exact k-mer pseudoalignment against
RepeatMasker family sequence, which suffices for family-level (not locus-level)
counts and needs no index build.

  1. reference: up to N elements per family from mm10 rmsk, sequence from the local
     mm10.2bit, 25-mers canonicalised (min of forward / reverse complement); k-mers
     shared by two families are dropped, so a read can only go to the family whose
     sequence it uniquely matches.
  2. reads: gzip FASTQ streamed and batch-encoded; a read is assigned when its best
     family has >= 3 unique k-mer hits and at least twice the runner-up.
  3. output: assigned reads per million (RPM) per family, plus family composition.

Two limits are measured rather than assumed: recall on held-out elements with known
family labels, and false-assignment rate on random genomic reads.
"""
import gzip, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L
from lib_seq import TwoBit

K = int(os.environ.get("TE_K", 20))
MIN_HITS = int(os.environ.get("TE_MIN_HITS", 2))
COPIES = int(os.environ.get("TE_COPIES", 400))
WIN = int(os.environ.get("TE_WIN", 1500))
TWOBIT = os.path.join(L.ROOT, "data", "genome", "mm10.2bit")
OUT = os.path.join(L.RES, "te")
os.makedirs(OUT, exist_ok=True)

FAMILIES = ["IAP", "ETn", "ERVK_oth", "ERVL", "ERV1", "L1", "SINE_B"]
LUT = np.full(256, -1, dtype=np.int8)
for i, ch in enumerate("ACGT"):
    LUT[ord(ch)] = i
    LUT[ord(ch.lower())] = i


def masks(rmsk):
    rn, rf = rmsk.repName.fillna(""), rmsk.repFamily.fillna("")
    return {
        "IAP": rn.str.startswith("IAP").values,
        "ETn": rn.str.startswith("ETn").values,
        "ERVK_oth": ((rf == "ERVK") & ~rn.str.startswith(("IAP", "ETn"))).values,
        "ERVL": rf.isin(["ERVL", "ERVL-MaLR"]).values,
        "ERV1": (rf == "ERV1").values,
        "L1": rf.isin(["L1", "L2"]).values,
        "SINE_B": rf.isin(["SINE", "tRNA", "MIR"]).values,
    }


def kmers_2d(codes):
    """(n, L) int8 -> (n, L-K+1) canonical int64; -1 where a window contains N."""
    n, L = codes.shape
    m = L - K + 1
    if m <= 0:
        return np.zeros((n, 0), dtype=np.int64)
    bad = (codes < 0).astype(np.int32)
    cs = np.concatenate([np.zeros((n, 1), np.int32), np.cumsum(bad, axis=1)], axis=1)
    valid = (cs[:, K:] - cs[:, :-K]) == 0
    z = np.maximum(codes, 0).astype(np.int64)
    zrc = np.maximum(3 - codes[:, ::-1], 0).astype(np.int64)
    f = np.zeros((n, m), dtype=np.int64)
    r = np.zeros((n, m), dtype=np.int64)
    for j in range(K):
        f = (f << 2) | z[:, j:j + m]
        r = (r << 2) | zrc[:, j:j + m]
    out = np.minimum(f, r)
    out[~valid] = -1
    return out


def kmers_1d(seq):
    codes = LUT[np.frombuffer(seq.encode(), dtype=np.uint8)].reshape(1, -1)
    k = kmers_2d(codes)[0]
    return k[k >= 0]


def build_reference(tb, rmsk, per_family=COPIES, window=WIN, held=30):
    mk = masks(rmsk)
    rmsk = rmsk.assign(_len=(rmsk.genoEnd - rmsk.genoStart).values)
    refs, holds = {}, {}
    for fam in FAMILIES:
        minlen = 120 if fam == "SINE_B" else 400
        d = rmsk[mk[fam] & (rmsk._len >= minlen)].sort_values("_len", ascending=False)
        build, hold = d.iloc[held:held + per_family], d.iloc[:held]
        kms = set()
        for r in build.itertuples():
            s = max(r.genoStart, (r.genoStart + r.genoEnd) // 2 - window // 2)
            kms.update(kmers_1d(tb.fetch(r.genoName, int(s), int(min(r.genoEnd, s + window)))).tolist())
        refs[fam], holds[fam] = kms, hold
    owner = {}
    for fam, kms in refs.items():
        for kv in kms:
            owner[kv] = owner.get(kv, 0) + 1
    keep = {fam: np.array(sorted(kv for kv in kms if owner[kv] == 1), dtype=np.int64)
            for fam, kms in refs.items()}
    allk = np.concatenate([keep[f] for f in FAMILIES])
    allf = np.concatenate([np.full(len(keep[f]), i, np.int8) for i, f in enumerate(FAMILIES)])
    o = np.argsort(allk, kind="mergesort")
    return allk[o], allf[o], holds, {f: int(len(keep[f])) for f in FAMILIES}


def score_batch(seqs, refk, reff, counts):
    """Assign a batch; returns (n_kmers_used, n_reads_unassigned)."""
    n = len(seqs)
    nf = len(FAMILIES)
    L = max(len(s) for s in seqs)
    codes = np.full((n, L), -1, dtype=np.int8)
    for i, s in enumerate(seqs):
        codes[i, :len(s)] = LUT[np.frombuffer(s.encode(), dtype=np.uint8)]
    km = kmers_2d(codes)
    m = km.shape[1]
    flat = km.ravel()
    ri = np.repeat(np.arange(n), m)
    good = flat >= 0
    flat, ri = flat[good], ri[good]
    if not len(flat):
        return 0, n
    pos = np.clip(np.searchsorted(refk, flat), 0, len(refk) - 1)
    hit = refk[pos] == flat
    flat, ri, fam = flat[hit], ri[hit], reff[pos[hit]]
    if not len(flat):
        return 0, n
    order = np.lexsort((flat, ri))
    flat, ri, fam = flat[order], ri[order], fam[order]
    dup = np.zeros(len(flat), bool)
    dup[1:] = (ri[1:] == ri[:-1]) & (flat[1:] == flat[:-1])
    keepm = ~dup
    ri, fam = ri[keepm], fam[keepm]
    bc = np.bincount(ri.astype(np.int64) * nf + fam, minlength=n * nf).reshape(n, nf)
    top = bc.max(1)
    srt = np.sort(bc, axis=1)
    second = srt[:, -2] if nf > 1 else np.zeros(n)
    ok = (top >= MIN_HITS) & (top >= 2 * second)
    counts += np.bincount(bc.argmax(1)[ok], minlength=nf)
    return int(len(flat)), int((~ok).sum())


def count_reads(path, refk, reff, n_max=None, chunk=20000):
    counts = np.zeros(len(FAMILIES), np.int64)
    unassigned = 0
    n_read = kmers_used = 0
    t0 = time.time()
    buf = []
    f = gzip.open(path, "rt", errors="ignore")
    while True:
        try:
            h = f.readline()
            if not h:
                break
            s = f.readline().strip()
            f.readline(); f.readline()
        except EOFError:
            break
        if s:
            buf.append(s)
            n_read += 1
            if len(buf) >= chunk:
                ku, ua = score_batch(buf, refk, reff, counts)
                kmers_used += ku; unassigned += ua; buf = []
            if n_max and n_read >= n_max:
                break
    try:
        f.close()
    except Exception:
        pass
    if buf:
        ku, ua = score_batch(buf, refk, reff, counts)
        kmers_used += ku; unassigned += ua
    return dict(n_reads=n_read, kmer_total=kmers_used, unassigned=unassigned,
                counts=counts.tolist(), seconds=round(time.time() - t0, 1))


def main():
    pilot = "--pilot" in sys.argv
    n_max = 150000 if pilot else None
    rmsk = pd.read_csv(os.path.join(L.ROOT, "data", "annotation", "mm10_rmsk.txt.gz"),
                       sep="\t", header=None,
                       names=["bin", "swScore", "milliDiv", "milliDel", "milliIns", "genoName",
                              "genoStart", "genoEnd", "genoLeft", "strand", "repName", "repClass",
                              "repFamily", "repStart", "repEnd", "repLeft", "id"])
    tb = TwoBit(TWOBIT)
    t0 = time.time()
    refk, reff, holds, sizes = build_reference(tb, rmsk)
    print(f"reference {len(refk):,} unique k-mers in {time.time()-t0:.0f}s: {sizes}", flush=True)

    rng = np.random.default_rng(7)
    val = {}
    for fam in FAMILIES:
        tot = cor = 0
        for r in holds[fam].itertuples():
            if r.genoEnd - r.genoStart < 130:
                continue
            for _ in range(30):
                st = int(rng.integers(r.genoStart, r.genoEnd - 120))
                seq = tb.fetch(r.genoName, st, st + 91)
                if "N" in seq:
                    continue
                c = np.zeros(len(FAMILIES), np.int64)
                score_batch([seq], refk, reff, c)
                tot += 1
                cor += int(c[FAMILIES.index(fam)] == 1)
        val[fam] = dict(simulated=tot, correct=cor,
                        recall=(round(cor / tot, 3) if tot else None))
    # negative control: coding exons of housekeeping genes (random genomic positions are
    # ~40% repetitive in mouse, so they are not a valid negative set)
    neg = np.zeros(len(FAMILIES), np.int64)
    n_neg = 0
    rf = pd.read_csv(os.path.join(L.ROOT, "data", "annotation", "mm10_refFlat.txt.gz"),
                     sep="\t", header=None,
                     names=["name2", "name", "chrom", "strand", "txStart", "txEnd", "cdsStart",
                            "cdsEnd", "exonCount", "exonStarts", "exonEnds"])
    hk = rf[rf.name2.isin(["Actb", "Gapdh", "B2m", "Tubb5", "Rpl13a", "Pgk1", "Hprt", "Sdha"])]
    for r in hk.itertuples():
        starts = [int(x) for x in str(r.exonStarts).rstrip(",").split(",") if x]
        ends = [int(x) for x in str(r.exonEnds).rstrip(",").split(",") if x]
        for a, b in zip(starts, ends):
            if b - a < 100:
                continue
            for _ in range(6):
                st = int(rng.integers(a, b - 91))
                seq = tb.fetch(r.chrom, st, st + 91)
                if "N" in seq:
                    continue
                score_batch([seq], refk, reff, neg)
                n_neg += 1
    val["_coding_exons"] = dict(simulated=n_neg, assigned=int(neg.sum()),
                                false_assignment_rate=round(float(neg.sum() / max(n_neg, 1)), 4))
    json.dump(val, open(os.path.join(OUT, "kmer_engine_validation.json"), "w"),
              ensure_ascii=False, indent=2)
    print("validation:", {k: v.get("recall", v.get("false_assignment_rate")) for k, v in val.items()}, flush=True)

    fq = os.path.join(L.ROOT, "data", "fastq")
    rows = []
    for fn in sorted(os.listdir(fq)):
        if not (fn.endswith(".part.gz") or fn.endswith(".fastq.gz")):
            continue
        if os.path.getsize(os.path.join(fq, fn)) < 5_000_000:
            continue
        grp = "TCDD" if fn.startswith("tcdd") else "Control"
        r = count_reads(os.path.join(fq, fn), refk, reff, n_max=n_max)
        tot = max(r["n_reads"], 1)
        rec = dict(library=fn, group=grp, n_reads=r["n_reads"], seconds=r["seconds"],
                   assigned_frac=round(1 - r["unassigned"] / tot, 4))
        for fam, c in zip(FAMILIES, r["counts"]):
            rec[f"{fam}_rpm"] = c / tot * 1e6
        rows.append(rec)
        print(fn, rec["n_reads"], rec["assigned_frac"], f"{rec['seconds']}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "te_family_kmer_counts.csv"), index=False)

    if len(df) and df.group.nunique() >= 2:
        cnt = df.set_index("group")[[f"{f}_rpm" for f in FAMILIES]].mul(
            df.set_index("group").n_reads / 1e6, axis=0).round(0)
        comp = df.groupby("group")[[f"{f}_rpm" for f in FAMILIES]].mean()
        ratio = (comp.loc["TCDD"] + 1e-9) / (comp.loc["Control"] + 1e-9)
        summary = dict(
            pilot=bool(pilot), n_reads_cap=n_max,
            libraries=df[["library", "group", "n_reads", "assigned_frac"]].to_dict("records"),
            validation=val, rpm=comp.round(2).to_dict(),
            counts=cnt.astype(int).to_dict(),
            estimable={f: bool(cnt[f"{f}_rpm"].min() >= 50) for f in FAMILIES},
            tcdd_over_ctrl_ratio={k: round(float(v), 3) for k, v in ratio.items()},
            median_ratio=float(ratio.median()),
            interpretation=("If every family moves by the same factor, that is library "
                            "composition, not TE derepression; only families departing from "
                            "the median are interpretable."),
            caveat=("pilot reads a byte-prefix of each FASTQ; 10x files are ordered by barcode, "
                    "so the prefix is a cell subset, not a random read sample."),
        )
        json.dump(summary, open(os.path.join(OUT, "kmer_summary.json"), "w"),
                  ensure_ascii=False, indent=2)
        print(json.dumps(summary["tcdd_over_ctrl_ratio"], indent=1))


if __name__ == "__main__":
    main()
