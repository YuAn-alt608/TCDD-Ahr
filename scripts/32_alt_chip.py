"""32 - A replacement AHR ChIP-seq / CUT&Tag set, admitted only if it passes Tier C.

Why this exists
---------------
GSE276152 (AHR ChIP-seq in HCASMC, DMSO vs TCDD) was the paper's only physical
binding evidence and it failed its own positive control: the six canonical TCDD
targets carry 0.00 peaks within TSS +-10 kb, against 0.20 for random genes
(scripts/26_ahr_gene_level.py). One dead dataset does not condemn the field, so
this script looks for another one and applies a gate *before* looking at any
target of interest.

The gate (Tier C, fixed before the search)
------------------------------------------
  pass  >= 2 of 6 canonical targets have a peak within TSS +-10 kb, AND
        the target set is enriched >= 2x over random genes at the same window.
  weak  enriched 1.3-2x, or passes only at +-50 kb -> usable only as a
        "binding potential" reference, never as a binding call.
  fail  otherwise -> the dataset is reported in the QC figure, not used.

Only datasets that pass are taken on to Tier A/B (the 16 loci in
results/chip/qpcr_candidates.csv).

How candidates are found
------------------------
NCBI eutils (gds) for AHR/AhR/aryl-hydrocarbon + ChIP-seq/CUT&Tag/CUT&RUN, then
the GEO supplementary directory listing; only processed peak files are fetched
(narrowPeak / broadPeak / bed), bigWig and raw reads are skipped, and any single
file over 80 MB is skipped rather than streamed.

Outputs (results/chip_alt/):
  candidates.csv          every GSE considered, with what was found and the verdict
  peak_files.csv          downloaded peak files and their peak counts
  tierC_qc.csv            per dataset x window: target vs random peak counts
  tierAB_binding.csv      only for datasets that passed: candidate loci vs background
  summary.json
"""
import os, sys, json, gzip, re, ssl, time, urllib.request, urllib.parse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

OUT = os.path.join(L.RES, "chip_alt")
RAW = os.path.join(L.ROOT, "data", "external", "chip_alt")
os.makedirs(OUT, exist_ok=True)
os.makedirs(RAW, exist_ok=True)

CTX = ssl.create_default_context()
UA = {"User-Agent": "Mozilla/5.0"}
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series/"
MAX_MB = 80

CANON = ["CYP1A1", "CYP1B1", "AHRR", "TIPARP", "NQO1", "AHR"]
WIN = {"10kb": 10_000, "50kb": 50_000}
PEAK_EXT = (".narrowpeak.gz", ".broadpeak.gz", ".narrowpeak", ".broadpeak",
            ".bed.gz", ".bed", ".peak", ".peaks.gz", ".txt.gz")
TSS_WIN = 2000


def get(url, timeout=90, binary=False):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        d = r.read()
    return d if binary else d.decode("utf-8", "replace")


def esearch(term, retmax=300):
    url = (EUTILS + "esearch.fcgi?db=gds&retmode=json&retmax=%d&term=" % retmax
           + urllib.parse.quote(term))
    return json.loads(get(url))["esearchresult"]


def esummary(ids):
    if not ids:
        return {}
    url = EUTILS + "esummary.fcgi?db=gds&retmode=json&id=" + ",".join(ids)
    d = json.loads(get(url))["result"]
    return {k: v for k, v in d.items() if k != "uids"}


def series_dir(acc):
    n = acc[3:]
    return FTP + f"GSE{n[:-3]}nnn/{acc}/suppl/"


def list_suppl(acc):
    try:
        html = get(series_dir(acc), timeout=60)
    except Exception:
        return []
    return sorted(set(re.findall(r'href="([^"]+)"', html)))


def download(url, dest, cap_mb=MAX_MB):
    try:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
            size = int(r.headers.get("Content-Length", 0))
        if size > cap_mb * 1024 * 1024:
            return None, f"skip {size/1e6:.0f} MB > cap"
        d = get(url, timeout=300, binary=True)
        open(dest, "wb").write(d)
        return dest, f"{len(d)/1e6:.1f} MB"
    except Exception as e:
        return None, f"error {e}"


def read_peaks(path, genome):
    """Return DataFrame(chrom,start,end) for a processed peak file; None if unknown."""
    op = gzip.open if path.endswith(".gz") else open
    rows = []
    try:
        with op(path, "rt", errors="replace") as f:
            for line in f:
                if line.startswith(("#", "track", "browser")):
                    continue
                p = line.rstrip("\n").split("\t")
                if len(p) < 3:
                    continue
                try:
                    s, e = int(float(p[1])), int(float(p[2]))
                except ValueError:
                    continue
                rows.append((p[0], s, e))
    except Exception:
        return None
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    return df[df.chrom.isin(chroms_of(genome))]


def chroms_of(genome):
    return ([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
            if genome == "hg38" else [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"])


def load_tss(genome):
    if genome == "hg38":
        p = os.path.join(L.ROOT, "data", "external", "hg38_refFlat.txt.gz")
        rf = pd.read_csv(p, sep="\t", header=None,
                         names=["gene", "name", "chrom", "strand", "txStart", "txEnd",
                                "cdsStart", "cdsEnd", "n_exon", "exon_starts", "exon_ends",
                                "a", "b", "c", "d", "e"])
        rf = rf[rf.chrom.isin(chroms_of("hg38"))]
        out = rf.groupby("gene").agg(chrom=("chrom", "first"),
                                     tss=("txStart", "min")).reset_index()
        return out.rename(columns={"gene": "gene"})
    p = os.path.join(L.ANN, "mm10_genes_tss.tsv")
    t = pd.read_csv(p, sep="\t")
    return t[["gene", "chrom", "tss"]].drop_duplicates("gene")


def peaks_per_gene(peaks, tss, win):
    """Number of peaks within +-win of each TSS (vectorised per chromosome)."""
    n = np.zeros(len(tss), dtype=int)
    t = tss.reset_index(drop=True)
    for chrom, sub in t.groupby("chrom", observed=True):
        pk = peaks[peaks.chrom == chrom]
        if len(pk) == 0:
            continue
        s = np.sort(pk.start.to_numpy())
        e = np.sort(pk.end.to_numpy())
        # starts within window, then check the nearest peak's end
        lo = np.searchsorted(s, t.tss.to_numpy()[sub.index] - win, side="left")
        hi = np.searchsorted(s, t.tss.to_numpy()[sub.index] + win, side="right")
        cnt = np.zeros(len(sub), dtype=int)
        ts = t.tss.to_numpy()[sub.index]
        for k, (a, b, p) in enumerate(zip(lo, hi, ts)):
            if b > a:
                cnt[k] = int(((s[a:b] <= p + win) & (e[a:b] >= p - win)).sum())
        n[sub.index.to_numpy()] = cnt
    return n


def tier_c(peaks, tss, label):
    """Tier C gate: the six canonical targets vs random genes, two windows."""
    rows = []
    rng = np.random.default_rng(0)
    for wname, win in WIN.items():
        cnt = peaks_per_gene(peaks, tss, win)
        up = {g.upper(): g for g in tss.gene.astype(str)}
        idx = [i for i, g in enumerate(tss.gene.astype(str)) if g.upper() in CANON]
        # keep the first TSS per symbol
        seen, keep = set(), []
        for i in idx:
            s = tss.gene.astype(str).iloc[i].upper()
            if s not in seen:
                seen.add(s); keep.append(i)
        tgt = cnt[keep] if keep else np.array([])
        bg = cnt[rng.choice(len(tss), min(3000, len(tss)), replace=False)]
        rows.append(dict(dataset=label, window=wname, n_targets=len(keep),
                         target_mean=float(tgt.mean()) if len(tgt) else np.nan,
                         target_hit_frac=float((tgt > 0).mean()) if len(tgt) else np.nan,
                         random_mean=float(bg.mean()), random_hit_frac=float((bg > 0).mean()),
                         enrichment=(float(tgt.mean() / bg.mean()) if len(tgt) and bg.mean() > 0
                                     else np.nan),
                         n_peaks=int(len(peaks)),
                         per_target=";".join(f"{tss.gene.astype(str).iloc[i]}={c}"
                                             for i, c in zip(keep, tgt))))
    return rows


def verdict(qc):
    q10 = qc[qc.window == "10kb"]
    q50 = qc[qc.window == "50kb"]
    if len(q10) == 0:
        return "no peaks"
    e10 = float(q10.enrichment.iloc[0]) if np.isfinite(q10.enrichment.iloc[0]) else 0.0
    hit = float(q10.target_hit_frac.iloc[0]) if len(q10) else 0.0
    e50 = float(q50.enrichment.iloc[0]) if len(q50) and np.isfinite(q50.enrichment.iloc[0]) else 0.0
    if e10 >= 2.0 and hit >= 2 / 6:
        return "PASS"
    if e10 >= 1.3 or e50 >= 2.0:
        return "weak"
    return "FAIL"


def main():
    print("=== searching GEO for AHR ChIP-seq / CUT&Tag ===", flush=True)
    queries = [
        '"aryl hydrocarbon receptor" AND (ChIP-seq OR ChIP OR CUT&Tag OR CUT&RUN)',
        'AhR AND (ChIP-seq OR CUT&Tag)',
        'AHR AND peaks AND TCDD',
    ]
    ids = []
    for q in queries:
        try:
            r = esearch(q)
            ids += r.get("idlist", [])
            print(f"  {q[:50]:52s} -> {r.get('count')} hits", flush=True)
        except Exception as e:
            print("  query failed:", e, flush=True)
        time.sleep(0.4)
    ids = list(dict.fromkeys(ids))
    print(f"  {len(ids)} unique UIDs", flush=True)

    meta = []
    for i in range(0, len(ids), 100):
        try:
            meta += list(esummary(ids[i:i + 100]).values())
        except Exception as e:
            print("  esummary failed:", e, flush=True)
        time.sleep(0.4)

    cand = []
    for m in meta:
        acc = m.get("accession", "")
        if not acc.startswith("GSE"):
            continue
        tax = (m.get("taxon", "") or "")
        title = (m.get("title", "") or "")
        summ = (m.get("summary", "") or "")
        blob = (title + " " + summ).lower()
        if "chip" not in blob and "cut&tag" not in blob and "cut&run" not in blob:
            continue
        if "ahr" not in blob and "aryl hydrocarbon" not in blob:
            continue
        genome = ("hg38" if ("human" in tax.lower() or "homo" in tax.lower())
                  else ("mm10" if ("mouse" in tax.lower() or "mus" in tax.lower()) else None))
        if genome is None:
            continue
        cand.append(dict(gse=acc, genome=genome, taxon=tax, title=title[:150],
                         n_samples=m.get("n_samples")))
    cand = pd.DataFrame(cand).drop_duplicates("gse")
    print(f"  {len(cand)} candidate series with AHR + ChIP/CUT&Tag, mouse or human",
          flush=True)

    rows, qc_rows, files_rows = [], [], []
    passed = []
    for r in cand.itertuples():
        files = [f for f in list_suppl(r.gse)
                 if f.lower().endswith(PEAK_EXT)]
        rec = dict(gse=r.gse, genome=r.genome, title=r.title, n_peak_files=len(files))
        if not files:
            rec["verdict"] = "no processed peak files"
            rows.append(rec)
            continue
        # prefer narrowPeak / broadPeak over generic bed
        files = sorted(files, key=lambda f: (0 if "peak" in f.lower() else 1, len(f)))
        got = []
        for f in files[:4]:
            dest = os.path.join(RAW, f"{r.gse}__{f}")
            if not os.path.exists(dest):
                path, msg = download(series_dir(r.gse) + f, dest)
                if path is None:
                    print(f"  {r.gse} {f}: {msg}", flush=True)
                    continue
            got.append(dest)
        if not got:
            rec["verdict"] = "peak files unavailable"
            rows.append(rec)
            continue
        tss = load_tss(r.genome)
        for dest in got:
            pk = read_peaks(dest, r.genome)
            if pk is None or len(pk) == 0:
                continue
            label = f"{r.gse}::{os.path.basename(dest)}"
            q = tier_c(pk, tss, label)
            qc_rows += q
            v = verdict(pd.DataFrame(q))
            files_rows.append(dict(dataset=label, genome=r.genome, n_peaks=len(pk),
                                   verdict=v))
            print(f"  {label}: {len(pk)} peaks, TierC = {v}, "
                  f"enrich10kb = {q[0]['enrichment']}", flush=True)
            if v == "PASS":
                passed.append((label, r.genome, pk, tss))
        rec["verdict"] = (files_rows[-1]["verdict"] if files_rows else "parsed 0 peaks")
        rows.append(rec)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "candidates.csv"), index=False)
    pd.DataFrame(files_rows).to_csv(os.path.join(OUT, "peak_files.csv"), index=False)
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(os.path.join(OUT, "tierC_qc.csv"), index=False)
    print("\n=== Tier C gate ===", flush=True)
    if len(qc):
        print(qc[["dataset", "window", "target_mean", "target_hit_frac", "random_mean",
                  "enrichment", "n_peaks"]].round(3).to_string(index=False), flush=True)

    # ---------------------------------------------------------- Tier A/B
    ab_rows = []
    qp = os.path.join(L.RES, "chip", "qpcr_candidates.csv")
    if passed and os.path.exists(qp):
        q = pd.read_csv(qp)
        for label, genome, pk, tss in passed:
            t = tss.copy()
            cnt = peaks_per_gene(pk, t, 10_000)
            t["n_peaks"] = cnt
            t["sym"] = t.gene.astype(str).str.upper()
            rng = np.random.default_rng(0)
            bg = t.iloc[rng.choice(len(t), min(3000, len(t)), replace=False)]
            for tier in ["A", "B", "C"]:
                sel = q[q.tier.astype(str).str.startswith(tier)]
                syms = set(str(x).upper() for x in sel.get("mouse", sel.get("human", [])))
                sub = t[t.sym.isin(syms)]
                if len(sub) == 0:
                    continue
                ab_rows.append(dict(dataset=label, tier=tier, n_loci=len(sub),
                                    mean_peaks=float(sub.n_peaks.mean()),
                                    frac_bound=float((sub.n_peaks > 0).mean()),
                                    bg_mean=float(bg.n_peaks.mean()),
                                    bg_frac=float((bg.n_peaks > 0).mean()),
                                    enrichment=float(sub.n_peaks.mean() / bg.n_peaks.mean())
                                    if bg.n_peaks.mean() > 0 else np.nan))
    ab = pd.DataFrame(ab_rows)
    ab.to_csv(os.path.join(OUT, "tierAB_binding.csv"), index=False)
    if len(ab):
        print("\n=== Tier A/B binding potential (passed datasets only) ===", flush=True)
        print(ab.round(3).to_string(index=False), flush=True)

    sm = {
        "query_hits": len(ids),
        "candidate_series": int(len(cand)),
        "series_with_peak_files": int(len(files_rows)),
        "n_passed_tierC": len(passed),
        "passed": [p[0] for p in passed],
        "gate": "PASS = >=2/6 canonical targets bound at TSS+-10kb AND >=2x enrichment "
                "over random genes; weak = only at +-50kb or 1.3-2x",
        "verdicts": pd.DataFrame(files_rows).verdict.value_counts().to_dict()
        if files_rows else {},
        "note": ("Datasets are admitted before any locus of interest is examined. If nothing "
                 "passes, the finding is not 'AHR does not bind these promoters' but 'no public "
                 "AHR occupancy map for vascular tissue passes its own positive control'."),
    }
    json.dump(sm, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=str)
    print("\n" + json.dumps(sm, indent=2, default=str), flush=True)
    print("done")


if __name__ == "__main__":
    main()
