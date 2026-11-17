"""22b - Base-pair overlap between the ATAC bins and real AHR ChIP-seq peaks.

22_ahr_chip.py works at orthologous-gene level, which is too coarse for the claim
"the bins that open are AHR binding sites". Here the peaks are lifted instead:
hg38 AHR peaks (GSE276152, HCASMC + TCDD) are mapped to mm10 with the UCSC
hg38ToMm10 chain, and every 5 kb ATAC bin is tested for direct overlap.

The lift is one-directional (mm10 coordinate -> hg38 coordinate) so no peak
interval has to be resized, and the fraction of bins that fail to lift is
reported rather than hidden.

Outputs (results/chip/):
  bin_ahr_overlap.csv     per bin: lifted?, overlaps an AHR peak?
  liftover_tests.csv      Fisher tests per lineage (open / close / background)
  summary.json
"""
import os, sys, gzip, json
from collections import defaultdict
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

EXT = os.path.join(L.ROOT, "data", "external")
OUT = os.path.join(L.RES, "chip")
os.makedirs(OUT, exist_ok=True)
CHAIN = os.path.join(EXT, "hg38ToMm10.over.chain.gz")
LINEAGES = ["Parenchyma", "Endothelial", "Mesenchymal", "Immune"]
CANON = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"]


def load_chain(path, keep):
    """mm10 (query) -> hg38 (target). Both query strands are handled: a large part
    of hg38ToMm10 is written with qStrand='-', and dropping those chains loses
    three quarters of the mappable genome."""
    blocks = defaultdict(list)
    kept = skipped = 0
    tName = None
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("chain"):
                p = line.split()
                tName, tSize, tStrand, tStart = p[2], int(p[3]), p[4], int(p[5])
                qName, qSize, qStrand, qStart = p[7], int(p[8]), p[9], int(p[10])
                tpos, qpos = tStart, qStart
                if qName not in keep:
                    tName = None
                    skipped += 1
                else:
                    kept += 1
                continue
            if tName is None:
                continue
            if not line.strip():
                tName = None
                continue
            tok = line.split()
            if len(tok) != 3:
                tName = None
                continue
            size, dt, dq = (int(x) for x in tok)
            if qStrand == "+":
                fs, fe = qpos, qpos + size
            else:                       # reverse-strand query: convert to forward coords
                fs, fe = qSize - qpos - size, qSize - qpos
            blocks[qName].append((fs, fe, qpos, tName, tpos, tStrand, tSize, qSize, qStrand))
            qpos += size + dq
            tpos += size + dt
    out = {}
    for k, v in blocks.items():
        v.sort()
        out[k] = (np.array([b[0] for b in v]), np.array([b[1] for b in v]),
                  np.array([b[2] for b in v]), [b[3] for b in v],
                  np.array([b[4] for b in v]), [b[5] for b in v],
                  np.array([b[6] for b in v]), np.array([b[7] for b in v]), [b[8] for b in v])
    return out, skipped, kept


def read_peaks(path):
    return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2],
                       names=["chrom", "start", "end"], comment="#")


def main():
    a = ad.read_h5ad(os.path.join(L.RES, "ATAC", "atac_processed.h5ad"), backed="r")
    keys = np.asarray(a.var_names, dtype=np.int64)
    code = (keys // 1_000_000_000).astype(int)
    start = keys % 1_000_000_000
    chrom = np.array(["NA"] * len(keys), dtype=object)
    for i, c in enumerate(CANON):
        chrom[code == i] = c
    ok = chrom != "NA"
    print(f"bins {len(keys)}, on canonical {int(ok.sum())}", flush=True)

    blocks, skipped, kept = load_chain(CHAIN, set(CANON))
    print(f"chain blocks loaded for {len(blocks)} mm10 chroms "
          f"(kept {kept} chains, skipped {skipped})", flush=True)

    peaks = read_peaks(os.path.join(EXT, "GSM8492045_AHR_pos_peaks.narrowPeak.gz"))
    peaks = peaks[peaks.chrom.str.startswith("chr")]
    pk = {c: (np.sort(g.start.to_numpy()), np.sort(g.end.to_numpy()))
          for c, g in peaks.groupby("chrom")}
    print(f"AHR peaks {len(peaks)} on {len(pk)} hg38 chroms", flush=True)

    lifted = np.zeros(len(keys), dtype=bool)
    hit = np.zeros(len(keys), dtype=bool)
    half = 2500
    for c in CANON:
        idx = np.where(chrom == c)[0]
        if len(idx) == 0 or c not in blocks:
            continue
        fs, fe, qs0, tnames, tstarts, tstrands, tsizes, qsizes, qstrands = blocks[c]
        mid = start[idx] + half
        j = np.searchsorted(fs, mid, "right") - 1
        for k in range(len(idx)):
            jj = j[k]
            if jj < 0:
                continue
            # a few chains can overlap the same forward interval: walk back until hit
            found = False
            for back in range(0, 4):
                jr = jj - back
                if jr < 0:
                    break
                if not (fs[jr] <= mid[k] < fe[jr]):
                    continue
                found = True
                qc = mid[k] if qstrands[jr] == "+" else qsizes[jr] - 1 - mid[k]
                off = qc - qs0[jr]
                tc = tnames[jr]
                if tc not in pk:
                    continue
                t0 = tstarts[jr] + off if tstrands[jr] == "+" else \
                    tsizes[jr] - 1 - (tstarts[jr] + off)
                lifted[idx[k]] = True
                s_, e_ = pk[tc]
                lo = np.searchsorted(e_, t0 - half, "left")
                hi = np.searchsorted(s_, t0 + half, "right")
                if hi > lo:
                    hit[idx[k]] = True
                break
            if found:
                continue
    print(f"bins lifted: {int(lifted.sum())} ({lifted[ok].mean()*100:.1f}% of canonical); "
          f"overlapping an AHR peak: {int(hit.sum())}", flush=True)

    pd.DataFrame({"bin_key": keys, "chrom": chrom, "start": start,
                  "lifted": lifted, "ahr_peak": hit}).to_csv(
        os.path.join(OUT, "bin_ahr_overlap.csv"), index=False)

    rows = []
    for ln in LINEAGES:
        p = os.path.join(L.RES, "motif", f"da_bins_{ln}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        pos = pd.Series(np.arange(len(keys)), index=keys)
        ii = pos.reindex(d.bin_key.to_numpy()).dropna().astype(int).to_numpy()
        keep = lifted[ii]
        ii = ii[keep]
        dd = d.iloc[keep].copy()
        dd["ahr"] = hit[ii]
        a1 = int((dd.sig_up & dd.ahr).sum()); b1 = int((dd.sig_up & ~dd.ahr).sum())
        a2 = int((~dd.sig_up & dd.ahr).sum()); b2 = int((~dd.sig_up & ~dd.ahr).sum())
        orr, pv = stats.fisher_exact([[a1, b1], [a2, b2]])
        d1 = int((dd.sig_dn & dd.ahr).sum()); e1 = int((dd.sig_dn & ~dd.ahr).sum())
        orr2, pv2 = stats.fisher_exact([[d1, e1], [a2, b2]])
        rows.append(dict(lineage=ln, n_open=int(dd.sig_up.sum()),
                         n_open_ahr=a1, frac_open_ahr=a1 / max(int(dd.sig_up.sum()), 1),
                         frac_other_ahr=a2 / max(len(dd) - int(dd.sig_up.sum()), 1),
                         or_open=orr, p_open=pv,
                         n_close=int(dd.sig_dn.sum()), n_close_ahr=d1,
                         or_close=orr2, p_close=pv2,
                         n_lifted=len(dd)))
    tdf = pd.DataFrame(rows)
    tdf.to_csv(os.path.join(OUT, "liftover_tests.csv"), index=False)
    print("\n" + tdf.round(4).to_string(index=False), flush=True)

    sm = {"n_bins": int(len(keys)), "bins_lifted": int(lifted.sum()),
          "lift_rate": float(lifted[ok].mean()),
          "bins_with_ahr_peak": int(hit.sum()),
          "peak_fraction_of_bins": float(hit[ok].mean()),
          "chain_skipped": int(skipped),
          "note": "mm10 bin midpoint -> hg38 via UCSC chain; bin window +-2.5 kb"}
    json.dump(sm, open(os.path.join(OUT, "liftover_summary.json"), "w"), indent=2)
    print(json.dumps(sm, indent=2))


if __name__ == "__main__":
    main()
