"""40 - One streaming pass over GSE131778 for the things R3 needs but 23 threw away.

Script 23 scored the programmes and kept only the scores. Two of the reviewer
objections that R3 has to survive need per-cell covariates that were not saved:

  * library depth (nCount_RNA) and the ribosomal / mitochondrial fractions -
    if the endothelial programme and the myeloid programme are anti-correlated
    only because deeply sequenced cells score high on everything, the branch
    point is an artefact;
  * the marker genes of the stromal continuum (MYH11/ACTA2 on one end,
    TNFRSF11B/LUM/POSTN on the other) - needed to ask whether the Matrix
    programme tracks SMC de-differentiation in human cells.

So this pass reads the same dense gene x cell matrix once and writes two small
tables. It re-uses 23's normalisation (log1p CPM with the library size taken
over ALL genes, z-scored per gene), so the scores stay comparable.

Outputs (results/human/):
  human_cell_qc.csv            nCount, nGene, pct_ribo, pct_mito per cell
  human_marker_logcpm.csv      log1p CPM for the selected marker genes
"""
from __future__ import annotations

import gzip
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(ROOT, "data", "external", "human_coronary.txt.gz")
OUT = os.path.join(ROOT, "results", "human")
os.makedirs(OUT, exist_ok=True)

GENES = [
    # contractile SMC
    "MYH11", "ACTA2", "TAGLN", "CNN1", "MYOCD",
    # fibroblast / matrix
    "DCN", "LUM", "PDGFRA", "COL1A1", "COL1A2", "COL3A1", "POSTN",
    "TNFRSF11B", "FN1", "TIMP1", "COL8A1", "SPP1",
    # de-differentiated / fibromyocyte
    "LGALS3", "VCAM1",
    # other compartments (for context and for the stratified check)
    "PECAM1", "VWF", "CDH5", "EMCN", "KDR",
    "CD68", "CD14", "LYZ", "CSF1R", "AIF1",
    "CD3D", "CD3E", "CD2", "IL7R", "PTPRC",
    # programmes
    "TREM2", "APOE", "GPNMB", "CD9", "FABP5", "LPL",
    "KLF2", "KLF4", "HIF1A", "SMAD2", "NR1H3",
    "MKI67",
]


def main():
    with gzip.open(F, "rt") as f:
        cells = f.readline().rstrip("\n").split("\t")
    n = len(cells)
    print(f"cells {n}", flush=True)

    lib = np.zeros(n, dtype=np.float64)
    ncount = np.zeros(n, dtype=np.float32)
    nribo = np.zeros(n, dtype=np.float32)
    nmito = np.zeros(n, dtype=np.float32)
    want = set(GENES)
    rows, names = [], []
    nline = 0

    with gzip.open(F, "rt") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            g = p[0]
            v = np.asarray(p[1:], dtype=np.float32)
            lib += v
            ncount += (v > 0)
            if g.startswith("MT-"):
                nmito += v
            elif g.startswith("RPS") or g.startswith("RPL"):
                nribo += v
            if g in want:
                names.append(g)
                rows.append(v)
            nline += 1
    print(f"genes {nline}, kept {len(names)}", flush=True)

    # 23 used per-gene z-scores over cells, so log-CPM is the raw material
    X = np.vstack(rows).astype(np.float32)
    cpm = np.log1p(X / np.maximum(lib, 1)[None, :] * 1e4)
    keep = ~pd.Index(names).duplicated()
    names = [g for g, k in zip(names, keep) if k]
    cpm = cpm[keep]

    qc = pd.DataFrame(dict(nCount=lib, nGene=ncount,
                           pct_ribo=100 * nribo / np.maximum(lib, 1),
                           pct_mito=100 * nmito / np.maximum(lib, 1)),
                      index=pd.Index(cells, name="cell"))
    qc.to_csv(os.path.join(OUT, "human_cell_qc.csv"), float_format="%.4f")

    ex = pd.DataFrame(cpm.T, index=cells, columns=names)
    ex.to_csv(os.path.join(OUT, "human_marker_logcpm.csv"), float_format="%.4f")

    print("\nQC summary:", flush=True)
    print(qc.describe().round(2).to_string(), flush=True)
    print("\nmissing panel genes:", [g for g in GENES if g not in names], flush=True)
    print(f"wrote {os.path.join(OUT, 'human_cell_qc.csv')} and human_marker_logcpm.csv",
          flush=True)


if __name__ == "__main__":
    main()
