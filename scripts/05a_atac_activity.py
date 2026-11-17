"""Cache the peak->gene activity matrix and check marker behaviour before labelling."""
import os, sys
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

RES = os.path.join(L.RES, "ATAC")


def parse_peaks(names):
    ch = np.array([n.split(":")[0] for n in names])
    st = np.array([int(n.split(":")[1].split("-")[0]) for n in names])
    en = np.array([int(n.split(":")[1].split("-")[1]) for n in names])
    return ch, st, en


def peak_to_gene(names, genes, window=2000):
    ch, st, en = parse_peaks(names)
    gene_names = genes.gene.to_numpy()
    out = np.full(len(names), -1, dtype=np.int32)
    for chrom, sub in genes.groupby("chrom", observed=True):
        idx = np.where(ch == chrom)[0]
        if len(idx) == 0:
            continue
        tss = sub.tss.to_numpy()
        o = np.argsort(tss)
        tss_s = tss[o]
        gi = sub.index.to_numpy()[o]
        lo = np.searchsorted(tss_s, st[idx] - window, side="left")
        hi = np.searchsorted(tss_s, en[idx] + window, side="right")
        cand = hi > lo
        for k in np.where(cand)[0]:
            a, b = lo[k], hi[k]
            mid = (st[idx][k] + en[idx][k]) // 2
            j = a + int(np.argmin(np.abs(tss_s[a:b] - mid)))
            out[idx[k]] = gi[j]   # gi maps sorted rank -> original gene row
    return out, gene_names


a = sc.read_h5ad(os.path.join(RES, "atac_processed.h5ad"))
genes = pd.read_csv(os.path.join(L.ANN, "mm10_genes_tss.tsv"), sep="\t").reset_index(drop=True)
p2g, gn = peak_to_gene(list(a.var_names), genes)
ok = p2g >= 0
G = sp.csr_matrix((np.ones(ok.sum(), dtype=np.float32), (p2g[ok], np.where(ok)[0])),
                  shape=(len(gn), a.n_vars))
act = np.asarray((a.X @ G.T).todense(), dtype=np.float32)
print("raw act", act.shape, flush=True)

# depth normalise: activity is proportional to sequencing depth, so convert to a fraction
# of the cell's total activity, then scale to a common 5,000 count budget, then log1p
tot = act.sum(1, keepdims=True)
act = np.log1p(act / np.maximum(tot, 1) * 5000.0)
np.savez_compressed(os.path.join(RES, "gene_activity.npz"), act=act, genes=gn)
print("saved gene_activity.npz", flush=True)

mk = ["Myh11", "Acta2", "Tagln", "Cnn1", "Pdgfra", "Col1a1", "Dcn", "Pecam1", "Cdh5",
      "Csf1r", "Cd68", "Lyz2", "Adipoq", "Plin1", "Fabp4", "Emcn", "Vwf", "Postn"]
mk = [g for g in mk if g in set(gn)]
df = pd.DataFrame(act[:, [list(gn).index(g) for g in mk]], columns=mk)
df["leiden"] = a.obs.leiden.to_numpy()
print(pd.DataFrame(df.groupby("leiden", observed=True).mean().round(2)).to_string(), flush=True)
print("\ncluster sizes:", df.leiden.value_counts().sort_index().to_dict(), flush=True)
