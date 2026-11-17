"""Recompute per-cell fingerprints on already clustered datasets (no re-clustering).

Fixes the column/index alignment bug and adds entropy + regulon entropy.
"""
import importlib.util, os, sys
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

spec = importlib.util.spec_from_file_location(
    "proc04", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_process.py"))
proc04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proc04)

METRIC_COLS = ["HGDI", "LADleak", "TES", "entropy_raw", "entropy_res", "regulon_entropy"]
KEEP_QC = {"n_genes_by_counts", "total_counts", "pct_counts_mt", "n_genes", "n_counts"}


def _is_metric(c):
    c = str(c)
    return (c.startswith(("LI_", "mod_", "z_", "TF_")) or c in METRIC_COLS
            or (c.startswith("n_") and c not in KEEP_QC))


def main(ds):
    out = os.path.join(L.RES, ds)
    p = os.path.join(out, "processed.h5ad")
    if not os.path.exists(p):
        p = os.path.join(out, "clustered.h5ad")
    print(f"[{ds}] loading {p}", flush=True)
    a = sc.read_h5ad(p)
    keep = [c for c in a.obs.columns if not _is_metric(c)]
    a.obs = a.obs[keep].copy()
    act0 = a.obsm.get("score_ulm")
    if isinstance(act0, np.ndarray) or isinstance(act0, pd.DataFrame):
        print("  reusing stored TF activity", flush=True)
    else:
        act0 = None
    a, act = proc04.compute_metrics(a, act=act0)
    a.write_h5ad(os.path.join(out, "processed.h5ad"), compression="gzip")
    act.to_csv(os.path.join(out, "regulon_activity.csv.gz"), compression="gzip", float_format="%.5g")
    slim = a.obs.drop(columns=[c for c in a.obs.columns if c.startswith("TF_")], errors="ignore")
    slim.to_csv(os.path.join(out, "cell_metrics.csv.gz"), compression="gzip", float_format="%.5g")
    nan = slim[[c for c in slim.columns if c in
                ("HGDI", "LADleak", "TES", "entropy_res", "regulon_entropy")]].isna().mean()
    print(nan.round(3).to_dict(), flush=True)
    print(f"[{ds}] recomputed", flush=True)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(d)
