"""In silico perturbation of epigenetic switches on TCDD-exposed coronary cells.

Model (transparent linear response, no black box):
  for each switch gene s and each highly variable gene g, fit slope b_g = cov(x_g, z_s)/var(z_s)
  across cells of that cell type; a perturbation that moves s to the 90th (activation) or 10th
  (repression) percentile of its own distribution displaces x_g by b_g * dz.
  Displacement is projected on the "exposure axis" (control centroid -> TCDD centroid).

Validation: the same machinery applied to Ahr is compared with the real SMC-specific Ahr-KO arm.
"""
import json, os, sys
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

SWITCHES = ["Sirt1", "Suv39h1", "Setdb1", "Ehmt2", "Trim28", "Hdac1", "Dnmt1", "Ezh2", "Ahr"]
N_NULL = 60


def subanalysis(a, celltype, lognorm):
    sub = a[a.obs.celltype == celltype]
    g = sub.obs.group.value_counts()
    if not {"Control", "TCDD_16wk_WT"} <= set(g.index):
        return None
    if min(g["Control"], g["TCDD_16wk_WT"]) < 60:
        return None

    s2 = sub.copy()
    sc.pp.highly_variable_genes(s2, n_top_genes=2000)
    hv = s2.var.highly_variable.to_numpy()
    hvg = np.array(s2.var_names)[hv]
    idx_hv = np.where(hv)[0]
    s2 = sub[:, hv].copy()
    sc.pp.scale(s2, max_value=10)
    sc.pp.pca(s2, n_comps=30, random_state=0)
    P = s2.obsm["X_pca"]
    PC = s2.varm["PCs"]

    is_c = (sub.obs.group == "Control").to_numpy()
    is_t = (sub.obs.group == "TCDD_16wk_WT").to_numpy()
    is_k = (sub.obs.group == "TCDD_16wk_AhrKO").to_numpy()
    axis = P[is_t].mean(0) - P[is_c].mean(0)
    axis /= np.linalg.norm(axis) + 1e-12
    s_axis = P @ axis
    base = s_axis[is_c].mean()
    shift_t = s_axis[is_t].mean() - base

    Xh = np.asarray(s2.X)                       # scaled, cells x 2000
    ridx = a.obs_names.get_indexer(sub.obs_names)
    Xn = np.asarray(lognorm[ridx][:, idx_hv].todense())

    def slopes(z, top_n=150):
        zc = z - z.mean()
        b = (Xh * zc[:, None]).sum(0) / ((zc ** 2).sum() + 1e-12)
        if top_n and top_n < len(b):
            keep = np.argsort(-np.abs(b))[:top_n]
            m = np.zeros_like(b); m[keep] = b[keep]; b = m
        return b

    pc_axis = PC @ axis

    def project(dx):
        # NOTE: do NOT centre dx per gene - the systemic (mean) component is the effect of interest
        return dx @ pc_axis

    res = {}
    rng = np.random.default_rng(0)
    picks = rng.choice(len(hvg), size=min(N_NULL, len(hvg)), replace=False)
    nulls = []
    for p_ in picks:
        z = Xn[:, p_]; z = (z - z.mean()) / (z.std() + 1e-9)
        b = slopes(z)
        da = np.percentile(z[is_t], 90) - z[is_t]
        nulls.append(float(np.nanmean(project(np.outer(da, b)))))
    nulls = np.array(nulls)
    res["null"] = {"mean": float(nulls.mean()), "sd": float(nulls.std()), "n": int(len(nulls))}

    for tf in SWITCHES:
        if tf in list(hvg):
            z = Xn[:, int(np.where(hvg == tf)[0][0])]
        elif tf in a.var_names:
            z = np.asarray(lognorm[ridx][:, a.var_names.get_loc(tf)].todense()).ravel()
        else:
            continue
        z = (z - z.mean()) / (z.std() + 1e-9)
        b = slopes(z)
        for lab, pct in [("activate", 90), ("repress", 10)]:
            da = np.percentile(z[is_t], pct) - z[is_t]
            ds = project(np.outer(da, b))
            res[f"{tf}_{lab}"] = {"mean_delta_axis": float(np.nanmean(ds)),
                                  "reversal_fraction": float(-np.nanmean(ds) / max(abs(shift_t), 1e-9)),
                                  "frac_cells_toward_control": float(np.mean((s_axis[is_t] + ds - base)
                                                                             < (s_axis[is_t] - base)))}

    val = {}
    if "Ahr" in s2.var_names and is_k.sum() > 20:
        obs = s_axis[is_k].mean() - s_axis[is_t].mean()
        pred = res.get("Ahr_repress", {}).get("mean_delta_axis", np.nan)
        val = {"n_ako": int(is_k.sum()),
               "observed_shift_AKO_vs_WT": float(obs),
               "predicted_shift_Ahr_repress": float(pred),
               "sign_agreement": bool(np.sign(obs) == np.sign(pred)) if np.isfinite(pred) else None,
               "calibration_factor": float(obs / pred) if pred not in (0, np.nan) and np.isfinite(pred) and pred != 0 else np.nan}

    return {"celltype": celltype, "n_control": int(is_c.sum()), "n_tcdd": int(is_t.sum()),
            "n_ako": int(is_k.sum()), "n_hvg": int(len(hvg)),
            "axis_shift_tcdd_vs_control": float(shift_t),
            "switches": {k: v for k, v in res.items() if k != "null"},
            "null": res.get("null", {}),
            "validation": val}


def main():
    a = sc.read_h5ad(os.path.join(L.RES, "D2", "processed.h5ad"))
    lognorm = sp.csr_matrix(a.X)
    print("cells", a.n_obs, "genes", a.n_vars, flush=True)
    out = []
    for ct, n in a.obs.celltype.value_counts().items():
        if n < 300:
            continue
        r = subanalysis(a, ct, lognorm)
        print(f"  {ct}: {'ok' if r else 'skipped'}", flush=True)
        if r:
            out.append(r)
    d = os.path.join(L.RES, "perturbation")
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "insilico_perturbation.json"), "w"), indent=1)

    rows = []
    for r in out:
        cal = r["validation"].get("calibration_factor", np.nan)
        for k, v in r["switches"].items():
            rows.append({"celltype": r["celltype"], "perturbation": k, **v,
                         "reversal_fraction_calibrated": v["reversal_fraction"] * (cal if np.isfinite(cal) else 1),
                         "null_mean": r["null"].get("mean", np.nan),
                         "null_sd": r["null"].get("sd", np.nan),
                         "z_vs_null": (v["mean_delta_axis"] - r["null"].get("mean", 0))
                                      / (r["null"].get("sd", 1) + 1e-12)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(d, "switch_effects.csv"), index=False)
    if not df.empty:
        print(df[["celltype", "perturbation", "reversal_fraction", "z_vs_null"]]
              .to_string(index=False), flush=True)
    for r in out:
        if r["validation"]:
            print(r["celltype"], json.dumps(r["validation"]), flush=True)


if __name__ == "__main__":
    main()
