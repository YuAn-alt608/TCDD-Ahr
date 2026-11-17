"""Figures for the Age3 report."""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

FIG = os.path.join(L.RES, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#555", "text.color": "#222", "axes.labelcolor": "#222",
    "xtick.color": "#444", "ytick.color": "#444", "savefig.dpi": 200,
})
C = {"Control": "#9aa5b1", "PM2.5": "#c0392b", "TCDD_8wk": "#e08b3a",
     "TCDD_16wk_WT": "#a3320b", "TCDD_16wk_AhrKO": "#5b7fa6", "PM2_5": "#c0392b"}

METRIC_LABEL = {"HGDI": "Heterochromatin-domain gene deviation (d)",
                "LADleak": "LAD-proxy domain gene deviation (d)",
                "TEderep": "TE de-repression proxy (d)",
                "Entropy": "Transcriptional entropy (d)",
                "RegulonEntropy": "Regulon entropy (d)"}
DS_LABEL = {"D1": "PM2.5 / lung", "D2": "TCDD / coronary", "D3": "PM2.5 maternal / iWAT"}


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  ->", name)
    return p


def fig_matrix(T):
    if T.empty:
        return
    metrics = ["HGDI", "LADleak", "TEderep", "Entropy"]
    rows, labels = [], []
    for ds in ["D1", "D2", "D3"]:
        s = T[T.dataset == ds]
        for g in sorted(s.group.unique()):
            rows.append(g)
            labels.append(f"{DS_LABEL[ds]} | {g}")
    cols = sorted(T.celltype.unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 0.45 * len(rows) + 1.6))
    for ax, m in zip(np.atleast_1d(axes), metrics):
        M = np.full((len(rows), len(cols)), np.nan)
        for i, g in enumerate(rows):
            sub = T[(T.metric == m) & (T.group == g)]
            for j, c in enumerate(cols):
                v = sub[sub.celltype == c].cohens_d
                if len(v):
                    M[i, j] = v.iloc[0]
        vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
        vmax = max(vmax, 0.3)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels if ax is np.atleast_1d(axes)[0] else [], fontsize=7)
        ax.set_title(METRIC_LABEL[m], fontsize=9)
        for i in range(len(rows)):
            for j in range(len(cols)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(M[i, j]) > vmax * 0.55 else "#222")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle("Exposure x lineage interaction matrix (Cohen's d vs matched control)", y=1.02, fontsize=11)
    return save(fig, "fig1_exposure_lineage_matrix.png")


def fig_timeline(T):
    s = T[(T.dataset == "D2")].copy()
    if s.empty:
        return
    order = {"TCDD_8wk": 8, "TCDD_16wk_WT": 16}
    s["wk"] = s.group.map(order)
    s = s[s.wk.notna()]
    metrics = ["HGDI", "LADleak", "TEderep", "Entropy"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    lineages = ["Endothelial", "Mesenchymal", "Parenchyma", "Immune"]
    colors = {"Endothelial": "#1f6f8b", "Mesenchymal": "#2e8b57",
              "Parenchyma": "#b8860b", "Immune": "#8e44ad"}
    for ax, m in zip(axes, metrics):
        sub = s[s.metric == m]
        for lin in lineages:
            d = sub[sub.lineage == lin].groupby("wk").delta.mean()
            if d.empty:
                continue
            xs = [0] + list(d.index)
            ys = [0.0] + list(d.values)
            ax.plot(xs, ys, "-o", color=colors[lin], label=lin if m == "HGDI" else None, lw=1.8, ms=5)
            half = d.max() / 2 if len(d) else None
            if half and abs(half) > 1e-9:
                ax.axhline(half, color=colors[lin], ls=":", lw=0.8, alpha=0.5)
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_title(METRIC_LABEL[m].replace(" (d)", ""), fontsize=9)
        ax.set_xlabel("TCDD exposure duration (weeks)")
        ax.set_xticks([0, 8, 16])
    axes[0].set_ylabel("Δ vs control")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Temporal order of heterochromatin barrier collapse by lineage (TCDD time series)", y=1.03)
    return save(fig, "fig2_collapse_timeline.png")


def fig_niche(M):
    if M.empty:
        return
    m = M[M.metric.isin(["HGDI", "TEderep", "Entropy"])].copy()
    m["lab"] = m.dataset.map(DS_LABEL) + " | " + m.group
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(m) + 1.5))
    y = np.arange(len(m))
    ax.scatter(m.mean_d_niche, y, color="#1f6f8b", s=55, label="niche (EC + mesenchymal)", zorder=3)
    ax.scatter(m.mean_d_parenchyma, y, color="#b8860b", s=55, marker="s", label="parenchyma", zorder=3)
    for i, (a, b) in enumerate(zip(m.mean_d_niche, m.mean_d_parenchyma)):
        ax.plot([a, b], [i, i], color="#bbb", lw=1, zorder=1)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.lab} · {r.metric}" for r in m.itertuples()], fontsize=7)
    ax.set_xlabel("mean Cohen's d vs control")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title("Microenvironment-support cells vs parenchyma: sensitivity gap", fontsize=10)
    return save(fig, "fig3_niche_vs_parenchyma.png")


def fig_perturb():
    """Two data-driven panels replacing the earlier linear-network prediction figure."""
    p = os.path.join(L.RES, "perturbation", "switch_evidence.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    if d.empty:
        return
    cts = [c for c in ["SMC", "ModifiedSMC", "Fibroblast", "Endothelial", "Macrophage"]
           if c in set(d.celltype)]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    # (A) association: rho between switch expression and position on the exposure axis
    ax = axes[0]
    Mx = d.pivot_table(index="gene", columns="celltype", values="rho_expr_vs_axis")
    Mx = Mx[[c for c in cts if c in Mx.columns]]
    Mx = Mx.loc[Mx.mean(1).sort_values().index]
    vmax = float(np.nanmax(np.abs(Mx.to_numpy()))) or 1.0
    im = ax.imshow(Mx.to_numpy(), cmap="RdBu_r", aspect="auto",
                   norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
    ax.set_xticks(range(Mx.shape[1]))
    ax.set_xticklabels(Mx.columns, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(Mx.shape[0]))
    ax.set_yticklabels(Mx.index, fontsize=8)
    for i in range(Mx.shape[0]):
        for j in range(Mx.shape[1]):
            v = Mx.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="#111" if abs(v) < vmax * 0.6 else "white")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ (switch expression vs exposure axis)")
    ax.set_title("Epigenetic switches lose expression along the exposure axis", fontsize=9.5)

    # (B) real genetic perturbation: Ahr knockout moves cells back toward control
    ax = axes[1]
    w = 0.26
    for k, (col, lab) in enumerate([("axis_pos_control", "Control"),
                                    ("axis_pos_tcdd", "TCDD 16 wk"),
                                    ("axis_pos_ahrko", "TCDD + SMC-Ahr-KO")]):
        vals = [d[d.celltype == c][col].mean() for c in cts]
        ax.bar(np.arange(len(cts)) + (k - 1) * w, vals, width=w,
               color=C.get({"Control": "Control", "TCDD 16 wk": "TCDD_16wk_WT",
                            "TCDD + SMC-Ahr-KO": "TCDD_16wk_AhrKO"}[lab], "#777"), label=lab)
    ax.axhline(0, color="#333", lw=0.9, ls="--")
    ax.set_xticks(range(len(cts)))
    ax.set_xticklabels([c.replace("Modified", "Modified\n") for c in cts], fontsize=8)
    ax.set_ylabel("coordinate on the exposure axis\n(0 = control centroid)")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Ahr knockout reverses the shift in every lineage", fontsize=9.5)
    fig.suptitle("Reversibility evidence: association + a real in-vivo genetic perturbation",
                 y=1.03, fontsize=11)
    return save(fig, "fig4_insilico_switches.png")


def fig_atac():
    p = os.path.join(L.RES, "ATAC", "atac_cell_metrics.csv.gz")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    lin = ["Endothelial", "Mesenchymal", "Parenchyma", "Immune"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col, ttl in zip(axes, ["frac_het", "frac_lad", "prom_het_rel"],
                            ["fraction of accessible reads\nin H3K9me3 domains",
                             "fraction in LAD-proxy domains",
                             "promoter accessibility of het genes\n(minus cell-wide mean)"]):
        sub = d[d.lineage.isin(lin)]
        groups = [g for g in ["Control", "TCDD_16wk_WT", "TCDD_16wk_AhrKO"] if g in sub.group.unique()]
        w = 0.8 / max(len(groups), 1)
        for k, g in enumerate(groups):
            vals, xs = [], []
            for i, l in enumerate(lin):
                v = sub[(sub.group == g) & (sub.lineage == l)][col].dropna()
                vals.append(v.mean() if len(v) else np.nan)
                xs.append(i + (k - (len(groups) - 1) / 2) * w)
            ax.bar(xs, vals, width=w, color=C.get(g, "#777"), label=g)
        ax.set_xticks(range(len(lin)))
        ax.set_xticklabels(lin, rotation=20, ha="right", fontsize=8)
        ax.set_title(ttl, fontsize=9)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.05)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.11))
    fig.suptitle("scATAC-seq: heterochromatin accessibility and AHR-KO rescue (GSE276870)", y=1.23)
    return save(fig, "fig5_atac_heterochromatin.png")


def fig_genetics():
    p = os.path.join(L.RES, "genetics", "gene_panel_summary.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p).sort_values("n_sig")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    ax.barh(d.gene, d.n_sig, color="#1f6f8b")
    ax.set_xlabel("genome-wide significant GWAS associations")
    ax.set_title("Epigenetic switch panel: human genetic signal", fontsize=10)
    ax = axes[1]
    cats = ["n_ageing", "n_cardiovascular", "n_lung", "n_kidney", "n_liver",
            "n_metabolic", "n_neuro", "n_immune"]
    lab = ["ageing", "cardiovascular", "lung", "kidney", "liver", "metabolic", "neuro", "immune"]
    bottom = np.zeros(len(d))
    cmap = plt.get_cmap("tab20")
    for i, (c, l) in enumerate(zip(cats, lab)):
        v = d[c].to_numpy()
        ax.barh(d.gene, v, left=bottom, label=l, color=cmap(i * 2 % 20))
        bottom += v
    ax.set_xlabel("trait-category mentions (all associations)")
    ax.set_title("Organ-degeneration / ageing trait pleiotropy", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    return save(fig, "fig6_gwas_panel.png")



def fig_motif():
    """EC vs SMC: which motif gains accessibility at heterochromatin promoters."""
    p = os.path.join(L.RES, "motif", "motif_activity_effects.csv")
    if not os.path.exists(p):
        return
    a = pd.read_csv(p)
    key = ["KLF2", "KLF4", "Nr1h3", "SMAD2", "HIF1A", "EPAS1", "Ahr::Arnt", "Arnt",
           "ATF2", "FOSL1::JUND", "REL", "RELA", "Nfe2l2"]
    sub = a[a.tf.isin(key)]
    piv = sub.pivot_table(index="tf", columns="lineage", values="d_tcdd_vs_ctrl")
    piv = piv.reindex([k for k in key if k in piv.index])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    y = np.arange(len(piv))
    ax.barh(y - 0.2, piv.get("Endothelial", np.nan), height=0.38,
            color="#1f6f8b", label="Endothelial")
    ax.barh(y + 0.2, piv.get("Parenchyma", np.nan), height=0.38,
            color="#a3320b", label="SMC (Parenchyma)")
    ax.set_yticks(y)
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_xlabel("motif accessibility change, TCDD vs control (Cohen's d)")
    ax.set_title("Motif deviation at heterochromatin-gene promoters", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    if "Parenchyma" in piv.columns and "Endothelial" in piv.columns:
        ax.scatter(piv["Parenchyma"], piv["Endothelial"], s=26, color="#333", zorder=3)
        off = {"Ahr::Arnt": (5, -9), "KLF2": (5, 4), "Nr1h3": (5, 4),
               "Nfe2l2": (6, 2), "RELA": (6, -9)}
        for k in piv.index:
            if k in off:
                ax.annotate(k, (piv.loc[k, "Parenchyma"], piv.loc[k, "Endothelial"]),
                            fontsize=7.5, xytext=off[k], textcoords="offset points")
        lim = np.nanmax(np.abs(piv[["Parenchyma", "Endothelial"]].values)) * 1.15
        ax.plot([-lim, lim], [-lim, lim], ls="--", lw=0.8, color="#999")
        ax.axhline(0, color="#ddd", lw=0.8)
        ax.axvline(0, color="#ddd", lw=0.8)
        ax.set_xlabel("d in SMC")
        ax.set_ylabel("d in endothelial")
        ax.set_title("Same motifs, two arms", fontsize=10)
    fig.tight_layout()
    return save(fig, "fig7_motif_ec_vs_smc.png")


def fig_paracrine():
    p = os.path.join(L.RES, "ccc", "smc_to_macrophage.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p).head(12).iloc[::-1]
    q = os.path.join(L.RES, "ccc", "module_scores_by_group.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    lab = [f"{r.ligand}–{r.receptor}\n({r.sender})" for r in d.itertuples()]
    cols = ["#1f6f8b" if v < 0 else "#b0863a" for v in d.d_ako_vs_wt.fillna(0)]
    ax.barh(np.arange(len(d)), d.d_tcdd_vs_ctrl, color=cols)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(lab, fontsize=6.5)
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_xlabel("communication score, TCDD 16wk − control")
    ax.set_title("SMC -> macrophage: what rises with TCDD\n(bar colour = does it fall when SMC Ahr is deleted)",
                 fontsize=9)
    ax = axes[1]
    if os.path.exists(q):
        m = pd.read_csv(q)
        m = m[m.celltype.isin(["Macrophage", "SMC", "Endothelial"])]
        gorder = ["Control", "TCDD_8wk", "TCDD_16wk_WT", "TCDD_16wk_AhrKO"]
        x = np.arange(len(gorder))
        for i, met in enumerate(["SASP", "INFLAM", "ISG"]):
            for j, c in enumerate(["Macrophage"]):
                v = [m[(m.celltype == c) & (m.group == g)][met].mean() for g in gorder]
                ax.plot(x + (i - 1) * 0.08, v, marker="o", ms=4,
                        color=["#8a6d3b", "#c0392b", "#2c6e49"][i], label=met)
        for j, c in enumerate(["SMC"]):
            v = [m[(m.celltype == c) & (m.group == g)].SASP.mean() for g in gorder]
            ax.plot(x, v, marker="s", ms=4, ls="--", color="#555", label="SASP (SMC)")
        ax.set_xticks(x)
        ax.set_xticklabels(["Ctrl", "TCDD 8wk", "16wk WT", "16wk AhrKO"], fontsize=8)
        ax.set_ylabel("module score (mean log expression)")
        ax.set_title("Macrophage programmes are not reverted by SMC Ahr-KO", fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return save(fig, "fig8_paracrine.png")


def main():
    T = pd.read_csv(os.path.join(L.RES, "matrix", "exposure_lineage_effects.csv")) \
        if os.path.exists(os.path.join(L.RES, "matrix", "exposure_lineage_effects.csv")) else pd.DataFrame()
    M = pd.read_csv(os.path.join(L.RES, "matrix", "niche_vs_parenchyma.csv")) \
        if os.path.exists(os.path.join(L.RES, "matrix", "niche_vs_parenchyma.csv")) else pd.DataFrame()
    fig_matrix(T)
    fig_timeline(T)
    fig_niche(M)
    fig_perturb()
    fig_atac()
    fig_genetics()
    fig_motif()
    fig_paracrine()


if __name__ == "__main__":
    main()
