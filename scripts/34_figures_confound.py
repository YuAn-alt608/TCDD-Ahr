"""34 - Figures for the confounding audit (section 10.7 / 10.8).

fig16_confounding.png  2x2:
  (a) Kitagawa/Oaxaca split of every frozen endpoint into composition vs
      within-state, with the retained fraction printed,
  (b) the Control-vs-8wk composition tests (log2 ratio of the cell fraction,
      Welch FDR and exact permutation p on the same panel),
  (c) n=1 arms: cell-bootstrap CI width vs the Control min-max envelope width,
  (d) matched-depth slope chart: |d| at full depth vs |d| after thinning every
      library to the Ahr-KO depth.

fig17_subsample_tierC.png  1x2:
  (a) 1,000 x 80% within-animal cell subsamples, each endpoint's draw
      distribution scaled by its full-depth d, jackknife range overlaid,
  (b) Tier C gate on every public AHR occupancy dataset (signal-based for the
      bigWig-only series): how many of the 6 canonical targets clear the 95th
      percentile of random TSS windows.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

OUT = os.path.join(L.RES, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 220, "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#666", "axes.linewidth": 0.8,
    "xtick.color": "#444", "ytick.color": "#444",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

VERDICT_COL = {"cell-intrinsic (survives composition matching)": "#2f7d6b",
               "mixed": "#c98a2b",
               "composition-sensitive": "#a3320b"}

SHORT = {
    ("RNA", "Endothelial", "het_derepression"): "内皮·去抑制",
    ("RNA", "Macrophage", "LAM"): "巨噬·LAM",
    ("ATAC", "Parenchyma", "frac_het"): "SMC·域可及",
    ("ATAC", "Endothelial", "prom_het_rel"): "内皮·启动子",
}


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved", name)


def _lab(r):
    base = SHORT.get((r.assay, r.celltype, r.endpoint)) or f"{r.celltype}·{r.endpoint}"
    wk = ("KO" if "AhrKO" in r.contrast else
          "WT" if "16wk" in r.contrast else "8wk")
    return f"{base} · {wk}"


def fig_confounding():
    v = pd.read_csv(os.path.join(L.RES, "audit", "endpoint_verdict.csv"))
    ct = pd.read_csv(os.path.join(L.RES, "audit", "composition_tests_8wk.csv"))
    st = pd.read_csv(os.path.join(L.RES, "audit", "substate_tests_8wk.csv"))
    env = pd.read_csv(os.path.join(L.RES, "audit", "composition_n1_envelope.csv"))
    ds = pd.read_csv(os.path.join(L.RES, "audit", "depth_summary.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))

    # ---------------------------------------------------------------- (a)
    ax = axes[0, 0]
    v = v.sort_values(["assay", "celltype", "endpoint", "contrast"])
    y = np.arange(len(v))
    scale = np.abs(v.delta_raw.to_numpy())
    comp = v.comp.to_numpy() / scale * 100
    state = v.state.to_numpy() / scale * 100
    ax.barh(y, comp, color="#3f6f9c", height=0.62, label="组成（细胞比例）")
    ax.barh(y, state, left=comp, color="#9dbf9b", height=0.62,
            label="态内（同一亚群内部）")
    ax.axvline(0, color="#555", lw=0.9)
    ax.axvline(100, color="#aaa", lw=0.8, ls=":")
    for i, r in enumerate(v.itertuples()):
        ax.text(112, i, f"{r.delta_retained:.2f}", va="center", ha="left", fontsize=7.2,
                color=VERDICT_COL.get(r.verdict, "#444"), fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([_lab(r) for r in v.itertuples()], fontsize=7.4)
    ax.set_xlabel("占 Δ_raw 的百分比（%）")
    ax.set_xlim(-170, 128)
    ax.set_ylim(-0.7, len(v) - 0.15)
    ax.set_title("a  把每个冻结终点拆成“组成”与“态内”两块\n"
                 "右侧数字 = 组成匹配后保留的 Δ 比例（<0.5 即组成敏感）",
                 loc="left", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left", frameon=False, ncol=1)
    ax.text(100, len(v) - 0.55, "100%", fontsize=6.5, color="#888", ha="center", va="top")

    # ---------------------------------------------------------------- (b)
    ax = axes[0, 1]
    t = pd.concat([ct, st]).reset_index(drop=True)
    t = t[t.welch_t.notna()].copy()
    t["lab"] = t.stratum.str.replace("RNA|", "", regex=False)
    col = np.where(t.fdr_welch < 0.05, "#a3320b", "#7f8c8d")
    ax.scatter(t.log2_ratio, -np.log10(t.p_welch), s=26, c=col, alpha=0.85,
               edgecolor="white", linewidth=0.4, zorder=3)
    t["_abs"] = t.log2_ratio.abs()
    for _, r in t.sort_values("_abs", ascending=False).head(4).iterrows():
        ax.annotate(f"{r.component}·{r.lab}", (r.log2_ratio, -np.log10(r.p_welch)),
                    fontsize=6.2, color="#333",
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(-np.log10(0.05), color="#c0392b", lw=0.8, ls="--")
    ax.text(ax.get_xlim()[0] + 0.05, -np.log10(0.05) + 0.05, "p = 0.05",
            fontsize=6.5, color="#c0392b")
    ax.axvline(0, color="#888", lw=0.7)
    ax.set_xlabel("细胞比例 log2 变化（8 周 TCDD / 对照）")
    ax.set_ylabel("−log10（Welch p）")
    nf = int((t.fdr_welch < 0.05).sum())
    ax.set_title(f"b  组成检验：L1 {len(ct)} 个 + L2 {len(st)} 个\n"
                 f"Welch 有 {nf} 个 FDR<0.05，动物水平精确置换 0 个——以精确置换为准",
                 loc="left", fontsize=9, fontweight="bold")

    # ---------------------------------------------------------------- (c)
    ax = axes[1, 0]
    a = env[env.boot_width.notna() & env.ctrl_width.notna()]
    ax.scatter(a.ctrl_width, a.boot_width, s=16, alpha=0.7,
               c=np.where(a.inside_envelope, "#3f6f9c", "#a3320b"),
               edgecolor="white", linewidth=0.3)
    hi = float(np.nanmax(np.r_[a.ctrl_width, a.boot_width])) * 1.05
    ax.plot([1e-3, hi], [1e-3, hi], color="#555", lw=0.9, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("3 只对照之间的跨度（max−min，比例）")
    ax.set_ylabel("细胞重抽样 95% CI 宽度")
    med = float((a.boot_width / a.ctrl_width).median())
    ax.set_title(f"c  n=1 的臂无法用组成辩护\n中位 细胞CI / 对照跨度 = {med:.2f}；"
                 f"仅 {a.inside_envelope.mean()*100:.0f}% 落在包络内",
                 loc="left", fontsize=9, fontweight="bold")
    ax.text(0.03, 0.93, "虚线以上：细胞抽样本身\n就比动物间差异更宽",
            transform=ax.transAxes, fontsize=6.8, color="#666", va="top")

    # ---------------------------------------------------------------- (d)
    ax = axes[1, 1]
    sel = ds[ds.target == "match_ako"].copy()
    frozen = [("ATAC", "Parenchyma", "frac_het"), ("ATAC", "Endothelial", "prom_het_rel"),
              ("RNA", "Endothelial", "het_derepression"), ("RNA", "Macrophage", "LAM")]
    sel = sel[sel.apply(lambda r: (r.assay, r.celltype, r.endpoint) in frozen, axis=1)]
    sel["ratio"] = sel.d_mean.abs() / sel.d_full.abs()
    sel["stat_lab"] = sel.statistic.map({"per_cell": "", "log1p": " · log1p",
                                         "share": " · share"})
    EP = {"frac_het": "H3K9me3 域可及", "prom_het_rel": "异染色质启动子",
          "het_derepression": "异染色质去抑制", "LAM": "LAM 程序"}
    sel["cat"] = (sel.celltype.map({"Parenchyma": "SMC", "Endothelial": "内皮",
                                    "Macrophage": "巨噬"}) + " · " +
                  sel.endpoint.map(EP).fillna(sel.endpoint) +
                  sel.stat_lab + " · " +
                  np.where(sel.contrast.str.contains("AhrKO"), "KO", "WT"))
    sel = sel.sort_values("ratio")
    pal = {"Parenchyma": "#2f7d6b", "Endothelial": "#a3320b", "Macrophage": "#3f6f9c"}
    ys = np.arange(len(sel))
    for i, (_, r) in enumerate(sel.iterrows()):
        ax.plot([1, r.ratio], [i, i], color="#cfd8dc", lw=1.6, zorder=1)
    ax.scatter(sel.ratio, ys, s=42, c=[pal[c] for c in sel.celltype],
               edgecolor="white", linewidth=0.6, zorder=3)
    xr = max(4.2, float(sel.ratio.max()) * 1.12)
    for i, (_, r) in enumerate(sel.iterrows()):
        ax.text(xr * 0.985, i, f"{r.ratio:.2f}", va="center", ha="right",
                fontsize=6.8, color="#555")
    ax.axvline(1, color="#555", lw=0.9, ls="--")
    ax.axvline(0.5, color="#c98a2b", lw=0.8, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels(sel.cat, fontsize=6.9)
    ax.set_xlim(-0.05, xr)
    ax.set_xlabel("|d（等深度）| / |d（满深度）|    1 = 不变，<1 = 衰减，>1 = 被放大")
    ax.set_title("d  等深度重算：所有终点方向都不变\n"
                 "比值只动幅度；>1 的是 log1p 凹性造成的放大",
                 loc="left", fontsize=9, fontweight="bold")
    ax.text(1.05, len(sel) - 0.7, "1 = 不变", fontsize=6.5, color="#555", va="center")
    ax.text(0.44, len(sel) - 0.7, "0.5", fontsize=6.5, color="#c98a2b", va="center",
            ha="right")
    ax.set_ylim(-0.7, len(sel) - 0.2)

    fig.tight_layout()
    save(fig, "fig16_confounding.png")


def fig_subsample_tierc():
    dr = pd.read_csv(os.path.join(L.RES, "audit", "subsample_draws.csv"))
    sm = pd.read_csv(os.path.join(L.RES, "audit", "subsample_summary.csv"))
    bw = pd.read_csv(os.path.join(L.RES, "chip_alt", "bigwig_tierC.csv"))
    tc = pd.read_csv(os.path.join(L.RES, "chip_alt", "tierC_qc.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.1))

    # ---------------------------------------------------------------- (a)
    ax = axes[0]
    sm["key"] = sm.endpoint_name + "|" + np.where(sm.contrast.str.contains("AhrKO"),
                                                  "KO", "WT")
    sm = sm.sort_values("d_full", key=lambda v: v.abs())
    labels = []
    for i, r in enumerate(sm.itertuples()):
        d = dr[(dr.celltype + "|" + dr.endpoint) == r.endpoint_name]
        d = d[d.contrast == r.contrast]
        v = d.d.to_numpy() / r.d_full
        ax.boxplot([v], positions=[i], widths=0.55, vert=True, patch_artist=True,
                   showfliers=False,
                   medianprops=dict(color="#222", lw=1.0),
                   boxprops=dict(facecolor="#cfd8dc", edgecolor="#607d8b", lw=0.7),
                   whiskerprops=dict(color="#607d8b", lw=0.7),
                   capprops=dict(color="#607d8b", lw=0.7))
        ax.plot([i, i], [r.jack_lo / r.d_full, r.jack_hi / r.d_full],
                color="#2f7d6b", lw=3.0, alpha=0.55, solid_capstyle="butt", zorder=4)
        labels.append(f"{r.endpoint_name.split('|')[0][:5]}·{r.endpoint_name.split('|')[-1]}\n{r.key.split('|')[-1]}")
    ax.axhline(1, color="#a3320b", lw=0.9, ls=":", zorder=0)
    ax.set_xticks(range(len(sm)))
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylabel("子抽样 d / 满深度 d")
    ax.set_ylim(0.35, 1.75)
    ax.set_title("每只动物内 1,000 次 80% 细胞重抽样", loc="left",
                 fontsize=9, fontweight="bold")
    ax.text(0.02, 0.97,
            f"全部同号（{sm.frac_same_sign.min()*100:.0f}%）\n"
            f"最差 {sm.frac_within_50pct.min()*100:.1f}% 落在 ±50% 内\n"
            "绿条 = 单细胞 jackknife 范围", transform=ax.transAxes,
            fontsize=6.8, va="top", color="#555")

    # ---------------------------------------------------------------- (b)
    ax = axes[1]
    su = pd.read_csv(os.path.join(L.RES, "chip_alt", "summit_tierC.csv"))
    NAME = {
        "GSE304148_AHR_ChIP_SKMEL2_control": "SKMEL2 · AHR ChIP（对照）",
        "GSE304148_AHR_ChIP_SKMEL2_AHROE": "SKMEL2 · AHR ChIP（过表达）",
        "GSE240495_AHR_CUTRUN_K562": "K562 · AHR CUT&RUN",
        "GSE276152_AHR_ChIP_HCASMC_TCDD": "HCASMC · AHR ChIP（TCDD）",
        "GSE276152_AHR_ChIP_HCASMC_DMSO": "HCASMC · AHR ChIP（DMSO）",
    }
    rows = []
    for ds, g in su.groupby("dataset"):
        nsum = int(g.pass_summit.sum())
        nmean = int(bw[(bw.dataset == ds) & (bw.window == "2kb") & bw.above_p95].shape[0])
        genes = ", ".join(g[g.pass_summit].gene.tolist()) or "—"
        rows.append(dict(label=NAME.get(ds, ds), nsum=nsum, nmean=nmean, genes=genes))
    rows.append(dict(label="人肝细胞 · GSE205502（差异峰集）", nsum=0, nmean=0, genes="—"))
    r = pd.DataFrame(rows).sort_values("nsum")
    ys = np.arange(len(r))
    cols = ["#a3320b" if v >= 2 else "#b0bec5" for v in r.nsum]
    ax.barh(ys, r.nsum, color=cols, height=0.58)
    for i, row in enumerate(r.itertuples()):
        ax.text(row.nsum + 0.08, i, f"{row.nsum}/6", va="center", fontsize=7.4,
                fontweight="bold" if row.nsum >= 2 else "normal")
        ax.text(2.62, i, f"{row.genes}（均值口径 {row.nmean}/6）", va="center",
                fontsize=6.1, color="#666")
    ax.axvline(2, color="#2f7d6b", lw=1.0, ls="--")
    ax.text(2.05, len(r) - 0.55, "门槛 2/6", fontsize=6.5, color="#2f7d6b")
    ax.set_yticks(ys)
    ax.set_yticklabels(r.label, fontsize=6.8)
    ax.set_xlim(0, 5.6)
    ax.set_xticks(range(0, 7))
    ax.set_xlabel("6 个经典靶基因中 summit 超过背景 p95 的个数")
    ax.set_title("Tier C 门禁（summit 口径）", loc="left", fontsize=9,
                 fontweight="bold")

    fig.tight_layout()
    save(fig, "fig17_subsample_tierC.png")


if __name__ == "__main__":
    fig_confounding()
    fig_subsample_tierc()
    print("done")
