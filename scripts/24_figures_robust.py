"""24 - Figures for the pre-submission verification block.

fig12_robustness.png  2x2: (a) depth sensitivity in EC, (b) depth sensitivity in SMC,
                      (c) GC/TSS-matched domain permutation null, (d) threshold sensitivity
fig13_external.png    1x2: (a) AHR ChIP-seq overlap, (b) human coronary compartments
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import gzip
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

OUT = os.path.join(L.RES, "figures")
os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(L.RES, "figures")

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 220, "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#666", "axes.linewidth": 0.8,
    "xtick.color": "#444", "ytick.color": "#444",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

TF_COL = {"Ahr::Arnt": "#a3320b", "KLF2": "#c0392b", "SMAD2": "#3f6f9c",
          "ATF2": "#7b5ea7", "FOSL1::JUND": "#2f7d6b"}
TARGET_LAB = {"none": "满深度", "match_ako": "等深度", "stress_60pct": "60% 压力"}


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved", name)


def fig_depth():
    eff = pd.read_csv(os.path.join(L.RES, "robust", "downsample_effects.csv"))
    tfs = ["Ahr::Arnt", "KLF2", "SMAD2", "ATF2"]
    targets = ["none", "match_ako", "stress_60pct"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), sharey=True)
    for ax, lin, title in [(axes[0], "Endothelial", "a  内皮：敲 SMC-Ahr 后的 motif 回落"),
                           (axes[1], "Parenchyma", "b  SMC：AhR 自身位点")]:
        w = 0.2
        for i, tf in enumerate(tfs):
            sub = eff[(eff.tf == tf) & (eff.lineage == lin)]
            vals = [sub[sub.target == t].d_ako_vs_wt.mean() for t in targets]
            errs = [sub[sub.target == t].d_ako_vs_wt.std() for t in targets]
            ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, yerr=errs, capsize=2,
                   color=TF_COL[tf], label=tf, error_kw=dict(lw=0.7, ecolor="#999"))
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels([TARGET_LAB[t] for t in targets])
        ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
        ax.set_ylim(-0.42, 0.28)
    axes[0].set_ylabel("d（Ahr-KO vs 16 周 WT）")
    axes[0].legend(frameon=False, fontsize=7.2, ncol=2, loc="lower left")
    ax = axes[1]
    ax.text(0.98, 0.04, "误差棒 = 5 个稀释种子", ha="right", transform=ax.transAxes,
            fontsize=7, color="#777")
    fig.suptitle("Ahr-KO 文库浅 27%：把深度拉到同一水平后重算", fontsize=10.5, y=1.02)
    fig.tight_layout()
    save(fig, "fig12a_depth.png")


def fig_perm():
    null = pd.read_csv(os.path.join(L.RES, "permutation", "permutation_null.csv.gz"))
    test = pd.read_csv(os.path.join(L.RES, "permutation", "permutation_test.csv"))
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    for lin, c in [("Parenchyma", "#a3320b"), ("Endothelial", "#2f6f9c")]:
        v = null[lin].dropna()
        name = "SMC" if lin == "Parenchyma" else "内皮"
        ax.hist(v, bins=40, histtype="step", lw=1.4, color=c, label=f"{name} 零分布")
        obs = float(test[test.lineage == lin].d_observed.iloc[0])
        ax.axvline(obs, color=c, ls="--", lw=1.7)
        ax.annotate(f"{name} 观察值", (obs, ax.get_ylim()[1] * 0.46), color=c, fontsize=7.2,
                    rotation=90, ha="right", va="bottom")
    ax.axvline(0, color="#999", lw=0.7)
    ax.set_xlabel("d（TCDD 16 周 vs 对照）")
    ax.set_ylabel("置换次数")
    ax.set_title("1,000 组 GC/TSS 匹配的常染色质对照：只有 SMC 越出零分布", loc="left",
                 fontsize=9.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.2, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=2)
    p_smc = float(test[test.lineage == "Parenchyma"].p_one_sided.iloc[0])
    p_ec = float(test[test.lineage == "Endothelial"].p_one_sided.iloc[0])
    ax.text(0.02, 0.72, f"SMC 观察值 p = {p_smc:.3f}\n内皮 观察值 p = {p_ec:.2f}",
            transform=ax.transAxes, fontsize=7.4, color="#333")
    fig.tight_layout()
    save(fig, "fig12b_permutation.png")


def fig_threshold():
    t = pd.read_csv(os.path.join(L.RES, "permutation", "threshold_sensitivity.csv"))
    order = ["H3K9me3_all", "H3K9me3_top20", "H3K9me3_top10", "LAD_proxy"]
    labs = ["全部峰", "Top20%", "Top10%", "LAD 代理"]
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    cols = {"Parenchyma": "#a3320b", "Endothelial": "#c0392b",
            "Mesenchymal": "#3f6f9c", "Immune": "#7b5ea7"}
    for lin, c in cols.items():
        ys = [t[(t.definition == o) & (t.lineage == lin)].d_wt_vs_ctrl.iloc[0] for o in order]
        ax.plot(range(4), ys, "-o", color=c, lw=1.6, ms=4, label=lin)
    ax.axhline(0, color="#999", lw=0.7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("d（TCDD 16 周 vs 对照）")
    ax.set_title("域定义从严：效应量单调上升，LAD 代理反向", loc="left", fontsize=9.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "fig12c_threshold.png")


def fig_external():
    lft = pd.read_csv(os.path.join(L.RES, "chip", "liftover_tests.csv"))
    hum = pd.read_csv(os.path.join(L.RES, "human", "human_programme_summary.csv"))
    sm = json.load(open(os.path.join(L.RES, "chip", "liftover_summary.json")))
    chip = json.load(open(os.path.join(L.RES, "chip", "summary.json")))
    n_peaks = chip.get("peaks", {}).get("ahr_tcdd_peaks", None)
    if n_peaks is None:
        pk = os.path.join(L.ROOT, "data", "external", "GSM8492045_AHR_pos_peaks.narrowPeak.gz")
        n_peaks = sum(1 for _ in gzip.open(pk, "rt"))
    n_dmso = sum(1 for _ in gzip.open(os.path.join(
        L.ROOT, "data", "external", "GSM8492044_AHR_neg_peaks.narrowPeak.gz"), "rt"))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5),
                       gridspec_kw={"width_ratios": [1.0, 1.45]})

    ax = axes[0]
    y = np.arange(len(lft))
    ax.barh(y, lft.or_open, color="#a3320b", height=0.55)
    ax.axvline(1, color="#666", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(lft.lineage, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("开放 bin 落在 AHR 峰内的 OR")
    ax.set_xlim(0, 2.2)
    for i, r in lft.reset_index().iterrows():
        ax.text(r.or_open + 0.05, i, f"p={r.p_open:.2f}  (n={int(r.n_open)})",
                va="center", fontsize=7, color="#444")
    ax.set_title("a  AHR ChIP-seq（人冠脉 SMC）与小鼠开放 bin 的直接交集",
                 loc="left", fontsize=9, fontweight="bold")
    ax.text(0.02, -0.30, f"liftOver 覆盖 {sm.get('lift_rate', float('nan'))*100:.0f}% 的 bin；"
                         f"GSE276152 AHR 峰 TCDD {n_peaks:,} 个 / DMSO {n_dmso:,} 个",
            transform=ax.transAxes, fontsize=7, color="#777")

    ax = axes[1]
    # programme x compartment 全矩阵：把“放错地方”的那一格显式画出来，
    # 否则 ModSMC 在成纤维上更高这件事在图上是看不见的。
    sc = pd.read_csv(os.path.join(L.RES, "human", "human_celltype_scores.csv"),
                     usecols=["compartment", "ModSMC", "LAM", "ECpanel", "Matrix"])
    progs = ["ModSMC", "LAM", "ECpanel", "Matrix"]
    comps = ["SMC", "EC", "Mac", "Fib", "T"]
    M = sc.groupby("compartment")[progs].mean().reindex(comps).T.values
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels([f"{c}\n(n={int((sc.compartment == c).sum()):,})" for c in comps],
                       fontsize=7.4)
    ax.set_yticks(range(len(progs)))
    ax.set_yticklabels(["ModSMC\n(小鼠调制型)", "LAM\n(Trem2+ 巨噬)",
                        "ECpanel\n(内皮面板)", "Matrix\n(基质配体)"], fontsize=7.4)
    for i in range(len(progs)):
        for j in range(len(comps)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.4,
                    color="white" if abs(v) > 0.45 * vmax else "#222")
    # 目标格：对角线（ModSMC→SMC、LAM→Mac、ECpanel→EC、Matrix→Fib）
    for i, j in [(0, 0), (1, 2), (2, 1), (3, 3)]:
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                   edgecolor="#1b7f4b", lw=1.6))
    # 反例格单独标红框，说明放在图脚注里，避免压住面板标题
    ax.add_patch(plt.Rectangle((3 - .5, 0 - .5), 1, 1, fill=False,
                               edgecolor="#a3320b", lw=1.8, ls="--"))
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("程序得分（z）", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_title("b  小鼠程序投影到人冠脉（GSE131778，11,756 细胞）",
                 loc="left", fontsize=9, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.5, 0.005,
             "绿框 = 该程序的目标细胞类型；红框 = ModSMC 面板在成纤维中反而更高（0.28 vs SMC 0.21），"
             "故小鼠标签不可直接移植到人。",
             ha="center", fontsize=7, color="#444")
    save(fig, "fig13_external.png")


if __name__ == "__main__":
    for f in [fig_depth, fig_perm, fig_threshold, fig_external]:
        try:
            f()
        except Exception as e:
            print("  FAILED", f.__name__, type(e).__name__, e)
    print("done")
