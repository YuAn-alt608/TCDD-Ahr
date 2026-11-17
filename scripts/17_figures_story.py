"""Story figures for the restructured manuscript: model cascade, validation panels, TE."""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

FIG = os.path.join(L.RES, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#555", "text.color": "#222", "axes.labelcolor": "#222",
    "xtick.color": "#444", "ytick.color": "#444", "savefig.dpi": 220,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})
C = {"Control": "#9aa5b1", "TCDD_8wk": "#e08b3a", "TCDD_16wk_WT": "#a3320b",
     "TCDD_16wk_AhrKO": "#3f6f9c"}
LAB = {"Control": "对照", "TCDD_8wk": "TCDD 8 周", "TCDD_16wk_WT": "TCDD 16 周",
       "TCDD_16wk_AhrKO": "SMC-Ahr-KO + TCDD"}


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  ->", name)
    return p


# --------------------------------------------------------------------------- model figure
def fig_model():
    t50 = pd.read_csv(os.path.join(L.RES, "matrix", "T50_by_lineage.csv"))
    first = t50.dropna(subset=["T50_weeks"]).sort_values("T50_weeks").groupby("celltype").first()
    order = first.sort_values("T50_weeks")
    keep = [c for c in order.index if c in
            ("Endothelial", "ModifiedSMC", "Macrophage", "Fibroblast")]
    meta = {
        "Endothelial": ("内皮 EC", "#c0392b", (8, 12)),
        "ModifiedSMC": ("调制型 SMC", "#a3320b", (10, 26)),
        "Macrophage": ("巨噬 Mac", "#7b5ea7", (12, 16)),
        "Fibroblast": ("成纤维 Fib", "#3f6f9c", (-40, 22)),
    }
    fig = plt.figure(figsize=(11.6, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1],
                          hspace=0.5, wspace=0.26)

    ax = fig.add_subplot(gs[:, 0])
    for ct in keep:
        r = order.loc[ct]
        xs = np.array([0, 8, 16])
        ys = np.array([0, float(r.delta_8wk), float(r.delta_16wk)])
        col = meta[ct][1]
        t50v = float(r.T50_weeks)
        ax.plot(xs, ys, "-o", color=col, lw=2.2, ms=5, zorder=3,
                label=f"{meta[ct][0]}   T50 = {t50v:.1f} 周")
        ax.plot(xs, ys, "o", color="white", ms=2.4, zorder=4)
        ax.plot([t50v, t50v], [0, np.interp(t50v, xs, ys)], ls=":", color=col, lw=1.0, zorder=2)
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.set_xlabel("TCDD 暴露时长（周）")
    ax.set_ylabel("缓冲崩溃幅度 Δ（各谱系最早失守指标）")
    ax.set_title("a  级联多米诺：内皮最早、成纤维最晚", loc="left", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.8, 17.5)
    ax.set_ylim(-0.005, 0.135)
    ax.legend(frameon=False, fontsize=8.4, loc="upper left", handlelength=1.8)


    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
    ax2.set_title("b  双层解耦的方向性", loc="left", fontsize=11, fontweight="bold")
    boxes = [
        (0.00, 0.56, "TCDD → SMC 的 AhR", "#a3320b", "细胞自主：AhR motif\n选择性开放（OR 2.0）"),
        (0.00, 0.06, "SMC 基质/分泌程序", "#b5651d", "Timp1↑ Fn1↑ Col1a2↑\n（不是 SASP）"),
        (0.54, 0.06, "内皮屏障失守", "#c0392b", "KLF2/SMAD2/HIF1A↑\n敲 SMC-Ahr 后回落"),
        (0.54, 0.56, "巨噬 LAM 切换", "#7b5ea7", "Trem2/Lgals3/Csf1r↑\nISG↑，非复原"),
    ]
    for x, y, t1, col, t2 in boxes:
        ax2.add_patch(Rectangle((x, y), 0.44, 0.32, facecolor="white", edgecolor=col,
                                lw=1.3, transform=ax2.transAxes, zorder=2))
        ax2.text(x + 0.025, y + 0.235, t1, fontsize=8.6, fontweight="bold", color=col,
                 transform=ax2.transAxes, zorder=3, va="center")
        ax2.text(x + 0.025, y + 0.10, t2, fontsize=7.2, color="#333",
                 transform=ax2.transAxes, zorder=3, va="center", linespacing=1.45)
    for (x0, y0, x1, y1) in [(0.44, 0.72, 0.54, 0.72), (0.44, 0.22, 0.54, 0.22),
                             (0.22, 0.56, 0.22, 0.38), (0.76, 0.38, 0.76, 0.56)]:
        ax2.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=ax2.transAxes,
                                      arrowstyle="-|>", mutation_scale=10, color="#666", lw=1.1))

    ax3 = fig.add_subplot(gs[1, 1])
    v = pd.read_csv(os.path.join(L.RES, "validation", "matrix_ligand_macrophage.csv"))
    v = v.set_index("group").loc[["Control", "TCDD_8wk", "TCDD_16wk_WT", "TCDD_16wk_AhrKO"]]
    for col, lab, c in [("lam_score", "LAM 程序", "#7b5ea7"), ("INFLAM", "炎性模块", "#c0392b"),
                        ("ISG", "ISG 模块", "#2f7d6b")]:
        y = v[col].values.astype(float)
        ax3.plot(range(4), (y - y[0]) / (np.abs(y - y[0]).max() + 1e-9), "-o", color=c,
                 label=lab, lw=1.8, ms=4)
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(["对照", "8 周", "16 周", "Ahr-KO"], fontsize=8)
    ax3.axhline(0, color="#bbb", lw=.8)
    ax3.set_ylabel("程序得分（相对对照）")
    ax3.legend(frameon=False, fontsize=7.4, loc="upper left")
    ax3.set_title("c  敲除≠复原：髓系程序过冲", loc="left", fontsize=11, fontweight="bold")
    save(fig, "fig9_model_cascade.png")


# --------------------------------------------------------------------------- validation
def fig_validation():
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6))
    order = ["Control", "TCDD_8wk", "TCDD_16wk_WT", "TCDD_16wk_AhrKO"]

    cm = pd.read_csv(os.path.join(L.RES, "validation", "cm_transfer.csv"))
    ax = axes[0]
    for sender, ls, alpha, col in [("ModifiedSMC", "-", 1.0, "#a3320b"), ("SMC", "--", .6, "#b5651d")]:
        d = cm[cm.sender == sender].set_index("group").loc[order]
        ax.plot(range(4), d.cm_score.values, ls, marker="o", lw=1.9, color=col, alpha=alpha,
                label=sender)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["对照", "8 周", "16 周", "Ahr-KO"], fontsize=8)
    ax.set_ylabel("条件培养基传递分\nΣ Hill(L$_{SMC}$, R$_{EC}$)")
    ax.set_title("a  旁分泌解耦（in silico CM）", loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.6, loc="lower left")
    ax.margins(y=.16)

    fp = pd.read_csv(os.path.join(L.RES, "validation", "cm_readout_prediction.csv"))
    ax = axes[1]
    y = np.arange(len(fp))
    ax.barh(y - .18, fp.obs_ako, height=.34, color="#3f6f9c", label="Ahr-KO 实测")
    ax.barh(y + .18, fp.pred_ako, height=.34, color="#c3d3e2", label="TCDD 臂标定后的预测")
    ax.set_yticks(y)
    ax.set_yticklabels(fp.readout, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("内皮表达（log 归一化拟批量）")
    ax.set_title("b  留出臂检验：4/4 方向正确", loc="left", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    ax.margins(x=.12)

    gi = json.load(open(os.path.join(L.RES, "validation", "genetic_independence.json")))
    ax = axes[2]
    ax.errorbar([gi["odds_ratio"]], [0], xerr=[[gi["odds_ratio"] - gi["ci_low"]],
                                             [gi["ci_high"] - gi["odds_ratio"]]],
                fmt="o", color="#a3320b", ms=7, capsize=4, lw=1.8)
    ax.axvline(1, color="#888", ls="--", lw=1)
    ax.set_yticks([])
    ax.set_ylim(-0.9, 0.9)
    ax.set_xlim(0.3, 1.18)
    ax.set_xlabel("表观开关面板的 GWAS 富集 OR（器官退变/衰老性状）")
    ax.set_title("c  遗传独立性", loc="left", fontsize=10.5, fontweight="bold")
    ax.text(gi["odds_ratio"], 0.30,
            f"OR = {gi['odds_ratio']:.2f}\n95% CI {gi['ci_low']:.2f}–{gi['ci_high']:.2f}",
            ha="center", fontsize=8.2, color="#a3320b")
    ax.text(0.98, -0.62, "环境表观重塑的靶点\n不是常见遗传变异的靶点", ha="right",
            fontsize=7.4, color="#666")
    fig.tight_layout()
    save(fig, "fig10_validation.png")


# --------------------------------------------------------------------------- TE
def fig_te():
    p = os.path.join(L.RES, "te", "kmer_summary.json")
    if not os.path.exists(p):
        print("  (te kmer summary missing)")
        return
    s = json.load(open(p))
    fams = ["IAP", "ETn", "ERVK_oth", "ERVL", "ERV1", "L1", "SINE_B"]
    rpm = pd.DataFrame(s["rpm"])
    cnt = pd.DataFrame(s.get("counts", {}))
    est = s.get("estimable", {f: True for f in fams})
    ratio = s["tcdd_over_ctrl_ratio"]
    x = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    c_vals = rpm.loc["Control", [f"{f}_rpm" for f in fams]].values
    t_vals = rpm.loc["TCDD", [f"{f}_rpm" for f in fams]].values
    ax.bar(x - .2, c_vals, .38, color="#9aa5b1", label="对照")
    ax.bar(x + .2, t_vals, .38, color="#a3320b", label="TCDD 16 周")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 3e4)
    for i, f in enumerate(fams):
        top = max(c_vals[i], t_vals[i])
        if est.get(f, True):
            ax.text(i, top * 1.25, f"×{ratio[f + '_rpm']:.2f}", ha="center", fontsize=7.6, color="#333")
        else:
            n1 = int(cnt.loc["Control", f"{f}_rpm"]) if len(cnt) else 0
            n2 = int(cnt.loc["TCDD", f"{f}_rpm"]) if len(cnt) else 0
            ax.text(i, top * 1.25, f"n.d.\n({n1} vs {n2})", ha="center", fontsize=6.8, color="#999")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in fams], fontsize=8)
    ax.set_ylabel("家族特异 k-mer 命中 reads / 百万（log 轴）")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    n0 = s["libraries"][0]["n_reads"]
    ax.set_title("D2 scRNA 原始 reads 上的 TE 家族定量（等字节前缀 pilot）", loc="left",
                 fontsize=10.5, fontweight="bold")
    ax.text(1.0, -0.19, f"每文库 {n0:,} 条 reads；仅 ≥50 计数的家族可解释",
            transform=ax.transAxes, fontsize=7.2, color="#666", ha="right")
    save(fig, "fig11_te_kmers.png")


def main():
    fig_model()
    fig_validation()
    fig_te()


if __name__ == "__main__":
    main()
