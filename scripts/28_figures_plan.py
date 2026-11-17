"""Figures for the two follow-ups: power of the n=2 upgrade, and what the public ChIP
actually contains (plus the gene-level re-test that replaces the 5 kb bin test).

fig14_power.png   (a) power curves for 2v2 / 2v3 / 3v3 / 4v4 / 6v6, with the observed
                      effects of the four pre-registered endpoints overlaid
                  (b) observed d per primary endpoint against the two MDE thresholds
fig15_chip_gene.png (a) positive control of GSE276152: canonical AHR targets vs random
                      genes -- the reason its negative cannot be read as biology
                  (b) gene-level AHR promoter-peak rate: het motif+, het motif-, background
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG = os.path.join(L.RES, "figures")
os.makedirs(FIG, exist_ok=True)
C_SMC = "#a3320b"
C_EC = "#2f6f9c"
C_MUT = "#8a8578"


def save(fig, name, dpi=200):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved", name, flush=True)


def fig_power():
    grid = pd.read_csv(os.path.join(L.RES, "power", "power_grid.csv"))
    pri = pd.read_csv(os.path.join(L.RES, "power", "primary_endpoints.csv"))
    summary = json.load(open(os.path.join(L.RES, "power", "summary.json")))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    styles = {(2, 2): (C_SMC, "-"), (2, 3): (C_EC, "-"), (3, 3): (C_MUT, "--"),
              (4, 4): ("#5b8c5a", "--"), (6, 6): ("#7a5195", ":")}
    for (n1, n2), sub in grid.groupby(["n_ko", "n_wt"]):
        sub = sub.sort_values("d")
        col, ls = styles.get((n1, n2), ("#999", ":"))
        ax.plot(sub.d, sub.power, color=col, ls=ls, lw=1.6,
                label=f"{n1} vs {n2}")
    ax.axhline(0.8, color="#555", lw=0.8, ls=":")
    ax.text(11.4, 0.82, "80%", fontsize=7, color="#555", ha="right")
    # observed effects: one annotation for the band, two thin lines
    d_atac = abs(float(pri[pri.assay == "ATAC"].d_ko_vs_wt.median()))
    d_rna = abs(float(pri[pri.assay == "RNA"].d_ko_vs_wt.median()))
    for d in (d_atac, d_rna):
        ax.axvline(d, color="#444", lw=0.9, ls="-.")
    ax.annotate(f"实测主终点 |d| = {min(d_atac, d_rna):.1f}–{max(d_atac, d_rna):.1f}",
                xy=(max(d_atac, d_rna), 0.42), xytext=(max(d_atac, d_rna) + 0.6, 0.55),
                fontsize=7.4, color="#333", va="center",
                arrowprops=dict(arrowstyle="->", color="#777", lw=0.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("标准化效应量 d（KO − WT）")
    ax.set_ylabel("功效（双侧 α = 0.05，6,000 次模拟）")
    ax.set_title("a  补一只动物买到的功效", loc="left", fontsize=9.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.2, loc="lower right", title="KO vs WT",
              title_fontsize=7.2)

    ax = axes[1]
    pri = pri.reset_index(drop=True)
    y = np.arange(len(pri))
    ax.barh(y - 0.18, pri.d_ko_vs_ctrl.abs(), 0.34, color=C_EC, label="|KO − Ctrl|")
    ax.barh(y + 0.18, pri.d_ko_vs_wt.abs(), 0.34, color=C_SMC, label="|KO − WT|")
    th = [(summary["mde_80pct"]["2v3"], C_EC, "2 vs 3 门槛", "right", -0.80),
          (summary["mde_80pct"]["2v2"], C_SMC, "2 vs 2 门槛", "left", -1.12)]
    for mde, col, lab, ha, yy in th:
        ax.axvline(mde, color=col, ls="--", lw=1.1)
        ax.text(mde + (-0.15 if ha == "right" else 0.15), yy, f"{lab} = {mde:.1f}",
                fontsize=7, color=col, ha=ha, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.celltype}\n{r.endpoint}" for r in pri.itertuples()], fontsize=7)
    ax.invert_yaxis()
    ax.set_ylim(len(pri) - 0.5, -1.35)
    ax.set_xlim(0, max(6.0, float(pri.d_ko_vs_ctrl.abs().max()) * 1.08))
    ax.set_xlabel("效应量 |d|（线性）")
    ax.set_title("b  四个主终点 vs 两个门槛", loc="left", fontsize=9.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    fig.tight_layout()
    save(fig, "fig14_power.png")


def fig_chip():
    pos = pd.read_csv(os.path.join(L.RES, "chip", "chip_positive_control.csv"))
    tests = pd.read_csv(os.path.join(L.RES, "chip", "gene_level_tests.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})

    ax = axes[0]
    sub = pos[pos.window_kb == 10]
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub.target_mean_peaks, 0.36, color=C_SMC, label="经典 AHR 靶基因（n=6）")
    ax.bar(x + 0.2, sub.random_mean_peaks, 0.36, color="#c9cfd4", label="随机基因（n=600）")
    ax.set_xticks(x)
    ax.set_xticklabels(["AHR / TCDD 峰", "AHR / DMSO 峰"], fontsize=7.6)
    ax.set_ylabel("TSS ±10 kb 内的峰数（均值）")
    for i, r in enumerate(sub.itertuples()):
        ax.text(i - 0.2, r.target_mean_peaks + 0.005, f"{r.target_mean_peaks:.2f}",
                ha="center", fontsize=7.2, color=C_SMC)
        ax.text(i + 0.2, r.random_mean_peaks + 0.005, f"{r.random_mean_peaks:.2f}",
                ha="center", fontsize=7.2, color="#666")
    ax.set_ylim(0, max(sub.target_mean_peaks.max(), sub.random_mean_peaks.max()) * 1.35)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left")
    ax.set_title("a  GSE276152 的阳性对照：经典靶基因上没有峰", loc="left",
                 fontsize=9.2, fontweight="bold")

    ax = axes[1]
    lab = ["het + AhR motif", "het 无 motif", "背景基因"]
    m1 = tests.iloc[1]
    m0 = tests.iloc[0]
    vals = [m1.rate_a, m1.rate_b, m0.rate_b]
    ns = [m1.n_a, m1.n_b, m0.n_b]
    cols = [C_SMC, "#c98b6b", "#c9cfd4"]
    ax.bar(range(3), vals, color=cols, width=0.6)
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(i, v + 0.004, f"{v*100:.1f}%\n(n={int(n)})", ha="center", fontsize=7.2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(lab, fontsize=7.6)
    ax.set_ylabel("启动子带 AHR 峰的比例")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.text(0.02, 0.94, f"het vs 背景：OR {m0.odds_ratio:.2f}，p = {m0.p:.2f}\n"
                        f"motif+ vs motif−：OR {m1.odds_ratio:.2f}，p = {m1.p:.2f}",
            transform=ax.transAxes, fontsize=7.2, color="#444", va="top")
    ax.set_title("b  基因级复测：分辨率不是问题", loc="left", fontsize=9.2,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "fig15_chip_gene.png")


if __name__ == "__main__":
    for f in [fig_power, fig_chip]:
        try:
            f()
        except Exception as e:
            print("  FAILED", f.__name__, type(e).__name__, e, flush=True)
    print("done", flush=True)
