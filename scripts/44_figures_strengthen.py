"""44 - Figures for the three strengthening analyses (report section 16).

  fig19  the branch point under donor stratification
         (a) the three correlations estimated inside each of the 8 donors, with
             the paired difference that carries the claim
         (b) the same correlations inside a single cell type, where composition
             cannot act
  fig20  the stromal continuum
         (a) mouse: the Matrix panel is high in ModifiedSMC *and* in fibroblasts,
             low in contractile SMC - the same animal, so not a species effect
         (b) human: inside SMC-labelled cells the Matrix score rises monotonically
             along a (non-circular) de-differentiation axis and approaches the
             fibroblast level
  fig21  what 3 vs 3 can and cannot deliver
         (a) the exact permutation floor as a function of design
         (b) Hedges' g with 95% CI for the four individual-level endpoints
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)

RED, GREY, BLUE = "#c0392b", "#7f8c8d", "#2c6fbb"
GOLD = "#b8860b"

# ---------------------------------------------------------------- fig19
r3 = json.load(open(os.path.join(RES, "human", "r3_summary.json")))
per = pd.read_csv(os.path.join(RES, "human", "r3_within_donor.csv"))
strat = pd.read_csv(os.path.join(RES, "human", "r3_stratified_partial.csv"))

fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6),
                       gridspec_kw=dict(width_ratios=[1.15, 1]))

PAIRS = [("LAM|Matrix", "髓系–基质"), ("ECpanel|LAM", "内皮–髓系"),
         ("ECpanel|Matrix", "内皮–基质")]
data = [per[k].dropna().to_numpy() for k, _ in PAIRS]
bp = ax[0].boxplot(data, widths=.5, patch_artist=True, showfliers=False,
                   medianprops=dict(color="k", lw=1.4))
for patch, c in zip(bp["boxes"], [BLUE, GREY, RED]):
    patch.set_facecolor(c)
    patch.set_alpha(.25)
    patch.set_edgecolor(c)
for i, (k, _) in enumerate(PAIRS, start=1):
    v = per[k].to_numpy()
    x = np.full(len(v), i, float) + np.random.default_rng(1).normal(0, .045, len(v))
    ax[0].scatter(x, v, s=26, c=c, zorder=3, edgecolors="w", linewidths=.6)
# the paired comparison the claim rests on
for _, row in per.iterrows():
    ax[0].plot([1, 3], [row["LAM|Matrix"], row["ECpanel|Matrix"]],
               c="k", lw=.7, alpha=.35, zorder=2)
ax[0].set_xticks([1, 2, 3])
ax[0].set_xticklabels([lab for _, lab in PAIRS], fontsize=9)
ax[0].axhline(0, c="k", lw=.8, ls=":")
ax[0].set_ylabel("Spearman r（每个供体单独估计，n = 8）", fontsize=9)
d = r3["within_donor_tests"]["LAM|Matrix > ECpanel|Matrix"]
ax[0].set_title(f"(a) 供体内相关：髓系–基质 − 内皮–基质 = "
                f"{d['median_diff']:+.2f}（{d['n_positive']}/8 供体为正，"
                f"Wilcoxon p = {d['p_wilcoxon']:.4f}）", fontsize=9.5, loc="left")

# (b) inside one cell type
order = ["Mac", "T", "Fib", "SMC", "EC", "other"]
s = strat.set_index("compartment").reindex([o for o in order if o in set(strat.compartment)])
w = .26
xs = np.arange(len(s))
for i, (k, lab) in enumerate([("LAM|Matrix", "髓系–基质"), ("ECpanel|LAM", "内皮–髓系"),
                              ("ECpanel|Matrix", "内皮–基质")]):
    ax[1].bar(xs + (i - 1) * w, s[k], w, label=lab,
              color=[BLUE, GREY, RED][i], alpha=.85)
ax[1].set_xticks(xs)
ax[1].set_xticklabels([f"{c}\n({int(n)})" for c, n in zip(s.index, s.n_cells)], fontsize=8)
ax[1].axhline(0, c="k", lw=.8)
ax[1].set_ylabel("偏相关 r（供体 + 深度已扣除）", fontsize=9)
ax[1].legend(fontsize=8, frameon=False, ncol=3, loc="upper center",
             bbox_to_anchor=(0.5, -0.16))
ax[1].set_ylim(-0.09, 0.56)
ax[1].set_title("(b) 单一细胞类型内部：内皮–基质始终 ≈ 0", fontsize=9.5, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig19_branch_donor.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- fig20
mm = pd.read_csv(os.path.join(RES, "human", "stromal_continuum_mouse.csv"))
sc = json.load(open(os.path.join(RES, "human", "stromal_continuum_summary.json")))
grad = pd.DataFrame(sc["gradient"])

fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.5))
cols = ["SMC", "ModifiedSMC", "Fibroblast"]
for r in mm.itertuples():
    ax[0].plot([0, 1, 2], [r.SMC, r.ModifiedSMC, r.Fibroblast],
               marker="o", ms=4, lw=.9, alpha=.8,
               c=RED if "8wk" in str(r.group) else (BLUE if "Control" in str(r.group) else GREY))
ax[0].set_xticks([0, 1, 2])
ax[0].set_xticklabels(["SMC\n（收缩型）", "ModifiedSMC\n（去分化）", "Fibroblast"], fontsize=8.5)
ax[0].set_ylabel("matrix_ligand 模块分数", fontsize=9)
ms = sc["mouse"]
ax[0].set_title(f"(a) 小鼠：去分化 SMC ≈ 成纤维（Δ = {ms['median_mod_minus_smc']:+.2f}，"
                f"{ms['n_up']}/8 只上升；与成纤维差 {ms['median_mod_minus_fib']:+.2f}，"
                f"p = {ms['p_mod_vs_fib']:.2f}）", fontsize=9.5, loc="left")

ax[1].plot(grad["dediff"], grad["matrix"], marker="o", ms=5, c=BLUE, lw=1.6,
           label="SMC 十分位均值")
for _, q in grad.iterrows():
    ax[1].errorbar(q.dediff, q.matrix, yerr=None, fmt="none")
ax[1].axhline(sc["fib_matrix_mean"], ls="--", c=RED, lw=1.2,
              label=f"成纤维均值 {sc['fib_matrix_mean']:.2f}")
ax[1].set_ylim(-0.08, 0.88)
ax[1].set_xlabel("去分化指数（ECM − 收缩，两个方向都不在 Matrix 面板内）", fontsize=8.5)
ax[1].set_ylabel("Matrix 面板分数", fontsize=9)
ax[1].legend(fontsize=8, frameon=False, loc="upper left")
ax[1].set_title(f"(b) 人 SMC 内部：沿去分化轴单调上升（ρ = {sc['gradient_rho']:+.2f}，"
                f"n = {sc['n_stromal']} 间质细胞）", fontsize=9.5, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig20_stromal_continuum.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- fig21
fl = pd.read_csv(os.path.join(RES, "power", "exact_permutation_floor.csv"))
et = pd.read_csv(os.path.join(RES, "power", "endpoint_effect_table.csv"))

fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.4),
                       gridspec_kw=dict(width_ratios=[1, 1.1]))
bal = fl[fl.n_per_group_if_balanced.notna()]
ax[0].plot(bal.n_per_group_if_balanced, bal.exact_floor_p, marker="o", c=BLUE, lw=1.6)
ax[0].axhline(0.05, ls="--", c=RED, lw=1.2)
ax[0].text(2.1, 0.055, "α = 0.05", color=RED, fontsize=8.5)
ax[0].set_yscale("log")
ax[0].set_xlabel("每组动物数", fontsize=9)
ax[0].set_ylabel("精确置换可达到的最小双侧 p", fontsize=9)
ax[0].annotate("3v3：0.143\n（无论效应多大）", xy=(3, 0.143), xytext=(3.4, 0.36),
               fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=.8))
ax[0].annotate("4v4：0.042\n（最小可拒绝设计）", xy=(4, 0.042), xytext=(4.6, 0.012),
               fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=.8))
ax[0].set_title("(a) 精确置换的下限由组合数决定，不由生物学决定", fontsize=9.5, loc="left")

y = np.arange(len(et))[::-1]
ax[1].errorbar(et.g, y, xerr=[et.g - et.g_lo, et.g_hi - et.g], fmt="o", ms=6,
               capsize=4, lw=1.3, color=BLUE)
for i, (_, r) in enumerate(et.iterrows()):
    ax[1].text(max(r.g_hi, 0.3) + .25, y[i],
               f"g = {r.g:+.2f}  p$_{{Welch}}$ = {r.p_welch:.3g}"
               f"  p$_{{perm}}$ = {r.p_exact:.3f}",
               va="center", fontsize=8)
ax[1].axvline(0, c="k", lw=.9)
ax[1].set_yticks(y)
ax[1].set_yticklabels(et.label, fontsize=8)
ax[1].set_xlim(-5.8, 13.5)
ax[1].set_xlabel("Hedges' g（95% CI，非中心 t）", fontsize=9)
ax[1].set_title("(b) 个体水平效应：方向 3/3 一致，但置换 p 触顶", fontsize=9.5, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig21_power_floor.png"), dpi=200)
plt.close(fig)

print("wrote fig19, fig20, fig21", flush=True)
