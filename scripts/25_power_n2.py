"""How much does one more animal actually buy?

The 2026-11 verification pass ended with one action item: "add a second Ahr-KO and a
second 16-week WT mouse". That sentence is easy to say and expensive to do, so before
anyone orders animals the honest question is: *what effect size becomes testable at
n = 2, and is that smaller than the effect we think we have?*

Nothing here is new data. Everything is a simulation on the between-animal variance we
already observe (RNA: 3 + 3 animals, ATAC: 2 control libraries), combined with the
one-animal KO-WT differences as the effect we *hope* to confirm.

Outputs
-------
results/power/power_grid.csv      power as a function of standardised effect, per (n1, n2)
results/power/mde.csv             minimum detectable effect at 80% power
results/power/endpoint_power.csv  per-endpoint observed d and power under each upgrade
results/power/atac_sigma_sensitivity.csv   what happens if the 2-animal SD is an underestimate
results/power/summary.json
results/power/preregistration.md  the frozen analysis plan (written before data exist)
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

OUT = os.path.join(L.RES, "power")
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.05
NSIM = 6000
D_GRID = np.round(np.arange(0.2, 12.01, 0.2), 2)   # 2v2 needs d > 6 to reach 80%
PAIRS = [(2, 2), (2, 3), (3, 2), (3, 3), (4, 4), (6, 6)]

# the four endpoints the manuscript actually argues from — frozen here, before data
PRIMARY = {
    ("ATAC", "Parenchyma", "frac_het"): "SMC 的 H3K9me3 域开放（细胞自主起点）",
    ("ATAC", "Endothelial", "prom_het_rel"): "内皮异染色质启动子相对开放（非细胞自主）",
    ("RNA", "Endothelial", "het_derepression"): "内皮异染色质去抑制模块",
    ("RNA", "Macrophage", "LAM"): "髓系重编程（Trem2/Lgs3 过冲）",
}


def power_curve(n1, n2, rng, dgrid=D_GRID, nsim=NSIM, equal_var=True):
    """Two-sided t-test power on equal-variance data, vectorised over simulations.

    equal_var=True (pooled / Student) is the primary because the estimand assumes a
    common between-animal sigma. Welch is reported alongside: at n=2 its df can collapse
    to 1, which makes it strictly worse and is worth knowing before ordering animals.
    """
    out = np.empty(len(dgrid))
    for i, d in enumerate(dgrid):
        a = rng.standard_normal((nsim, n1))
        b = rng.standard_normal((nsim, n2)) + d
        p = stats.ttest_ind(a, b, axis=1, equal_var=equal_var).pvalue
        out[i] = float(np.mean(p < ALPHA))
    return out


def mde_from_curve(dgrid, pw, target=0.80):
    if pw.max() < target:
        return float("nan")
    j = int(np.argmax(pw >= target))
    if j == 0:
        return float(dgrid[0])
    x0, x1, y0, y1 = dgrid[j - 1], dgrid[j], pw[j - 1], pw[j]
    return float(x0 + (target - y0) * (x1 - x0) / max(y1 - y0, 1e-9))


def power_at(d, n1, n2, rng, nsim=NSIM, equal_var=True):
    a = rng.standard_normal((nsim, n1))
    b = rng.standard_normal((nsim, n2)) + d
    p = stats.ttest_ind(a, b, axis=1, equal_var=equal_var).pvalue
    return float(np.mean(p < ALPHA))


def n_required(d, rng, target=0.80, nmax=40):
    """How many animals per arm would actually be needed at this effect size?"""
    for n in range(2, nmax + 1):
        if power_at(d, n, n, rng, nsim=3000) >= target:
            return n
    return nmax + 1


def main():
    rng = np.random.default_rng(20261117)
    rna = pd.read_csv(os.path.join(L.RES, "pseudobulk", "rna_module_by_mouse.csv"))
    atac = pd.read_csv(os.path.join(L.RES, "pseudobulk", "atac_by_library.csv"))

    # ------------------------------------------------------------------ 1 power grid
    rows = []
    for n1, n2 in PAIRS:
        pw = power_curve(n1, n2, rng)
        for d, p in zip(D_GRID, pw):
            rows.append({"n_ko": n1, "n_wt": n2, "d": d, "power": p})
    grid = pd.DataFrame(rows)
    # Welch 对照：只算 2v2，用于说明“不假设等方差时更差”
    w22 = power_curve(2, 2, rng, equal_var=False)
    grid.to_csv(os.path.join(OUT, "power_grid.csv"), index=False)

    pd.DataFrame({"d": D_GRID, "power_welch_2v2": w22}).to_csv(
        os.path.join(OUT, "power_welch_2v2.csv"), index=False)
    mde = []
    for (n1, n2), sub in grid.groupby(["n_ko", "n_wt"]):
        sub = sub.sort_values("d")
        mde.append({"n_ko": n1, "n_wt": n2,
                    "mde_d_80pct": mde_from_curve(sub.d.to_numpy(), sub.power.to_numpy()),
                    "mde_d_60pct": mde_from_curve(sub.d.to_numpy(), sub.power.to_numpy(), 0.60)})
    mde = pd.DataFrame(mde).sort_values(["n_ko", "n_wt"]).reset_index(drop=True)
    # the σ→raw-unit conversion depends on the endpoint, so d is the portable column
    mde.to_csv(os.path.join(OUT, "mde.csv"), index=False)
    print(mde.to_string(index=False), flush=True)

    # ------------------------------------------------------------------ 2 RNA endpoints
    ep = []
    for (ct, mod), sub in rna.groupby(["celltype", "module"]):
        g = sub.groupby("group").score.apply(list)
        if not {"Control", "TCDD_8wk", "TCDD_16wk_WT", "TCDD_16wk_AhrKO"} <= set(g.index):
            continue
        c8 = np.asarray(g["TCDD_8wk"], float)
        c0 = np.asarray(g["Control"], float)
        wt = float(np.mean(g["TCDD_16wk_WT"]))
        ko = float(np.mean(g["TCDD_16wk_AhrKO"]))
        if len(c8) < 2 or len(c0) < 2:
            continue
        # pooled within-animal SD from the only two groups with replicates (df = 4)
        sp = np.sqrt(((len(c0) - 1) * c0.var(ddof=1) + (len(c8) - 1) * c8.var(ddof=1))
                     / max(len(c0) + len(c8) - 2, 1))
        if sp <= 0:
            continue
        d_kw = (ko - wt) / sp
        d_kc = (ko - c0.mean()) / sp
        ep.append({"assay": "RNA", "celltype": ct, "endpoint": mod,
                   "sigma_within": sp, "df_sigma": len(c0) + len(c8) - 2,
                   "delta_ko_vs_wt": ko - wt, "d_ko_vs_wt": d_kw,
                   "delta_ko_vs_ctrl": ko - c0.mean(), "d_ko_vs_ctrl": d_kc,
                   "power_2v2_kowt": power_at(d_kw, 2, 2, rng),
                   "power_2v2_kowt_welch": power_at(d_kw, 2, 2, rng, equal_var=False),
                   "power_3v2_kowt": power_at(d_kw, 3, 2, rng),
                   "power_2v3_koctrl": power_at(d_kc, 2, 3, rng),
                   "power_3v3_koctrl": power_at(d_kc, 3, 3, rng),
                   "n_per_arm_for_80pct": n_required(abs(d_kw), rng)})

    # ------------------------------------------------------------------ 3 ATAC endpoints
    INFLATE = 1.5   # 2 control animals cannot estimate animal-to-animal SD; inflate by default
    sens = []
    for lineage, sub in atac.groupby("lineage"):
        ctrl = sub[sub.group == "Control"]
        wt = sub[sub.group == "TCDD_16wk_WT"]
        ko = sub[sub.group == "TCDD_16wk_AhrKO"]
        if len(ctrl) < 2 or len(wt) != 1 or len(ko) != 1:
            continue
        for metric in ["frac_het", "frac_lad", "prom_het_rel"]:
            x = ctrl[metric].to_numpy(float)
            s_hat = abs(x[0] - x[1]) / np.sqrt(2)     # df = 1
            if s_hat <= 0:
                continue
            dkw = (float(ko[metric].iloc[0]) - float(wt[metric].iloc[0])) / s_hat
            dkc = (float(ko[metric].iloc[0]) - x.mean()) / s_hat
            for k in [1.0, 1.5, 2.0, 3.0]:
                sens.append({"lineage": lineage, "metric": metric, "sigma_inflate": k,
                             "d_ko_vs_wt": dkw / k, "d_ko_vs_ctrl": dkc / k})
            ep.append({"assay": "ATAC", "celltype": lineage, "endpoint": metric,
                       "sigma_within": s_hat * INFLATE, "df_sigma": 1,
                       "delta_ko_vs_wt": float(ko[metric].iloc[0]) - float(wt[metric].iloc[0]),
                       "d_ko_vs_wt": dkw / INFLATE,
                       "delta_ko_vs_ctrl": float(ko[metric].iloc[0]) - x.mean(),
                       "d_ko_vs_ctrl": dkc / INFLATE,
                       "power_2v2_kowt": power_at(dkw / INFLATE, 2, 2, rng),
                       "power_2v2_kowt_welch": power_at(dkw / INFLATE, 2, 2, rng, equal_var=False),
                       "power_3v2_kowt": power_at(dkw / INFLATE, 3, 2, rng),
                       "power_2v3_koctrl": power_at(dkc / INFLATE, 2, 3, rng),
                       "power_3v3_koctrl": power_at(dkc / INFLATE, 3, 3, rng),
                       "n_per_arm_for_80pct": n_required(abs(dkw / INFLATE), rng)})

    ep = pd.DataFrame(ep)
    ep["primary"] = [bool((r.assay, r.celltype, r.endpoint) in PRIMARY) for r in ep.itertuples()]
    ep["sig_2v2"] = ep.power_2v2_kowt >= 0.80
    ep["sig_3v3"] = ep.power_3v3_koctrl >= 0.80
    ep = ep.sort_values(["assay", "celltype", "endpoint"]).reset_index(drop=True)
    ep.to_csv(os.path.join(OUT, "endpoint_power.csv"), index=False)
    sens = pd.DataFrame(sens)
    sens.to_csv(os.path.join(OUT, "atac_sigma_sensitivity.csv"), index=False)

    # ------------------------------------------------------------------ 4 summary
    def frac(sub, col):
        return float((sub[col]).mean()) if len(sub) else float("nan")

    at = ep[ep.assay == "ATAC"]
    rn = ep[ep.assay == "RNA"]
    mde_d = dict(zip(zip(mde.n_ko, mde.n_wt), mde.mde_d_80pct))
    summary = {
        "alpha": ALPHA, "nsim": NSIM,
        "mde_80pct": {f"{k[0]}v{k[1]}": round(float(v), 2) for k, v in mde_d.items()},
        "atac_sigma_inflate": INFLATE,
        "atac": {
            "n_endpoints": int(len(at)),
            "median_d_ko_vs_wt": round(float(at.d_ko_vs_wt.median()), 2),
            "frac_power80_at_2v2": round(frac(at, "sig_2v2"), 2),
            "frac_power80_at_3v3": round(frac(at, "sig_3v3"), 2),
        },
        "rna": {
            "n_endpoints": int(len(rn)),
            "median_d_ko_vs_wt": round(float(rn.d_ko_vs_wt.median()), 2),
            "median_d_ko_vs_ctrl": round(float(rn.d_ko_vs_ctrl.median()), 2),
            "frac_power80_at_2v2": round(frac(rn, "sig_2v2"), 2),
            "frac_power80_at_3v3": round(frac(rn, "sig_3v3"), 2),
        },
        "design_in_data": {"rna": "Control 3 / 8wk 3 / 16wk WT 1 / KO 1",
                           "atac": "Control 2 / 16wk WT 1 / KO 1"},
        "note": ("Power is simulated for the two-sided pooled-variance t-test at alpha=0.05 "
                 "(Welch reported separately: at n=2 its df collapses toward 1 and power halves). "
                 "Between-animal SD is estimated from the replicated groups: RNA df=4, ATAC df=1 "
                 "(inflated x1.5 by default, sensitivity x1-x3 in atac_sigma_sensitivity.csv)."),
    }
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)

    # ------------------------------------------------------------------ 5 pre-registration
    pri = ep[ep.primary].copy()
    pri["n_needed_koctl"] = [n_required(abs(v), rng) for v in pri.d_ko_vs_ctrl]
    pri["n_needed_kowt"] = [n_required(abs(v), rng) for v in pri.d_ko_vs_wt]
    pri.to_csv(os.path.join(OUT, "primary_endpoints.csv"), index=False)

    def prow(r):
        return (f"| {r.celltype} | {r.endpoint} | {r.d_ko_vs_ctrl:+.2f} | {r.power_2v3_koctrl:.2f} | "
                f"{int(r.n_needed_koctl)} | {r.d_ko_vs_wt:+.2f} | {r.power_2v2_kowt:.2f} | "
                f"{int(r.n_needed_kowt)} |")

    lines = []
    A = lines.append
    A("# 预注册：如果再补动物，检验什么、怎么判、判不出来怎么说")
    A("")
    A("> 本文件**在拿到任何新数据之前**写定，目的是把“补动物之后的分析”锁死，避免事后挑指标。")
    A("> 全部数字来自 `results/power/`（脚本 `scripts/25_power_n2.py`，双侧 t，α = 0.05，6,000 次模拟）。")
    A("")
    A("## 0. 一句话结论")
    A("")
    A("**再补一只 Ahr-KO 买不到“回落了多少”的检验，只买得到“KO 仍然不等于对照”的检验。**")
    A("原因不是效应太小，而是效应相对个体间变异太小：四个主终点的 KO−WT 效应量 d = "
      f"{pri[pri.index.isin(pri.index)].d_ko_vs_wt.min():+.2f} ~ {pri.d_ko_vs_wt.max():+.2f}，"
      f"而 2 vs 2 的 80% 功效门槛是 d = {mde_d.get((2, 2), float('nan')):.2f}。")
    A("同样的效应量要达到 80% 功效，需要 **"
      f"{int(pri.n_needed_kowt.min())}–{int(pri.n_needed_kowt.max())} 只/组**。这不是一次补样能解决的。")
    A("")
    A("## 1. 现有设计的确切形状")
    A("")
    A("| 层 | Control | TCDD 8 周 | TCDD 16 周 WT | TCDD 16 周 Ahr-KO |")
    A("|---|---|---|---|---|")
    A("| scRNA（只数） | 3 | 3 | **1** | **1** |")
    A("| scATAC（文库数） | 2 | — | **1** | **1** |")
    A("")
    A("所以现在有两类对比，可检验性完全不同：")
    A("")
    A("- **KO vs Control**：只要再补 1 只 KO 就是 2 vs 3，有自由度，可检验。")
    A("- **KO vs WT（“回落了多少”）**：必须 KO、WT 各 ≥ 2 才有自由度；补 1+1 是 2 vs 2，门槛很高。")
    A("")
    A("## 2. 不同补样方案买到的功效（80% 功效下的最小可检效应 d）")
    A("")
    A("| 方案 | KO vs WT | KO vs Control | 最小可检 d（KO vs WT） | 最小可检 d（KO vs Control） |")
    A("|---|---|---|---|---|")
    A(f"| 现状 | 1 vs 1（无自由度） | 1 vs 3 | — | — |")
    A(f"| **A：+1 KO** | 1 vs 2（仍不可检） | **2 vs 3** | — | **{mde_d.get((2, 3), float('nan')):.2f}** |")
    A(f"| B：+1 KO，+1 WT | 2 vs 2 | 2 vs 3 | {mde_d.get((2, 2), float('nan')):.2f} | {mde_d.get((2, 3), float('nan')):.2f} |")
    A(f"| C：+2 KO，+1 WT | 3 vs 2 | 3 vs 3 | {mde_d.get((3, 2), float('nan')):.2f} | {mde_d.get((3, 3), float('nan')):.2f} |")
    A(f"| D：+3 KO，+3 WT | 4 vs 4 | 4 vs 4 | {mde_d.get((4, 4), float('nan')):.2f} | {mde_d.get((4, 4), float('nan')):.2f} |")
    A("")
    A("注：以上为合并方差 t（假设组间同方差，与伪批量的估计量一致）。若改用 Welch，"
      "n = 2 时自由度可塌到 1，2 vs 2 的功效进一步下降约一半（见 `power_welch_2v2.csv`）。")
    A("")
    A("## 3. 四个主终点（冻结，不再增删）")
    A("")
    A("| 细胞类型 | 指标 | KO−Ctrl d | 功效（2 vs 3） | 需要只数/组 | KO−WT d | 功效（2 vs 2） | 需要只数/组 |")
    A("|---|---|---|---|---|---|---|---|")
    for r in pri.itertuples():
        A(prow(r))
    A("")
    A("把这张表和方案 A 对照，结论很清楚：")
    A("")
    A("- **KO vs Control 在补 1 只之后基本可检验**（功效 "
      f"{pri.power_2v3_koctrl.min():.2f}–{pri.power_2v3_koctrl.max():.2f}）。这对应论文里那句"
      "“敲除≠复原”——它是本文最稳、也最不依赖大效应的主张。")
    A("- **KO vs WT 的“回落幅度”在 2 vs 2 下功效只有 "
      f"{pri.power_2v2_kowt.min():.2f}–{pri.power_2v2_kowt.max():.2f}**，补一只等于没补。")
    A("")
    A("### 3.1 但 ATAC 的 σ 本身不可靠，要单独看")
    A("")
    A("ATAC 的个体间 SD 只能由 **2 个对照文库**估出（df = 1），默认放大 1.5 倍使用。"
      "把放大倍数从 1 推到 3，主终点的 d 变化如下（`atac_sigma_sensitivity.csv`）：")
    A("")
    A("| 终点 | σ×1 | σ×1.5（默认） | σ×3 |")
    A("|---|---|---|---|")
    for _lin, _met in [("Parenchyma", "frac_het"), ("Endothelial", "prom_het_rel")]:
        q = sens[(sens.lineage == _lin) & (sens.metric == _met)].set_index("sigma_inflate")
        A(f"| {_lin} {_met}（KO−Ctrl） | {q.loc[1.0,'d_ko_vs_ctrl']:+.2f} | "
          f"{q.loc[1.5,'d_ko_vs_ctrl']:+.2f} | {q.loc[3.0,'d_ko_vs_ctrl']:+.2f} |")
    A("")
    A("所以“补一只就能检验”只对内皮 `prom_het_rel` 成立（σ 放大 3 倍仍有 d ≈ 6.7）；"
      "SMC 的 `frac_het` 会从 d ≈ 2.4 掉到 ≈ 0.8，功效随之从 ~0.5 掉到 ~0.2。")
    A("**结论：补 1 只 KO 能让“内皮启动子层面未复原”变成可检验命题；不能让 SMC 的域指标变成可检验命题，"
      "更不能让 RNA 层变成可检验命题。**")
    A("")
    A("## 4. 因此预注册只做两件事")
    A("")
    A("**H1（可检验）：** Ahr-KO 的染色质/转录表型**未回到对照水平**。")
    A("检验：KO vs Control，2 vs 3（补 1 只 KO 后），合并方差 t，双侧 α = 0.05；")
    A("四个主终点做 Holm 校正。判据：校正后 p < 0.05 且方向为“偏离对照”，则写“敲除未使 X 复原”；")
    A("若 p ≥ 0.05，则写“未能排除完全复原”，并给出 Δ 的 95% CI。")
    A("")
    A("**H2（不检验，只描述）：** Ahr-KO 相对 WT 的**回落幅度**。")
    A("处理：报告每只动物的值与 Δ 的 bootstrap 95% CI（按细胞重采样 + 按动物重采样两版），")
    A("明确标注“n = 2，不足以做假设检验”，**不写 p 值**。")
    A("这是把 2 只动物的证据按 2 只动物写，而不是按 20 只写。")
    A("")
    A("## 5. 判据（写死）")
    A("")
    A("| 结果 | 允许写 | 禁止写 |")
    A("|---|---|---|")
    A("| H1 校正 p < 0.05 | “KO 未复原（Δ = …，95% CI …）” | “完全阻断”“彻底逆转” |")
    A("| H1 校正 p ≥ 0.05 | “未能排除复原（Δ = …，CI 跨 0）” | 任何“证明”类措辞 |")
    A("| H2 的 CI 跨 0 | “回落幅度的方向与单只动物一致，但区间跨 0” | “显著回落” |")
    A("| 新动物偏离组内 > 3 SD | 报告并给剔除/保留两版敏感性分析 | 直接剔除 |")
    A("")
    A("## 6. 数据到位后的流程（照抄即可）")
    A("")
    A("```bash")
    A("python scripts/04_process.py D2          # 新样本并入计数矩阵")
    A("python scripts/19_pseudobulk.py          # 按小鼠聚合，自动识别新 mouse id")
    A("python scripts/25b_n2_analysis.py        # 只跑 H1 的四个终点 + Holm，输出 H2 的 bootstrap CI")
    A("```")
    A("")
    A("`25b_n2_analysis.py` 已实现上述判定与“证据不足”分支；`--selftest` 用合成动物跑通全流程，")
    A("不写入 `results/`。")
    A("")
    A("## 7. 明确不做")
    A("")
    A("- 不把细胞当重复：细胞数只用于估计比例，永不进入检验自由度。")
    A("- 不在四个主终点之外挑“显著的那一个”；其余标探索性并给 FDR。")
    A("- 不为凑 n 把不同批次、不同测序深度的动物混在一起而不校正批次；批次作为协变量进入模型。")
    A("- 不因为“补了动物”就重新解释已有结论；已有结论按第 10 节的核验结果为准。")
    A("")
    with open(os.path.join(OUT, "preregistration.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("written", os.path.join(OUT, "preregistration.md"), flush=True)


if __name__ == "__main__":
    main()
