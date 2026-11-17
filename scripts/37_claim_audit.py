"""37 - Claim audit: every number the story line quotes, recomputed from the
result files and checked against the rule the claim needs.

Why this exists
---------------
By this point the report has ~14 sections and dozens of numbers, and several of
them were typed into prose by hand. A claim audit is not a re-analysis: it takes
the specific sentence the paper wants to make, pulls the number out of the file
that produced it, and applies the threshold that sentence implies. Three outcomes:

  PASS      number reproduces and clears the stated rule
  REWRITE   number reproduces but does NOT clear the rule -> the sentence changes
  FAIL      number does not reproduce at all (file changed, or wrong column)

The third outcome is the one that matters: a hand-typed number that no file
supports is the kind of thing a referee finds and nobody forgives.

Outputs (results/audit/):
  claim_audit.csv
  claim_audit.md
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "audit")
os.makedirs(OUT, exist_ok=True)


def _csv(p):
    q = os.path.join(RES, p)
    return pd.read_csv(q) if os.path.exists(q) else pd.DataFrame()


def _json(p):
    q = os.path.join(RES, p)
    return json.load(open(q)) if os.path.exists(q) else {}


def _f(x, n=3):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


ROWS = []


def add(cid, claim, value, rule, ok, src, note=""):
    ROWS.append(dict(id=cid, claim=claim, value=value, rule=rule,
                     verdict="PASS" if ok is True else ("FAIL" if ok is None else "REWRITE"),
                     source=src, note=note))


def main():
    lin = _csv("ATAC/atac_lineage_effects.csv")
    perm = _csv("permutation/permutation_test.csv")
    mot = _csv("motif/motif_enrichment.csv")
    domv = _csv("robust/downsample_domain_verdict.csv")
    dep = _csv("audit/depth_summary.csv")
    sub = _csv("audit/subsample_summary.csv")
    ver = _csv("audit/endpoint_verdict.csv")
    comp = _csv("audit/composition_tests_8wk.csv")
    pb = _csv("pseudobulk/cell_vs_pseudobulk_p.csv")
    hum = _csv("human/human_programme_summary.csv")
    hcs = _csv("human/human_celltype_scores.csv")
    gen = _json("genetics/trait_enrichment.json")
    dom = _json("chip_alt/occupancy_summary.json")
    occ = _csv("chip_alt/occupancy_tests.csv")

    def g(df, col, **kw):
        if not len(df):
            return np.nan
        m = pd.Series(True, index=df.index)
        for k, v in kw.items():
            m &= (df[k] == v)
        s = df[m]
        return float(s[col].iloc[0]) if len(s) else np.nan

    # ---------------------------------------------------------------- C1
    d = g(lin, "d_wt_vs_control", lineage="Parenchyma", metric="frac_het")
    q = g(lin, "fdr", lineage="Parenchyma", metric="frac_het")
    add("C1", "SMC 的 H3K9me3 域在 TCDD 后可及性上升（d>0，FDR<0.05）",
        f"d={_f(d)}, FDR={q:.1e}".replace("e-0", "e-"), "d>0 且 FDR<0.05", bool(d > 0 and q < 0.05),
        "ATAC/atac_lineage_effects.csv")

    # ---------------------------------------------------------------- C2
    pv = g(perm, "p_one_sided", lineage="Parenchyma")
    pe = g(perm, "p_one_sided", lineage="Endothelial")
    pm = g(perm, "p_one_sided", lineage="Mesenchymal")
    pi = g(perm, "p_one_sided", lineage="Immune")
    ok = bool(pv < 0.05 and min(pe, pm, pi) > 0.05)
    add("C2", "GC/TSS 匹配置换下，只有 SMC 的域开放偏移能区别于零模型",
        f"SMC p={_f(pv,4)}；EC p={_f(pe,3)}；Mes p={_f(pm,3)}；Imm p={_f(pi,3)}",
        "SMC p<0.05 且其余三条 p>0.05", ok,
        "permutation/permutation_test.csv",
        "所以“跨谱系普遍击穿”这句必须删——非 SMC 三条都落在零分布内")

    # ---------------------------------------------------------------- C3
    mc = g(lin, "mean_control", lineage="Parenchyma", metric="frac_het")
    mw = g(lin, "mean_wt", lineage="Parenchyma", metric="frac_het")
    mk = g(lin, "mean_ahrko", lineage="Parenchyma", metric="frac_het")
    # (mk - mc)/(mw - mc) is the fraction of the WT gain that SURVIVES the knockout;
    # the reversal is its complement. Reporting the survival fraction as "reversal"
    # inflates the effect by ~6 points and was the bug fixed on 2026-11-18.
    kept = (mk - mc) / (mw - mc) if (mw - mc) else np.nan
    rev = 1 - kept
    add("C3", "SMC 特异敲除 Ahr 后，域开放只部分回落（0<回落比例<1）",
        f"对照 {_f(mc,5)} / WT {_f(mw,5)} / AhrKO {_f(mk,5)}；回落 {rev:.0%}"
        f"（AhrKO 仍保留 WT 增益的 {kept:.0%}）",
        "AhrKO 高于对照且低于 WT", bool(mc < mk < mw),
        "ATAC/atac_lineage_effects.csv")

    # ---------------------------------------------------------------- C4
    mo = mot[(mot.lineage == "Parenchyma") & (mot.direction == "open") &
             (mot.tf == "Ahr::Arnt")]
    orv = float(mo.odds_ratio.iloc[0]) if len(mo) else np.nan
    fdr = float(mo.fdr.iloc[0]) if len(mo) else np.nan
    add("C4", "SMC 开放 bin 富集 Ahr::Arnt motif（OR>1.5，FDR<0.05）",
        f"OR={_f(orv,2)}, FDR={_f(fdr,4)}, n={int(mo.n_target.iloc[0]) if len(mo) else 0}",
        "OR>1.5 且 FDR<0.05", bool(orv > 1.5 and fdr < 0.05),
        "motif/motif_enrichment.csv")

    # ---------------------------------------------------------------- C5
    eo = mot[(mot.lineage == "Endothelial") & (mot.direction == "open")]
    n_ok = int((eo.fdr < 0.05).sum()) if len(eo) else 0
    n_t = int(eo.n_target.iloc[0]) if len(eo) else 0
    ea = eo[eo.tf == "Ahr::Arnt"]
    add("C5", "内皮开放 bin 的 motif 富集能指认出一个上游因子（如 KLF2/HIF1A）",
        f"n={n_t}，FDR<0.05 的 motif {n_ok} 个；Ahr::Arnt p={_f(float(ea.p.iloc[0]),3) if len(ea) else 'NA'}",
        "至少 1 个 motif 过 FDR<0.05", None if not len(eo) else (n_ok > 0),
        "motif/motif_enrichment.csv",
        "n=30 个 bin，一个都没过；因此“内皮走缺氧/剪切通道”只能当假设写，不能当结论")

    # ---------------------------------------------------------------- C6
    dw = g(lin, "d_wt_vs_control", lineage="Endothelial", metric="prom_het_rel")
    ec_c = g(lin, "mean_control", lineage="Endothelial", metric="prom_het_rel")
    ec_w = g(lin, "mean_wt", lineage="Endothelial", metric="prom_het_rel")
    ec_k = g(lin, "mean_ahrko", lineage="Endothelial", metric="prom_het_rel")
    gap = abs(ec_k - ec_w) / abs(ec_w - ec_c) if (ec_w - ec_c) else np.nan
    add("C6", "内皮启动子可及性改变不被 SMC 特异的 Ahr 敲除逆转",
        f"d={_f(dw,3)}；对照 {_f(ec_c,5)} / WT {_f(ec_w,5)} / AhrKO {_f(ec_k,5)}；KO 与 WT 相差 {gap:.0%} 的 WT−Ctrl 落差",
        "AhrKO 与 WT 接近（<25% 落差）且两者都远离对照", bool(gap < 0.25),
        "ATAC/atac_lineage_effects.csv")

    # ---------------------------------------------------------------- C7
    if len(domv):
        same = int((np.sign(domv.full_d) == np.sign(domv.match_ako_d)).sum() +
                   (np.sign(domv.full_d) == np.sign(domv.stress_60pct_d)).sum())
        n_cmp = 2 * len(domv)
        holds = int((domv.verdict == "holds").sum())
        add("C7", "等深度重算后，域指标的方向全部不变（12/12），显著性不全部保持",
            f"方向 {same}/{n_cmp}；判定 holds {holds}/{len(domv)}，其余 direction only",
            "方向 100% 一致", bool(same == n_cmp),
            "robust/downsample_domain_verdict.csv",
            f"符号全保，但只有 {holds}/{len(domv)} 连显著性一起保住——正文不能写“方向与显著性全保”")

    # ---------------------------------------------------------------- C8
    if len(dep):
        n33 = int((dep.sign_match == "3/3").sum())
        bad = dep[dep.sign_match != "3/3"]
        add("C8", "逐细胞终点在等深度下方向稳定",
            f"{n33}/{len(dep)} 个组合三次稀释全部同号；非全同号的 {len(bad)} 个：" +
            ", ".join(f"{r.celltype}/{r.endpoint}/{'AhrKO' if 'AhrKO' in r.contrast else 'WT'}({r.sign_match})"
                      for r in bad.itertuples()),
            "主终点（EC het_derepression / SMC frac_het / Mac LAM / Fib matrix）不出现 0/3",
            bool(not (bad[(bad.celltype == "Macrophage") & (bad.endpoint == "LAM")].sign_match == "0/3").any()),
            "audit/depth_summary.csv",
            "Mac LAM（WT 16 周）只 2/3 同号、衰减 93.8%——必须写进局限，不能只说 12/12")

    # ---------------------------------------------------------------- C9
    if len(sub):
        fs = float(sub.frac_same_sign.min())
        add("C9", "每只动物内 80% 细胞重抽样，主终点同号比例 100%",
            f"最低同号比例 {fs:.3f}（{len(sub)} 个终点）", "全部 =1.0", bool(fs >= 0.999),
            "audit/subsample_summary.csv")

    # ---------------------------------------------------------------- C10
    if len(ver):
        vc = ver.verdict.value_counts()
        ci = int(vc.get("cell-intrinsic (survives composition matching)", 0))
        mx = int(vc.get("mixed", 0))
        cs = int(vc.get("composition-sensitive", 0))
        add("C10", "组成分解后 10 个终点中 6 个细胞自主、2 个混合、2 个组成敏感",
            f"{ci} / {mx} / {cs}", "6/2/2", bool((ci, mx, cs) == (6, 2, 2)),
            "audit/endpoint_verdict.csv")

    # ---------------------------------------------------------------- C11
    if len(comp):
        pmin = float(comp.p_exact_perm.min())
        add("C11", "细胞组成的改变不能宣称显著（精确置换最小可达 p≈0.048）",
            f"精确置换最小 p = {_f(pmin,3)}（{len(comp)} 个成分）", "最小值 >0.05 或 = 1/21",
            bool(pmin >= 1 / 21 - 1e-9),
            "audit/composition_tests_8wk.csv",
            "Welch 会给出 FDR<0.05 的成分，属 n=3 的正态假设，不采信")

    # ---------------------------------------------------------------- C12
    def pbr(ct, m):
        return (g(pb, "p_animal_level", celltype=ct, metric=m),
                g(pb, "d_animal_level", celltype=ct, metric=m))
    pe_, de_ = pbr("Endothelial", "LI_ConstitutiveHet")
    ps_, ds_ = pbr("SMC", "LI_ConstitutiveHet")
    pma, dma = pbr("Macrophage", "mod_AgingSen")
    add("C12", "个体水平上只有巨噬 AgingSen 与成纤维基质配体站得住，内皮与 SMC 不显著",
        f"EC p={_f(pe_,3)} d={_f(de_,2)}；SMC p={_f(ps_,3)} d={_f(ds_,2)}；Mac AgingSen p={_f(pma,4)} d={_f(dma,2)}",
        "EC/SMC p>0.05 且 Mac p<0.01", bool(pe_ > 0.05 and ps_ > 0.05 and pma < 0.01),
        "pseudobulk/cell_vs_pseudobulk_p.csv")

    # ---------------------------------------------------------------- C13
    if len(hum):
        mod_smc = g(hum, "mean", compartment="SMC", programme="ModSMC")
        mod_fib = float(hcs[hcs.compartment == "Fib"].ModSMC.mean()) if "ModSMC" in hcs else np.nan
        lam_mac = g(hum, "mean", compartment="Mac", programme="LAM")
        ec_ec = g(hum, "mean", compartment="EC", programme="ECpanel")
        mat_smc = g(hum, "mean", compartment="SMC", programme="Matrix")
        mat_fib = g(hum, "mean", compartment="Fib", programme="Matrix")
        add("C13", "三类程序在人体冠脉图谱里落到正确细胞类型",
            f"LAM：Mac {_f(lam_mac,2)}；ECpanel：EC {_f(ec_ec,2)}；"
            f"ModSMC：SMC {_f(mod_smc,2)} vs Fib {_f(mod_fib,2)}；"
            f"Matrix：SMC {_f(mat_smc,2)} vs Fib {_f(mat_fib,2)}",
            "每个程序在其目标细胞类型最高", False,
            "human/human_programme_summary.csv + human/human_celltype_scores.csv",
            "LAM 与 ECpanel 落对；ModSMC 在成纤维更高（已知反例）；"
            "Matrix 也在成纤维更高（0.687 vs 0.162）→ “基质配体落对”这句不成立，只保留“可检测”")

    # ---------------------------------------------------------------- C14
    if gen:
        orr = gen.get("odds_ratio")
        add("C14", "环境重塑的靶点与常见遗传变异的靶点不重叠（OR<1）",
            f"OR={_f(orr,3)}, Fisher p={gen.get('fisher_p'):.1e}, "
            f"面板 {gen.get('n_panel_assoc')} vs 背景 {gen.get('n_background_assoc')}",
            "OR<1 且 p<0.05", bool(orr < 1 and gen.get("fisher_p", 1) < 0.05),
            "genetics/trait_enrichment.json")

    # ---------------------------------------------------------------- C15
    if dom:
        dd = g(occ, "z", set="SMC_open", statistic="ligand_delta_mean")
        dp = g(occ, "p", set="SMC_open", statistic="ligand_delta_mean")
        dd2 = g(occ, "z", set="SMC_open[cov>=.5]", statistic="ligand_delta_mean")
        add("C15", "环五：SMC 开放 bin 的 AHR 占据没有配体特异性富集",
            f"配对差 z={_f(dd,2)}（p={_f(dp,2)}）；覆盖≥50% 后 z={_f(dd2,2)}",
            "z<0（即不高于背景）", bool(dd < 0),
            "chip_alt/occupancy_tests.csv",
            f"lift 率 {dom.get('lift_rate', float('nan')):.1%}；阳性对照只有 AHRR/CYP1A1 过 90 分位")

    # ------------------------------------------------- 16-20 donor-stratified R3
    r3 = _json("human/r3_summary.json")
    sc = _json("human/stromal_continuum_summary.json")
    fl = _json("power/floor_summary.json")
    if r3:
        wt = r3["within_donor_tests"]["LAM|Matrix > ECpanel|Matrix"]
        add("C16", "R3 在供体内部也成立：髓系–基质 > 内皮–基质",
            f"{wt['n_positive']}/{wt['n_donors']} 供体为正，中位差 {_f(wt['median_diff'],2)}，"
            f"Wilcoxon p={_f(wt['p_wilcoxon'],4)}",
            "≥7/8 供体为正且 p<0.05", bool(wt["n_positive"] >= 7 and wt["p_wilcoxon"] < 0.05),
            "human/r3_summary.json")
        pc = r3["partial_depth_ribo_mito_donor"]
        add("C17", "控制深度/核糖体/线粒体后，内皮–基质仍低于髓系–基质",
            f"偏 r：髓系–基质 {_f(pc['LAM|Matrix'])}, 内皮–基质 {_f(pc['ECpanel|Matrix'])}",
            "髓系–基质 > 内皮–基质", bool(pc["LAM|Matrix"] > pc["ECpanel|Matrix"]),
            "human/r3_summary.json")
        add("C18", "髓系–基质是三者中最紧的一对",
            f"偏 r：髓系–基质 {_f(pc['LAM|Matrix'])}, 内皮–髓系 {_f(pc['ECpanel|LAM'])}",
            "髓系–基质 为三者最大",
            bool(pc["LAM|Matrix"] > pc["ECpanel|LAM"]),
            "human/r3_summary.json",
            "扣除深度后 0.137 < 0.169；“最紧的一对”这句撤回，只保留“内皮与基质正交”")
    if sc:
        g_smc = {r["target"]: r for r in sc["human_correlations"] if r["compartment"] == "SMC"}
        t11 = g_smc.get("TNFRSF11B", {})
        add("C19", "人 SMC 内部 Matrix 随纤维肌细胞标志上升",
            f"TNFRSF11B 偏 r={_f(t11.get('partial_r'))}（CI {_f(t11.get('ci_lo'))}–{_f(t11.get('ci_hi'))}）",
            "r>0.1 且 CI 不含 0",
            bool(t11.get("partial_r", 0) > 0.1 and t11.get("ci_lo", 1) > 0),
            "human/stromal_continuum_summary.json")
        t22 = g_smc.get("MYH11", {})
        add("C20", "人 SMC 内部 Matrix 与 MYH11 强负相关（r<-0.3）",
            f"MYH11 偏 r={_f(t22.get('partial_r'))}；ACTA2 {_f(g_smc.get('ACTA2',{}).get('partial_r'))}",
            "r<-0.3", bool(t22.get("partial_r", 0) < -0.3),
            "human/stromal_continuum_summary.json",
            "实测 MYH11 −0.04、ACTA2 −0.21：收缩标志的负相关只是中等，"
            "梯度主要由纤维肌细胞端驱动，正文按此改写")
    if fl:
        add("C21", "3v3 的精确置换检验可以判定组成改变（p<0.05）",
            f"{_f(fl['exact_floor_3v3'],4)}（C(6,3)=20 种划分）",
            "floor<0.05 才可能拒绝", bool(fl["exact_floor_3v3"] < 0.05),
            "power/floor_summary.json",
            "0.143>0.05：3v3 的精确置换在任何效应量下都无法拒绝，"
            "最小可拒绝设计是 4v4（floor 0.042）")

    t = pd.DataFrame(ROWS)
    t.to_csv(os.path.join(OUT, "claim_audit.csv"), index=False)

    md = ["# Claim audit", "",
          f"共 {len(t)} 条；PASS {int((t.verdict=='PASS').sum())}，"
          f"REWRITE {int((t.verdict=='REWRITE').sum())}，"
          f"FAIL {int((t.verdict=='FAIL').sum())}", "",
          "| 编号 | 主张 | 重算值 | 判据 | 判定 | 来源 |", "|---|---|---|---|---|---|"]
    for r in t.itertuples():
        md.append(f"| {r.id} | {r.claim} | {r.value} | {r.rule} | **{r.verdict}** | `{r.source}` |")
    md.append("")
    md.append("## 需要改写的地方")
    for r in t[t.verdict != "PASS"].itertuples():
        md.append(f"- **{r.id} {r.verdict}**：{r.note}")
    with open(os.path.join(OUT, "claim_audit.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(t[["id", "verdict", "value"]].to_string(index=False))
    print("\n改写清单：")
    for r in t[t.verdict != "PASS"].itertuples():
        print(f"  {r.id} {r.verdict}: {r.note}")


if __name__ == "__main__":
    main()
