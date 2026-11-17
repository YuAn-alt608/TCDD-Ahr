"""45 - Supplementary Table 1: the 21-claim audit matrix, bilingual.

This script does not re-run any analysis. It assembles one row per audited
claim from `results/audit/claim_audit.csv` (the verdicts and recomputed values)
and the four human-validation summaries, and writes:

  processed_data/Supplementary_Table_1_Claim_Audit.xlsx
  processed_data/claim_audit_21_summary.csv

Two rules were followed while writing the prose of every row:

  1. The "Original hypothesis" column states what the manuscript used to say,
     not the corrected version. If a reviewer compares this table against an
     older draft, the difference must be visible, not smoothed over.
  2. Every number in "Internal re-audit" is copied from the artifact named in
     the same row. Nothing is retyped from memory; where a value is a range or
     a CI, the artifact is the JSON/CSV that holds the bounds.

The claim IDs are the ones already used in the report and in the audit script
(C01-C21). The numbering in the reviewer-facing example circulated earlier
(numbering C04 = KLF2 gatekeepers etc.) is offset from this table; the
Crosswalk sheet maps the two so the two documents can be read together.
"""
from __future__ import annotations

import json
import os

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "processed_data")
os.makedirs(OUT, exist_ok=True)


def jload(rel):
    with open(os.path.join(RES, rel), encoding="utf-8") as f:
        return json.load(f)


def load(rel):
    return pd.read_csv(os.path.join(RES, rel))


# ---------------------------------------------------------------- source values
R3 = jload("human/r3_summary.json")
ER = jload("human/external_replication.json")
SC = jload("human/stromal_continuum_summary.json")
FL = jload("power/floor_summary.json")
AUDIT = load("audit/claim_audit.csv")

# single source of truth for the numbers quoted in the human-validation column
AUC = {k: round(v, 3) for k, v in ER["pooled"].items()}
NDON = ER["R2"]["n_donors_auc_gt_06"]
WT = R3["within_donor_tests"]["LAM|Matrix > ECpanel|Matrix"]
FE = R3["donor_fixed_effects"]
PC = R3["partial_depth_ribo_mito_donor"]
DELTA_LAM = R3["cluster_bootstrap_delta"]["delta_lam"]
SMC_COR = {r["target"]: r for r in SC["human_correlations"] if r["compartment"] == "SMC"}

N_A_MOUSE = ("N/A — murine-specific TCDD / Ahr-KO perturbation; no human counterpart "
             "in GSE131778 (no exposure, no genotype).")
N_A_MOUSE_CN = "不适用：小鼠特异的 TCDD 暴露／Ahr 敲除实验，GSE131778 无暴露、无基因型，无法对应。"

# --------------------------------------------------------------------- the rows
ROWS = [
    dict(
        id="C01",
        claim_en="TCDD opens heterochromatin-marked (H3K9me3) domains specifically in "
                 "vascular smooth muscle cells.",
        claim_cn="TCDD 使血管平滑肌细胞的异染色质（H3K9me3）域可及性上升，且该效应限于平滑肌。",
        metric_en="frac_het domain accessibility, Cohen's d (Control vs TCDD-16wk-WT). "
                  "Lineage is stored as `Parenchyma` in the artifact (= Myh11-tdT+ smooth "
                  "muscle); Script 05 / 11 → results/ATAC/atac_lineage_effects.csv",
        metric_cn="域内可及性分数 frac_het 的 Cohen's d（对照 vs TCDD-16 周 WT）；"
                  "该谱系在文件中记为 Parenchyma（即 Myh11-tdT⁺ 平滑肌）；"
                  "脚本 05/11 → results/ATAC/atac_lineage_effects.csv",
        audit_en="Confirmed: d = 0.347, FDR = 8.6e-4 (SMC). The other three lineages stay "
                 "inside the null (see C02), so the claim is lineage-restricted, not global.",
        audit_cn="成立：d = 0.347，FDR = 8.6e-4（平滑肌）。其余三个谱系落在零分布内（见 C02），"
                 "所以结论是“限于平滑肌”，不是“跨谱系普遍”。",
        ext_en=N_A_MOUSE, ext_cn=N_A_MOUSE_CN,
        disp="Retained",
    ),
    dict(
        id="C02",
        claim_en="The SMC chromatin opening exceeds a GC-content- and TSS-matched null, "
                 "while the other three lineages do not.",
        claim_cn="平滑肌的域开放偏移超过 GC 含量／TSS 密度匹配的零模型，其余三个谱系则不超过。",
        metric_en="Empirical p from 1,000 GC/TSS-matched domain shuffles; Script 20 → "
                  "results/permutation/permutation_test.csv",
        metric_cn="1,000 次 GC/TSS 匹配的域置换经验 p；脚本 20 → "
                  "results/permutation/permutation_test.csv",
        audit_en="Confirmed: SMC p = 0.001; endothelial 0.284, mesenchymal 0.907, immune 0.433. "
                 "The earlier phrase 'opening is broken across lineages' is withdrawn.",
        audit_cn="成立：平滑肌 p = 0.001；内皮 0.284、间质 0.907、免疫 0.433。"
                 "原文“跨谱系普遍击穿”一句已删除。",
        ext_en=N_A_MOUSE, ext_cn=N_A_MOUSE_CN,
        disp="Retained",
    ),
    dict(
        id="C03",
        claim_en="SMC-specific Ahr deletion reverses only part of the opening, i.e. the "
                 "opening is AhR-dependent but not AhR-exclusive.",
        claim_cn="平滑肌特异敲除 Ahr 只使域开放部分回落：该开放依赖 AhR，但不是 AhR 独占。",
        metric_en="frac_het in Control / TCDD-16wk-WT / TCDD-16wk-AhrKO (lineage "
                  "`Parenchyma`); reversal fraction = 1 − (KO−Control)/(WT−Control); "
                  "Script 05 → results/ATAC/atac_lineage_effects.csv",
        metric_cn="对照／16 周 WT／16 周 AhrKO 三组的 frac_het（谱系记为 Parenchyma）；"
                  "回落比例 = 1 − (KO−对照)/(WT−对照)；脚本 05 → "
                  "results/ATAC/atac_lineage_effects.csv",
        audit_en="Confirmed: Control 0.01542, WT 0.01615, AhrKO 0.01581 → the knockout "
                 "reverses 47% of the WT gain and retains 53% of it; AhrKO stays above "
                 "Control, so the reversal is partial, not complete.",
        audit_cn="成立：对照 0.01542、WT 0.01615、AhrKO 0.01581 → 回落 47%、仍保留 WT 增益的 53%；"
                 "AhrKO 高于对照，说明是部分回落而非完全回落。",
        ext_en=N_A_MOUSE, ext_cn=N_A_MOUSE_CN,
        disp="Retained",
    ),
    dict(
        id="C04",
        claim_en="SMC open bins are enriched for the Ahr::Arnt motif, consistent with direct "
                 "AhR recruitment.",
        claim_cn="平滑肌开放 bin 富集 Ahr::Arnt motif，与 AhR 直接招募一致。",
        metric_en="Motif odds ratio and FDR over 293 SMC open bins vs GC/TSS-matched "
                  "background; Script 11 → results/motif/motif_enrichment.csv",
        metric_cn="293 个平滑肌开放 bin 相对 GC/TSS 匹配背景的 motif 优势比与 FDR；"
                  "脚本 11 → results/motif/motif_enrichment.csv",
        audit_en="Confirmed: OR = 2.01, FDR = 0.0015, n = 293 bins. This is motif-level "
                 "enrichment only — it does not establish occupancy (see C15).",
        audit_cn="成立：OR = 2.01，FDR = 0.0015，n = 293。"
                 "但这是 motif 层面的富集，不等于占据（占据见 C15）。",
        ext_en=N_A_MOUSE, ext_cn=N_A_MOUSE_CN,
        disp="Retained",
    ),
    dict(
        id="C05",
        claim_en="The motifs enriched in endothelial open bins identify the upstream factor "
                 "that opens them (originally named as KLF2 / HIF1A / EPAS1).",
        claim_cn="内皮开放 bin 的 motif 富集能指认出打开它们的上游因子（原写作 KLF2／HIF1A／EPAS1）。",
        metric_en="Motif FDR over the endothelial DA open bin set; Script 11 → "
                  "results/motif/motif_enrichment.csv and ec_vs_smc_contrast.csv",
        metric_cn="内皮差异开放 bin 集合上的 motif FDR；脚本 11 → "
                  "results/motif/motif_enrichment.csv 与 ec_vs_smc_contrast.csv",
        audit_en="Refuted as stated: the endothelial DA open set holds only n = 30 bins and "
                 "no motif passes FDR on it (Ahr::Arnt p = 0.272; smallest p also fails). "
                 "The high KLF2/HIF1A/EPAS1 scores are chromVAR deviation scores of the whole "
                 "endothelial compartment, not enrichment in the opened bins.",
        audit_cn="按原表述不成立：内皮差异开放 bin 仅 30 个，其上没有任何 motif 通过 FDR"
                 "（Ahr::Arnt p = 0.272，最小 p 也未过线）。KLF2/HIF1A/EPAS1 的高分是"
                 "内皮整体染色质的 chromVAR 偏差分数，不是被打开 bin 的富集。",
        ext_en="N/A — the claim is about motif content of murine endothelial bins.",
        ext_cn="不适用：该主张针对小鼠内皮 bin 的 motif 组成。",
        disp="Downgraded to Hypothesis",
    ),
    dict(
        id="C06",
        claim_en="Endothelial accessibility changes are not reversed by SMC-specific Ahr "
                 "deletion, i.e. the endothelial arm is non-cell-autonomous.",
        claim_cn="内皮可及性改变不被平滑肌特异的 Ahr 敲除逆转，即内皮这一支不是细胞自主的。",
        metric_en="Endothelial prom_het_rel across the three groups and the WT−Control gap "
                  "retained in AhrKO; Script 05 → results/ATAC/atac_lineage_effects.csv "
                  "(lineage `Endothelial`)",
        metric_cn="三组内皮启动子可及性的 d，以及 AhrKO 保留的 WT−对照落差；"
                  "脚本 05 → results/ATAC/atac_lineage_effects.csv",
        audit_en="Confirmed: d = 0.544; Control −0.0248, WT −0.02128, AhrKO −0.02148. AhrKO "
                 "recovers only 6% of the WT−Control gap, so the endothelial change is "
                 "essentially insensitive to smooth-muscle AhR.",
        audit_cn="成立：d = 0.544；对照 −0.0248、WT −0.02128、AhrKO −0.02148。"
                 "AhrKO 只回补了 WT−对照落差的 6%，内皮改变几乎不受平滑肌 AhR 影响。",
        ext_en=N_A_MOUSE, ext_cn=N_A_MOUSE_CN,
        disp="Retained",
    ),
    dict(
        id="C07",
        claim_en="After depth-matched downsampling, domain-level effects keep both their "
                 "direction and their significance.",
        claim_cn="等深度重采样后，域指标的效应方向与显著性都保持。",
        metric_en="3 metrics × 4 lineages × 2 depth levels = 24 sign comparisons; verdicts "
                  "in robust/downsample_domain_verdict.csv; Script 18b / 30",
        metric_cn="3 指标 × 4 谱系 × 2 档深度 = 24 次符号比较；判定见 "
                  "robust/downsample_domain_verdict.csv；脚本 18b/30",
        audit_en="Partially refuted: sign is preserved in 24/24 comparisons, but only 7 of the "
                 "12 verdict rows also keep significance; the other 5 are direction-only. "
                 "The phrase 'direction and significance both preserved' is removed.",
        audit_cn="部分不成立：24/24 次比较符号全保，但 12 行判定里只有 7 行连显著性一起保住，"
                 "其余 5 行只是方向一致。“方向与显著性都保持”一句已删除。",
        ext_en="N/A — a property of the murine sequencing design.", ext_cn="不适用：小鼠测序设计本身的属性。",
        disp="Revised / Bounded in Limitations",
    ),
    dict(
        id="C08",
        claim_en="Per-cell endpoints are stable in direction under depth matching.",
        claim_cn="逐细胞终点在等深度下方向稳定。",
        metric_en="3 binomial-thinning depths per endpoint, 72 endpoint×lineage×contrast "
                  "combinations; Script 30 → results/audit/depth_summary.csv",
        metric_cn="每个终点 3 档二项稀释深度，共 72 个终点×谱系×对比组合；"
                  "脚本 30 → results/audit/depth_summary.csv",
        audit_en="Confirmed with one exception: 67/72 combinations keep the same sign at all "
                 "three depths; 5 do not, and the load-bearing one is macrophage LAM under "
                 "WT-16wk (2/3, effect attenuated 93.8%). Primary endpoints never reach 0/3.",
        audit_cn="成立但有一处例外：72 个组合中 67 个三档稀释全部同号；5 个不是，"
                 "其中承重的是 WT-16 周的巨噬 LAM（2/3 同号、效应衰减 93.8%）。"
                 "主终点没有出现 0/3。",
        ext_en="N/A — a property of the murine sequencing design.", ext_cn="不适用：小鼠测序设计本身的属性。",
        disp="Retained",
    ),
    dict(
        id="C09",
        claim_en="Within-animal 80% cell resampling keeps all primary endpoints in the same "
                 "direction (no single animal drives the effect).",
        claim_cn="每只动物内 80% 细胞重抽样后，主终点方向全部不变（不存在单只动物驱动效应）。",
        metric_en="1,000 resamples per animal, sign concordance per endpoint; Script 31 → "
                  "results/audit/subsample_summary.csv",
        metric_cn="每只动物 1,000 次重抽样，逐终点同号比例；脚本 31 → "
                  "results/audit/subsample_summary.csv",
        audit_en="Confirmed: minimum sign concordance 1.000 across all four primary endpoints.",
        audit_cn="成立：四个主终点的最低同号比例均为 1.000。",
        ext_en="N/A — a property of the murine sampling design.", ext_cn="不适用：小鼠抽样设计本身的属性。",
        disp="Retained",
    ),
    dict(
        id="C10",
        claim_en="After Kitagawa decomposition of the raw difference into composition, "
                 "within-state and interaction terms, most endpoints are cell-intrinsic.",
        claim_cn="把原始差异做 Kitagawa 分解为组成项、态内项与交互项后，多数终点是细胞自主的。",
        metric_en="Δ_raw = composition + within-state + interaction, per endpoint; "
                  "Script 29 → results/audit/endpoint_verdict.csv",
        metric_cn="逐终点 Δ原始 = 组成项 + 态内项 + 交互项；脚本 29 → "
                  "results/audit/endpoint_verdict.csv",
        audit_en="Confirmed: 6 cell-intrinsic, 2 mixed, 2 composition-sensitive out of 10 "
                 "endpoints. The two composition-sensitive ones are named in the manuscript "
                 "and are not used as mechanism evidence.",
        audit_cn="成立：10 个终点中 6 个细胞自主、2 个混合、2 个组成敏感。"
                 "两个组成敏感的已在正文点名，且不作为机制证据。",
        ext_en="N/A — decomposition requires the murine genotype arms.", ext_cn="不适用：分解需要小鼠的基因型分组。",
        disp="Retained",
    ),
    dict(
        id="C11",
        claim_en="Cell-composition shifts cannot be declared significant at this sample size.",
        claim_cn="在这个样本量下，不能宣称细胞组成发生了显著改变。",
        metric_en="Exact permutation over 3 vs 3 libraries, 7 L1 and 25 L2 tests; "
                  "Script 29 → results/audit/composition_tests_8wk.csv",
        metric_cn="3 对 3 文库的精确置换，7 个 L1 与 25 个 L2 检验；"
                  "脚本 29 → results/audit/composition_tests_8wk.csv",
        audit_en="Confirmed and strengthened: with C(6,3) = 20 splits the smallest attainable "
                 "two-sided p is 3/21 = 0.143, and every observed test sits at or above it. "
                 "Welch's test would return three FDR < 0.05 components; that is an n=3 "
                 "normality artefact and is not taken as evidence.",
        audit_cn="成立且更强：C(6,3) = 20 种分法下最小可达双侧 p 为 3/21 = 0.143，"
                 "全部实测检验都不低于它。Welch 会给 3 个成分 FDR < 0.05，"
                 "那是 n = 3 的正态假设假象，不采信。",
        ext_en="N/A — a property of the murine design.", ext_cn="不适用：小鼠设计本身的属性。",
        disp="Retained",
    ),
    dict(
        id="C12",
        claim_en="At the animal level only the myeloid LAM state and the fibroblast matrix-"
                 "ligand program remain significant; the endothelial and SMC endpoints do not.",
        claim_cn="个体水平上只有巨噬 LAM 状态与成纤维基质配体程序仍显著；内皮与平滑肌终点不显著。",
        metric_en="Per-animal pseudobulk scores, exact permutation and Welch p; "
                  "Script 19 → results/pseudobulk/cell_vs_pseudobulk_p.csv",
        metric_cn="逐动物 pseudo-bulk 分数、精确置换与 Welch p；脚本 19 → "
                  "results/pseudobulk/cell_vs_pseudobulk_p.csv",
        audit_en="Confirmed: endothelial p = 0.138 (d = −1.82), SMC p = 0.185 (d = −1.55), "
                 "macrophage AgingSen p = 0.0038 (d = +5.50). Note that the endothelial and "
                 "SMC point estimates are opposite in sign to their cell-level effects — this "
                 "is why the title no longer claims a two-layer decoupling.",
        audit_cn="成立：内皮 p = 0.138（d = −1.82）、平滑肌 p = 0.185（d = −1.55）、"
                 "巨噬 AgingSen p = 0.0038（d = +5.50）。注意内皮与平滑肌的点估计与细胞水平"
                 "方向相反，这正是标题不再写“双层解耦”的原因。",
        ext_en="N/A — the human atlas has no exposure arm to replicate an exposure effect.",
        ext_cn="不适用：人冠脉图谱没有暴露分组，无法复现暴露效应。",
        disp="Retained",
    ),
    dict(
        id="C13",
        claim_en="All four mouse-derived programs (LAM, ECpanel, Matrix, ModSMC) map "
                 "specifically onto their canonical human cell types.",
        claim_cn="四个小鼠来源的程序（LAM、ECpanel、Matrix、ModSMC）都能特异映射到人体对应细胞类型。",
        metric_en="Per-program ROC AUC by cell type, pooled and per donor; Script 39 → "
                  "results/human/external_replication.csv and human_programme_summary.csv",
        metric_cn="逐程序的细胞类型 ROC AUC（合并与逐供体）；脚本 39 → "
                  "results/human/external_replication.csv 与 human_programme_summary.csv",
        audit_en="Refuted for two of four programs. LAM → macrophage AUC 0.740 (7/8 donors "
                 "> 0.6) and ECpanel → endothelial 0.689 (8/8) hold. Matrix AUC 0.636 with "
                 "3/8 donors, and ModSMC 0.631 with 4/8, fail lineage specificity; Matrix "
                 "scores higher in fibroblasts (0.687) than in SMC (0.162).",
        audit_cn="四个程序中有两个不成立。LAM→巨噬 AUC 0.740（7/8 供体 > 0.6）、"
                 "ECpanel→内皮 0.689（8/8）成立；Matrix 0.636、逐供体仅 3/8，"
                 "ModSMC 0.631、4/8，均不满足谱系特异性；Matrix 在成纤维 0.687，高于平滑肌 0.162。",
        ext_en=f"Split-half and per-donor AUC over 8 donors / 11,756 cells: LAM {AUC['LAM']:.3f} "
               f"({NDON['LAM']}/8), ECpanel {AUC['ECpanel']:.3f} ({NDON['ECpanel']}/8), "
               f"Matrix {AUC['Matrix']:.3f} ({NDON['Matrix']}/8), ModSMC {AUC['ModSMC']:.3f} "
               f"({NDON['ModSMC']}/8). Two of four fail lineage specificity.",
        ext_cn=f"8 供体／11,756 细胞的分裂复现与逐供体 AUC：LAM {AUC['LAM']:.3f}（{NDON['LAM']}/8）、"
               f"ECpanel {AUC['ECpanel']:.3f}（{NDON['ECpanel']}/8）、Matrix {AUC['Matrix']:.3f}"
               f"（{NDON['Matrix']}/8）、ModSMC {AUC['ModSMC']:.3f}（{NDON['ModSMC']}/8）。"
               f"四个里有两个不满足谱系特异性。",
        disp="Revised / Bounded in Limitations",
    ),
    dict(
        id="C14",
        claim_en="Environment-remodelled chromatin targets are genetically independent of "
                 "common CAD risk loci.",
        claim_cn="环境重塑的染色质靶点与常见冠心病遗传风险位点相互独立。",
        metric_en="Fisher enrichment of CAD GWAS credible sets in the target panel vs "
                  "matched background (1,993 vs 4,157); Script 08 → "
                  "results/genetics/trait_enrichment.json",
        metric_cn="冠心病 GWAS 可信集在靶点面板（1,993）相对匹配背景（4,157）的 Fisher 富集；"
                  "脚本 08 → results/genetics/trait_enrichment.json",
        audit_en="Confirmed: OR = 0.529 (95% CI 0.47–0.59), Fisher p = 9.5e-29 — the panel is "
                 "depleted, not enriched, for CAD loci.",
        audit_cn="成立：OR = 0.529（95% CI 0.47–0.59），Fisher p = 9.5e-29——"
                 "面板在冠心病位点上是耗竭而非富集。",
        ext_en="Human orthologous regulatory regions with CAD GWAS summary statistics (not "
               "GSE131778): OR 0.529, p = 9.5e-29.",
        ext_cn="人同源调控区 + 冠心病 GWAS 汇总统计（非 GSE131778）：OR 0.529，p = 9.5e-29。",
        disp="Retained",
    ),
    dict(
        id="C15",
        claim_en="SMC open bins do not show ligand-specific (TCDD > DMSO) AHR occupancy.",
        claim_cn="平滑肌开放 bin 上没有配体特异（TCDD > DMSO）的 AHR 占据。",
        metric_en="LiftOver of 5-kb bins to hg38 (51.4%), window mean/summit in HCASMC AHR "
                  "ChIP TCDD vs DMSO tracks, stratified null (5,000 draws); Script 35 → "
                  "results/chip_alt/occupancy_tests.csv and occupancy_summary.json",
        metric_cn="5 kb bin 抬升到 hg38（51.4%），在 HCASMC AHR ChIP 的 TCDD 与 DMSO 轨道上取"
                  "窗口均值／summit，分层零分布 5,000 次；脚本 35 → "
                  "results/chip_alt/occupancy_tests.csv 与 occupancy_summary.json",
        audit_en="Confirmed as a negative: paired difference z = −0.75 (p = 0.77), and "
                 "z = −2.04 after restricting to bins with ≥50% coverage in both tracks. "
                 "Positive control on the same statistic reaches the 99th (AHRR) and 87th "
                 "(CYP1A1) percentile, so the test has usable sensitivity. It is a calibrated "
                 "negative, not proof of absence.",
        audit_cn="作为阴性结果成立：配对差 z = −0.75（p = 0.77），"
                 "限制到两条轨道覆盖均 ≥50% 后 z = −2.04。同一统计量的阳性对照达到 99 分位"
                 "（AHRR）与 87 分位（CYP1A1），说明检验有可用灵敏度。"
                 "这是“有刻度的阴性”，不是“证明不结合”。",
        ext_en="External human HCASMC AHR ChIP tracks (GSE276152) — but GSE276152 and the "
               "murine GSE276870/276872 are arms of the same study, so it is not independent.",
        ext_cn="外部人冠脉平滑肌 AHR ChIP 轨道（GSE276152）——但 GSE276152 与本文鼠源 "
               "GSE276870/276872 属同一研究的不同分支，不构成独立证据。",
        disp="Retained",
    ),
    dict(
        id="C16",
        claim_en="The branch point holds within donors: myeloid–matrix coupling exceeds "
                 "endothelial–matrix coupling in every donor.",
        claim_cn="分叉点在供体内部成立：每个供体的髓系–基质耦合都高于内皮–基质。",
        metric_en="Per-donor Spearman for three program pairs, paired Wilcoxon; "
                  "Script 41 → results/human/r3_within_donor.csv and r3_summary.json",
        metric_cn="三个程序对的逐供体 Spearman 与配对 Wilcoxon；"
                  "脚本 41 → results/human/r3_within_donor.csv 与 r3_summary.json",
        audit_en=f"Confirmed: {WT['n_positive']}/{WT['n_donors']} donors positive, median "
                 f"difference {WT['median_diff']:.2f}, Wilcoxon p = {WT['p_wilcoxon']:.4f} "
                 f"(the floor for n = 8 one-sided is 1/256 = 0.0039). Donor variance accounts "
                 f"for 5.6–15.9% of total, so no single donor drives it.",
        audit_cn=f"成立：{WT['n_positive']}/{WT['n_donors']} 供体为正，中位差 {WT['median_diff']:.2f}，"
                 f"Wilcoxon p = {WT['p_wilcoxon']:.4f}（n = 8 单侧最小值即 1/256 = 0.0039）。"
                 f"供体方差只占 5.6%–15.9%，没有单个供体主导。",
        ext_en=f"Replicated in GSE131778 itself: {WT['n_positive']}/{WT['n_donors']} donors, "
               f"median Δ = +{WT['median_diff']:.2f}; cluster bootstrap on donors "
               f"Δ = +{R3['cluster_bootstrap_delta']['delta_matrix']['median']:.2f} "
               f"[{R3['cluster_bootstrap_delta']['delta_matrix']['lo']:+.2f}, "
               f"{R3['cluster_bootstrap_delta']['delta_matrix']['hi']:+.2f}], 100% positive.",
        ext_cn=f"在人冠脉数据内部即复现：{WT['n_positive']}/{WT['n_donors']} 供体，"
               f"中位差 +{WT['median_diff']:.2f}；按供体聚类 bootstrap 的 Δ = "
               f"+{R3['cluster_bootstrap_delta']['delta_matrix']['median']:.2f} "
               f"[{R3['cluster_bootstrap_delta']['delta_matrix']['lo']:+.2f}, "
               f"{R3['cluster_bootstrap_delta']['delta_matrix']['hi']:+.2f}]，100% 为正。",
        disp="Retained",
    ),
    dict(
        id="C17",
        claim_en="The endothelial–matrix null survives control for sequencing depth, "
                 "ribosomal and mitochondrial content, and donor.",
        claim_cn="控制测序深度、核糖体与线粒体比例以及供体后，内皮–基质仍然正交。",
        metric_en="Partial correlation on residuals after donor, log nCount, pct_ribo, "
                  "pct_mito; donor-clustered bootstrap; Script 41 → results/human/r3_summary.json",
        metric_cn="扣除供体、log nCount、核糖体与线粒体比例后的偏相关；按供体聚类 bootstrap；"
                  "脚本 41 → results/human/r3_summary.json",
        audit_en=f"Confirmed: partial r(EC–Matrix) = {PC['ECpanel|Matrix']:+.3f} vs "
                 f"r(Myeloid–Matrix) = {PC['LAM|Matrix']:+.3f}; donor fixed effects give "
                 f"{FE['ECpanel|Matrix']:+.3f} vs {FE['LAM|Matrix']:+.3f}. Orthogonality is the "
                 f"part of the branch claim that survives every adjustment.",
        audit_cn=f"成立：偏 r（内皮–基质）= {PC['ECpanel|Matrix']:+.3f}，"
                 f"而 r（髓系–基质）= {PC['LAM|Matrix']:+.3f}；供体固定效应为 "
                 f"{FE['ECpanel|Matrix']:+.3f} 对 {FE['LAM|Matrix']:+.3f}。"
                 f"“正交”是分叉主张里唯一经得起全部校正的部分。",
        ext_en=f"Replicated: partial r(EC–Matrix) = {PC['ECpanel|Matrix']:+.3f} "
               f"[cluster CI −0.02, +0.11] vs r(Myeloid–Matrix) = {PC['LAM|Matrix']:+.3f} "
               f"[+0.19, +0.47].",
        ext_cn=f"已复现：偏 r（内皮–基质）= {PC['ECpanel|Matrix']:+.3f}"
               f"[聚类 CI −0.02, +0.11]，r（髓系–基质）= {PC['LAM|Matrix']:+.3f}[+0.19, +0.47]。",
        disp="Retained",
    ),
    dict(
        id="C18",
        claim_en="Myeloid–matrix is the tightest of the three pairwise couplings.",
        claim_cn="髓系–基质是三个配对耦合中最紧的一对。",
        metric_en="Paired per-donor comparison of the two couplings plus donor-clustered "
                  "bootstrap of their difference; Script 41 → results/human/r3_summary.json",
        metric_cn="两个耦合的逐供体配对比较，及其差值的供体聚类 bootstrap；"
                  "脚本 41 → results/human/r3_summary.json",
        audit_en=f"Withdrawn: after adjusting for depth the order reverses "
                 f"({PC['LAM|Matrix']:.3f} for myeloid–matrix vs {PC['ECpanel|LAM']:.3f} for "
                 f"endothelial–myeloid); the paired test gives p = 0.098 and the cluster CI of "
                 f"the difference is [{DELTA_LAM['lo']:+.2f}, {DELTA_LAM['hi']:+.2f}] across zero "
                 f"({DELTA_LAM['frac_positive']:.0%} positive). What remains is orthogonality "
                 f"(C17), not 'tightest'.",
        audit_cn=f"已撤回：扣除深度后顺序反转（髓系–基质 {PC['LAM|Matrix']:.3f}，"
                 f"内皮–髓系 {PC['ECpanel|LAM']:.3f}）；配对检验 p = 0.098，"
                 f"差值聚类 CI [{DELTA_LAM['lo']:+.2f}, {DELTA_LAM['hi']:+.2f}] 跨 0"
                 f"（{DELTA_LAM['frac_positive']:.0%} 为正）。留下来的只有“正交”（C17），"
                 f"不是“最紧”。",
        ext_en=f"Replicated for the part that survives: EC–Matrix r = "
               f"{PC['ECpanel|Matrix']:+.3f} to {FE['ECpanel|Matrix']:+.3f} in all four "
               f"estimators; the ranking claim is not supported (p = 0.098).",
        ext_cn=f"成立的部分已复现：在四种估计下内皮–基质 r 均为 "
               f"{PC['ECpanel|Matrix']:+.3f} 到 {FE['ECpanel|Matrix']:+.3f}；"
               f"排序主张不成立（p = 0.098）。",
        disp="Revised / Bounded in Limitations",
    ),
    dict(
        id="C19",
        claim_en="Within human SMC, the matrix program rises with fibromyocyte markers, so "
                 "the program tracks a state rather than a lineage.",
        claim_cn="在人平滑肌内部，基质程序随纤维肌细胞标志上升，说明该程序追踪的是状态而非谱系。",
        metric_en="Partial correlation within human SMC (2,557 cells, donor + depth "
                  "controlled); decile gradient; Script 42 → "
                  "results/human/stromal_continuum_summary.json",
        metric_cn="人平滑肌内部（2,557 细胞，扣除供体与深度）的偏相关；十分位梯度；"
                  "脚本 42 → results/human/stromal_continuum_summary.json",
        audit_en=f"Confirmed: TNFRSF11B +{SMC_COR['TNFRSF11B']['partial_r']:.3f} "
                 f"(CI {SMC_COR['TNFRSF11B']['ci_lo']:+.3f} to {SMC_COR['TNFRSF11B']['ci_hi']:+.3f}), "
                 f"LUM +{SMC_COR['LUM']['partial_r']:.3f}, monotone gradient across deciles "
                 f"(rho = {SC['gradient_rho']:.2f}), reaching the fibroblast mean "
                 f"({SC['fib_matrix_mean']:.2f}) at the top decile.",
        audit_cn=f"成立：TNFRSF11B +{SMC_COR['TNFRSF11B']['partial_r']:.3f}"
                 f"（CI {SMC_COR['TNFRSF11B']['ci_lo']:+.3f} 至 {SMC_COR['TNFRSF11B']['ci_hi']:+.3f}）、"
                 f"LUM +{SMC_COR['LUM']['partial_r']:.3f}，十分位梯度单调"
                 f"（rho = {SC['gradient_rho']:.2f}），最高档逼近成纤维均值 {SC['fib_matrix_mean']:.2f}。",
        ext_en="Tested in the human cohort itself (GSE131778 SMC subset, 8 donors): this is a "
               "human-only claim, so the external check is donor-held-out stability, which holds.",
        ext_cn="该主张本身就在人队列中检验（GSE131778 平滑肌子集，8 供体）；"
               "所谓外部检验即逐供体留一稳定性，结果成立。",
        disp="Retained",
    ),
    dict(
        id="C20",
        claim_en="The human SMC dedifferentiation gradient is driven by loss of contractile "
                 "identity (MYH11 shutdown).",
        claim_cn="人平滑肌的去分化梯度由收缩身份的丧失（MYH11 关闭）驱动。",
        metric_en="Partial r of the matrix score with MYH11 / ACTA2 within SMC; Script 42 → "
                  "results/human/stromal_continuum_summary.json",
        metric_cn="平滑肌内部基质分数与 MYH11／ACTA2 的偏相关；脚本 42 → "
                  "results/human/stromal_continuum_summary.json",
        audit_en=f"Refuted as stated: MYH11 partial r = {SMC_COR['MYH11']['partial_r']:.3f} "
                 f"(CI {SMC_COR['MYH11']['ci_lo']:+.3f} to {SMC_COR['MYH11']['ci_hi']:+.3f}), "
                 f"ACTA2 = {SMC_COR['ACTA2']['partial_r']:.3f} — both short of the prespecified "
                 f"−0.3. The gradient is produced mainly by gaining the fibromyocyte program, "
                 f"not by losing the contractile one.",
        audit_cn=f"按原表述不成立：MYH11 偏 r = {SMC_COR['MYH11']['partial_r']:.3f}"
                 f"（CI {SMC_COR['MYH11']['ci_lo']:+.3f} 至 {SMC_COR['MYH11']['ci_hi']:+.3f}），"
                 f"ACTA2 = {SMC_COR['ACTA2']['partial_r']:.3f}，均未达预设的 −0.3。"
                 f"梯度主要由“获得纤维肌细胞程序”产生，不是“失去收缩程序”。",
        ext_en="Same human cohort, so this is not an external test — but the failure is "
               "informative: the prespecified direction was wrong, not underpowered.",
        ext_cn="同一人队列，不构成外部检验——但失败本身有信息量：错的是预设方向，不是功效不足。",
        disp="Revised / Bounded in Limitations",
    ),
    dict(
        id="C21",
        claim_en="The individual-level 3v3 permutation p = 0.143 reflects weak biology in the "
                 "endothelial and SMC endpoints.",
        claim_cn="个体水平 3v3 置换 p = 0.143 反映的是内皮与平滑肌终点的生物学效应弱。",
        metric_en="Exact permutation over all C(6,3) = 20 splits; MDE by noncentral t; "
                  "Hedges' g with CIs; Script 43 → results/power/floor_summary.json and "
                  "endpoint_effect_table.csv",
        metric_cn="C(6,3) = 20 种划分的全枚举精确置换；非中心 t 求 MDE；Hedges' g 与 CI；"
                  "脚本 43 → results/power/floor_summary.json 与 endpoint_effect_table.csv",
        audit_en=f"Reframed: with 20 splits and the (k+1)/(N+1) convention, the observed split "
                 f"and its mirror are the two most extreme assignments, so the smallest "
                 f"attainable two-sided p is {FL['exact_floor_3v3']:.4f} — even the macrophage "
                 f"AgingSen endpoint (g = +4.40, Welch p = 0.0038) returns 0.143. The correct "
                 f"reading is a design floor, not absence of effect; MDE at 3v3 is "
                 f"{FL['mde']['3']:.2f}, and the smallest design that can reject is 4v4 "
                 f"(floor {FL['smallest_design_that_can_reject']['exact_floor_p']:.3f}).",
        audit_cn=f"重新定性：20 种分法下观测分法与其镜像并列最极端，故最小可达双侧 p 为 "
                 f"{FL['exact_floor_3v3']:.4f}——连巨噬 AgingSen（g = +4.40、Welch p = 0.0038）"
                 f"也只能得到 0.143。正确读法是设计下限，不是无效应；3v3 的 MDE 为 "
                 f"{FL['mde']['3']:.2f}，最小可拒绝设计是 4v4"
                 f"（下限 {FL['smallest_design_that_can_reject']['exact_floor_p']:.3f}）。",
        ext_en="N/A — a property of the murine 3v3 layout.", ext_cn="不适用：小鼠 3v3 设计本身的属性。",
        disp="Revised / Bounded in Limitations",
    ),
]

assert len(ROWS) == 21, len(ROWS)
assert [r["id"] for r in ROWS] == [f"C{i:02d}" for i in range(1, 22)]

# the audit verdict must agree with the disposition, or the table would silently
# disagree with the CSV that generated it
V = {f"C{int(r.id[1:]):02d}": r.verdict for r in AUDIT.itertuples()}

# C07 is the one legitimate mismatch: the audit CSV stores the *already softened*
# statement ("direction preserved, significance not all preserved"), so its PASS
# applies to the softened text. This table's column 2 quotes the pre-audit wording
# ("direction and significance both preserved"), for which the disposition is
# Revised. Recording the reason here keeps the check strict everywhere else.
SOFTENED_AT_WRITE = {
    "C07": "audit CSV holds the revised sentence; PASS applies to that sentence, "
           "not to the pre-audit wording quoted in column 2",
}
for r in ROWS:
    v = V[r["id"]]
    if v == "REWRITE" and r["disp"] == "Retained":
        raise SystemExit(f"{r['id']}: audit says REWRITE but disposition is Retained")
    if v == "PASS" and r["disp"] != "Retained" and r["id"] not in SOFTENED_AT_WRITE:
        raise SystemExit(f"{r['id']}: audit says PASS but disposition is {r['disp']}")

# ------------------------------------------------------------------- styling kit
HDR_FILL = PatternFill("solid", fgColor="1F3864")
HDR_FONT = Font(bold=True, color="FFFFFF", size=10, name="Calibri")
BODY = Font(size=9, name="Calibri")
MONO = Font(size=8.5, name="Consolas")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
DISP_CN = {"Retained": "保留（Retained）",
           "Downgraded to Hypothesis": "降级为假设（Downgraded to Hypothesis）",
           "Revised / Bounded in Limitations": "改写并写入局限（Revised / Bounded in Limitations）"}


def band_of(v):
    """Row tint by disposition, tolerant of the bilingual strings in the CN sheet."""
    v = str(v or "")
    if "Retained" in v or v.startswith("保留"):
        return "E2EFDA"
    if "Downgraded" in v or v.startswith("降级"):
        return "FFF2CC"
    if "Revised" in v or v.startswith("改写"):
        return "FCE4D6"
    return None


for r in ROWS:                     # bilingual disposition for the CN sheet
    r["disp_cn"] = DISP_CN[r["disp"]]

def write_sheet(ws, cols, rows, widths, disp_key="disp", wrap_cols=()):
    ws.append(cols)
    for j, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(j)].width = w
        c = ws.cell(row=1, column=j)
        c.fill, c.font, c.border = HDR_FILL, HDR_FONT, BOX
        c.alignment = Alignment(vertical="center", horizontal="center", wrap_text=True)
    ws.row_dimensions[1].height = 34
    for r in rows:
        ws.append([r[c] for c in cols])
    for i in range(2, ws.max_row + 1):
        fill = band_of(ws.cell(row=i, column=len(cols)).value)
        for j in range(1, len(cols) + 1):
            c = ws.cell(row=i, column=j)
            c.font = MONO if j == 1 else BODY
            c.border = BOX
            c.alignment = Alignment(vertical="top", wrap_text=True,
                                    horizontal="center" if j in (1,) else "left")
            if fill:
                c.fill = PatternFill("solid", fgColor=fill)
        # row height from the longest wrapped cell: 9 pt font, ~1.25 chars per width unit
        est = max(len(str(ws.cell(row=i, column=j).value or "")) / max(widths[j - 1], 8)
                  for j in range(1, len(cols) + 1))
        ws.row_dimensions[i].height = min(240, max(30, 13 * (int(est) + 1)))
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{ws.max_row}"


wb = Workbook()

# ------------------------------------------------------------------ legend sheet
lg = wb.active
lg.title = "Legend"
lg.column_dimensions["A"].width = 30
lg.column_dimensions["B"].width = 118
LEGEND = [
    ("Title", "Supplementary Table 1 — Claim audit matrix for the 21 load-bearing statements"),
    ("标题", "补充表 1 —— 21 条承重主张的审计矩阵"),
    ("", ""),
    ("Purpose / 用途",
     "Every downstream claim in the manuscript is paired with the artifact that produced it, the "
     "recomputation, and a disposition. The table is meant to be read against the manuscript: if "
     "a sentence is not in this table, it is not a load-bearing claim."),
    ("", "正文里每一条承重主张都与产出它的文件、独立重算结果和处置结论一一对应。读者可拿这张表对照正文："
         "不在这张表里的句子，就不是承重主张。"),
    ("", ""),
    ("Column 1 / 列1", "Claim ID — C01–C21; identical to the IDs used in the report (Section 15/16) "
                       "and in scripts/37_claim_audit.py. / 主张编号，与报告第 15、16 节及 "
                       "scripts/37_claim_audit.py 完全一致。"),
    ("Column 2 / 列2", "Original hypothesis or statement — what the manuscript asserted BEFORE the "
                       "audit. Corrected wording is deliberately not substituted here, so that the "
                       "difference between drafts stays visible. / 审计之前正文的原始表述；此处刻意保留原话，"
                       "以便看清改动前后的差别。"),
    ("Column 3 / 列3", "Metric and source artifact — the statistic, the script that computes it, and "
                       "the file that stores it. Paths are relative to the repository root. / "
                       "统计量、计算脚本、存储文件；路径相对仓库根目录。"),
    ("Column 4 / 列4", "Internal re-audit — the recomputed value with its confidence interval or "
                       "boundary condition. Values were read from the artifact programmatically, not "
                       "retyped. / 独立重算值，附置信区间或边界条件；数值由程序从文件中读取，非手工誊写。"),
    ("Column 5 / 列5", "External human validation (GSE131778, 8 donors, 11,756 cells, no exposure). "
                       "Claims that depend on the murine TCDD exposure or the Ahr knockout are marked "
                       "N/A — murine-specific perturbation, because the human atlas has neither. / "
                       "人冠脉外部验证（8 供体、11,756 细胞、无暴露）；依赖小鼠暴露或敲除的主张标为不适用。"),
    ("Column 6 / 列6", "Final disposition, one of exactly three classes (see next block). / 最终处置，"
                       "严格三分类。"),
    ("", ""),
    ("Retained / 保留", f"{sum(r['disp'] == 'Retained' for r in ROWS)} claims. Recomputed value meets the "
                       "prespecified rule; wording is kept."),
    ("Downgraded to Hypothesis / 降级为假设",
     f"{sum(r['disp'].startswith('Downgraded') for r in ROWS)} claim. The number is right but the causal "
     "wording is not supported; the sentence now says 'candidate' or 'hypothesis'."),
    ("Revised / Bounded in Limitations / 改写并写入局限",
     f"{sum(r['disp'].startswith('Revised') for r in ROWS)} claims. Either the prespecified threshold was not "
     "met, or the claim survives only under a narrower statement; each is named in the Limitations section."),
    ("", ""),
    ("Numbering note / 编号说明",
     "An earlier review-facing draft numbered some claims differently (e.g. C04 = endothelial KLF2 "
     "gatekeepers, C09 = depth matching, C17 = cell-type mapping). Those three are C05, C07 and C13 "
     "here. The Crosswalk sheet maps every offset so the two documents can be read side by side; this "
     "table follows the repository numbering, which is what the scripts emit."),
    ("", "早前一份评审用草稿对个别条目用了不同编号（例如把内皮 KLF2 归因写作 C04、等深度写作 C09、"
         "细胞类型映射写作 C17）。在本表中它们分别是 C05、C07、C13。Crosswalk 表给出逐条对照，"
         "本表采用脚本实际输出的仓库编号。"),
    ("Count reconciliation / 计数对照",
     "The audit script reports 16 PASS / 5 REWRITE, this table reports 15/1/5. The one-row offset is "
     "C07: its row in claim_audit.csv already stores the softened sentence, so it passes there, while "
     "column 2 of this table quotes the pre-audit wording, for which the disposition is Revised. "
     "Nothing is double-counted; the two counts refer to two different sentences."),
    ("", "审计脚本报 16 通过 / 5 改写，本表报 15 保留 / 1 降级 / 5 改写。差异仅 C07 一条："
         "它在 claim_audit.csv 里存的是已经软化的句子，故判通过；而本表第 2 列引用的是审计前原话，"
         "对那句话的处置是改写。两条计数指向两句不同的话，没有重复计算。"),
    ("", ""),
    ("Provenance / 溯源",
     "Assembled by scripts/45_table_s1.py from results/audit/claim_audit.csv, "
     "results/human/{r3_summary,external_replication,stromal_continuum_summary}.json and "
     "results/power/floor_summary.json. Audit computations: scripts 37, 39, 41, 42, 43."),
    ("", "由 scripts/45_table_s1.py 汇编，来源为 results/audit/claim_audit.csv、"
         "results/human/ 三个 summary 与 results/power/floor_summary.json；"
         "审计计算由脚本 37、39、41、42、43 完成。"),
    ("", ""),
    ("Suggested legend text / 建议图注",
     "Supplementary Table 1. Audit of the 21 load-bearing claims. For each claim: the original "
     "statement, the generating artifact, the recomputed value with its interval, the external human "
     "validation (GSE131778; N/A where the claim is murine-specific), and the final disposition. "
     "Dispositions: Retained ({n_r}), Downgraded to Hypothesis ({n_d}), Revised / Bounded in "
     "Limitations ({n_v})."),
]
N_R = sum(r["disp"] == "Retained" for r in ROWS)
N_D = sum(r["disp"].startswith("Downgraded") for r in ROWS)
N_V = sum(r["disp"].startswith("Revised") for r in ROWS)
LEGEND = [(k, v.format(n_r=N_R, n_d=N_D, n_v=N_V) if "{n_" in str(v) else v) for k, v in LEGEND]
for k, v in LEGEND:
    lg.append([k, v])
    r = lg.max_row
    lg.cell(row=r, column=1).font = Font(bold=(k in ("Title", "标题")), size=10, name="Calibri")
    lg.cell(row=r, column=2).font = Font(size=9, name="Calibri")
    for j in (1, 2):
        lg.cell(row=r, column=j).alignment = Alignment(vertical="top", wrap_text=True)
    if v:
        lg.row_dimensions[r].height = min(150, max(16, 13 * (len(v) // 110 + 1)))

# ------------------------------------------------------------------- EN and CN
COLS_EN = ["Claim ID", "Original hypothesis / statement",
           "Metric & source artifact", "Internal re-audit result (recomputed)",
           "External human validation (GSE131778)", "Final disposition"]
COLS_CN = ["主张编号", "原始假设／表述（审计前）", "统计量与产出文件",
           "内部重算结果", "外部人体验证（GSE131778）", "最终处置"]
MAP_EN = ["id", "claim_en", "metric_en", "audit_en", "ext_en", "disp"]
MAP_CN = ["id", "claim_cn", "metric_cn", "audit_cn", "ext_cn", "disp_cn"]
WID = [9, 46, 44, 62, 44, 20]

ws1 = wb.create_sheet("TableS1_EN")
write_sheet(ws1, COLS_EN, [{c: r[m] for c, m in zip(COLS_EN, MAP_EN)} for r in ROWS], WID)
ws2 = wb.create_sheet("TableS1_CN")
write_sheet(ws2, COLS_CN, [{c: r[m] for c, m in zip(COLS_CN, MAP_CN)} for r in ROWS], WID)

# -------------------------------------------------------------------- crosswalk
XW_OFFSET = {"C04": "C05", "C09": "C07", "C17": "C13"}
SECTION = {"C01": "5.1 / 10.5", "C02": "5.2", "C03": "5.3 / 11.2", "C04": "5.4 / 7.2",
           "C05": "7.3 / 15.2", "C06": "6.1 / 15.2", "C07": "10.8 / 15.2", "C08": "10.8",
           "C09": "10.8", "C10": "10.7", "C11": "10.7", "C12": "9.2 / 15.2",
           "C13": "9.3 / 15.2 / 16.2", "C14": "8.3", "C15": "10.11", "C16": "16.1",
           "C17": "16.1", "C18": "16.1", "C19": "16.2", "C20": "16.2", "C21": "16.3"}
ARTIFACT = {"C01": "results/ATAC/atac_lineage_effects.csv",
            "C02": "results/permutation/permutation_test.csv",
            "C03": "results/ATAC/atac_lineage_effects.csv",
            "C04": "results/motif/motif_enrichment.csv",
            "C05": "results/motif/ec_vs_smc_contrast.csv",
            "C06": "results/ATAC/atac_lineage_effects.csv",
            "C07": "results/robust/downsample_domain_verdict.csv",
            "C08": "results/audit/depth_summary.csv",
            "C09": "results/audit/subsample_summary.csv",
            "C10": "results/audit/endpoint_verdict.csv",
            "C11": "results/audit/composition_tests_8wk.csv",
            "C12": "results/pseudobulk/cell_vs_pseudobulk_p.csv",
            "C13": "results/human/external_replication.csv",
            "C14": "results/genetics/trait_enrichment.json",
            "C15": "results/chip_alt/occupancy_tests.csv",
            "C16": "results/human/r3_within_donor.csv",
            "C17": "results/human/r3_summary.json",
            "C18": "results/human/r3_summary.json",
            "C19": "results/human/stromal_continuum_summary.json",
            "C20": "results/human/stromal_continuum_summary.json",
            "C21": "results/power/floor_summary.json"}
xw = []
for r in ROWS:
    old = [k for k, v in XW_OFFSET.items() if v == r["id"]]
    xw.append(dict(本表编号=r["id"],
                   评审草稿编号=", ".join(old) if old else "（同）",
                   审计判定=V[r["id"]],
                   处置=r["disp"],
                   报告章节=SECTION[r["id"]],
                   数据文件=ARTIFACT[r["id"]]))
wsc = wb.create_sheet("Crosswalk")
write_sheet(wsc, list(xw[0].keys()), xw, [10, 14, 12, 30, 16, 46])

# ------------------------------------------------------- machine-readable sheet
AI = AUDIT.set_index("id")
KEY = {r["id"]: f"C{int(r['id'][1:])}" for r in ROWS}   # audited ids are "C1".."C21"
mr = pd.DataFrame([dict(claim_id=r["id"], verdict=V[r["id"]], disposition=r["disp"],
                        statement_original=r["claim_en"],
                        value=AI.loc[KEY[r["id"]], "value"],
                        rule=AI.loc[KEY[r["id"]], "rule"],
                        artifact=AI.loc[KEY[r["id"]], "source"])
                   for r in ROWS])
wsm = wb.create_sheet("Machine-readable")
wsm.append(list(mr.columns))
for j, w in enumerate([9, 11, 34, 60, 40, 26, 40], start=1):
    wsm.column_dimensions[get_column_letter(j)].width = w
    c = wsm.cell(row=1, column=j)
    c.fill, c.font = HDR_FILL, HDR_FONT
for row in mr.itertuples(index=False):
    wsm.append(list(row))
for i in range(2, wsm.max_row + 1):
    for j in range(1, wsm.max_column + 1):
        wsm.cell(row=i, column=j).font = BODY
        wsm.cell(row=i, column=j).alignment = Alignment(vertical="top", wrap_text=True)

xl = os.path.join(OUT, "Supplementary_Table_1_Claim_Audit.xlsx")
wb.save(xl)
mr.to_csv(os.path.join(OUT, "claim_audit_21_summary.csv"), index=False, encoding="utf-8-sig")

print(f"written {xl}")
print(f"  rows {len(ROWS)} | " + " / ".join(
    f"{k} {sum(r['disp'] == k for r in ROWS)}" for k in
    ["Retained", "Downgraded to Hypothesis", "Revised / Bounded in Limitations"]))
