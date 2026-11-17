# TCDD-AhR — AhR-dependent heterochromatin opening in vascular smooth muscle

Frozen analysis release **v1.0.0-submission**. This repository contains the code, the derived
result tables, the 21 figures and Supplementary Table 1 for a single study:
8- and 16-week TCDD exposure in ApoE-null mice, with and without smooth-muscle-specific
*Ahr* deletion, profiled by single-cell ATAC and RNA sequencing.

The repository exists so that every load-bearing sentence in the manuscript can be
traced to a number, and every number to a file, and every file to a script.

## The claim in one paragraph

TCDD opens heterochromatin-marked domains in vascular smooth muscle in an
AhR-dependent but not AhR-exclusive way (47% of the gain is reversed, 53% retained). The
endothelial change is **not** reversed by deleting *Ahr* in smooth muscle — the
endothelial arm is non-cell-autonomous. Downstream, the myeloid LAM state and the
fibroblast matrix-ligand program are the only endpoints that survive aggregation to
the animal level. The two arms are separable in an independent human coronary atlas:
endothelial and matrix programs are orthogonal across all eight donors, while the
myeloid–matrix coupling is positive. Direct AhR occupancy of the opened bins is
**not** supported by any available dataset and is stated as an open question.

## What is *not* claimed

- No ChIP-seq, ChIP-qPCR or reporter assay was performed here. The manuscript says
  "AhR-dependent", not "AhR-bound". A calibrated negative on external AHR occupancy
  tracks is reported (Claim C15) rather than left out.
- The pre-registered animal-level endpoints for endothelium and smooth muscle are
  directionally consistent but not significant at n = 3/group. With 20 possible splits
  the exact permutation floor is 0.143, so those tests **cannot** reject at this sample
  size; the manuscript reports Hedges' g with intervals and the MDE instead of a p value
  alone (Claim C21).
- Two of four mouse programs do not map onto their canonical human cell types. The
  matrix program tracks a *state* (matrix production), not a lineage, and is described
  that way (Claim C13, C19, C20).

## Layout

```
scripts/            flat, numbered 00-46; scripts/INDEX.md groups them into 7 modules
processed_data/     every table behind a claim or a figure; Table S1 in .xlsx
  results/          mirrors the analysis tree (audit/, human/, motif/, chip_alt/, ...)
  figure_source_data/fig_manifest.csv   figure -> generating script -> source tables
figures/            fig1-fig21 (PNG, 300 dpi)
report/             the full self-contained HTML report (16 sections, all figures inline)
data/meta/          sample sheet and library metadata
data/DATA_AVAILABILITY.md   what is shipped, what is re-downloadable, and from where
environment/        environment.yml, requirements.txt, session_info.txt
MANIFEST.sha256     sha256 of every shipped file
VERSION.json        freeze metadata
```

## Reproduce the tables and figures

```bash
conda env create -f environment/environment.yml && conda activate tcdd-ahr
bash run_all_figures.sh      # figures + Table S1 + report from processed_data/
bash run_all_audits.sh       # the audit chain behind Table S1 (claims, replication, power)
```

The audit and figure steps need only numpy / pandas / scipy / matplotlib / openpyxl.
The single-cell stack (scanpy, anndata) is imported lazily inside `lib_core.py`, so the
whole reporting layer runs in a plain pandas interpreter — which is what makes the
freeze checkable without the 8 GB of raw data.

Rebuilding from raw counts additionally requires the GEO objects listed in
`data/DATA_AVAILABILITY.md`; the scripts that consume them are 00-08 and will stop
with a clear message if the files are absent.

## Supplementary Table 1

`processed_data/Supplementary_Table_1_Claim_Audit.xlsx` — 21 rows, 6 columns, bilingual
(English sheet + Chinese sheet + crosswalk + machine-readable sheet). Columns: claim ID,
original hypothesis, metric and source artifact, internal re-audit, external human
validation, final disposition. Dispositions are exactly three: **Retained (15)**,
**Downgraded to Hypothesis (1)**, **Revised / Bounded in Limitations (5)**.

## Integrity check

```bash
sha256sum -c MANIFEST.sha256        # or: shasum -a 256 -c MANIFEST.sha256
```

## 中文要点

本仓库是一个环境毒理＋单细胞表观遗传研究的**冻结分析版本**：小鼠 TCDD 暴露（8/16 周）
叠加平滑肌特异 *Ahr* 敲除，单细胞 ATAC＋RNA。核心结论是“起点在平滑肌、传播会分叉”：
平滑肌异染色质开放依赖 AhR 但不被 AhR 独占（敲除后回落 47%、仍保留 53% 的增益），内皮改变不被平滑肌
AhR 敲除逆转（非细胞自主），个体水平只有髓系 LAM 与成纤维基质配体站得住。

**明确不主张**：没有做任何 ChIP 或报告基因实验，所以只说“AhR 依赖”，不说“AhR 结合”；
个体水平内皮与平滑肌终点在 n=3/组 下方向一致但不显著，且 3v3 精确置换的最小可达双侧
p 本身就有 0.143 的天花板；四个小鼠程序里只有两个能落到人体对应细胞类型。
补充表 1 把这 21 条主张逐条列出重算值与处置结论（保留 15、降级 1、改写并进局限 5）。

## Licence

Code MIT; derived tables, figures and Table S1 under CC BY 4.0. Upstream data
remain under their repository terms.
