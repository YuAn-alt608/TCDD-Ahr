"""46 - Freeze a submission-ready release tree under dist/TCDD-Ahr.

What this does, in order:

  1. copies the code, the audited result tables, the figures and the report into a
     clean tree (raw FASTQ / bigWig / genome FASTA and the per-cell matrices are
     NOT shipped; they are regenerable or re-downloadable, and are listed with
     their accessions in data/DATA_AVAILABILITY.md);
  2. writes the environment capture, the module index, the figure-to-source-data
     manifest, LICENSE / CITATION and a repository README;
  3. hashes every shipped file into MANIFEST.sha256 and records the freeze in
     VERSION.json.

Design note on directory layout: scripts stay FLAT in scripts/. The reviewer-facing
plan was to split them into scripts/01_.../05_... subfolders, but every script
resolves its project root as dirname(dirname(__file__)) and 47 of them also insert
their own directory into sys.path. Moving them one level deeper would break both,
which is a worse outcome than a flat directory with an index. scripts/INDEX.md and
the per-module runner scripts give the same navigability without touching a line
of analysis code.

Usage:  python scripts/46_package_release.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist")
REPO = os.path.join(DIST, "TCDD-Ahr")
VERSION = "v1.0.0-submission"

# result families that ship: tables that back a claim or a figure
SHIP_RESULTS = ["ATAC", "audit", "power", "human", "genetics", "pseudobulk", "permutation",
                "robust", "motif", "validation", "matrix", "chip_alt", "ccc", "te", "chip"]
MAX_FILE = 8 * 1024 * 1024          # per-file cap; the 31 MB per-cell chromVAR table is excluded
SKIP_SUFFIX = (".bw", ".bam", ".h5ad", ".fastq", ".fq", ".rds")
SKIP_NAMES = {"motif_deviation_per_cell.csv.gz"}

MODULES = [
    ("01_data_acquisition", "00-04", "GEO retrieval, annotation build, per-library processing"),
    ("02_smc_heterochromatin", "05-11", "lineage ATAC metrics, heterochromatin domains, motif dissection"),
    ("03_cascade_and_te", "12-17", "cell-cell communication, transposable elements, in-silico validation, trajectory"),
    ("04_pseudobulk_and_null", "18-21", "depth downsampling, per-animal pseudobulk, domain permutation"),
    ("05_external_ahr_mapping", "22-28", "external AHR ChIP, liftOver, human coronary projection, primer design"),
    ("06_confounding_audit", "29-34", "composition decomposition, depth matching, cell subsampling, Tier C gates"),
    ("07_occupancy_and_replication", "35-45", "occupancy specificity, claim audit, donor-stratified replication, Table S1"),
]


def sh(*cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy(src, dst, manifest, note=""):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    rel = os.path.relpath(dst, REPO).replace(os.sep, "/")
    manifest.append((rel, sha256(dst), os.path.getsize(dst), note))


def main():
    if os.path.isdir(DIST):
        shutil.rmtree(DIST)
    os.makedirs(REPO)
    manifest = []

    # ---------------------------------------------------------------- 1. scripts
    sdir = os.path.join(ROOT, "scripts")
    for f in sorted(os.listdir(sdir)):
        p = os.path.join(sdir, f)
        if os.path.isfile(p) and f.endswith((".py", ".md", ".sh")) and not f.startswith("_"):
            copy(p, os.path.join(REPO, "scripts", f), manifest)
    libs = [f for f in os.listdir(sdir) if f.startswith("lib_") and f.endswith(".py")]
    print(f"scripts: {len(os.listdir(sdir))} files, {len(libs)} shared libs")

    # ------------------------------------------------------------- 2. result tables
    skipped = []
    for fam in SHIP_RESULTS:
        base = os.path.join(ROOT, "results", fam)
        for dirpath, _, files in os.walk(base):
            for f in files:
                p = os.path.join(dirpath, f)
                rel = os.path.relpath(p, os.path.join(ROOT, "results")).replace(os.sep, "/")
                if f in SKIP_NAMES or f.endswith(SKIP_SUFFIX) or os.path.getsize(p) > MAX_FILE:
                    skipped.append((rel, os.path.getsize(p)))
                    continue
                copy(p, os.path.join(REPO, "processed_data", "results", rel), manifest,
                     "result table")

    # ---------------------------------------------------------------- 3. figures
    fdir = os.path.join(ROOT, "results", "figures")
    for f in sorted(os.listdir(fdir)):
        if f.endswith(".png"):
            copy(os.path.join(fdir, f), os.path.join(REPO, "figures", f), manifest, "main figure")

    # ------------------------------------------------------- 4. report + table S1
    rp = os.path.join(ROOT, "report", "Age3_report.html")
    if os.path.exists(rp):
        copy(rp, os.path.join(REPO, "report", "Age3_report.html"), manifest, "full report")
    for f in ["Supplementary_Table_1_Claim_Audit.xlsx", "claim_audit_21_summary.csv"]:
        p = os.path.join(ROOT, "processed_data", f)
        if os.path.exists(p):
            copy(p, os.path.join(REPO, "processed_data", f), manifest, "Table S1")

    # ------------------------------------------------------------- 5. input metadata
    md = os.path.join(ROOT, "data", "meta")
    if os.path.isdir(md):
        for dirpath, _, files in os.walk(md):
            for f in files:
                p = os.path.join(dirpath, f)
                if os.path.getsize(p) <= MAX_FILE:
                    rel = os.path.relpath(p, os.path.join(ROOT, "data")).replace(os.sep, "/").replace(os.sep, "/")
                    copy(p, os.path.join(REPO, "data", rel), manifest, "sample metadata")

    # ------------------------------------------- 6. figure -> source data manifest
    shipped = {m[0] for m in manifest}      # only list files that are actually in the freeze
    shipped = {m[0] for m in manifest}      # only list files that are actually in the freeze
    figrows = []
    fmap = {"fig1": "09_figures.py", "fig2": "17_figures_story.py", "fig3": "17_figures_story.py",
            "fig4": "14_insilico_validation.py", "fig5": "17_figures_story.py",
            "fig6": "17_figures_story.py", "fig7": "11_motif.py", "fig8": "17_figures_story.py",
            "fig9": "17_figures_story.py", "fig10": "14_insilico_validation.py",
            "fig11": "16_te_kmers.py", "fig12": "24_figures_robust.py",
            "fig12a": "24_figures_robust.py", "fig12b": "24_figures_robust.py",
            "fig12c": "24_figures_robust.py",
            "fig13": "22_ahr_chip.py", "fig14": "25_power_n2.py", "fig15": "26_ahr_gene_level.py",
            "fig16": "34_figures_confound.py", "fig17": "34_figures_confound.py",
            "fig18": "36_figures_occupancy.py", "fig19": "44_figures_strengthen.py",
            "fig20": "44_figures_strengthen.py", "fig21": "44_figures_strengthen.py"}
    # two ways the figure scripts name their inputs: a plain "dir/file" literal,
    # and os.path.join(<RES>, "dir", "file"). Both are collected, then filtered to
    # paths that actually exist, so the manifest cannot list a phantom file.
    lit_pat = re.compile(r"""["']([A-Za-z0-9_.\-]+/[^"']+\.(?:csv|json|npy|npz|gz))["']""")
    join_pat = re.compile(
        r'os\.path\.join\(\s*(?:L\.)?RES\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"')
    for png in sorted(os.listdir(fdir)):
        m = re.match(r"(fig\d+[a-z]?)", png)
        if not m or not png.endswith(".png"):
            continue
        stem = m.group(1)
        script = fmap.get(stem, "")
        srcs = []
        sp = os.path.join(sdir, script) if script else ""
        if sp and os.path.exists(sp):
            txt = open(sp, encoding="utf-8", errors="ignore").read()
            cands = set(lit_pat.findall(txt))
            cands |= {a + "/" + b for a, b in join_pat.findall(txt)}
            for cand in sorted(cands):
                rel = "processed_data/results/" + cand.replace("\\", "/")
                if rel in shipped:
                    srcs.append(rel)
        figrows.append(dict(figure=f"figures/{png}", generating_script=f"scripts/{script}",
                            source_tables_read_by_script="; ".join(srcs) or "(computed in-script)"))
    import csv as _csv
    os.makedirs(os.path.join(REPO, "processed_data", "figure_source_data"), exist_ok=True)
    with open(os.path.join(REPO, "processed_data", "figure_source_data", "fig_manifest.csv"),
              "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["figure", "generating_script",
                                           "source_tables_read_by_script"])
        w.writeheader()
        w.writerows(figrows)
    with open(os.path.join(REPO, "processed_data", "figure_source_data", "README.md"),
              "w", encoding="utf-8") as f:
        f.write("""# Figure source data

`fig_manifest.csv` maps each figure to the script that draws it and to the result
tables that script reads. The granularity is the *script*, not the panel: several
panels in one figure are drawn from several tables, and a script that draws three
figures (e.g. `24_figures_robust.py` for fig12a-c) necessarily lists the same
tables three times. Rows marked `(computed in-script)` are figures whose inputs are
objects built earlier in the same script rather than a file on disk.

Every path in the manifest is also present in the freeze, so the list can be walked
programmatically without hitting a missing file.
""")
    with open(os.path.join(REPO, "processed_data", "figure_source_data", "README.md"),
              "w", encoding="utf-8") as f:
        f.write("""# Figure source data

`fig_manifest.csv` maps each figure to the script that draws it and to the result
tables that script reads. The granularity is the *script*, not the panel: several
panels in one figure are drawn from several tables, and a script that draws three
figures (e.g. `24_figures_robust.py` for fig12a-c) necessarily lists the same
tables three times. Rows marked `(computed in-script)` are figures whose inputs are
objects built earlier in the same script rather than a file on disk.

Every path in the manifest is also present in the freeze, so the list can be walked
programmatically without hitting a missing file.
""")
    manifest.append(("processed_data/figure_source_data/fig_manifest.csv",
                     sha256(os.path.join(REPO, "processed_data", "figure_source_data",
                                         "fig_manifest.csv")), 0, "figure provenance"))

    # ---------------------------------------------------------- 7. script index
    idx = ["# Script index",
           "",
           "Scripts are kept flat and numbered in execution order. Every file starts with a",
           "docstring saying *why* it exists, not only what it does; the first line of that",
           "docstring is reproduced here.",
           "",
           "| Module | Range | Purpose | Scripts |",
           "|---|---|---|---|"]
    for name, rng, purpose in MODULES:
        lo, hi = [int(x) for x in rng.split("-")]
        idx.append(f"| `{name}` | {rng} | {purpose} | see below |")
    idx.append("")
    for name, rng, purpose in MODULES:
        lo, hi = [int(x) for x in rng.split("-")]
        idx += ["", f"## {name} — {purpose}", ""]
        for f in sorted(os.listdir(sdir)):
            m = re.match(r"(\d+)", f)
            if not f.endswith(".py") or not m:
                continue
            if not (lo <= int(m.group(1)) <= hi):
                continue
            txt = open(os.path.join(sdir, f), encoding="utf-8", errors="ignore").read()
            doc = ""
            mm = re.search(r'^"""(.+?)$', txt, re.S | re.M)
            if mm:
                doc = mm.group(1).strip().split("\n")[0]
                doc = re.sub(r"^\d+[a-z]?\s*-\s*", "", doc).rstrip('"').strip()
            idx.append(f"- `{f}` — {doc}")
    idx += ["", "## Execution order", "",
            "```bash",
            "python scripts/09_figures.py            # main figures from processed results",
            "python scripts/37_claim_audit.py        # the 21-claim audit (Table S1 source)",
            "python scripts/39_external_replication.py",
            "python scripts/41_human_branch_stats.py",
            "python scripts/42_stromal_continuum.py",
            "python scripts/43_power_floor.py",
            "python scripts/45_table_s1.py           # Supplementary Table 1 (.xlsx)",
            "python scripts/10_report.py             # rebuild report/Age3_report.html",
            "```", ""]
    with open(os.path.join(REPO, "scripts", "INDEX.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(idx))

    # --------------------------------------------------------- 8. environment capture
    envdir = os.path.join(REPO, "environment")
    os.makedirs(envdir, exist_ok=True)
    pi = sh(sys.executable, "-m", "pip", "freeze")
    with open(os.path.join(envdir, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write("# python " + platform.python_version() + " on " + platform.platform() + "\n")
        f.write(pi.stdout)
    with open(os.path.join(envdir, "session_info.txt"), "w", encoding="utf-8") as f:
        f.write("freeze: " + VERSION + "\n")
        f.write("built:  " + datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n")
        f.write("host:   " + platform.platform() + "\n")
        f.write("python: " + sys.version.replace("\n", " ") + "\n")
        f.write("git:    " + sh("git", "--version").stdout.strip() + "\n")
        f.write("\n# No R is required. The original pipeline was pure Python; there is no\n"
                "# renv.lock because no analysis step calls R. If a downstream step is added\n"
                "# in R, snapshot it then rather than shipping an empty lockfile.\n")
        f.write("\n# python -m pip list (top-level)\n")
        f.write(sh(sys.executable, "-m", "pip", "list", "--not-required").stdout)
    with open(os.path.join(envdir, "environment.yml"), "w", encoding="utf-8") as f:
        f.write("""# Conda environment for the freeze %s.
# The single-cell steps need scanpy/anndata; the audit and figure steps need only
# numpy/pandas/scipy/matplotlib/openpyxl, which is why lib_core imports the
# single-cell stack lazily (a plain pandas interpreter can rebuild every table
# and the report without it).
name: tcdd-ahr
channels: [conda-forge, bioconda]
dependencies:
  - python=3.11
  - numpy>=1.26
  - pandas>=2.1
  - scipy>=1.11
  - matplotlib>=3.8
  - openpyxl>=3.1
  - scikit-learn>=1.3
  - statsmodels>=0.14
  - pip
  - pip:
      - scanpy>=1.9
      - anndata>=0.9
      # not on conda-forge: pure-python B+ tree bigWig reader is in scripts/lib_bigwig.py
""" % VERSION)

    # ----------------------------------------------------------------- 9. licences
    with open(os.path.join(REPO, "LICENSE"), "w", encoding="utf-8") as f:
        f.write(f"""MIT License

Copyright (c) 2026 the Age3/TCDD-AhR authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---
The derived result tables, figures and Supplementary Table 1 under
processed_data/, figures/ and report/ are released under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). The upstream sequencing data
remain under the terms of their repositories (see data/DATA_AVAILABILITY.md).
""")
    with open(os.path.join(REPO, "CITATION.cff"), "w", encoding="utf-8") as f:
        f.write(f"""cff-version: 1.2.0
title: "AhR-dependent heterochromatin opening in vascular smooth muscle initiates a bifurcated myeloid-stromal cascade"
message: "If you use this code or the derived tables, please cite the article and this archive."
version: "{VERSION}"
date-released: "{datetime.now().strftime('%Y-%m-%d')}"
license: MIT
type: software
repository-code: "https://github.com/YuAn-alt608/TCDD-Ahr"
keywords:
  - TCDD
  - aryl hydrocarbon receptor
  - vascular smooth muscle
  - single-cell ATAC
  - heterochromatin
  - reproducibility
""")

    # ----------------------------------------------------- 10. data availability note
    with open(os.path.join(REPO, "data", "DATA_AVAILABILITY.md"), "w", encoding="utf-8") as f:
        f.write("""# Data availability

## Shipped here (processed, de-identified)
`processed_data/results/` holds every table that backs a claim or a figure: domain
metrics, motif enrichment, permutation nulls, composition decomposition, depth
matching, external AHR occupancy, GWAS enrichment, human donor scores, the
claim-audit outputs and the power/MDE tables. `processed_data/claim_audit_21_summary.csv`
is the machine-readable form of Supplementary Table 1.

`data/meta/` holds the sample sheet and library metadata.

## Not shipped (raw or third-party, re-downloadable)
| Object | Size | Where it comes from |
|---|---|---|
| scATAC / scRNA count matrices (GSE276870, GSE276872) | ~4.5 GB | GEO, series-level supplementary files |
| External AHR ChIP bigWig + peak sets (GSE276152, GSE205502) | ~1.5 GB | GEO supplementary |
| Human coronary atlas expression matrix (GSE131778) | ~1.2 GB | GEO supplementary |
| mm10 / hg38 FASTA + annotation | ~1.5 GB | UCSC / Ensembl |
| FASTQ from SRA | ~39 MB (subset used) | SRA, accessions in `data/meta/` |

Two files that *are* shipped are large enough to note: `results/human/human_marker_logcpm.csv`
(4.0 MB, per-cell marker expression for the human atlas) and
`results/pseudobulk/rna_pseudobulk_cpm.csv` (5.3 MB, per-animal pseudobulk CPM).
`results/motif/motif_deviation_per_cell.csv.gz` (31 MB, per-cell chromVAR deviations)
is excluded by the 8 MB per-file cap; it is regenerated by `scripts/11_motif.py`.

## Accessions
- Murine scATAC + scRNA, TCDD 8/16 wk, Ahr-KO: **GSE276870**, **GSE276872**
- Human coronary artery atlas (no exposure): **GSE131778**
- Human coronary SMC AHR ChIP (TCDD vs DMSO): **GSE276152** (same study as the murine arms)
- AHR differential peak set, second study: **GSE205502**

## Reference genome and chain files
mm10 and hg38 as distributed by UCSC; `mm10ToHg38.over.chain.gz` from
https://hgdownload.soe.ucsc.edu/goldenPath/mm10/liftOver/ (70.7 MB). The liftOver in
`scripts/lib_chain.py` is a streaming parser written for this project because the
off-the-shelf Python bindings materialise the 31 Mb blocks and exhaust memory.
""")

    # --------------------------------------------------------------- 11. README
    n_tab = sum(1 for m in manifest if m[3] == "result table")
    with open(os.path.join(REPO, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# TCDD-AhR — AhR-dependent heterochromatin opening in vascular smooth muscle

Frozen analysis release **{VERSION}**. This repository contains the code, the derived
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
""")

    # ------------------------------------------------- 11b. Zenodo deposit metadata
    # Zenodo reads .zenodo.json from the tagged commit when the GitHub-Zenodo
    # integration is switched on, and it OVERRIDES the metadata scraped from the
    # repo. The creator list is deliberately a placeholder: a DOI record with a
    # guessed author name is worse than one the author has to fill in.
    with open(os.path.join(REPO, ".zenodo.json"), "w", encoding="utf-8") as f:
        json.dump({
            "title": "TCDD-AhR: AhR-dependent heterochromatin opening in vascular "
                     "smooth muscle initiates a bifurcated myeloid-stromal cascade "
                     "(analysis release)",
            "description": "Code, 172 derived result tables, 23 figures and the "
                           "21-claim audit matrix (Supplementary Table 1) for a study of "
                           "8- and 16-week TCDD exposure in ApoE-null mice with "
                           "smooth-muscle-specific Ahr deletion, profiled by single-cell "
                           "ATAC and RNA sequencing. Raw sequencing data are not included; "
                           "see data/DATA_AVAILABILITY.md for accessions "
                           "(GSE276870, GSE276872, GSE131778, GSE276152, GSE205502).",
            "upload_type": "software",
            "license": "MIT",
            "access_right": "open",
            "keywords": ["TCDD", "aryl hydrocarbon receptor", "vascular smooth muscle",
                         "single-cell ATAC-seq", "heterochromatin", "reproducibility",
                         "claim audit"],
            "related_identifiers": [
                {"relation": "isSupplementTo", "identifier": "TODO: DOI of the article",
                 "resource_type": "publication-article"}],
            "creators": [
                {"name": "TODO: family name, given name",
                 "affiliation": "TODO: affiliation",
                 "orcid": "TODO: ORCID (optional)"}],
            "notes": "EDIT BEFORE ENABLING THE ZENODO INTEGRATION. The creator block and "
                     "the related_identifiers DOI are placeholders on purpose; Zenodo "
                     "publishes them verbatim and the DOI is permanent.",
        }, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------- 11b. Zenodo deposit metadata
    # Zenodo reads .zenodo.json from the tagged commit when the GitHub-Zenodo
    # integration is switched on, and it OVERRIDES the metadata scraped from the
    # repo. The creator list is deliberately a placeholder: a DOI record with a
    # guessed author name is worse than one the author has to fill in.
    with open(os.path.join(REPO, ".zenodo.json"), "w", encoding="utf-8") as f:
        json.dump({
            "title": "TCDD-AhR: AhR-dependent heterochromatin opening in vascular "
                     "smooth muscle initiates a bifurcated myeloid-stromal cascade "
                     "(analysis release)",
            "description": "Code, 172 derived result tables, 23 figures and the "
                           "21-claim audit matrix (Supplementary Table 1) for a study of "
                           "8- and 16-week TCDD exposure in ApoE-null mice with "
                           "smooth-muscle-specific Ahr deletion, profiled by single-cell "
                           "ATAC and RNA sequencing. Raw sequencing data are not included; "
                           "see data/DATA_AVAILABILITY.md for accessions "
                           "(GSE276870, GSE276872, GSE131778, GSE276152, GSE205502).",
            "upload_type": "software",
            "license": "MIT",
            "access_right": "open",
            "keywords": ["TCDD", "aryl hydrocarbon receptor", "vascular smooth muscle",
                         "single-cell ATAC-seq", "heterochromatin", "reproducibility",
                         "claim audit"],
            "related_identifiers": [
                {"relation": "isSupplementTo", "identifier": "TODO: DOI of the article",
                 "resource_type": "publication-article"}],
            "creators": [
                {"name": "TODO: family name, given name",
                 "affiliation": "TODO: affiliation",
                 "orcid": "TODO: ORCID (optional)"}],
            "notes": "EDIT BEFORE ENABLING THE ZENODO INTEGRATION. The creator block and "
                     "the related_identifiers DOI are placeholders on purpose; Zenodo "
                     "publishes them verbatim and the DOI is permanent.",
        }, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------- 12. one-click runners
    with open(os.path.join(REPO, "run_all_figures.sh"), "w", encoding="utf-8", newline="\n") as f:
        f.write("""#!/usr/bin/env bash
# Rebuild every figure, Supplementary Table 1 and the report from processed_data/.
# No raw sequencing data required; needs numpy, pandas, scipy, matplotlib, openpyxl.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

for s in 09_figures.py 17_figures_story.py 24_figures_robust.py 28_figures_plan.py \\
         34_figures_confound.py 36_figures_occupancy.py 44_figures_strengthen.py; do
  echo "== $s" && $PY "scripts/$s"
done

echo "== 45_table_s1.py" && $PY scripts/45_table_s1.py
echo "== 10_report.py"   && $PY scripts/10_report.py
echo "done: figures/, processed_data/Supplementary_Table_1_Claim_Audit.xlsx, report/"
""")
    with open(os.path.join(REPO, "run_all_audits.sh"), "w", encoding="utf-8", newline="\n") as f:
        f.write("""#!/usr/bin/env bash
# Rebuild the audit chain behind Table S1: claim audit, external replication,
# donor-stratified branch statistics, stromal continuum, power floor.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
for s in 37_claim_audit.py 39_external_replication.py 41_human_branch_stats.py \\
         42_stromal_continuum.py 43_power_floor.py 45_table_s1.py; do
  echo "== $s" && $PY "scripts/$s"
done
""")
    os.chmod(os.path.join(REPO, "run_all_figures.sh"), 0o755)
    os.chmod(os.path.join(REPO, "run_all_audits.sh"), 0o755)
    with open(os.path.join(REPO, ".gitignore"), "w", encoding="utf-8") as f:
        f.write("__pycache__/\n*.pyc\n.ipynb_checkpoints/\n.DS_Store\nThumbs.db\n")
    # every file is hashed in MANIFEST.sha256, so line-ending or encoding
    # normalisation on checkout would break the integrity check. Ship bytes as-is.
    with open(os.path.join(REPO, ".gitattributes"), "w", encoding="utf-8") as f:
        f.write("* -text -filter\n")

    # -------------------------------------------------------- 13. manifest + version
    total = sum(m[2] for m in manifest)
    ver = dict(version=VERSION,
               frozen_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               files=len(manifest),
               bytes=total,
               result_tables=n_tab,
               figures=len([m for m in manifest if m[3] == "main figure"]),
               scripts=len([m for m in manifest if m[0].startswith("scripts/")]),
               claims=21,
               dispositions={"Retained": 15, "Downgraded to Hypothesis": 1,
                             "Revised / Bounded in Limitations": 5},
               excluded=[dict(file=r, bytes=b) for r, b in skipped],
               upstream=["GSE276870", "GSE276872", "GSE131778", "GSE276152", "GSE205502"],
               licence=dict(code="MIT", derived="CC BY 4.0"))
    with open(os.path.join(REPO, "VERSION.json"), "w", encoding="utf-8") as f:
        json.dump(ver, f, indent=2, ensure_ascii=False)

    # The manifest must cover EVERY shipped byte, including the files written by
    # hand above (README, LICENCE, CITATION, INDEX, the runners). Hashing only the
    # copy() calls would leave the integrity check silently partial.
    lines = []
    for dirpath, dirnames, files in os.walk(REPO):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for fn in sorted(files):
            if fn == "MANIFEST.sha256":
                continue
            p_ = os.path.join(dirpath, fn)
            rel = os.path.relpath(p_, REPO).replace(os.sep, "/")
            lines.append(f"{sha256(p_)}  {rel}")
    with open(os.path.join(REPO, "MANIFEST.sha256"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sorted(lines)) + "\n")
    ver["files_hashed"] = len(lines)
    with open(os.path.join(REPO, "VERSION.json"), "w", encoding="utf-8") as f:
        json.dump(ver, f, indent=2, ensure_ascii=False)
    # VERSION.json just changed, so redo its own line
    lines = [ln for ln in lines if not ln.endswith("  VERSION.json")]
    lines.append(f"{sha256(os.path.join(REPO, 'VERSION.json'))}  VERSION.json")
    with open(os.path.join(REPO, "MANIFEST.sha256"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sorted(lines)) + "\n")

    print(f"frozen {VERSION}: {len(manifest)} files, {total / 1e6:.1f} MB")
    print(f"  result tables {n_tab} | figures {ver['figures']} | excluded {len(skipped)}")
    for r, b in sorted(skipped, key=lambda x: -x[1])[:5]:
        print(f"  excluded {r} ({b / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
