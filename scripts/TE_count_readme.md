# 直接把转座子计入表达的接口说明

本目录的主分析只使用 GEO 提供的**基因水平**表达矩阵（10x `filtered_feature_bc_matrix`），
其中不含转座子（TE）的独立计数。因此 `TES` 指标是三个基因层组分的合成量（见主报告 3.2），
而不是 bp 级 TE 表达。若要把 TE 计数纳入，需要回到 BAM/FASTQ，按下面两条路线之一执行。

## 路线 A：scTE（单细胞，推荐）

```bash
# 1. 依赖
pip install scTE            # 需要 samtools 在 PATH 中
# 2. 构建 mm10 TE 注释（UCSC rmsk -> GTF）
scTE_build -g mm10 -o scTE_mm10
# 3. 从 BAM 计数（BAM 来自 Cell Ranger ATAC/RNA 的 possorted_genome_bam.bam）
scTE -i possorted_genome_bam.bam -o out_prefix -x mm10 -csv -h5
# 4. 输出 out_prefix.h5 中的 TE 家族计数可直接并入 AnnData：
#    adata.obsm['TE_counts'] = sc.read_10x_h5('out_prefix.h5').X
```

## 路线 B：TEtranscripts + TEcount（批量/伪批量）

```bash
TEcount --sortByPos --format BAM \
        -b sample.bam \
        --GTF mm10_refseq.gtf \
        --TE mm10_rmsk.gtf \
        --project sample_TE
```

`mm10_rmsk.gtf` 可由本项目的 `data/annotation/mm10_rmsk.txt.gz` 转换：

```python
import pandas as pd
cols = ["bin","swScore","milliDiv","milliDel","milliIns","chrom","start","end","genoLeft",
        "strand","repName","repClass","repFamily","repStart","repEnd","repLeft","id"]
d = pd.read_csv("data/annotation/mm10_rmsk.txt.gz", sep="\t", header=None, names=cols)
d = d[d.repClass.isin(["LTR","LINE","SINE","DNA"])]
with open("mm10_rmsk.gtf","w") as f:
    for r in d.itertuples():
        f.write(f"{r.chrom}\tRepeatMasker\texon\t{r.start+1}\t{r.end}\t.\t{r.strand}\t.\t"
                f'gene_id "{r.repName}_{r.id}"; transcript_id "{r.repName}_{r.id}"; '
                f'family_id "{r.repFamily}"; class_id "{r.repClass}";\n')
```

## 与主流程的对接

得到 TE 家族计数矩阵（细胞 × 家族）后，把主报告中的 `TES` 组分 (a) 替换为真实 TE 计数：

```python
# 家族水平去抑制分数：每个细胞对各家族计数做 CP10k + log1p，
# 再用对照细胞拟合的均值/方差做 z 分数，按 LTR/ERVK/ERVL/ERV1 子集平均
te = te_counts[:, ["LTR","ERVK","ERVL","ERV1"]]
score_te = zscore(te) .mean(axis=1)
```

## 数据可达性说明

本批次选定的三个数据集均只提交了基因水平矩阵，未附 BAM。若需 BAM，可从 SRA 重新比对
（`prefetch` + `fasterq-dump` + `cellranger count`），每个样本约需 2–6 小时与 30–50 GB 中间文件，
超出本次分析的计算预算，故未执行。

## 本次实际做了什么（results/te/）

因为拿不到 BAM，`scripts/13_te.py` 只做了两件**可审计**的事，并在报告 4.8 节写清了边界：

1. **家族级启动子代理**：用 `data/annotation/mm10_rmsk.txt.gz` 把 TSS±1 kb 内含某一家族元件的基因挑出来
   （IAP 54、ETn 9、ERVK-其它 2,073、ERVL 3,491、ERV1 978、L1 3,938、SINE 12,512 个基因），
   以这些基因的 log 表达均值作为该家族的代理活性。这是**基因层**指标，不是 TE 转录本计数。
2. **符号检查（重要）**：七个家族在 TCDD 16 周**同比例**降到约 0.62 倍、Ahr-KO 组又同比例回升。
   家族之间没有选择性 → 这是文库/检出结构差异，不是家族特异去抑制。因此报告**不呈现**跨组的 TE 代理差异，
   只呈现同一文库内的细胞级共变（TE 代理 vs ISG，巨噬 ERV1/L1/SINE ρ ≈ 0.4，对照 0.1–0.2）。

这个检查应该成为常规步骤：任何“某家族被特异性激活”的结论，先看其它家族是否同向同幅移动。

## 若拿到 BAM 的最小执行清单

| 步骤 | 命令/参数 | 预计资源 |
|---|---|---|
| 取原始 reads | `prefetch --max-size 100G <SRR>` + `fasterq-dump --split-files` | 每样本 20–60 GB |
| 比对 | `STAR --soloType CB_UMI_Simple --soloMultiMappers EM`（保留 EM 多重比对，scTE 需要） | 每样本 3–6 h，60 GB 中间文件 |
| TE 计数 | `scTE -i possorted_genome_bam.bam -x mm10 -csv -h5` | 每样本 1–2 h |
| 合并 | 按 barcode 前缀合并 16 个样本的家族计数，得到 cells × family | — |
| 统计 | 家族内 CP10k+log1p → 以对照细胞拟合的均值/方差做 z → 与 ISG 模块做细胞级相关 | 分钟级 |

按 GSE276872 的 16 个样本估算，全流程约 1.2 TB 中间文件与 2–3 天机时；这是本报告没有做这件事的**唯一**原因。

---

## 2026-11-17 更新：原始 reads 其实是公开的，已建立免比对器管线

上一版把“GEO 没给 BAM”误读成“没有 reads”。核查结果：

| 项目 | 登录号 | 规模 |
|---|---|---|
| scRNA（GSE276872） | `PRJNA1159315` / `SRP531896` | 41 个 run，约 130 GB FASTQ |
| scATAC（GSE276870） | `PRJNA1159314` / `SRP531933` | 8 个 run，约 72 GB FASTQ |
| bulk ATAC | `PRJNA1155180` / `SRP529913` | — |

缺的仍然只是 fragments/BAM（所以足迹分析做不了），不是 reads。

### 已实现的替代方案：`16_te_kmers.py`

不需要 STAR、不需要 scTE：从 rmsk 取每家族 400 个元件的中心 1.5 kb，
用 20-mer 规范化 + 跨家族去重，做成 798,195 个家族特异 k-mer；
reads 批量编码后一次 `searchsorted` 完成伪比对；判定 = 最佳家族 ≥2 个唯一 k-mer 且 ≥2× 次优。

- 引擎召回（留出元件模拟 reads）：IAP 26%、ETn 86%、ERVK 44%、ERVL 29%、ERV1 33%、L1 62%；
  SINE_B 0/900（不可用，本文不对 SINE 下结论）。
- 管家基因编码外显子阴性对照：318 条 reads 中 1 条被错误分配（0.31%）。
- 速度：约 27 万条 reads / 10 s（单核）。

### 全量运行怎么做

```bash
# 1) 分块下载（直连 300 KB/s 且会断流，必须分块 + 校验）
bash scripts/get_fastq_chunked.sh SRR30622183 083 data/fastq/ctrl_r1.part.gz 3400000000
bash scripts/get_fastq_chunked.sh SRR30622170 070 data/fastq/tcdd_r1.part.gz 2200000000
# 2) 计数（去掉 --pilot 即全量；脚本自动扫描 data/fastq/*.part.gz）
python scripts/16_te_kmers.py
```

判定标准：四个高丰度家族（ETn/ERVK/L1/SINE）的每文库命中数 ≥ 5,000 时才够做跨组比较；
细胞类型分层需要 barcode 白名单（从 `results/D2/clustered.h5ad` 导出 tdT⁺/tdT⁻ 的 barcode），
`16_te_kmers.py` 目前是文库级，分层版本需要在 `score_batch` 后按 barcode 前缀聚合。

### 当前 pilot 的结论（写进报告的口径）

等字节前缀 20 MB × 2 文库（27 万 reads 级），四个可解释家族全部 ×0.96–1.07，中位 ×1.04：
**在这个深度上检测不到家族级 TE 去抑制**。这既不是阳性结果，也不能反证阴性——
它证明的是管线可用、且没有文库层面的系统性偏移。

### 投稿周期的决定（2026-11-17）

**不再跑全量 TE。** 理由不是省事，而是三条互相独立的判断：

1. **它不改任何主结论。** 报告的因果链停在「motif 层 + 域可及性」，
   TE 只是第 7 节的一个方法学附注；把 pilot 升级成家族×组检验，最好的结果也是
   “四个家族里 1–2 个 ×1.2–1.5”，仍需原位验证才能写进机制。
2. **带宽是硬约束。** 4 个文库约 11 GB，直连 300 KB/s 且频繁断流，分块下载 + 校验
   在本机至少需要两个整晚，期间无法访问同一网络做别的下载。为一条补充材料占用两周投稿窗口不划算。
3. **现有的 pilot 已经能挡住最要命的那一问。** 审稿人会问“你的 TE 结论有没有归一化偏移”，
   等字节前缀 + 四家族 ×0.96–1.07 正好回答这个问题。**pilot 作为 Supplementary 的
   “引擎验证 + 归一化阴性对照”，而不是生物学结论**，这是它唯一诚实的用法。

若审稿人明确要求 TE 家族定量，再按上面的分块下载脚本补跑；届时须同时补 barcode 白名单，
否则仍只能给文库级结果。
