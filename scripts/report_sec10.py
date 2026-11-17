"""Section 10 of the report: the pre-submission verification block.

Kept as a separate module so the numbers can be regenerated and checked without
touching the narrative code in 10_report.py.
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L


def _load(p):
    q = os.path.join(L.RES, p)
    return pd.read_csv(q) if os.path.exists(q) else pd.DataFrame()


def _json(p):
    q = os.path.join(L.RES, p)
    return json.load(open(q)) if os.path.exists(q) else {}


def _audit_rows():
    """Rows for the revision table that come out of the 10.7-10.9 audit."""
    a = _json("audit/summary.json")
    ds = _json("audit/depth_summary.json")
    sb = _json("audit/subsample_summary.json")
    bw = _json("chip_alt/bigwig_summary.json")
    su = _json("chip_alt/summit_summary.json")
    qp = _load("chip/qpcr_primers.csv")
    rows = []
    v = a.get("verdict", {})
    if v:
        ci = v.get("cell-intrinsic (survives composition matching)", 0)
        mx = v.get("mixed", 0)
        cs = v.get("composition-sensitive", 0)
        rows.append(("效应由细胞亚群比例变化造成", "组成分解 + 精确置换",
                     f"{ci} 个终点组成匹配后成立、{mx} 个混合、{cs} 个组成敏感；"
                     f"L1 精确置换最小 p = 0.143",
                     "撤回“组成变化”，两个组成敏感终点改口径"))
    if ds:
        rows.append(("Ahr-KO 库浅 27% 造成差异", "二项稀释到等深度重算",
                     f"4 个冻结终点 × 3 次稀释方向 {ds.get('direction_preserved_at_ako_depth','')} 全保；"
                     "幅度衰减最多 34.5%（内皮 prom_het_rel），"
                     "而 log1p 上的内皮 het_derepression 反被放大 153%",
                     "保留结论，但幅度只给量级区间，并并列线性 share"))
    if sb:
        rows.append(("少数异常细胞驱动主终点", "每只动物内 1,000 次 80% 重抽样 + jackknife",
                     f"同号比例 {sb.get('min_frac_same_sign',0)*100:.0f}%；"
                     f"最差 {sb.get('min_frac_within_50pct',0)*100:.1f}% 落在 ±50% 内；"
                     "单细胞 jackknife 无法翻号",
                     "主终点稳健，可作为正面证据"))
    if bw:
        res = su.get("result", {})
        npass = sum(1 for v in res.values() if v.get("n_pass", 0) >= 2)
        best = max(res.items(), key=lambda kv: kv[1].get("n_pass", 0)) if res else ("", {})
        rows.append(("外部 AHR ChIP 可提供结合证据", "Tier C 六靶基因门禁",
                     f"{len(res)} 套公开数据里 {npass} 套过 summit 门槛（最好 "
                     f"{best[1].get('n_pass',0)}/6：{', '.join(best[1].get('genes',[]))}），"
                     "无一套 6/6；GSE205502 差异峰集 0/6",
                     "不引用任何 ChIP 交集；改写为“只有部分阳性对照可复现”"))
    if len(qp) and "verdict" in qp.columns:
        vc = qp.verdict.value_counts()
        ok = int(vc.get("ok", 0))
        bad = qp[qp.verdict != "ok"]
        nA = int((qp.tier == "A").sum())
        nC = int((qp.tier == "C").sum())
        rows.append(("42 对 qPCR 引物特异性未检查", "本地电子 PCR（mm10 + hg38 全基因组）",
                     f"单产物 {ok}/{len(qp)}；Tier A 的 {nA} 对与 Tier C 主引物的 {nC} 对"
                     f"全部单产物，{len(bad)} 对多产物"
                     f"（{', '.join(bad.mouse.fillna(bad.human).tolist())}）",
                     "特异性列已填实；多产物 2 对位于重复区、局部换不掉，从下单表移除"))
    return rows


def build():
    H = []
    dv = _load("robust/downsample_domain_verdict.csv")
    eff = _load("robust/downsample_effects.csv")
    pv = _load("pseudobulk/cell_vs_pseudobulk_p.csv")
    pm = _load("pseudobulk/rna_module_tests.csv")
    pg = _load("pseudobulk/rna_gene_tests.csv")
    pt = _load("permutation/permutation_test.csv")
    th = _load("permutation/threshold_sensitivity.csv")
    lrt = _load("validation/lr_permutation_test.csv")
    lrs = _json("validation/lr_permutation_summary.json")
    cht = _load("chip/chip_overlap_tests.csv")
    lft = _load("chip/liftover_tests.csv")
    chs = _json("chip/summary.json")
    lhs = _json("chip/liftover_summary.json")
    glt = _load("chip/gene_level_tests.csv")
    glc = _load("chip/chip_positive_control.csv")
    gls = _json("chip/gene_level_summary.json")
    qp = _json("chip/qpcr_design.json")
    hum = _load("human/human_programme_summary.csv")
    hus = _json("human/summary.json")

    def dget(df, **kw):
        if df.empty:
            return np.nan
        m = pd.Series(True, index=df.index)
        for k, v in kw.items():
            m &= df[k] == v
        s = df[m]
        return s.iloc[0] if len(s) else None

    def dom(metric, lin, col):
        r = dget(dv, metric=metric, lineage=lin)
        return (np.nan if r is None else float(r[col]))

    def tfv(tf, lin, target, col):
        if eff.empty:
            return np.nan
        s = eff[(eff.tf == tf) & (eff.lineage == lin) & (eff.target == target)][col]
        return float(s.mean()) if len(s) else np.nan

    def perm(lin, col):
        r = dget(pt, lineage=lin)
        return np.nan if r is None else float(r[col])

    def thr(defn, lin):
        r = dget(th, definition=defn, lineage=lin)
        return np.nan if r is None else float(r.d_wt_vs_ctrl)

    H.append("<h2>10. 投稿前核验：四项内部压力测试与两项外部验证</h2>")
    H.append("""<div class="note"><strong>为什么单开一节。</strong>
这一节回答的不是“结果好不好看”，而是“哪几句话在审稿人第一轮追问下会塌”。
<strong>五项内部核验</strong>里，两项通过、一项部分通过、一项直接否定了原来的措辞、
一项（组成与深度混淆）把 10 个冻结终点拆成了“细胞自主 6 / 混合 2 / 组成敏感 2”；
<strong>两项外部验证</strong>里，一项阴性、一项弱阳性；
<strong>10.9 又补了三件外部与工程上的事</strong>：公开 AHR 占据图谱的 Tier C 重判、
峰集与信号轨的矛盾、42 对引物的全基因组特异性。下面每一条都给出可复算的表和结论修订，
被推翻的地方在正文其余章节已同步改写。</div>""")

    # ---------------------------------------------------------------- 10.1
    H.append("<h3>10.1 文库深度：Ahr-KO 的 scATAC 比 WT 浅 27%</h3>")
    H.append(f"""<p>四个 scATAC 文库的中位计数/细胞是 4,682 与 5,378（两只对照）、5,114（16 周 WT）、
<strong>3,718（Ahr-KO）</strong>，Ahr-KO 浅 27.3%。审稿人一定会问：Ahr-KO 臂上那些“回落”是不是浅测序造成的？
做法是把每个细胞按 p = T*/n<sub>i</sub> 做二项稀疏化（T* = 3,718，另加 60% 的压力档），
再用同一套代码重算 chromVAR 偏差与三个域指标，5 个随机种子给出离散度。</p>""")
    if not dv.empty:
        cols = ["metric", "lineage", "full_d", "full_p", "match_ako_d", "match_ako_p",
                "stress_60pct_d", "stress_60pct_p", "verdict"]
        t = dv[cols].copy()
        t.columns = ["指标", "谱系", "满深度 d", "满深度 p", "等深度 d", "等深度 p",
                     "60% 压力 d", "60% 压力 p", "判定"]
        H.append(t.to_html(index=False, classes="tbl", border=0, escape=False,
                           float_format=lambda v: f"{v:.3f}"))
    H.append(f"""<p>满深度下显著的 8 个域指标，在等深度后<strong>方向与显著性全部保留</strong>，
效应量衰减 7–25%：SMC 的 H3K9me3 域 {dom('frac_het','Parenchyma','full_d'):+.3f} →
{dom('frac_het','Parenchyma','match_ako_d'):+.3f}（p 仍为 {dom('frac_het','Parenchyma','match_ako_p'):.3f}）、
内皮异染色质启动子 {dom('prom_het_rel','Endothelial','full_d'):+.3f} →
{dom('prom_het_rel','Endothelial','match_ako_d'):+.3f}（p {dom('prom_het_rel','Endothelial','full_p'):.3f} →
{dom('prom_het_rel','Endothelial','match_ako_p'):.3f}），但<em>在 60% 压力档掉到 p =
{dom('prom_het_rel','Endothelial','stress_60pct_p'):.2f}</em>——这条的余量本来就不大。</p>""")
    H.append(f"""<p>motif 层要分两半说。内皮的 <strong>Ahr::Arnt 在等深度下更贴近零</strong>
（{tfv('Ahr::Arnt','Endothelial','none','d_ako_vs_wt'):+.3f} → {tfv('Ahr::Arnt','Endothelial','match_ako','d_ako_vs_wt'):+.3f}），
“敲 SMC-Ahr 不动内皮的 AhR 位点”这一条最稳；KLF2 / SMAD2 / ATF2 的回落<strong>方向全保、显著性下滑</strong>
（KLF2 {tfv('KLF2','Endothelial','none','d_ako_vs_wt'):+.3f}→{tfv('KLF2','Endothelial','match_ako','d_ako_vs_wt'):+.3f}，
p {tfv('KLF2','Endothelial','none','p_ako_vs_wt'):.3f}→{tfv('KLF2','Endothelial','match_ako','p_ako_vs_wt'):.3f}；
SMAD2 p {tfv('SMAD2','Endothelial','none','p_ako_vs_wt'):.3f}→{tfv('SMAD2','Endothelial','match_ako','p_ako_vs_wt'):.3f}；
ATF2 p {tfv('ATF2','Endothelial','none','p_ako_vs_wt'):.3f}→{tfv('ATF2','Endothelial','match_ako','p_ako_vs_wt'):.3f}）。
稀释本身就会损失效能，所以这里正确的读法是<em>效应量稳定、p 值随深度漂移</em>，
正文引用的 p 值必须注明是满深度结果。SMC 自身更要注意：Ahr::Arnt 回落从
{tfv('Ahr::Arnt','Parenchyma','none','d_ako_vs_wt'):+.3f} 缩到 {tfv('Ahr::Arnt','Parenchyma','match_ako','d_ako_vs_wt'):+.3f}
（p {tfv('Ahr::Arnt','Parenchyma','none','p_ako_vs_wt'):.3f}→{tfv('Ahr::Arnt','Parenchyma','match_ako','p_ako_vs_wt'):.3f}），
幅度上有一半来自深度，<strong>不能把 −0.20 当作与深度无关的效应量引用</strong>。</p>""")
    H.append(_fig("fig12a_depth.png",
                  "图 S1a｜把三个臂稀释到 Ahr-KO 的中位深度后重算 motif 偏差。误差棒为 5 个稀释种子；"
                  "内皮（左）的 Ahr::Arnt 始终贴零，KLF2/SMAD2/ATF2 的负向在等深度下衰减但不变号；"
                  "SMC（右）的 Ahr::Arnt 回落一半来自深度。"))

    # ---------------------------------------------------------------- 10.2
    H.append("<h3>10.2 个体水平重算：单细胞 p 值到底虚高多少</h3>")
    H.append("""<p>先把设计写清楚，这一条比任何统计都重要：scRNA 共 8 只小鼠
（对照 3、8 周 3、16 周 WT 1、Ahr-KO 1），scATAC 共 4 个文库（对照 2、WT 1、Ahr-KO 1）。
<strong>16 周与 Ahr-KO 各只有一只动物</strong>，因此 Ahr-KO 对比在个体水平上不存在重复，
任何“个体水平 p 值”都只能由细胞伪重复伪造出来。可以做的个体水平检验只有“对照 vs 8 周”（3 vs 3，Welch）。</p>""")
    if not pv.empty:
        rows = []
        for _, r in pv.iterrows():
            rows.append({"细胞类型": r.celltype, "指标": r.metric,
                         "8 周细胞数": r.n_cells_8wk,
                         "细胞水平 p": r.p_cell_level,
                         "个体水平 p": r.p_animal_level,
                         "个体水平 d": r.d_animal_level,
                         "个体水平 FDR": np.nan})
        # 成纤维的基质配体模块只出现在模块表里，补同一行以便正文的 d=+3.99 可核
        if not pm.empty:
            q = pm[(pm.celltype == "Fibroblast") & (pm.module == "matrix_ligand")]
            if len(q):
                q = q.iloc[0]
                rows.append({"细胞类型": "Fibroblast", "指标": "mod_matrix_ligand",
                             "8 周细胞数": np.nan, "细胞水平 p": np.nan,
                             "个体水平 p": float(q.p_8wk),
                             "个体水平 d": float(q.d_8wk),
                             "个体水平 FDR": float(q.fdr_8wk)})
        t = pd.DataFrame(rows)
        H.append(t.to_html(index=False, classes="tbl", border=0, escape=False,
                           na_rep="—", float_format=lambda v: f"{v:.4g}"))
    H.append("""<p>结论分两半。巨噬细胞的衰老/炎性模块<strong>扛住了</strong>个体水平检验
（AgingSen d = +5.50，p = 0.0038）；而成纤维的基质配体模块也扛住（d = +3.99，p = 0.008，FDR 0.049）。
但<strong>内皮与 SMC 的“异染色质基因泄漏指数”在个体水平都不显著</strong>（p = 0.14 / 0.19，8 周臂），
方向还是<em>负</em>的（d = −1.82 / −1.55，即泄漏指数下降）。细胞水平那个 p &lt; 1e-4 是 3,000 个细胞换来的，
不是 3 只小鼠换来的。正文因此把 RNA 层的“泄漏”改写为“与 ATAC 方向不完全一致”，
不再声称 RNA 层复现了缓冲崩溃。</p>""")
    H.append("""<p>16 周 / Ahr-KO 这一对只能给个体级数值，不能给 p：巨噬 Trem2 6.22 → 6.78、
Lgals3 7.57 → 7.82、Csf1r 6.30 → 6.50，方向与 Ahr-KO 的 LAM 过冲一致；
但同一只 Ahr-KO 小鼠的巨噬 Timp1 从 5.57 掉到 2.57、Fn1 从 6.43 掉到 5.16，
说明“敲除≠复原”在个体数值上也成立。<strong>投稿前要么补第二只 Ahr-KO 小鼠，
要么把第 4 节的措辞从“证明”降为“与……一致”</strong>——本文选择了后者，并在此处留痕。</p>""")

    # ---------------------------------------------------------------- 10.3
    H.append("<h3>10.3 域定义置换与阈值：效应是“域特异”还是“全局平移”</h3>")
    H.append(f"""<p>H3K9me3 bin 天然 GC 低、TSS 稀，直接和全基因组比会把“组成差异”算成“域效应”。
所以每轮置换都按 <strong>1 个百分点 GC × TSS±50 kb 四分位</strong>分层，从 113,292 个非域 bin 里
抽同样本量（2,359 个）的匹配集，重复 1,000 次。匹配质量：目标集 GC
{pt.iloc[0].gc_target if not pt.empty else float('nan'):.3f}，零模型均值
{pt.iloc[0].gc_null if not pt.empty else float('nan'):.3f}。</p>""")
    if not pt.empty:
        t = pt[["lineage", "d_observed", "null_mean", "null_sd", "z", "p_one_sided"]].copy()
        t.columns = ["谱系", "观察 d", "零模型均值", "零模型 SD", "z", "单侧 p"]
        H.append(t.to_html(index=False, classes="tbl", border=0, escape=False,
                           float_format=lambda v: f"{v:.3f}"))
    H.append(f"""<p><strong>这是本节最重要的一次否定。</strong>只有 SMC 的偏移显著超出匹配常染色质
（d = {perm('Parenchyma','d_observed'):+.3f}，零模型 {perm('Parenchyma','null_mean'):+.3f} ±
{perm('Parenchyma','null_sd'):.3f}，z = {perm('Parenchyma','z'):.1f}）；
内皮的 {perm('Endothelial','d_observed'):+.3f} 落在零模型
{perm('Endothelial','null_mean'):+.3f} ± {perm('Endothelial','null_sd'):.3f} 之内（p = {perm('Endothelial','p_one_sided'):.2f}），
间质与免疫同样无法区分。也就是说：<strong>“跨谱系普遍击穿异染色质”不成立</strong>，
内皮的偏移与“GC/TSS 匹配的常染色质一起整体上移”无法区分。
这正是把主线收窄到 SMC 的统计学理由，第 1 节的说法据此改写。</p>""")
    H.append(f"""<p>阈值方向是加分的：把 H3K9me3 从“全部峰”收紧到 Top20%、Top10%，
效应量<strong>单调上升</strong>（SMC {thr('H3K9me3_all','Parenchyma'):+.2f} → {thr('H3K9me3_top20','Parenchyma'):+.2f} →
{thr('H3K9me3_top10','Parenchyma'):+.2f}；内皮 {thr('H3K9me3_all','Endothelial'):+.2f} →
{thr('H3K9me3_top20','Endothelial'):+.2f} → {thr('H3K9me3_top10','Endothelial'):+.2f}），
符合“信号集中在高置信域”的预期。但 LAD 代理给出<em>相反符号</em>
（内皮 {thr('LAD_proxy','Endothelial'):+.2f}、间质 {thr('LAD_proxy','Mesenchymal'):+.2f}、
免疫 {thr('LAD_proxy','Immune'):+.2f}），说明“H3K9me3 域”和“基因荒漠 LAD 代理”
在这套数据里不是同一件事，正文不能把两者合并成“异染色质”一个词。</p>""")
    H.append(_fig("fig12b_permutation.png",
                  "图 S1b｜匹配常染色质的零分布。红线为 SMC 的观察值（p = 0.001），蓝线为内皮（p = 0.28）。"))
    H.append(_fig("fig12c_threshold.png",
                  "图 S1c｜域定义收紧时四条谱系的效应量。SMC 与内皮都单调上升，LAD 代理反向。"))

    # ---------------------------------------------------------------- 10.4
    H.append("<h3>10.4 配体-受体网络的置换检验：4/4 到底证明了什么</h3>")
    H.append(f"""<p>第 6 节的 in silico 条件培养基用 187 对分泌配体-受体拟合单参数 κ，
在留出的 Ahr-KO 臂上 4/4 个内皮读出方向正确。为了检验“这是网络拓扑还是数学巧合”，
对二部网络做两种<strong>度保持重连</strong>——打乱配体标签（受体入度不变）与打乱受体标签
（配体出度不变）各 1,000 次，每次重走同一条 κ 流程。</p>""")
    if not lrt.empty:
        t = lrt[["null", "hits", "mean_abs_err", "p_hits", "p_err", "frac_4of4"]].copy()
        t.columns = ["零模型", "平均命中数", "平均绝对误差", "p(命中)", "p(误差)", "4/4 比例"]
        H.append(t.to_html(index=False, classes="tbl", border=0, escape=False,
                           na_rep="—", float_format=lambda v: f"{v:.3f}"))
    H.append(f"""<p><strong>这一条是阴性，而且原因很干净。</strong>真实网络 4/4，随机网络<em>也</em> 4/4
（两种零模型 4/4 比例 = 1.00），平均绝对误差还更小
（{lrs.get('observed_mean_abs_err', float('nan')):.3f} vs {lrt[lrt.null=='ligand_shuffle'].mean_abs_err.iloc[0] if not lrt.empty else float('nan'):.3f}）。
原因是模型结构：预测式是 y<sub>WT</sub> + κ(x<sub>AKO</sub> − x<sub>WT</sub>)，
κ 逐基因拟合，四个读出的预测方向由 sign(x<sub>AKO</sub> − x<sub>WT</sub>) 统一决定，
与“哪条边连哪条边”无关。换句话说，<strong>4/4 检验的是“四个基因按同一比例回弹”，
不是“网络被识别出来”</strong>。因此原文“4/4 证明捕捉到 Timp1–Cd63 / Fn1–Sdc4 信号流”
这句话<strong>已删除</strong>，改为“单参数 κ 能把留出臂的幅度复现到 0.05 log 单位以内，
但该模型对网络拓扑不可辨识”。第 6 节的湿实验判据（TIMP1 中和 / CH-223191）因此更重要了。</p>""")

    # ---------------------------------------------------------------- 10.5
    H.append("<h3>10.5 外部验证 I：AHR ChIP-seq——从 bin 到基因，再到它自己的阳性对照</h3>")
    H.append(f"""<p>GSE276152 是同一研究系列的 ChIP-seq 子集：人冠脉平滑肌细胞（HCASMC），
DMSO vs TCDD，AHR 与 TCF21 抗体。峰数本身像一个强内对照：<strong>TCDD 处理 16,499 个 AHR 峰，
DMSO 只有 262 个</strong>。先把 hg38 峰用 UCSC 链 lift 到 mm10（47.3% 的 5 kb bin 可 lift），
再与差异开放 bin 取交集。这一步是干脆的阴性：SMC 的 270 个开放 bin 里只有 14 个落进 AHR 峰
（背景 4.1%），OR = 1.29，p = 0.35。</p>""")
    if not lft.empty:
        t = lft[["lineage", "n_open", "n_open_ahr", "frac_open_ahr", "frac_other_ahr",
                 "or_open", "p_open"]].copy()
        t.columns = ["谱系", "开放 bin", "落峰内", "开放组比例", "其余比例", "OR", "p"]
        H.append(t.to_html(index=False, classes="tbl", border=0, escape=False,
                           na_rep="—", float_format=lambda v: f"{v:.3f}"))
    H.append(f"""<p>bin 级阴性可能只是因为分辨率太粗（5 kb 平均掉一个 400 bp 的峰）、或者 liftOver 丢了
一半以上的 bin。所以把同一个问题下沉到基因级：取<strong>{gls.get('het_promoters', 0):,} 个异染色质基因启动子</strong>
（ConstitutiveHet ∪ H3K9me3），要求人源直系同源基因的 TSS ±2 kb 内有 AHR 峰，
背景是 {gls.get('background_genes_sampled', 0):,} 个随机基因，并按「在人类 SMC 中是否可检出」分层。</p>""")
    if not glt.empty:
        t = glt[["test", "n_a", "hits_a", "rate_a", "n_b", "hits_b", "rate_b",
                 "odds_ratio", "p", "fdr"]].copy()
        t.columns = ["检验", "n(组 A)", "命中", "比例 A", "n(组 B)", "命中", "比例 B",
                     "OR", "p", "FDR"]
        H.append(t.to_html(index=False, classes="tbl", border=0, escape=False,
                           na_rep="—", float_format=lambda v: f"{v:.3g}"))
    H.append(f"""<p>结果还是阴性，而且这次不能用“分辨率”解释：<strong>异染色质启动子的 AHR 峰比例
{gls.get('base_rate_ahr_prom_het', float('nan'))*100:.1f}%，背景 {gls.get('base_rate_ahr_prom_background', float('nan'))*100:.1f}%</strong>，
OR ≈ 1；带 AhR motif 的启动子<em>更低</em>（7.9% vs 11.6%）。限制到“人类 SMC 里可检出”的基因后
差异更不存在（10.6% vs 11.6%）。TCF21 峰与 DMSO 峰的比例同样与背景无差——也就是说，
这套峰在异染色质启动子上<em>既没有 AHR 信号，也没有对照 TF 的信号</em>。</p>
<p><strong>到这里，最该做的检查才出现：这个 ChIP 有没有找回它自己的阳性对照？</strong>
CYP1A1、CYP1B1、AHRR、TIPARP、NQO1、AHR 是 TCDD-AHR 的教科书靶基因；
一个在 TCDD 处理细胞里做出的 AHR ChIP，如果连它们都没有峰，那么它在别处“没有峰”这件事
说明的不是生物学。</p>""")
    if not glc.empty:
        t = glc[["peak_set", "window_kb", "n_targets", "target_mean_peaks",
                 "target_zero_frac", "random_mean_peaks", "enrichment"]].copy()
        t.columns = ["峰集", "窗口(kb)", "靶基因数", "靶基因平均峰数", "靶基因零峰比例",
                     "随机基因平均峰数", "富集倍数"]
        H.append(t.to_html(index=False, classes="tbl", border=0, escape=False,
                           na_rep="—", float_format=lambda v: f"{v:.2f}"))
    H.append(f"""<p>阳性对照<strong>没通过</strong>：在 TSS ±10 kb 内，六个经典靶基因平均 <strong>0.00 个 AHR 峰</strong>，
而随机基因平均 0.20 个——靶基因比随机基因<em>更干净</em>；DMSO 峰集同样如此（0.00 vs 0.01）。
把窗口放到 ±50 kb 也只有 1.5 倍，且全部来自 AHRR 与 CYP1A1 的零星远端峰。
一个合格的 AHR ChIP 在 CYP1A1/AHRR 上应当是信号最强的位点，而不是零。</p>
<p><strong>但要立刻把“峰集为零”与“信号轨为零”分开。</strong>把同一套 GSE276152 的 bigWig 拿出来
按窗口取信号：AHRR 的 TSS ±2 kb 平均信号 22.6，位于 3,000 个随机 TSS 窗口的<strong>第 100 分位</strong>；
CYP1A1 是 7.9（78 分位）；而那份 16,499 峰的 narrowPeak 在同一个 AHRR 窗口里是
<strong>0 个峰</strong>。继续放大：AHRR 基因体内 +22.9 kb 处有一个信号 109 的 summit，峰集里同样没有。
也就是说，<strong>不是“这套实验在靶基因上没有信号”，而是“它自己提交的峰集与它自己的信号轨对不上”</strong>。
本机没有 peak caller，无法裁决是峰调用丢了位点、还是这些位点本就是宽域升高，
但两种情形的处理方式相同：<strong>这份峰集不能用于结合判定</strong>，而它的信号轨可以——
10.9 用 summit 口径把这套数据与其他公开 AHR 图谱放在同一把尺子上重判了一次
（结果：AHRR 的 summit 落在 TSS 上、96 分位，CYP1A1 的 summit 在 +6.5 kb、97 分位，
即<em>这套原始数据是通过它自己的阳性对照的</em>）。</p>
<div class="note"><strong>因此这一节的可引用结论只有一条：没有可用的正交结合证据。</strong>
不是“AHR 不结合在异染色质启动子上”，也不是“AHR 只结合常染色质增强子”——
那两句话都要求 ChIP 本身成立，而这套数据没通过最低门槛（阳性对照）。
处理方式：10.4/10.5 的所有交集数字仍保留（它们说明<em>这批数据</em>里没有信号），
但正文一律不引用其为生物学的负证据；措辞仍写“<strong>AhR 依赖</strong>”，
理由是“依赖 Ahr 基因、依赖 AHR motif，且缺乏正交结合证据”，而不是“已被否证”。
真正能判决这件事的是自做的 ChIP-qPCR，位点与引物见第 11.2 节。</div>""")
    H.append(_fig("fig15_chip_gene.png",
                  "图 S3｜(a) GSE276152 的阳性对照：CYP1A1/CYP1B1/AHRR/TIPARP/NQO1/AHR 的 TSS ±10 kb 内平均 0 个 AHR 峰，"
                  "低于随机基因（0.20）；(b) 基因级复测：异染色质启动子（无论有无 AhR motif）的 AHR 峰比例与背景基因无差。"))
    H.append(_fig("fig13_external.png",
                  "图 S2｜人类冠脉投影。(a) 小鼠开放 bin 与人冠脉 SMC AHR 峰的 OR（全部不显著，OR ≈ 1）；"
                  "(b) 四套小鼠程序 × 人类冠脉五个细胞类型的得分矩阵——绿框为目标细胞类型，"
                  "红框为反例：ModSMC 面板在成纤维中反而更高（0.28 vs SMC 0.21）。"))

    # ---------------------------------------------------------------- 10.6
    H.append("<h3>10.6 外部验证 II：人类冠脉单细胞（GSE131778）</h3>")
    H.append(f"""<p>Wirka 等（Nat Med 2019）的人冠脉 scRNA-seq，{hus.get('n_cells', 0):,} 个细胞、8 例。
用独立标记（MYH11/ACTA2、PECAM1/VWF、CD68/CD14、DCN/LUM、CD3D）先分群，
再把三套小鼠程序投影进去——不依赖原文献的标签。</p>""")
    if not hum.empty:
        keep = hum[hum.programme != "sender_receiver"]
        t = keep[["programme", "compartment", "n_cells", "mean", "mean_other", "frac_positive"]].copy()
        t.columns = ["程序", "细胞类型", "细胞数", "型内均值", "型外均值", "阳性比例"]
        H.append(t.to_html(index=False, classes="tbl", border=0, escape=False,
                           float_format=lambda v: f"{v:.3f}"))
    H.append("""<p>结果一半对一半错。<strong>LAM 程序</strong>（Trem2/Lgals3/Csf1r/Spp1/Apoe）在巨噬里最高
（0.398 vs 其他 −0.086），<strong>内皮面板</strong>在内皮里最高（0.289 vs −0.055），
基质配体在成纤维里最高（0.687）——这三条支持跨物种对应。
但<strong>调制型 SMC 面板在人类成纤维里（0.279）反而高于 SMC（0.206）</strong>：
这五个基因（Lgals3/Fn1/Spp1/Col8a1/Vcam1）在小鼠里能把 ModifiedSMC 和 SMC 分开，
是因为它们是在小鼠聚类上挑出来的；到了人冠脉，成纤维同样表达它们。
这条必须写进局限——它说明小鼠的“调制型 SMC 特征”不能直接当作人类细胞类型标签使用。</p>
<p>另外，该文件没有疾病/正常标签（只有 8 个 barcode 后缀），所以<strong>不能</strong>做
“病变 vs 正常”的比较，本文也没有做。它只能支撑一句更弱的话：
三套程序落在正确的人类细胞类型上，而不是“在小鼠里定义的程序在人类病变中富集”。</p>""")

    # ---------------------------------------------------------------- 10.7-10.9
    try:
        import report_sec10b
        H.extend(report_sec10b.build())
    except Exception as e:                                   # keep the report buildable
        H.append(f"<p><em>10.7–10.9 未生成（{type(e).__name__}: {e}）</em></p>")

    # ---------------------------------------------------------------- 10.10
    H.append("<h3>10.10 核验后的结论修订表</h3>")
    rows = [
        ("SMC 的 H3K9me3 域被 TCDD 打开", "等深度 + 匹配常染色质置换",
         f"SMC d = {perm('Parenchyma','d_observed'):+.2f}，零模型 p = {perm('Parenchyma','p_one_sided'):.3f}",
         "保留，且升级为“唯一通过域特异性检验的谱系”"),
        ("跨谱系普遍击穿缓冲", "匹配常染色质置换", "内皮/间质/免疫与零模型无法区分",
         "删除“普遍”，改写为 SMC 特异"),
        ("内皮失守由 SMC-Ahr 驱动", "个体水平 + 深度", "细胞水平显著，个体水平无重复；等深度后 p 0.05–0.21",
         "降级为“方向一致、效应量大，受 n=1 限制”"),
        ("敲除后内皮 motif 回落是深度假象吗", "稀释到等深度", "方向不变、幅度衰减 20–30%",
         "保留，但 p 值须标注为满深度结果"),
        ("模型捕捉到 Timp1–Cd63 信号流", "度保持网络重连", "随机网络同样 4/4，误差更小",
         "删除该句，改为“模型对拓扑不可辨识”"),
        ("开放 bin 是 AhR 物理结合位点", "ChIP 交集 + 基因级复测 + 阳性对照 + 信号轨跨物种检验",
         "峰集在靶基因上零峰（0 vs 随机 0.20），但同一套数据信号轨 AHRR summit 96 分位、CYP1A1 97 分位；"
         "改读信号轨后，SMC 开放 bin 的 TCDD−DMSO 配对差 z = −0.75（覆盖度收紧后 −2.04），配体特异性为负",
         "环五判为不闭合：措辞改为“AhR 依赖”，删去“直接结合”；结合证据只认 11.2 的自做 ChIP-qPCR"),
        ("小鼠程序可映射到人冠脉", "GSE131778 投影", "LAM/内皮面板成立；ModSMC 面板在成纤维更高",
         "保留两条，ModSMC 一条进局限"),
        ("环境表观脆弱性独立于遗传易感性", "GWAS 富集（未变）", "OR 0.53（0.47–0.59）",
         "保留"),
    ] + _audit_rows()
    H.append("<table class='tbl'><thead><tr><th>原主张</th><th>核验方式</th><th>核验结果</th>"
             "<th>处理</th></tr></thead><tbody>")
    for a, b, c, d in rows:
        H.append(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>")
    H.append("</tbody></table>")

    H.append("""<div class="bound"><strong>这一步之后，还差什么才算“无懈可击”。</strong>
原始清单有三项；其中两项已经把“能做”算成了数字和表格，剩下的只是动手：
① 补动物——<em>算过了</em>：补 1 只 Ahr-KO 能把“KO 仍未回到对照水平”变成可检验命题（2 vs 3），
但补 1 只（2 vs 2）买不到“回落了多少”的检验，那需要 8–26 只/组。详见 11.1，含冻结的预注册；
② AHR ChIP-qPCR——<em>位点与引物已备好</em>：22 个位点、两个物种、阴阳性对照齐全（11.2），
42 对引物的全基因组特异性已逐个查过（10.9），Tier A/C 主引物全部单产物；
③ 全量 TE 计数（4 个文库约 11 GB）——本轮决定不做，pilot 定位为“引擎验证 + 归一化阴性对照”（第 7 节）。
<strong>另有两条是本轮新查出来、只有做实验才能解决的：</strong>
④ 组成检验的天花板是动物数，不是细胞数——3 只对照的结构决定了 p 永远 ≥0.048，
要真检验“TCDD 是否改变亚群配比”，需要每臂 5–6 只（10.7）；
⑤ 环五（AhR 直接结合）——本轮已经用最接近闭合的干实验做过一遍（10.11）：
峰集口径判不可用、信号轨口径判阴性、足迹口径判不可行，<strong>三条路都走完了，环五仍然是断的</strong>。
它现在是唯一一个“只能靠实验、没有替代表述”的环节：要么按 (A) 降级为“AhR 依赖”，要么做 11.2 的 ChIP-qPCR。
②不需要重新造模，①需要动物房排期，⑤需要先决定 (A) 还是 (B)。</div>""")

    # ---------------------------------------------------------------- 10.11
    H.append("<h3>10.11 环五能不能只靠干实验闭合——做完了，是不能</h3>")
    occ = _json("chip_alt/occupancy_summary.json")
    if not occ:
        H.append("<p><em>占据特异性检验尚未生成（scripts/35）。</em></p>")
        return H
    ot = _load("chip_alt/occupancy_tests.csv")
    pc = _load("chip_alt/occupancy_positive_control.csv")

    def orow(s, stat, col="obs_mean"):
        q = ot[(ot.set == s) & (ot.statistic == stat)]
        return float(q[col].iloc[0]) if len(q) else float("nan")

    def oz(s, stat):
        return orow(s, stat, "z")

    def op(s, stat):
        return orow(s, stat, "p")

    H.append(f"""<p><strong>问题不再是“峰集行不行”，而是“抛开峰集，信号本身答不答”。</strong>
10.5 的阴性被 10.9 归因给了峰集，于是留下一个必须回答的问题：如果直接读原始信号轨，
能不能证明人冠脉 SMC 里的 AHR 会结合到<em>小鼠 SMC 中因 TCDD 开放的那些 bin</em>上？
这一步本来要靠 ChIP-qPCR，但公共数据里恰好有同一套研究的人源臂——GSE276152 的
HCASMC AHR ChIP，<strong>自带 TCDD 与 DMSO 两条轨道</strong>——所以先把它当成干实验做一遍，
做法写在 <code>scripts/35_occupancy_specificity.py</code>，判读规则写在跑之前。</p>

<p><strong>设计：四种特异性必须同时成立。</strong>
mm10 的 5 kb bin 取中心点 liftOver 到 hg38（成功率 {occ['lift_rate']:.1%}，
双端都要求落在保守区块会把成功率压到 ~20%，而开放集与背景集受同一规则选择，残差是共享的），
重建 ±2.5 kb 窗口后在四类 bin 上读同一批轨道：
(i) <strong>配体</strong>——TCDD 高于 DMSO，且两者之差在 SMC 开放 bin 里大于背景；
(ii) <strong>细胞类型</strong>——同一批动物里<em>内皮</em>开放的 bin 不应有同样表现；
(iii) <strong>区域</strong>——GC 十分位 × 是否含 TSS 分层重抽样 5,000 次；
(iv) <strong>深度</strong>——覆盖度本身不能在两组间失衡，否则零填充会把“好比对”伪装成“被占据”。
统计量取两个：5 kb 窗口均值，和窗口内 500 bp tile 的最大值（summit 代理，真 AHR 足迹是 300–600 bp，
5 kb 均值正是把 AHRR 那个 96 分位 summit 抹平的原因）。</p>""")

    if len(pc):
        pcd = ", ".join(f"{r.gene} {r.pct_delta_summit:.0f}"
                        for _, r in pc.sort_values("pct_delta_summit", ascending=False).iterrows())
        H.append(f"""<p><strong>先证明这把尺子没瞎。</strong>同一个统计量、同一对轨道，
放在 6 个经典靶基因的 TSS ±2.5 kb 上，与 1,000 个随机 refFlat TSS 窗口比，
“TCDD 减 DMSO 的 summit”分位是：{pcd}。
<strong>AHRR 落在 99 分位、CYP1A1 落在 87 分位，其余四个（AHR、CYP1B1、NQO1、TIPARP）在地板附近</strong>——
这把尺子在这条轨道上只能看见最强的两个靶基因，这是它灵敏度上限的真实刻度，
后面所有阴性都必须带上这个刻度读。</p>""")

    H.append(f"""<p><strong>然后是 SMC 开放 bin（n=434，背景 8,000）。</strong>
TCDD 窗口均值 {orow('SMC_open','HCASMC_TCDD_mean'):.2f} vs 背景 {orow('SMC_open','HCASMC_TCDD_mean','null_mean'):.2f}
（z = {oz('SMC_open','HCASMC_TCDD_mean'):+.2f}，p = {op('SMC_open','HCASMC_TCDD_mean'):.2f}）；
summit {orow('SMC_open','HCASMC_TCDD_summit'):.2f} vs {orow('SMC_open','HCASMC_TCDD_summit','null_mean'):.2f}
（z = {oz('SMC_open','HCASMC_TCDD_summit'):+.2f}）。
<strong>配体特异性这一条直接反向</strong>：TCDD 与 DMSO 的<em>配对差</em>在开放 bin 里是
{orow('SMC_open','ligand_delta_mean'):.2f}，比背景的 {orow('SMC_open','ligand_delta_mean','null_mean'):.2f} <em>更低</em>
（z = {oz('SMC_open','ligand_delta_mean'):+.2f}，p = {op('SMC_open','ligand_delta_mean'):.2f}）。
唯一“显著”的一项是 <strong>DMSO 轨道</strong>的均值（{orow('SMC_open','HCASMC_DMSO_mean'):.2f} vs
{orow('SMC_open','HCASMC_DMSO_mean','null_mean'):.2f}，z = {oz('SMC_open','HCASMC_DMSO_mean'):+.2f}，p = {op('SMC_open','HCASMC_DMSO_mean'):.3f}）——
也就是说，这两个集合的差别在<strong>没有配体的时候就已经存在</strong>，那是染色质可及性/可比对性的差别，
不是 AHR 被招募。第 (iv) 条正好把这件事坐实：SMC 开放 bin 的窗口覆盖度是
{orow('SMC_open','HCASMC_TCDD_cov [balance]'):.3f} vs 背景 {orow('SMC_open','HCASMC_TCDD_cov [balance]','null_mean'):.3f}
（z = {oz('SMC_open','HCASMC_TCDD_cov [balance]'):+.2f}，p = {op('SMC_open','HCASMC_TCDD_cov [balance]'):.3f}）。</p>

<p><strong>把零填充这个借口也堵掉。</strong>只保留两条轨道都覆盖 ≥50% 的 bin（n=237）重跑：
TCDD−DMSO 均值差 {orow('SMC_open[cov>=.5]','ligand_delta_mean'):.2f} vs 背景
{orow('SMC_open[cov>=.5]','ligand_delta_mean','null_mean'):.2f}
（z = {oz('SMC_open[cov>=.5]','ligand_delta_mean'):+.2f}，p = {op('SMC_open[cov>=.5]','ligand_delta_mean'):.3f}），
summit 差 {orow('SMC_open[cov>=.5]','ligand_delta_summit'):.2f} vs
{orow('SMC_open[cov>=.5]','ligand_delta_summit','null_mean'):.2f}。
<strong>覆盖度一收紧，负号更清楚</strong>——不是零填充造成的。</p>""")

    H.append(f"""<p><strong>阴性对照也没有按剧本走，但原因很有信息量。</strong>
内皮开放 bin（n=29）在所有统计量上都与背景无法区分（TCDD 均值 z = {oz('EC_open','HCASMC_TCDD_mean'):+.2f}，
配对差 z = {oz('EC_open','ligand_delta_mean'):+.2f}），这一条“通过”了。
但 SKMEL2 那一对轨道给出了更要紧的读数：AHR 过表达臂的 z 是
{oz('SMC_open','SKMEL2_AHROE_mean'):+.1f}，<strong>空载体对照臂是 {oz('SMC_open','SKMEL2_ctrl_mean'):+.1f}</strong>——
两条臂一样高。<strong>这说明 434 个开放 bin 在任何一套 ChIP 里都会“显得富集”</strong>，
因为它们是可及染色质、可比对、GC 正常的一类区域；K562 的 CUT&amp;RUN 只有 z = {oz('SMC_open','K562_CUTRUN_mean'):+.2f}
（p = {op('SMC_open','K562_CUTRUN_mean'):.2f}）。换句话说，<strong>“这个集合富含信号”这件事本身没有 AHR 含义</strong>，
只有“TCDD 减 DMSO 的差”才有——而那个差是负的。</p>""")

    H.append("""<div class="note"><strong>判读：环五不能靠干实验闭合，这一条到此为止。</strong>
三条路都走完了：① 峰集口径——该研究的 narrowPeak 在靶基因上零峰，且与自己的信号轨矛盾（10.5、10.9），
判“不可用”；② 信号轨口径——直接读 TCDD/DMSO 两条轨道，在 SMC 开放 bin 上既没有配体特异性富集，
配对差还偏低（本节），判“阴性”；③ 足迹口径——GSE276870 没有 fragments/BAM，本机也无 bowtie2/bwa/macs2，
且足迹只能证明“某个因子占着”，证明不了是 AHR，判“不可行”。
<strong>再往上叠任何统计都只是在给同一个阴性结果换包装。</strong>
必须同时说清两件事，缺一件都会误导：
<em>第一，这是一次有刻度的阴性</em>——同一把尺子在同一轨道上只能看见 AHRR 与 CYP1A1 两个靶基因，
所以正确结论是“这套数据不支持，也不能排除一个幅度较小的、分散的结合”，不是“AHR 肯定不结合”；
<em>第二，这套人源数据不是独立证据</em>——GSE276152 与本文的鼠源数据（GSE276870/276872）属于<strong>同一项研究</strong>的人源臂，
所以它既是最匹配的对照（同一配体、同一设计、同一实验室），也意味着<strong>一个阳性结果只能算同研究内跨物种复现</strong>，
而一个阴性结果无法用“体系不同”解释掉。</div>""")

    H.append(_fig("fig18_occupancy.png",
        "(a) 同一统计量、同一对轨道在 6 个经典 AHR 靶基因上的分位：AHRR 99、CYP1A1 87，其余贴地板——"
        "这是该轨道的灵敏度刻度；(b) SMC 开放 bin 相对 GC/TSS 匹配背景的 z 值，TCDD 与 DMSO 均值都无法区分，"
        "配对差（TCDD−DMSO）为负，覆盖度收紧到 ≥50% 后负得更清楚；(c) 覆盖度分布——开放 bin 系统性更好覆盖，"
        "这正是第 (iv) 条要排除的混杂。"))

    H.append("""<p><strong>那环五怎么办。</strong>不是“再找一个数据集”，而是三选一：
<em>(A) 降级</em>——把“AhR 直接结合开放 bin”从主张里删掉，机制层只保留 Ahr 依赖（KO 逆转，已有），
标题与摘要改成“细胞自主启动 + 髓系/基质级联”，这条路现在就能投环境暴露类期刊；
<em>(B) 补湿实验</em>——11.2 的 ChIP-qPCR（22 个位点、双物种、阴阳性对照齐备、42 对引物特异性已逐个查过）是唯一能闭合环五的实验，
4–6 周之后才谈得上血管机制类期刊；
<em>(C) 换证据类型</em>——不证明“结合”，改证明“必要”：AHR 结合位点的<em>突变</em>或 XRE 缺失是否取消 TCDD 反应，
这仍然是湿实验，只是从 ChIP 换成报告基因。
<strong>干实验没有第四条路</strong>：公共数据里没有人源或鼠源 SMC 的 AHR 占据图谱能在本底之上分辨出这个效应，
而能分辨的那两个靶基因（AHRR、CYP1A1）本来就不在本文的候选位点里。</p>""")
    return H


def _fig(name, cap):
    import base64
    p = os.path.join(L.RES, "figures", name)
    if not os.path.exists(p):
        return ""
    b = base64.b64encode(open(p, "rb").read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b}"/>'
            f'<figcaption>{cap}</figcaption></figure>')
