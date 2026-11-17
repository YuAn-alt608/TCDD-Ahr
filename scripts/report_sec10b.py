"""Section 10.7-10.9 of the report: the confounding audit.

Kept as its own module because these three sections regenerate from five tables
that are produced by scripts 29-34 and must be quotable line by line:

  10.7  composition  (scripts/29)  - is the effect a shift in cell proportions?
  10.8  depth + cells (scripts/30/31) - is it a shallow library or a few cells?
  10.9  outside data (scripts/32/32b/33) - does any public AHR map, and are the
        primers actually specific?
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


def _f(x, n=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def _fig(name, cap):
    import base64
    p = os.path.join(L.RES, "figures", name)
    if not os.path.exists(p):
        return ""
    b = base64.b64encode(open(p, "rb").read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b}"/>'
            f'<figcaption>{cap}</figcaption></figure>')


def _tbl(df, cls="tbl"):
    return df.to_html(index=False, classes=cls, border=0, escape=False,
                      na_rep="—")


def build():
    H = []
    osum = _json("audit/summary.json")
    ct = _load("audit/composition_tests_8wk.csv")
    st = _load("audit/substate_tests_8wk.csv")
    env = _load("audit/composition_n1_envelope.csv")
    dec = _load("audit/endpoint_decomposition.csv")
    ver = _load("audit/endpoint_verdict.csv")
    dsum = _load("audit/depth_summary.csv")
    dlib = _load("audit/depth_by_library.csv")
    ssum = _load("audit/subsample_summary.csv")
    sg = _load("audit/sort_gate_check.csv")
    bwt = _load("chip_alt/bigwig_tierC.csv")
    bws = _json("chip_alt/bigwig_summary.json")
    sut = _load("chip_alt/summit_tierC.csv")
    sus = _json("chip_alt/summit_summary.json")
    altc = _load("chip_alt/candidates.csv")
    tcq = _load("chip_alt/tierC_qc.csv")
    qp = _load("chip/qpcr_primers.csv")

    # ================================================================ 10.7
    H.append("<h3>10.7 细胞组成：效应是“比例变了”还是“细胞里变了”</h3>")
    H.append("""<div class="note"><strong>这一节要挡的那句话。</strong>
“你看到的染色质开放变化，也许只是 TCDD 让某个亚群变多、Ahr-KO 让它变少，
你测的是细胞比例，不是细胞状态。”这句话对任何单细胞结论都成立，除非把比例显式
当成一个变量算出来。<strong>做法</strong>：CLR（多分类）或 logit（二分类）变换后做
Welch 检验并按族做 BH，同时给一个<em>动物水平精确置换 p</em>（3 vs 3 只有
C(6,3)=20 种标签划分，双侧最小可达 p = 1/21 ≈ 0.048，这是设计的天花板，不是方法的选择）；
再用 Kitagawa/Oaxaca 把每个冻结终点的 Δ 拆成组成项、态内项与交互项。</div>""")

    if len(sg):
        g = sg.copy()
        H.append("<p><strong>先记一条设计约束：分选门就是一道组成学的墙。</strong>"
                 "tdT<sup>+</sup> 库里只可能有 SMC/ModifiedSMC，tdT<sup>−</sup> 库里物理上不含 SMC，"
                 "所以组成只能在 fraction 内部比较，跨 fraction 的比例没有任何生物学含义。"
                 "本数据的实际构成（每个亚群的细胞数）为：")
        H.append(_tbl(g))
        H.append("</p>")

    # ---- L1
    if len(ct):
        c = ct.copy()
        c["stratum"] = c.stratum.str.replace("RNA|", "", regex=False)
        c = c.rename(columns={"stratum": "分选", "component": "亚群",
                              "p_ctrl": "对照比例", "p_8wk": "8周比例",
                              "log2_ratio": "log2", "p_welch": "Welch p",
                              "fdr_welch": "BH FDR", "p_exact_perm": "精确置换 p"})
        c = c[["分选", "亚群", "对照比例", "8周比例", "log2", "Welch p", "BH FDR",
               "精确置换 p"]]
        H.append("<p><strong>L1：fraction 内的细胞类型比例。</strong>"
                 "7 个检验，每个族一个 BH 校正：</p>")
        H.append(_tbl(c.round(4)))
        n_fdr = int((ct.fdr_welch < 0.05).sum())
        best = ct.loc[ct.fdr_welch.idxmin()]
        H.append(f"""<p>两个 p 值在这里<strong>结论相反，必须当场裁决</strong>：
按 Welch，{n_fdr} 个成分的 FDR 掉到 0.05 以下（最极端的是 {best.component}，
log2 = {best.log2_ratio:+.2f}，Welch p = {best.p_welch:.4f}，BH FDR = {best.fdr_welch:.3f}）；
按动物水平精确置换，<strong>一个都不显著</strong>（最小 p = {ct.p_exact_perm.min():.3f}，
即 21 个可达取值里第 3 极端）。</p>
<p>裁决理由只能有一条：<strong>每臂 3 个文库时，Welch 的正态假设无法辩护</strong>——
细胞数上万并不会让“3 只动物的比例”变成正态，它只会让标准误看起来很小。
标度上的证据是直接的：精确置换的可达最小值是 1/21 = 0.048，而所有 7 个检验的置换 p 都卡在
0.143–0.524，说明观测到的标签划分从来不是最极端的那一种。<strong>因此本节采信精确置换，
把“组成显著改变”的结论撤回</strong>；Welch 的数字保留在表里，作为“如果硬套参数检验会得到什么”的
对照，正文任何地方都不单独引用它。</p>
<p>方向本身仍值得记下：内皮 0.120 → 0.056（降一倍）、单核 0.028 → 0.097、T 细胞 0.010 → 0.034。
这三个成分的细胞级 bootstrap 区间都不跨零，但<strong>动物间跨度</strong>跨零——
也就是说，能确定的是“这几类细胞在 8 周这一臂的几只小鼠里都偏少/偏多”，
不能确定的是“这是处理效应”。后者的门槛是动物数，不是细胞数。</p>""")

    # ---- L2
    if len(st):
        best2 = st.loc[st.p_welch.idxmin()]
        n_small = int((st.fdr_welch < 0.30).sum())
        H.append(f"""<p><strong>L2：亚群比例（de novo 重算，不依赖被辩护的 leiden 注释）。</strong>
在每个细胞类型内部，用 harmonised embedding 上的 k-means（按细胞数取 k = 3–5）重划亚群，
再跑同一套检验，共 {len(st)} 个。<strong>FDR &lt; 0.05 的 0 个</strong>，
最小的 p 是 {best2.stratum.split('|')[-1]} 的 substate {best2.component}
（{best2.p_ctrl:.3f} → {best2.p_8wk:.3f}，Welch p = {best2.p_welch:.3f}，FDR = {best2.fdr_welch:.3f}）；
成纤维的三个 substate 恰好并列在 FDR ≈ 0.27，全部落在“看起来有变化、但 3 vs 3 分不开”的区间里。
这一步是必要的，因为如果把“亚群比例”当成解释变量、而亚群本身是照着终点指标划出来的，
就会循环论证；这里重算亚群只用表达/可及性嵌入，不看任何终点。</p>""")

    # ---- n=1 envelope
    if len(env):
        e = env[env.boot_width.notna()]
        med = float((e.boot_width / e.ctrl_width).median())
        H.append(f"""<p><strong>n=1 的两只动物怎么办：包络法，而不是 p 值。</strong>
16 周 WT 与 Ahr-KO 各只有 1 只小鼠，任何检验都无自由度。可量化的是“它离对照有多远，
以及这个距离是否超过测量噪声”：对每个亚群，取 3 只对照的 min–max 作包络，
再给单只动物的比例配一个细胞级 bootstrap 区间（固定动物、重抽细胞）。
结果：64 个（亚群 × 臂）组合里只有 <strong>{e.inside_envelope.mean()*100:.0f}%</strong>
落在 3 只对照的包络内；而<strong>细胞重抽样 CI 宽度与对照跨度宽度的中位数之比是 {med:.2f}</strong>
——即细胞抽样噪声<em>本身就比</em>动物间差异更宽。这句话的含义要读准：不是“细胞噪声大所以没事”，
而是<strong>在 n=3 的对照包络下，单只动物的组成数值既无法被判定为异常，也无法被判定为正常</strong>。
所以 10.7 的全部结论只用于“排除组成作为唯一解释”，不用于声称组成发生了变化。</p>""")

    # ---- decomposition
    if len(dec):
        d = dec[dec.estimator == "substate_pseudobulk"] if (dec.estimator ==
            "substate_pseudobulk").any() else dec
        H.append("""<p><strong>把 Δ 拆开：组成项、态内项、交互项。</strong>
对每个冻结终点，用逐细胞线性得分（均值满足 mean(all) = Σ w<sub>c</sub>·mean(c)，
伪批量得分<em>不</em>满足，因此分解与伪批量分开报）做 Kitagawa 分解：
Δ<sub>raw</sub> = Σ(w₁−w₀)μ₀ + Σw₀(μ₁−μ₀) + Σ(w₁−w₀)(μ₁−μ₀)，
再做 post-stratification 重加权，把处理组的亚群权重换成对照组的，得到 Δ<sub>adj</sub>。
“保留比例”= Δ<sub>adj</sub>/Δ<sub>raw</sub>：接近 1 说明效应与组成无关，接近 0 说明效应几乎全由组成解释。</p>""")
        rows = []
        for _, r in ver.iterrows():
            rows.append({
                "终点": f"{r.celltype} · {r.endpoint}",
                "对比": ("Ahr-KO" if "AhrKO" in r.contrast else
                        "16 周 WT" if "16wk" in r.contrast else "8 周"),
                "Δ_raw": r.delta_raw, "组成项": r.comp, "态内项": r.state,
                "组成占比 %": r.pct_composition,
                "保留 Δ": r.delta_retained,
                "同号亚群": f"{int(round(r.frac_strata_same_sign*100))}%",
                "判语": r.verdict.replace(" (survives composition matching)", ""),
            })
        H.append(_tbl(pd.DataFrame(rows).round(4)))
        nv = ver.verdict.value_counts()
        H.append(f"""<p>汇总：<strong>细胞自主 {int(nv.get('cell-intrinsic (survives composition matching)', 0))} 个、
混合 {int(nv.get('mixed', 0))} 个、组成敏感 {int(nv.get('composition-sensitive', 0))} 个</strong>。
需要点名的是两个组成敏感的：<strong>16 周 WT 的巨噬 LAM</strong>（Δ_raw = +0.22，
组成项 −0.33，保留比例 −0.02——单细胞水平上这个“上升”几乎完全是巨噬内部亚群配比变化）
与 <strong>Ahr-KO 的 SMC frac_het</strong>（保留 0.41）。
这两条在正文里已经不再作为“细胞状态改变”的证据；
而 <strong>Ahr-KO 的内皮异染色质去抑制（保留 0.70）与巨噬 LAM 过冲（保留 0.98）</strong>
经得起组成匹配，可以继续叫“细胞自主”。</p>""")
        H.append(_fig("fig16_confounding.png",
                      "(a) 每个冻结终点的 Δ 被拆成组成项与态内项，横轴是占 Δ_raw 的百分比，"
                      "右侧数字是组成匹配后保留的 Δ；"
                      "(b) 组成检验的效应量–显著性平面，红点为 Welch FDR&lt;0.05 的三个成分，"
                      "但它们全部过不了动物水平精确置换；"
                      "(c) 单只动物臂的细胞重抽样 CI 宽度 vs 三只对照的跨度，"
                      "虚线以上说明细胞噪声本身就比动物间差异宽；"
                      "(d) 削到 Ahr-KO 深度后 |d| 的比值，全部终点方向不变，"
                      "比值 &gt;1 的两个是 log1p 凹性导致的放大。"))
        H.append("""<p>还有一个反面用途值得写出来：<em>分解本身是有用的阴性对照</em>。
如果一个终点的 Δ 在组成匹配后<em>变大</em>（保留 &gt; 1，例如 8 周内皮 0.86、8 周 LAM 1.29），
说明组成变化的方向与态内变化相反、在<em>掩盖</em>效应，这也必须报——
否则读者会以为“组成无关”只能靠“保留 ≈ 1”来证明。</p>""")

    # ================================================================ 10.8
    H.append("<h3>10.8 深度与细胞抽样：把“浅 27%”与“少数细胞”两刀一起挡掉</h3>")
    if len(dlib):
        at = dlib[dlib.assay == "ATAC"]
        rn = dlib[dlib.assay == "RNA"]

        def _med(name):
            q = rn[rn["sample"] == name]
            return f"{int(q['median'].iloc[0]):,}" if len(q) else "—"

        H.append(f"""<p><strong>先承认事实。</strong>scATAC 四个文库的 bin 中位计数是
{', '.join(f"{str(r['sample']).replace('D2_atac_','')} {int(r['median']):,}" for _, r in at.iterrows())}，
<strong>Ahr-KO 比两个对照浅约 27%</strong>，这一点在 10.1 已按域级指标核过一遍。
scRNA 侧还有一颗更响的雷：16 周 WT 的 tdT<sup>−</sup> 库中位 UMI 只有
{_med('D2_rna_WT16_NTDT')}，而同臂 tdT<sup>+</sup> 是
{_med('D2_rna_WT16_TDT')}、Ahr-KO 的 tdT<sup>−</sup> 是
{_med('D2_rna_AKO16_NTDT')}——差 5–12 倍，
所以 16 周 WT 的 tdT<sup>−</sup> 读数（内皮、巨噬、成纤维都在这一侧）先天不可与大库比。</p>""")
        H.append("<p>逐库深度与细胞数：</p>")
        H.append(_tbl(dlib.rename(columns={"assay": "组学", "group": "组", "frac": "分选",
                                           "sample": "文库", "median": "中位计数",
                                           "n_cells": "细胞数"})))

    if len(dsum):
        s = dsum[dsum.target == "match_ako"]
        rows = []
        for (a, c, e, st_), g in s.groupby(["assay", "celltype", "endpoint", "statistic"]):
            for _, r in g.iterrows():
                rows.append({
                    "组学": a, "细胞": c, "终点": e,
                    "统计量": st_ + ("（log1p）" if st_ == "log1p" else
                                   "（线性 share）" if st_ == "share" else ""),
                    "对比": ("Ahr-KO" if "AhrKO" in r.contrast else "16 周 WT"),
                    "满深度 d": r.d_full,
                    "等深度 d": r.d_mean,
                    "3 次稀释 SD": r.d_sd,
                    "方向一致": r.sign_match,
                    "变化 %": r.attenuation_pct,
                })
        t = pd.DataFrame(rows)
        H.append("""<p><strong>把每个文库稀释到 Ahr-KO 的深度，再重算 4 个冻结终点。</strong>
方法上必须说准：GSE276870/276872 没有释放 fragments/BAM，所以<em>不能</em>做 fragment 级下采样。
这里做的是<strong>二项稀释</strong>——对每个细胞的计数矩阵，每个条目以 p = T*/n<sub>i</sub>
独立保留——它是“多项式 reads 下采样”的精确采样模型，等价于把测序仪读数按比例削掉，
但不是把文件读短。σ 固定为冻结值（ATAC：|c₁−c₂|/√2×1.5，df=1；RNA：两个重复臂的合并个体间 SD，df=4），
所以下表“变化 %”只反映 Δ，不掺 σ 的重新估计。</p>""")
        H.append(_tbl(t.round(4)))
        js = _json("audit/depth_summary.json")

        def _d(cell, ep, col, stat="per_cell"):
            q = dsum[(dsum.celltype == cell) & (dsum.endpoint == ep) &
                     (dsum.contrast.str.contains("AhrKO")) &
                     (dsum.target == "match_ako") & (dsum.statistic == stat)]
            return float(q[col].iloc[0]) if len(q) else np.nan

        H.append(f"""<p>三个结论，按可靠性排序：</p>
<ol>
<li><strong>方向 12/12 全保</strong>（4 个终点 × 3 次独立稀释，含 RNA 的两个统计量）。
即使把高深度样本削到 Ahr-KO 的水平，没有一个终点翻号。这条直接堵住“深度伪影”的质疑。</li>
<li><strong>幅度衰减是不均匀的，而且不能一刀切地说“衰减”</strong>：
SMC 的 frac_het 只掉 3.2%（d {_d('Parenchyma','frac_het','d_full'):.2f}
→ {_d('Parenchyma','frac_het','d_mean'):.2f}），
巨噬 LAM 掉 10–15%，内皮的 prom_het_rel 掉 34.5%（d {_d('Endothelial','prom_het_rel','d_full'):.2f}
→ {_d('Endothelial','prom_het_rel','d_mean'):.2f}）；
而内皮的 het_derepression 在 log1p 上<strong>被放大</strong>（{_d('Endothelial','het_derepression','d_full','log1p'):.2f}
→ {_d('Endothelial','het_derepression','d_mean','log1p'):.2f}）。</li>
<li><strong>那个“放大”是真偏差，但方向对论文有利，所以更要解释清楚</strong>：
log1p(CPM) 是凹函数，浅库的 Poisson 噪声经 Jensen 不等式被系统性压低，
于是“处理组 vs 对照组”的差被<em>夸大</em>。为此另报一个<strong>逐细胞线性</strong>的统计量
share = ΣCPM/n<sub>genes</sub>：它对单个细胞的深度不敏感，方向与 log1p 完全一致。
但<strong>它的幅度也变了</strong>（内皮 het_derepression 的 Δ 从 −1.67 到 −3.62），
原因不是凹性，而是稀释只在每个细胞内按 min(1, T*/n<sub>i</sub>) 截断——
深细胞被削得多、浅细胞一动不动，伪批量的权重因此向浅细胞偏移。
诚实地说，<strong>两个统计量都只在方向上稳健，幅度都随深度变</strong>；
所以本文引用的是方向与量级区间，不引用小数点后第二位。</li>
</ol>
<p>60% 压力档（把深度再砍到 2,231）保留在表里但不作主张：4 个终点里有 2 个（免疫与间质的
prom_het_rel）在那里翻号，说明“再浅一倍”就到了这套数据的悬崖边。这个边界写出来比藏起来有用。</p>""")

    if len(ssum):
        t = ssum.copy()
        t["终点"] = (t.celltype + " · " + t.endpoint + " · " +
                     np.where(t.contrast.str.contains("AhrKO"), "Ahr-KO", "16 周 WT"))
        t = t[["终点", "d_full", "draw_lo", "draw_hi", "draw_sd", "frac_same_sign",
               "frac_within_50pct", "jack_lo", "jack_hi", "d_trim1pct"]]
        t.columns = ["终点", "满深度 d", "子抽样 2.5%", "97.5%", "SD",
                     "同号比例", "落在 ±50%", "jackknife 下界", "上界", "1% 截尾 d"]
        H.append("""<p><strong>“是不是几只异常细胞撑起来的”：每只动物内 1,000 次 80% 无放回重抽。</strong>
重抽样<em>只在动物内部</em>做（不跨动物），因此它只回答“细胞抽样”这一个问题；
σ 仍用冻结值，所以 d 的散布不掺任何重新标准化。同时给精确的 leave-one-cell-out
jackknife（均值可解析：删掉一个细胞后 (S−x<sub>i</sub>)/(n−1)），那是“单个细胞最多能翻多大浪”的上界。</p>""")
        H.append(_tbl(t.round(4)))
        H.append(f"""<p>结果：<strong>1,000 次子抽样里 100% 保持同号</strong>，
最低的“落在满深度 d 的 ±50% 内”比例是 {ssum.frac_within_50pct.min()*100:.1f}%；
最脆的是 {ssum.loc[ssum.frac_within_50pct.idxmin()].celltype} 的
{ssum.loc[ssum.frac_within_50pct.idxmin()].endpoint}
（{('Ahr-KO' if 'AhrKO' in ssum.loc[ssum.frac_within_50pct.idxmin()].contrast else '16 周 WT')} 对比），
它的 jackknife 范围是 [{ssum.loc[ssum.frac_within_50pct.idxmin()].jack_lo:.2f},
{ssum.loc[ssum.frac_within_50pct.idxmin()].jack_hi:.2f}]——<strong>删掉任意单个细胞都无法翻号</strong>。
1% 截尾后的 d 与全量 d 的差在 0.1 以内。<strong>结论：主终点不是离群细胞驱动的。</strong>
但要说清它<em>不</em>证明什么：它不提高动物水平的功效，也不替代 10.2 的个体水平重算。</p>""")

    # ---- pseudo-bulk stance
    H.append("""<div class="bound"><strong>Pseudo-bulk 的 p 值：本文不再“求显著”，改报两件可核的事。</strong>
<code>results/power/preregistration.md</code> 已经把最小可检效应量算死：
合并方差 t 下 80% 功效要求 2v2 的 d = 5.58、2v3 = 3.88、3v3 = 3.07，
而四个主终点的 KO−WT 实测 |d| 只有 0.81–1.54。这意味着<strong>追求伪批量 p &lt; 0.05 在数学上就是无解的</strong>，
再跑一百次也不会变，继续报“p = 0.14，趋势显著”只是把功效不足伪装成发现。因此本节改成两条：
<strong>① 方向一致性审计</strong>——同一终点在细胞水平、亚群伪批量、深度残差化三种估计量下方向是否一致
（10.7 的分解表与 10.8 的稀释表已给出：一致）；
<strong>② 浅测序下采样敏感性</strong>——把高深度样本削到 Ahr-KO 水平后，|d| 的<em>排序与方向</em>是否稳定
（4 个终点 × 3 次稀释：12/12）。
正文与摘要里凡是“个体水平不显著”的地方，都同时给出 d 与功效上限，
让读者自己能算出“这不是没效应，是买不起这个检验”。</div>""")

    # ================================================================ 10.9
    H.append("<h3>10.9 外部 AHR 占据图谱的替补，与 Tier C 门禁</h3>")
    n_cand = len(altc) if len(altc) else 0
    H.append(f"""<p><strong>为什么要替补。</strong>10.5 的结论建立在“GSE276152 找不到自己的阳性对照”
之上——一个连 CYP1A1 都没有峰的 AHR ChIP-seq，既不能支持也不能反驳任何结合假说。
但一套数据废了不等于全世界都废，所以重新检索 GEO（<code>AHR[Title] AND (ChIP-seq OR CUT&amp;RUN OR CUT&amp;Tag)</code>，
人/鼠，共 {n_cand} 个候选系列），并把<strong>Tier C 门禁放在任何目标基因分析之前</strong>：
6 个经典 AHR 靶基因（CYP1A1/CYP1B1/AHRR/TIPARP/NQO1/AHR）里至少 2 个要在 TSS ±10 kb 内
有自己的峰（或 summit），门槛才叫过。门禁不过，数据集就不进入本节任何推断。</p>""")

    # ---- peak-based result
    H.append("""<p><strong>峰口径：几乎没有数据可判。</strong>
九个系列里只有 GSE205502（原代人肝细胞，TCDD）提供了 processed peak，而且是一份
<em>差异峰集</em>（1,984 峰，TCDD vs 溶剂对照），不是全峰集。拿它跑 Tier C 的结果是
<strong>6 个靶基因的 TSS ±10 kb 内平均 0.0 个峰、enrichment 0.0，直接 FAIL</strong>。
这不是“肝细胞里 AHR 不结合靶基因”，而是差异峰集这个设计本身就不能当占据图谱用
（它只保留组间差异，靶基因上恒强的组成型占据反而不在里面）。</p>""")

    # ---- mean-window result
    if len(bwt):
        by = (bwt.groupby(["dataset", "window"])
              .agg(n_above=("above_p95", "sum")).reset_index())
        rows = []
        for ds in by.dataset.unique():
            for w in ["2kb", "10kb"]:
                g = bwt[(bwt.dataset == ds) & (bwt.window == w)]
                if not len(g):
                    continue
                hit = ", ".join(g[g.above_p95].gene.tolist()) or "无"
                rows.append({"数据集": ds.replace("GSE", "GSE "), "窗口": f"±{w[0]} kb",
                             "超过背景 p95 的靶基因": hit,
                             "个数": f"{int(g.above_p95.sum())}/{len(g)}"})
        H.append("""<p><strong>信号口径一：TSS 窗口均值。</strong>
其余系列只放了 bigWig。本机没有可用的 peak caller/pyBigWig，于是手写了一个
bigWig 读取器（B+ tree + cirTree + bedGraph 块，<code>scripts/lib_bigwig.py</code>），
直接取窗口均值，与同一轨道内 3,000 个随机 TSS 窗口的 95 分位比较。</p>""")
        H.append(_tbl(pd.DataFrame(rows)))
        H.append(f"""<p>{bws.get('caveat','')}</p>
<p>均值口径对两种情形都会误判：一个真 summit 若离注释 TSS 有 5–8 kb，会被平均掉；
一片宽而低的高原（低可比对性、拷贝数、hyper-ChIPable 区）则会不用任何 summit 就抬上去。
<strong>这两种情况在本文的数据里同时出现了</strong>，所以必须再加一个口径。</p>""")

    # ---- summit result
    if len(sut):
        rows = []
        for ds, g in sut.groupby("dataset"):
            for _, r in g.sort_values("pct", ascending=False).iterrows():
                rows.append({"数据集": ds.replace("GSE", "GSE "), "靶基因": r.gene,
                             "summit 信号": r.summit, "距 TSS": f"{int(r.offset):+d} bp",
                             "背景 p95": r.null_p95, "分位": r.pct,
                             "过门禁": "✓" if r.pass_summit else ""})
        H.append("""<p><strong>信号口径二：±10 kb 内的最高 500 bp bin（summit 代理）。</strong>
这个统计量对 TSS 注释的 ±1 kb 误差不敏感，也不会被远程高原拉高，因为比较的是分布的高端
（与同轨道 500 个随机 TSS 窗口的 summit 分布比）。门禁仍是 ≥2/6。</p>""")
        H.append(_tbl(pd.DataFrame(rows).round(3)))
        npass = sut.groupby("dataset").pass_summit.sum().sort_values(ascending=False)
        H.append(f"""<p><strong>换成 summit 口径，结论反转了一半。</strong>
<s>没有任何一套能找回靶基因</s> —— 正确的说法是：<strong>没有任何一套能找回全部六个，
但 {int((npass>=2).sum())} 套能过 2/6 的门槛</strong>：SKMEL2 的 AHR 过表达库
（CYP1A1、CYP1B1，两个 summit 都在 +6.5/+9.5 kb 的远端增强子区）与
<strong>HCASMC 的 TCDD 库（CYP1A1 +6.5 kb、AHRR 恰好落在 TSS 上）</strong>。
而同一个 HCASMC 系列的 DMSO 对照库里，AHRR 只有 57 分位、CYP1A1 只有 16 分位，
唯一过线的 NQO1（97 分位）在 TCDD 库里掉到 48 分位——<strong>这不是一套乱噪，
它给出的是 TCDD 特异性的 AHRR/CYP1A1 结合</strong>。</p>
<div class="note"><strong>于是 10.5 的阴性必须重新归因。</strong>
同一套 GSE276152 实验：信号轨在 AHRR 的 TSS 上有 96 分位的 summit、在 CYP1A1 的 +6.5 kb
有 97 分位的 summit，而它自己提交的 narrowPeak 在这两个窗口里<strong>一个峰都没有</strong>
（AHRR 基因体内 +22.9 kb 处甚至有一个信号 109 的强 summit，峰集里同样没有）。
峰集与信号轨对不上，只能说明<strong>“这份峰集不能用”，不能说明“AHR 不结合”</strong>。
10.5 的措辞因此收紧为：<em>该研究的 processed peak 集没有通过它自己的阳性对照，
故不能用于结合判定；其原始信号轨在 CYP1A1/AHRR 上是通过阳性对照的。</em></div>
<p>反过来，这也给了本文一个<strong>可引用的小结论</strong>：在唯一一套人冠脉 SMC 的
TCDD-AHR 占据数据里，六个经典靶基因有两个（CYP1A1、AHRR）能被找回，
其余四个（CYP1B1、TIPARP、NQO1、AHR）不能——而后者在另外三套数据里也各自只偶尔过线。
<strong>“AHR 占据”这件事在不同细胞类型之间的可复现性本来就只在最强的两个靶基因上成立</strong>，
这既解释了为什么本文的异染色质位点检验注定阴性，也说明用“有没有峰”去否定
异染色质结合在方法上是不成立的。判决仍交给 11.2 的自做 ChIP-qPCR，
它自带 6 个阳性对照与不含 motif 的阴性对照，而任何外部数据都没有。</p>""")

    # ---- primer specificity
    if len(qp) and "verdict" in qp.columns:
        vc = qp.verdict.value_counts()
        ok = int(vc.get("ok", 0)); rv = int(vc.get("review", 0)); rd = int(vc.get("redesign", 0))
        H.append(f"""<p><strong>42 对引物的特异性：从“未检查”变成逐个数字。</strong>
UCSC hgPcr 被 Cloudflare 拦，本机也没有 isPcr/BLAST+ 二进制，于是直接实现了电子 PCR
（<code>scripts/33_primer_specificity.py</code>）：mm10 与 hg38 的 2bit 逐条染色体展开为 bytes，
以 3′ 端 15 nt 精确锚定（<code>bytes.find</code>，two-way 算法）定位候选，再允许 5′ 余部 ≤2 错配，
产物长度限定 40–3,000 bp，反向引物取反向互补后同样索引到正链。
判定：{ok} 对单产物（unique）、{rv} 对双产物（下单前复核）、{rd} 对多产物（需重新设计）。
每个基因组的唯一产物数与产物长度都已写回 <code>results/chip/qpcr_primers.csv</code>
的 <code>n_products</code> / <code>sizes</code> / <code>verdict</code> 列，
逐产物坐标在 <code>results/chip/ispcr_hits.csv</code>，可复算。</p>""")
        bad = qp[qp.verdict != "ok"]
        if len(bad):
            b = bad[["genome", "cell", "tier", "mouse" if "mouse" in bad else "human",
                     "n_products", "sizes", "verdict"]].copy()
            b.columns = ["基因组", "细胞", "层级", "基因", "产物数", "产物长度", "判定"]
            H.append("<p>需要处理的引物对：</p>")
            H.append(_tbl(b))
            H.append("""<p><strong>不能停在这里说“已查、有 2 对要重设计”——直接把重做的一步跑完。</strong>
<code>scripts/33b_primer_redesign.py</code> 对这两对做了闭环替换：以原扩增子 ±300 bp 为模板，
primer3 强制跨过原扩增子设计，失败则把窗口放宽到 TSS ±800 bp / ±1500 bp 并要求跨 TSS；
每个候选都拿同一套<em>全基因组</em>电子 PCR 复检，只有“恰好 1 个产物、且长度与原设计相差 ≤30 bp”
才接受。<strong>结果：两对都换不掉</strong>——Iffo2 的两个候选在 mm10 全基因组上仍有
2,825 / 2,827 个 144–146 bp 产物，TIPARP 备选对的三个候选仍有 32–36 个 798–2,040 bp 产物。
这说明问题不在引物本身而在<strong>位点</strong>：这两处启动子落在重复序列里，
任何 20-mer 都找不到唯一 15-mer 锚点。</p>
<p>处理因此不是“重设计”而是“<strong>从下单表移除</strong>”，并写明代价：
<strong>Tier A 的 14 对与 Tier C 的 12 对主引物全部是单产物</strong>，
失效的两对分别是 mm10 的 tier B 对照（Iffo2）与 hg38 的 Tier C <em>备选</em>引物（TIPARP）——
即<strong>主面板不受影响，损失的是一个对照位点与一个备份</strong>。
Tier B 的对照少一个不改变判读逻辑（A 与 C 都有各自的对照），
但如果要补，建议换到同一基因的另一个 TSS 或第一内含子，而不是在重复区里继续挪。</p>""")
        else:
            H.append("<p>全部引物对在各自基因组里都是单产物，42 对可直接下单。</p>")

    H.append(_fig("fig17_subsample_tierC.png",
        "(a) 每只动物内 1,000 次 80% 细胞重抽样得到的 d 分布（已除以满深度 d），绿条是单细胞 "
        "jackknife 的范围——最极端的一只细胞也翻不了号；(b) Tier C 门禁逐数据集的结果，柱上是 "
        "TSS ±2 kb 的通过数，柱下是 ±10 kb 的结果，虚线为 2/6 的通过门槛。"))
    return H
