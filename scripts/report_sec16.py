"""Section 16 of the report: the three responses to the "reviewer will push here"
list - donor stratification of the branch point, the stromal continuum behind the
Matrix panel, and what a 3-vs-3 p-value can physically mean.

Separate module for the same reason as report_sec10: the prose has to move with
the numbers, and the numbers come from three JSON files written by scripts 41-43.
"""
import json
import os

import pandas as pd

import lib_core as L


def _load(p):
    q = os.path.join(L.RES, p)
    return pd.read_csv(q) if os.path.exists(q) else pd.DataFrame()


def _json(p):
    q = os.path.join(L.RES, p)
    return json.load(open(q)) if os.path.exists(q) else {}


def _fig(name, cap):
    import base64
    p = os.path.join(L.RES, "figures", name)
    if not os.path.exists(p):
        return ""
    b = base64.b64encode(open(p, "rb").read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b}"/>'
            f'<figcaption>{cap}</figcaption></figure>')


def _tbl(df, fmt="{:.3f}"):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    return d.to_html(index=False, classes="tbl", border=0, escape=True)


def build():
    H = []
    r3 = _json("human/r3_summary.json")
    sc = _json("human/stromal_continuum_summary.json")
    fl = _json("power/floor_summary.json")
    et = _load("power/endpoint_effect_table.csv")
    per = _load("human/r3_within_donor.csv")
    strat = _load("human/r3_stratified_partial.csv")
    floor = _load("power/exact_permutation_floor.csv")

    H.append("<h2>16. 三处会被死磕的地方：分叉的统计单位、基质面板的归属、3v3 的 p 值上限</h2>")
    H.append("""<p>第 15 节把“这句话的数字对不对”查完了。这一节处理另一类问题：
<strong>数字对，但推理的单位或口径会被审稿人一枪打穿</strong>。
三件事各有一个具体的攻击面——合并相关可能只是 Simpson 悖论（16.1）、
把面板标成 SMC 专属可能本来就把靶子设错了（16.2）、
把 0.143 读成“接近显著”其实是被组合数锁死（16.3）。
三条都不是补一句话能解决的，所以各配了一个脚本与一个判据，结果有好有坏，坏的照写。</p>""")

    # ================================================================ 16.1
    H.append("<h3>16.1 分叉的统计单位：从“一次合并计算”改成“四种算法 + 供体聚类”</h3>")
    H.append("""<p>第 15.3 节的 R3 只有一个数：把 11,756 个细胞混在一起算 Spearman。
这个算法有一个致命弱点——八个供体贡献的细胞数从几百到几千不等，
只要有两三个供体同时在内皮和髓系上都偏高，合并相关就会被它们拉动，
而这件事与“两条通路是不是分开的”完全无关（Simpson 悖论）。
所以 R3 重算成四个版本，每个版本挡一种攻击：</p>
<ul>
<li><strong>供体内（within-donor）</strong>：八个供体各自算一次，得到八个独立估计，
再用配对 Wilcoxon 比较“髓系–基质”与“内皮–基质”的差——挡“某几个供体主导”；</li>
<li><strong>供体固定效应</strong>：先把每个供体的均值减掉再合并，等价于随机截距混合模型
（斜率同质时）——挡“供体均值不同造成的虚假相关”；</li>
<li><strong>偏相关</strong>：把每个程序分数先对 log nCount、核糖体比例、线粒体比例和供体哑变量回归，
取残差再算相关——挡“测序深度高的细胞什么都高”；</li>
<li><strong>单一细胞类型内部</strong>：只在巨噬/内皮/SMC/成纤维/ T 细胞内部算——挡“细胞类型构成”。</li>
</ul>
<p>推断单位是<strong>供体不是细胞</strong>：置信区间用按供体重抽样的聚类 bootstrap（2,000 次），
因为同一供体内的细胞不独立。</p>""")

    if r3:
        icc = r3["donor_variance_share"]
        fe = r3["donor_fixed_effects"]
        pc = r3["partial_depth_ribo_mito_donor"]
        ci = r3["cluster_bootstrap"]
        # the pooled point estimates are the bootstrap medians; the fixed-effect
        # and partial versions are exact numbers from the same file
        t = pd.DataFrame([
            dict(算法="合并（第 15 节原口径，聚类 bootstrap 中位）", **{
                "髓系–基质": ci["LAM|Matrix"]["median"],
                "内皮–髓系": ci["ECpanel|LAM"]["median"],
                "内皮–基质": ci["ECpanel|Matrix"]["median"]}),
            dict(算法="供体固定效应（每供体去均值）", **{
                "髓系–基质": fe["LAM|Matrix"], "内皮–髓系": fe["ECpanel|LAM"],
                "内皮–基质": fe["ECpanel|Matrix"]}),
            dict(算法="偏相关（供体 + 深度 + 核糖体/线粒体）", **{
                "髓系–基质": pc["LAM|Matrix"], "内皮–髓系": pc["ECpanel|LAM"],
                "内皮–基质": pc["ECpanel|Matrix"]}),
            dict(算法="各细胞类型内（6 个区室中位数）", **{
                "髓系–基质": float(strat["LAM|Matrix"].median()),
                "内皮–髓系": float(strat["ECpanel|LAM"].median()),
                "内皮–基质": float(strat["ECpanel|Matrix"].median())}),
        ])
        H.append(_tbl(t))
        H.append(f"""<p><strong>供体解释了多少方差：</strong>LAM {icc['LAM']:.1%}、Matrix {icc['Matrix']:.1%}、
ECpanel {icc['ECpanel']:.1%}、ModSMC {icc['ModSMC']:.1%}。
也就是说供体间差异只占一成上下，<em>没有</em>哪一个供体能独自决定合并相关的符号。</p>""")

        wt = r3["within_donor_tests"]["LAM|Matrix > ECpanel|Matrix"]
        wl = r3["within_donor_tests"]["LAM|Matrix > ECpanel|LAM"]
        d1 = r3["cluster_bootstrap_delta"]["delta_matrix"]
        d2 = r3["cluster_bootstrap_delta"]["delta_lam"]
        H.append(f"""<p><strong>能站住的：</strong>供体内 {wt['n_positive']}/{wt['n_donors']} 个供体都是
“髓系–基质 &gt; 内皮–基质”，中位差 {wt['median_diff']:+.2f}（Wilcoxon p = {wt['p_wilcoxon']:.4f}，
n = 8 的单侧 Wilcoxon 最小可能 p 就是 1/2⁸ = 0.0039，所以这已经贴着下限）；
聚类 bootstrap 的差是 {d1['median']:+.2f}，95% CI [{d1['lo']:+.2f}, {d1['hi']:+.2f}]，
{d1['frac_positive']:.0%} 的重抽样为正。<strong>换上任何算法，内皮–基质都在 0 附近，髓系–基质都在 0.13–0.35。</strong></p>
<p><strong>站不住、必须撤回的：</strong>本文原先在 15.3 里暗示“髓系与基质是三者中最紧的一对”。
把这个判断也做成供体配对比较后：中位差 {wl['median_diff']:+.2f}、{wl['n_positive']}/{wl['n_donors']} 供体为正、
Wilcoxon p = {wl['p_wilcoxon']:.3f}，聚类 CI [{d2['lo']:+.2f}, {d2['hi']:+.2f}] 跨过 0
（{d2['frac_positive']:.0%} 为正）；扣掉深度后更是 {pc['LAM|Matrix']:.3f} 对 {pc['ECpanel|LAM']:.3f}，顺序反过来了。
所以这句删掉。留下的准确说法只有一句：

<strong>内皮程序与基质程序正交（四种算法下 |r| ≤ 0.06）；髓系–基质在合并、固定效应、偏相关三种口径下都保持中等正耦合（0.14–0.24），而内皮–髓系只在合并口径下是 0.24、进到细胞类型内部就掉到 0.03 左右。</strong>
分叉的实质是“内皮这一端不接基质那一端”，不是“髓系与基质最亲密”。</p>""")
        H.append("""<div class="note"><p><strong>顺带查出一件事，正好反过来支持审稿人的担心。</strong>
合并口径下“内皮–髓系”是 0.24，看着不小；但一进到单一细胞类型内部，它就掉到 0.03–0.09。
也就是说这 0.24 基本<em>全部</em>来自细胞类型构成（比如内皮分数在 T/other 里也偏高），
不是两条通路真的耦合。这条同时证明两件事：合并相关确实会被构成拉高（审稿人是对的），
而我们改成“细胞类型内 + 供体聚类”的口径是必要的（不是走过场）。</p></div>""")
        H.append(_fig("fig19_branch_donor.png",
                      "图 19　分叉的四种算法。(a) 八个供体各自的相关估计，连线是同一个供体在"
                      "“髓系–基质”与“内皮–基质”上的配对；(b) 单一细胞类型内部的偏相关，"
                      "内皮–基质在每一个区室里都贴着 0。"))

    # ================================================================ 16.2
    H.append("<h3>16.2 基质面板：不是“小鼠和人对不上”，而是靶子本来就设错了</h3>")
    H.append("""<p>第 15.3 节把“Matrix 面板在人冠脉里落在成纤维（AUC 0.687）而不是 SMC（0.162）”
写进了局限。写完还是不舒服：这句话读起来像跨物种不一致，而跨物种不一致是最难辩解的一类缺陷。
于是回到小鼠侧查同一件事——<strong>这个面板在小鼠里到底是不是 SMC 专属</strong>。
答案是不是，而且在小鼠里就已经不是了。</p>""")
    if sc:
        ms = sc["mouse"]
        H.append(f"""<p><strong>小鼠侧（同一只动物内比较，8 只）：</strong>matrix_ligand 模块分数在
ModifiedSMC 比收缩型 SMC 高 <strong>{ms['median_mod_minus_smc']:+.2f}</strong>
（{ms['n_up']}/{ms['n_animals']} 只上升，Wilcoxon p = {ms['p_wilcoxon']:.4f}），
而 <strong>ModifiedSMC 与成纤维细胞几乎完全一样</strong>（差 {ms['median_mod_minus_fib']:+.2f}，
p = {ms['p_mod_vs_fib']:.2f}）。
一个“SMC 专属”的程序不可能在成纤维细胞里同高。所以问题不在人，在于
<strong>R2 预先声明的目标（SMC）与面板实际标记的东西（基质生产状态）不是一回事</strong>。</p>""")
        g = {r["target"]: r for r in sc["human_correlations"] if r["compartment"] == "SMC"}
        H.append(f"""<p><strong>人侧（SMC 内部，2,557 个细胞，供体 + 深度已扣除）：</strong>
如果面板标记的是状态而非谱系，那么在同一批被判为 SMC 的细胞内部，它就该随去分化程度移动。
事实如此：Matrix 与纤维肌细胞标志 <strong>TNFRSF11B r = {g['TNFRSF11B']['partial_r']:+.2f}</strong>
（CI {g['TNFRSF11B']['ci_lo']:+.2f}–{g['TNFRSF11B']['ci_hi']:+.2f}）、
<strong>LUM r = {g['LUM']['partial_r']:+.2f}</strong>，与非循环的 ECM 基因（COL1A1/POSTN）r = {g['ecm']['partial_r']:+.2f}；
把 SMC 按去分化指数分十档，Matrix 从第 1 档的 {sc['gradient'][0]['matrix']:+.2f} 单调升到第 10 档的
{sc['gradient'][-1]['matrix']:+.2f}（ρ = {sc['gradient_rho']:+.2f}），逼近而略低于成纤维均值
{sc['fib_matrix_mean']:+.2f}。间质内 Fib vs SMC 的 AUC 是 {sc['auc_fib_vs_smc']:.3f}——
这条连续体在人体里是连续可测的，只是不适合用“SMC 是/不是”的二分类去判它。</p>""")
        H.append("""<div class="bound"><p><strong>但预设的判据没达标，这条得写清楚。</strong>
原先的预期是“Matrix 与 MYH11/ACTA2 强负相关（r &lt; −0.3）”。实测：
<strong>MYH11 r = {m:.2f}（95% CI {ml:.3f} 至 {mh:.3f}）、ACTA2 r = {a:.2f}</strong>，
两个都没到 −0.3。也就是说<em>“失去收缩程序”这一侧的证据只有中等强度</em>，
梯度的主要驱动力是“获得纤维肌细胞/基质程序”那一侧（TNFRSF11B、LUM、COL1A1 都是 +0.3 上下）。
诚实的写法是：这个面板追踪的是基质生产的<em>开启</em>，不是收缩表型的<em>关闭</em>；
两者在体内不完全同步，MYH11 的稀疏检出（dropout）也会压低这一侧的相关。</p></div>""".replace(
            "{m:.2f}", f"{g['MYH11']['partial_r']:+.2f}").replace(
            "{ml:+.2f}", f"{g['MYH11']['ci_lo']:+.2f}").replace(
            "{mh:+.2f}", f"{g['MYH11']['ci_hi']:+.2f}").replace(
            "{a:.2f}", f"{g['ACTA2']['partial_r']:+.2f}"))
        H.append(f"""<p><strong>所以 15.3 的判定保留、解释换掉。</strong>
AUC 口径“Matrix 不落对”是事实，不改；但原因不再是“物种不一致”，而是两件事叠加：
① 面板本身就是状态标记，在小鼠里 ModifiedSMC 与成纤维同高（差 {ms['median_mod_minus_fib']:+.2f}，p = {ms['p_mod_vs_fib']:.2f}）；
② 人的 SMC 注释是标记驱动的，去分化的 SMC 掉了 MYH11 就被分到别的区室里去，
于是“SMC vs 其余”这个二分类必然把它判错。
换成“这条程序在人体里是否随去分化移动”来问，答案是<strong>是</strong>，而且与小鼠同向。
这是把缺陷变成结论的唯一诚实路径：<em>不是解释掉 AUC，而是承认 AUC 不是这个面板的正确度量</em>。</p>""")
        H.append(_fig("fig20_stromal_continuum.png",
                      "图 20　间质连续体。(a) 小鼠同一只动物内 SMC → ModifiedSMC → 成纤维的 matrix_ligand 分数，"
                      "去分化 SMC 与成纤维齐平；(b) 人 SMC 内部沿去分化十分位的 Matrix 分数，"
                      "虚线是成纤维均值。"))

    # ================================================================ 16.3
    H.append("<h3>16.3 3v3 的 p 值上限：0.143 是“到顶”，不是“接近”</h3>")
    H.append("""<p>第 15 节里两个数字并排放着，读者很容易误读：
<em>“精确置换最小 p = 0.143”</em> 与 <em>“个体水平 EC 0.138 / SMC 0.185 / Mac 0.0038”</em>。
前者看起来像“差一点就显著”，后者像“只有巨噬有戏”。两句话都需要一个前置说明，
而这个说明是纯组合数学，不涉及生物学。</p>
<p>精确置换检验把 6 个样本分成 3+3，一共只有 C(6,3) = 20 种分法；
而任何一种分法与它的镜像（两组对调）给出绝对值相同的统计量，
所以最极端的那个分法也一定有一个并列者。按本文统一的 (计数+1)/(分法+1) 口径，
<strong>3v3 能达到的最小双侧 p 恒等于 3/21 = 0.143</strong>，与效应多大无关。
用 200 组随机向量实测，最小值最大值都是 0.1429，没有一次例外。</p>""")
    if len(floor):
        f2 = floor[floor.n_per_group_if_balanced.notna()].copy()
        f2 = f2[["n_per_group_if_balanced", "n_splits", "exact_floor_p", "can_reach_005"]]
        f2.columns = ["每组动物数", "分法数", "最小可达双侧 p", "能否达到 0.05"]
        f2["每组动物数"] = f2["每组动物数"].astype(int).astype(str)
        f2["分法数"] = f2["分法数"].astype(int).astype(str)
        f2["能否达到 0.05"] = f2["能否达到 0.05"].map({True: "能", False: "不能"})
        H.append(_tbl(f2, "{:.3f}"))
    H.append("""<p>读法：<strong>4v4 是最小的、精确置换能拒绝的设计（3/71 = 0.042）</strong>；
3v4 也不行（3/36 = 0.083）。这解释了两件此前只能含糊处理的事：
① 10.7 的组成检验里“精确置换 p 全部 ≥ 0.143”——那不是“接近显著”，是<em>已经到顶</em>，
在这套设计下“组成是否改变”这个问题<strong>结构上就无法被检验</strong>，只能靠分解去做；
② 反过来，个体水平那四个终点用的是 Welch t（不受组合数限制），所以 Mac 能到 0.0038。</p>
<div class="note"><p><strong>最锋利的一条对照：</strong>髓系 AgingSen 的效应量是 Hedges' g = {g_mac}，
95% CI [{g_lo}, {g_hi}]，Welch p = {p_mac}；但把同一个终点送进精确置换，p 仍然是 0.143——
<strong>恰好等于上限</strong>。一个 g ≈ 4.4 的效应在 3v3 的置换检验里也拿不到 0.05。
这条同时说明了“p = 0.143 不代表没效应”与“置换检验不适合这个 n”，比任何辩解都直接。</p></div>""".replace(
        "{g_mac}", f"{et.loc[et.metric=='mod_AgingSen','g'].iloc[0]:+.2f}" if len(et) else "n/a").replace(
        "{g_lo}", f"{et.loc[et.metric=='mod_AgingSen','g_lo'].iloc[0]:+.2f}" if len(et) else "n/a").replace(
        "{g_hi}", f"{et.loc[et.metric=='mod_AgingSen','g_hi'].iloc[0]:+.2f}" if len(et) else "n/a").replace(
        "{p_mac}", f"{et.loc[et.metric=='mod_AgingSen','p_welch'].iloc[0]:.4f}" if len(et) else "n/a"))

    if len(et):
        tt = et[["label", "g", "g_lo", "g_hi", "p_welch", "p_exact", "concordance",
                 "power_at_3v3", "n_required"]].copy()
        tt.columns = ["终点", "Hedges g", "CI 下", "CI 上", "Welch p", "置换 p",
                      "方向一致率", "3v3 功效", "达 80% 功效所需 n/组"]
        H.append("<p><strong>四个个体水平终点的完整效应量（每臂 3 只）：</strong></p>")
        H.append(_tbl(tt, "{:.3f}"))
        H.append("""<p>三个真实变化的终点，方向一致率都是 <strong>3/3</strong>
（三只处理动物全部越过对照中位数），但只有巨噬那个同时越过了功效门槛。
内皮与 SMC 的 |g| 约 1.2–1.5，而 3v3 的最小可检效应是 <strong>3.07</strong>，
所以它们的 p = 0.138 / 0.185 是<em>功效不足</em>，不是零效应——
这句话现在有 CI 撑着了：两个 CI 都包含 0，但下界分别是 −3.0 与 −2.7，
与“中等偏大效应”完全相容。</p>
<p><strong>顺带把 11.1 的补样门槛补齐到 3v3。</strong>11.1 用 6,000 次模拟报的是
2v2 = 5.58、2v3 = 3.88、3v3 = 3.07；本节的解析解（非中心 t）给 2v2 = 5.65、3v3 = 3.07，
两种算法在 3v3 上完全一致、2v2 差 0.07（模拟误差）。
补上去的是逐终点的那一列：<strong>内皮要 6 只/组、SMC 要 8 只/组、髓系 3 只/组就够</strong>——
因为三者的效应量分别是 1.46、1.24、4.40。这解释了为什么“再补一只”对不同终点价值不同：
对髓系早就够了，对内皮/SMC 只是把 MDE 从 3.88 压到 3.07，仍高于观测效应。</p>""")
        H.append(_fig("fig21_power_floor.png",
                      "图 21　3v3 的能力边界。(a) 精确置换的最小可达 p 随设计的变化，"
                      "3v3 恒为 0.143、4v4 才降到 0.05 以下；(b) 四个个体水平终点的 Hedges g 与 95% CI。"))

    # ================================================================ 16.4
    H.append("<h3>16.4 这一轮改了什么：一处撤回、三处改写口径、两条新增证据</h3>")
    H.append("""<table class="tbl"><thead><tr><th>位置</th><th>原表述</th><th>现在</th><th>依据</th></tr></thead><tbody>
<tr><td>15.3 R3</td><td>“内皮那条线与髓系/基质那条线分开，差了三到二十倍”</td>
<td>“内皮–基质正交（四种算法 |r| ≤ 0.06）；内皮–髓系仍 0.14–0.24，分叉不等于内皮与两者都无关”</td>
<td>供体内 8/8、偏相关、聚类 CI</td></tr>
<tr><td>15.3 R3</td><td>暗示“髓系–基质是三者最紧的一对”</td>
<td><strong>撤回</strong>：扣深度后 0.137 &lt; 0.169，配对检验 p = 0.098、CI 跨 0</td>
<td>16.1 配对 Wilcoxon</td></tr>
<tr><td>15.3 / C13</td><td>Matrix 在人体落错，写入局限</td>
<td>判定保留，原因改写为“面板是状态标记 + 人体注释的二分法不适用”，并给出人 SMC 内的连续体证据</td>
<td>16.2 小鼠 8/8 与 5,390 间质细胞</td></tr>
<tr><td>11.1 / 10.8</td><td>补样门槛只报到 2v2 / 2v3（5.58 / 3.88）</td>
<td>补到 3v3 = 3.07，并给出逐终点所需 n：内皮 6、SMC 8、髓系 3（只/组）</td>
<td>非中心 t 功效曲线</td></tr>
<tr><td>新增</td><td>—</td><td>精确置换的 3v3 上限 0.143，属组合数而非生物学；置换与 t 检验的适用边界写进方法学</td>
<td>20 种划分 + 200 次随机实测</td></tr>
<tr><td>新增</td><td>—</td><td>供体方差占比（5.6%–15.9%）作为“供体不主导”的正面证据写进 15.3</td>
<td>组间/总方差</td></tr>
</tbody></table>""")
    H.append("""<p class="files">新增脚本：<code>scripts/40_human_qc_pass.py</code>（逐细胞 QC 与标志基因表达）、
<code>scripts/41_human_branch_stats.py</code>（供体内 / 固定效应 / 偏相关 / 分层 + 聚类 bootstrap）、
<code>scripts/42_stromal_continuum.py</code>（小鼠—人间质连续体）、
<code>scripts/43_power_floor.py</code>（精确置换下限、Hedges g、MDE）、
<code>scripts/44_figures_strengthen.py</code>（图 19–21）。
新增结果：<code>results/human/r3_{within_donor,stratified,stratified_partial}.csv</code>、
<code>results/human/stromal_continuum_*.{csv,json}</code>、
<code>results/power/{exact_permutation_floor,endpoint_effect_table,n_required}.csv</code>。</p>""")
    return H
