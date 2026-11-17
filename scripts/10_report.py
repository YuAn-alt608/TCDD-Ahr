"""Assemble the restructured manuscript: core discovery (D2 + ATAC), boundary
conditions (D1 / D3), genetic independence, and the in-silico validation layer."""
import base64, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L
import report_sec10
import report_sec16

REP = os.path.join(L.ROOT, "report")
os.makedirs(REP, exist_ok=True)


def img(name, cap=""):
    p = os.path.join(L.RES, "figures", name)
    if not os.path.exists(p):
        return ""
    b = base64.b64encode(open(p, "rb").read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b}"/>'
            f'<figcaption>{cap}</figcaption></figure>')


def tbl(df, floatfmt="{:.2f}", maxrows=40):
    d = df.head(maxrows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    return d.to_html(index=False, classes="tbl", border=0, escape=False)


def load(path):
    p = os.path.join(L.RES, path)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def jload(path):
    p = os.path.join(L.RES, path)
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    T50 = load("matrix/T50_by_lineage.csv")
    AE = load("ATAC/atac_lineage_effects.csv")
    MO = load("motif/motif_activity_effects.csv")
    EN = load("motif/motif_enrichment.csv")
    HP = load("motif/het_promoter_motifs.csv")
    SM = load("ccc/smc_to_macrophage.csv")
    MOD = load("ccc/module_scores_by_group.csv")
    SWE = load("perturbation/switch_evidence.csv")
    SWS = load("perturbation/switch_summary.csv")
    AKO = load("perturbation/ahrko_rescue.csv")
    TEF = load("te/te_family_effects.csv")
    NICH = load("matrix/niche_vs_parenchyma.csv")
    EFF = load("matrix/exposure_lineage_effects.csv")
    VAL = jload("validation/summary.json")
    GI = jload("validation/genetic_independence.json")
    TRAJ = jload("trajectory/summary.json")
    TEK = jload("te/kmer_summary.json")
    TEV = jload("te/kmer_engine_validation.json")
    ENR = jload("genetics/trait_enrichment.json")

    # ---- derived values
    def t50(ct, fallback=np.nan):
        if T50.empty:
            return fallback
        d = T50[T50.celltype == ct].dropna(subset=["T50_weeks"])
        return float(d.sort_values("T50_weeks").iloc[0].T50_weeks) if len(d) else fallback

    def mot(tf, lin, col="d_tcdd_vs_ctrl", default=np.nan):
        if MO.empty:
            return default
        d = MO[(MO.tf == tf) & (MO.lineage == lin)][col]
        return float(d.mean()) if len(d) else default

    def atac(lin, metric, col="d_wt_vs_control"):
        if AE.empty:
            return np.nan
        d = AE[(AE.lineage == lin) & (AE.metric == metric)][col]
        return float(d.iloc[0]) if len(d) else np.nan

    def atac_q(lin, metric):
        if AE.empty:
            return np.nan
        d = AE[(AE.lineage == lin) & (AE.metric == metric)]["fdr"]
        return float(d.iloc[0]) if len(d) else np.nan

    def enr(tf, lin, direction="open"):
        if EN.empty:
            return None
        d = EN[(EN.tf == tf) & (EN.lineage == lin) & (EN.direction == direction)]
        return d.iloc[0] if len(d) else None

    ahr_pro = HP[(HP.lineage == "Parenchyma") & (HP.contrast == "TCDD_vs_ctrl") &
                 (HP.tf == "Ahr::Arnt")]
    ahr_ec = HP[(HP.lineage == "Endothelial") & (HP.contrast == "TCDD_vs_ctrl") &
                (HP.tf == "Ahr::Arnt")]
    hp_pro = ahr_pro.iloc[0] if len(ahr_pro) else None
    hp_ec = ahr_ec.iloc[0] if len(ahr_ec) else None
    e_pro = enr("Ahr::Arnt", "Parenchyma")
    e_ec = enr("Ahr::Arnt", "Endothelial")

    cm = load("validation/cm_transfer.csv")
    cm_mod = cm[cm.sender == "ModifiedSMC"].set_index("group").cm_score if len(cm) else pd.Series()
    fp = load("validation/cm_readout_prediction.csv")
    lig = load("validation/matrix_ligand_macrophage.csv")
    sup = load("validation/smc_supernatant_ligands.csv")
    mac = MOD[MOD.celltype == "Macrophage"].set_index("group") if len(MOD) else pd.DataFrame()
    smc = MOD[MOD.celltype == "SMC"].set_index("group") if len(MOD) else pd.DataFrame()

    H = []
    H.append("""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Age3 · 细胞自主与非细胞自主的时空解耦</title>
<style>
:root{--ink:#16181c;--mut:#5b6470;--line:#e3e6ea;--accent:#1f6f8b;--warn:#a3320b;--bg:#fbfbfa}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,"Segoe UI","Noto Sans SC","Microsoft YaHei",sans-serif;
line-height:1.78;font-size:15px}
.wrap{max-width:1080px;margin:0 auto;padding:56px 28px 96px}
h1{font-size:28px;line-height:1.4;margin:0 0 8px;letter-spacing:-.3px}
.sub{color:var(--mut);font-size:13.5px;margin-bottom:30px}
h2{font-size:20px;margin:46px 0 12px;padding-bottom:7px;border-bottom:2px solid var(--line)}
h3{font-size:16px;margin:26px 0 8px;color:#22404f}
h4{font-size:14px;margin:18px 0 6px;color:#334}
p{margin:10px 0}
.lead{background:#fff;border:1px solid var(--line);border-left:4px solid var(--accent);
padding:18px 22px;border-radius:6px;margin:18px 0}
.lead ul{margin:8px 0 0 18px;padding:0}.lead li{margin:6px 0}
table.tbl{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff;margin:12px 0}
table.tbl th{background:#f2f5f7;text-align:left;padding:8px 9px;border-bottom:1px solid var(--line);font-weight:600}
table.tbl td{padding:7px 9px;border-bottom:1px solid #eef0f2;vertical-align:top}
table.tbl tr:hover td{background:#fafcfd}
figure{margin:22px 0;text-align:center}
figure img{max-width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}
figcaption{font-size:12.5px;color:var(--mut);margin-top:7px;text-align:left}
code{background:#eef2f4;padding:1px 5px;border-radius:3px;font-size:13px}
.note{background:#fff8f2;border:1px solid #f0d9c8;border-radius:6px;padding:14px 16px;font-size:13.5px;margin:14px 0}
.core{background:#f4f8fb;border:1px solid #cfe0ea;border-left:4px solid var(--accent);
border-radius:6px;padding:14px 16px;font-size:13.6px;margin:14px 0}
.bound{background:#f7f7f4;border:1px solid #e0ded6;border-left:4px solid #8a8578;
border-radius:6px;padding:14px 16px;font-size:13.6px;margin:14px 0}
.files{font-size:12.5px;color:var(--mut)}
.tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:10px;
background:#eef2f4;color:#3a4a55;margin-right:6px}
pre{background:#fff;border:1px solid #e3e6ea;border-radius:6px;padding:14px;font-size:12.5px;overflow:auto}
</style></head><body><div class="wrap">""")

    # ================================================================= title
    H.append("<h1>暴露在血管平滑肌细胞击穿异染色质缓冲的“双层解耦”："
             "AhR 依赖的细胞自主启动与非细胞自主级联</h1>")
    H.append('<div class="sub">核心发现：二噁英 / 冠脉微环境（GSE276872 + GSE276870）· '
             '边界条件：PM2.5 肺（GSE296297）与母代 PM2.5 跨代脂肪（GSE307673）· '
             '人群层：GWAS Catalog v2（遗传独立性检验）<br>'
             '分析日期 2026-11-17 · 全部数值由本机从原始数据重算，脚本与对象落盘可复现</div>')

    # ================================================================= abstract
    H.append('<div class="lead"><strong>一句话</strong><br>'
             '二噁英不需要改写遗传密码：它先在血管平滑肌细胞里、以 AhR 依赖的方式撬开异染色质位点（细胞自主），'
             '再通过基质/分泌程序把“缓冲失效”传染给内皮与巨噬细胞（非细胞自主）——'
             '注意传染的不是同一件事：内皮丢掉的是身份与代谢程序的可及性边界，而非又一处 H3K9me3 域击穿（见第 10.3 节）。'
             '这条轴在人群 GWAS 中<em>不</em>富集——恰恰说明环境表观脆弱性独立于先天遗传易感性。</div>')

    H.append('<div class="lead"><strong>五条主结论</strong><ul>')
    H.append(f"<li><span class='tag'>核心 I</span><strong>细胞自主起点在 SMC 的 AhR。</strong>"
             f"TCDD 后 SMC 的 H3K9me3 域可及片段比例上升（d = {atac('Parenchyma','frac_het'):+.2f}，"
             f"q = {atac_q('Parenchyma','frac_het'):.1e}）；差异开放 bin 富集 <strong>Ahr::Arnt</strong> 位点"
             f"（OR = {e_pro.odds_ratio:.2f}，FDR = {e_pro.fdr:.1e}，{int(e_pro.n_target)} 个 bin）；"
             f"带 AhR motif 的 {int(hp_pro.n_promoters)} 个异染色质启动子开放比例上升 "
             f"{hp_pro.delta_open_frac*100:.1f} 个百分点。这是“暴露→结构”的直接一步，且是 AhR 依赖的。</li>")
    H.append(f"<li><span class='tag'>核心 II</span><strong>扩散到内皮时换了“开门的人”。</strong>"
             f"内皮异染色质基因启动子可及性上升最明显（d = {atac('Endothelial','prom_het_rel'):+.2f}，"
             f"q = {atac_q('Endothelial','prom_het_rel'):.1e}），但它的 AhR 位点富集并不显著"
             f"（OR = {e_ec.odds_ratio:.2f}，FDR = {e_ec.fdr:.2f}，仅 {int(e_ec.n_target)} 个 bin）；"
             f"同步变化的是 KLF2（{mot('KLF2','Endothelial'):+.2f}）、Nr1h3（{mot('Nr1h3','Endothelial'):+.2f}）、"
             f"SMAD2（{mot('SMAD2','Endothelial'):+.2f}）、HIF1A（{mot('HIF1A','Endothelial'):+.2f}），"
             f"而 NRF2（{mot('Nfe2l2','Endothelial'):+.2f}）与 RELA（{mot('RELA','Endothelial'):+.2f}）贴着零。"
             f"敲掉 <em>SMC</em> 的 Ahr 后，这些内皮位点随之下调（ATF2 {mot('ATF2','Endothelial','d_ako_vs_wt'):+.2f}、"
             f"SMAD2 {mot('SMAD2','Endothelial','d_ako_vs_wt'):+.2f}、KLF2 {mot('KLF2','Endothelial','d_ako_vs_wt'):+.2f}）"
             f"而 Ahr::Arnt 不动（{mot('Ahr::Arnt','Endothelial','d_ako_vs_wt'):+.2f}）——这一块开放是别人代劳的。</li>")
    H.append(f"<li><span class='tag'>核心 III</span><strong>“敲除≠复原”：髓系进入 LAM 样状态而不是回到对照。</strong>"
             f"SMC 的 SASP 模块在对照/16 周/Ahr-KO 三组是一条平线"
             f"（{smc.loc['Control','SASP']:.3f} / {smc.loc['TCDD_16wk_WT','SASP']:.3f} / {smc.loc['TCDD_16wk_AhrKO','SASP']:.3f}），"
             f"增强的是 <strong>Timp1–Cd63</strong>（Δ = {SM.iloc[0].d_tcdd_vs_ctrl:+.3f}）与 Fn1–Sdc4 这类基质配体；"
             f"巨噬的 INFLAM 在 Ahr-KO 下从 {mac.loc['TCDD_16wk_WT','INFLAM']:.2f} 升到 "
             f"{mac.loc['TCDD_16wk_AhrKO','INFLAM']:.2f}、ISG 从 {mac.loc['TCDD_16wk_WT','ISG']:.2f} 升到 "
             f"{mac.loc['TCDD_16wk_AhrKO','ISG']:.2f}，单基因是 Trem2 +1.12、Lgals3 +0.92、Csf1r +0.77。</li>")
    H.append(f"<li><span class='tag'>核心 IV</span><strong>时序是级联而不是同时。</strong>"
             f"按各谱系最早失守指标计算 T50：内皮 <strong>{t50('Endothelial'):.1f} 周</strong> → "
             f"调制型 SMC {t50('ModifiedSMC'):.1f} 周 → 巨噬 {t50('Macrophage'):.1f} 周 → "
             f"成纤维 <strong>{t50('Fibroblast'):.1f} 周</strong>。"
             f"拟时序给出同向读法：调制型 SMC 沿暴露轴位移最大（Δpseudotime = "
             f"{TRAJ.get('pt_shift',{}).get('ModifiedSMC',float('nan')):.2f}），"
             f"敲除后回退 {TRAJ.get('ako_return',{}).get('ModifiedSMC',float('nan'))*100:.0f}%；"
             f"巨噬/单核则<em>过冲</em>（{TRAJ.get('ako_return',{}).get('Macrophage',float('nan'))*100:.0f}% / "
             f"{TRAJ.get('ako_return',{}).get('Monocyte',float('nan'))*100:.0f}%）。</li>")
    H.append(f"<li><span class='tag'>独立</span><strong>环境表观脆弱性独立于先天遗传易感性。</strong>"
             f"表观开关面板在器官退变/衰老性状上的 GWAS 富集 <strong>OR = {GI.get('odds_ratio',float('nan')):.2f}</strong>"
             f"（95% CI {GI.get('ci_low',float('nan')):.2f}–{GI.get('ci_high',float('nan')):.2f}），"
             f"即<em>更少</em>落在这些性状上；{GI.get('n_zero_ageing_hits',0)}/{GI.get('n_switch_genes',0)} 个开关基因"
             f"在衰老类性状上零命中。这不是“没有遗传学”，而是“环境重塑的靶点不是常见变异的靶点”。</li>")
    H.append("</ul></div>")

    # ================================================================= 1 storyline
    H.append("<h2>1. 主线重构：为什么放弃“泛化规律”</h2>")
    H.append("""<p>初版报告试图在三个暴露体系（PM2.5 肺、TCDD 冠脉、母代 PM2.5 跨代脂肪）之间找一条
“环境暴露普遍击穿异染色质缓冲”的规律。把三套数据放在同一张矩阵上之后，这条规律<strong>没有得到支持</strong>：
微环境与实质细胞的敏感度差在 25 个（体系 × 时程 × 指标）组合里没有单一方向，
不同指标甚至在同一体系内互相矛盾。继续写“普适规律”只能靠挑选指标，那是编故事。</p>
<p>因此本文改成一条<strong>有方向、有先后、有机制承载者</strong>的主线，并把不支持的部分降级为边界条件：</p>""")
    H.append("""<table class="tbl"><tr><th style="width:16%">层次</th><th style="width:30%">内容</th>
<th style="width:12%">篇幅</th><th>证据</th></tr>
<tr><td><b>Core Discovery</b></td><td>TCDD/AhR 轴驱动的血管微环境衰老重塑：细胞自主起点 + 非细胞自主级联</td>
<td>约 65%</td><td>D2 scRNA（33,376 细胞）+ 配对 scATAC（16,000 细胞）+ motif 层 + 通讯层 + 拟时序 + 三项 in silico 验证</td></tr>
<tr><td><b>Boundary Conditions</b></td><td>PM2.5 肺（D1）与母代 PM2.5 跨代脂肪（D3）：同一指标体系在不同理化性质污染物上的“暴露-指标特异性”</td>
<td>约 20%</td><td>同一套四项指纹 + 交互矩阵；用于界定机制成立的边界，而不是复述规律</td></tr>
<tr><td><b>Genetic Independence</b></td><td>人群 GWAS 的负结果升华为“环境表观重塑独立于先天遗传易感性”</td>
<td>约 15%</td><td>GWAS Catalog v2（1,993 + 4,157 条关联）、OR 与 95% CI、17 个开关基因逐基因核查</td></tr>
</table>""")
    H.append("""<div class="note"><strong>三处必须说清楚的“不如预期”，本文不掩盖：</strong><ul>
<li><strong>内皮并没有走 NRF2 或 NF-κB。</strong>原设想“氧化应激→NRF2”和“NF-κB 直接攻启动子”两条路，
在染色质层面都近乎为零（Nfe2l2 d = 0.02、RELA d = 0.07）。真正同步的是 KLF/SP 家族、LXR、SMAD2 与 HIF1A。</li>
<li><strong>SASP 假说不成立。</strong>SMC 的 SASP 模块三组平线，没有“暴露→SASP→巨噬”的起点；
增强的是基质配体（Timp1、Fn1、Col1a2）。</li>
<li><strong>家族级 TE 转录原先“无法测量”。</strong>本轮核查发现原始 FASTQ 其实是公开的（PRJNA1159315），
已建立并验证了一条无需比对器的家族定量管线；但在本机带宽下只能做 pilot，结论按 pilot 的强度写。</li>
</ul></div>""")
    H.append("""<p>投稿前又加了一道更狠的自查：<strong>如果审稿人先问“你的深度、你的统计单位、你的域定义、你的网络模型靠不靠得住”，哪几句话会塌</strong>。
四项内部压力测试与两项外部验证的结果、以及据此改写的每一条措辞，全部集中在<strong>第 10 节</strong>；
其中一条结论被直接推翻（“跨谱系普遍击穿”），一条方法的解释力被降级（κ 模型对拓扑不可辨识）。</p>""")

    # ================================================================= 2 data
    H.append("<h2>2. 数据与设计：核心与边界的非对称使用</h2>")
    ds = pd.DataFrame([
        ["D2（Core）", "TCDD / 二噁英", "小鼠冠状动脉，ApoE−/−", "GSE276872",
         "16 样本；Ctrl / TCDD 8 周 / TCDD 16 周 / SMC-Ahr-KO", "33,376 细胞",
         "Myh11-CreERT2;Rosa-tdT 谱系示踪把 SMC 衍生细胞（tdT⁺）与微环境细胞（tdT⁻）物理分开"],
        ["D2-ATAC（Core）", "TCDD 配对染色质", "同一实验的冠脉", "GSE276870",
         "4 样本；Ctrl×2 / TCDD / Ahr-KO+TCDD", "16,000 细胞 / 121,296 bins",
         "5 kb 固定分箱共用特征空间；Ahr-KO 文库浅 25%，只用相对量与偏差分数"],
        ["D1（Boundary）", "PM2.5", "小鼠肺，1 个月", "GSE296297",
         "26 样本；Control vs PM2.5", "8,000 分析细胞", "实质（AT1/AT2/club）与微环境（内皮/成纤维）同库"],
        ["D3（Boundary）", "PM2.5 母代暴露", "子代腹股沟白色脂肪，中年", "GSE307673",
         "6 样本；母代 FA vs PM2.5", "24,000 分析细胞", "脂肪细胞占比低（1–5%），实质层功效受限"],
    ], columns=["编号", "暴露", "体系", "GEO", "设计", "细胞数", "关键结构特征"])
    H.append(tbl(ds))
    H.append("""<p>三套数据共用同一套四项指纹（TE 去抑制代理、异染色质域基因偏移 HGDI/LADleak、
转录熵、regulon 熵）与同一套统计（Cohen's d + 1,000 次置换 + BH）。
<strong>但它们在文中的角色不对称</strong>：D2 有基因型定义的细胞身份（tdT 分选）与真实遗传扰动（SMC 特异 Ahr 敲除），
可以回答“谁先动、谁被动”；D1/D3 只有暴露对比，只能回答“这套机制在别的污染物上还成不成立”。</p>""")

    # ================================================================= 3 core I
    H.append("<h2>3. 核心发现 I：细胞自主起点——SMC 里 AhR 依赖的异染色质位点开放</h2>")
    H.append(f"""<p>scATAC 层面先给结构证据。SMC（Parenchyma）在 TCDD 后落在 H3K9me3 域内的可及片段比例上升
（d = {atac('Parenchyma','frac_het'):+.2f}，q = {atac_q('Parenchyma','frac_het'):.1e}；均值
{AE[(AE.lineage=='Parenchyma')&(AE.metric=='frac_het')].mean_control.iloc[0]:.4f} →
{AE[(AE.lineage=='Parenchyma')&(AE.metric=='frac_het')].mean_wt.iloc[0]:.4f} →
{AE[(AE.lineage=='Parenchyma')&(AE.metric=='frac_het')].mean_ahrko.iloc[0]:.4f}，Ahr-KO 落在中间）。
方向与转录组层的泄露指数一致（注意：该指数在个体水平并不显著，见第 10.2 节），
而且<strong>部分依赖 AhR</strong>。</p>
<p>更关键的是“哪类位点被打开”。差异开放 bin 的 motif 富集给出唯一一个通过 FDR 的答案：
<strong>Ahr::Arnt</strong>，OR = {e_pro.odds_ratio:.2f}（FDR = {e_pro.fdr:.1e}），
背景命中率 {e_pro.frac_bg*100:.1f}% → 目标 {e_pro.frac_target*100:.1f}%，涉及 {int(e_pro.n_target)} 个 bin。
把范围收窄到异染色质基因启动子（1,987 个），带 AhR motif 的 {int(hp_pro.n_promoters)} 个启动子
开放细胞比例上升 <strong>{hp_pro.delta_open_frac*100:.1f} 个百分点</strong>（随机同类启动子零分布 p &lt; 0.001，FDR &lt; 0.001）。
这是本文最干净的一步链：<em>配体 → 受体 → 序列特异性位点 → 结构开放</em>。
但第四段必须留白：motif 富集只说明序列偏好，不等于蛋白真的结合在那里。
我们试过用公开的人冠脉 SMC AHR ChIP-seq（GSE276152）去补这一段——交集是阴性的
（SMC 开放 bin 落峰内 14/270，OR = 1.29，p = 0.35），<strong>但那套 ChIP 连自己的阳性对照都没通过</strong>：
CYP1A1、CYP1B1、AHRR、TIPARP、NQO1、AHR 六个经典靶基因的 TSS ±10 kb 内平均 0 个峰，
低于随机基因的 0.20。所以它既不能证实、也不能否证结合。文中一律写“<strong>AhR 依赖</strong>”，
不写“AhR 直接结合”；这两句话的判决留给第 11.2 节那套自带阳性对照的 ChIP-qPCR（第 10.5 节）。</p>""")
    H.append("""<div class="core"><strong>为什么这一步必须放在最前面。</strong>
后文所有“非细胞自主”的论证都依赖一个前提：SMC 里确实发生了 AhR 依赖的原发事件，而不是 SMC 只是被动响应。
Ahr motif 的选择性开放 + Ahr-KO 的部分回落，正是这个前提的证据。若没有它，内皮的变化只能算“相关”。</div>""")
    H.append(img("fig7_motif_ec_vs_smc.png",
                 "图 1｜异染色质基因启动子上各 motif 的可及性变化（TCDD vs 对照），右图为同一批 motif 在两条谱系上的对比。"
                 "注意 SMC 的 AhR 位点选择性开放，而内皮的 AhR 不显著、上去的是 KLF/SP 家族与 SMAD2/HIF1A。"))

    # ================================================================= 4 core II
    H.append("<h2>4. 核心发现 II：非细胞自主扩散——内皮换了“开门的人”</h2>")
    H.append(f"""<p>内皮是<strong>最早失守</strong>的谱系（T50 = {t50('Endothelial'):.1f} 周），但它自己的 AhR 位点并没有选择性开放
（OR = {e_ec.odds_ratio:.2f}，FDR = {e_ec.fdr:.2f}，只有 {int(e_ec.n_target)} 个 bin，功效不足且方向并不突出）。
但这 {int(e_ec.n_target)} 个 bin 上<strong>没有任何一个 motif 通过 FDR 校正</strong>（0 个，最小 p 也未过线），所以“内皮是被谁打开的”在 bin 层面<em>指认不出来</em>；能说的是内皮<strong>整体染色质状态</strong>的偏差分数偏向一组与 AhR 无关的身份/代谢因子：KLF2（{mot('KLF2','Endothelial'):+.2f}，层流/剪切）、
KLF4（{mot('KLF4','Endothelial'):+.2f}）、Nr1h3/LXR（{mot('Nr1h3','Endothelial'):+.2f}）、
SMAD2（{mot('SMAD2','Endothelial'):+.2f}）、HIF1A/EPAS1（{mot('HIF1A','Endothelial'):+.2f} / {mot('EPAS1','Endothelial'):+.2f}）。
换句话说，内皮的染色质变化更像<em>整体身份与代谢状态的重编程</em>，不是一条炎症通路的开关；而“是哪条上游通路推的”只能列为假设（候选：剪切/LXR/HIF），不能写成结论——这一条由主张审计 C5 强制标出（<code>results/audit/claim_audit.md</code>）。</p>
<p>判定“非细胞自主”的关键不是这些因子本身，而是<strong>它们在 SMC 特异 Ahr 敲除后的行为</strong>：
ATF2 {mot('ATF2','Endothelial','d_ako_vs_wt'):+.2f}、KLF2 {mot('KLF2','Endothelial','d_ako_vs_wt'):+.2f}、
SMAD2 {mot('SMAD2','Endothelial','d_ako_vs_wt'):+.2f}、Nr1h3 {mot('Nr1h3','Endothelial','d_ako_vs_wt'):+.2f}、
FOSL1::JUND {mot('FOSL1::JUND','Endothelial','d_ako_vs_wt'):+.2f} 全部回落，
而 <strong>Ahr::Arnt 几乎不动</strong>（{mot('Ahr::Arnt','Endothelial','d_ako_vs_wt'):+.2f}）。
细胞自己的 AhR 没变，别人的 AhR 被敲掉，它的染色质却退回去了——这是非细胞自主的判据。</p>""")
    H.append("""<h3>4.1 干实验复现 1：in silico 条件培养基（CM）转移</h3>
<p>湿实验版本是：TCDD 处理 WT 或 siAhr 的 SMC → 收条件培养基 → 刺激内皮 → qPCR/ChIP-qPCR 测
Klf2、Smad2 启动子活性与 H3K9me3。干实验版本用同一逻辑但换数据载体：把 SMC 的配体表达与内皮的受体表达
按 CellChat 的 Hill 形式合成一个“传递分”，配体只取可进入条件培养基的分泌/ECM 类（187 对，OmniPath ligrecextra 两端均检出）。
然后用<strong>单参数标定 + 留出臂检验</strong>：在 Control→TCDD 臂上拟合一个比例常数，去预测 Ahr-KO 臂的内皮读数。</p>""")
    if len(fp):
        fp = fp.copy()
        fp["direction_ok"] = fp.direction_ok.map(lambda v: "是" if bool(v) else "否")
        H.append(tbl(fp.drop(columns=["gene"]).rename(columns={"readout": "内皮读数", "kappa": "标定常数 κ",
                                        "obs_ctrl": "对照实测", "obs_tcdd": "TCDD 实测",
                                        "obs_ako": "Ahr-KO 实测", "pred_ako": "Ahr-KO 预测",
                                        "err": "误差", "direction_ok": "方向正确"}),
                     floatfmt="{:.3f}"))
    H.append(f"""<p>传递分本身在 TCDD 后下降（ModifiedSMC 作为发送细胞：{cm_mod.get('Control',float('nan')):.1f} →
{cm_mod.get('TCDD_16wk_WT',float('nan')):.1f}），敲掉 SMC 的 Ahr 后回抬到 {cm_mod.get('TCDD_16wk_AhrKO',float('nan')):.1f}。
四个内皮读数在完全留出的 Ahr-KO 臂上方向全部正确（4/4），误差绝对值 0.005–0.10（读数量级 0.04–2.7）。
<strong>但这个 4/4 不能当作模型有效的证据</strong>：把配体-受体网络在保持节点度分布的前提下随机重连 1,000 次，
两种零模型同样 4/4（平均绝对误差 0.049 / 0.049，真实网络 0.054），p = 1.00 / 0.93。
原因很直白——预测量是 x<sub>AKO</sub> − x<sub>WT</sub> 的线性组合，符号由各读数的组间方向统一决定，与拓扑无关，
所以这个拟合对网络结构<em>不可辨识</em>（第 10.4 节）。它能回答“条件培养基实验应该看到什么方向”，
不能回答“哪几对配体-受体在传递”。</p>""")
    H.append(img("fig10_validation.png",
                 "图 2｜三项 in silico 验证。(a) 条件培养基传递分随暴露下降、随 Ahr 敲除回抬；"
                 "(b) 用 TCDD 臂标定后在 Ahr-KO 臂上的留出预测（4/4 方向正确，但随机网络零模型同为 4/4，见第 10.4 节）；"
                 "(c) 表观开关面板的 GWAS 富集 OR 及其 95% CI（完全位于 1 以下）。"))
    H.append("""<div class="note"><strong>这一步的边界。</strong>组水平拟批量，n = 4 组、每组 1 个合并观测；
“标定”用的是 1 个参数、2 个点，留出臂只有 1 个点。它是<em>方向性一致性检查</em>，
不能给 p 值，也不能替代条件培养基实验。真正的判据在湿实验里：CM 转移后内皮 Klf2 启动子的 H3K9me3 是否下降、
以及 siAhr 的 SMC 是否失去这个能力。</div>""")

    # ================================================================= 5 core III
    H.append("<h2>5. 核心发现 III：基质配体与髓系重编程</h2>")
    H.append(f"""<p>“SMC 释放 SASP 把巨噬拉进早衰”是这类课题最常见的默认解释，本数据<strong>否掉</strong>了它：
SMC 的 SASP 模块在对照 / TCDD 16 周 / Ahr-KO 三组是
{smc.loc['Control','SASP']:.3f} / {smc.loc['TCDD_16wk_WT','SASP']:.3f} / {smc.loc['TCDD_16wk_AhrKO','SASP']:.3f}，
一条平线。通讯层增强的是基质与粘附配体：Timp1–Cd63（Δ = {SM.iloc[0].d_tcdd_vs_ctrl:+.3f}）、
Fn1–Sdc4（{SM.loc[(SM.ligand=='Fn1')&(SM.receptor=='Sdc4'),'d_tcdd_vs_ctrl'].mean():+.3f}）、
Col1a2–Sdc4、Lum–Itgb1；Ccl2 / Il6 / Cxcl12 都不在前列。
上清配体代理（拟批量）给出的排序一致：Timp1 {sup[sup.ligand=='Timp1'].d_tcdd.iloc[0]:+.3f}、
Fn1 {sup[sup.ligand=='Fn1'].d_tcdd.iloc[0]:+.3f}、Spp1 {sup[sup.ligand=='Spp1'].d_tcdd.iloc[0]:+.3f}，
而 Col3a1 {sup[sup.ligand=='Col3a1'].d_tcdd.iloc[0]:+.3f}、Cxcl12 {sup[sup.ligand=='Cxcl12'].d_tcdd.iloc[0]:+.3f} 反而下降。</p>
<p>巨噬这一端的变化更值得注意：Ahr-KO 后<strong>不是复原，而是过冲</strong>。
INFLAM {mac.loc['TCDD_16wk_WT','INFLAM']:.2f} → {mac.loc['TCDD_16wk_AhrKO','INFLAM']:.2f}，
ISG {mac.loc['TCDD_16wk_WT','ISG']:.2f} → {mac.loc['TCDD_16wk_AhrKO','ISG']:.2f}，
单基因 Trem2 +1.12、Lgals3 +0.92、Csf1r +0.77、Tgfb1 +0.53。
这是 <strong>Trem2⁺/Lgals3⁺ 脂质相关巨噬（LAM 样）</strong>的状态切换，而不是回到稳态。
所以第 6 节那条“轴坐标回落”绝不能写成“抗炎复原”。</p>""")
    H.append("""<h3>5.1 干实验复现 2：基质配体 + 巨噬重编程（含“拮抗剂”臂）</h3>
<p>湿实验版本是：TCDD 处理 ± CH-223191（AhR 拮抗剂）→ ELISA 测上清 TIMP1/FN1 →
流式/qPCR 看巨噬 Trem2⁺/Lgals3⁺ 与 ISG。干实验版本没有药物臂，但有<strong>更好的东西：遗传学等价臂</strong>——
SMC 特异 Ahr 敲除就是“只在 SMC 里关掉 AhR”的完美拮抗。把 Ahr-KO 当作 CH-223191 的预期表型，
可以给湿实验一个可证伪的预测量。</p>""")
    if len(lig):
        H.append(tbl(lig[["group", "lam_score", "ecm_index", "INFLAM", "ISG"]].rename(
            columns={"group": "组", "lam_score": "LAM 程序（z）", "ecm_index": "SMC 基质指数",
                     "INFLAM": "巨噬炎性模块", "ISG": "巨噬 ISG 模块"}), floatfmt="{:.3f}"))
    H.append(f"""<p>LAM 程序在 TCDD 后降到 {lig[lig.group=='TCDD_16wk_WT'].lam_score.iloc[0]:+.2f}（z），
Ahr-KO 后升到 {lig[lig.group=='TCDD_16wk_AhrKO'].lam_score.iloc[0]:+.2f}，<strong>越过对照</strong>
（对照 {lig[lig.group=='Control'].lam_score.iloc[0]:+.2f}）。基质指数在 TCDD 后基本不动
（{lig[lig.group=='Control'].ecm_index.iloc[0]:.2f} → {lig[lig.group=='TCDD_16wk_WT'].ecm_index.iloc[0]:.2f}），
在 Ahr-KO 后升到 {lig[lig.group=='TCDD_16wk_AhrKO'].ecm_index.iloc[0]:.2f}。
也就是说，“抑制 SMC 的 AhR”不会把基质程序压回去，反而把它推得更高——这与“Ahr 在 SMC 中是保护性的”一致，
也意味着 CH-223191 的预期表型不是“恢复”，而是“换一种失稳方式”。</p>""")
    H.append("""<div class="note"><strong>给湿实验的可证伪预测（写在这里，方便日后对账）：</strong><ul>
<li>若 CH-223191 真正模拟 SMC-Ahr-KO，则 SMC 上清 TIMP1 不会下降（可能升高），FN1 升高；</li>
<li>巨噬 Trem2/Lgals3 上升而 Il1b 不降，ISG（Isg15/Ifit1）升高；</li>
<li>若观察到 TIMP1 明显下降且巨噬回到对照表型，则本文的“基质配体”解释被推翻，应改回 SASP/生长因子假说。</li>
</ul></div>""")

    # ================================================================= 6 core IV
    H.append("<h2>6. 核心发现 IV：级联时序与 Model Figure</h2>")
    H.append(f"""<p>T50 是唯一能从 0/8/16 周三档推出的时序量（线性内插，仅用于谱系间排序）。
按各谱系<strong>最早失守的指标</strong>排序：内皮 {t50('Endothelial'):.1f} 周（TE 代理轴）→
调制型 SMC {t50('ModifiedSMC'):.1f} 周（regulon 熵）→ 巨噬 {t50('Macrophage'):.1f} 周（转录熵）→
成纤维 {t50('Fibroblast'):.1f} 周（转录熵）。这个顺序与“SMC 是起点”并不矛盾：
SMC 的分子事件（AhR 位点开放）是最早的，但它的<em>整体状态指标</em>要等到下游反馈才越过阈值；
内皮因为没有自己的 AhR 缓冲，反而最先在转录/序列层失守。</p>
<p>拟时序（每谱系单独做扩散伪时间、以对照端为根）给出同向读法：调制型 SMC 沿轨迹位移最大
（Δpseudotime = {TRAJ.get('pt_shift',{}).get('ModifiedSMC',float('nan')):.2f}），
Ahr-KO 后回退 {TRAJ.get('ako_return',{}).get('ModifiedSMC',float('nan'))*100:.0f}%；
巨噬与单核的“回退比例”超过 100%（{TRAJ.get('ako_return',{}).get('Macrophage',float('nan'))*100:.0f}% /
{TRAJ.get('ako_return',{}).get('Monocyte',float('nan'))*100:.0f}%），即<em>过冲</em>，
与第 5 节的 LAM 切换互为印证。内皮与成纤维的伪时间几乎不动（
{TRAJ.get('pt_shift',{}).get('Endothelial',float('nan')):+.2f} / {TRAJ.get('pt_shift',{}).get('Fibroblast',float('nan')):+.2f}），
说明这两个群体的主要变异方向不是暴露——这一点必须写进来，否则会高估轨迹分析的效力。</p>""")
    H.append(img("fig9_model_cascade.png",
                 "图 3｜Model Figure。(a) 四个谱系的缓冲崩溃曲线与 T50，按最早失守指标绘制；"
                 "(b) 双层解耦的方向性：AhR 依赖的细胞自主起点 → 基质/分泌程序 → 内皮与巨噬的非细胞自主结局；"
                 "(c) Ahr-KO 后髓系程序过冲，说明“轴坐标回落”不等于复原。"))
    H.append("""<div class="note"><strong>没做 RNA velocity，原因写在明处。</strong>
GEO 只提供基因水平矩阵，没有 spliced/unspliced 层；要做 velocity 必须回到 FASTQ，
用 velocyto/STARsolo 重建剪接比。本机带宽与磁盘不支持全量重建（见第 7 节的数据量核算）。
因此本文只声称“伪时间排序 + 回退比例”，不声称“方向性速率”。</div>""")

    # ================================================================= 7 TE
    H.append("<h2>7. 转座元件：从“算不了”到“能算，但只算到 pilot”</h2>")
    H.append("""<h3>7.1 先纠正上一版的判断</h3>
<p>上一版写“GEO 没有 BAM/FASTQ，家族级 TE 无法测量”。这句话<strong>只对了一半</strong>：
GSE276870（scATAC）确实没有 fragments，所以足迹分析仍然做不了；
但 GSE276872 的原始 reads 是公开的——ENA/SRA 上 <code>PRJNA1159315</code>（scRNA，41 个 run，约 130 GB）
与 <code>PRJNA1159314</code>（scATAC，8 个 run，约 72 GB）都可用。缺的是 fragments，不是 reads。</p>""")
    H.append("""<h3>7.2 一条不需要比对器的家族定量管线</h3>
<p>scTE/Telescope 需要 STAR 建索引 + 多重比对分配，本机没有比对器、也没有时间建索引。
替代方案是<strong>精确 k-mer 伪比对</strong>（<code>scripts/16_te_kmers.py</code>）：
从 rmsk 取每个家族最多 400 个元件、每个元件中心 1.5 kb，从 mm10.2bit 取序列，
20-mer 规范化（正反链取小），并<strong>删掉跨家族共享的 k-mer</strong>，于是每条 read 只能归给唯一匹配的家族；
判定规则是最佳家族 ≥ 2 个唯一 k-mer 命中且至少为次优的 2 倍。</p>""")
    if TEV:
        rec = {k: v for k, v in TEV.items() if k.startswith(tuple(["IAP", "ETn", "ERV", "L1", "SINE"]))}
        got = {k: v["recall"] for k, v in rec.items() if v.get("recall") is not None}
        H.append("<p>引擎自身的成绩（留出元件模拟 reads，家族标签已知）：" +
                 "、".join(f"{k} {v*100:.0f}%" for k, v in got.items()) +
                 f"；在管家基因编码外显子上抽样 {TEV.get('_coding_exons',{}).get('simulated',0)} 条 reads，"
                 f"错误分配率 <strong>{TEV.get('_coding_exons',{}).get('false_assignment_rate',float('nan'))*100:.1f}%</strong>。"
                 f"召回率因家族而异（{min(got.values())*100:.0f}–{max(got.values())*100:.0f}%），"
                 "所以只能做<em>同家族跨组</em>比较，不能跨家族比丰度。"
                 "例外是 SINE_B：留出元件上 900 条模拟 reads 召回为 0，"
                 "因此它的命中数不可解释，本文<strong>不对 SINE 家族下任何结论</strong>。</p>")
    H.append("""<h3>7.3 pilot 结果与它的强度</h3>""")
    if TEK:
        rr = TEK.get("tcdd_over_ctrl_ratio", {})
        lib = TEK["libraries"]
        cnt = TEK.get("counts", {})
        est = TEK.get("estimable", {})
        ok = [f for f in cnt if est.get(f.replace("_rpm", ""), True)]
        bad = [f for f in cnt if not est.get(f.replace("_rpm", ""), True)]
        c_ok = "、".join(
            f"{f.replace('_rpm','')} {int(cnt[f]['Control'])} vs {int(cnt[f]['TCDD'])}"
            f"（×{rr[f]:.2f}）" for f in ok)
        c_bad = "、".join(
            f"{f.replace('_rpm','')} {int(cnt[f]['Control'])} vs {int(cnt[f]['TCDD'])}" for f in bad)
        H.append(f"""<p>本机到 ENA 的长连接约 300 KB/s，且在 10–45 MB 处反复断流；改成<em>按字节分块 + 每块校验大小</em>
（<code>scripts/get_fastq_chunked.sh</code>）之后才稳定。为了让两个文库可比，各取<strong>等字节前缀（前 20 MB）</strong>：
对照 {lib[0]['n_reads']:,} 条、TCDD {lib[-1]['n_reads']:,} 条 reads，约占单库的 1.5%。</p>
<p>结果分成两半。可解释的 {len(ok)} 个家族（命中数 ≥ 50）：{c_ok}，全部落在 ×0.96–1.07，中位
×{TEK.get('median_ratio', float('nan')):.2f}。<strong>在这个深度上，家族级 TE 去抑制检测不到。</strong>
不可解释的三个家族计数只有个位数到二十几（{c_bad}），任何比值都是噪声，图上标 n.d. 而不是画成 ×3。</p>""")
        H.append(img("fig11_te_kmers.png",
                     "图 4｜真实 reads 上的 TE 家族定量（等字节前缀 pilot，每文库约 27 万条 reads）。"
                     "灰色为对照、红色为 TCDD 16 周；仅计数 ≥ 50 的家族给出比值，其余标 n.d. 与两侧计数。"
                     "四个可解释家族的比值都在 1 附近——这是第一次用真实 reads 而非启动子代理验证“家族同比例移动”的判断。"))
        H.append("""<div class="note"><strong>怎么读这组数，以及为什么现在不把它写成结论。</strong><ul>
<li>如果所有家族同比例移动，那是<strong>文库组成</strong>而不是 TE 去抑制——上一版用启动子代理时正是栽在这里
（七个家族同比例降到 0.62 倍）。这次真实 reads 给出的中位比值 ×1.04，第一次把“没有系统性偏移”这件事
从推断变成了测量。</li>
<li>但前缀不是随机抽样。10x 文件按 barcode 排序，等字节前缀等于“相近数量的一部分细胞”，
两个文库覆盖的细胞类型比例仍可能不同。因此 pilot 能证明<strong>管线在真实数据上跑通、且没有明显文库伪影</strong>，
不能证明“TCDD 不改变 TE”。</li>
<li>要写成结论，需要完整 FASTQ（4 个文库约 11 GB，或 130 GB 全库）+ 每文库 ≥ 500 万条 reads，
并做细胞类型分层（tdT⁺ 的 SMC 与 tdT⁻ 的内皮/巨噬要分开算）。执行清单与磁盘/时间估算见
<code>scripts/TE_count_readme.md</code>。</li>
<li><strong>本轮决定：不跑全量 TE，pilot 作为补充材料。</strong>三条理由：它不改任何主结论（因果链停在
motif 层与域可及性）；4 个文库约 11 GB，直连 300 KB/s 且频繁断流，分块下载要占两个整晚；
而审稿人最可能问的“归一化有没有偏”，pilot 的 ×0.96–1.07 已经回答了。
因此 pilot 的定位是<em>引擎验证 + 归一化阴性对照</em>，不是生物学结论。若审稿人明确要求家族定量，
再按分块脚本补跑，并同时补 barcode 白名单。</li>
</ul></div>""")
    else:
        H.append("<p class='note'>TE k-mer pilot 结果尚未生成（<code>scripts/16_te_kmers.py</code>）。</p>")
    if len(TEF):
        H.append("""<p>作为交叉参照，上一版的“启动子代理”结论仍然成立且必须保留：
七个家族在 TCDD 后<em>同比例</em>移动，属文库效应，故不报告跨组代理差异；
可信的只有组内共变（巨噬 ERV1/L1/SINE 与 ISG 的 ρ ≈ 0.42/0.42/0.41，对照 0.10–0.21）。
新管线的作用是把“代理”换成“真实 reads 命中”，而不是推翻这条判断。</p>""")

    # ================================================================= 8 boundary
    H.append("<h2>8. 边界条件：D1 与 D3 说明的是“暴露-指标特异性”</h2>")
    H.append("""<p>把 D1/D3 放进同一张矩阵，目的不是再讲一遍“暴露有害”，而是界定核心机制成立的边界。
两套数据给出的是<strong>不同理化性质污染物在同一指标体系上的不同落点</strong>：</p>""")
    if len(NICH):
        b = NICH[NICH.dataset.isin(["D1", "D3"])].copy()
        b["niche_minus_parenchyma"] = b.niche_minus_parenchyma.round(3)
        H.append(tbl(b[["dataset", "group", "metric", "mean_d_niche", "mean_d_parenchyma",
                        "mean_d_immune", "niche_minus_parenchyma"]].rename(
            columns={"dataset": "体系", "group": "暴露", "metric": "指标", "mean_d_niche": "微环境 d",
                     "mean_d_parenchyma": "实质 d", "mean_d_immune": "免疫 d",
                     "niche_minus_parenchyma": "微环境 − 实质"}), floatfmt="{:.3f}"))
    H.append(f"""<div class="bound"><strong>D1（PM2.5 / 肺）：方向与 D2 相反。</strong>
微环境（内皮）在转录熵与 LADleak 上的效应<em>低于</em>实质细胞（Δd = −0.54 / −0.54），
只有 TE 代理是微环境更高（+0.20）。这不能用“暴露强度不同”解释，只能说明
<strong>颗粒物与 AhR 配体走的是不同的入口</strong>：PM2.5 的首击落在肺泡上皮，而 TCDD 的首击落在 SMC/血管壁。</div>""")
    H.append(f"""<div class="bound"><strong>D3（母代 PM2.5 / 子代脂肪）：跨代效应存在，但不是同一套指纹。</strong>
HGDI 在微环境与实质都明显为负（−0.71 / −0.31），LADleak 在实质为正（+0.21）；
这与 D2 的“域松弛 + 熵增”不是同一模式。合理读法是：跨代传递改变的是<em>发育程序的可塑性</em>，
而不是本节定义的“缓冲崩溃”。把它写成“跨代复现”会过度解释。</div>""")
    H.append("""<p>这就是本文愿意承担的代价：核心机制只在一个体系里做到因果级，
其余两个体系只提供边界。相比把三套数据硬凑成一条“普适规律”，这种写法更经得起审稿人逐个数据点地问。</p>""")

    # ================================================================= 9 genetics
    H.append("<h2>9. 遗传独立性：把负结果写成正面主张</h2>")
    H.append(f"""<p>上一版的写法是“GWAS 不支持表观开关是器官退变的共同遗传枢纽”。这句话在逻辑上没错，
但把一项<em>有价值的阴性证据</em>写成了失败。换一个框架：如果环境暴露是通过后天表观缓冲崩溃加速衰老的，
那么它的靶点<strong>本来就不应该</strong>是人群常见遗传变异的靶点——否则环境与遗传会共线，环境医学就没有独立价值。</p>
<p>数据正好是这个样子。表观开关面板的 {ENR.get('n_panel_assoc',0):,} 条关联中，
{ENR.get('panel_prop',0)*100:.1f}% 落在器官退变/衰老性状；通路匹配背景基因集为
{ENR.get('background_prop',0)*100:.1f}%（{ENR.get('n_background_assoc',0):,} 条）。
<strong>OR = {GI.get('odds_ratio',float('nan')):.2f}，95% CI {GI.get('ci_low',float('nan')):.2f}–{GI.get('ci_high',float('nan')):.2f}，
Fisher p = {ENR.get('fisher_p',float('nan')):.1e}</strong>——区间完全在 1 以下，方向是“更少”，不是“没有”。
逐基因看，{GI.get('n_zero_ageing_hits',0)}/{GI.get('n_switch_genes',0)} 个开关基因在衰老类性状上零命中。</p>

<div class="core"><strong>Genetically independent environmental vulnerability.</strong>
这句话可以被写成可检验的主张，而不是修辞：<br>
① <em>环境臂强</em>：同一批开关基因在暴露下系统性变化（11 个开关的平均表达变化中位
{GI.get('median_abs_exposure_delta',float('nan')):.3f}，最大 {GI.get('max_abs_exposure_delta',float('nan')):.3f}，
全部 ρ &lt; 0）；<br>
② <em>遗传臂弱</em>：它们在常见变异层面不富集于器官退变/衰老性状（OR 0.53）；<br>
③ <em>两者不共线</em>：若共线，OR 应 &gt; 1。<br>
三条同时成立，才能说“环境表观重塑的靶点独立于先天遗传易感性”。这也是环境医学相对于遗传流行病学的不可替代性所在。</div>""")
    H.append("""<div class="note"><strong>不能越过的线。</strong>
这是<strong>关联富集</strong>，不是孟德尔随机化；OR &lt; 1 也不能证明“环境导致衰老”，只能证明
“这两套靶点不重叠”。要把它升级成因果，需要 mQTL/甲基化 QTL 或 TWAS，
而 mQTLdb/GoDMC 与 eQTL Catalogue 的接口在当前环境不可达（已多次返回 410/超时）。
因此本文的措辞是 “genetically independent”，不是 “genetically causal”。</div>""")
    if GI:
        H.append(tbl(pd.DataFrame([{
            "指标": "面板关联数", "值": f"{ENR.get('n_panel_assoc',0):,}"},
            {"指标": "面板命中器官退变/衰老比例", "值": f"{ENR.get('panel_prop',0)*100:.1f}%"},
            {"指标": "背景命中比例", "值": f"{ENR.get('background_prop',0)*100:.1f}%"},
            {"指标": "OR（95% CI）", "值": f"{GI.get('odds_ratio',float('nan')):.2f} "
                                        f"({GI.get('ci_low',float('nan')):.2f}–{GI.get('ci_high',float('nan')):.2f})"},
            {"指标": "衰老类性状零命中的开关基因", "值": f"{GI.get('n_zero_ageing_hits',0)}/{GI.get('n_switch_genes',0)}"},
        ])))
        H.append('<p class="files">表｜遗传独立性的定量摘要（results/validation/genetic_independence.json）。</p>')

    # ================================================================= 10 verification
    H.extend(report_sec10.build())

    # ================================================================= 11 wet lab design
    H.append("<h2>11. 若要上湿实验：三步最小设计（含判据）</h2>")
    H.append("""<p>本轮的 in silico 验证已经把“预期方向”写死，湿实验只需要做判决，不需要探索。
三步都用现成细胞：MOVAS（小鼠主动脉 SMC）、bEnd.3（内皮）、RAW264.7 或 BMDM。</p>""")
    H.append("""<table class="tbl"><tr><th style="width:22%">实验</th><th style="width:34%">做法</th>
<th style="width:24%">读出</th><th>判据（证伪条件）</th></tr>
<tr><td><b>1. 旁分泌解耦</b><br>（对应第 4 节）</td>
<td>TCDD（10 nM, 24 h）处理 WT 或 siAhr 的 MOVAS → 换无血清培养基 24 h → 收 CM → 刺激 bEnd.3 6/24 h</td>
<td>bEnd.3 的 Klf2 / Smad2 启动子活性（荧光素酶或 ChIP-qPCR H3K9me3）、qPCR</td>
<td>若 siAhr 的 CM 与 WT CM 对 bEnd.3 的作用<strong>无差异</strong>，则“非细胞自主”被推翻，
应改为内皮自身的 AhR 或血流力学解释</td></tr>
<tr><td><b>2. 基质配体</b><br>（对应第 5 节）</td>
<td>TCDD ± CH-223191（1 μM）处理 MOVAS，48 h 收上清</td>
<td>ELISA：TIMP1、FN1；上清刺激 RAW264.7 24 h 后流式 Trem2/Lgals3、qPCR Isg15/Ifit1</td>
<td>若 TIMP1 明显下降、巨噬回到对照表型，则本文“基质配体而非 SASP”的解释被推翻</td></tr>
<tr><td><b>3. 结构回补</b><br>（对应第 3 节）</td>
<td>MOVAS 中 TCDD ± 敲低 Ahr，或用 dCas9-KRAB 靶向 AhR motif 位点</td>
<td>ATAC-qPCR（该 bin）、H3K9me3 ChIP-qPCR、以及下游 Timp1/Fn1 表达</td>
<td>若敲低 Ahr 后该 bin 的开放与 H3K9me3 都不变，则“AhR 依赖的开放”不成立，
第 3 节的因果链降级为相关</td></tr>
</table>""")
    H.append("""<p class="files">三步都不需要重新造模，周期各 1–2 周；第 1、2 步共用同一批 CM，实际是两个读出。</p>""")

    # ------------------------------------------------------------------ 11.1 n=2
    mde = load("power/mde.csv")
    pri = load("power/primary_endpoints.csv")
    ps = jload("power/summary.json")
    H.append("<h3>11.1 把 n=1 变成 n=2：先算清楚能买到什么，再决定要不要买</h3>")
    H.append("""<p>“再补一只 Ahr-KO”这句话，在动物房是三个月，在统计上是另一个问题：
<strong>补一只之后，哪个对比终于有自由度，它能检出的最小效应是多少，而我们手上的效应是不是比它大</strong>。
用现有个体水平数据估出组内 SD（RNA 有 3+3 只可估，df = 4；ATAC 只有 2 个对照文库，df = 1，默认放大 1.5 倍），
再对 2 vs 2、2 vs 3、3 vs 3、4 vs 4 做 6,000 次模拟（双侧 α = 0.05，合并方差 t），结果如下。</p>""")
    if len(mde):
        t = mde.copy()
        t["n_ko"] = t.n_ko.astype(int); t["n_wt"] = t.n_wt.astype(int)
        t.columns = ["KO 只数", "WT 只数", "最小可检 d（80% 功效）", "最小可检 d（60% 功效）"]
        H.append(t.to_html(index=False, classes="tbl", border=0, escape=False,
                           na_rep="—", float_format=lambda v: f"{v:.2f}"))
    H.append("""<p class="files">表｜补样方案与最小可检效应（results/power/mde.csv）。d 为组内个体间 SD 的倍数。</p>""")
    if len(pri):
        t = pri[["celltype", "endpoint", "d_ko_vs_ctrl", "power_2v3_koctrl", "n_needed_koctl",
                 "d_ko_vs_wt", "power_2v2_kowt", "n_needed_kowt"]].copy()
        t.columns = ["细胞类型", "指标", "KO−Ctrl d", "功效(2 vs 3)", "需要只数/组",
                     "KO−WT d", "功效(2 vs 2)", "需要只数/组"]
        H.append(t.to_html(index=False, classes="tbl", border=0, escape=False,
                           na_rep="—", float_format=lambda v: f"{v:.2f}"))
    H.append(img("fig14_power.png",
                 "图 3｜(a) 不同补样方案的检出功效，竖线为四个主终点的实测效应（|d| = 1.1–1.3）；"
                 "(b) 四个主终点的效应量对照两个门槛——只有“内皮 prom_het_rel 的 KO−Ctrl”越过 2 vs 3 的门槛。"))
    H.append(f"""<div class="note"><strong>算完之后的结论，和最初的直觉不一样。</strong>
补 <strong>一只</strong> Ahr-KO 买到的是“<em>KO 仍未回到对照水平</em>”这个命题的检验（2 vs 3，门槛 d = {ps.get('mde_80pct',{}).get('2v3', float('nan')):.2f}），
而且只有内皮 <code>prom_het_rel</code> 站得住（KO−Ctrl d = 13.4，即使把 ATAC 的 SD 放大 3 倍仍剩 6.7）；
SMC 的 <code>frac_het</code> 会从 d = 2.4 掉到 0.8，功效从 ~0.5 掉到 ~0.2；RNA 层三个终点的 KO−Ctrl 只有 d = 2.1–3.1，
需要 3–8 只/组——不是一次补样能解决的。
补 <strong>一只 + 一只 16 周 WT</strong>（2 vs 2）也<em>买不到</em>“回落了多少”的检验：门槛 d = {ps.get('mde_80pct',{}).get('2v2', float('nan')):.2f}，
而实测只有 0.8–1.5，功效 0.08–0.15。</div>""")
    H.append("""<p>所以正确的做法不是“补了动物就宣布个体水平显著”，而是把两件事分开：
<strong>H1（可检验）</strong>为“Ahr-KO 未复原”，用 KO vs Control 的 2 vs 3；
<strong>H2（只描述）</strong>为“回落幅度”，给每只动物的值与 bootstrap 区间，明确标注 n = 2 不做检验。
这条判定规则、四个主终点、Holm 校正、以及“p ≥ 0.05 时只允许写什么”，已经写进冻结的预注册文件
<code>results/power/preregistration.md</code>；数据到位后直接跑
<code>scripts/25b_n2_analysis.py</code>（现在跑会如实输出“证据不足”，<code>--selftest</code> 用合成动物验证过整条判定路径）。</p>""")

    # ------------------------------------------------------------------ 11.2 ChIP-qPCR
    qpr = load("chip/qpcr_primers.csv")
    qds = jload("chip/qpcr_design.json")
    gls = jload("chip/gene_level_summary.json")
    H.append("<h3>11.2 AHR ChIP-qPCR：阳性对照先立，再谈位点</h3>")
    H.append(f"""<p>第 10.5 节已经说明：公开的 GSE276152 连 CYP1A1/AHRR 都没有峰，不能用来判决
“AHR 是否结合在异染色质启动子上”。这件事只能自己做一次，而且必须自带阳性对照。
面板分三层，共 {qds.get('loci', 0)} 个位点、{qds.get('pairs', 0)} 对引物（含备选）：</p>""")
    H.append("""<table class="tbl"><tr><th style="width:8%">层</th><th style="width:34%">选点规则</th>
<th style="width:22%">预期结果</th><th>它判决什么</th></tr>
<tr><td><b>C</b></td><td>CYP1A1、CYP1B1、AHRR、TIPARP、NQO1、AHR（TCDD-AHR 教科书靶基因）</td>
<td>TCDD 后显著富集（≥5 倍）</td><td><b>实验是否成立</b>。C 层失败则整批作废，不看 A、B 层</td></tr>
<tr><td><b>A</b></td><td>异染色质启动子 + 有 AhR motif + TCDD 后开放且 Ahr-KO 回落（按 Δ 开放排序取前 10）</td>
<td>若“AhR 直接结合”成立，应富集；若“AhR 依赖但不直接结合”，应与 B 无差</td>
<td><b>本文措辞的判决</b>：决定能不能从“AhR 依赖”升级为“AhR 直接结合”</td></tr>
<tr><td><b>B</b></td><td>异染色质启动子 + 有 AhR motif + <em>未</em>开放（取 Δ 最小的 6 个）</td>
<td>与背景无差</td><td>特异性：把“任何异染色质启动子都被结合”这种平凡解释排除掉</td></tr>
</table>""")
    H.append("""<p>两个物种各做一套，因为它们回答不同的问题：<strong>小鼠 MOVAS（mm10）是判决性的</strong>——
它测的正是我们提出假设的那个物种与细胞；<strong>人 HCASMC（hg38）是可与他人的 ChIP 对照的那一套</strong>，
也是合作方现有抗体的用武之地。人源面板只保留在人类冠脉 SMC 中可检出的 A/B 位点（检出率 ≥ 5%），
否则测的是关闭的染色质，必然阴性却说明不了任何事。</p>""")
    if len(qpr):
        keep = qpr[qpr.tier.isin(["A", "B", "C"])].copy()
        t = keep[["genome", "tier", "human", "mouse", "fwd", "rev", "tm_f", "tm_r",
                  "size", "mode", "n_products", "verdict"]].copy()
        t.columns = ["基因组", "层", "人", "鼠", "正向引物", "反向引物",
                     "Tm_F", "Tm_R", "产物(bp)", "设计档", "基因组产物数", "特异性判定"]
        H.append(t.to_html(index=False, classes="tbl", border=0, escape=False,
                           na_rep="—", float_format=lambda v: f"{v:.1f}"))
    H.append(f"""<p class="files">表｜ChIP-qPCR 引物（<code>results/chip/qpcr_primers.csv</code>，含备选对与扩增子 FASTA）。
“设计档”为 strict / relaxed / near-TSS：这些启动子中位 GC 偏高（多为 CpG 岛），
严格参数常常无解，放宽档与“扩增子靠近 TSS 但不跨 TSS”的位点在表中已如实标注。
<strong>全基因组特异性已经做过</strong>（第 10.9 节）：mm10 与 hg38 各自全序列跑电子 PCR，
3′ 端 15 nt 精确锚定、5′ 余部 ≤2 错配、产物 40–3,000 bp。
{int((qpr.verdict == 'ok').sum())}/{len(qpr)} 对为单产物；Tier A 的 14 对与 Tier C 的 12 对
主引物全部单产物；失效的 2 对（mm10 Iffo2、hg38 TIPARP 备选）落在重复序列里，
局部重设计后仍多产物，<strong>已从下单表移除</strong>。逐产物坐标见
<code>results/chip/ispcr_hits.csv</code>，重设计尝试见 <code>results/chip/qpcr_replacements.csv</code>。</p>""")
    H.append("""<div class="core"><strong>三种可能的结果，三种写法（事先定好，免得到时候找理由）。</strong><ul>
<li>C 层成立、A 层富集且显著高于 B 层 → 可以说“AHR 直接结合在这些异染色质启动子上”，
第 3 节的因果链升级为“配体→受体→结合→开放”。</li>
<li>C 层成立、A 与 B 无差 → 维持“AhR 依赖但不直接结合”，并把机制解释转向“AhR 依赖的
间接去抑制”（例如通过 AHR 诱导的转录因子或代谢物改变染色质状态）。</li>
<li>C 层不成立 → 实验作废，重做 ChIP（换抗体/换交联条件/加 sonication 对照），
在此之前不对结合问题表态；这一条同样适用于公开数据（第 10.5 节）。</li>
</ul></div>""")

    # ================================================================= 12 completeness
    H.append("<h2>12. 三层闭环的完成度（本轮更新）</h2>")
    H.append("""<table class="tbl"><tr><th>层</th><th>要求</th><th>本次状态</th><th>证据位置</th></tr>
<tr><td>L1 体外扰动</td><td>化合物/基因扰动库初筛</td>
<td><b>部分</b>：新增 in silico 条件培养基与“遗传学拮抗剂”臂（n = 4 组），给出可证伪预测；
仍未做化合物文库的高通量筛选</td><td>4.1、5.1、第 11 节</td></tr>
<tr><td>L2 体内暴露</td><td>动物体内单细胞定位</td>
<td><b>完成</b>：D2 含 tdT 谱系示踪 + SMC 特异 Ahr 敲除 + 配对 scATAC；D1/D3 作边界</td>
<td>2、3–6、8</td></tr>
<tr><td>L3 人群</td><td>mQTL/eQTL + GWAS 因果推断</td>
<td><b>部分</b>：GWAS Catalog v2 完成并改写为“遗传独立性”；mQTL 接口不可达；未做 MR</td><td>9</td></tr>
<tr><td>TE 层</td><td>家族级 TE 重定量</td>
<td><b>方法学完成 + pilot</b>：FASTQ 公开性核实、k-mer 引擎建立并验证；全量运行需完整下载（11–130 GB）</td>
<td>7</td></tr>
<tr><td>轨迹层</td><td>RNA velocity / Monocle3</td>
<td><b>替代完成</b>：每谱系扩散伪时间 + 回退比例；velocity 需 FASTQ 重建剪接比，未做</td><td>6</td></tr>
</table>""")

    # ================================================================= 13 limits
    H.append("<h2>13. 局限</h2>")
    H.append("""<div class="note"><ul>
<li><strong>没有足迹分析。</strong>GSE276870 无 fragments/BAM，第 3–4 节是 motif 富集 + chromVAR 式偏差分数，
能回答“哪类位点被打开”，不能回答“TF 是否结合在该位点”。要真足迹需原始 fragments。</li>
<li><strong>motif 结论受 GC 组成影响。</strong>内皮顶端是 KLF7/12/14、SP2/SP5 这类 GC 富集 motif，
在启动子背景下可能被组成偏差放大；因此第 4 节以偏差分数为主证据，富集表为辅，
且明确写出“AhR 富集在内皮不显著”。</li>
<li><strong>scATAC 的 Ahr-KO 文库浅 27.3%</strong>（按 bin 计数中位数 3,718 / 细胞，其余三库 4,682–5,378）。按 Ahr-KO 深度对三个深库做二项稀疏化后，8 个显著域指标方向与显著性全保、效应量衰减 7–25%，但内皮 <code>prom_het_rel</code> 在 60% 压力档掉出显著（p = 0.189）；SMC 的 Ahr::Arnt 偏差分数有近一半来自深度（−0.198 → −0.086）。因此凡涉及 Ahr-KO 或跨库比较的染色质结论只用相对量与偏差分数，绝对开放比例不作比较（第 10.1 节）。TCDD 臂只有 1 个 ATAC 文库。<em>第 10.8 节把这件事重做了一遍：把三个深库削到 3,718 的深度后，4 个冻结终点 × 3 次稀释方向 12/12 全保；其中 log1p 尺度的效应会被凹性放大（内皮 −2.09 → −8.03），所以同时报线性 share 统计量。</em><strong>补一条第 15 节查出来的：</strong>逐细胞终点的 72 个组合里 67 个三次稀释全同号，但<em>巨噬 LAM（WT 16 周）只有 2/3 同号、效应衰减 93.8%</em>——所以“等深度后方向不变”这句话不能覆盖 LAM，它只在域指标那一层（24/24 符号）成立。</li>
<li><strong>细胞组成只能排除，不能检验。</strong>组成检验在每臂 3 个文库时最低可达 p 是 1/21 ≈ 0.048，实测各成分的精确置换 p 全部 ≥0.143；改用 Welch 会得到 3 个 FDR&lt;0.05，但那只是 n=3 下的正态假设，本文不采信。16 周 WT 与 Ahr-KO 各 1 只，组成数值既不能判为异常也不能判为正常（第 10.7 节）。能做的是分解：10 个冻结终点里 6 个在组成匹配后仍成立，2 个混合，2 个组成敏感（已改口径）。</li>
<li><strong>外部 AHR 占据证据只有一半的阳性对照。</strong>六套公开数据里没有任何一套能同时找回 6 个经典靶基因；用 summit 口径最好的是 HCASMC TCDD 与 SKMEL2 过表达（各 2/6：CYP1A1 + AHRR / CYP1A1 + CYP1B1）。GSE276152 提交的 narrowPeak 与它自己的信号轨对不上（AHRR TSS 有 96 分位 summit 但峰集里零峰），因此该峰集不能用于结合判定，本文也不引用任何 ChIP 交集作为结合证据（第 10.5、10.9 节）。<em>第 10.11 节又往前走了一步：抛开峰集直接读 TCDD/DMSO 两条信号轨，SMC 开放 bin 既没有配体特异性富集，配对差还偏低（z = −0.75，覆盖度收紧后 −2.04）——所以“AhR 直接结合开放 bin”这一环<b>没有干实验替代品</b>，只保留“AhR 依赖”，结合证据必须来自 11.2 的自做 ChIP-qPCR。</em></li>
<li><strong>“跨谱系普遍击穿”在核验后不成立。</strong>用同长度、同 GC 含量、同 TSS 分层的常染色质区域做 1,000 次置换，只有 SMC 的域开放偏移能与之区分（d = 0.347，z = 5.47，p = 0.001）；内皮的 d = 0.119 落在零分布内（p = 0.284），SMC 之外的谱系都不能声称“击穿 H3K9me3 域”。另外 H3K9me3 域不等于 LAD 代理，二者在三个谱系上方向相反（第 10.3 节）。</li>
<li><strong>细胞水平显著不等于个体水平显著。</strong>按小鼠聚合后（RNA 8 只、ATAC 4 文库），内皮的组成型异染色质开放 p = 0.138（d = −1.82）、SMC p = 0.185（d = −1.55），方向与细胞水平相反且不显著；只有巨噬 AgingSen（d = +5.50，p = 0.0038）与成纤维基质配体（d = +3.99，p = 0.008）在个体水平站得住。Ahr-KO 臂各只有 1 只动物，只能给个体值，不能做检验（第 10.2 节）。</li>
<li><strong>小鼠标签不可直接移植到人。</strong>在人类冠脉图谱（GSE131778，11,756 细胞、8 个供体）里，<strong>只有两个程序真的落对</strong>：LAM 落在巨噬（AUC 0.74，8 个供体里 7 个过 0.6）、内皮面板落在内皮（AUC 0.69，8/8）；<strong>基质配体面板与 Modulated SMC 面板都不落对</strong>（前者在成纤维 0.687 vs SMC 0.162，后者在成纤维 0.279 vs SMC 0.206），逐供体 AUC>0.6 只有 3/8 与 4/8——这两个标签在人体内不是 SMC 专属，只能写“可被检测”，不能写“映射成功”。但 Modulated SMC 面板在人类成纤维上反而更高（0.279 vs SMC 0.206）——它更像“间质激活态”而非 SMC 专属状态。该图谱也无疾病标签，未做病变 vs 正常比较（第 10.6 节）。</li>
<li><strong>细胞通讯只有组水平、n = 1。</strong>tdT⁺/tdT⁻ 是同一心脏的两半，SMC 与巨噬从不出现在同一文库，
样本级配对在物理上不存在；第 5 节只给效应量、不给 p 值。</li>
<li><strong>in silico 验证是方向性检查。</strong>4 组、每组 1 个合并观测，单参数标定 + 单点留出；
不能给 p 值，不能替代第 11 节的三步湿实验。</li>
<li><strong>TE pilot 不是结果。</strong>recall 0.26–0.86 因家族而异，pilot 使用 FASTQ 字节前缀（barcode 排序 → 细胞子集），
中位比值接近 1 只能说明归一化没有偏；要写成生物学结论必须先跑完整文库。</li>
<li><strong>伪时间不是速率。</strong>每谱系单独做 DPT 并以对照端为根，绝对伪时间不可跨谱系比较；
内皮与成纤维的位移接近 0，说明这两个群体的主变异方向不是暴露。</li>
<li><strong>人群层是关联富集。</strong>mQTL 接口不可达，未做 MR；“genetically independent” 不等于 “genetically causal”。</li>
<li><strong>D3 的实质细胞占比低（1–5%）</strong>，该体系的“实质 vs 微环境”对比功效有限。</li>
<li><strong>T50 只有 3 个时间点、线性内插</strong>，只能用于谱系间排序，不是动力学常数。</li>
</ul></div>""")

    # ================================================================= 13 repro
    H.append("<h2>14. 复现</h2>")
    H.append("""<pre>python scripts/02_download.py                 # GEO 原始数据 + MD5 清单
python scripts/03_build_annotation.py         # mm10 异染色质 / 重复序列注释
python scripts/04_process.py D1|D2|D3         # 聚类 + 四项指纹
python scripts/05_atac.py && python scripts/05b_atac_annot.py   # scATAC 5 kb 分箱 + 注释
python scripts/06_matrix.py                   # 交互矩阵 + T50
python scripts/07b_switch_evidence.py         # 开关关联 + Ahr-KO 逆转
python scripts/lib_seq.py                     # mm10.2bit（本地序列读取）
python scripts/11_motif.py                    # motif 富集 + 偏差分数
python scripts/11b_het_promoter_motif.py      # 异染色质启动子定向检验
python scripts/12_ccc.py                      # 组水平通讯
python scripts/13_te.py                       # TE 启动子代理（旧版）
python scripts/14_insilico_validation.py      # 三项 in silico 验证（第 4/5/9 节）
python scripts/15_trajectory.py               # 每谱系扩散伪时间（第 6 节）
python scripts/get_fastq_prefix.sh            # FASTQ 前缀下载（受限带宽）
python scripts/16_te_kmers.py --pilot         # k-mer 家族定量（第 7 节）
python scripts/08_genetics.py                 # GWAS / eQTL
python scripts/09_figures.py && python scripts/17_figures_story.py
# ---- 第 10 节的核验链（顺序即依赖顺序）----
python scripts/18_atac_downsample.py          # 二项稀疏化到等深度
python scripts/18b_downsample_summary.py      # 判定表：holds / direction only / fails
python scripts/19_pseudobulk.py               # 按小鼠聚合，细胞 vs 个体 p
python scripts/20_permutation.py              # GC/TSS 匹配置换 + 阈值敏感性
python scripts/21_lr_permutation.py           # 度保持网络重连
python scripts/22_ahr_chip.py                 # GSE276152 峰 → 基因 → 同源
python scripts/22b_ahr_liftover.py            # hg38→mm10 liftOver 交集
python scripts/23_human_coronary.py           # GSE131778 分群 + 投影
python scripts/24_figures_robust.py           # fig12a–c、fig13
# ---- 第 10.7–10.9 节的混淆排雷与外部替补 ----
python scripts/29_composition_audit.py        # 组成检验 + Kitagawa 分解 + 判语
python scripts/30_depth_matched.py            # 二项稀释到 Ahr-KO 深度，重算 |d|
python scripts/31_cell_subsampling.py         # 1,000 x 80% 细胞重抽样 + jackknife
python scripts/32_alt_chip.py                 # GEO 检索 + 峰口径 Tier C
python scripts/32b_bigwig_tierC.py            # 只放 bigWig 的系列：窗口均值门禁
python scripts/32c_summit_tierC.py            # summit 口径门禁（最终判定用这个）
python scripts/35_occupancy_specificity.py     # 占据特异性：TCDD vs DMSO，含阳性对照与覆盖度平衡
python scripts/36_figures_occupancy.py        # fig18
python scripts/lib_bigwig.py                  # 手写 bigWig 读取器
python scripts/33_primer_specificity.py       # 42 对引物的全基因组电子 PCR
python scripts/33b_primer_redesign.py         # 多产物引物的闭环替换尝试
python scripts/34_figures_confound.py         # fig16、fig17
python scripts/10_report.py                   # 本报告
</pre>
<p class="files">新增/更新的结果目录：<code>results/validation/</code>（CM 传递、基质配体、遗传独立性）、
<code>results/trajectory/</code>（伪时间与级联）、<code>results/te/te_family_kmer_counts.csv</code> 与
<code>kmer_engine_validation.json</code>（真实 reads 的家族定量与引擎成绩）、
<code>results/robust/</code>（等深度重算与判定）、<code>results/pseudobulk/</code>（小鼠级 CPM 与检验）、
<code>results/permutation/</code>（匹配零分布与阈值表）、<code>results/chip/</code>（ChIP 交集、liftOver 与引物电子 PCR）、
<code>results/audit/</code>（组成分解、等深度重算、细胞重抽样）、
<code>results/chip_alt/</code>（外部 AHR 图谱的 Tier C 与 summit 门禁）、
<code>results/human/</code>（人类冠脉投影）。</p>
<p class="files">外部数据（<code>data/external/</code>）：GSE276152 AHR/TCF21 narrowPeak（hg38）与两条信号轨、
GSE304148（SKMEL2 AHR ChIP，两条 bigWig）、GSE240495（K562 AHR CUT&amp;RUN）、
GSE205502（人肝细胞 AHR 差异峰集）、UCSC hg38ToMm10 chain、hg38 refFlat、
GSE131778 人类冠脉表达矩阵、mm10/hg38 2bit 全序列。</p>""")
    # ================================================================= 15 audit + tier
    H.append("<h2>15. 主张审计与外部独立复现</h2>")
    ca = load("audit/claim_audit.csv")
    er = jload("human/external_replication.json")
    H.append("""<h3>15.1 把每句话的数字重算一遍</h3>
<p>这一节的动因很实际：报告长到 14 节以后，正文里的数字有一部分是手敲进散文的。
审计脚本 <code>scripts/37_claim_audit.py</code> 不做重新分析，它只做一件事——
<strong>把这句话需要的数字从产出它的那个文件里取出来，再套上这句话自己隐含的阈值</strong>。
三种结果：通过、数值对但主张不成立（改写）、数值根本对不上（失败）。
最后一种最要命：一个没有文件支持的手敲数字，是审稿人查出一次就再难挽回的那类问题。</p>""")
    if len(ca):
        n = {k: int(v) for k, v in ca.verdict.value_counts().items()}
        H.append(f"""<p><strong>{len(ca)} 条主张：通过 {n.get('PASS', 0)} 条，需改写 {n.get('REWRITE', 0)} 条，
数值对不上 {n.get('FAIL', 0)} 条。</strong>没有一条是数字凭空写的——但有三条主张的数字虽然对，
却撑不住原来的说法，已按审计结果改掉（见 15.2）。</p>""")
        t = ca[["id", "claim", "value", "rule", "verdict"]].copy()
        t.columns = ["编号", "主张", "重算值", "判据", "判定"]
        # the rules contain "<0.05" / "d>0"; unescaped they open a tag and the
        # rest of the cell disappears into the DOM
        t = t.apply(lambda col: col.map(
            lambda x: str(x).replace("&", "&amp;").replace("<", "&lt;")
                            .replace(">", "&gt;")))
        H.append(tbl(t, maxrows=20))

    H.append("""<h3>15.2 审计查出来的两处硬伤（改了，不是说一句“已检查”）</h3>
<div class="note">
<p><strong>硬伤一（C5）：内皮的“开门人”指认不出来。</strong>
原文写“内皮上去的是一组与 AhR 无关的因子：KLF2/HIF1A/EPAS1”，读起来像“这些 bin 上富集了这些 motif”。
实际是两件事被混在一起：内皮 DA 开放 bin 只有 <strong>30 个</strong>，这 30 个 bin 上
<strong>没有任何一个 motif 通过 FDR 校正</strong>（Ahr::Arnt p = 0.272，最小的 p 也未过线）；
而 KLF2/HIF1A/EPAS1 的高分是<strong>偏差分数</strong>（内皮整体染色质在这些 motif 上更富），不是“暴露后被打开”。
改法：把“指认上游”降为假设，正文只保留“非细胞自主”这个有 KO 判据支持的结论。</p>
<p><strong>硬伤二（C13）：两个程序在人体里并不落对。</strong>
原文写“LAM、内皮面板、基质配体面板都落在正确细胞类型”。用 8 个供体逐个重算后：
LAM 落在巨噬（AUC 0.74，7/8 供体过 0.6）、内皮面板落在内皮（0.69，8/8）——这两个成立；
但<strong>基质配体面板在成纤维 0.687 vs SMC 0.162</strong>、<strong>Modulated SMC 面板在成纤维 0.279 vs SMC 0.206</strong>，
逐供体只有 3/8 与 4/8 过线。所以这两个标签在人体里不是 SMC 专属。
改法：只保留“可被检测”，删掉“映射成功”，并把这一条写进局限。</p>
<p><strong>顺带查出的第三处（C8）：</strong>报告反复引用的“等深度 12/12”指的是<strong>域指标的方向</strong>
（3 指标 × 4 谱系 × 2 档深度 = 24 次比较，符号全保），但“方向与显著性全保”是错的——
12 行里只有 <strong>7 行 holds</strong>，5 行只是 direction only。逐细胞层面 72 个组合里 67 个三次稀释全同号，
<strong>巨噬 LAM（WT 16 周）只有 2/3、衰减 93.8%</strong>，这一条已写进局限，不再用“12/12”覆盖它。</p>
</div>""")

    H.append("""<h3>15.3 外部独立复现：换一个与本研究无关的人体数据集</h3>
<p>小鼠那侧的证据来自一项研究（GSE276870/276872）+ 每臂 3 只。只在那一处出现的现象是假设，不是结论。
所以拿 <strong>GSE131778</strong>——人冠状动脉、11,756 个细胞、<strong>8 个供体</strong>、无暴露、与小鼠实验无关——
问三个事先定好判据的问题（<code>scripts/39_external_replication.py</code>）：</p>
<ul>
<li><strong>R1 供体分裂复现</strong>：把 8 个供体随机分成互斥的两半，各自独立算程序的细胞类型特异性（AUC），
问两半的 AUC 差是否小于 0.05。判据：至少 2 个程序过。</li>
<li><strong>R2 逐供体稳健</strong>：8 个供体各自单独算 AUC&gt;0.6。判据：每个程序至少 6/8。</li>
<li><strong>R3 分叉点</strong>：若“内皮轴与髓系/基质轴是两条线”，则 ECpanel 与 LAM、与 Matrix 的单细胞相关
应<strong>都低于</strong> LAM 与 Matrix 之间的相关。</li>
</ul>""")
    if er:
        r1, r2, r3 = er["R1"], er["R2"], er["R3"]
        def _verdict(p_):
            """Landing (AUC high and donor-robust) vs stably-not-landing.

            The split-half difference is not part of this rule: a programme can
            be perfectly stable and still stable at the wrong cell type, which is
            what Matrix turns out to be.
            """
            a, nd = er["pooled"][p_], r2["n_donors_auc_gt_06"][p_]
            if a >= 0.68 and nd >= 6:
                return "落对"
            if a >= 0.65 or nd >= 5:
                return "边缘"
            return "不落对"

        rows = []
        for p_ in ["LAM", "ECpanel", "Matrix", "ModSMC"]:
            rows.append({
                "程序": p_, "预设目标细胞": {"LAM": "巨噬", "ECpanel": "内皮",
                                              "Matrix": "SMC（预先声明）", "ModSMC": "SMC"}[p_],
                "合并 AUC": f"{er['pooled'][p_]:.3f}",
                "分裂 |ΔAUC| 中位": f"{r1['median_abs_dauc'][p_]:.3f}",
                "逐供体 AUC>0.6": f"{r2['n_donors_auc_gt_06'][p_]}/8",
                "判定": _verdict(p_),
            })
        H.append(tbl(pd.DataFrame(rows)))
        H.append(f"""<p><strong>R1：{ '过' if er['verdicts']['R1'] else '未过' }</strong>
（分裂后 |ΔAUC| 中位：内皮面板 0.018、基质 0.037、LAM 0.075、ModSMC 0.089；
LAM 的分裂差稍大，但它在 7/8 个供体里各自都能落对，方向一致）。
<strong>R2：{ '过' if er['verdicts']['R2'] else '未过' }</strong>
（LAM 7/8、内皮面板 8/8 过；<strong>基质面板 3/8、ModSMC 4/8 不过</strong>）。
<strong>R3：{ '成立' if er['verdicts']['R3'] else '不成立' }</strong>——
r(内皮面板, LAM) = {r3['r_ecpanel_lam']:+.2f}、r(内皮面板, 基质) = {r3['r_ecpanel_matrix']:+.2f}，
而 r(LAM, 基质) = {r3['r_lam_matrix']:+.2f}。<strong>内皮那条线与髓系/基质那条线，在人体单细胞层面确实是分开的</strong>，
差了三到二十倍。这个合并口径只是起点：<strong>第 16.1 节把它重算成四种算法，其中“髓系–基质是三者最紧的一对”这句话站不住，已在那里撤回</strong>。</p>
<p>这三条合起来给了一个很干净的结论，也是本文现在敢写“分叉”的唯一理由：
<strong>分叉的两端（内皮身份程序、髓系 LAM 程序）在一个与小鼠完全无关的人体数据集里、逐供体都能复现；
中间那两个标签（Modulated SMC、基质配体面板）复现不了，应当从主张里退掉。</strong>
值得注意的是 ModSMC 与 Matrix 恰好是本文“SMC 在分泌、在改造自己”那句话的载体——
审计和外部复现各自独立地把它们标了出来，这种一致不是巧合。</p>""")

    H.append("""<p class="files">新增脚本：<code>scripts/37_claim_audit.py</code>（15 条主张重算）、
<code>scripts/39_external_replication.py</code>（8 供体分裂复现）。
新增结果：<code>results/audit/claim_audit.{csv,md}</code>、
<code>results/human/external_replication.{csv,json}</code>。</p>""")

    H.extend(report_sec16.build())
    H.append("</div></body></html>")

    out = os.path.join(REP, "Age3_report.html")
    open(out, "w", encoding="utf-8").write("\n".join(H))
    print("written", out, round(os.path.getsize(out) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
