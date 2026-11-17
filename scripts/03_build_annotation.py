"""Build mm10 heterochromatin / repeat annotation resources.

Outputs (data/annotation/):
  mm10_genes_tss.tsv          gene, chrom, tss, strand, gene_body_start/end
  mm10_gene_density.tsv       1-Mb window gene count + repeat fraction
  domains_LADproxy.bed        gene-desert + repeat-rich 1-Mb windows (LAD proxy)
  domains_H3K9me3.bed         merged ENCODE H3K9me3 liver+heart domains
  sets/*.txt                  gene sets used by the scoring module
  annotation_summary.json
"""
import gzip, json, os
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = os.path.join(ROOT, "data", "annotation")
OUT = os.path.join(ANN, "processed")
SETS = os.path.join(OUT, "sets")
os.makedirs(SETS, exist_ok=True)

CHR_KEEP = {f"chr{i}" for i in range(1, 20)} | {"chrX", "chrY"}

REF = os.path.join(ANN, "mm10_refFlat.txt.gz")
RMSK = os.path.join(ANN, "mm10_rmsk.txt.gz")
CYTO = os.path.join(ANN, "mm10_cytoBand.txt.gz")


def load_genes():
    cols = ["gene", "tx", "chrom", "strand", "txStart", "txEnd", "cdsStart", "cdsEnd"]
    df = pd.read_csv(REF, sep="\t", header=None, usecols=range(8), names=cols,
                     dtype={"chrom": str})
    df = df[df.chrom.isin(CHR_KEEP)].copy()
    # one representative transcript per gene = longest
    df["len"] = df.txEnd - df.txStart
    df = df.sort_values("len", ascending=False).drop_duplicates("gene")
    df["tss"] = np.where(df.strand == "+", df.txStart, df.txEnd)
    df["body_start"] = df.txStart
    df["body_end"] = df.txEnd
    return df[["gene", "chrom", "tss", "strand", "body_start", "body_end"]].reset_index(drop=True)


def load_rmsk():
    # UCSC rmsk.txt has 17 cols: bin, swScore, milliDiv, milliDel, milliIns, genoName, ...
    cols = ["bin", "swScore", "milliDiv", "milliDel", "milliIns", "chrom", "start", "end",
            "genoLeft", "strand", "repName", "repClass", "repFamily", "repStart",
            "repEnd", "repLeft", "id"]
    df = pd.read_csv(RMSK, sep="\t", header=None, usecols=range(17), names=cols,
                     dtype={"chrom": str, "repClass": str, "repFamily": str})
    for c in ("start", "end"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["start", "end"])
    df["start"] = df.start.astype("int64")
    df["end"] = df.end.astype("int64")
    df = df[df.chrom.isin(CHR_KEEP)]
    return df


def merge_intervals(df, gap=0):
    """df: chrom,start,end -> merged"""
    out = []
    for chrom, sub in df.sort_values(["chrom", "start"]).groupby("chrom", sort=False):
        s = sub.start.to_numpy(); e = sub.end.to_numpy()
        cs, ce = s[0], e[0]
        for a, b in zip(s[1:], e[1:]):
            if a <= ce + gap:
                ce = max(ce, b)
            else:
                out.append((chrom, cs, ce)); cs, ce = a, b
        out.append((chrom, cs, ce))
    return pd.DataFrame(out, columns=["chrom", "start", "end"])


def genes_overlapping(genes, dom):
    """Return set of gene names whose body overlaps any domain."""
    hit = set()
    for chrom, sub in dom.groupby("chrom"):
        g = genes[genes.chrom == chrom]
        if g.empty:
            continue
        gs = g.body_start.to_numpy(); ge = g.body_end.to_numpy()
        for _, d in sub.iterrows():
            m = (gs < d.end) & (ge > d.start)
            if m.any():
                hit.update(g.gene.to_numpy()[m])
    return hit


def main():
    genes = load_genes()
    genes.to_csv(os.path.join(OUT, "mm10_genes_tss.tsv"), sep="\t", index=False)
    print("genes:", len(genes), flush=True)

    rmsk = load_rmsk()
    print("rmsk elements:", len(rmsk), flush=True)

    # ---------- 1. TE-promoter genes (TSS +/- 2 kb overlapped by LTR / LINE / DNA) ----------
    TECLASS = {"LTR", "LINE", "DNA", "RC", "Retroposon"}
    te = rmsk[rmsk.repClass.isin(TECLASS)]
    te_prom = set()
    for chrom, sub in te.groupby("chrom"):
        g = genes[genes.chrom == chrom]
        if g.empty:
            continue
        tss = g.tss.to_numpy(); gn = g.gene.to_numpy()
        for s, e in zip(sub.start.to_numpy(), sub.end.to_numpy()):
            m = (tss >= s - 2000) & (tss <= e + 2000)
            if m.any():
                te_prom.update(gn[m])
    # LTR/ERV only (most relevant to heterochromatin de-repression)
    ltr = rmsk[rmsk.repClass == "LTR"]
    ltr_prom = set()
    for chrom, sub in ltr.groupby("chrom"):
        g = genes[genes.chrom == chrom]
        if g.empty:
            continue
        tss = g.tss.to_numpy(); gn = g.gene.to_numpy()
        for s, e in zip(sub.start.to_numpy(), sub.end.to_numpy()):
            m = (tss >= s - 2000) & (tss <= e + 2000)
            if m.any():
                ltr_prom.update(gn[m])
    # strict subset: intact (>=1 kb) LTR element overlapping TSS +/- 1 kb
    ltr_strict = ltr[(ltr.end - ltr.start) >= 1000]
    ltr_sp = set()
    for chrom, sub in ltr_strict.groupby("chrom"):
        g = genes[genes.chrom == chrom]
        if g.empty:
            continue
        tss = g.tss.to_numpy(); gn = g.gene.to_numpy()
        for s, e in zip(sub.start.to_numpy(), sub.end.to_numpy()):
            m = (tss >= s - 1000) & (tss <= e + 1000)
            if m.any():
                ltr_sp.update(gn[m])
    # young/active families in mouse (IAP, ERVK, ERVL, ERV1)
    young = ltr[ltr.repFamily.isin({"ERVK", "ERVL", "ERV1", "ERVL-MaLR"})]
    young_genes = set()
    for chrom, sub in young.groupby("chrom"):
        g = genes[genes.chrom == chrom]
        if g.empty:
            continue
        tss = g.tss.to_numpy(); gn = g.gene.to_numpy()
        for s, e in zip(sub.start.to_numpy(), sub.end.to_numpy()):
            m = (tss >= s - 1000) & (tss <= e + 1000)
            if m.any():
                young_genes.update(gn[m])
    print("strict LTR-promoter genes:", len(ltr_sp), "| young-family LTR promoter:", len(young_genes), flush=True)

    # ---------- 2. 1-Mb windows: gene density + repeat fraction -> LAD proxy ----------
    genes["win"] = (genes.tss // 1_000_000).astype(int)
    dens = genes.groupby(["chrom", "win"]).size().rename("n_genes").reset_index()
    rmsk["wstart"] = (rmsk.start // 1_000_000) * 1_000_000
    rmsk["wlen"] = np.minimum(rmsk.end, rmsk.wstart + 1_000_000) - rmsk.start
    rep = rmsk.groupby(["chrom", "wstart"]).wlen.sum().rename("rep_bp").reset_index()
    rep["frac_rep"] = (rep.rep_bp / 1_000_000).clip(0, 1)
    rep["win"] = (rep.wstart // 1_000_000).astype(int)
    win = rep.merge(dens, on=["chrom", "win"], how="left").fillna({"n_genes": 0})
    win["n_genes"] = win.n_genes.astype(int)
    win["win_start"] = win.win * 1_000_000
    win["win_end"] = win.win_start + 1_000_000
    win.to_csv(os.path.join(OUT, "mm10_gene_density.tsv"), sep="\t", index=False)

    lad = win[(win.n_genes <= 1) & (win.frac_rep >= 0.35)][["chrom", "win_start", "win_end"]]
    lad.columns = ["chrom", "start", "end"]
    lad = merge_intervals(lad, gap=200_000)
    lad.to_csv(os.path.join(OUT, "domains_LADproxy.bed"), sep="\t", index=False, header=False)
    lad_genes = genes_overlapping(genes, lad)
    print("LAD-proxy domains:", len(lad), "| genes in LAD-proxy:", len(lad_genes), flush=True)

    # ---------- 3. H3K9me3 domains (ENCODE mm10 liver + heart, merged within 5 kb) ----------
    hk = []
    for f in ["H3K9me3_liver.bed.gz", "H3K9me3_heart.bed.gz"]:
        p = os.path.join(ANN, f)
        if os.path.exists(p):
            d = pd.read_csv(p, sep="\t", header=None, usecols=[0, 1, 2],
                            names=["chrom", "start", "end"], dtype={"chrom": str})
            d = d[d.chrom.isin(CHR_KEEP)]
            hk.append(d)
    hk = pd.concat(hk, ignore_index=True)
    hk_dom = merge_intervals(hk, gap=5000)
    hk_dom.to_csv(os.path.join(OUT, "domains_H3K9me3.bed"), sep="\t", index=False, header=False)
    hk_genes = genes_overlapping(genes, hk_dom)
    print("H3K9me3 domains:", len(hk_dom), "| genes:", len(hk_genes), flush=True)

    # ---------- 4. pericentromeric (mm10 cytoBand has no 'acen'; use gap 'centromere') ----------
    pc = []
    gapf = os.path.join(ANN, "mm10_gap.txt.gz")
    if os.path.exists(gapf):
        gp = pd.read_csv(gapf, sep="\t", header=None,
                         names=["bin", "chrom", "start", "end", "ix", "n", "size", "type", "bridge"],
                         dtype={"chrom": str})
        cen = gp[gp.type == "centromere"].copy()
        cen["start"] = (cen.start - 2_000_000).clip(lower=0)
        cen["end"] = cen.end + 2_000_000
        pc = cen[["chrom", "start", "end"]]
    pc.to_csv(os.path.join(OUT, "domains_pericentromeric.bed"), sep="\t", index=False, header=False)

    # ---------- 5. writable gene sets ----------
    def write_set(name, s):
        with open(os.path.join(SETS, name + ".txt"), "w") as f:
            f.write("\n".join(sorted(s)))
        print(f"  set {name}: {len(s)}")

    write_set("TEprom_LTR", ltr_prom)
    write_set("TEprom_LTR_strict", ltr_sp)
    write_set("TEprom_young", young_genes)
    write_set("TEprom_all", te_prom)
    write_set("LADproxy", lad_genes)
    write_set("H3K9me3", hk_genes)
    write_set("ConstitutiveHet", lad_genes | hk_genes)

    summary = {
        "n_genes": int(len(genes)),
        "n_rmsk": int(len(rmsk)),
        "n_TEprom_LTR": len(ltr_prom),
        "n_TEprom_all": len(te_prom),
        "n_LADproxy_domains": int(len(lad)),
        "n_LADproxy_genes": len(lad_genes),
        "n_H3K9me3_domains": int(len(hk_dom)),
        "n_H3K9me3_genes": len(hk_genes),
        "n_pc_domains": int(len(pc)),
        "genome": "mm10",
        "sources": {
            "refFlat": "UCSC hgdownload mm10/database/refFlat.txt.gz",
            "rmsk": "UCSC hgdownload mm10/database/rmsk.txt.gz (RepeatMasker)",
            "H3K9me3": "ENCODE mm10 replicated peaks: liver ENCFF546PMU, heart ENCFF018ZTP",
            "cytoBand": "UCSC mm10 cytoBand (acen)",
        },
        "LADproxy_definition": "1-Mb windows with <=1 RefSeq TSS and >=35% RepeatMasker coverage, merged within 200 kb",
    }
    json.dump(summary, open(os.path.join(OUT, "annotation_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
