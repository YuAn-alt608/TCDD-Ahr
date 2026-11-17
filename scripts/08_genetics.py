"""Layer 3: human genetic evidence for the epigenetic switch panel (GWAS Catalog v2 + GTEx eQTL).

Outputs: results/genetics/*.csv
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_core as L

GWAS = "https://www.ebi.ac.uk/gwas/rest/api/v2"
HDR = {"Accept": "application/json", "User-Agent": "Mozilla/5.0 (research)"}

PANEL = {
    "SIRT1": "heterochromatin/sirtuin switch",
    "SUV39H1": "H3K9me2/3 writer",
    "SETDB1": "H3K9me3 writer",
    "TRIM28": "KRAB-ZNF co-repressor",
    "EHMT2": "G9a H3K9me1/2",
    "DNMT1": "maintenance methylation",
    "DNMT3A": "de novo methylation",
    "UHRF1": "hemimethylation reader",
    "MORC2": "heterochromatin compaction",
    "HDAC1": "deacetylation",
    "HDAC2": "deacetylation",
    "SIRT6": "H3K9/H3K56 deacetylation",
    "KDM6B": "H3K27me3 demethylase",
    "EZH2": "H3K27me3 writer",
    "TET2": "DNA demethylation",
    "AHR": "dioxin receptor (exposure sensor)",
    "NFE2L2": "oxidative stress response",
    "FOXO3": "longevity",
}

# traits considered "organ degeneration / ageing"
AGING_KW = ["aging", "ageing", "longevity", "senescen", "telomere", "frailty",
            "age at", "biological age", "epigenetic age", "grip strength",
            "bone mineral density", "sarcopenia", "osteoporosis"]
ORGAN_KW = {
    "cardiovascular": ["coronary", "myocardial", "heart", "cardiovascular", "blood pressure",
                       "atheroscler", "stroke", "artery", "atrial", "hypertens", "lipid",
                       "cholesterol", "triglyceride"],
    "lung": ["lung", "pulmonary", "asthma", "copd", "respiratory", "fev1", "spirometr",
             "bronch", "emphysema", "idiopathic pulmonary fibrosis"],
    "kidney": ["kidney", "renal", "creatinine", "egfr", "nephro", "albuminuria"],
    "liver": ["liver", "hepatic", "cirrhos", "fibrosis", "bilirubin", "ggt", "alt ",
              "alkaline phosphatase", "nafld", "steato"],
    "metabolic": ["body mass", "obesity", "diabet", "insulin", "glucose", "hba1c",
                  "adipos", "waist", "bmi"],
    "neuro": ["alzheimer", "parkinson", "cognitive", "neurodegener", "dementia",
              "amyotrophic", "multiple sclerosis", "brain volume", "schizophren"],
    "immune_inflam": ["inflamm", "autoimmun", "arthritis", "lupus", "crohn", "colitis",
                      "psoriasis", "immun", "infection", "covid"],
    "cancer": ["cancer", "carcinoma", "tumor", "tumour", "neoplas", "leukemia",
               "lymphoma", "melanoma", "glioma"],
}


def get(url, params=None, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HDR, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 + 2 * i); continue
            return None
        except Exception:
            time.sleep(2 + 2 * i)
    return None


def fetch_gene(g, size=500):
    j = get(f"{GWAS}/associations", {"mapped_gene": g, "page": 0, "size": size})
    if not j:
        return []
    rows = []
    assoc = (j.get("_embedded") or {}).get("associations", [])
    for a in assoc:
        p = a.get("p_value")
        if p is None:
            try:
                p = float(a.get("pvalue_mantissa", 0)) * 10 ** int(a.get("pvalue_exponent", 0))
            except Exception:
                p = np.nan
        traits = [t.get("efo_trait") for t in (a.get("efo_traits") or [])]
        rows.append({
            "gene": g, "rs_id": (a.get("snp_allele") or [{}])[0].get("rs_id") if a.get("snp_allele") else None,
            "p_value": p, "efo_traits": ";".join([t for t in traits if t]),
            "reported_trait": ";".join(a.get("reported_trait") or []),
            "study": a.get("accession_id"), "pubmed": a.get("pubmed_id"),
            "locations": ";".join(a.get("locations") or []),
        })
    return rows


def classify(txt):
    t = (txt or "").lower()
    cats = []
    if any(k in t for k in AGING_KW):
        cats.append("ageing")
    for c, kws in ORGAN_KW.items():
        if any(k in t for k in kws):
            cats.append(c)
    return cats if cats else ["other"]


def main():
    out = os.path.join(L.RES, "genetics")
    os.makedirs(out, exist_ok=True)

    rows = []
    for g in PANEL:
        r = fetch_gene(g)
        print(f"  {g}: {len(r)} associations", flush=True)
        rows.extend(r)
        time.sleep(0.15)
    A = pd.DataFrame(rows)
    if A.empty:
        print("no GWAS data"); return
    A = A.drop_duplicates(subset=["gene", "rs_id", "reported_trait", "study"])
    A["sig"] = A.p_value < 5e-8
    A["trait_text"] = (A.efo_traits.fillna("") + ";" + A.reported_trait.fillna(""))
    A["categories"] = A.trait_text.apply(classify)
    A.to_csv(os.path.join(out, "gwas_associations_panel.csv"), index=False)

    # per-gene summary
    summ = []
    for g, s in A.groupby("gene"):
        sig = s[s.sig]
        cats = [c for lst in s.categories for c in lst]
        summ.append({
            "gene": g, "role": PANEL[g], "n_assoc": len(s), "n_sig": int(s.sig.sum()),
            "n_distinct_traits": s.reported_trait.nunique(),
            "top_traits": "; ".join(s[s.sig].reported_trait.value_counts().head(4).index.astype(str)),
            "n_ageing": cats.count("ageing"),
            "n_cardiovascular": cats.count("cardiovascular"),
            "n_lung": cats.count("lung"), "n_kidney": cats.count("kidney"),
            "n_liver": cats.count("liver"), "n_metabolic": cats.count("metabolic"),
            "n_neuro": cats.count("neuro"), "n_immune": cats.count("immune_inflam"),
            "min_p": s.p_value.min(),
        })
    S = pd.DataFrame(summ).sort_values("n_sig", ascending=False)
    S.to_csv(os.path.join(out, "gene_panel_summary.csv"), index=False)
    print(S.to_string(index=False), flush=True)

    # enrichment of organ-degeneration traits vs. a background gene panel
    bg_genes = ["APOE", "TP53", "BRCA1", "MYC", "PTEN", "EGFR", "VEGFA", "IL6", "TNF",
                "MTHFR", "ACE", "AGT", "NOS3", "PPARG", "ADRB2", "COMT", "MTR", "SOD2",
                "CAT", "GPX1", "NOS2", "CRP", "FTO", "MC4R", "TCF7L2", "KCNJ11", "IRS1",
                "LPL", "APOB", "PCSK9", "LDLR", "CETP", "MTHFR2" if False else "MTRR"]
    brows = []
    for g in bg_genes:
        brows.extend(fetch_gene(g, size=200))
        time.sleep(0.12)
    B = pd.DataFrame(brows)
    if not B.empty:
        B = B.drop_duplicates(subset=["gene", "rs_id", "reported_trait", "study"])
        B["trait_text"] = (B.efo_traits.fillna("") + ";" + B.reported_trait.fillna(""))
        B["categories"] = B.trait_text.apply(classify)
        B.to_csv(os.path.join(out, "gwas_associations_background.csv"), index=False)

    def prop_age(df):
        cats = [c for lst in df.categories for c in lst]
        return (cats.count("ageing") + cats.count("cardiovascular") + cats.count("lung")
                + cats.count("kidney") + cats.count("liver") + cats.count("metabolic")) / max(len(df), 1)

    if not B.empty:
        from scipy.stats import fisher_exact
        panel_hit = A.categories.apply(lambda c: any(x in c for x in
                     ["ageing", "cardiovascular", "lung", "kidney", "liver", "metabolic"])).sum()
        bg_hit = B.categories.apply(lambda c: any(x in c for x in
                    ["ageing", "cardiovascular", "lung", "kidney", "liver", "metabolic"])).sum()
        table = [[panel_hit, len(A) - panel_hit], [bg_hit, len(B) - bg_hit]]
        orr, p = fisher_exact(table)
        enr = {"panel_prop": panel_hit / max(len(A), 1), "background_prop": bg_hit / max(len(B), 1),
               "odds_ratio": float(orr), "fisher_p": float(p),
               "n_panel_assoc": int(len(A)), "n_background_assoc": int(len(B))}
        json.dump(enr, open(os.path.join(out, "trait_enrichment.json"), "w"), indent=1)
        print("\nenrichment:", json.dumps(enr, indent=1), flush=True)

    # GTEx eQTL look-up for the strongest panel genes
    eqtl = []
    for g in ["SIRT1", "SUV39H1", "SETDB1", "AHR", "EHMT2"]:
        for tissue in ["Artery_Coronary", "Lung", "Liver", "Kidney_Cortex"]:
            j = get("https://gtexportal.org/api/v2/association/singleTissueEqtl",
                    {"geneSymbol": g, "tissueSiteDetailId": tissue, "itemsPerPage": 50})
            if not j:
                continue
            for d in j.get("data", [])[:50]:
                eqtl.append({"gene": g, "tissue": tissue, "variant": d.get("variantId"),
                             "pvalue": d.get("pValue"), "nes": d.get("nes"),
                             "snp": d.get("snpId")})
            time.sleep(0.1)
    if eqtl:
        E = pd.DataFrame(eqtl)
        E.to_csv(os.path.join(out, "gtex_eqtl_panel.csv"), index=False)
        print(f"\nGTEx eQTL rows: {len(E)}", flush=True)


if __name__ == "__main__":
    main()
