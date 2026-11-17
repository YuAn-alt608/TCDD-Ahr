"""GEO dataset discovery via NCBI eutils (no API key -> keep <=3 req/s)."""
import json, time, sys, urllib.parse, urllib.request

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
HEAD = {"User-Agent": "Mozilla/5.0 (research-metadata-query)"}


def get(url):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=HEAD)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            time.sleep(2 + attempt * 2)
    return ""


def esearch(term, db="gds", retmax=40):
    u = BASE + "esearch.fcgi?" + urllib.parse.urlencode(
        {"db": db, "term": term, "retmode": "json", "retmax": retmax})
    j = get(u)
    try:
        return json.loads(j)["esearchresult"].get("idlist", [])
    except Exception:
        return []


def esummary(ids, db="gds"):
    if not ids:
        return []
    u = BASE + "esummary.fcgi?" + urllib.parse.urlencode(
        {"db": db, "id": ",".join(ids), "retmode": "json"})
    j = get(u)
    try:
        res = json.loads(j)["result"]
    except Exception:
        return []
    out = []
    for uid in res.get("uids", []):
        r = res[uid]
        out.append({
            "gse": r.get("accession"),
            "title": r.get("title"),
            "taxon": r.get("taxon"),
            "n_samples": r.get("n_samples"),
            "gdstype": r.get("gdstype"),
            "pdat": r.get("pdat"),
            "summary": (r.get("summary") or "")[:400],
        })
    return out


QUERIES = {
    "pm25_lung_sc": '(PM2.5 OR "fine particulate") AND ("single cell" OR "single-cell" OR scRNA) AND "expression profiling by high throughput sequencing"',
    "cadmium_sc": 'cadmium AND ("single cell" OR "single-cell" OR scRNA) AND "expression profiling by high throughput sequencing"',
    "tcdd_liver_sc": '(TCDD OR dioxin) AND ("single cell" OR "single-cell") AND "expression profiling by high throughput sequencing"',
    "bap_lung_sc": '("benzo[a]pyrene" OR "benzo(a)pyrene" OR BPDE) AND ("single cell" OR "single-cell") AND "expression profiling by high throughput sequencing"',
    "arsenic_sc": 'arsenic AND ("single cell" OR "single-cell") AND "expression profiling by high throughput sequencing"',
    "bpa_sc": '("bisphenol" OR BPA) AND ("single cell" OR "single-cell") AND "expression profiling by high throughput sequencing"',
    "pm25_atac": 'PM2.5 AND ("ATAC" OR "ATAC-seq")',
    "cadmium_atac": 'cadmium AND (ATAC OR chromatin accessibility)',
    "pm25_bulk_lung": '(PM2.5 OR "fine particulate") AND lung AND "expression profiling by high throughput sequencing"',
}

allrows = {}
for k, q in QUERIES.items():
    ids = esearch(q)
    time.sleep(0.4)
    rows = esummary(ids)
    for r in rows:
        r["query"] = k
        allrows.setdefault(r["gse"], r)
    time.sleep(0.4)
    print(f"### {k}: {len(rows)} hits", flush=True)
    for r in rows[:12]:
        print(f"  {r['gse']:>10} | {r['taxon'][:28]:28} | n={r['n_samples']:>4} | {str(r['pdat'])[:10]} | {r['title'][:80]}")

with open(sys.argv[1] if len(sys.argv) > 1 else "geo_candidates.json", "w", encoding="utf-8") as f:
    json.dump(list(allrows.values()), f, ensure_ascii=False, indent=1)
print(f"\nunique GSE: {len(allrows)}")
