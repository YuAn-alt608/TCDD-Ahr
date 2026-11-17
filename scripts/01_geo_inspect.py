"""Inspect GEO series: download family SOFT, parse supplementary files + sample titles, HEAD sizes."""
import gzip, io, json, os, re, sys, time, urllib.request

OUT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(os.path.dirname(OUT), "data", "meta")
os.makedirs(META, exist_ok=True)
HEAD = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def soft_url(gse):
    stem = gse[:-3] + "nnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stem}/{gse}/soft/{gse}_family.soft.gz"


def fetch(gse):
    path = os.path.join(META, f"{gse}_family.soft.gz")
    if not os.path.exists(path) or os.path.getsize(path) < 200:
        for a in range(3):
            try:
                req = urllib.request.Request(soft_url(gse), headers=HEAD)
                with urllib.request.urlopen(req, timeout=90) as r:
                    data = r.read()
                open(path, "wb").write(data)
                break
            except Exception as e:
                print("  retry", gse, e)
                time.sleep(2)
    if not os.path.exists(path):
        return None
    return gzip.decompress(open(path, "rb").read()).decode("utf-8", "replace")


def head_size(url):
    try:
        req = urllib.request.Request(url, headers=HEAD, method="HEAD")
        with urllib.request.urlopen(req, timeout=45) as r:
            return int(r.headers.get("Content-Length", -1))
    except Exception:
        return -1


def parse(gse, txt):
    d = {"gse": gse, "supp": [], "samples": [], "organism": None, "platform": [],
         "summary": "", "design": "", "type": []}
    title = None
    cur_sample = None
    for line in txt.splitlines():
        if line.startswith("!Series_title"):
            d["title"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Series_summary"):
            d["summary"] += line.split("=", 1)[1].strip() + " "
        elif line.startswith("!Series_overall_design"):
            d["design"] += line.split("=", 1)[1].strip() + " "
        elif line.startswith("!Series_type"):
            d["type"].append(line.split("=", 1)[1].strip())
        elif line.startswith("!Series_platform_id"):
            d["platform"].append(line.split("=", 1)[1].strip())
        elif line.startswith("!Series_sample_id"):
            d.setdefault("sample_ids", []).append(line.split("=", 1)[1].strip())
        elif line.startswith("!Series_relation"):
            v = line.split("=", 1)[1].strip()
            if "BioSample" in v:
                d.setdefault("biosamples", []).append(v.split("/")[-1])
        elif line.startswith("!Series_supplementary_file"):
            d["supp"].append(line.split("=", 1)[1].strip())
        elif line.startswith("^SAMPLE"):
            cur_sample = {"gsm": line.split("=", 1)[1].strip(), "title": "", "organism": "",
                          "source": "", "characteristics": [], "supp": []}
            d["samples"].append(cur_sample)
        elif cur_sample is not None:
            if line.startswith("!Sample_title"):
                cur_sample["title"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_organism_ch1"):
                cur_sample["organism"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_source_name_ch1"):
                cur_sample["source"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_characteristics_ch1"):
                cur_sample["characteristics"].append(line.split("=", 1)[1].strip())
            elif line.startswith("!Sample_supplementary_file"):
                cur_sample["supp"].append(line.split("=", 1)[1].strip())
    if d["samples"]:
        d["organism"] = d["samples"][0]["organism"]
    d["summary"] = d["summary"][:800]
    d["design"] = d["design"][:600]
    return d


def main(gses):
    infos = []
    for gse in gses:
        print(f"\n=== {gse} ===", flush=True)
        txt = fetch(gse)
        if not txt:
            print("  SOFT unavailable")
            continue
        d = parse(gse, txt)
        print(" ", d.get("title"))
        print("  organism:", d["organism"], "| type:", d["type"], "| n_samples:", len(d["samples"]))
        uniq_supp = sorted(set(d["supp"]))
        for u in uniq_supp:
            sz = head_size(u)
            d.setdefault("supp_size", {})[u] = sz
            print(f"    supp {sz/1e6:9.1f} MB  {u.split('/')[-1]}")
        # sample-level supplementary (often per-sample 10x dirs)
        ss = sorted(set(x for s in d["samples"] for x in s["supp"]))
        if ss:
            print(f"    sample-level supp files: {len(ss)} e.g. {ss[0].split('/')[-1]}")
        ch = {}
        for s in d["samples"]:
            for c in s["characteristics"]:
                k = c.split(":", 1)[0].strip()
                v = c.split(":", 1)[1].strip() if ":" in c else ""
                ch.setdefault(k, set()).add(v[:40])
        for k, v in ch.items():
            vals = sorted(v)
            if len(vals) <= 8:
                print(f"    char[{k}] = {vals}")
            else:
                print(f"    char[{k}] : {len(vals)} levels")
        d["sample_chars"] = {k: sorted(v) for k, v in ch.items()}
        infos.append(d)
    with open(os.path.join(META, "geo_series_info.json"), "w", encoding="utf-8") as f:
        json.dump(infos, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
