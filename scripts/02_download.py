"""Download all raw inputs with resume + manifest. Run with --only <tag>."""
import hashlib, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
ANN = os.path.join(ROOT, "data", "annotation")
META = os.path.join(ROOT, "data", "meta")
for d in (RAW, ANN, META):
    os.makedirs(d, exist_ok=True)

G = "https://ftp.ncbi.nlm.nih.gov/geo"
S = G + "/samples"
SER = G + "/series"
U = "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/database"

FILES = [
    # --- D1: PM2.5 lung, mouse, 26 samples, normalized h5ad ---
    ("d1_pm25_lung", f"{SER}/GSE296nnn/GSE296297/suppl/GSE296297_batch1_normalized.h5ad",
     "D1_PM25_lung.h5ad"),
    # --- D2: TCDD/dioxin coronary vasculature, per-sample 10x RNA matrices (GSE276872) ---
    ("d2_rna", f"{S}/GSM8507nnn/GSM8507976/suppl/GSM8507976_CTR_TDT1_feature_bc_matrix.h5", "D2_rna_CTR_TDT1.h5"),
    ("d2_rna", f"{S}/GSM8507nnn/GSM8507977/suppl/GSM8507977_CTR_TDT2_feature_bc_matrix.h5", "D2_rna_CTR_TDT2.h5"),
    ("d2_rna", f"{S}/GSM8507nnn/GSM8507978/suppl/GSM8507978_CTR_TDT3_feature_bc_matrix.h5", "D2_rna_CTR_TDT3.h5"),
    ("d2_rna", f"{S}/GSM8507nnn/GSM8507979/suppl/GSM8507979_CTR_NTDT1_feature_bc_matrix.h5", "D2_rna_CTR_NTDT1.h5"),
    ("d2_rna", f"{S}/GSM8507nnn/GSM8507980/suppl/GSM8507980_CTR_NTDT2_feature_bc_matrix.h5", "D2_rna_CTR_NTDT2.h5"),
    ("d2_rna", f"{S}/GSM8507nnn/GSM8507981/suppl/GSM8507981_CTR_NTDT3_feature_bc_matrix.h5", "D2_rna_CTR_NTDT3.h5"),
    ("d2_rna", f"{S}/GSM8507nnn/GSM8507982/suppl/GSM8507982_1482_TdT_feature_bc_matrix.h5", "D2_rna_TCDD_TDT1.h5"),
    ("d2_rna", f"{S}/GSM8507nnn/GSM8507983/suppl/GSM8507983_1483_TdTa_feature_bc_matrix.h5", "D2_rna_TCDD_TDT2.h5"),
    ("d2_rna", f"{S}/GSM8507nnn/GSM8507984/suppl/GSM8507984_1483_TdTb_feature_bc_matrix.h5", "D2_rna_TCDD_TDT3.h5"),
    ("d2_rna", f"{S}/GSM8507nnn/GSM8507985/suppl/GSM8507985_1482_NoTdT_feature_bc_matrix.h5", "D2_rna_TCDD_NTDT1.h5"),
    ("d2_rna", f"{S}/GSM8507nnn/GSM8507986/suppl/GSM8507986_1483_NoTdTa_feature_bc_matrix.h5", "D2_rna_TCDD_NTDT2.h5"),
    ("d2_rna", f"{S}/GSM8507nnn/GSM8507987/suppl/GSM8507987_1483_NoTdTb_feature_bc_matrix.h5", "D2_rna_TCDD_NTDT3.h5"),
    ("d2_rna", f"{S}/GSM8507nnn/GSM8507988/suppl/GSM8507988_WT_TCDD16_T_feature_bc_matrix.h5", "D2_rna_WT16_TDT.h5"),
    ("d2_rna", f"{S}/GSM8507nnn/GSM8507989/suppl/GSM8507989_WT_TCDD16_NT_feature_bc_matrix.h5", "D2_rna_WT16_NTDT.h5"),
    ("d2_rna", f"{S}/GSM8507nnn/GSM8507990/suppl/GSM8507990_AKO_TCDD16_T_feature_bc_matrix.h5", "D2_rna_AKO16_TDT.h5"),
    ("d2_rna", f"{S}/GSM8507nnn/GSM8507991/suppl/GSM8507991_AKO_TCDD16_NT_feature_bc_matrix.h5", "D2_rna_AKO16_NTDT.h5"),
    # --- D2 paired scATAC (GSE276870), filtered peak matrices ---
    ("d2_atac", f"{S}/GSM8507nnn/GSM8507962/suppl/GSM8507962_CTRL1_filtered_peak_bc_matrix.h5", "D2_atac_CTRL1.h5"),
    ("d2_atac", f"{S}/GSM8507nnn/GSM8507963/suppl/GSM8507963_CTRL2_filtered_peak_bc_matrix.h5", "D2_atac_CTRL2.h5"),
    ("d2_atac", f"{S}/GSM8507nnn/GSM8507964/suppl/GSM8507964_WT_TCDD_filtered_peak_bc_matrix.h5", "D2_atac_WT_TCDD.h5"),
    ("d2_atac", f"{S}/GSM8507nnn/GSM8507965/suppl/GSM8507965_AHRKO_TCDD_filtered_peak_bc_matrix.h5", "D2_atac_AHRKO_TCDD.h5"),
    # --- D3: maternal PM2.5, offspring iWAT, per-sample matrices ---
    ("d3_iwat", f"{S}/GSM9229nnn/GSM9229087/suppl/GSM9229087_InMFA1_auto_matrix.tar.gz", "D3_FA1.tar.gz"),
    ("d3_iwat", f"{S}/GSM9229nnn/GSM9229088/suppl/GSM9229088_InMFA2_auto_matrix.tar.gz", "D3_FA2.tar.gz"),
    ("d3_iwat", f"{S}/GSM9229nnn/GSM9229089/suppl/GSM9229089_InMFA3_auto_matrix.tar.gz", "D3_FA3.tar.gz"),
    ("d3_iwat", f"{S}/GSM9229nnn/GSM9229090/suppl/GSM9229090_InMPM1_matrix.tar.gz", "D3_PM1.tar.gz"),
    ("d3_iwat", f"{S}/GSM9229nnn/GSM9229091/suppl/GSM9229091_InMPM2_matrix.tar.gz", "D3_PM2.tar.gz"),
    ("d3_iwat", f"{S}/GSM9229nnn/GSM9229092/suppl/GSM9229092_InMPM3_matrix.tar.gz", "D3_PM3.tar.gz"),
    # --- reference annotation: mm10 ---
    ("ann", f"{U}/refFlat.txt.gz", "mm10_refFlat.txt.gz"),
    ("ann", f"{U}/rmsk.txt.gz", "mm10_rmsk.txt.gz"),
    ("ann", f"{U}/cytoBand.txt.gz", "mm10_cytoBand.txt.gz"),
    ("ann", f"{U}/gap.txt.gz", "mm10_gap.txt.gz"),
]


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main(only=None):
    man_path = os.path.join(META, "download_manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else []
    have = {(m["dest"]) for m in man}
    for tag, url, name in FILES:
        if only and tag != only:
            continue
        dest = os.path.join(ANN if tag == "ann" else RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"[skip] {name} ({os.path.getsize(dest)/1e6:.1f} MB)", flush=True)
        else:
            print(f"[get ] {name} <- {url}", flush=True)
            t0 = time.time()
            r = subprocess.run(["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "3",
                                "-C", "-", "-A", "Mozilla/5.0", "-o", dest, url],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print("  FAILED:", r.stderr[-300:], flush=True)
                continue
            print(f"  ok {os.path.getsize(dest)/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)
        if name not in have:
            man = [m for m in man if m["dest"] != name]
            man.append({"tag": tag, "url": url, "dest": name,
                        "bytes": os.path.getsize(dest), "md5": md5(dest)})
            json.dump(man, open(man_path, "w"), indent=1)
    print("manifest entries:", len(man))


if __name__ == "__main__":
    only = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--only" else None
    main(only)
