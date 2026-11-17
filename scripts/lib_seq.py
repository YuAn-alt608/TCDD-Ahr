"""Minimal mm10 2bit reader + parallel ranged downloader.

The per-region REST calls are latency bound (~1.2 s each, thousands of regions);
pulling the 790 MB 2bit once with 8 ranged connections runs at ~5 MB/s and then
every interval lookup is local. No pysam / py2bit needed - the format is simple.
"""
import os, sys, time, ssl, urllib.request, json
from concurrent.futures import ThreadPoolExecutor
import numpy as np

URL = "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.2bit"
CTX = ssl.create_default_context()
_CODE = "TCAG"          # 0=T, 1=C, 2=A, 3=G (verified against the UCSC API)


def download(path, n_threads=8, chunk=8 * 1024 * 1024):
    if os.path.exists(path) and os.path.getsize(path) > 700 * 1024 * 1024:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(
            urllib.request.Request(URL, method="HEAD", headers={"User-Agent": "Mozilla/5.0"}),
            timeout=60, context=CTX) as r:
        total = int(r.headers["Content-Length"])
    parts = [None] * ((total + chunk - 1) // chunk)

    def get(i):
        s, e = i * chunk, min((i + 1) * chunk, total) - 1
        for _ in range(4):
            try:
                req = urllib.request.Request(
                    URL, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={s}-{e}"})
                with urllib.request.urlopen(req, timeout=180, context=CTX) as r:
                    buf = bytearray()
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        buf += b
                if len(buf) == e - s + 1:
                    return bytes(buf)
            except Exception:
                time.sleep(1)
        raise RuntimeError(f"chunk {i} failed")

    t0 = time.time()
    with ThreadPoolExecutor(n_threads) as ex:
        for i, b in enumerate(ex.map(get, range(len(parts)))):
            parts[i] = b
            if i % 10 == 0:
                print(f"  {i+1}/{len(parts)} chunks, {time.time()-t0:.0f}s", flush=True)
    with open(path, "wb") as f:
        for b in parts:
            f.write(b)
    print(f"downloaded {total/1e6:.0f} MB in {time.time()-t0:.0f}s", flush=True)
    return path


class TwoBit:
    def __init__(self, path):
        self.f = open(path, "rb")
        hdr = self.f.read(16)
        magic = np.frombuffer(hdr[:4], dtype="<u4")[0]
        if magic != 0x1A412743:
            raise ValueError("not a 2bit file")
        self.n = int(np.frombuffer(hdr[8:12], dtype="<u4")[0])
        self.idx = {}
        for _ in range(self.n):
            nl = self.f.read(1)[0]
            name = self.f.read(nl).decode()
            off = int(np.frombuffer(self.f.read(4), dtype="<u4")[0])
            self.idx[name] = off

    def _record(self, chrom):
        f = self.f
        f.seek(self.idx[chrom])
        dna = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        nb = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        nstarts = np.frombuffer(f.read(4 * nb), dtype="<u4") if nb else np.array([], dtype="<u4")
        nsizes = np.frombuffer(f.read(4 * nb), dtype="<u4") if nb else np.array([], dtype="<u4")
        mb = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        f.seek(4 * mb, 1)
        f.seek(4 * mb, 1)
        f.seek(4, 1)                      # reserved uint32, always zero, before packed dna
        return dna, (nstarts, nsizes), f.tell()

    def fetch(self, chrom, start, end):
        """0-based half-open interval -> upper-case sequence, N inside hard-masked blocks."""
        if chrom not in self.idx:
            return "N" * (end - start)
        dna, (nstarts, nsizes), dna_off = self._record(chrom)
        req_start, req_end = start, end
        start = max(0, start)
        end = min(end, dna)
        if end <= start:
            return "N" * (req_end - req_start)
        self.f.seek(dna_off + start // 4)
        nbytes = (end + 3) // 4 - start // 4
        buf = np.frombuffer(self.f.read(nbytes), dtype=np.uint8)
        bits = np.empty(nbytes * 4, dtype=np.uint8)
        bits[0::4] = (buf >> 6) & 3
        bits[1::4] = (buf >> 4) & 3
        bits[2::4] = (buf >> 2) & 3
        bits[3::4] = buf & 3
        idx = bits[start % 4: start % 4 + (end - start)]
        out = "".join(np.array(list(_CODE), dtype="<U1")[idx].tolist())
        pad_l = start - req_start
        pad_r = req_end - end
        if len(nstarts):
            arr = np.frombuffer(out.encode(), dtype=np.uint8).copy()
            for st, ln in zip(nstarts, nsizes):
                a, b = max(int(st), start), min(int(st) + int(ln), end)
                if b > a:
                    arr[a - start:b - start] = ord("N")
            out = arr.tobytes().decode()
        return "N" * pad_l + out + "N" * pad_r


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = download(os.path.join(root, "data", "genome", "mm10.2bit"))
    tb = TwoBit(p)
    print("sequences:", len(tb.idx))
    # sanity: compare against the UCSC API for one interval
    s = tb.fetch("chr1", 100_000_000, 100_000_060)
    print("local :", s)
    try:
        with urllib.request.urlopen(
                "https://api.genome.ucsc.edu/getData/sequence?genome=mm10;chrom=chr1"
                ";start=100000000;end=100000060", timeout=60, context=CTX) as r:
            api = json.loads(r.read())["dna"]
        print("api   :", api)
        print("match :", s == api)
    except Exception as e:
        print("api check skipped:", e)
