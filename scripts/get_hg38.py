"""Ranged download of hg38.2bit (790 MB) with 8 connections.

Same trick as lib_seq.download (which is mm10-hardcoded): UCSC throughput per
connection is poor (~300-600 kb/s from this host) but 8 in parallel gives ~5 MB/s,
and a 2bit file is randomly addressable so a failed chunk costs nothing.
"""
import os, sys, time, ssl, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit"
OUT = os.path.join(ROOT, "data", "genome", "hg38.2bit")
CTX = ssl.create_default_context()
CHUNK = 8 * 1024 * 1024


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT) and os.path.getsize(OUT) > 700 * 1024 * 1024:
        print("already present", os.path.getsize(OUT) / 1e6, "MB")
        return
    with urllib.request.urlopen(
            urllib.request.Request(URL, method="HEAD", headers={"User-Agent": "Mozilla/5.0"}),
            timeout=60, context=CTX) as r:
        total = int(r.headers["Content-Length"])
    print(f"size {total/1e6:.0f} MB", flush=True)
    parts = [None] * ((total + CHUNK - 1) // CHUNK)

    def get(i):
        s, e = i * CHUNK, min((i + 1) * CHUNK, total) - 1
        for k in range(5):
            try:
                req = urllib.request.Request(
                    URL, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={s}-{e}"})
                with urllib.request.urlopen(req, timeout=240, context=CTX) as r:
                    buf = bytearray()
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        buf += b
                if len(buf) == e - s + 1:
                    return bytes(buf)
                print(f"  short chunk {i}: {len(buf)}/{e-s+1} (try {k+1})", flush=True)
            except Exception as ex:
                print(f"  chunk {i} err {ex} (try {k+1})", flush=True)
                time.sleep(2)
        raise RuntimeError(f"chunk {i} failed")

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(8) as ex:
        for i, b in enumerate(ex.map(get, range(len(parts)))):
            parts[i] = b
            done += 1
            if done % 10 == 0 or done == len(parts):
                el = time.time() - t0
                print(f"  {done}/{len(parts)} chunks  {el:.0f}s  "
                      f"{done*CHUNK/1e6/max(el,1):.1f} MB/s", flush=True)
    with open(OUT, "wb") as f:
        for b in parts:
            f.write(b)
    print(f"downloaded {total/1e6:.0f} MB in {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
