"""Streaming UCSC chain reader for point liftover.

The mm10->hg38 chain has 83k chains and 31M block lines. Materialising every
block costs many GB, so this module never stores the chain: it sorts the query
points, then makes one streaming pass over the blocks and resolves any query
that falls inside the current block. Memory is O(number of queries).

API
---
liftover_points(chain_path, points)
    points : list of (chrom, pos) in the source assembly
    returns: list of (chrom, pos) in the target assembly, or None per point
"""
from __future__ import annotations

import bisect
import gzip


def liftover_points(chain_path, points):
    order = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    out = [None] * len(points)

    # per source chromosome: sorted positions plus the original indices
    by_chrom = {}
    for i in order:
        c, p = points[i]
        by_chrom.setdefault(c, ([], []))
        by_chrom[c][0].append(p)
        by_chrom[c][1].append(i)

    opener = gzip.open if chain_path.endswith(".gz") else open
    t_name = None
    t_cur = q_cur = q_size = 0
    q_name = None
    fwd = True
    with opener(chain_path, "rt") as fh:
        for line in fh:
            if line.startswith("chain"):
                p = line.split()
                t_name = p[2]
                t_cur = int(p[5])
                q_name, q_size, q_cur = p[7], int(p[8]), int(p[10])
                fwd = p[9] == "+"
                continue
            if t_name is None or t_name not in by_chrom:
                # still need to keep the coordinates in sync for this chain
                pass
            parts = line.split()
            if not parts:
                continue
            size = int(parts[0])
            qs, qi = by_chrom.get(t_name, (None, None))
            if qs:
                lo = bisect.bisect_left(qs, t_cur)
                hi = bisect.bisect_left(qs, t_cur + size)
                if hi > lo:
                    if fwd:
                        q_plus = q_cur
                    else:
                        q_plus = q_size - (q_cur + size)
                    for k in range(lo, hi):
                        idx = qi[k]
                        if out[idx] is not None:
                            continue
                        off = qs[k] - t_cur
                        out[idx] = (q_name,
                                    q_plus + off if fwd else q_plus + (size - 1 - off))
            t_cur += size
            q_cur += size
            if len(parts) == 3:
                t_cur += int(parts[1])
                q_cur += int(parts[2])
    return out


if __name__ == "__main__":
    import os
    import time
    p = os.path.expanduser("~/.pyliftover/mm10ToHg38.over.chain.gz")
    pts = [("chr1", 3115000), ("chr7", 17298651), ("chr11", 96000000),
           ("chr11", 96005000), ("chr5", 122000000)]
    t = time.time()
    r = liftover_points(p, pts)
    print(f"{time.time()-t:.1f}s")
    for a, b in zip(pts, r):
        print(a, "->", b)
