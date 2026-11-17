"""32b - Minimal bigWig reader, so a dataset that only deposits bigWig can still be
        gated on Tier C.

Why this exists
---------------
scripts/32_alt_chip.py found nine AHR ChIP-seq / CUT&Tag series for mouse or human
and only two deposit called peaks: GSE276152 (fails its own positive control) and
GSE205502 (which deposits a *differential* peak set, 1,984 peaks - not an occupancy
map). The rest deposit bigWig only, and this machine has no isPcr/BLAST/minimap2
and no pyBigWig wheel for Python 3.13.

So the bigWig format is read directly. Only two operations are needed:
  * the chromosome B+ tree  -> chromosome name/id/size
  * the cirTree index       -> the data blocks for a region
  * the data blocks         -> bedGraph / variableStep / fixedStep items, zlib
                               decompressed when the header asks for it
Signal is summarised as the coverage-weighted mean over a window, which is what a
peak call would threshold on anyway.

This is a reader, not a peak caller. Tier C therefore becomes:
  "is the mean AHR signal over the six canonical TSS windows above the 95th
   percentile of 2,000 random TSS windows?"
which is a weaker statement than a peak call and is labelled as such.
"""
import os, sys, zlib, struct
import numpy as np

BIGWIG_MAGIC = 0x888FFC26


class BigWig:
    def __init__(self, path):
        self.f = open(path, "rb")
        hdr = self.f.read(64)
        magic, ver, nzoom, self.chrom_tree_off, self.full_data_off, \
            self.full_index_off, nfields, defined, autosql, totalsum, \
            self.uncompress, reserved = struct.unpack("<IHHQQQHHQQIQ", hdr)
        if magic != BIGWIG_MAGIC:
            raise ValueError("not a bigWig file")
        self.chroms = {}          # name -> (id, size)
        self.id2chrom = {}
        self._read_chrom_tree()
        self.blocks = self._read_index()
        self._build_region_index()

    # ------------------------------------------------------------------ B+ tree
    def _read_chrom_tree(self):
        f = self.f
        f.seek(self.chrom_tree_off)
        magic, block_size, key_size, val_size, item_count, reserved = \
            struct.unpack("<IIIIQQ", f.read(32))
        self.key_size = key_size

        def walk(off):
            f.seek(off)
            is_leaf, _, count = struct.unpack("<BBH", f.read(4))
            if is_leaf:
                for _ in range(count):
                    key = f.read(key_size).rstrip(b"\x00").decode("latin-1")
                    cid, csize = struct.unpack("<II", f.read(8))
                    self.chroms[key] = (cid, csize)
                    self.id2chrom[cid] = key
            else:
                children = []
                for _ in range(count):
                    f.read(key_size)
                    children.append(struct.unpack("<Q", f.read(8))[0])
                for c in children:
                    walk(c)

        walk(self.chrom_tree_off + 32)   # root node follows the 32-byte header

    # ------------------------------------------------------------------ cirTree
    def _read_index(self):
        f = self.f
        f.seek(self.full_index_off)
        magic, block_size, item_count, sc, sb, ec, eb, end_off, items_per_slot, res = \
            struct.unpack("<IIQIIIIQII", f.read(48))
        out = []

        def walk(off):
            f.seek(off)
            is_leaf, _, count = struct.unpack("<BBH", f.read(4))
            if is_leaf:
                for _ in range(count):
                    sc, sb, ec, eb, doff, dsize = struct.unpack("<IIIIQQ", f.read(32))
                    out.append((sc, sb, ec, eb, doff, dsize))
            else:
                children = []
                for _ in range(count):
                    f.read(16)
                    children.append(struct.unpack("<Q", f.read(8))[0])
                for c in children:
                    walk(c)

        walk(self.full_index_off + 48)
        return out

    # ------------------------------------------------------------ region index
    def _build_region_index(self):
        """Per-chromosome, start-sorted block arrays.

        The index leaves are non-overlapping intervals, so for a query window the
        overlapping blocks are exactly the slice between two bisections. Without
        this, every window walks the whole leaf list - 112k blocks for the HCASMC
        TCDD track - which is why the first attempt at 35 died part-way through.
        """
        if not self.blocks:
            self.ridx = {}
            return
        arr = np.array([(a, b, c, d, e, f) for a, b, c, d, e, f in self.blocks],
                       dtype="int64")
        sc, sb, ec, eb, off, size = (arr[:, i] for i in range(6))
        self.ridx = {}
        for cid in np.unique(np.concatenate([sc, ec])):
            m = (sc <= cid) & (ec >= cid)
            if not m.any():
                continue
            csb = sb[m]
            order = np.argsort(csb, kind="stable")
            self.ridx[int(cid)] = (csb[order], eb[m][order],
                                   off[m][order], size[m][order])
        self._max_span = int((eb - sb).max()) if len(arr) else 0

    def _blocks_for(self, cid, start, end):
        """Slice of block arrays that can overlap [start, end)."""
        rec = self.ridx.get(int(cid))
        if rec is None:
            return None
        csb, ceb, coff, csize = rec
        lo = int(np.searchsorted(csb, start - self._max_span, side="left"))
        hi = int(np.searchsorted(csb, end, side="left"))
        if hi <= lo:
            return None
        return csb[lo:hi], ceb[lo:hi], coff[lo:hi], csize[lo:hi]

    # ------------------------------------------------------------------ blocks
    def _parse_block(self, data):
        """Yield (chromId, start, end, value) for the sections in one block."""
        p = 0
        n = len(data)
        while p + 24 <= n:
            cid, cstart, cend, step, span, typ, _, icount = struct.unpack(
                "<IIIIIBBH", data[p:p + 24])
            p += 24
            if typ == 1:                      # bedGraph
                need = icount * 12
                if p + need > n:
                    return
                # bedGraph items are (start u32, end u32, value f32) = 12 bytes
                raw = np.frombuffer(data[p:p + need], dtype=np.uint8).reshape(-1, 12)
                st = raw[:, :4].copy().view("<u4").ravel()
                en = raw[:, 4:8].copy().view("<u4").ravel()
                va = raw[:, 8:].copy().view("<f4").ravel()
                p += need
                for s, e, v in zip(st, en, va):
                    yield int(cid), int(s), int(e), float(v)
            elif typ == 2:                    # variableStep
                need = icount * 8
                if p + need > n:
                    return
                raw = np.frombuffer(data[p:p + need], dtype=np.uint8).reshape(-1, 8)
                st = raw[:, :4].copy().view("<u4").ravel()
                va = raw[:, 4:].copy().view("<f4").ravel()
                p += need
                for s, v in zip(st, va):
                    yield int(cid), int(s), int(s) + int(span), float(v)
            elif typ == 3:                    # fixedStep
                need = icount * 4
                if p + need > n:
                    return
                va = np.frombuffer(data[p:p + need], dtype="<f4")
                p += need
                for i, v in enumerate(va):
                    s = int(cstart) + i * int(step)
                    yield int(cid), s, s + int(span), float(v)
            else:
                return

    def _items(self, chrom, start, end):
        """All (s, e, v) items overlapping [start, end) for one chromosome."""
        if chrom not in self.chroms:
            return
        cid = self.chroms[chrom][0]
        rec = self._blocks_for(cid, start, end)
        if rec is None:
            return
        csb, ceb, coff, csize = rec
        f = self.f
        for sb_, eb_, off_, size_ in zip(csb, ceb, coff, csize):
            if eb_ <= start or sb_ >= end:
                continue
            f.seek(int(off_))
            raw = f.read(int(size_))
            if self.uncompress:
                try:
                    raw = zlib.decompress(raw)
                except zlib.error:
                    continue
            for c, s, e, v in self._parse_block(raw):
                if c != cid or e <= start or s >= end:
                    continue
                yield s, e, v

    def window_mean(self, chrom, start, end):
        """Coverage-weighted mean signal over [start, end). NaN if no data."""
        if not np.isfinite(start) or not np.isfinite(end):
            return np.nan
        num = den = 0.0
        for s, e, v in self._items(chrom, int(start), int(end)):
            a, b = max(s, start), min(e, end)
            if b > a:
                num += v * (b - a)
                den += (b - a)
        return num / den if den > 0 else np.nan

    def window_stats(self, chrom, start, end, tile=500):
        """One pass over the window -> (mean, max tile mean, covered bp fraction).

        The tile maximum is a summit proxy: a real AHR peak is ~300-600 bp, and
        diluting it into a 5-kb mean is exactly the failure mode that made the
        published narrowPeak set look empty at AHRR.
        """
        start, end = int(start), int(end)
        width = end - start
        if width <= 0:
            return (np.nan, np.nan, 0.0)
        n_t = (width + tile - 1) // tile
        t_num = np.zeros(n_t)
        t_den = np.zeros(n_t)
        num = den = 0.0
        for s, e, v in self._items(chrom, start, end):
            a, b = max(s, start), min(e, end)
            if b <= a:
                continue
            num += v * (b - a)
            den += (b - a)
            i0 = (a - start) // tile
            i1 = (b - 1 - start) // tile
            for i in range(i0, i1 + 1):
                ta = max(a, start + i * tile)
                tb = min(b, start + (i + 1) * tile)
                if tb > ta:
                    t_num[i] += v * (tb - ta)
                    t_den[i] += (tb - ta)
        mean = num / den if den > 0 else np.nan
        cov = den / width
        with np.errstate(invalid="ignore", divide="ignore"):
            tm = np.where(t_den > 0, t_num / np.maximum(t_den, 1), np.nan)
        summit = np.nanmax(tm) if np.isfinite(tm).any() else np.nan
        return mean, summit, cov


if __name__ == "__main__":
    p = sys.argv[1]
    bw = BigWig(p)
    print(len(bw.chroms), "chromosomes,", len(bw.blocks), "index blocks")
    for c in ["chr1", "chr7", "chr15"]:
        if c in bw.chroms:
            print(c, bw.chroms[c], "mean chr2:2.0-2.1Mb =",
                  round(bw.window_mean(c, 2_000_000, 2_100_000), 4))
