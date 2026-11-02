"""Pick TF motif sites on one or more chromosomes and label them bound / unbound by ChIP-seq.

    python select_sites.py REF.fa CHROM[,CHROM...] CHIP_PEAKS.bed.gz OUT_PREFIX [n_per_class=250] [seed=1] [MOTIF.jaspar]

* Scans both strands with a JASPAR matrix (default MA0139.2, CTCF, 15 bp) and keeps strong matches
  (relative score >= 0.85, the JASPAR convention).
* bound   = motif centre inside an IDR ChIP peak, within 100 bp of the summit.
* unbound = no ChIP peak within 2 kb of the motif.
* Sites closer than 2 kb to each other are thinned so molecules are not shared.
* Unbound sites are drawn to match the motif-score distribution of bound sites, so a
  tool cannot score well just by preferring stronger motifs.
Writes OUT_PREFIX.sites.bed (with labels) and OUT_PREFIX.regions.bed (site +/- 100 bp, for samtools -L).
"""
import bisect
import gzip
import sys

import numpy as np
import pysam

# JASPAR MA0139.2 position frequency matrix (A, C, G, T rows)
PFM = np.array([
    [281, 56, 8, 744, 40, 107, 851, 5, 333, 54, 12, 56, 104, 372, 82],
    [49, 800, 903, 13, 528, 433, 11, 0, 3, 12, 0, 8, 733, 13, 482],
    [449, 21, 0, 65, 334, 48, 32, 903, 566, 504, 890, 775, 5, 507, 307],
    [134, 36, 2, 91, 11, 324, 18, 3, 9, 341, 8, 71, 67, 17, 37],
], dtype=float)


def read_jaspar(path):
    """JASPAR-format matrix file (">ID NAME" then rows 'A [ ... ]') -> 4 x L counts."""
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        vals = line.split("[")[1].split("]")[0].split() if "[" in line else line.split()[1:]
        rows.append([float(v) for v in vals])
    assert len(rows) == 4, f"{path}: expected 4 rows (A C G T), got {len(rows)}"
    return np.array(rows)


def pwm(pfm=None):
    pfm = PFM if pfm is None else pfm
    p = (pfm + 0.8 * 0.25) / (pfm.sum(0) + 0.8)
    return np.log2(p / 0.25)


def scan(seq, W, rel=0.85):
    L = W.shape[1]
    code = np.full(len(seq), 4, dtype=np.int8)
    for i, b in enumerate(b"ACGT"):
        code[seq == b] = i
    n = len(seq) - L + 1
    W5 = np.vstack([W, np.full((1, L), -99.0)])            # N -> never matches
    Wrc = W[::-1, ::-1]                                    # reverse complement
    Wrc5 = np.vstack([Wrc, np.full((1, L), -99.0)])
    sf = np.zeros(n); sr = np.zeros(n)
    for j in range(L):
        c = code[j:j + n]
        sf += W5[c, j]
        sr += Wrc5[c, j]
    mn, mx = W.min(0).sum(), W.max(0).sum()
    thr = mn + rel * (mx - mn)
    out = []
    for strand, sc in (("+", sf), ("-", sr)):
        idx = np.flatnonzero(sc >= thr)
        out += [(int(i), int(i + L), float((sc[i] - mn) / (mx - mn)), strand) for i in idx]
    return sorted(out)


def read_peaks(path, chrom):
    op = gzip.open if path.endswith(".gz") else open
    pk = []
    with op(path, "rt") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] != chrom:
                continue
            s, e = int(p[1]), int(p[2])
            summit = s + int(p[9]) if len(p) > 9 and p[9] not in (".", "-1") else (s + e) // 2
            pk.append((s, e, summit, float(p[6]) if len(p) > 6 else 0.0))
    pk.sort()
    return pk


def classify(fa, chrom, W, peaks_p):
    seq = np.frombuffer(fa.fetch(chrom).upper().encode(), dtype=np.uint8)
    hits = scan(seq, W)
    pk = read_peaks(peaks_p, chrom)
    starts = np.array([p[0] for p in pk]); ends = np.array([p[1] for p in pk])
    summits = np.array([p[2] for p in pk]); sig = np.array([p[3] for p in pk])
    order = np.argsort(starts, kind="stable")
    starts, ends, summits, sig = starts[order], ends[order], summits[order], sig[order]
    maxend = np.maximum.accumulate(ends) if len(pk) else ends   # running max of peak ends, by start
    bound, unbound = [], []
    for s, e, sc, strand in hits:
        c = (s + e) // 2
        i = np.searchsorted(starts, c, side="right") - 1 if len(pk) else -1
        inside = i >= 0 and starts[i] <= c < ends[i]
        j = np.searchsorted(starts, c + 2000, side="right") - 1 if len(pk) else -1
        near = bool(j >= 0 and maxend[j] + 2000 > c)   # some peak with start-2000 <= c < end+2000
        if inside and abs(summits[i] - c) <= 100:
            bound.append((chrom, s, e, sc, strand, float(sig[i])))
        elif not near:
            unbound.append((chrom, s, e, sc, strand, 0.0))
    return bound, unbound, len(hits)


def main(ref, chroms, peaks_p, prefix, n=250, seed=1, motif=None):
    n = int(n); rng = np.random.default_rng(int(seed))
    W = pwm(read_jaspar(motif) if motif else None)
    fa = pysam.FastaFile(ref)
    bound, unbound, n_hits = [], [], 0
    for chrom in chroms.split(","):
        b, u, h = classify(fa, chrom, W, peaks_p)
        bound += b; unbound += u; n_hits += h
    sc_of = lambda x: x[3]  # noqa: E731
    mid = lambda x: (x[1] + x[2]) // 2  # noqa: E731

    def thin(sites):
        sites = sorted(sites, key=lambda x: -sc_of(x))     # keep the stronger motif when two are close
        kept, taken = [], {}
        for x in sites:
            c = mid(x); t = taken.setdefault(x[0], [])   # sorted centres already kept
            k = bisect.bisect_left(t, c)
            if (k == len(t) or t[k] - c >= 2000) and (k == 0 or c - t[k - 1] >= 2000):
                kept.append(x); t.insert(k, c)
        return kept
    bound, unbound = thin(bound), thin(unbound)
    # also thin across classes
    bc = {}
    for x in bound:
        bc.setdefault(x[0], []).append(mid(x))
    bc = {k: np.array(v) for k, v in bc.items()}
    unbound = [x for x in unbound if x[0] not in bc or np.min(np.abs(bc[x[0]] - mid(x))) >= 2000]
    b_sel = [bound[i] for i in rng.choice(len(bound), size=min(n, len(bound)), replace=False)]
    # score-matched unbound: sample within motif-score bins of the chosen bound sites
    bins = np.linspace(0.85, 1.0, 7)
    u_sc = np.array([sc_of(x) for x in unbound])
    u_sel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        need = sum(1 for x in b_sel if lo <= sc_of(x) < hi or (hi == 1.0 and sc_of(x) == 1.0))
        pool = np.flatnonzero((u_sc >= lo) & ((u_sc < hi) | ((hi == 1.0) & (u_sc == 1.0))))
        take = rng.choice(pool, size=min(need, len(pool)), replace=False) if len(pool) else []
        u_sel += [unbound[i] for i in take]
    with open(prefix + ".sites.bed", "w") as f:
        for lab, sel in (("bound", b_sel), ("unbound", u_sel)):
            for k, (chrom, s, e, sc, strand, sg) in enumerate(sorted(sel)):
                f.write(f"{chrom}\t{s}\t{e}\t{lab}_{k}\t{sc:.3f}\t{strand}\t{lab}\t{sg:.2f}\n")
    with open(prefix + ".regions.bed", "w") as f:
        for chrom, s, e, *_ in sorted(b_sel + u_sel):
            f.write(f"{chrom}\t{max(0, s - 100)}\t{e + 100}\n")
    print(f"motif hits {n_hits}; bound {len(bound)}, unbound {len(unbound)} after thinning; "
          f"selected {len(b_sel)} bound, {len(u_sel)} unbound (score-matched)")
    print("mean motif score bound %.3f unbound %.3f" % (np.mean([sc_of(x) for x in b_sel]), np.mean([sc_of(x) for x in u_sel])))


if __name__ == "__main__":
    main(*sys.argv[1:])
