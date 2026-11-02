"""Score any tool's annotated Fiber-seq BAM (fibertools `Ma`, FiberHMM `MA`) on a region.

    python -I bench/score_annotated_bam.py ANNOTATED.bam FIRE_PEAKS.bed REF.fa CHROM START END OUT.json [label] [DNASE_PEAKS.bed[.gz]]

Same checks as realdata_gm12878.py so tools are scored identically:
nucleosome length and repeat, back-to-back nucleosomes, FIRE-peak AUROC of per-base
accessibility (not inside a nucleosome or footprint), footprint enrichment in FIRE peaks,
nucleosome occupancy in GC-rich DNA, and an orientation sanity check (m6A inside calls).
"""
import json
import os
import sys

import numpy as np
import pysam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.ftparse import ma_frame, parse_annotations  # noqa: E402
from bench.metrics import auroc  # noqa: E402
from fiberstate.io import fiber_arrays  # noqa: E402


LONG_PATCH = 100


def mode(x, a, b, step=2):
    x = [v for v in x if a <= v <= b]
    h, e = np.histogram(x, bins=np.arange(a, b + 1, step))
    return float(e[h.argmax()] + step / 2 - 1) if len(x) else float("nan")


def read_intervals(p, chrom):
    import gzip
    op = gzip.open if p.endswith(".gz") else open
    out = []
    with op(p, "rt") as f:
        for line in f:
            if line.startswith(("#", "track", "browser")):
                continue
            c = line.split("\t")
            if c[0] == chrom:
                out.append((int(c[1]), int(c[2])))
    return out


def main(bam_p, peaks_p, ref_p, chrom, start, end, out_p, label="tool", dnase_p=None):
    lo, hi = int(start), int(end)
    n = hi - lo
    bam = pysam.AlignmentFile(bam_p)
    frame = ma_frame(bam.header)
    cov = np.zeros(n); nucc = np.zeros(n); tfc = np.zeros(n); m6 = np.zeros(n); at = np.zeros(n)
    longp = np.zeros(n)     # molecules where the base sits in an open stretch >= LONG_PATCH bp
    nuc_len, spacing, gaps, foot_centers = [], [], [], []
    m6_in_nuc = [0, 0]
    nf = 0
    seen = set()
    for r in bam.fetch(chrom, lo, hi):
        if r.is_secondary or r.is_supplementary or r.query_name in seen:
            continue
        seen.add(r.query_name)
        nf += 1
        s, m = fiber_arrays(r)
        ann = parse_annotations(r, features=("nuc", "tf", "unr"), MA_frame=frame)
        L = len(s)
        q2r = np.full(L, -1, dtype=np.int64)
        for qp, rp in r.get_aligned_pairs(matches_only=True):
            q2r[qp] = rp
        ok = (q2r >= lo) & (q2r < hi)
        idx = q2r[ok] - lo
        nm = np.zeros(L, bool); tm = np.zeros(L, bool)
        for a, b in ann["nuc"] + ann["unr"]:          # unresolved protected stretches count as protected
            nm[max(0, a):min(L, b)] = True
        for a, b in ann["tf"]:
            tm[max(0, a):min(L, b)] = True
            c = q2r[min(L - 1, (a + b) // 2)]
            if lo <= c < hi:
                foot_centers.append(int(c))
        isat = np.isin(s, [65, 84])
        cov[idx] += 1; nucc[idx] += nm[ok]; tfc[idx] += tm[ok]
        prot = sorted(ann["nuc"] + ann["unr"] + ann["tf"])
        lp = np.zeros(L, bool)
        prev = 0
        for a_, b_ in prot + [(L, L)]:
            if a_ - prev >= LONG_PATCH:
                lp[prev:a_] = True
            prev = max(prev, b_)
        longp[idx] += lp[ok]
        at[idx] += isat[ok]; m6[idx] += ((m > 0) & isat)[ok]
        m6_in_nuc[0] += int(((m > 0) & isat & nm).sum()); m6_in_nuc[1] += int((isat & nm).sum())
        nucs = sorted(ann["nuc"])
        nuc_len += [b - a for a, b in nucs]
        spacing += [nucs[i + 1][0] - nucs[i][0] for i in range(len(nucs) - 1)]
        gaps += [nucs[i + 1][0] - nucs[i][1] for i in range(len(nucs) - 1)]
    peaks = []
    with open(peaks_p) as f:
        for line in f:
            if not line.startswith("#"):
                p = line.split("\t")
                if float(p[20]) <= 0.05:
                    peaks.append((int(p[1]), int(p[2])))
    pm = np.zeros(n, bool)
    for a, b in peaks:
        pm[max(0, a - lo):max(0, min(hi, b) - lo)] = True
    good = cov >= 20
    c1 = np.maximum(cov, 1)
    acc = 1 - (nucc + tfc) / c1
    B = 100; nb = n // B
    binned = lambda x: x[:nb * B].reshape(nb, B).mean(1)  # noqa: E731
    lab = binned(pm.astype(float)) >= 0.5
    gb = binned(good.astype(float)) >= 0.9
    fa = pysam.FastaFile(ref_p)
    seq = np.frombuffer(fa.fetch(chrom, lo, hi).upper().encode(), dtype=np.uint8)
    gc200 = np.convolve(np.isin(seq, [67, 71]).astype(float), np.ones(200) / 200, mode="same")
    rich = (gc200 >= 0.65) & good
    occ = lambda mk: float((nucc[mk] / c1[mk]).mean()) if mk.any() else None  # noqa: E731
    fc = np.array(foot_centers, dtype=int)
    in_pk = pm[fc - lo] if len(fc) else np.array([], bool)
    cov_p = cov[pm & good].sum(); cov_b = cov[~pm & good].sum()
    rate_p = in_pk.sum() / max(1, cov_p) * 1e3; rate_b = (~in_pk).sum() / max(1, cov_b) * 1e3
    g = np.array(gaps)
    res = dict(
        label=label, MA_frame=frame, n_fibers=nf,
        m6a_per_AT_inside_nucleosomes=m6_in_nuc[0] / max(1, m6_in_nuc[1]),
        nuc_len_mode=mode(nuc_len, 60, 260), nuc_len_median=float(np.median(nuc_len)) if nuc_len else None,
        n_nucleosomes=len(nuc_len), repeat_mode=mode(spacing, 100, 400),
        repeat_median=float(np.median([x for x in spacing if 100 <= x <= 400])) if spacing else None,
        back_to_back=float((g < 10).mean()) if len(g) else None,
        fire_auroc=auroc(binned(acc)[gb], lab[gb]),
        n_footprints=int(len(fc)), footprint_frac_in_peaks=float(in_pk.mean()) if len(fc) else None,
        footprint_enrichment=float(rate_p / max(1e-12, rate_b)) if len(fc) else None,
        nuc_occ_gc_rich_in_peaks=occ(rich & pm), nuc_occ_normal_outside=occ(~rich & ~pm & good),
    )
    if dnase_p:
        dm = np.zeros(n, bool)
        for a_, b_ in read_intervals(dnase_p, chrom):
            dm[max(0, a_ - lo):max(0, min(hi, b_) - lo)] = True
        dlab = binned(dm.astype(float)) >= 0.5
        res["dnase_auroc"] = auroc(binned(acc)[gb], dlab[gb])
        res["dnase_bins_positive"] = int((dlab & gb).sum())
        res["dnase_auroc_raw_m6a"] = auroc(binned(m6 / np.maximum(at, 1))[gb], dlab[gb])
        res["dnase_auroc_long_patches"] = auroc(binned(longp / c1)[gb], dlab[gb])
    json.dump(res, open(out_p, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
