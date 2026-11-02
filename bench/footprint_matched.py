"""Footprint enrichment in FIRE peaks at a matched number of calls (top-N by each tool's own score).

    python -I bench/footprint_matched.py FIBERHMM.bam FIBERSTATE.bam FIRE_PEAKS.bed CHROM START END OUT.json
"""
import json
import os
import sys

import numpy as np
import pysam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.ftparse import ma_frame  # noqa: E402


def calls(bam_p, lo, hi, chrom, score_from_aq):
    b = pysam.AlignmentFile(bam_p)
    frame = ma_frame(b.header)
    out = []   # (score, ref_center, length)
    seen = set()
    for r in b.fetch(chrom, lo, hi):
        if r.is_secondary or r.is_supplementary or r.query_name in seen or not r.has_tag("MA"):
            continue
        seen.add(r.query_name)
        L = r.query_length
        q2r = np.full(L, -1, dtype=np.int64)
        for qp, rp in r.get_aligned_pairs(matches_only=True):
            q2r[qp] = rp
        aq = list(r.get_tag("AQ")) if r.has_tag("AQ") else []
        k = 0
        for part in str(r.get_tag("MA")).split(";")[1:]:
            head, data = part.split(":", 1)
            i = next(j for j, ch in enumerate(head) if ch in "+-.")
            name, qspec = head[:i], head[i + 1:]
            for tok in filter(None, data.split(",")):
                s, l = tok.split("-", 1)
                s0 = int(s) - 1; e0 = s0 + int(l)
                q = aq[k] if (score_from_aq and qspec and k < len(aq)) else None
                k += len(qspec)
                if name not in ("tf",):
                    continue
                if frame == "molecular" and r.is_reverse:
                    s0, e0 = L - e0, L - s0
                c = q2r[min(L - 1, max(0, (s0 + e0) // 2))]
                if lo <= c < hi:
                    out.append((q if q is not None else 0, int(c), e0 - s0))
    return out


def main(fh_p, fs_p, peaks_p, chrom, start, end, out_p):
    lo, hi = int(start), int(end)
    pm = np.zeros(hi - lo, bool)
    with open(peaks_p) as f:
        for line in f:
            if not line.startswith("#"):
                p = line.split("\t")
                if float(p[20]) <= 0.05:
                    pm[max(0, int(p[1]) - lo):max(0, min(hi, int(p[2])) - lo)] = True
    frac_peaks = float(pm.mean())
    fh = calls(fh_p, lo, hi, chrom, True)
    fs = calls(fs_p, lo, hi, chrom, False)
    n = len(fs)
    fh_sorted = sorted(fh, key=lambda x: -x[0])
    res = {}
    for label, sel in (("fiberstate_all", fs), ("fiberhmm_all", fh), ("fiberhmm_topN", fh_sorted[:n]),
                       ("fiberhmm_len_ge15_topN", sorted([x for x in fh if x[2] >= 15], key=lambda x: -x[0])[:n])):
        c = np.array([x[1] for x in sel])
        inp = pm[c - lo].mean() if len(c) else float("nan")
        res[label] = dict(n=len(sel), frac_in_peaks=float(inp), enrichment_vs_bp=float(inp / frac_peaks),
                          median_len=float(np.median([x[2] for x in sel])) if sel else None)
    res["fire_peak_bp_fraction"] = frac_peaks
    # TF-sized calls only (15-50 bp), FiberHMM matched to fiberstate's count by its own score
    fs_m = [x for x in fs if 15 <= x[2] <= 50]
    fh_m = sorted([x for x in fh if 15 <= x[2] <= 50], key=lambda x: -x[0])
    enr = lambda sel: float(pm[np.array([x[1] for x in sel]) - lo].mean() / frac_peaks) if sel else float("nan")  # noqa: E731
    res["length_15_50"] = dict(n_fiberstate=len(fs_m), n_fiberhmm=len(fh_m), matched_n=min(len(fs_m), len(fh_m)),
                               fiberstate_enrichment=round(enr(fs_m), 2),
                               fiberhmm_top_enrichment=round(enr(fh_m[:len(fs_m)]), 2),
                               fiberhmm_all_enrichment=round(enr(fh_m), 2))
    json.dump(res, open(out_p, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
