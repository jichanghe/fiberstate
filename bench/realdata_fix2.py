"""Fix 2 evaluation on a real Fiber-seq region.

    python -I bench/realdata_fix2.py REGION.bam FIRE_PEAKS.bed FT_BIN CHROM START END OUT.json
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pysam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.realdata_gm12878 import run_ft  # noqa: E402
from fiberstate.context import ContextTable  # noqa: E402
from fiberstate.io import fiber_arrays  # noqa: E402
from fiberstate.model import Model, Params  # noqa: E402

CTX = ContextTable.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "fiberstate", "data", "hia5_k5_gm12878.tsv"))


def _work(args):
    bam_p, names, ftn, chrom, lo, hi, peaks = args
    names = set(names)
    pm = np.zeros(hi - lo, bool)
    for a, b in peaks:
        pm[max(0, a - lo):max(0, min(hi, b) - lo)] = True
    out = dict(gaps_old=[], gaps_new=[], spacing_new=[], spacing_old=[], unres_len=[], unres_in_peak_bp=0,
               unres_bp=0, prot_bp=0, nuc_bp_new=0, unres_m6a=0, unres_at=0, ft_cov_in_unres=0,
               ft_long_in_unres=0, ft_nuc_n_in_unres=0, unres_n=0, est_nucs=0, cov_bp=0, peak_cov_bp=0)
    m_new = Model(Params(), context=CTX)
    m_old = Model(Params(merge_unsupported=False), context=CTX)
    with pysam.AlignmentFile(bam_p) as bam:
        for r in bam.fetch(chrom, lo, hi):
            if r.query_name not in names or r.is_secondary or r.is_supplementary:
                continue
            names.discard(r.query_name)
            s, m = fiber_arrays(r)
            c = m_new.call_read(s, m)
            c0 = m_old.call_read(s, m)
            q2r = np.full(len(s), -1, dtype=np.int64)
            for qp, rp in r.get_aligned_pairs(matches_only=True):
                q2r[qp] = rp
            ok = (q2r >= lo) & (q2r < hi)
            out["cov_bp"] += int(ok.sum())
            out["peak_cov_bp"] += int(pm[q2r[ok] - lo].sum())
            for segs, gk, sk in ((c.segments, "gaps_new", "spacing_new"), (c0.segments, "gaps_old", "spacing_old")):
                prot = sorted(segs, key=lambda x: x[1])
                for a, b in zip(prot, prot[1:]):
                    if a[0] == 1 and b[0] == 1:
                        out[gk].append(b[1] - a[2])
                        out[sk].append(b[1] - a[1])
            ftm = np.zeros(len(s), bool)
            ftsegs = ftn.get(r.query_name, [])
            for a, b in ftsegs:
                ftm[a:b] = True
            isat = np.isin(s, [65, 84])
            for k, a, b, q in c.segments:
                if k in (1, 2, 3):
                    out["prot_bp"] += b - a
                if k == 1:
                    out["nuc_bp_new"] += b - a
                if k != 3:
                    continue
                out["unres_n"] += 1
                out["unres_len"].append(b - a)
                out["unres_bp"] += b - a
                out["est_nucs"] += max(1, int(round((b - a) / 185)))
                rp = q2r[a:b]
                inr = (rp >= lo) & (rp < hi)
                out["unres_in_peak_bp"] += int(pm[rp[inr] - lo].sum())
                out["unres_m6a"] += int(((m[a:b] > 0) & isat[a:b]).sum())
                out["unres_at"] += int(isat[a:b].sum())
                out["ft_cov_in_unres"] += int(ftm[a:b].sum())
                inside = [(x, y) for x, y in ftsegs if x < b and y > a]
                out["ft_nuc_n_in_unres"] += len(inside)
                out["ft_long_in_unres"] += sum(1 for x, y in inside if y - x > 250)
    return out


def main(bam_p, peaks_p, ft, chrom, start, end, out_p):
    lo, hi = int(start), int(end)
    peaks = []
    with open(peaks_p) as f:
        for line in f:
            if not line.startswith("#"):
                p = line.split("\t")
                if float(p[20]) <= 0.05:
                    peaks.append((int(p[1]), int(p[2])))
    with pysam.AlignmentFile(bam_p) as bam:
        names = sorted({r.query_name for r in bam.fetch(chrom, lo, hi) if not (r.is_secondary or r.is_supplementary)})
    ftn, _ = run_ft(ft, bam_p)
    ft_lens = [b - a for v in ftn.values() for a, b in v]
    chunks = [names[i::2] for i in range(2)]
    with ProcessPoolExecutor(2) as ex:
        parts = list(ex.map(_work, [(bam_p, ch, {n: ftn.get(n, []) for n in ch}, chrom, lo, hi, peaks) for ch in chunks]))
    agg = {}
    for p in parts:
        for k, v in p.items():
            agg[k] = agg.get(k, [] if isinstance(v, list) else 0) + v
    def mode(x, a=100, b=400):
        x = [v for v in x if a <= v <= b]
        h, e = np.histogram(x, bins=np.arange(a, b + 1, 2))
        return float(e[h.argmax()] + 1), float(np.median(x))
    go, gn = np.array(agg["gaps_old"]), np.array(agg["gaps_new"])
    res = dict(
        back_to_back_before=float((go < 10).mean()), back_to_back_after=float((gn < 10).mean()),
        n_pairs_before=len(go), n_pairs_after=len(gn),
        repeat_before=mode(agg["spacing_old"]), repeat_after=mode(agg["spacing_new"]),
        unresolved_n=agg["unres_n"], unresolved_frac_of_protected_bp=agg["unres_bp"] / agg["prot_bp"],
        unresolved_len_median=float(np.median(agg["unres_len"])),
        unresolved_len_q90=float(np.quantile(agg["unres_len"], .9)),
        unresolved_est_nucleosomes=agg["est_nucs"],
        unresolved_frac_in_peaks=agg["unres_in_peak_bp"] / max(1, agg["unres_bp"]),
        all_dna_frac_in_peaks=agg["peak_cov_bp"] / agg["cov_bp"],
        unresolved_m6a_per_AT=agg["unres_m6a"] / max(1, agg["unres_at"]),
        ft_coverage_of_unresolved=agg["ft_cov_in_unres"] / max(1, agg["unres_bp"]),
        ft_nucs_per_unresolved=agg["ft_nuc_n_in_unres"] / max(1, agg["unres_n"]),
        ft_frac_long_calls_in_unresolved=agg["ft_long_in_unres"] / max(1, agg["ft_nuc_n_in_unres"]),
        ft_long_calls_overall=float(np.mean(np.array(ft_lens) > 250)),
        ft_repeat=mode([b[0] - a[0] for v in ftn.values() for a, b in zip(v, v[1:])]),
    )
    json.dump(res, open(out_p, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
