"""Inspect fibers where fiberstate and ft footprint disagree at CTCF motifs.

    python -I bench/realdata_disagree.py CTCF.bam MOTIFS.bed.gz FT_FOOTPRINTS.bed.gz OUT.json
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pysam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.realdata import read_bed  # noqa: E402
from fiberstate.io import fiber_arrays  # noqa: E402
from fiberstate.context import ContextTable  # noqa: E402
_CTX = ContextTable.load(os.environ["FS_CONTEXT"]) if os.environ.get("FS_CONTEXT") else None
from fiberstate.model import Model, Params  # noqa: E402
FRAC_NUC = float(os.environ.get('FS_FRAC_NUC', '0.75'))


def _call(a):
    n, s, m = a
    c = Model(Params(frac_nuc=FRAC_NUC), context=_CTX).call_read(s, m)
    return n, c.post_linker, c.post_nuc, c.post_foot, c.segments


def main(bam_p, motif_p, fp_p, out_p):
    bam = pysam.AlignmentFile(bam_p)
    motifs = read_bed(motif_p)
    ftfp = {(r[0], int(r[1])): r for r in read_bed(fp_p)}
    jobs, meta = {}, []
    for mf in motifs:
        chrom, s, e, name = mf[0], int(mf[1]), int(mf[2]), mf[3]
        mstrand = mf[5] if len(mf) > 5 else "+"
        row = ftfp.get((chrom, s))
        codes = dict(zip(row[-1].split(","), map(int, row[-3].split(",")))) if row else {}
        for r in bam.fetch(chrom, s, e):
            if r.is_secondary or r.is_supplementary or r.query_name not in codes:
                continue
            r2q = {rp: qp for qp, rp in r.get_aligned_pairs(matches_only=True)}
            qs = [r2q.get(p) for p in range(s, e)]
            qs = [q for q in qs if q is not None]
            if not qs:
                continue
            seq, m6a = fiber_arrays(r)
            jobs[r.query_name] = (r.query_name, seq, m6a)
            meta.append((name, r.query_name, min(qs), max(qs) + 1, mstrand, codes[r.query_name]))
    with ProcessPoolExecutor(2) as ex:
        calls = {c[0]: c for c in ex.map(_call, list(jobs.values()), chunksize=8)}
    rows = []
    for name, fb, qa, qb, mstrand, code in meta:
        _, s, m = jobs[fb]
        _, gl, gn, gf, segs = calls[fb]
        qc = (qa + qb) // 2
        at = np.isin(s[qa:qb], [65, 84])
        n_m6a = int((m[qa:qb] > 0).sum())
        # m6A in the 100 bp either side (is the motif inside a long protected stretch?)
        lo, hi = max(0, qa - 100), min(len(s), qb + 100)
        flank_rate = float((m[lo:hi] > 0).sum() / max(1, np.isin(s[lo:hi], [65, 84]).sum()))
        r_is_rev = False
        seg = next(((k, a, b, q) for k, a, b, q in segs if a <= qc < b), None)
        # modules 1b-4 (motif bp 9-35) in motif orientation; reads are reference-oriented
        ca, cb = (qa + 9, qb) if mstrand == "+" else (qa, qb - 9)
        core = (gn[ca:cb] + gf[ca:cb]) > 0.5
        n_m6a_core = int((m[ca:cb] > 0).sum())
        n_m6a_outer = n_m6a - n_m6a_core
        fs_rule = os.environ.get("FS_RULE", "center")
        fs_prot = bool(core.mean() >= 0.9) if fs_rule == "core" else float(gn[qc] + gf[qc]) > 0.5
        rows.append(dict(site=name, ft_prot=(code & 0b1111100) == 0b1111100,
                         fs_prot=fs_prot, fs_state=int(np.argmax([gl[qc], gn[qc], gf[qc]])),
                         n_m6a_motif=n_m6a, n_m6a_core=n_m6a_core, n_m6a_outer=n_m6a_outer,
                         strand_minus=bool(r_is_rev), n_at_motif=int(at.sum()), flank_rate=flank_rate,
                         seg_len=(seg[2] - seg[1]) if seg else 0, seg_conf=seg[3] if seg else None))
    groups = {}
    for r in rows:
        key = ("both_protected" if r["ft_prot"] and r["fs_prot"] else "ft_only" if r["ft_prot"]
               else "fs_only" if r["fs_prot"] else "both_open")
        groups.setdefault(key, []).append(r)
    summ = {}
    for k, g in groups.items():
        nm = np.array([r["n_m6a_motif"] for r in g])
        nc = np.array([r["n_m6a_core"] for r in g])
        summ[k] = dict(n=len(g), m6a_in_core_dist={str(i): int((nc == i).sum()) for i in range(3)} | {"3+": int((nc >= 3).sum())},
                       m6a_in_motif_dist={str(i): int((nm == i).sum()) for i in range(4)} | {"4+": int((nm >= 4).sum())},
                       median_flank_m6a_rate=float(np.median([r["flank_rate"] for r in g])),
                       fs_state={"linker": sum(r["fs_state"] == 0 for r in g), "nucleosome": sum(r["fs_state"] == 1 for r in g),
                                 "footprint": sum(r["fs_state"] == 2 for r in g)},
                       median_seg_len=float(np.median([r["seg_len"] for r in g])),
                       median_seg_conf=float(np.median([r["seg_conf"] for r in g if r["seg_conf"] is not None])) if any(r["seg_conf"] is not None for r in g) else None)
    json.dump(summ, open(out_p, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
