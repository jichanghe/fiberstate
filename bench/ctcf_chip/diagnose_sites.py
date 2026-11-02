"""Per-molecule comparison of fiberstate vs FiberHMM / fibertools calls at chosen CTCF sites.

    python -I diagnose_sites.py DIAG_DIR OUT.json

For each site and each molecule spanning it: what fiberstate calls over the motif
(footprint / nucleosome / unresolved / linker, with segment length), whether FiberHMM
calls a footprint there, and the m6A evidence (marks in motif core, distance to the
nearest mark on each side). Summarised by category of disagreement.
"""
import collections
import json
import os
import sys

import numpy as np
import pysam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bench.ftparse import ma_frame, parse_annotations  # noqa: E402
from fiberstate.io import fiber_arrays  # noqa: E402

NAMES = {"nuc": "nucleosome", "tf": "footprint", "unr": "unresolved"}


def calls_at(bam_p, site, feats):
    b = pysam.AlignmentFile(bam_p)
    frame = ma_frame(b.header)
    out = {}
    for r in b.fetch(site["chrom"], site["s"], site["e"]):
        if r.is_secondary or r.is_supplementary or r.reference_start > site["s"] - 200 or r.reference_end < site["e"] + 200:
            continue
        r2q = {rp: qp for qp, rp in r.get_aligned_pairs(matches_only=True) if site["s"] - 200 <= rp < site["e"] + 200}
        qs = [r2q[p] for p in range(site["s"], site["e"]) if p in r2q]
        if len(qs) < 12:
            continue
        qa, qb = min(qs), max(qs) + 1
        ann = parse_annotations(r, features=feats, MA_frame=frame)
        cover = None
        for f in feats:
            for a, e in ann[f]:
                ov = min(e, qb) - max(a, qa)
                if ov >= 0.8 * (qb - qa):
                    cover = (f, e - a, a - qa, e - qb)   # type, length, left overhang (neg), right overhang
                    break
            if cover:
                break
        out[r.query_name] = dict(cover=cover, qa=qa, qb=qb, read=r)
    return out


def main(d, out_p):
    sites = []
    for line in open(os.path.join(d, "sites.bed")):
        c = line.rstrip("\n").split("\t")
        sites.append(dict(chrom=c[0], s=int(c[1]), e=int(c[2]), name=c[3], chip=float(c[7])))
    res = {}
    for st in sites:
        fs = calls_at(os.path.join(d, "ctcf_fs.bam"), st, ("tf", "nuc", "unr"))
        fh = calls_at(os.path.join(d, "ctcf_fh.bam"), st, ("tf", "nuc"))
        rows = collections.Counter()
        lens = collections.defaultdict(list)
        evid = collections.defaultdict(list)
        for name, x in fs.items():
            y = fh.get(name)
            if y is None:
                continue
            fs_t = NAMES[x["cover"][0]] if x["cover"] else "linker"
            fh_t = "footprint" if (y["cover"] and y["cover"][0] == "tf") else ("nucleosome" if y["cover"] else "open")
            rows[(fs_t, fh_t)] += 1
            if x["cover"]:
                lens[(fs_t, fh_t)].append(x["cover"][1])
            # m6A evidence around the motif on this molecule
            s, m = fiber_arrays(x["read"])
            qa, qb = x["qa"], x["qb"]
            marked = np.flatnonzero((m > 0) & np.isin(s, [65, 84]))
            core = int(((marked >= qa) & (marked < qb)).sum())
            left = marked[marked < qa]
            right = marked[marked >= qb]
            dl = int(qa - left.max()) if len(left) else 9999
            dr = int(right.min() - qb + 1) if len(right) else 9999
            evid[(fs_t, fh_t)].append((core, dl, dr))
        summ = {}
        for k, n in rows.most_common():
            e = np.array(evid[k])
            summ[f"fiberstate={k[0]} | FiberHMM={k[1]}"] = dict(
                n=n, median_seg_len=float(np.median(lens[k])) if lens[k] else None,
                median_marks_in_motif=float(np.median(e[:, 0])),
                median_gap_to_mark_left=float(np.median(e[:, 1])), median_gap_to_mark_right=float(np.median(e[:, 2])))
        res[st["name"]] = dict(chip=st["chip"], n_molecules=sum(rows.values()), breakdown=summ)
    json.dump(res, open(out_p, "w"), indent=1)
    for name, v in res.items():
        print(f"\n{name}  ChIP {v['chip']:.0f}  molecules {v['n_molecules']}")
        for k, x in list(v["breakdown"].items())[:6]:
            print(f"  {k:45s} {x['n']:4d}  seg {x['median_seg_len']}  marks-in-motif {x['median_marks_in_motif']:.0f}  "
                  f"gap L/R {x['median_gap_to_mark_left']:.0f}/{x['median_gap_to_mark_right']:.0f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
