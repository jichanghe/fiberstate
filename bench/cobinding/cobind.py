"""Co-binding on real Fiber-seq: do two TF sites on the same molecule bind together more often
than chance, once shared accessibility is accounted for?

Sites are all strong CTCF, REST and NRF1 motif matches within 1 kb of the sites used in the TF
benchmarks; each is labelled bound / unbound by ChIP-seq of its own factor (bound: within 100 bp
of an IDR peak summit; unbound: no peak within 2 kb). For every pair 60-1000 bp apart that at
least MIN_MOL molecules span, fiberstate's per-molecule posteriors at the two motif centres give
a log odds ratio of co-binding, conditioned on accessibility (fiberstate's test) and unconditioned.

Negative control: pairs of two unbound sites, where no coupling is expected. A calibrated test
gives at most 5% of those pairs p < 0.05. Only testable pairs are analysed: each site bound, and
unbound, on at least 3 effective molecules (fiberstate.sites.co_occupancy 'informative').

    python cobind.py OUT_PREFIX REF.fa --bams a.bam,b.bam --windows w1.bed,w2.bed \
        --tf CTCF:MOTIF.jaspar:PEAKS.bed.gz --tf REST:... --tf NRF1:...
BAMs must come from `fiberstate call --posterior-tags` (tags fp and np).
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pysam

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ctcf_chip"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from select_sites import PFM, pwm, read_jaspar, read_peaks, scan  # noqa: E402
from fiberstate.io import query_to_ref  # noqa: E402
from fiberstate.sites import co_occupancy  # noqa: E402

MIN_MOL = 20
DMIN, DMAX = 60, 1000


def windows(paths, flank=1000):
    iv = {}
    for p in paths:
        for line in open(p):
            c = line.split("\t")
            iv.setdefault(c[0], []).append((max(0, int(c[1]) - flank), int(c[2]) + flank))
    out = {}
    for ch, v in iv.items():
        v.sort(); m = []
        for s, e in v:
            if m and s <= m[-1][1]:
                m[-1] = (m[-1][0], max(m[-1][1], e))
            else:
                m.append((s, e))
        out[ch] = m
    return out


def label(c, pk):
    if not pk:
        return "unbound"
    starts = np.array([p[0] for p in pk]); ends = np.array([p[1] for p in pk]); summ = np.array([p[2] for p in pk])
    i = np.searchsorted(starts, c, side="right") - 1
    if i >= 0 and starts[i] <= c < ends[i] and abs(summ[i] - c) <= 100:
        return "bound"
    near = np.any((starts - 2000 <= c) & (c < ends + 2000))
    return "ambiguous" if near else "unbound"


def find_sites(fa, win, tfs):
    sites = []
    for ch, ivs in win.items():
        pk = {name: read_peaks(peaks, ch) for name, _, peaks in tfs}
        for s0, e0 in ivs:
            seq = np.frombuffer(fa.fetch(ch, s0, e0).upper().encode(), dtype=np.uint8)
            for name, W, _ in tfs:
                for s, e, sc, strand in scan(seq, W):
                    c = s0 + (s + e) // 2
                    sites.append(dict(chrom=ch, s=s0 + s, e=s0 + e, c=c, tf=name, score=sc, strand=strand,
                                      label=label(c, pk[name])))
    # one site per locus: overlapping matches keep the strongest
    sites.sort(key=lambda x: (x["chrom"], x["c"]))
    kept = []
    for x in sites:
        if kept and kept[-1]["chrom"] == x["chrom"] and x["s"] < kept[-1]["e"]:
            if x["score"] > kept[-1]["score"]:
                kept[-1] = x
            continue
        kept.append(x)
    return kept


def posteriors(bams, sites):
    """{site index: {read: (pL, pN, pF)}} at the motif centre, from fp/np tags."""
    obs = {i: {} for i in range(len(sites))}
    by_chrom = {}
    for i, st in enumerate(sites):
        by_chrom.setdefault(st["chrom"], []).append(i)
    for bp in bams:
        bam = pysam.AlignmentFile(bp)
        seen = set()
        for ch, idx in by_chrom.items():
            cs = np.array([sites[i]["c"] for i in idx])
            for st_i in idx:
                st = sites[st_i]
                for r in bam.fetch(ch, st["c"], st["c"] + 1):
                    if r.is_secondary or r.is_supplementary or r.query_name in seen:
                        continue
                    seen.add(r.query_name)
                    if not (r.has_tag("fp") and r.has_tag("np")):
                        raise SystemExit(f"{bp}: needs fp/np tags (fiberstate call --posterior-tags, version >= 0.5)")
                    q2r = query_to_ref(r)
                    fp = np.asarray(r.get_tag("fp"), dtype=np.float64) / 255.0
                    npp = np.asarray(r.get_tag("np"), dtype=np.float64) / 255.0
                    order = np.argsort(q2r)
                    rs = q2r[order]
                    lo, hi = r.reference_start, r.reference_end
                    for j in np.flatnonzero((cs >= lo) & (cs < hi)):
                        si = idx[j]
                        if r.query_name in obs[si]:
                            continue
                        k = np.searchsorted(rs, cs[j])
                        if k >= len(rs) or rs[k] != cs[j]:
                            continue
                        q = order[k]
                        pF, pN = fp[q], npp[q]
                        obs[si][r.query_name] = (max(0.0, 1.0 - pF - pN), pN, pF)
    return obs


def bh(p):
    p = np.asarray(p); n = len(p)
    if n == 0:
        return p
    o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("ref")
    ap.add_argument("--bams", required=True); ap.add_argument("--windows", required=True)
    ap.add_argument("--tf", action="append", required=True, help="NAME:MOTIF.jaspar|-:PEAKS")
    a = ap.parse_args()
    fa = pysam.FastaFile(a.ref)
    tfs = []
    for spec in a.tf:
        name, motif, peaks = spec.split(":", 2)
        W = pwm(PFM if motif == "-" else read_jaspar(motif))
        tfs.append((name, W, peaks))
    win = windows(a.windows.split(","))
    win = {ch: v for ch, v in win.items() if ch in fa.references}
    sites = find_sites(fa, win, tfs)
    print(f"{len(sites)} motif sites: " + ", ".join(f"{t} {sum(s['tf'] == t for s in sites)}" for t, _, _ in tfs)
          + "; " + ", ".join(f"{l} {sum(s['label'] == l for s in sites)}" for l in ("bound", "unbound", "ambiguous")))
    obs = posteriors(a.bams.split(","), sites)
    rows = []
    n_skip = 0
    by_chrom = {}
    for i, st in enumerate(sites):
        if st["label"] != "ambiguous":
            by_chrom.setdefault(st["chrom"], []).append(i)
    for ch, idx in by_chrom.items():
        idx.sort(key=lambda i: sites[i]["c"])
        for x, i in enumerate(idx):
            for j in idx[x + 1:]:
                d = sites[j]["c"] - sites[i]["c"]
                if d > DMAX:
                    break
                if d < DMIN:
                    continue
                common = [r for r in obs[i] if r in obs[j]]
                if len(common) < MIN_MOL:
                    continue
                A = [obs[i][r] for r in common]; B = [obs[j][r] for r in common]
                rc = co_occupancy(A, B, condition_on_access=True)
                ru = co_occupancy(A, B, condition_on_access=False)
                if not rc["informative"]:      # a rarely bound site gives no stable odds ratio
                    n_skip += 1
                    continue
                li, lj = sites[i]["label"], sites[j]["label"]
                cat = "both bound" if li == lj == "bound" else ("both unbound" if li == lj == "unbound" else "one bound")
                occ_i = float(np.mean([v[2] for v in A])); occ_j = float(np.mean([v[2] for v in B]))
                rows.append(dict(chrom=ch, ci=sites[i]["c"], cj=sites[j]["c"], tf_i=sites[i]["tf"], tf_j=sites[j]["tf"],
                                 label_i=li, label_j=lj, category=cat, distance=d, n=len(common),
                                 occ_i=occ_i, occ_j=occ_j,
                                 lor_cond=rc["log_or"], p_cond=rc["p"], lor_uncond=ru["log_or"], p_uncond=ru["p"]))
    print(f"{len(rows)} testable pairs ({n_skip} more pairs had a site bound or unbound on < 3 molecules)")
    for key in ("p_cond", "p_uncond"):
        q = bh([r[key] for r in rows])
        for r, v in zip(rows, q):
            r["q" + key[1:]] = float(v)
    with open(a.out + ".pairs.tsv", "w") as f:
        cols = list(rows[0].keys()) if rows else []
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    summ = {}
    def block(sel):
        if not sel:
            return dict(n_pairs=0)
        lc = np.array([r["lor_cond"] for r in sel]); lu = np.array([r["lor_uncond"] for r in sel])
        pc = np.array([r["p_cond"] for r in sel]); pu = np.array([r["p_uncond"] for r in sel])
        return dict(n_pairs=len(sel), median_lor_cond=float(np.median(lc)), median_lor_uncond=float(np.median(lu)),
                    frac_p05_cond=float(np.mean(pc < 0.05)), frac_p05_uncond=float(np.mean(pu < 0.05)),
                    frac_pos_p05_cond=float(np.mean((pc < 0.05) & (lc > 0))), frac_pos_p05_uncond=float(np.mean((pu < 0.05) & (lu > 0))),
                    n_q10_cond=int(sum(r["q_cond"] < 0.1 for r in sel)), n_q10_uncond=int(sum(r["q_uncond"] < 0.1 for r in sel)),
                    median_molecules=float(np.median([r["n"] for r in sel])))
    for cat in ("both unbound", "one bound", "both bound"):
        summ[cat] = block([r for r in rows if r["category"] == cat])
    bb = [r for r in rows if r["category"] == "both bound"]
    summ["both bound by distance"] = {f"{lo}-{hi} bp": block([r for r in bb if lo <= r["distance"] < hi])
                                      for lo, hi in ((60, 150), (150, 300), (300, 1001))}
    pairs_tf = {}
    for r in bb:
        pairs_tf.setdefault("-".join(sorted((r["tf_i"], r["tf_j"]))), []).append(r)
    summ["both bound by factor pair"] = {k: block(v) for k, v in pairs_tf.items()}
    top = sorted([r for r in rows if r["q_cond"] < 0.1], key=lambda r: r["p_cond"])[:15]
    summ["top_pairs_q10_cond"] = top
    json.dump(summ, open(a.out + ".summary.json", "w"), indent=1)
    print(f"\n{'pairs':14s} {'n':>6s}  {'median log OR':>22s}  {'share p<0.05':>22s}  {'q<0.1':>12s}")
    print(f"{'':14s} {'':>6s}  {'conditioned / uncond.':>22s}  {'conditioned / uncond.':>22s}  {'cond/uncond':>12s}")
    for cat in ("both unbound", "one bound", "both bound"):
        b = summ[cat]
        if b["n_pairs"]:
            print(f"{cat:14s} {b['n_pairs']:6d}  {b['median_lor_cond']:+9.2f} / {b['median_lor_uncond']:+9.2f}  "
                  f"{100 * b['frac_p05_cond']:8.1f}% / {100 * b['frac_p05_uncond']:6.1f}%   {b['n_q10_cond']:4d}/{b['n_q10_uncond']}")
    print("both bound, by distance:")
    for k, b in summ["both bound by distance"].items():
        if b["n_pairs"]:
            print(f"  {k:12s} {b['n_pairs']:6d}  {b['median_lor_cond']:+9.2f} / {b['median_lor_uncond']:+9.2f}  "
                  f"{100 * b['frac_p05_cond']:8.1f}% / {100 * b['frac_p05_uncond']:6.1f}%")
    print("both bound, by factor pair:")
    for k, b in summ["both bound by factor pair"].items():
        print(f"  {k:12s} {b['n_pairs']:6d}  {b['median_lor_cond']:+9.2f} / {b['median_lor_uncond']:+9.2f}  "
              f"{100 * b['frac_p05_cond']:8.1f}% / {100 * b['frac_p05_uncond']:6.1f}%")
    if top:
        print("strongest coupled pairs (conditioned, q < 0.1):")
        for r in top[:8]:
            print(f"  {r['chrom']}:{r['ci']}-{r['cj']} {r['tf_i']}-{r['tf_j']} {r['distance']} bp, {r['n']} molecules, "
                  f"log OR {r['lor_cond']:+.2f} (uncond. {r['lor_uncond']:+.2f}), q = {r['q_cond']:.2g}")


if __name__ == "__main__":
    main()
