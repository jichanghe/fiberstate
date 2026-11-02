"""Evaluate the real fibertools-rs `ft add-nucleosomes` on the same simulated conditions.

    python -m bench.ft_eval --ft /path/to/ft --out results/
fibertools does not call TF footprints de novo, so only nucleosome metrics are reported.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time

import numpy as np
import pysam

from fiberstate.io import write_sim_bam
from .metrics import false_calls_in_mask, prf, seg_prf
from .simulate import NUC, SimConfig, simulate


def eval_cfg(ft, cfg):
    loci, reads, _ = simulate(cfg)
    with tempfile.TemporaryDirectory() as d:
        inp, out = os.path.join(d, "in.bam"), os.path.join(d, "out.bam")
        write_sim_bam(inp, reads)
        t0 = time.perf_counter()
        subprocess.run([ft, "add-nucleosomes", inp, out], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dt = time.perf_counter() - t0
        calls = {}
        with pysam.AlignmentFile(out, "rb", check_sq=False) as f:
            for rd in f:
                from bench.ftparse import parse_annotations
                segs = [(NUC, a, b, 1.0) for a, b in parse_annotations(rd)["nuc"]]
                calls[rd.query_name] = segs
    tp = nt = np_ = 0
    dy = []
    fi = fb = 0
    ib = bb = 0
    for r in reads:
        segs = calls.get(r.name, [])
        a, b, c, d_ = seg_prf(r.segments, segs, NUC)
        tp += a; nt += b; np_ += c; dy += d_
        im = np.zeros(len(r.seq), bool)
        for s, e in loci[r.locus]["islands"]:
            im[s:e] = True
        ib += im.sum(); bb += (~im).sum()
        fi += false_calls_in_mask(r.segments, segs, NUC, im)
        fb += false_calls_in_mask(r.segments, segs, NUC, ~im)
    p, rc, f1 = prf(tp, nt, np_)
    return dict(nuc_precision=p, nuc_recall=rc, nuc_f1=f1,
                dyad_median_err=float(np.median(dy)) if dy else float("nan"),
                false_nuc_per_10kb_island=1e4 * fi / ib, false_nuc_per_10kb_bg=1e4 * fb / bb,
                ms_per_read=1000 * dt / len(reads))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    base = dict(n_loci=12, reads_per_locus=30)
    res = dict(sensitivity=[], fp_rate=[])
    for sens in (0.3, 0.45, 0.6, 0.75, 0.9):
        for sd in range(1, a.seeds + 1):
            res["sensitivity"].append(dict(value=sens, seed=sd, metrics=eval_cfg(
                a.ft, SimConfig(**base, sensitivity_mean=sens, seed=sd))))
    for fp in (0.005, 0.02, 0.05, 0.08):
        for sd in range(1, a.seeds + 1):
            res["fp_rate"].append(dict(value=fp, seed=sd, metrics=eval_cfg(
                a.ft, SimConfig(**base, fp_rate=fp, seed=100 + sd))))
    with open(os.path.join(a.out, "ft_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["sensitivity"][6], indent=1))


if __name__ == "__main__":
    main()
