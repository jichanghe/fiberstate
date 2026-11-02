"""Run the full fiberstate benchmark.

    python -m bench.run --out results/            # everything
    python -m bench.run --quick --out results/    # small smoke run
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from fiberstate.model import Model, Params
from fiberstate.sites import co_occupancy, occupancy
from . import baselines
from .metrics import auroc, false_calls_in_mask, prf, seg_prf
from .simulate import FOOT, NUC, SimConfig, simulate

METHODS = ["fiberstate", "fiberstate_noseq", "fiberstate_noEM", "gap_threshold", "naive_hmm"]
TUNED = {"gap_threshold": dict(nuc_min=85, foot_min=15), "naive_hmm": dict(p_access=0.35)}


def run_method(name, read, tuned):
    if name == "fiberstate":
        c = Model().call_read(read.seq, read.m6a)
        return c.path, c.segments, c
    if name == "fiberstate_noseq":
        # same model, but every base is an observation (G/C counted as unmethylated)
        p = Params(p_access=0.35, p_protect=0.012)
        c = Model(p).call_read(read.seq, read.m6a, sequence_aware=False)
        return c.path, c.segments, c
    if name == "fiberstate_noEM":
        c = Model(Params(learn_per_read=False)).call_read(read.seq, read.m6a)
        return c.path, c.segments, c
    if name == "gap_threshold":
        path, segs = baselines.gap_caller(read.m6a, **tuned["gap_threshold"])
        return path, segs, None
    if name == "naive_hmm":
        path, segs = baselines.naive_hmm(read.m6a, **tuned["naive_hmm"])
        return path, segs, None
    raise ValueError(name)


def _island_mask(locus, L):
    m = np.zeros(L, dtype=bool)
    for s, e in locus["islands"]:
        m[s:e] = True
    return m


def _site_probs(out, site):
    path, segs, call = out
    c = (site.start + site.end) // 2
    if call is not None:
        return (float(call.post_linker[c]), float(call.post_nuc[c]), float(call.post_foot[c]))
    v = [0.0, 0.0, 0.0]
    v[int(path[c])] = 1.0
    return tuple(v)


def _work(args):
    cfg_dict, methods, tuned, lo, hi = args
    cfg = SimConfig(**cfg_dict)
    loci, reads, sites = simulate(cfg)
    reads = [r for r in reads if lo <= r.locus < hi]
    acc = {m: dict(nuc=[0, 0, 0], foot=[0, 0, 0], dyad=[], base_ok=0, base_n=0,
                   false_nuc_island=0, false_nuc_bg=0, time=0.0) for m in methods}
    site_obs = {m: {} for m in methods}   # site idx -> list of (read name, probs)
    truth_bound = {}
    island_bp = 0
    bg_bp = 0
    for r in reads:
        loc = loci[r.locus]
        L = len(r.seq)
        im = _island_mask(loc, L)
        island_bp += im.sum(); bg_bp += (~im).sum()
        for gi, b in r.bound.items():
            truth_bound.setdefault(gi, []).append((r.name, b))
        for m in methods:
            t0 = time.perf_counter()
            out = run_method(m, r, tuned)
            acc[m]["time"] += time.perf_counter() - t0
            path, segs, call = out
            a = acc[m]
            for kind, key in ((NUC, "nuc"), (FOOT, "foot")):
                tp, nt, np_, dy = seg_prf(r.segments, segs, kind)
                a[key][0] += tp; a[key][1] += nt; a[key][2] += np_
                if kind == NUC:
                    a["dyad"].extend(dy)
            a["base_ok"] += int((path == r.state).sum()); a["base_n"] += L
            a["false_nuc_island"] += false_calls_in_mask(r.segments, segs, NUC, im)
            a["false_nuc_bg"] += false_calls_in_mask(r.segments, segs, NUC, ~im)
            off = loc["site_offset"]
            for j, st in enumerate(loc["sites"]):
                site_obs[m].setdefault(off + j, []).append((r.name, _site_probs(out, st)))
    return acc, site_obs, truth_bound, int(island_bp), int(bg_bp)


def run_condition(cfg: SimConfig, methods=METHODS, tuned=TUNED, workers=2):
    n = cfg.n_loci
    chunks = [(dataclasses.asdict(cfg), methods, tuned, i, min(n, i + max(1, n // workers)))
              for i in range(0, n, max(1, n // workers))]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_work, chunks))
    acc = {m: dict(nuc=[0, 0, 0], foot=[0, 0, 0], dyad=[], base_ok=0, base_n=0,
                   false_nuc_island=0, false_nuc_bg=0, time=0.0) for m in methods}
    site_obs = {m: {} for m in methods}
    truth_bound = {}
    island_bp = bg_bp = 0
    for a_, so, tb, ib, bb in parts:
        island_bp += ib; bg_bp += bb
        truth_bound.update(tb)
        for m in methods:
            for k in ("nuc", "foot"):
                for i in range(3):
                    acc[m][k][i] += a_[m][k][i]
            acc[m]["dyad"].extend(a_[m]["dyad"])
            for k in ("base_ok", "base_n", "false_nuc_island", "false_nuc_bg", "time"):
                acc[m][k] += a_[m][k]
            site_obs[m].update(so[m])
    n_reads = n * cfg.reads_per_locus
    res = {}
    for m in methods:
        a = acc[m]
        npr = prf(*a["nuc"]); fpr = prf(*a["foot"])
        # occupancy error per site
        occ_err = []
        for gi, lst in site_obs[m].items():
            tb = truth_bound.get(gi, [])
            if not tb:
                continue
            true_occ = np.mean([b for _, b in tb])
            est = occupancy([p for _, p in lst])["occupancy"]
            occ_err.append(abs(est - true_occ))
        res[m] = dict(
            nuc_precision=npr[0], nuc_recall=npr[1], nuc_f1=npr[2],
            foot_precision=fpr[0], foot_recall=fpr[1], foot_f1=fpr[2],
            dyad_median_err=float(np.median(a["dyad"])) if a["dyad"] else float("nan"),
            base_accuracy=a["base_ok"] / a["base_n"],
            false_nuc_per_10kb_island=1e4 * a["false_nuc_island"] / max(1, island_bp),
            false_nuc_per_10kb_bg=1e4 * a["false_nuc_bg"] / max(1, bg_bp),
            occupancy_mae=float(np.mean(occ_err)) if occ_err else float("nan"),
            ms_per_read=1000 * a["time"] / n_reads,
        )
    return res, site_obs, truth_bound


def cooccupancy_eval(cfg, tuned, methods=("fiberstate", "gap_threshold", "naive_hmm"), workers=2):
    _, site_obs, _ = run_condition(cfg, list(methods), tuned, workers=workers)
    # simulate() is deterministic given the seed: re-run it to recover the site table
    _, _, sites = simulate(cfg)
    pairs = {}
    for gi, s in enumerate(sites):
        if s.pair_id >= 0:
            pairs.setdefault(s.pair_id, []).append(gi)
    out = {}
    for m in methods:
        for cond in (True, False):
            scores, labels, called, inf = [], [], [], []
            for pid, (i, j) in ((k, v) for k, v in pairs.items() if len(v) == 2):
                oi = dict(site_obs[m][i]); oj = dict(site_obs[m][j])
                names = [n for n in oi if n in oj]
                r = co_occupancy([oi[n] for n in names], [oj[n] for n in names],
                                 condition_on_access=cond)
                scores.append(r["z"]); labels.append(sites[i].coupled)
                # call: positive coupling, one-sided p < 0.05 (Fisher exact), informative pairs only
                called.append(bool(r["informative"] and r["log_or"] > 0 and r["p"] / 2 < 0.05))
                inf.append(bool(r["informative"]))
            key = f"{m}|{'conditioned' if cond else 'unconditioned'}"
            labels_a = np.array(labels)
            called = np.array(called)
            tp = int((called & labels_a).sum()); fp = int((called & ~labels_a).sum())
            out[key] = dict(auroc=auroc(scores, labels), n_pairs=len(labels),
                            n_coupled=int(labels_a.sum()), tp=tp, fp=fp, n_informative=int(np.sum(inf)),
                            fdr=fp / max(1, tp + fp), power=tp / max(1, labels_a.sum()))
    return out


def tune_baselines(workers=2):
    """Grid-search baseline thresholds on a held-out simulation (seed 999)."""
    cfg = SimConfig(n_loci=6, reads_per_locus=20, seed=999)
    best_gap, best_hmm = None, None
    for nuc_min in (60, 75, 85, 100, 120):
        for foot_min in (8, 12, 15, 20, 25):
            t = dict(TUNED, gap_threshold=dict(nuc_min=nuc_min, foot_min=foot_min))
            r, _, _ = run_condition(cfg, ["gap_threshold"], t, workers)
            sc = (r["gap_threshold"]["nuc_f1"] + r["gap_threshold"]["foot_f1"]) / 2
            if best_gap is None or sc > best_gap[0]:
                best_gap = (sc, dict(nuc_min=nuc_min, foot_min=foot_min))
    for pa in (0.1, 0.15, 0.25, 0.35, 0.45):
        for pp in (0.0005, 0.001, 0.005, 0.012, 0.03):
            t = dict(TUNED, naive_hmm=dict(p_access=pa, p_protect=pp))
            r, _, _ = run_condition(cfg, ["naive_hmm"], t, workers)
            sc = (r["naive_hmm"]["nuc_f1"] + r["naive_hmm"]["foot_f1"]) / 2
            if best_hmm is None or sc > best_hmm[0]:
                best_hmm = (sc, dict(p_access=pa, p_protect=pp))
    return {"gap_threshold": best_gap[1], "naive_hmm": best_hmm[1],
            "_scores": {"gap_threshold": best_gap[0], "naive_hmm": best_hmm[0]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    w = a.workers
    base = dict(n_loci=4 if a.quick else 12, reads_per_locus=10 if a.quick else 30)

    t0 = time.time()
    tuned = tune_baselines(w)
    print("tuned", tuned, flush=True)
    global TUNED
    TUNED = {k: v for k, v in tuned.items() if not k.startswith("_")}
    results = dict(tuned=tuned, sensitivity=[], fp_rate=[], cooccupancy=None)

    seeds = range(1, (2 if a.quick else a.seeds) + 1)
    for sens in (0.3, 0.45, 0.6, 0.75, 0.9):
        for sd in seeds:
            cfg = SimConfig(**base, sensitivity_mean=sens, seed=sd)
            r, _, _ = run_condition(cfg, METHODS, TUNED, w)
            results["sensitivity"].append(dict(value=sens, seed=sd, metrics=r))
            print(f"sens={sens} seed={sd} {time.time()-t0:.0f}s", flush=True)
    for fp in (0.005, 0.02, 0.05, 0.08):
        for sd in seeds:
            cfg = SimConfig(**base, fp_rate=fp, seed=100 + sd)
            r, _, _ = run_condition(cfg, METHODS, TUNED, w)
            results["fp_rate"].append(dict(value=fp, seed=sd, metrics=r))
            print(f"fp={fp} seed={sd} {time.time()-t0:.0f}s", flush=True)
    cc = SimConfig(n_loci=10 if a.quick else 50, reads_per_locus=30 if a.quick else 60,
                   ndrs_per_locus=4, sites_per_ndr=(2, 2), seed=7)
    results["cooccupancy"] = cooccupancy_eval(cc, TUNED, workers=w)
    print("cooccupancy", json.dumps(results["cooccupancy"], indent=1), flush=True)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
