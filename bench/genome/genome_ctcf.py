"""Genome-wide CTCF footprint benchmark: every strong CTCF motif, labelled by ChIP-seq.

Subcommands
  sites REF.fa CHROM PEAKS.bed.gz OUT.bed
      all CTCF motif matches (JASPAR MA0139.2, relative score >= 0.85) on CHROM, labelled
      bound (within 100 bp of an IDR peak summit) / unbound (no peak within 2 kb); ambiguous
      matches are dropped. No thinning, no score matching (motif strength is stratified later).
  scan SITES.bed WINDOW START END FS.bam FH.bam FT.tsv OUT_PREFIX
      one pass over the molecules that START in [START, END): per-site counts of scored and
      footprinted molecules per tool (default calls), and per-window histograms of each tool's
      per-molecule score at bound and unbound sites (for matched false-call comparisons).
      FS.bam = fiberstate --posterior-tags; FH.bam = FiberHMM; FT.tsv = ft footprint on the sites.
  summarize OUT_DIR OUT.json
      pools all windows: site AUROC (all sites and by motif-score band), false-call rate,
      recovery at matched false-call rates, and window-bootstrap 95% intervals.

A molecule is scored at a site when at least 12 motif bases and the motif centre are aligned
(as in the TF benchmarks). Default call: a footprint covering >= 80% of the motif.
"""
import glob
import json
import os
import sys

import numpy as np
import pysam

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ctcf_chip"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from select_sites import classify, pwm  # noqa: E402
from score_sites import auroc, ft_footprint_states  # noqa: E402
from bench.ftparse import ma_frame, parse_annotations  # noqa: E402
from fiberstate.io import query_to_ref  # noqa: E402

TOOLS = ("fiberstate", "FiberHMM", "fibertools")
NBIN = 256
MIN_FIBERS = 10


def cmd_sites(ref, chrom, peaks, out):
    fa = pysam.FastaFile(ref)
    bound, unbound, n = classify(fa, chrom, pwm(), peaks)
    rows = [(c, s, e, sc, st, "bound", sg) for c, s, e, sc, st, sg in bound] + \
           [(c, s, e, sc, st, "unbound", sg) for c, s, e, sc, st, sg in unbound]
    rows.sort(key=lambda r: r[1])
    with open(out, "w") as f:
        for k, (c, s, e, sc, st, lab, sg) in enumerate(rows):
            f.write(f"{c}\t{s}\t{e}\t{c}_{k}\t{sc:.3f}\t{st}\t{lab}\t{sg:.2f}\n")
    print(f"{chrom}: {n} motif matches; {len(bound)} bound, {len(unbound)} unbound (ambiguous dropped)")


def load_sites(p):
    out = []
    for line in open(p):
        c = line.rstrip("\n").split("\t")
        out.append(dict(chrom=c[0], s=int(c[1]), e=int(c[2]), name=c[3], score=float(c[4]),
                        strand=c[5], bound=c[6] == "bound", signal=float(c[7])))
    return out


def _spans(r, sites, centres):
    """[(site index, qa, qb, qs)] for sites this molecule scores."""
    q2r = query_to_ref(r)
    aq = np.flatnonzero(q2r >= 0)
    ar = q2r[aq]
    lo = np.searchsorted(centres, r.reference_start)
    hi = np.searchsorted(centres, r.reference_end)
    out = []
    for i in range(lo, hi):
        st = sites[i]
        a = np.searchsorted(ar, st["s"]); b = np.searchsorted(ar, st["e"])
        if b - a < min(12, int(0.8 * (st["e"] - st["s"]))):
            continue
        k = np.searchsorted(ar, centres[i])
        if k >= len(ar) or ar[k] != centres[i]:
            continue
        qs = aq[a:b]
        out.append((i, int(qs.min()), int(qs.max()) + 1, qs))
    return out


def _starts_in(r, start, end):
    return (not r.is_secondary and not r.is_supplementary and r.reference_start is not None
            and start <= r.reference_start < end)


def cmd_scan(sites_p, window, start, end, fs_p, fh_p, ft_p, out_prefix):
    start, end = int(start), int(end)
    sites = load_sites(sites_p)
    centres = np.array([(s["s"] + s["e"]) // 2 for s in sites])
    n = {t: np.zeros(len(sites), np.int32) for t in TOOLS}
    foot = {t: np.zeros(len(sites), np.int32) for t in TOOLS}
    hist = {t: {"bound": np.zeros(NBIN, np.int64), "unbound": np.zeros(NBIN, np.int64)} for t in TOOLS}
    calls = {t: 0 for t in TOOLS}
    chrom = sites[0]["chrom"] if sites else None
    in_window = set()

    # fiberstate: default call from MA (seq frame); score = mean footprint posterior over the motif
    bam = pysam.AlignmentFile(fs_p)
    frame = ma_frame(bam.header)
    for r in bam.fetch(chrom, start, end):
        if not _starts_in(r, start, end):
            continue
        in_window.add(r.query_name)
        ann = parse_annotations(r, features=("tf",), MA_frame=frame)["tf"]
        calls["fiberstate"] += len(ann)
        fp = np.asarray(r.get_tag("fp"), dtype=np.float64) if r.has_tag("fp") else None
        for i, qa, qb, qs in _spans(r, sites, centres):
            called = any(min(b, qb) - max(a, qa) >= 0.8 * (qb - qa) for a, b in ann)
            n["fiberstate"][i] += 1; foot["fiberstate"][i] += called
            if fp is not None:
                b_ = min(NBIN - 1, int(round(fp[qs].mean())))
                hist["fiberstate"]["bound" if sites[i]["bound"] else "unbound"][b_] += 1

    # FiberHMM: default = any footprint covering >= 80% of the motif; score = its best quality
    bam = pysam.AlignmentFile(fh_p)
    molecular = ma_frame(bam.header) == "molecular"
    for r in bam.fetch(chrom, start, end):
        if not _starts_in(r, start, end) or not r.has_tag("MA"):
            continue
        L = r.query_length
        aq = list(r.get_tag("AQ")) if r.has_tag("AQ") else []
        tf, k = [], 0
        for part in str(r.get_tag("MA")).split(";")[1:]:
            head, data = part.split(":", 1)
            j = next(m for m, ch in enumerate(head) if ch in "+-.")
            name, qspec = head[:j], head[j + 1:]
            for tok in filter(None, data.split(",")):
                s, l = tok.split("-", 1)
                s0 = int(s) - 1; e0 = s0 + int(l)
                q = aq[k] if (qspec and k < len(aq)) else 255
                k += len(qspec)
                if name == "tf":
                    if molecular and r.is_reverse:
                        s0, e0 = L - e0, L - s0
                    tf.append((s0, e0, q))
        calls["FiberHMM"] += len(tf)
        for i, qa, qb, qs in _spans(r, sites, centres):
            cov = [q for a, b, q in tf if min(b, qb) - max(a, qa) >= 0.8 * (qb - qa)]
            n["FiberHMM"][i] += 1; foot["FiberHMM"][i] += bool(cov)
            hist["FiberHMM"]["bound" if sites[i]["bound"] else "unbound"][int(max(cov)) if cov else 0] += 1

    # fibertools footprint (motif-guided, yes/no); only molecules that start in this window
    for (i, rn), stt in ft_footprint_states(ft_p, sites).items():
        if rn not in in_window:
            continue
        called = stt == "foot"
        n["fibertools"][i] += 1; foot["fibertools"][i] += called
        hist["fibertools"]["bound" if sites[i]["bound"] else "unbound"][255 if called else 0] += 1
        calls["fibertools"] += called

    with open(out_prefix + ".sites.tsv", "w") as f:
        f.write("site\tbound\tscore\tsignal\t" + "\t".join(f"n_{t}\tfoot_{t}" for t in TOOLS) + "\n")
        for i, st in enumerate(sites):
            if any(n[t][i] for t in TOOLS):
                f.write(f"{st['name']}\t{int(st['bound'])}\t{st['score']}\t{st['signal']}\t"
                        + "\t".join(f"{n[t][i]}\t{foot[t][i]}" for t in TOOLS) + "\n")
    json.dump(dict(window=window, chrom=chrom, start=start, end=end, molecules=len(in_window), calls=calls,
                   hist={t: {k: v.tolist() for k, v in hist[t].items()} for t in TOOLS}),
              open(out_prefix + ".hist.json", "w"))
    print(f"{window}: {len(in_window)} molecules; scored sites "
          + ", ".join(f"{t} {int((n[t] > 0).sum())}" for t in TOOLS))


def _interp(hb, hu, target):
    """bound rate at unbound rate = target, from score histograms (calls: score >= threshold)."""
    cb = np.cumsum(hb[::-1]) / max(1, hb.sum())     # share >= each bin, from the top
    cu = np.cumsum(hu[::-1]) / max(1, hu.sum())
    xs = np.r_[0.0, cu]; ys = np.r_[0.0, cb]
    return float(np.interp(target, xs, ys))


def cmd_summarize(out_dir, out_json):
    rows = {}
    for p in glob.glob(os.path.join(out_dir, "*.sites.tsv")):
        with open(p) as f:
            h = f.readline().rstrip("\n").split("\t")
            for line in f:
                c = line.rstrip("\n").split("\t")
                d = dict(zip(h, c))
                key = d["site"]
                if key not in rows:
                    rows[key] = dict(bound=d["bound"] == "1", score=float(d["score"]),
                                     **{f"n_{t}": 0 for t in TOOLS}, **{f"foot_{t}": 0 for t in TOOLS})
                for t in TOOLS:
                    rows[key][f"n_{t}"] += int(d[f"n_{t}"]); rows[key][f"foot_{t}"] += int(d[f"foot_{t}"])
    wins = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out_dir, "*.hist.json")))]
    res = dict(windows=len(wins), molecules=int(sum(w["molecules"] for w in wins)),
               footprint_calls={t: int(sum(w["calls"][t] for w in wins)) for t in TOOLS})

    def site_block(sel):
        out = {}
        for t in TOOLS:
            ok = [r for r in sel if r[f"n_{t}"] >= MIN_FIBERS]
            y = np.array([r["bound"] for r in ok], bool)
            occ = np.array([r[f"foot_{t}"] / r[f"n_{t}"] for r in ok])
            fb = sum(r[f"foot_{t}"] for r in ok if r["bound"]); nb = sum(r[f"n_{t}"] for r in ok if r["bound"])
            fu = sum(r[f"foot_{t}"] for r in ok if not r["bound"]); nu = sum(r[f"n_{t}"] for r in ok if not r["bound"])
            out[t] = dict(n_bound=int(y.sum()), n_unbound=int((~y).sum()),
                          auroc=float(auroc(occ, y)) if y.any() and (~y).any() else float("nan"),
                          false_call_rate=fu / max(1, nu), bound_rate=fb / max(1, nb))
        return out

    allr = list(rows.values())
    res["sites_all"] = site_block(allr)
    res["sites_by_motif_score"] = {f"{lo:.2f}-{hi:.2f}": site_block([r for r in allr if lo <= r["score"] < hi])
                                   for lo, hi in ((0.85, 0.90), (0.90, 0.95), (0.95, 1.01))}
    # matched false calls: pooled histograms, window bootstrap
    H = {t: {k: np.array([w["hist"][t][k] for w in wins]) for k in ("bound", "unbound")} for t in TOOLS}
    rng = np.random.default_rng(0)
    res["matched_false_calls"] = {}
    for f in (0.005, 0.01, 0.02):
        block = {}
        for t in TOOLS:
            block[t] = _interp(H[t]["bound"].sum(0), H[t]["unbound"].sum(0), f)
        boots = {t: [] for t in TOOLS}
        for _ in range(1000):
            idx = rng.integers(0, len(wins), len(wins))
            for t in TOOLS:
                boots[t].append(_interp(H[t]["bound"][idx].sum(0), H[t]["unbound"][idx].sum(0), f))
        diffs = {}
        for t in TOOLS[1:]:
            d = np.array(boots["fiberstate"]) - np.array(boots[t])
            diffs[t] = dict(diff=block["fiberstate"] - block[t], ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                            p_two_sided=float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))))
        res["matched_false_calls"][str(f)] = dict(bound_rate=block, fiberstate_minus=diffs)
    # per-chromosome consistency at 1%
    per_chrom = {}
    for w in wins:
        per_chrom.setdefault(w["chrom"], []).append(w)
    cons = {}
    for ch, ws in per_chrom.items():
        cons[ch] = {t: _interp(np.sum([w["hist"][t]["bound"] for w in ws], 0), np.sum([w["hist"][t]["unbound"] for w in ws], 0), 0.01)
                    for t in TOOLS}
    res["per_chromosome_at_1pct"] = cons
    res["chromosomes_where_fiberstate_leads_at_1pct"] = {t: int(sum(v["fiberstate"] > v[t] for v in cons.values())) for t in TOOLS[1:]}
    res["n_chromosomes"] = len(cons)
    json.dump(res, open(out_json, "w"), indent=1)

    print(f"\n{res['windows']} windows, {res['molecules']:,} molecules, {len(rows):,} CTCF motif sites with molecules")
    fc = res["footprint_calls"]
    print(f"footprint calls on these molecules: fiberstate {fc['fiberstate']:,}, FiberHMM {fc['FiberHMM']:,} "
          f"(genome-wide); fibertools {fc['fibertools']:,} (only at the motif sites it is given)")
    print(f"\n{'all sites':24s}" + "".join(f"{t:>26s}" for t in TOOLS))
    a = res["sites_all"]
    print(f"{'  sites (bound/unbound)':24s}" + "".join(f"{a[t]['n_bound']:>13,}/{a[t]['n_unbound']:<12,}" for t in TOOLS))
    print(f"{'  site AUROC':24s}" + "".join(f"{a[t]['auroc']:>26.3f}" for t in TOOLS))
    print(f"{'  false-call rate':24s}" + "".join(f"{100 * a[t]['false_call_rate']:>25.2f}%" for t in TOOLS))
    print(f"{'  footprinted at bound':24s}" + "".join(f"{100 * a[t]['bound_rate']:>25.1f}%" for t in TOOLS))
    print("\nsite AUROC by motif score:")
    for band, b in res["sites_by_motif_score"].items():
        print(f"  {band:22s}" + "".join(f"{b[t]['auroc']:>26.3f}" for t in TOOLS) + f"   ({b['fiberstate']['n_bound']:,} bound)")
    print("\nbound molecules footprinted at matched false calls (window-bootstrap 95% CI for fiberstate minus other):")
    for f, b in res["matched_false_calls"].items():
        print(f"  {100 * float(f):.1f}% false calls: " + ", ".join(f"{t} {100 * v:.1f}%" for t, v in b["bound_rate"].items()))
        for t, d in b["fiberstate_minus"].items():
            print(f"      fiberstate - {t:10s} {100 * d['diff']:+.1f} points (CI {100 * d['ci95'][0]:+.1f} to {100 * d['ci95'][1]:+.1f}, "
                  f"p {'< 0.001' if d['p_two_sided'] == 0 else '= %.3g' % d['p_two_sided']})")
    print("\nchromosomes where fiberstate recovers more at 1% false calls: "
          + ", ".join(f"vs {t} {v}/{res['n_chromosomes']}" for t, v in res["chromosomes_where_fiberstate_leads_at_1pct"].items()))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"sites": cmd_sites, "scan": cmd_scan, "summarize": cmd_summarize}[cmd](*args)
