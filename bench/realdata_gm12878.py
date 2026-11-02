"""fiberstate vs fibertools on a real GM12878 Fiber-seq region, checked against FIRE peaks.

    python -I bench/realdata_gm12878.py REGION.bam FIRE_PEAKS.bed REF.fa FT_BIN CHROM START END OUT.json

Truth is not known on real data, so the checks are indirect:
  * nucleosome length and repeat length (expected ~147 bp core, ~185-195 bp repeat in human LCLs)
  * how well each tool's per-base accessibility separates FIRE peaks from background
    (FIRE peaks are built from fibertools output, so this check is biased toward fibertools)
  * enrichment of fiberstate TF footprints inside FIRE peaks (real footprints should be
    concentrated in regulatory elements; spurious gaps should not)
  * nucleosome calls in GC-rich windows, split by FIRE peak status
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pysam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.metrics import auroc  # noqa: E402
from fiberstate.io import fiber_arrays  # noqa: E402
from fiberstate.context import ContextTable  # noqa: E402
_CTX = ContextTable.load(os.environ["FS_CONTEXT"]) if os.environ.get("FS_CONTEXT") else None
from fiberstate.model import Model  # noqa: E402

KEYS = ("cov", "fs_link", "fs_nuc", "fs_foot", "fs_nuc_call", "fs_foot_call", "ft_nuc", "m6a", "at")


def parse_ma(rd):
    from bench.ftparse import parse_annotations
    return parse_annotations(rd)["nuc"]


def run_ft(ft, bam):
    d = tempfile.mkdtemp()
    out = os.path.join(d, "ft.bam")
    t0 = time.perf_counter()
    subprocess.run([ft, "add-nucleosomes", bam, out], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dt = time.perf_counter() - t0
    res = {}
    with pysam.AlignmentFile(out, "rb", check_sq=False) as f:
        for rd in f:
            if not (rd.is_secondary or rd.is_supplementary):
                res[rd.query_name] = parse_ma(rd)
    return res, dt


def _work(args):
    bam_p, names, ftn, chrom, lo, hi = args
    names = set(names)
    n = hi - lo
    acc = {k: np.zeros(n) for k in KEYS}
    fs_nuc_len, fs_foot_len, ft_nuc_len, fs_spacing, ft_spacing, pa = [], [], [], [], [], []
    foot_centers = []
    t = 0.0
    with pysam.AlignmentFile(bam_p) as bam:
        for r in bam.fetch(chrom, lo, hi):
            if r.query_name not in names or r.is_secondary or r.is_supplementary:
                continue
            s, m = fiber_arrays(r)
            t0 = time.perf_counter()
            c = Model(context=_CTX).call_read(s, m)
            t += time.perf_counter() - t0
            pa.append(c.p_access)
            q2r = np.full(len(s), -1, dtype=np.int64)
            for qp, rp in r.get_aligned_pairs(matches_only=True):
                q2r[qp] = rp
            ok = (q2r >= lo) & (q2r < hi)
            idx = q2r[ok] - lo
            acc["cov"][idx] += 1
            acc["fs_link"][idx] += c.post_linker[ok]
            acc["fs_nuc"][idx] += c.post_nuc[ok]
            acc["fs_foot"][idx] += c.post_foot[ok]
            acc["fs_nuc_call"][idx] += (c.path[ok] == 1)
            acc["fs_foot_call"][idx] += (c.path[ok] == 2)
            ftm = np.zeros(len(s), bool)
            segs = ftn.get(r.query_name, [])
            for a, b in segs:
                ftm[a:b] = True
            acc["ft_nuc"][idx] += ftm[ok]
            isat = np.isin(s, [65, 84])
            acc["at"][idx] += isat[ok]
            acc["m6a"][idx] += ((m > 0) & isat)[ok]
            nucs = [(a, b) for k, a, b, _ in c.segments if k == 1]
            fs_nuc_len += [b - a for a, b in nucs]
            fs_spacing += [nucs[i + 1][0] - nucs[i][0] for i in range(len(nucs) - 1)]
            ft_nuc_len += [b - a for a, b in segs]
            ft_spacing += [segs[i + 1][0] - segs[i][0] for i in range(len(segs) - 1)]
            for k, a, b, q in c.segments:
                if k == 2:
                    fs_foot_len.append(b - a)
                    rp = q2r[(a + b) // 2]
                    if lo <= rp < hi:
                        foot_centers.append((int(rp), float(q)))
    return acc, fs_nuc_len, fs_foot_len, ft_nuc_len, fs_spacing, ft_spacing, pa, foot_centers, t


def mode_hist(x, lo, hi):
    h, e = np.histogram(x, bins=np.arange(lo, hi + 1, 2))
    return float(e[h.argmax()] + 1), h.tolist()


def main(bam_p, peaks_p, ref_p, ft, chrom, start, end, out_p):
    lo, hi = int(start), int(end)
    with pysam.AlignmentFile(bam_p) as bam:
        names = sorted({r.query_name for r in bam.fetch(chrom, lo, hi)
                        if not (r.is_secondary or r.is_supplementary)})
    ftn, ft_time = run_ft(ft, bam_p)
    chunks = [names[i::2] for i in range(2)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(2) as ex:
        parts = list(ex.map(_work, [(bam_p, ch, {n: ftn.get(n, []) for n in ch}, chrom, lo, hi) for ch in chunks]))
    wall = time.perf_counter() - t0
    acc = {k: sum(p[0][k] for p in parts) for k in KEYS}
    cat = lambda i: [x for p in parts for x in p[i]]  # noqa: E731
    fs_nuc_len, fs_foot_len, ft_nuc_len = cat(1), cat(2), cat(3)
    fs_spacing, ft_spacing, pa, foots = cat(4), cat(5), cat(6), cat(7)
    fs_cpu = sum(p[8] for p in parts)

    # FIRE peaks (FDR <= 0.05) as a mask
    peaks = []
    with open(peaks_p) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t")
            if float(p[20]) <= 0.05:
                peaks.append((int(p[1]), int(p[2])))
    pm = np.zeros(hi - lo, bool)
    for a, b in peaks:
        pm[max(0, a - lo):max(0, min(hi, b) - lo)] = True

    cov = np.maximum(acc["cov"], 1)
    good = acc["cov"] >= 20
    fs_acc = acc["fs_link"] / cov                      # P(accessible)
    ft_acc = 1 - acc["ft_nuc"] / cov                   # not in a fibertools nucleosome
    raw = acc["m6a"] / np.maximum(acc["at"], 1)        # raw m6A per A/T

    # 100 bp bins
    B = 100
    nb = (hi - lo) // B
    def binned(x):
        return x[:nb * B].reshape(nb, B).mean(1)
    lab = binned(pm.astype(float)) >= 0.5
    gb = binned(good.astype(float)) >= 0.9
    roc = {nm: auroc(binned(v)[gb], lab[gb]) for nm, v in
           (("fiberstate", fs_acc), ("fibertools", ft_acc), ("raw_m6a_density", raw))}

    # footprint enrichment in FIRE peaks
    fc = np.array([c for c, q in foots])
    fq = np.array([q for c, q in foots])
    in_pk = pm[fc - lo] if len(fc) else np.array([], bool)
    frac_bp_peak = float((pm & good).sum() / good.sum())
    cov_peak = acc["cov"][pm & good].sum()
    cov_bg = acc["cov"][~pm & good].sum()
    def enr(mask):
        ip = (in_pk & mask).sum(); ob = (~in_pk & mask).sum()
        rate_p = ip / max(1, cov_peak) * 1e3
        rate_b = ob / max(1, cov_bg) * 1e3
        return dict(n=int(mask.sum()), per_kb_fiber_in_peaks=float(rate_p), per_kb_fiber_outside=float(rate_b),
                    enrichment=float(rate_p / max(1e-12, rate_b)), frac_in_peaks=float(ip / max(1, mask.sum())))
    foot = dict(all=enr(np.ones(len(fc), bool)), conf_0_9=enr(fq >= 0.9), conf_0_99=enr(fq >= 0.99))

    # GC-rich windows from the reference
    fa = pysam.FastaFile(ref_p)
    seq = np.frombuffer(fa.fetch(chrom, lo, hi).upper().encode(), dtype=np.uint8)
    gcb = np.isin(seq, [67, 71]).astype(float)
    gc200 = np.convolve(gcb, np.ones(200) / 200, mode="same")
    rich = (gc200 >= 0.65) & good
    def occ(mask):
        if mask.sum() == 0:
            return None
        return dict(bp=int(mask.sum()),
                    fs_nuc=float((acc["fs_nuc_call"][mask] / cov[mask]).mean()),
                    fs_nuc_post=float((acc["fs_nuc"][mask] / cov[mask]).mean()),
                    ft_nuc=float((acc["ft_nuc"][mask] / cov[mask]).mean()),
                    raw_m6a_per_AT=float(acc["m6a"][mask].sum() / max(1, acc["at"][mask].sum())),
                    at_frac=float(1 - gcb[mask].mean()))
    gc = dict(gc_rich_in_peaks=occ(rich & pm), gc_rich_outside=occ(rich & ~pm),
              normal_in_peaks=occ(~rich & pm & good), normal_outside=occ(~rich & ~pm & good))

    # tracks for plotting (500 bp resolution)
    T = 500
    nt = (hi - lo) // T
    tr = lambda x: x[:nt * T].reshape(nt, T).mean(1).round(4).tolist()  # noqa: E731

    fs_nrl, fs_sp_h = mode_hist([x for x in fs_spacing if 100 <= x <= 400], 100, 400)
    ft_nrl, ft_sp_h = mode_hist([x for x in ft_spacing if 100 <= x <= 400], 100, 400)
    res = dict(
        region=f"{chrom}:{lo}-{hi}", n_fibers=len(names), n_fire_peaks=len(peaks),
        frac_bp_in_peaks=frac_bp_peak, median_coverage=float(np.median(acc["cov"][good])),
        fs_cpu_sec=fs_cpu, fs_wall_sec=wall, ft_sec=ft_time,
        fs_ms_per_fiber=1000 * fs_cpu / len(names), ft_ms_per_fiber=1000 * ft_time / max(1, len(ftn)),
        p_access=dict(q10=float(np.quantile(pa, .1)), median=float(np.median(pa)), q90=float(np.quantile(pa, .9))),
        nuc_len=dict(fs_mode=mode_hist(fs_nuc_len, 60, 260)[0], ft_mode=mode_hist(ft_nuc_len, 60, 260)[0],
                     fs_median=float(np.median(fs_nuc_len)), ft_median=float(np.median(ft_nuc_len)),
                     fs_hist=np.histogram(fs_nuc_len, bins=np.arange(40, 262, 4))[0].tolist(),
                     ft_hist=np.histogram(ft_nuc_len, bins=np.arange(40, 262, 4))[0].tolist(),
                     n_fs=len(fs_nuc_len), n_ft=len(ft_nuc_len)),
        repeat_length=dict(fs_mode=fs_nrl, ft_mode=ft_nrl,
                           fs_median=float(np.median([x for x in fs_spacing if 100 <= x <= 400])),
                           ft_median=float(np.median([x for x in ft_spacing if 100 <= x <= 400]))),
        footprint_len_median=float(np.median(fs_foot_len)) if fs_foot_len else None,
        fire_auroc=roc, footprints=foot, gc=gc,
        tracks=dict(step=T, start=lo, fs_acc=tr(fs_acc), ft_acc=tr(ft_acc), raw=tr(raw),
                    peak=tr(pm.astype(float)), cov=tr(acc["cov"])),
    )
    json.dump(res, open(out_p, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "tracks"}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main(*sys.argv[1:])
