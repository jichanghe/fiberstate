"""fiberstate command line.

  fiberstate call  IN.bam -o calls.bed [--out-bam tagged.bam] [--soft] [-t 4]
  fiberstate sites IN.bam SITES.bed -o occupancy.tsv [--pairs pairs.tsv] [--max-dist 1000]
  fiberstate simulate -o sim.bam --truth truth.bed [--reads 200]
"""
from __future__ import annotations

import argparse
import array
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np

from . import __version__
from .model import Model, Params


def _context_from_args(a):
    from .context import ContextTable
    if a.context == "none" or a.no_seq:
        return None
    if a.context == "default":
        import os
        return ContextTable.load(os.path.join(os.path.dirname(__file__), "data", "hia5_k5_gm12878.tsv"))
    return ContextTable.load(a.context)


def _params_from_args(a):
    return Params(p_access=a.p_access, p_protect=a.p_protect, mean_linker=a.mean_linker,
                  frac_nuc=a.frac_nuc, learn_per_read=not a.no_em,
                  merge_unsupported=not a.keep_inferred, em_iters=a.em_iters,
                  foot_posterior=None if a.foot_posterior <= 0 else a.foot_posterior)


_WORKER = {}


def _init_worker(params, ctx, seq_aware, header=None, opts=None):
    # belt and braces: also cap BLAS threads if numpy was imported before our env settings
    try:
        from threadpoolctl import threadpool_limits
        _WORKER["tp"] = threadpool_limits(limits=1)
    except ImportError:
        pass
    _WORKER["model"] = Model(params, context=ctx)
    _WORKER["seq_aware"] = seq_aware
    if header is not None:
        import pysam
        _WORKER["header"] = pysam.AlignmentHeader.from_dict(header)
    _WORKER["opts"] = opts or {}


class _LiteCall:
    """Segments and per-fiber rates (what BED rows and BAM tags are built from)."""
    __slots__ = ("segments", "p_access", "p_protect", "length")

    def __init__(self, segments, p_access, p_protect, length):
        self.segments, self.p_access, self.p_protect, self.length = segments, p_access, p_protect, length


def _call_chunk(chunk):
    """Worker: SAM text records in -> (BED text, tagged SAM text or None, called?) per record.
    Decoding m6A, the model, BED rows and tags all run here, so the main process only moves text."""
    import time
    import pysam
    from .io import add_tags, fiber_arrays, query_to_ref, segments_to_bed_rows
    m = _WORKER["model"]
    h = _WORKER["header"]
    o = _WORKER["opts"]
    out = []
    t0 = time.perf_counter()
    for rec in chunk:
        read = pysam.AlignedSegment.fromstring(rec, h)
        s, m6a = fiber_arrays(read, ml_threshold=o["ml_threshold"], soft=o["soft"])
        if s is None or len(s) < 200:
            out.append(None)
            continue
        m6a = m6a.astype(np.float32) if o["soft"] else (m6a > 0).astype(np.uint8)
        c = m.call_read(s, m6a, sequence_aware=_WORKER["seq_aware"])
        lc = _LiteCall(c.segments, c.p_access, c.p_protect, len(s))
        q2r = query_to_ref(read) if not read.is_unmapped else None
        rows = segments_to_bed_rows(read, lc, q2r)
        if o.get("calib") is not None:
            cut, rate = o["calib"]
            est = lambda conf: float(np.interp(conf, cut, rate, left=rate[0], right=rate[-1]))  # noqa: E731
            bed = "".join("%s\t%d\t%d\t%s\t%s\t%.3f\t%d\t%d\t%.3f\t%.4f" % row
                          + ("\t%.4g\n" % est(row[5]) if row[3] == "footprint" else "\tNA\n") for row in rows)
        else:
            bed = "".join("%s\t%d\t%d\t%s\t%s\t%.3f\t%d\t%d\t%.3f\t%.4f\n" % row for row in rows)
        tagged = None
        if o["bam"]:
            add_tags(read, lc)
            if o.get("calib") is not None:
                cut, rate = o["calib"]
                fe = [np.interp(cf, cut, rate, left=rate[0], right=rate[-1]) for k, _, _, cf in lc.segments if k == 2]
                read.set_tag("fe", array.array("f", fe))   # estimated false-call rate per footprint (fs/fl order)
            if o.get("posterior_tags"):
                # per-base posteriors, 0-255, stored-SEQ frame: fp = TF footprint, np = nucleosome
                read.set_tag("fp", array.array("B", np.rint(255 * np.clip(c.post_foot, 0, 1)).astype(np.uint8).tobytes()))
                read.set_tag("np", array.array("B", np.rint(255 * np.clip(c.post_nuc, 0, 1)).astype(np.uint8).tobytes()))
            tagged = read.to_string()
        out.append((bed, tagged))
    return out, time.perf_counter() - t0


def cmd_call(a):
    import collections
    import time
    import pysam
    calib = None
    if a.false_call_rate is not None:
        from .calibrate import Calibration
        calib = Calibration(a.calibration)
        a.foot_posterior = calib.cutoff_for(a.false_call_rate)
        print(f"fiberstate: footprint cut-off {a.foot_posterior:.2f} for a false-call rate <= {a.false_call_rate} "
              f"({'built-in table' if a.calibration in (None, 'default') else a.calibration})", file=sys.stderr)
    params = _params_from_args(a)
    ctx = _context_from_args(a)
    io_threads = max(1, min(4, a.threads // 2))     # htslib (de)compression threads
    bam = pysam.AlignmentFile(a.bam, "rb", check_sq=False, threads=io_threads)
    hdr = None
    if a.out_bam:
        hdr = bam.header.to_dict()
        hdr.setdefault("CO", []).append("fiberstate: MA tag coordinates are in stored SEQ frame (coord=seq)")
        hdr.setdefault("PG", []).append({"ID": "fiberstate", "PN": "fiberstate", "VN": __version__,
                                         "CL": "fiberstate " + " ".join(sys.argv[1:])})
    out_bam = pysam.AlignmentFile(a.out_bam, "wb", header=hdr, threads=io_threads) if a.out_bam else None
    out = open(a.output, "w") if a.output != "-" else sys.stdout
    out.write("#chrom\tstart\tend\ttype\tread\tconfidence\tq_start\tq_end\tread_p_access\tread_p_protect"
              + ("\test_false_call_rate" if calib is not None else "") + "\n")
    n = 0
    t_start = time.perf_counter()
    t_read = t_write = t_wait = model_cpu = 0.0
    opts = dict(ml_threshold=a.ml_threshold, soft=a.soft, bam=out_bam is not None,
                posterior_tags=getattr(a, "posterior_tags", False),
                calib=(calib.cut, calib.rate) if calib is not None else None)
    if opts["posterior_tags"] and out_bam is None:
        raise SystemExit("--posterior-tags needs --out-bam")

    def emit(results):
        nonlocal n, t_write
        t0 = time.perf_counter()
        for r in results:
            if r is None:
                continue
            bed, tagged = r
            out.write(bed)
            if out_bam is not None:
                out_bam.write(pysam.AlignedSegment.fromstring(tagged, out_bam.header))
            n += 1
        t_write += time.perf_counter() - t0

    chunk_size = max(1, a.chunk_size)
    window = max(2, 3 * a.threads)          # chunks in flight: keeps every worker busy
    pending = collections.deque()
    it = (r.to_string() for r in bam if not (r.is_secondary or r.is_supplementary))
    with ProcessPoolExecutor(a.threads, initializer=_init_worker,
                             initargs=(params, ctx, not a.no_seq, bam.header.to_dict(), opts)) as ex:
        done_reading = False
        while not done_reading or pending:
            while not done_reading and len(pending) < window:
                t0 = time.perf_counter()
                payload = []
                for rec in it:
                    payload.append(rec)
                    if len(payload) >= chunk_size:
                        break
                t_read += time.perf_counter() - t0
                if not payload:
                    done_reading = True
                    break
                pending.append(ex.submit(_call_chunk, payload))
            if pending:
                fut = pending.popleft()
                t0 = time.perf_counter()
                results, cpu = fut.result()
                t_wait += time.perf_counter() - t0
                model_cpu += cpu
                emit(results)
    if out_bam is not None:
        out_bam.close()
    if out is not sys.stdout:
        out.close()
    wall = time.perf_counter() - t_start
    print(f"fiberstate: called {n} fibers in {wall:.1f} s ({1000 * wall / max(1, n):.1f} ms/fiber wall, "
          f"{a.threads} threads)", file=sys.stderr)
    if a.timing:
        print(f"fiberstate timing: workers {model_cpu:.1f} s CPU incl. m6A decoding and tags "
              f"({1000 * model_cpu / max(1, n):.1f} ms/fiber), main process: reading {t_read:.1f} s, "
              f"writing {t_write:.1f} s, waiting on workers {t_wait:.1f} s", file=sys.stderr)


def _read_bed(path):
    sites = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            p = line.rstrip("\n").split("\t")
            name = p[3] if len(p) > 3 else f"{p[0]}:{p[1]}-{p[2]}"
            sites.append((p[0], int(p[1]), int(p[2]), name))
    return sites


def cmd_sites(a):
    import pysam
    from .io import fiber_arrays, ref_to_query
    from .sites import co_occupancy, occupancy
    params = _params_from_args(a)
    model = Model(params, context=_context_from_args(a))
    bam = pysam.AlignmentFile(a.bam, "rb")
    sites = _read_bed(a.sites)
    by_chrom = {}
    for i, s in enumerate(sites):
        by_chrom.setdefault(s[0], []).append(i)
    obs = {i: {} for i in range(len(sites))}
    for chrom, idxs in by_chrom.items():
        lo = min(sites[i][1] for i in idxs)
        hi = max(sites[i][2] for i in idxs)
        seen = set()
        for read in bam.fetch(chrom, lo, hi):
            if read.is_secondary or read.is_supplementary or read.query_name in seen:
                continue
            seen.add(read.query_name)
            s, m = fiber_arrays(read, a.ml_threshold, a.soft)
            if s is None:
                continue
            r2q = None
            hits = [i for i in idxs if read.reference_start <= (sites[i][1] + sites[i][2]) // 2 < read.reference_end]
            if not hits:
                continue
            r2q = ref_to_query(read)
            c = model.call_read(s, m, sequence_aware=not a.no_seq)
            for i in hits:
                ctr = (sites[i][1] + sites[i][2]) // 2
                qp = r2q.get(ctr)
                if qp is None:
                    continue
                obs[i][read.query_name] = (float(c.post_linker[qp]), float(c.post_nuc[qp]),
                                           float(c.post_foot[qp]))
    with open(a.output, "w") as f:
        f.write("chrom\tstart\tend\tname\tn_fibers\toccupancy\tci_low\tci_high\taccessible\tocc_given_accessible\n")
        for i, (ch, s, e, nm) in enumerate(sites):
            o = occupancy(list(obs[i].values()))
            f.write(f"{ch}\t{s}\t{e}\t{nm}\t{o['n']}\t{o['occupancy']:.4f}\t{o['ci_low']:.4f}\t"
                    f"{o['ci_high']:.4f}\t{o['accessible']:.4f}\t{o['occ_given_accessible']:.4f}\n")
    if a.pairs:
        with open(a.pairs, "w") as f:
            f.write("name_i\tname_j\tdistance\tn_eff\tlog_odds_ratio\tse\tz\tp\tinformative\n")
            for chrom, idxs in by_chrom.items():
                idxs = sorted(idxs, key=lambda i: sites[i][1])
                for i, j in combinations(idxs, 2):
                    d = abs((sites[j][1] + sites[j][2]) // 2 - (sites[i][1] + sites[i][2]) // 2)
                    if d > a.max_dist:
                        continue
                    common = [r for r in obs[i] if r in obs[j]]
                    if len(common) < a.min_fibers:
                        continue
                    r = co_occupancy([obs[i][x] for x in common], [obs[j][x] for x in common],
                                     condition_on_access=not a.unconditioned)
                    f.write(f"{sites[i][3]}\t{sites[j][3]}\t{d}\t{r['n']:.1f}\t{r['log_or']:.4f}\t"
                            f"{r['se']:.4f}\t{r['z']:.3f}\t{r['p']:.3g}\t{int(r['informative'])}\n")
    print(f"fiberstate: wrote {len(sites)} sites", file=sys.stderr)


def cmd_calibrate(a):
    from .calibrate import calibrate, merge_tables, write_table
    if a.merge:
        merge_tables(a.inputs, a.output)
        print(f"fiberstate: pooled {len(a.inputs)} tables into {a.output}", file=sys.stderr)
        return
    if len(a.inputs) != 1 or not a.negatives:
        raise SystemExit("calibrate needs one BAM and --negatives (or --merge TABLE.tsv ...)")
    rows = calibrate(a.inputs[0], a.negatives, a.positives, params=_params_from_args(a), ctx=_context_from_args(a),
                     seq_aware=not a.no_seq, ml_threshold=a.ml_threshold, threads=a.threads)
    write_table(rows, a.output)
    for r in rows:
        if abs(r["cutoff"] - 0.5) < 1e-9:
            print(f"fiberstate: {r['n_negative']} molecules at unbound sites, {r['n_positive']} at bound sites; "
                  f"cut-off 0.50: false-call rate {100 * r['false_call_rate']:.2f}%, "
                  f"recovered {100 * r['recovered_rate']:.2f}%", file=sys.stderr)


def cmd_simulate(a):
    import dataclasses
    from .io import write_sim_bam
    try:
        from bench.simulate import SimConfig, simulate
    except ImportError:
        sys.exit("fiberstate simulate needs the bench/ package (run from the repo root)")
    cfg = SimConfig(n_loci=max(1, a.reads // 20), reads_per_locus=20,
                    sensitivity_mean=a.sensitivity, seed=a.seed)
    loci, reads, sites = simulate(cfg)
    write_sim_bam(a.output, reads)
    names = {1: "nucleosome", 2: "footprint"}
    with open(a.truth, "w") as f:
        for r in reads:
            for k, s, e in r.segments:
                f.write(f"{r.name}\t{s}\t{e}\t{names[k]}\n")
    print(f"fiberstate: simulated {len(reads)} fibers -> {a.output}", file=sys.stderr)


def cmd_learn_context(a):
    import pysam
    from .io import fiber_arrays
    from .learn import learn_context
    fibers = []
    with pysam.AlignmentFile(a.bam, "rb", check_sq=False) as b:
        for r in b:
            if r.is_secondary or r.is_supplementary:
                continue
            s, m = fiber_arrays(r, a.ml_threshold)
            if s is not None and len(s) >= 1000:
                fibers.append((s, m))
            if len(fibers) >= a.max_fibers:
                break
    ctx, stats = learn_context(fibers, k=a.k, workers=a.threads)
    ctx.save(a.output)
    st = stats[-1]
    print(f"fiberstate: learned {a.k}-mer marking weights from {len(fibers)} fibers "
          f"({st['n_sites']:.0f} accessible sites; 5-95% range {st['w_q05']:.2f}-{st['w_q95']:.2f}) -> {a.output}",
          file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fiberstate",
                                 description="Sequence-aware single-molecule chromatin calling for Fiber-seq.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--ml-threshold", type=float, default=0.5,
                       help="ML probability above which an A is called m6A (hard mode)")
        p.add_argument("--soft", action="store_true", help="use ML probabilities as soft evidence")
        p.add_argument("--p-access", type=float, default=0.6)
        p.add_argument("--p-protect", type=float, default=0.03)
        p.add_argument("--mean-linker", type=float, default=45.0)
        p.add_argument("--frac-nuc", type=float, default=0.75)
        p.add_argument("--no-em", action="store_true", help="disable per-fiber rate estimation")
        p.add_argument("--no-seq", action="store_true", help="disable sequence awareness (ablation)")
        p.add_argument("--em-iters", type=int, default=4,
                       help="per-fiber EM rounds (fewer = faster; 2 is ~1.5x faster, less precise rates)")
        p.add_argument("--foot-posterior", type=float, default=0.5,
                       help="also call a footprint where its posterior exceeds this inside Viterbi linker "
                            "runs (default 0.5, chosen on CTCF ChIP-seq in GM12878 and K562; higher = more "
                            "specific, less sensitive; 0 = Viterbi only, as in versions before 0.3)")
        p.add_argument("--keep-inferred", action="store_true",
                       help="keep nucleosome positions inside unmarked stretches instead of reporting "
                            "them as one 'protected_unresolved' segment")
        p.add_argument("--context", default="default",
                       help="Hia5 sequence-context table: 'default' (learned on GM12878 Fiber-seq), "
                            "'none', or a path from `fiberstate learn-context`")

    p = sub.add_parser("call", help="call nucleosomes / footprints on every fiber")
    p.add_argument("bam"); p.add_argument("-o", "--output", default="-")
    p.add_argument("--out-bam"); p.add_argument("-t", "--threads", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=16, help="fibers per worker task")
    p.add_argument("--timing", action="store_true", help="report where the time went")
    p.add_argument("--posterior-tags", action="store_true",
                   help="also write per-base posteriors to --out-bam: fp = footprint, np = nucleosome (0-255, SEQ frame)")
    p.add_argument("--false-call-rate", type=float, default=None,
                   help="choose the footprint cut-off for this false-call rate (share of molecules called "
                        "footprinted at unbound motif sites, e.g. 0.01) from a calibration table; adds each "
                        "footprint's estimated false-call rate (BED column 11, BAM tag fe)")
    p.add_argument("--calibration", default="default",
                   help="calibration table from `fiberstate calibrate` ('default' = built-in, from ChIP-labelled "
                        "CTCF, REST and NRF1 sites in GM12878 and K562)")
    common(p); p.set_defaults(func=cmd_call)

    p = sub.add_parser("calibrate", help="false-call rate of each footprint cut-off, at motif sites known to be unbound")
    p.add_argument("inputs", nargs="+", help="aligned, indexed BAM with MM/ML (or calibration tables with --merge)")
    p.add_argument("--negatives", help="BED of motif sites known to be unbound (e.g. no ChIP peak within 2 kb)")
    p.add_argument("--positives", help="optional BED of bound motif sites (reports the recovered share)")
    p.add_argument("--merge", action="store_true", help="pool several calibration tables (molecule-weighted)")
    p.add_argument("-o", "--output", required=True); p.add_argument("-t", "--threads", type=int, default=1)
    common(p)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("sites", help="per-site occupancy and pairwise co-occupancy")
    p.add_argument("bam", help="aligned, indexed BAM"); p.add_argument("sites", help="BED of sites")
    p.add_argument("-o", "--output", required=True); p.add_argument("--pairs")
    p.add_argument("--max-dist", type=int, default=1000); p.add_argument("--min-fibers", type=int, default=10)
    p.add_argument("--unconditioned", action="store_true",
                   help="do not condition co-occupancy on accessibility")
    common(p); p.set_defaults(func=cmd_sites)

    p = sub.add_parser("learn-context", help="learn Hia5 sequence-context marking weights from your data")
    p.add_argument("bam"); p.add_argument("-o", "--output", required=True)
    p.add_argument("-k", type=int, default=5); p.add_argument("--max-fibers", type=int, default=3000)
    p.add_argument("-t", "--threads", type=int, default=2)
    p.add_argument("--ml-threshold", type=float, default=0.5)
    p.set_defaults(func=cmd_learn_context)

    p = sub.add_parser("simulate", help="write a simulated Fiber-seq BAM with ground truth")
    p.add_argument("-o", "--output", required=True); p.add_argument("--truth", required=True)
    p.add_argument("--reads", type=int, default=200); p.add_argument("--sensitivity", type=float, default=0.65)
    p.add_argument("--seed", type=int, default=0); p.set_defaults(func=cmd_simulate)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
