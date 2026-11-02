"""Calibrate the footprint cut-off to a false-call rate.

A *false call* is a molecule called footprinted at a motif site that ChIP-seq says is unbound:
one footprint call covers at least 80% of the motif (the rule used in the benchmarks). The
false-call rate at a cut-off is the share of molecules at unbound sites called at that cut-off.

`fiberstate calibrate` runs the model once per molecule at the given sites and applies the exact
calling steps of `fiberstate call` at every cut-off (posterior footprints inside linker runs,
then merging of unsupported segments), so the table describes what `call` will report.
`fiberstate call --false-call-rate R` reads such a table, uses the lowest cut-off whose false-call
rate is <= R, and reports for every footprint the false-call rate at a cut-off equal to its own
mean posterior (its estimated false-call rate).
"""
from __future__ import annotations

import os

import numpy as np

GRID = tuple(round(x, 2) for x in np.arange(0.05, 0.96, 0.05))
DEFAULT_TABLE = os.path.join(os.path.dirname(__file__), "data", "footprint_calibration.tsv")
_W = {}


def _read_sites(p):
    out = []
    with open(p) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            c = line.rstrip("\n").split("\t")
            out.append((c[0], int(c[1]), int(c[2])))
    return out


def _init(params, ctx, seq_aware, header, ml_threshold, grid):
    import pysam
    from .model import Model
    try:
        from threadpoolctl import threadpool_limits
        _W["tp"] = threadpool_limits(limits=1)
    except ImportError:
        pass
    _W.update(model=Model(params, context=ctx), seq_aware=seq_aware, ml=ml_threshold, grid=grid,
              header=pysam.AlignmentHeader.from_dict(header))


def _work(items):
    """items: (SAM text, [(site_id, label, start, end)]) -> [(site_id, label, [called at each cut-off])]."""
    import pysam
    from .io import fiber_arrays, query_to_ref
    out = []
    for rec, sites in items:
        r = pysam.AlignedSegment.fromstring(rec, _W["header"])
        q2r = query_to_ref(r)
        spans = []
        for sid, lab, s, e in sites:
            qs = np.flatnonzero((q2r >= s) & (q2r < e))
            c = (s + e) // 2
            if len(qs) < min(12, int(0.8 * (e - s))) or not np.any(q2r[qs] == c):
                continue
            spans.append((sid, lab, int(qs.min()), int(qs.max()) + 1))
        if not spans:
            continue
        seq, m = fiber_arrays(r, ml_threshold=_W["ml"])
        if seq is None or len(seq) < 200:
            continue
        feet = _W["model"].footprints_at_cutoffs(seq, (m > 0).astype(np.uint8), _W["grid"], _W["seq_aware"])
        for sid, lab, qa, qb in spans:
            need = 0.8 * (qb - qa)
            called = [any(min(b, qb) - max(a, qa) >= need for a, b in feet[t]) for t in _W["grid"]]
            out.append((sid, lab, called))
    return out


def calibrate(bam_path, negatives, positives=None, params=None, ctx=None, seq_aware=True,
              ml_threshold=0.5, threads=1, grid=GRID, chunk=16):
    import pysam
    from concurrent.futures import ProcessPoolExecutor
    from .model import Params
    params = params or Params()
    labeled = [(s, 0) for s in _read_sites(negatives)] + ([(s, 1) for s in _read_sites(positives)] if positives else [])
    bam = pysam.AlignmentFile(bam_path)
    by_read = {}
    for sid, ((ch, s, e), lab) in enumerate(labeled):
        for r in bam.fetch(ch, s, e):
            if r.is_secondary or r.is_supplementary or r.reference_start > s or r.reference_end < e:
                continue
            ent = by_read.setdefault(r.query_name, [r.to_string(), []])
            ent[1].append((sid, lab, s, e))
    items = list(by_read.values())
    calls = {0: [], 1: []}
    with ProcessPoolExecutor(max(1, threads), initializer=_init,
                             initargs=(params, ctx, seq_aware, bam.header.to_dict(), ml_threshold, grid)) as ex:
        for res in ex.map(_work, [items[i:i + chunk] for i in range(0, len(items), chunk)]):
            for sid, lab, called in res:
                calls[lab].append(called)
    neg = np.array(calls[0], dtype=bool).reshape(-1, len(grid))
    pos = np.array(calls[1], dtype=bool).reshape(-1, len(grid))
    rows = []
    for j, t in enumerate(grid):
        rows.append(dict(cutoff=t, false_call_rate=float(neg[:, j].mean()) if len(neg) else float("nan"),
                         recovered_rate=float(pos[:, j].mean()) if len(pos) else float("nan"),
                         n_negative=len(neg), n_positive=len(pos)))
    return rows


def write_table(rows, path):
    with open(path, "w") as f:
        f.write("cutoff\tfalse_call_rate\trecovered_rate\tn_negative\tn_positive\n")
        for r in rows:
            f.write(f"{r['cutoff']:.2f}\t{r['false_call_rate']:.5f}\t{r['recovered_rate']:.5f}\t"
                    f"{r['n_negative']}\t{r['n_positive']}\n")


def merge_tables(paths, out_path):
    """Pool several calibration tables (molecule-weighted) into one."""
    acc = {}
    for p in paths:
        with open(p) as f:
            f.readline()
            for line in f:
                c = line.rstrip("\n").split("\t")
                t = float(c[0]); fr, rr, nn, npos = float(c[1]), float(c[2]), int(c[3]), int(c[4])
                a = acc.setdefault(t, [0.0, 0, 0.0, 0])
                a[0] += fr * nn; a[1] += nn
                if npos and rr == rr:
                    a[2] += rr * npos; a[3] += npos
    rows = [dict(cutoff=t, false_call_rate=a[0] / max(1, a[1]), recovered_rate=(a[2] / a[3]) if a[3] else float("nan"),
                 n_negative=a[1], n_positive=a[3]) for t, a in sorted(acc.items())]
    write_table(rows, out_path)
    return rows


class Calibration:
    """cut-off <-> false-call rate, monotone (rates made non-increasing in the cut-off)."""

    def __init__(self, path=None):
        path = DEFAULT_TABLE if path in (None, "default") else path
        if not os.path.exists(path):
            raise SystemExit(f"no calibration table at {path}; make one with `fiberstate calibrate`")
        cut, rate = [], []
        with open(path) as f:
            f.readline()
            for line in f:
                c = line.split("\t")
                if c[1] not in ("nan", ""):
                    cut.append(float(c[0])); rate.append(float(c[1]))
        o = np.argsort(cut)
        self.cut = np.array(cut)[o]
        self.rate = np.minimum.accumulate(np.array(rate)[o])   # a higher cut-off never raises the rate
        self.path = path

    def cutoff_for(self, target):
        ok = np.flatnonzero(self.rate <= target)
        if len(ok) == 0:
            raise SystemExit(f"no cut-off in {self.path} reaches a false-call rate of {target} "
                             f"(lowest is {self.rate.min():.4f} at {self.cut[-1]})")
        return float(self.cut[ok[0]])

    def rate_at(self, posterior):
        """Estimated false-call rate of a footprint whose mean posterior is `posterior`."""
        return float(np.interp(posterior, self.cut, self.rate, left=self.rate[0], right=self.rate[-1]))
