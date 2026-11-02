"""Learn the sequence-context marking table from real fibers."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .context import ContextTable, context_codes
from .model import Model, Params

MIN_LINKER = 60   # only count bases inside long accessible stretches
POST = 0.95


def _counts(args):
    items, k, ctx = args
    m = Model(Params(), context=ctx)
    cm = np.zeros(4 ** k)
    cn = np.zeros(4 ** k)
    for seq, m6a in items:
        c = m.call_read(seq, m6a)
        codes = context_codes(seq, k)
        acc = np.zeros(len(seq), bool)
        # long Viterbi linker runs with high posterior
        path = c.path
        change = np.flatnonzero(np.diff(path.astype(np.int16))) + 1
        st = np.concatenate(([0], change)); en = np.concatenate((change, [len(path)]))
        for s, e in zip(st, en):
            if path[s] == 0 and e - s >= MIN_LINKER:
                acc[s + 5:e - 5] = True
        sel = acc & (c.post_linker > POST) & (codes >= 0)
        # normalise by the fiber's own rate so fibers with high marking don't dominate
        r = max(1e-3, c.p_access)
        np.add.at(cm, codes[sel], (m6a[sel] > 0) / r)
        np.add.at(cn, codes[sel], 1.0)
    return cm, cn


def learn_context(fibers, k=5, iterations=2, workers=2, shrink=20.0):
    """fibers: list of (seq uint8, m6a array). Returns (ContextTable, stats)."""
    ctx = None
    stats = []
    for it in range(iterations):
        chunks = [(fibers[i::workers], k, ctx) for i in range(workers)]
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(_counts, chunks))
        cm = sum(p[0] for p in parts)
        cn = sum(p[1] for p in parts)
        ctx = ContextTable.fit(cm, cn, k, shrink)
        w = ctx.weights[cn > 50]
        stats.append(dict(iteration=it, n_sites=float(cn.sum()), kmers_with_data=int((cn > 50).sum()),
                          w_q05=float(np.quantile(w, .05)), w_q50=float(np.median(w)),
                          w_q95=float(np.quantile(w, .95))))
    return ctx, stats
