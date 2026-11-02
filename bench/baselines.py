"""Baseline callers used in the benchmark.

gap_caller   - fibertools-style: protected segments are runs of the read with no m6A
               call; length thresholds split nucleosomes from footprints.
               (A re-implementation of the idea, with thresholds tuned on held-out
               simulated data so the baseline is given its best shot.)
naive_hmm    - classic 3-state per-base HMM with geometric durations; every base is an
               observation (G/C bases count as "unmethylated"), global emission rates.
"""
from __future__ import annotations

import numpy as np
from numba import njit

LINKER, NUC, FOOT = 0, 1, 2


def gap_caller(m6a: np.ndarray, nuc_min=85, foot_min=15):
    T = len(m6a)
    pos = np.flatnonzero(m6a)
    edges = np.concatenate(([-1], pos, [T]))
    path = np.zeros(T, dtype=np.uint8)
    segs = []
    for a, b in zip(edges[:-1], edges[1:]):
        s, e = a + 1, b
        ln = e - s
        if ln >= nuc_min:
            path[s:e] = NUC
            segs.append((NUC, int(s), int(e), 1.0))
        elif ln >= foot_min:
            path[s:e] = FOOT
            segs.append((FOOT, int(s), int(e), 1.0))
    return path, segs


@njit(cache=True)
def _viterbi3(obs, logA, logE1, logE0, logpi):
    T = obs.shape[0]
    K = 3
    d = np.empty((T, K))
    bp = np.zeros((T, K), dtype=np.int8)
    for k in range(K):
        d[0, k] = logpi[k] + (logE1[k] if obs[0] else logE0[k])
    for t in range(1, T):
        for k in range(K):
            best = -1e300; arg = 0
            for j in range(K):
                v = d[t - 1, j] + logA[j, k]
                if v > best:
                    best = v; arg = j
            d[t, k] = best + (logE1[k] if obs[t] else logE0[k])
            bp[t, k] = arg
    path = np.empty(T, dtype=np.uint8)
    k = int(np.argmax(d[T - 1]))
    for t in range(T - 1, -1, -1):
        path[t] = k
        k = bp[t, k]
    return path


def naive_hmm(m6a: np.ndarray, p_access=0.35, p_protect=0.012,
              mean_linker=45.0, mean_nuc=147.0, mean_foot=28.0, frac_nuc=0.92):
    """p_access is the per-*base* (not per-A) methylation rate, since G/C count as zeros."""
    A = np.zeros((3, 3))
    A[0, 0] = 1 - 1 / mean_linker
    A[0, 1] = (1 / mean_linker) * frac_nuc
    A[0, 2] = (1 / mean_linker) * (1 - frac_nuc)
    A[1, 1] = 1 - 1 / mean_nuc
    A[1, 0] = 1 / mean_nuc
    A[2, 2] = 1 - 1 / mean_foot
    A[2, 0] = 1 / mean_foot
    p = np.array([p_access, p_protect, p_protect])
    path = _viterbi3(m6a.astype(np.bool_), np.log(A + 1e-300), np.log(p), np.log(1 - p),
                     np.log(np.array([0.3, 0.6, 0.1])))
    segs = []
    change = np.flatnonzero(np.diff(path.astype(np.int16))) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(path)]))
    for s, e in zip(starts, ends):
        k = int(path[s])
        if k != LINKER:
            segs.append((k, int(s), int(e), 1.0))
    return path, segs
