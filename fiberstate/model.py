"""High-level model: parameters, per-read EM, segment calling."""
from __future__ import annotations

import dataclasses
import numpy as np

from .hsmm_lin import fb as forward_backward, viterbi

LINKER, NUC, FOOT, UNRESOLVED = 0, 1, 2, 3
NUC_REPEAT = 185.0   # typical human nucleosome repeat length, used only to estimate counts
_METHYLATABLE = np.zeros(256, dtype=bool)
for _b in b"ATat":
    _METHYLATABLE[_b] = True


def _discrete_mixture(R, comps):
    """pmf over durations 1..R from a list of (weight, mean, sd, lo, hi) normals."""
    d = np.arange(1, R + 1, dtype=float)
    p = np.zeros(R)
    for w, mu, sd, lo, hi in comps:
        x = np.exp(-0.5 * ((d - mu) / sd) ** 2)
        x[(d < lo) | (d > hi)] = 0
        p += w * x / x.sum()
    return p / p.sum()


@dataclasses.dataclass
class Params:
    p_access: float = 0.6       # m6A detection prob at methylatable base in accessible DNA
    p_protect: float = 0.03     # m6A call prob inside protected DNA (false positives / breathing)
    mean_linker: float = 45.0
    frac_nuc: float = 0.75      # P(next protected segment is a nucleosome | leaving linker); tuned on held-out sim seed 999
    nuc_max: int = 200
    foot_max: int = 90
    learn_per_read: bool = True
    em_iters: int = 4
    prior_strength: float = 50.0   # Beta pseudo-count weight toward the global rates
    min_informative: int = 30      # skip EM on reads with fewer methylatable bases
    foot_posterior: float | None = 0.5   # inside Viterbi linker runs, call a footprint where its posterior
                                   # exceeds this (None = pure Viterbi). Footprint lengths vary, so the
                                   # footprint state's probability is spread over many paths and Viterbi
                                   # under-calls it; posterior marginals do not have that bias.
    foot_min_len: int = 8
    merge_unsupported: bool = True # merge protected calls not separated by any m6A into one
                                   # 'unresolved' segment (positions inside are not identifiable)

    def durations(self):
        dN = _discrete_mixture(self.nuc_max, [(0.88, 147, 7, 120, 200), (0.12, 110, 10, 90, 130)])
        dF = _discrete_mixture(self.foot_max, [(1.0, 28, 12, 8, 90)])
        return dN, dF


class Model:
    def __init__(self, params: Params | None = None, context=None):
        """``context``: optional ContextTable of Hia5 sequence-context marking weights."""
        self.p = params or Params()
        self.context = context
        self.dN, self.dF = self.p.durations()
        # residual-life initial distributions for left-truncated first segment
        sN = np.cumsum(self.dN[::-1])[::-1]
        sF = np.cumsum(self.dF[::-1])[::-1]
        self.initN = sN / sN.sum()
        self.initF = sF / sF.sum()
        mN = (np.arange(1, len(self.dN) + 1) * self.dN).sum()
        mF = (np.arange(1, len(self.dF) + 1) * self.dF).sum()
        wL = self.p.mean_linker
        wN = self.p.frac_nuc * mN
        wF = (1 - self.p.frac_nuc) * mF
        tot = wL + wN + wF
        self.pi = (wL / tot, wN / tot, wF / tot)
        self.q = 1.0 - 1.0 / self.p.mean_linker

    # ------------------------------------------------------------------
    @staticmethod
    def observations(seq: np.ndarray, m6a: np.ndarray, sequence_aware: bool = True) -> np.ndarray:
        """Build the observation vector. ``seq`` is uint8 ASCII, ``m6a`` is bool or prob in [0,1]."""
        obs = np.asarray(m6a, dtype=np.float64).copy()
        if sequence_aware:
            obs[~_METHYLATABLE[seq]] = -1.0
        return obs

    def _run(self, obs, w, pL, pP, decode=True):
        args = (obs, w, pL, pP, pP, self.q, self.p.frac_nuc, self.dN, self.dF,
                self.initN, self.initF, *self.pi)
        gL, gN, gF, ll = forward_backward(*args)
        path = viterbi(*args) if decode else None
        return gL, gN, gF, ll, path

    def fit_read(self, obs, w):
        """Per-read EM over the two emission rates (MAP with Beta prior toward globals)."""
        pL, pP = self.p.p_access, self.p.p_protect
        inf = obs >= 0
        if not self.p.learn_per_read or inf.sum() < self.p.min_informative:
            return pL, pP
        k = self.p.prior_strength
        o = obs[inf]
        wi = w[inf]
        for _ in range(self.p.em_iters):
            gL, gN, gF, _, _ = self._run(obs, w, pL, pP, decode=False)
            wl = gL[inf]
            wp = 1.0 - wl
            # accessible rate is pL * w[t]: MLE of the shared factor (small-rate approximation)
            pL = (np.dot(wl, o) + k * self.p.p_access) / (np.dot(wl, wi) + k)
            pP = (np.dot(wp, o) + k * self.p.p_protect) / (wp.sum() + k)
            pL = float(np.clip(pL, 0.05, 0.99))
            pP = float(np.clip(pP, 1e-4, min(0.5, pL * 0.8)))
        return pL, pP

    def call_read(self, seq, m6a, sequence_aware=True):
        obs = self.observations(seq, m6a, sequence_aware)
        w = self.context.per_base(seq) if (self.context is not None and sequence_aware) else np.ones(len(obs))
        pL, pP = self.fit_read(obs, w)
        gL, gN, gF, ll, path = self._run(obs, w, pL, pP, decode=True)
        if self.p.foot_posterior is not None:
            path = add_posterior_footprints(path, gF, self.p.foot_posterior, self.p.foot_min_len)
        segs = segments_from_path(path, gN, gF)
        if self.p.merge_unsupported:
            path, segs = merge_unsupported(path, segs, obs, gN, gF)
        return ReadCall(path=path, post_linker=gL, post_nuc=gN, post_foot=gF,
                        p_access=pL, p_protect=pP, loglik=ll, segments=segs)

    def footprints_at_cutoffs(self, seq, m6a, cutoffs, sequence_aware=True):
        """Run the model once, then apply the call steps (posterior footprints, merging) at each
        cut-off. Returns {cutoff: [(start, end), ...] footprint calls}; None = Viterbi only.
        At the cut-off in use, this equals call_read's footprint segments."""
        obs = self.observations(seq, m6a, sequence_aware)
        w = self.context.per_base(seq) if (self.context is not None and sequence_aware) else np.ones(len(obs))
        pL, pP = self.fit_read(obs, w)
        gL, gN, gF, ll, vpath = self._run(obs, w, pL, pP, decode=True)
        out = {}
        for t in cutoffs:
            path = vpath if t is None else add_posterior_footprints(vpath, gF, t, self.p.foot_min_len)
            segs = segments_from_path(path, gN, gF)
            if self.p.merge_unsupported:
                path, segs = merge_unsupported(path, segs, obs, gN, gF)
            out[t] = [(s_, e_) for k, s_, e_, _ in segs if k == FOOT]
        return out


@dataclasses.dataclass
class ReadCall:
    path: np.ndarray
    post_linker: np.ndarray
    post_nuc: np.ndarray
    post_foot: np.ndarray
    p_access: float
    p_protect: float
    loglik: float
    segments: list      # (state, start, end, mean_posterior); state 3 = protected, positions unresolved

    def n_nucleosomes_unresolved(self):
        return [max(1, int(round((e - s) / NUC_REPEAT))) for k, s, e, _ in self.segments if k == UNRESOLVED]


def segments_from_path(path, gN=None, gF=None):
    segs = []
    T = len(path)
    if T == 0:
        return segs
    change = np.flatnonzero(np.diff(path.astype(np.int16))) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [T]))
    for s, e in zip(starts, ends):
        k = int(path[s])
        if k == LINKER:
            continue
        conf = float(np.mean((gN if k == NUC else gF)[s:e])) if gN is not None else 1.0
        segs.append((k, int(s), int(e), conf))
    return segs


def merge_unsupported(path, segs, obs, gN, gF):
    """Merge runs of protected segments whose separating linkers carry no m6A call.

    A boundary between two protected calls is only evidenced by marks in the linker
    between them. When a linker has no m6A, the split point (and so the nucleosome
    positions) is set by the length prior alone, so the whole run is reported as one
    'unresolved' protected segment instead.
    Runs of a single segment are left unchanged.
    """
    if len(segs) < 2:
        return path, segs
    marked = obs > 0
    out = []
    path = path.copy()
    i = 0
    n = len(segs)
    while i < n:
        j = i
        while j + 1 < n and not marked[segs[j][2]:segs[j + 1][1]].any():
            j += 1
        if j > i:
            s, e = segs[i][1], segs[j][2]
            conf = float(np.mean(gN[s:e] + gF[s:e]))
            out.append((UNRESOLVED, s, e, conf))
            path[s:e] = UNRESOLVED
        else:
            out.append(segs[i])
        i = j + 1
    return path, out


def add_posterior_footprints(path, gF, thr=0.5, min_len=8):
    """Relabel linker bases whose footprint posterior exceeds `thr` as footprint (runs >= min_len).

    Only linker bases are changed, so Viterbi nucleosome calls are untouched."""
    cand = (path == LINKER) & (gF > thr)
    if not cand.any():
        return path
    path = path.copy()
    d = np.diff(np.concatenate(([0], cand.view(np.int8), [0])))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a >= min_len:
            path[a:b] = FOOT
    return path
