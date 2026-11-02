"""Numba kernels for a sequence-aware explicit-duration HSMM over single DNA molecules.

States
------
0  linker / accessible DNA         geometric duration (self-loop prob q)
1  nucleosome (protected)          explicit duration pmf dN[r], r = 1..RN
2  footprint (small protected)     explicit duration pmf dF[r], r = 1..RF

Observation at each base ``obs[t]``:
  * ``-1``      base cannot be methylated (G/C) -> emission = 1 for every state,
                i.e. the base carries no evidence. This is the "sequence-aware" part:
                an m6A-free stretch that is GC-rich is *not* evidence of protection.
  * ``0..1``    probability that the methylatable base carries m6A (0/1 for hard calls,
                or the caller's ML probability for soft calls).

``wL[t]`` multiplies the accessible marking rate at base t (sequence-context weight of the
Hia5 enzyme; all ones for a context-blind model).

Durations use the "countdown" formulation: entering a segment draws its total length d,
the state then carries the remaining length r = d, d-1, ..., 1.
"""
import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _cl(p):
    return min(0.99, max(1e-4, p))


@njit(cache=True, inline="always")
def _emit(o, p):
    if o < 0.0:
        return 1.0
    return o * p + (1.0 - o) * (1.0 - p)


@njit(cache=True)
def forward_backward(obs, wL, pL, pN, pF, q, aN, dN, dF, initN, initF, piL, piN, piF):
    """Scaled forward-backward. Returns (gL, gN, gF, loglik), per-base posteriors."""
    T = obs.shape[0]
    RN = dN.shape[0]
    RF = dF.shape[0]
    aF = 1.0 - aN
    aL = np.empty(T)
    aNm = np.empty((T, RN))
    aFm = np.empty((T, RF))
    c = np.empty(T)

    # t = 0
    eL = _emit(obs[0], _cl(pL * wL[0])); eN = _emit(obs[0], pN); eF = _emit(obs[0], pF)
    s = 0.0
    aL[0] = piL * eL
    s += aL[0]
    for r in range(RN):
        aNm[0, r] = piN * initN[r] * eN
        s += aNm[0, r]
    for r in range(RF):
        aFm[0, r] = piF * initF[r] * eF
        s += aFm[0, r]
    c[0] = s
    inv = 1.0 / s
    aL[0] *= inv
    for r in range(RN):
        aNm[0, r] *= inv
    for r in range(RF):
        aFm[0, r] *= inv

    for t in range(1, T):
        o = obs[t]
        eL = _emit(o, _cl(pL * wL[t])); eN = _emit(o, pN); eF = _emit(o, pF)
        lexit = aL[t - 1] * (1.0 - q)
        aL[t] = (aL[t - 1] * q + aNm[t - 1, 0] + aFm[t - 1, 0]) * eL
        s = aL[t]
        enN = lexit * aN
        enF = lexit * aF
        for r in range(RN):
            prev = aNm[t - 1, r + 1] if r + 1 < RN else 0.0
            v = (prev + enN * dN[r]) * eN
            aNm[t, r] = v
            s += v
        for r in range(RF):
            prev = aFm[t - 1, r + 1] if r + 1 < RF else 0.0
            v = (prev + enF * dF[r]) * eF
            aFm[t, r] = v
            s += v
        c[t] = s
        inv = 1.0 / s
        aL[t] *= inv
        for r in range(RN):
            aNm[t, r] *= inv
        for r in range(RF):
            aFm[t, r] *= inv

    # backward
    bL = np.ones(T)
    bN = np.ones((T, RN))
    bF = np.ones((T, RF))
    for t in range(T - 2, -1, -1):
        o = obs[t + 1]
        ic = 1.0 / c[t + 1]
        eL1 = _emit(o, _cl(pL * wL[t + 1])) * ic
        eN1 = _emit(o, pN) * ic
        eF1 = _emit(o, pF) * ic
        for r in range(RN - 1, 0, -1):
            bN[t, r] = eN1 * bN[t + 1, r - 1]
        bN[t, 0] = eL1 * bL[t + 1]
        for r in range(RF - 1, 0, -1):
            bF[t, r] = eF1 * bF[t + 1, r - 1]
        bF[t, 0] = eL1 * bL[t + 1]
        sN = 0.0
        for r in range(RN):
            sN += dN[r] * bN[t + 1, r]
        sF = 0.0
        for r in range(RF):
            sF += dF[r] * bF[t + 1, r]
        bL[t] = q * eL1 * bL[t + 1] + (1.0 - q) * (aN * sN * eN1 + aF * sF * eF1)

    gL = np.empty(T)
    gN = np.empty(T)
    gF = np.empty(T)
    ll = 0.0
    for t in range(T):
        ll += np.log(c[t])
        x = aL[t] * bL[t]
        y = 0.0
        for r in range(RN):
            y += aNm[t, r] * bN[t, r]
        z = 0.0
        for r in range(RF):
            z += aFm[t, r] * bF[t, r]
        tot = x + y + z
        gL[t] = x / tot
        gN[t] = y / tot
        gF[t] = z / tot
    return gL, gN, gF, ll


@njit(cache=True)
def viterbi(obs, wL, pL, pN, pF, q, aN, dN, dF, initN, initF, piL, piN, piF):
    """Max-product decoding. Returns per-base state path (uint8: 0 L, 1 N, 2 F)."""
    T = obs.shape[0]
    RN = dN.shape[0]
    RF = dF.shape[0]
    NEG = -1e300
    lq = np.log(q)
    lexq = np.log(1.0 - q)
    laN = np.log(aN)
    laF = np.log(1.0 - aN)
    ldN = np.log(dN + 1e-300)
    ldF = np.log(dF + 1e-300)
    dL = np.empty(T)
    dNm = np.empty((T, RN))
    dFm = np.empty((T, RF))
    bpL = np.zeros(T, dtype=np.int8)       # 0 from L, 1 from N(r=1), 2 from F(r=1)
    bpN = np.zeros((T, RN), dtype=np.int8)  # 0 continue, 1 entered from L
    bpF = np.zeros((T, RF), dtype=np.int8)

    eL = np.log(_emit(obs[0], _cl(pL * wL[0]))); eN = np.log(_emit(obs[0], pN)); eF = np.log(_emit(obs[0], pF))
    dL[0] = np.log(piL) + eL
    for r in range(RN):
        dNm[0, r] = np.log(piN * initN[r] + 1e-300) + eN
    for r in range(RF):
        dFm[0, r] = np.log(piF * initF[r] + 1e-300) + eF

    for t in range(1, T):
        o = obs[t]
        eL = np.log(_emit(o, _cl(pL * wL[t]))); eN = np.log(_emit(o, pN)); eF = np.log(_emit(o, pF))
        best = dL[t - 1] + lq
        arg = 0
        if dNm[t - 1, 0] > best:
            best = dNm[t - 1, 0]; arg = 1
        if dFm[t - 1, 0] > best:
            best = dFm[t - 1, 0]; arg = 2
        dL[t] = best + eL
        bpL[t] = arg
        ent = dL[t - 1] + lexq
        for r in range(RN):
            cont = dNm[t - 1, r + 1] if r + 1 < RN else NEG
            new = ent + laN + ldN[r]
            if cont >= new:
                dNm[t, r] = cont + eN; bpN[t, r] = 0
            else:
                dNm[t, r] = new + eN; bpN[t, r] = 1
        for r in range(RF):
            cont = dFm[t - 1, r + 1] if r + 1 < RF else NEG
            new = ent + laF + ldF[r]
            if cont >= new:
                dFm[t, r] = cont + eF; bpF[t, r] = 0
            else:
                dFm[t, r] = new + eF; bpF[t, r] = 1

    # termination: any state (right-censored)
    k = 0; rr = 0; best = dL[T - 1]
    for r in range(RN):
        if dNm[T - 1, r] > best:
            best = dNm[T - 1, r]; k = 1; rr = r
    for r in range(RF):
        if dFm[T - 1, r] > best:
            best = dFm[T - 1, r]; k = 2; rr = r
    path = np.empty(T, dtype=np.uint8)
    for t in range(T - 1, -1, -1):
        path[t] = k
        if t == 0:
            break
        if k == 0:
            a = bpL[t]
            if a == 0:
                k = 0
            elif a == 1:
                k = 1; rr = 0
            else:
                k = 2; rr = 0
        elif k == 1:
            if bpN[t, rr] == 0:
                rr = rr + 1
            else:
                k = 0
        else:
            if bpF[t, rr] == 0:
                rr = rr + 1
            else:
                k = 0
    return path
