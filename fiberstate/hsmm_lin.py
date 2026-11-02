"""Linear-history kernels for the same explicit-duration HSMM as hsmm.py / hsmm_fast.py.

hsmm_fast keeps, for each segment type, a ring buffer indexed by remaining duration and adds
every new segment entry into ~100 ring slots per base (an axpy that wraps around the ring).
Here the same quantities are written as dot products over a plain, contiguous history:

  forward   exit mass at t  = K_t * sum_r d[r] * E[t - r]        (E = scaled entry mass)
  backward  sum at t        = K_t * sum_r d[r] * B[t + 1 + r]    (B = scaled exit betas)
  viterbi   best exit at t  = O_t + max_r (en[t - r] + log d[r]) (en = entry score - offset)

The arithmetic is identical up to summation order (rounding-level differences); Viterbi
keeps hsmm.viterbi's tie rule (an earlier entry wins a tie) and only stores one back-pointer
per base and segment type instead of one per base and duration.
"""
import numpy as np
from numba import njit

_LO, _HI = 1e-150, 1e150


@njit(cache=True, inline="always")
def _cl(p):
    return min(0.99, max(1e-4, p))


@njit(cache=True, inline="always")
def _emit(o, p):
    if o < 0.0:
        return 1.0
    return o * p + (1.0 - o) * (1.0 - p)


@njit(cache=True)
def _support(d):
    lo = 0
    while lo < d.shape[0] and d[lo] <= 0.0:
        lo += 1
    hi = d.shape[0]
    while hi > lo and d[hi - 1] <= 0.0:
        hi -= 1
    return lo, hi


@njit(cache=True, fastmath={"reassoc", "contract"})
def _dot(a, h, i0, n):
    """sum_{k<n} a[k] * h[i0 + k]; both operands ascend so the loop vectorises."""
    z = 0.0
    for k in range(n):
        z += a[k] * h[i0 + k]
    return z


@njit(cache=True, fastmath={"reassoc", "contract"})
def fb(obs, wL, pL, pN, pF, q, aN, dN, dF, initN, initF, piL, piN, piF):
    """Forward-backward. Returns (gL, gN, gF, loglik); same outputs as hsmm_fast.fb."""
    T = obs.shape[0]
    RN = dN.shape[0]
    RF = dF.shape[0]
    aF_ = 1.0 - aN
    loN, hiN = _support(dN)
    loF, hiF = _support(dF)
    # survival S[k] = sum_{r >= k} d[r]  (mass of an entry still present k bases later)
    SN = np.zeros(RN + 1)
    SF = np.zeros(RF + 1)
    for r in range(RN - 1, -1, -1):
        SN[r] = SN[r + 1] + dN[r]
    for r in range(RF - 1, -1, -1):
        SF[r] = SF[r + 1] + dF[r]
    nN = hiN - loN
    nF = hiF - loF
    dNs = dN[loN:hiN].copy()          # support, ascending (backward sums)
    dFs = dF[loF:hiF].copy()
    dNr = dNs[::-1].copy()            # support, descending (forward sums over past entries)
    dFr = dFs[::-1].copy()

    eLa = np.empty(T)
    eNa = np.empty(T)
    eFa = np.empty(T)
    for t in range(T):
        o = obs[t]
        eLa[t] = _emit(o, _cl(pL * wL[t]))
        eNa[t] = _emit(o, pN)
        eFa[t] = _emit(o, pF)

    # ---------------- forward ----------------
    # History arrays are offset by R so index t - r never goes negative: H[R + s] holds time s.
    EN = np.zeros(T + RN)
    EF = np.zeros(T + RF)
    IN = np.zeros(RN)          # in-progress segments at t = 0 (exit at time r)
    IF = np.zeros(RF)
    aL = np.empty(T)
    c = np.empty(T)
    exN = np.empty(T)
    exF = np.empty(T)
    eN = eNa[0]
    eF = eFa[0]
    MN = 0.0
    MF = 0.0
    for r in range(RN):
        IN[r] = piN * initN[r] * eN
        MN += IN[r]
    for r in range(RF):
        IF[r] = piF * initF[r] * eF
        MF += IF[r]
    aLu = piL * eLa[0]
    s = aLu + MN + MF
    c[0] = s
    inv = 1.0 / s
    aL[0] = aLu * inv
    MN *= inv
    MF *= inv
    KN = inv
    KF = inv
    a0N = IN * inv
    a0F = IF * inv
    exN[0] = KN * IN[0]
    exF[0] = KF * IF[0]
    for t in range(1, T):
        eL = eLa[t]; eN = eNa[t]; eF = eFa[t]
        xn = exN[t - 1]
        xf = exF[t - 1]
        lexit = aL[t - 1] * (1.0 - q)
        enN = lexit * aN
        enF = lexit * aF_
        aLu = (aL[t - 1] * q + xn + xf) * eL
        MNu = eN * (MN - xn + enN)
        MFu = eF * (MF - xf + enF)
        s = aLu + MNu + MFu
        c[t] = s
        inv = 1.0 / s
        EN[RN + t] = enN / KN
        EF[RF + t] = enF / KF
        KN = KN * eN * inv
        KF = KF * eF * inv
        aL[t] = aLu * inv
        MN = MNu * inv
        MF = MFu * inv
        if KN < _LO or KN > _HI:
            for i in range(max(0, t - RN + 1), t + 1):
                EN[RN + i] *= KN
            for i in range(RN):
                IN[i] *= KN
            KN = 1.0
        if KF < _LO or KF > _HI:
            for i in range(max(0, t - RF + 1), t + 1):
                EF[RF + i] *= KF
            for i in range(RF):
                IF[i] *= KF
            KF = 1.0
        zN = _dot(dNr, EN, RN + t - hiN + 1, nN)     # = sum_r dN[r] * EN[RN + t - r]
        zF = _dot(dFr, EF, RF + t - hiF + 1, nF)
        if t < RN:
            zN += IN[t]
        if t < RF:
            zF += IF[t]
        exN[t] = KN * zN
        exF[t] = KF * zF
        if (t & 31) == 0:
            # re-anchor running totals: entries still present at t (exit time >= t)
            z = 0.0
            for k in range(0, min(RN, t + 1)):
                z += SN[k] * EN[RN + t - k]
            for r in range(t, RN):
                z += IN[r]
            MN = KN * z
            z = 0.0
            for k in range(0, min(RF, t + 1)):
                z += SF[k] * EF[RF + t - k]
            for r in range(t, RF):
                z += IF[r]
            MF = KF * z

    # ---------------- backward ----------------
    # BN[s] = scaled beta of a segment ending at s (index s, s >= T-1 is "past the end" = 1).
    BN = np.ones(T + RN + 1)
    BF = np.ones(T + RF + 1)
    KBN = 1.0
    KBF = 1.0
    bL_next = 1.0
    gL = np.empty(T)
    stN = np.zeros(T)
    stF = np.zeros(T)
    edN = np.zeros(T)
    edF = np.zeros(T)
    gL[T - 1] = aL[T - 1]
    edN[T - 1] = exN[T - 1]
    edF[T - 1] = exF[T - 1]
    for t in range(T - 2, -1, -1):
        ic = 1.0 / c[t + 1]
        eL1 = eLa[t + 1] * ic
        eN1 = eNa[t + 1] * ic
        eF1 = eFa[t + 1] * ic
        sN = KBN * _dot(dNs, BN, t + 1 + loN, nN)    # = sum_r dN[r] * BN[t + 1 + r]
        sF = KBF * _dot(dFs, BF, t + 1 + loF, nF)
        lexit = aL[t] * (1.0 - q)
        stN[t + 1] = lexit * aN * eN1 * sN
        stF[t + 1] = lexit * aF_ * eF1 * sF
        bL = q * eL1 * bL_next + (1.0 - q) * (aN * sN * eN1 + aF_ * sF * eF1)
        bexit = eL1 * bL_next
        KBN *= eN1
        BN[t] = bexit / KBN
        KBF *= eF1
        BF[t] = bexit / KBF
        if KBN < _LO or KBN > _HI:
            for i in range(t, min(T + RN + 1, t + RN + 1)):
                BN[i] *= KBN
            KBN = 1.0
        if KBF < _LO or KBF > _HI:
            for i in range(t, min(T + RF + 1, t + RF + 1)):
                BF[i] *= KBF
            KBF = 1.0
        edN[t] = exN[t] * bexit
        edF[t] = exF[t] * bexit
        gL[t] = aL[t] * bL
        bL_next = bL
    # segments in progress at t = 0 (remaining r -> ends at time r)
    z = 0.0
    for r in range(RN):
        z += a0N[r] * BN[r]
    stN[0] = z * KBN
    z = 0.0
    for r in range(RF):
        z += a0F[r] * BF[r]
    stF[0] = z * KBF

    gN = np.empty(T)
    gF = np.empty(T)
    cN = 0.0
    cF = 0.0
    ll = 0.0
    for t in range(T):
        cN += stN[t]
        cF += stF[t]
        if t > 0:
            cN -= edN[t - 1]
            cF -= edF[t - 1]
        gN[t] = cN if cN > 0.0 else 0.0
        gF[t] = cF if cF > 0.0 else 0.0
        ll += np.log(c[t])
    return gL, gN, gF, ll


@njit(cache=True, inline="always")
def _upd(v, k, m, i):
    """lane update keeping the earliest index on ties (strictly greater replaces)."""
    if v > m:
        return v, k
    return m, i


@njit(cache=True)
def _vargmax(a, h, i0, n):
    """(max_k a[k] + h[i0 + k], smallest k attaining it). Eight independent lanes, then merged:
    exact, and the same answer as a left-to-right scan with strict '>'."""
    m0 = m1 = m2 = m3 = m4 = m5 = m6 = m7 = -1e308
    i_0 = i_1 = i_2 = i_3 = i_4 = i_5 = i_6 = i_7 = -1
    k = 0
    while k + 8 <= n:
        j = i0 + k
        m0, i_0 = _upd(a[k] + h[j], k, m0, i_0)
        m1, i_1 = _upd(a[k + 1] + h[j + 1], k + 1, m1, i_1)
        m2, i_2 = _upd(a[k + 2] + h[j + 2], k + 2, m2, i_2)
        m3, i_3 = _upd(a[k + 3] + h[j + 3], k + 3, m3, i_3)
        m4, i_4 = _upd(a[k + 4] + h[j + 4], k + 4, m4, i_4)
        m5, i_5 = _upd(a[k + 5] + h[j + 5], k + 5, m5, i_5)
        m6, i_6 = _upd(a[k + 6] + h[j + 6], k + 6, m6, i_6)
        m7, i_7 = _upd(a[k + 7] + h[j + 7], k + 7, m7, i_7)
        k += 8
    while k < n:
        m0, i_0 = _upd(a[k] + h[i0 + k], k, m0, i_0)
        k += 1
    bm = m0; bi = i_0
    ms = (m1, m2, m3, m4, m5, m6, m7)
    iz = (i_1, i_2, i_3, i_4, i_5, i_6, i_7)
    for j in range(7):
        m = ms[j]; i = iz[j]
        if i >= 0 and (m > bm or (m == bm and (bi < 0 or i < bi))):
            bm = m; bi = i
    return bm, bi


@njit(cache=True)
def viterbi(obs, wL, pL, pN, pF, q, aN, dN, dF, initN, initF, piL, piN, piF):
    """Max-product decoding with hsmm.viterbi's tie rules; one back-pointer per base and type."""
    T = obs.shape[0]
    RN = dN.shape[0]
    RF = dF.shape[0]
    NEG = -1e300
    loN, hiN = _support(dN)
    loF, hiF = _support(dF)
    lq = np.log(q)
    lexq = np.log(1.0 - q)
    laN = np.log(aN)
    laF = np.log(1.0 - aN)
    ldN = np.log(dN + 1e-300)
    ldF = np.log(dF + 1e-300)
    nN = hiN - loN
    nF = hiF - loF
    ldNr = ldN[loN:hiN][::-1].copy()      # k = 0 <-> r = hiN - 1 (earliest entry)
    ldFr = ldF[loF:hiF][::-1].copy()
    # entry scores minus the running emission offset, history offset by R
    enN = np.full(T + RN, NEG)
    enF = np.full(T + RF, NEG)
    IN = np.empty(RN)           # in-progress at t = 0: exits at time r (offset-free)
    IF = np.empty(RF)
    bpL = np.zeros(T, dtype=np.int8)
    rN = np.full(T, -1, dtype=np.int32)   # duration index of the N segment that exits at t (-1: initial)
    rF = np.full(T, -1, dtype=np.int32)
    o = obs[0]
    eL = np.log(_emit(o, _cl(pL * wL[0]))); eN = np.log(_emit(o, pN)); eF = np.log(_emit(o, pF))
    dL = np.log(piL) + eL
    for r in range(RN):
        IN[r] = np.log(piN * initN[r] + 1e-300) + eN
    for r in range(RF):
        IF[r] = np.log(piF * initF[r] + 1e-300) + eF
    ON = 0.0                     # emission sums since t = 0, excluding t = 0 (already in IN / IF)
    OF = 0.0
    exNv = IN[0]                 # best exit score at t = 0
    exFv = IF[0]
    # log emissions for every base up front (one vectorisable pass instead of 3 logs per base)
    lEL = np.empty(T)
    lEN = np.empty(T)
    lEF = np.empty(T)
    for t in range(T):
        o = obs[t]
        lEL[t] = _emit(o, _cl(pL * wL[t])); lEN[t] = _emit(o, pN); lEF[t] = _emit(o, pF)
    lEL = np.log(lEL)
    lEN = np.log(lEN)
    lEF = np.log(lEF)
    for t in range(1, T):
        eL = lEL[t]; eN = lEN[t]; eF = lEF[t]
        best = dL + lq
        arg = 0
        if exNv > best:
            best = exNv; arg = 1
        if exFv > best:
            best = exFv; arg = 2
        ent = dL + lexq
        dL = best + eL
        bpL[t] = arg
        enN[RN + t] = ent + laN - ON
        enF[RF + t] = ent + laF - OF
        ON += eN
        OF += eF
        # best N exit at t: initial segment (earliest) first, then entries from earliest to latest;
        # a later entry only replaces on a strictly better score (earlier entry wins ties)
        # earliest candidate wins ties: the initial segment, then entries from earliest to latest
        i0 = RN + t - hiN + 1
        m, kk = _vargmax(ldNr, enN, i0, nN)
        if t < RN and IN[t] >= m:
            bv = IN[t]
            br = -1
        elif m > NEG:
            bv = m
            br = hiN - 1 - kk
        else:
            bv = NEG
            br = -2
        exNv = bv + ON
        rN[t] = br
        # earliest candidate wins ties: the initial segment, then entries from earliest to latest
        i0 = RF + t - hiF + 1
        m, kk = _vargmax(ldFr, enF, i0, nF)
        if t < RF and IF[t] >= m:
            bv = IF[t]
            br = -1
        elif m > NEG:
            bv = m
            br = hiF - 1 - kk
        else:
            bv = NEG
            br = -2
        exFv = bv + OF
        rF[t] = br
    # final state: linker, or a segment still in progress at T-1 (truncated at the read end)
    best = dL
    k = 0
    start = 0
    # in-progress N with remaining rr at T-1: entered at s with duration index r = rr + (T-1-s)
    for rr in range(RN):
        v = NEG
        sbest = -2
        if rr + T - 1 < RN:
            v = IN[rr + T - 1]
            sbest = -1
        for s in range(max(1, T - RN + rr), T):
            r = rr + (T - 1 - s)
            if r < loN or r >= hiN:
                continue
            x = enN[RN + s] + ldN[r]
            if x > v:
                v = x; sbest = s
        if v + ON > best:
            best = v + ON; k = 1; start = sbest
    for rr in range(RF):
        v = NEG
        sbest = -2
        if rr + T - 1 < RF:
            v = IF[rr + T - 1]
            sbest = -1
        for s in range(max(1, T - RF + rr), T):
            r = rr + (T - 1 - s)
            if r < loF or r >= hiF:
                continue
            x = enF[RF + s] + ldF[r]
            if x > v:
                v = x; sbest = s
        if v + OF > best:
            best = v + OF; k = 2; start = sbest
    path = np.empty(T, dtype=np.uint8)
    t = T - 1
    while t >= 0:
        if k == 0:
            path[t] = 0
            if t == 0:
                break
            a = bpL[t]
            if a == 0:
                t -= 1
            else:
                # the segment that exited at t-1
                k = int(a)
                rexit = rN[t - 1] if k == 1 else rF[t - 1]
                start = -1 if rexit == -1 else (t - 1) - rexit
                t -= 1
        else:
            s0 = 0 if start == -1 else start
            for u in range(s0, t + 1):
                path[u] = k
            if s0 == 0:
                break
            t = s0 - 1
            k = 0
    return path
