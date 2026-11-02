"""Fast kernels for the same explicit-duration HSMM as hsmm.py (identical results).

Three exact tricks make each pass cheap:

1. Ring buffer. The countdown state (type, remaining r) shifts by one every base. Stored
   at slot (r + t) mod R, the shift becomes "no move", so nothing is copied.
2. Lazy scaling. Every state of one type is multiplied by the same emission each base,
   so the buffer stores values divided by a running scalar instead of touching all R
   entries. Only states that can actually be *entered* (durations with non-zero
   probability, e.g. 90-200 bp for nucleosomes) are written each base.
3. Posteriors from segment starts and ends. The probability that a base lies in a
   nucleosome is (starts at or before t) - (ends before t). Both are O(1) per base given
   quantities the backward pass already computes, so the full posterior costs no more
   than the linker-only pass.
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


@njit(cache=True, fastmath={"reassoc", "contract"})   # lets the sums vectorise; rounding-level change
def fb(obs, wL, pL, pN, pF, q, aN, dN, dF, initN, initF, piL, piN, piF):
    """Forward-backward. Returns (gL, gN, gF, loglik)."""
    T = obs.shape[0]
    RN = dN.shape[0]
    RF = dF.shape[0]
    aF_ = 1.0 - aN
    loN, hiN = _support(dN)
    loF, hiF = _support(dF)

    # ---------------- forward ----------------
    aL = np.empty(T)
    c = np.empty(T)
    exN = np.empty(T)          # normalised alpha_t[r=0] (segment ends at t)
    exF = np.empty(T)
    A_N = np.empty(RN)
    A_F = np.empty(RF)
    o = obs[0]
    eL = _emit(o, _cl(pL * wL[0])); eN = _emit(o, pN); eF = _emit(o, pF)
    MN = 0.0
    MF = 0.0
    for r in range(RN):
        A_N[r] = piN * initN[r] * eN
        MN += A_N[r]
    for r in range(RF):
        A_F[r] = piF * initF[r] * eF
        MF += A_F[r]
    aLu = piL * eL
    s = aLu + MN + MF
    c[0] = s
    inv = 1.0 / s
    aL[0] = aLu * inv
    MN *= inv
    MF *= inv
    KN = inv
    KF = inv
    a0N = A_N * inv            # alpha_0 rows, needed for the t = 0 start mass
    a0F = A_F * inv
    exN[0] = KN * A_N[0]
    exF[0] = KF * A_F[0]
    for t in range(1, T):
        o = obs[t]
        eL = _emit(o, _cl(pL * wL[t])); eN = _emit(o, pN); eF = _emit(o, pF)
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
        # N ring: exit slot (t-1)%RN becomes r = RN-1 (no continuation)
        offN = t % RN
        A_N[(t - 1) % RN] = 0.0
        add = enN / KN
        cut = RN - offN                       # r < cut -> slot offN + r, else slot r - cut
        for r in range(loN, min(hiN, cut)):
            A_N[offN + r] += add * dN[r]
        for r in range(max(loN, cut), hiN):
            A_N[r - cut] += add * dN[r]
        KN = KN * eN * inv
        offF = t % RF
        A_F[(t - 1) % RF] = 0.0
        add = enF / KF
        cut = RF - offF
        for r in range(loF, min(hiF, cut)):
            A_F[offF + r] += add * dF[r]
        for r in range(max(loF, cut), hiF):
            A_F[r - cut] += add * dF[r]
        KF = KF * eF * inv
        aL[t] = aLu * inv
        MN = MNu * inv
        MF = MFu * inv
        if KN < _LO or KN > _HI:
            for i in range(RN):
                A_N[i] *= KN
            KN = 1.0
        if KF < _LO or KF > _HI:
            for i in range(RF):
                A_F[i] *= KF
            KF = 1.0
        exN[t] = KN * A_N[offN]
        exF[t] = KF * A_F[offF]
        if (t & 31) == 0:
            # re-anchor the running totals to the buffers so rounding cannot accumulate
            z = 0.0
            for i in range(RN):
                z += A_N[i]
            MN = KN * z
            z = 0.0
            for i in range(RF):
                z += A_F[i]
            MF = KF * z

    # ---------------- backward ----------------
    B_N = np.ones(RN)
    B_F = np.ones(RF)
    KBN = 1.0
    KBF = 1.0
    bL_next = 1.0
    gL = np.empty(T)
    stN = np.zeros(T)          # P(N segment starts at t)
    stF = np.zeros(T)
    edN = np.zeros(T)          # P(N segment's last base is t)
    edF = np.zeros(T)
    gL[T - 1] = aL[T - 1]
    edN[T - 1] = exN[T - 1]
    edF[T - 1] = exF[T - 1]
    for t in range(T - 2, -1, -1):
        o = obs[t + 1]
        ic = 1.0 / c[t + 1]
        eL1 = _emit(o, _cl(pL * wL[t + 1])) * ic
        eN1 = _emit(o, pN) * ic
        eF1 = _emit(o, pF) * ic
        o1 = (t + 1) % RN
        cut = RN - o1
        sN = 0.0
        for r in range(loN, min(hiN, cut)):
            sN += dN[r] * B_N[o1 + r]
        for r in range(max(loN, cut), hiN):
            sN += dN[r] * B_N[r - cut]
        sN *= KBN
        o1 = (t + 1) % RF
        cut = RF - o1
        sF = 0.0
        for r in range(loF, min(hiF, cut)):
            sF += dF[r] * B_F[o1 + r]
        for r in range(max(loF, cut), hiF):
            sF += dF[r] * B_F[r - cut]
        sF *= KBF
        lexit = aL[t] * (1.0 - q)
        stN[t + 1] = lexit * aN * eN1 * sN
        stF[t + 1] = lexit * aF_ * eF1 * sF
        bL = q * eL1 * bL_next + (1.0 - q) * (aN * sN * eN1 + aF_ * sF * eF1)
        bexit = eL1 * bL_next               # beta_t[r = 0] for either protected type
        KBN *= eN1
        B_N[t % RN] = bexit / KBN
        KBF *= eF1
        B_F[t % RF] = bexit / KBF
        if KBN < _LO or KBN > _HI:
            for i in range(RN):
                B_N[i] *= KBN
            KBN = 1.0
        if KBF < _LO or KBF > _HI:
            for i in range(RF):
                B_F[i] *= KBF
            KBF = 1.0
        edN[t] = exN[t] * bexit
        edF[t] = exF[t] * bexit
        gL[t] = aL[t] * bL
        bL_next = bL
    # mass of segments already in progress at t = 0 (slot(0, r) = r)
    z = 0.0
    for r in range(RN):
        z += a0N[r] * B_N[r]
    stN[0] = z * KBN
    z = 0.0
    for r in range(RF):
        z += a0F[r] * B_F[r]
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


@njit(cache=True)
def viterbi(obs, wL, pL, pN, pF, q, aN, dN, dF, initN, initF, piL, piN, piF):
    """Max-product decoding, same tie-breaking as hsmm.viterbi."""
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
    D_N = np.empty(RN)         # stored minus running offset ON
    D_F = np.empty(RF)
    bpL = np.zeros(T, dtype=np.int8)
    bpN = np.zeros((T, RN), dtype=np.int8)
    bpF = np.zeros((T, RF), dtype=np.int8)
    o = obs[0]
    eL = np.log(_emit(o, _cl(pL * wL[0]))); eN = np.log(_emit(o, pN)); eF = np.log(_emit(o, pF))
    dL = np.log(piL) + eL
    for r in range(RN):
        D_N[r] = np.log(piN * initN[r] + 1e-300) + eN
    for r in range(RF):
        D_F[r] = np.log(piF * initF[r] + 1e-300) + eF
    ON = 0.0
    OF = 0.0
    for t in range(1, T):
        o = obs[t]
        eL = np.log(_emit(o, _cl(pL * wL[t]))); eN = np.log(_emit(o, pN)); eF = np.log(_emit(o, pF))
        exN = D_N[(t - 1) % RN] + ON
        exF = D_F[(t - 1) % RF] + OF
        best = dL + lq
        arg = 0
        if exN > best:
            best = exN; arg = 1
        if exF > best:
            best = exF; arg = 2
        ent = dL + lexq
        dL = best + eL
        bpL[t] = arg
        # exit slot carries no continuation; then compare entries on the support only.
        D_N[(t - 1) % RN] = NEG
        offN = t % RN
        en = ent + laN - ON
        for r in range(loN, hiN):
            sl = offN + r
            if sl >= RN:
                sl -= RN
            new = en + ldN[r]
            if D_N[sl] < new:            # cont >= new keeps continuation (old tie rule)
                D_N[sl] = new
                bpN[t, r] = 1
        ON += eN
        D_F[(t - 1) % RF] = NEG
        offF = t % RF
        en = ent + laF - OF
        for r in range(loF, hiF):
            sl = offF + r
            if sl >= RF:
                sl -= RF
            new = en + ldF[r]
            if D_F[sl] < new:
                D_F[sl] = new
                bpF[t, r] = 1
        OF += eF
    k = 0; rr = 0; best = dL
    for r in range(RN):
        v = D_N[(r + T - 1) % RN] + ON
        if v > best:
            best = v; k = 1; rr = r
    for r in range(RF):
        v = D_F[(r + T - 1) % RF] + OF
        if v > best:
            best = v; k = 2; rr = r
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
                rr += 1
            else:
                k = 0
        else:
            if bpF[t, rr] == 0:
                rr += 1
            else:
                k = 0
    return path
