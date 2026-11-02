"""Sequence-context model of Hia5 m6A marking in accessible DNA.

Hia5 does not mark every accessible A equally: the local sequence around the A changes
how often it is marked (it strongly prefers A/T-rich context). A short run of A's in a
poor context can stay unmarked even when the DNA is open, which a context-blind model
reads as a small protected footprint.

We describe each methylatable base by the k-mer centred on the A that would carry the
mark (for a T on the top strand, the reverse complement, so the A is always in the
centre). A table of relative marking rates per k-mer is learned from confidently
accessible DNA on real fibers and multiplies the per-fiber accessible rate.
"""
from __future__ import annotations

import numpy as np

_CODE = np.full(256, -1, dtype=np.int64)
for _i, _b in enumerate(b"ACGT"):
    _CODE[_b] = _i
    _CODE[ord(chr(_b).lower())] = _i


def context_codes(seq: np.ndarray, k: int = 5) -> np.ndarray:
    """Integer k-mer code (centred on the marked A) for every base; -1 if not A/T or near an end."""
    assert k % 2 == 1
    h = k // 2
    L = len(seq)
    b = _CODE[seq]
    out = np.full(L, -1, dtype=np.int64)
    if L < k:
        return out
    # forward code over the window, base-4, leftmost base most significant
    fwd = np.zeros(L - k + 1, dtype=np.int64)
    rev = np.zeros(L - k + 1, dtype=np.int64)
    bad = np.zeros(L - k + 1, dtype=bool)
    for j in range(k):
        x = b[j:L - k + 1 + j]
        bad |= x < 0
        xx = np.where(x < 0, 0, x)
        fwd = fwd * 4 + xx
    for j in range(k - 1, -1, -1):
        x = b[j:L - k + 1 + j]
        xx = np.where(x < 0, 0, x)
        rev = rev * 4 + (3 - xx)      # complement, read right-to-left
    centre = b[h:L - h]
    code = np.where(centre == 0, fwd, np.where(centre == 3, rev, -1))
    code[bad] = -1
    out[h:L - h] = code
    return out


def kmer_string(code: int, k: int) -> str:
    s = []
    for _ in range(k):
        s.append("ACGT"[code % 4])
        code //= 4
    return "".join(reversed(s))


class ContextTable:
    """Relative marking weight per k-mer (mean weight 1 over accessible A's)."""

    def __init__(self, k: int, weights: np.ndarray):
        self.k = k
        self.weights = np.asarray(weights, dtype=np.float64)

    def per_base(self, seq: np.ndarray) -> np.ndarray:
        c = context_codes(seq, self.k)
        w = np.ones(len(seq))
        ok = c >= 0
        w[ok] = self.weights[c[ok]]
        return w

    @classmethod
    def fit(cls, counts_m: np.ndarray, counts_n: np.ndarray, k: int, shrink: float = 20.0):
        """Shrunken rate ratio per k-mer from marked / total counts in accessible DNA."""
        tot = counts_n.sum()
        base = counts_m.sum() / max(1, tot)
        rate = (counts_m + shrink * base) / (counts_n + shrink)
        w = rate / base
        # renormalise so the count-weighted mean weight is 1
        w = w / (np.dot(w, counts_n) / max(1, tot))
        return cls(k, w)

    def save(self, path):
        with open(path, "w") as f:
            f.write(f"#k={self.k}\nkmer\tweight\n")
            for c, w in enumerate(self.weights):
                f.write(f"{kmer_string(c, self.k)}\t{w:.5f}\n")

    @classmethod
    def load(cls, path):
        rows = []
        k = None
        with open(path) as f:
            for line in f:
                if line.startswith("#k="):
                    k = int(line[3:])
                    continue
                if line.startswith("kmer") or not line.strip():
                    continue
                km, w = line.split("\t")
                rows.append((km, float(w)))
        k = k or len(rows[0][0])
        w = np.ones(4 ** k)
        for km, x in rows:
            c = 0
            for ch in km:
                c = c * 4 + "ACGT".index(ch)
            w[c] = x
        return cls(k, w)
