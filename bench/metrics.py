"""Benchmark metrics against simulator ground truth."""
from __future__ import annotations

import numpy as np

LINKER, NUC, FOOT = 0, 1, 2


def match_segments(truth, pred, min_recip=0.5):
    """Greedy one-to-one matching by reciprocal overlap. Returns list of (ti, pi)."""
    truth = sorted(truth, key=lambda x: x[0])
    pred = sorted(pred, key=lambda x: x[0])
    pairs = []
    cand = []
    j0 = 0
    for ti, (ts, te) in enumerate(truth):
        while j0 < len(pred) and pred[j0][1] <= ts:
            j0 += 1
        j = j0
        while j < len(pred) and pred[j][0] < te:
            ps, pe = pred[j]
            ov = min(te, pe) - max(ts, ps)
            if ov > 0:
                r = min(ov / (te - ts), ov / (pe - ps))
                if r >= min_recip:
                    cand.append((r, ti, j))
            j += 1
    cand.sort(reverse=True)
    used_t, used_p = set(), set()
    for r, ti, pj in cand:
        if ti in used_t or pj in used_p:
            continue
        used_t.add(ti); used_p.add(pj)
        pairs.append((truth[ti], pred[pj]))
    return pairs


def seg_prf(true_segs, pred_segs, kind):
    t = [(s, e) for k, s, e, *_ in true_segs if k == kind]
    p = [(s, e) for k, s, e, *_ in pred_segs if k == kind]
    m = match_segments(t, p)
    tp = len(m)
    dyad_err = [abs((a[0] + a[1]) / 2 - (b[0] + b[1]) / 2) for a, b in m]
    return tp, len(t), len(p), dyad_err


def false_calls_in_mask(true_segs, pred_segs, kind, mask):
    """Predicted segments of `kind` lying >=50% in mask that match no true segment."""
    t = [(s, e) for k, s, e, *_ in true_segs if k == kind]
    p = [(s, e) for k, s, e, *_ in pred_segs if k == kind]
    matched = {pp for _, pp in match_segments(t, p)}
    n = 0
    for s, e in p:
        if (s, e) in matched:
            continue
        if mask[s:e].mean() >= 0.5:
            n += 1
    return n


def prf(tp, nt, np_):
    prec = tp / np_ if np_ else 0.0
    rec = tp / nt if nt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def auroc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, bool)
    ok = np.isfinite(scores)
    scores, labels = scores[ok], labels[ok]
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata
    ranks = rankdata(np.concatenate([pos, neg]))   # average ranks for ties
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
