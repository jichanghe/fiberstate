"""Site-level statistics from single-molecule calls.

occupancy        - fraction of molecules with a TF footprint over a site, using
                   per-molecule posteriors (soft) or hard calls; Wilson / effective-n CI.
co_occupancy     - single-molecule coupling between two sites, measured as a log odds
                   ratio of binding.  By default it is *conditioned on accessibility*:
                   molecules on which either site is wrapped in a nucleosome are
                   down-weighted, so that two sites that merely share an accessible
                   element do not look coupled.
"""
from __future__ import annotations

import math
import numpy as np


def site_state_probs(call, center):
    """(P_linker, P_nuc, P_foot) at a base, from a ReadCall or a hard path."""
    if hasattr(call, "post_foot"):
        return (float(call.post_linker[center]), float(call.post_nuc[center]),
                float(call.post_foot[center]))
    k = int(call[center])
    v = [0.0, 0.0, 0.0]
    v[k] = 1.0
    return tuple(v)


def occupancy(probs):
    """probs: list of (pL, pN, pF) over molecules spanning the site."""
    if len(probs) == 0:
        return dict(n=0, occupancy=float("nan"), accessible=float("nan"),
                    occ_given_accessible=float("nan"), ci_low=float("nan"), ci_high=float("nan"))
    P = np.asarray(probs)
    n = len(P)
    occ = P[:, 2].mean()
    acc = 1 - P[:, 1].mean()
    occ_acc = P[:, 2].sum() / max(1e-9, (1 - P[:, 1]).sum())
    z = 1.96
    den = 1 + z * z / n
    ctr = (occ + z * z / (2 * n)) / den
    hw = z * math.sqrt(occ * (1 - occ) / n + z * z / (4 * n * n)) / den
    return dict(n=n, occupancy=float(occ), accessible=float(acc),
                occ_given_accessible=float(occ_acc), ci_low=float(ctr - hw), ci_high=float(ctr + hw))


def co_occupancy(probs_i, probs_j, condition_on_access=True, pseudo=0.5, min_expected=3.0):
    """probs_i/j: aligned lists of (pL, pN, pF) on molecules spanning both sites.

    Builds a 2x2 table of (soft, accessibility-weighted) molecule counts. Returns the log odds
    ratio (with ``pseudo`` added to each cell), its standard error, z, a two-sided p-value from
    Fisher's exact test on the rounded table, and ``informative``: whether each site is bound,
    and unbound, on at least ``min_expected`` effective molecules. Pairs that are not
    informative have no stable odds ratio (a rarely bound site gives a near-empty table) and
    should not be interpreted.
    """
    from scipy.stats import fisher_exact
    Pi = np.asarray(probs_i, dtype=float)
    Pj = np.asarray(probs_j, dtype=float)
    if len(Pi) == 0:
        return dict(n=0, log_or=float("nan"), se=float("nan"), z=float("nan"), p=float("nan"), informative=False)
    if condition_on_access:
        w = (1 - Pi[:, 1]) * (1 - Pj[:, 1])
        bi = Pi[:, 2] / np.maximum(1e-9, 1 - Pi[:, 1])
        bj = Pj[:, 2] / np.maximum(1e-9, 1 - Pj[:, 1])
    else:
        w = np.ones(len(Pi))
        bi, bj = Pi[:, 2], Pj[:, 2]
    bi = np.clip(bi, 0, 1); bj = np.clip(bj, 0, 1)
    c11 = (w * bi * bj).sum()
    c10 = (w * bi * (1 - bj)).sum()
    c01 = (w * (1 - bi) * bj).sum()
    c00 = (w * (1 - bi) * (1 - bj)).sum()
    n11, n10, n01, n00 = c11 + pseudo, c10 + pseudo, c01 + pseudo, c00 + pseudo
    lor = math.log(n11 * n00 / (n10 * n01))
    se = math.sqrt(1 / n11 + 1 / n10 + 1 / n01 + 1 / n00)
    z = lor / se
    tab = np.rint([[c11, c10], [c01, c00]]).astype(int)
    p = float(fisher_exact(tab)[1]) if tab.sum() > 0 else 1.0
    informative = min(c11 + c10, c01 + c00, c11 + c01, c10 + c00) >= min_expected
    return dict(n=float(w.sum()), log_or=lor, se=se, z=z, p=p, informative=bool(informative))
