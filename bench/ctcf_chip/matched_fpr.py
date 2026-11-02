"""Compare footprint callers at a matched false-positive rate per molecule.

Site AUROC only ranks sites by footprinted share; it does not reward a low false-positive
rate as such. Here every tool gets a per-molecule score at each motif, and the threshold is
moved until the share of molecules called footprinted at *unbound* sites equals a target
rate. The share footprinted at *bound* sites at that threshold is what the tool recovers.

    python matched_fpr.py SITES.bed RAW.bam OUT.json label=kind:path ...
kinds
    fsposterior:BAM   fiberstate run with --posterior-tags: mean footprint posterior over the motif
    fiberhmm:BAM      FiberHMM: best quality score (AQ, first value) of a footprint covering >= 80% of the motif
    hfscore:TSV       HiddenFoot adapter table: motif_posterior column
    ftfootprint:TSV   fibertools `ft footprint` (yes/no: a single operating point)
Molecules are those each tool scores in score_sites.py (spanning the motif, centre aligned).
"""
import json
import os
import sys

import numpy as np
import pysam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from score_sites import MIN_FIBERS, auroc, ft_footprint_states, load_sites  # noqa: E402
from bench.ftparse import ma_frame  # noqa: E402
from fiberstate.io import query_to_ref  # noqa: E402

TARGETS = (0.005, 0.01, 0.02, 0.03, 0.05)


def _motif_query(r, st):
    """Query positions of the motif's aligned bases, or None if the molecule does not qualify
    (same rule as score_sites.tool_states: enough motif bases aligned and the centre aligned)."""
    q2r = query_to_ref(r)
    qs = np.flatnonzero((q2r >= st["s"]) & (q2r < st["e"]))
    c = (st["s"] + st["e"]) // 2
    if len(qs) < min(12, int(0.8 * (st["e"] - st["s"]))) or not np.any(q2r[qs] == c):
        return None
    return qs


def _spanning(bam, st):
    for r in bam.fetch(st["chrom"], st["s"], st["e"]):
        if r.is_secondary or r.is_supplementary or r.reference_start > st["s"] or r.reference_end < st["e"]:
            continue
        yield r


def fs_scores(bam_p, sites):
    bam = pysam.AlignmentFile(bam_p)
    out = {}
    for i, st in enumerate(sites):
        for r in _spanning(bam, st):
            if not r.has_tag("fp"):
                raise SystemExit(f"{bam_p}: no fp tag; rerun fiberstate call with --posterior-tags")
            qs = _motif_query(r, st)
            if qs is None:
                continue
            fp = np.asarray(r.get_tag("fp"), dtype=np.float64)
            out[(i, r.query_name)] = float(fp[qs].mean()) / 255.0
    return out


def fh_scores(bam_p, sites):
    bam = pysam.AlignmentFile(bam_p)
    molecular = ma_frame(bam.header) == "molecular"
    out = {}
    for i, st in enumerate(sites):
        for r in _spanning(bam, st):
            qs = _motif_query(r, st)
            if qs is None:
                continue
            qa, qb = int(qs.min()), int(qs.max()) + 1
            L = r.query_length
            best = 0.0
            if r.has_tag("MA"):
                aq = list(r.get_tag("AQ")) if r.has_tag("AQ") else []
                k = 0
                for part in str(r.get_tag("MA")).split(";")[1:]:
                    head, data = part.split(":", 1)
                    j = next(n for n, ch in enumerate(head) if ch in "+-.")
                    name, qspec = head[:j], head[j + 1:]
                    for tok in filter(None, data.split(",")):
                        s, l = tok.split("-", 1)
                        s0 = int(s) - 1; e0 = s0 + int(l)
                        q = aq[k] if (qspec and k < len(aq)) else 255
                        k += len(qspec)
                        if name != "tf":
                            continue
                        if molecular and r.is_reverse:
                            s0, e0 = L - e0, L - s0
                        if min(e0, qb) - max(s0, qa) >= 0.8 * (qb - qa):
                            best = max(best, q / 255.0)
            out[(i, r.query_name)] = best
    return out


def hf_scores(tsv_p, sites):
    idx = {st["name"]: i for i, st in enumerate(sites)}
    out = {}
    with open(tsv_p) as f:
        h = f.readline().rstrip("\n").split("\t")
        col = {k: j for j, k in enumerate(h)}
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[0] in idx:
                out[(idx[c[0]], c[1])] = float(c[col["motif_posterior"]])
    return out


def ft_scores(tsv_p, sites):
    return {k: 1.0 if v == "foot" else 0.0 for k, v in ft_footprint_states(tsv_p, sites).items()}


def operating_points(scores, sites):
    """Molecule-level (unbound rate, bound rate) for every threshold, plus per-site arrays."""
    per = {}
    for (i, _), v in scores.items():
        per.setdefault(i, []).append(v)
    keep = [i for i in per if len(per[i]) >= MIN_FIBERS]
    bound = np.array([v for i in keep if sites[i]["bound"] for v in per[i]])
    unb = np.array([v for i in keep if not sites[i]["bound"] for v in per[i]])
    return per, keep, bound, unb


def at_rate(per, keep, bound, unb, sites, target):
    """Lowest threshold whose unbound rate <= target (calls are score > threshold)."""
    n = len(unb)
    if n == 0:
        return None
    allowed = int(np.floor(target * n + 1e-9))
    su = np.sort(unb)
    thr = -1.0 if allowed >= n else float(su[n - allowed - 1])
    y = np.array([sites[i]["bound"] for i in keep], dtype=bool)
    occ = np.array([np.mean(np.array(per[i]) > thr) for i in keep])
    return dict(threshold=thr, unbound_rate=float(np.mean(unb > thr)), bound_rate=float(np.mean(bound > thr)),
                site_auroc=float(auroc(occ, y)))


def roc_points(bound, unb):
    """(unbound rate, bound rate) at every distinct threshold, from (0,0) to (1,1)."""
    v = np.concatenate([bound, unb])
    lab = np.concatenate([np.ones(len(bound)), np.zeros(len(unb))])
    o = np.argsort(-v, kind="stable")
    v, lab = v[o], lab[o]
    last = np.r_[np.flatnonzero(np.diff(v) != 0), len(v) - 1]
    tp = np.cumsum(lab)[last] / max(1, len(bound))
    fp = np.cumsum(1 - lab)[last] / max(1, len(unb))
    return np.r_[0.0, fp, 1.0], np.r_[0.0, tp, 1.0]


def interp_rate(bound, unb, target):
    """Bound rate at unbound rate = target, interpolated on the ROC curve (a tool with tied
    scores, or yes/no calls, reaches rates between its operating points by random tie-breaking)."""
    xs, ys = roc_points(bound, unb)
    return float(np.interp(target, xs, ys))


def partial_auc(bound, unb, max_fpr=0.05):
    """Area under (unbound rate, bound rate) up to max_fpr, divided by max_fpr (1 = perfect)."""
    v = np.concatenate([bound, unb])
    lab = np.concatenate([np.ones(len(bound)), np.zeros(len(unb))])
    o = np.argsort(-v, kind="stable")
    v, lab = v[o], lab[o]
    last = np.r_[np.flatnonzero(np.diff(v) != 0), len(v) - 1]     # end of each tie group
    tp = np.cumsum(lab)[last] / max(1, len(bound))
    fp = np.cumsum(1 - lab)[last] / max(1, len(unb))
    xs = np.r_[0.0, fp]
    ys = np.r_[0.0, tp]
    area = 0.0
    for k in range(1, len(xs)):
        x0, y0, x1, y1 = xs[k - 1], ys[k - 1], xs[k], ys[k]
        if x0 >= max_fpr:
            break
        if x1 > max_fpr:
            y1 = y0 + (y1 - y0) * (max_fpr - x0) / (x1 - x0); x1 = max_fpr
        area += (x1 - x0) * (y0 + y1) / 2
    return area / max_fpr


OWN_THRESHOLD = {"fsposterior": 0.5, "fiberhmm": 0.0, "hfscore": 0.5, "ftfootprint": 0.5}


def main(sites_p, raw_bam, out_p, *tools):
    sites = load_sites(sites_p)
    kinds = {"fsposterior": fs_scores, "fiberhmm": fh_scores, "hfscore": hf_scores, "ftfootprint": ft_scores}
    res = {}
    rng = np.random.default_rng(0)
    data = {}
    for spec in tools:
        label, rest = spec.split("=", 1)
        kind, path = rest.split(":", 1)
        sc = kinds[kind](path, sites)
        per, keep, bound, unb = operating_points(sc, sites)
        data[label] = (per, keep, bound, unb)
        binary = kind == "ftfootprint"
        r = dict(kind=kind, n_molecules_bound=int(len(bound)), n_molecules_unbound=int(len(unb)),
                 own_point=dict(threshold=OWN_THRESHOLD[kind], unbound_rate=float(np.mean(unb > OWN_THRESHOLD[kind])),
                                bound_rate=float(np.mean(bound > OWN_THRESHOLD[kind]))),
                 at_target={})
        r["bound_rate_interpolated"] = {}
        for f in TARGETS:
            r["at_target"][str(f)] = None if binary else at_rate(per, keep, bound, unb, sites, f)
            r["bound_rate_interpolated"][str(f)] = interp_rate(bound, unb, f)
        r["partial_auc_fpr_le_0.05"] = float(partial_auc(bound, unb, 0.05))
        if kind == "fsposterior":
            # fixed cut-offs on the motif posterior, for choosing one on a tuning set and testing elsewhere
            y = np.array([sites[i]["bound"] for i in keep], dtype=bool)
            r["threshold_sweep"] = {}
            for thr in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7):
                occ = np.array([np.mean(np.array(per[i]) > thr) for i in keep])
                r["threshold_sweep"][str(thr)] = dict(site_auroc=float(auroc(occ, y)), unbound_rate=float(np.mean(unb > thr)),
                                                       bound_rate=float(np.mean(bound > thr)))
        res[label] = r
    # every tool's own operating point: what does each continuous tool recover at that same false-positive rate?
    for label, r in res.items():
        f = r["own_point"]["unbound_rate"]
        r["others_at_my_rate"] = {}
        for other, (per, keep, bound, unb) in data.items():
            if other == label or res[other]["kind"] == "ftfootprint":
                continue
            r["others_at_my_rate"][other] = at_rate(per, keep, bound, unb, sites, f)
    # bootstrap over sites (bound and unbound resampled): fiberstate minus other, interpolated bound rate at 1%
    if "fiberstate" in data:
        boots = {}
        pf, kf = data["fiberstate"][:2]
        for other in data:
            if other == "fiberstate":
                continue
            po, ko = data[other][:2]
            common = sorted(set(kf) & set(ko))
            cb = [i for i in common if sites[i]["bound"]]
            cu = [i for i in common if not sites[i]["bound"]]
            diffs = []
            for _ in range(1000):
                sb = rng.choice(len(cb), len(cb)); su = rng.choice(len(cu), len(cu))
                def arr(per, idx, pool):
                    return np.concatenate([per[pool[j]] for j in idx])
                d = (interp_rate(arr(pf, sb, cb), arr(pf, su, cu), 0.01)
                     - interp_rate(arr(po, sb, cb), arr(po, su, cu), 0.01))
                diffs.append(d)
            point = (interp_rate(np.concatenate([pf[i] for i in cb]), np.concatenate([pf[i] for i in cu]), 0.01)
                     - interp_rate(np.concatenate([po[i] for i in cb]), np.concatenate([po[i] for i in cu]), 0.01))
            boots[other] = dict(diff=float(point), ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                                p_two_sided=float(min(1.0, 2 * min(np.mean(np.array(diffs) <= 0), np.mean(np.array(diffs) >= 0)))))
        res["bootstrap_bound_rate_diff_at_1pct"] = boots
    json.dump(res, open(out_p, "w"), indent=1)
    # short printout
    print(f"{'tool':22s} own point (unbound / bound)   bound rate at unbound rate 0.5%   1%     2%     5%    pAUC(<=5%)")
    for label, r in res.items():
        if label.startswith("bootstrap"):
            continue
        op = r["own_point"]
        bi = r["bound_rate_interpolated"]
        ats = "  ".join(f"{100 * bi[str(f)]:5.1f}%" for f in (0.005, 0.01, 0.02, 0.05))
        print(f"{label:22s} {100 * op['unbound_rate']:4.1f}% / {100 * op['bound_rate']:4.1f}%              {ats}   {r['partial_auc_fpr_le_0.05']:.3f}")
    for other, b in res.get("bootstrap_bound_rate_diff_at_1pct", {}).items():
        print(f"  at 1% false calls, fiberstate minus {other:20s} {100 * b['diff']:+5.1f} points of bound molecules "
              f"(95% CI {100 * b['ci95'][0]:+.1f} to {100 * b['ci95'][1]:+.1f}, "
              f"p {'< 0.001' if b['p_two_sided'] == 0 else '= %.3g' % b['p_two_sided']})")


if __name__ == "__main__":
    main(*sys.argv[1:])
