"""Score how well each tool's per-molecule footprints recover ChIP-seq-bound CTCF sites.

    python score_sites.py SITES.bed RAW.bam OUT.json label1=tool1.bam [label2=tool2.bam ...]
    (a fibertools `ft footprint` table is given as  label=ftfootprint:path.tsv,
     a site/read/state table, e.g. from the HiddenFoot adapter, as  label=states:path.tsv)

For every site and every molecule spanning it, a tool "footprints" the site when one of
its footprint calls covers at least 80% of the 15 bp motif. Nucleosome calls over the
motif centre are counted separately (an occluded site cannot be bound).
A data-only reference ("raw: motif unmarked") calls a molecule bound when no A/T within
the motif +/- 10 bp (35 bp, the window fibertools' CTCF footprinting uses) carries m6A
while both 50 bp flanks do (an unmarked island inside accessible DNA).

Reported per method:
  auroc_site      bound vs unbound sites ranked by footprint fraction (main metric)
  auroc_site_open same, using footprint fraction among molecules not nucleosome-covered
  rate_bound / rate_unbound / ratio  per-molecule footprint rates
  spearman_chip   footprint fraction vs ChIP signal across bound sites
with 95% bootstrap intervals over sites, and paired bootstrap differences vs the raw rule.
"""
import json
import os
import sys

import numpy as np
import pysam
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bench.ftparse import ma_frame, parse_annotations  # noqa: E402
from fiberstate.io import fiber_arrays  # noqa: E402

MIN_FIBERS = 10
FOOT_NAMES = ("tf",)
PROT_NAMES = ("nuc", "unr")


def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def load_sites(p):
    out = []
    with open(p) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            out.append(dict(chrom=c[0], s=int(c[1]), e=int(c[2]), name=c[3], score=float(c[4]),
                            strand=c[5], bound=c[6] == "bound", signal=float(c[7])))
    return out


def tool_states(bam_p, sites):
    """{(site, read): state} with state in {'foot', 'nuc', 'open'}."""
    bam = pysam.AlignmentFile(bam_p)
    frame = ma_frame(bam.header)
    out = {}
    for i, st in enumerate(sites):
        c = (st["s"] + st["e"]) // 2
        for r in bam.fetch(st["chrom"], st["s"], st["e"]):
            if r.is_secondary or r.is_supplementary or r.reference_start > st["s"] or r.reference_end < st["e"]:
                continue
            r2q = {}
            for qp, rp in r.get_aligned_pairs(matches_only=True):
                if st["s"] <= rp < st["e"]:
                    r2q[rp] = qp
            qs = [r2q[p] for p in range(st["s"], st["e"]) if p in r2q]
            # need most of the motif aligned: 12 bases (as for CTCF and REST), or 80% of a shorter motif
            if len(qs) < min(12, int(0.8 * (st["e"] - st["s"]))) or c not in r2q:
                continue
            ann = parse_annotations(r, features=FOOT_NAMES + PROT_NAMES, MA_frame=frame)
            qa, qb = min(qs), max(qs) + 1
            state = "open"
            for a, b in ann["tf"]:
                if min(b, qb) - max(a, qa) >= 0.8 * (qb - qa):
                    state = "foot"; break
            if state == "open":
                qc = r2q[c]
                if any(a <= qc < b for a, b in ann["nuc"] + ann["unr"]):
                    state = "nuc"
            out[(i, r.query_name)] = state
    return out


def ft_footprint_states(tsv_p, sites):
    """fibertools `ft footprint` output on motif +/- 10 bp windows with modules
    [0,10) left flank, [10,25) motif core, [25,35) right flank.
    Code bit 0 = an accessible patch (MSP) spans the window; bit i+1 = module i has no m6A.
    fibertools' footprint call: spanned by an MSP and the core module unmarked."""
    key = {(st["chrom"], st["s"] - 10): i for i, st in enumerate(sites)}
    out = {}
    with open(tsv_p) as f:
        header = f.readline().lstrip("#").rstrip("\n").split("\t")
        col = {h: j for j, h in enumerate(header)}
        for line in f:
            c = line.rstrip("\n").split("\t")
            i = key.get((c[col["chrom"]], int(c[col["start"]])))
            if i is None or not c[col["fiber_names"]]:
                continue
            codes = [int(x) for x in c[col["footprint_codes"]].split(",") if x != ""]
            names = c[col["fiber_names"]].split(",")
            for code, nm in zip(codes, names):
                core_unmarked = bool(code & (1 << 2))
                if core_unmarked and code & 1:
                    out[(i, nm)] = "foot"
                elif core_unmarked:
                    out[(i, nm)] = "nuc"
                else:
                    out[(i, nm)] = "open"
    return out


def states_tsv(p, sites):
    """Per-molecule states from a table with columns site, read, state (e.g. HiddenFoot adapter)."""
    idx = {st["name"]: i for i, st in enumerate(sites)}
    out = {}
    with open(p) as f:
        f.readline()
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[0] in idx:
                out[(idx[c[0]], c[1])] = c[2]
    return out


def raw_states(bam_p, sites):
    """Data-only rule, like fibertools' CTCF footprinting: the motif +/- 10 bp carries no m6A
    while both flanks (10-60 bp beyond it) do, i.e. an unmarked island inside accessible DNA.
    Unmarked motif with an unmarked flank counts as nucleosome-like ('nuc')."""
    bam = pysam.AlignmentFile(bam_p)
    out = {}
    for i, st in enumerate(sites):
        lo, hi = st["s"] - 10, st["e"] + 10
        for r in bam.fetch(st["chrom"], lo - 60, hi + 60):
            if r.is_secondary or r.is_supplementary or r.reference_start > lo - 60 or r.reference_end < hi + 60:
                continue
            s, m = fiber_arrays(r)
            pairs = r.get_aligned_pairs(matches_only=True)
            core = [qp for qp, rp in pairs if lo <= rp < hi]
            left = [qp for qp, rp in pairs if lo - 60 <= rp < lo - 10]
            right = [qp for qp, rp in pairs if hi + 10 <= rp < hi + 60]
            if len(core) < 30 or not left or not right:
                continue
            def marked(q):
                q = np.array(q)
                return bool(((m[q] > 0) & np.isin(s[q], [65, 84])).any())
            if np.isin(s[np.array(core)], [65, 84]).sum() < 3:
                continue
            if marked(core):
                out[(i, r.query_name)] = "open"
            elif marked(left) and marked(right):
                out[(i, r.query_name)] = "foot"
            else:
                out[(i, r.query_name)] = "nuc"
    return out


def summarize(states, sites, rng, n_boot=1000):
    per = {}
    for (i, _), stt in states.items():
        per.setdefault(i, []).append(stt)
    idx = [i for i in range(len(sites)) if len(per.get(i, [])) >= MIN_FIBERS]
    if not idx:
        raise SystemExit("no site has enough scored molecules for this tool; check its input")
    y = np.array([sites[i]["bound"] for i in idx], dtype=bool)
    occ = np.array([np.mean([x == "foot" for x in per[i]]) for i in idx])
    occ_open = np.array([(np.sum([x == "foot" for x in per[i]]) / max(1, np.sum([x != "nuc" for x in per[i]]))) for i in idx])
    fb = sum(x == "foot" for i in idx if sites[i]["bound"] for x in per[i])
    nb = sum(len(per[i]) for i in idx if sites[i]["bound"])
    fu = sum(x == "foot" for i in idx if not sites[i]["bound"] for x in per[i])
    nu = sum(len(per[i]) for i in idx if not sites[i]["bound"])
    sig = np.array([sites[i]["signal"] for i in idx])
    res = dict(n_sites=len(idx), n_bound=int(y.sum()), n_unbound=int((~y).sum()),
               median_fibers_per_site=float(np.median([len(per[i]) for i in idx])) if idx else 0,
               auroc_site=auroc(occ, y), auroc_site_open=auroc(occ_open, y),
               rate_bound=fb / max(1, nb), rate_unbound=fu / max(1, nu),
               ratio=(fb / max(1, nb)) / max(1e-9, fu / max(1, nu)),
               spearman_chip=float(spearmanr(occ[y], sig[y])[0]) if y.sum() > 5 else float("nan"))
    boots = []
    for _ in range(n_boot):
        k = rng.integers(0, len(idx), len(idx))
        boots.append(auroc(occ[k], y[k]))
    res["auroc_site_ci95"] = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
    return res, dict(zip(idx, occ))


def _delong_paired(a, b, y):
    """DeLong test for two correlated AUROCs on the same sites. Returns (diff, se, z, p)."""
    from math import erfc, sqrt
    y = np.asarray(y, bool)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    m, n = len(pos), len(neg)
    V10, V01 = [], []
    for s in (a, b):
        s = np.asarray(s, float)
        sp, sn = s[pos][:, None], s[neg][None, :]
        psi = (sp > sn) + 0.5 * (sp == sn)
        V10.append(psi.mean(1)); V01.append(psi.mean(0))
    auc = [v.mean() for v in V10]
    S10 = np.cov(np.vstack(V10)); S01 = np.cov(np.vstack(V01))
    var = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / m + (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n
    se = sqrt(max(var, 1e-12))
    z = (auc[0] - auc[1]) / se
    return float(auc[0] - auc[1]), float(se), float(z), float(erfc(abs(z) / sqrt(2)))


def paired_comparisons(occs, sites, rng, n_boot=2000):
    """Every pair of methods on the sites both scored: AUROC difference (DeLong + bootstrap),
    and per-site footprint fraction at unbound sites (false footprints) and bound sites."""
    from scipy.stats import wilcoxon
    out = {}
    labels = list(occs)
    for x in range(len(labels)):
        for z_ in range(x + 1, len(labels)):
            A, B = labels[x], labels[z_]
            common = sorted(set(occs[A]) & set(occs[B]))
            y = np.array([sites[i]["bound"] for i in common])
            a = np.array([occs[A][i] for i in common]); b = np.array([occs[B][i] for i in common])
            diff, se, zz, p = _delong_paired(a, b, y)
            boots = []
            for _ in range(n_boot):
                k = rng.integers(0, len(common), len(common))
                boots.append(auroc(a[k], y[k]) - auroc(b[k], y[k]))
            boots = np.array(boots)
            def wil(u, v):
                try:
                    return float(wilcoxon(u, v).pvalue)
                except ValueError:
                    return float("nan")
            out[f"{A} vs {B}"] = dict(
                n_sites=len(common),
                auroc_diff=diff, delong_se=se, delong_z=zz, delong_p=p,
                bootstrap_ci95=[float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
                bootstrap_frac_A_better=float(np.nanmean(boots > 0)),
                unbound_mean_occ=[float(a[~y].mean()), float(b[~y].mean())],
                unbound_wilcoxon_p=wil(a[~y], b[~y]),
                bound_mean_occ=[float(a[y].mean()), float(b[y].mean())],
                bound_wilcoxon_p=wil(a[y], b[y]),
            )
    return out


def main(sites_p, raw_bam, out_p, *tools):
    sites = load_sites(sites_p)
    rng = np.random.default_rng(0)
    results, occs = {}, {}
    results["raw: motif unmarked"], occs["raw"] = summarize(raw_states(raw_bam, sites), sites, rng)
    for t in tools:
        label, path = t.split("=", 1)
        if path.startswith("ftfootprint:"):
            states = ft_footprint_states(path.split(":", 1)[1], sites)
        elif path.startswith("states:"):
            states = states_tsv(path.split(":", 1)[1], sites)
        else:
            states = tool_states(path, sites)
        results[label], occs[label] = summarize(states, sites, rng)
    # paired bootstrap: each tool's AUROC minus the raw rule, on sites scored by both
    for label in list(occs):
        if label == "raw":
            continue
        common = sorted(set(occs[label]) & set(occs["raw"]))
        y = np.array([sites[i]["bound"] for i in common])
        a = np.array([occs[label][i] for i in common]); b = np.array([occs["raw"][i] for i in common])
        d = []
        for _ in range(1000):
            k = rng.integers(0, len(common), len(common))
            d.append(auroc(a[k], y[k]) - auroc(b[k], y[k]))
        results[label]["auroc_minus_raw"] = float(auroc(a, y) - auroc(b, y))
        results[label]["auroc_minus_raw_ci95"] = [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
    results["paired"] = paired_comparisons(occs, sites, rng)
    # per-site table for follow-up analysis
    labels = list(occs)
    with open(os.path.splitext(out_p)[0] + ".per_site.tsv", "w") as f:
        f.write("site\tbound\tmotif_score\tchip_signal\t" + "\t".join(f"occ_{l}" for l in labels) + "\n")
        for i, st in enumerate(sites):
            vals = [occs[l].get(i) for l in labels]
            if all(v is None for v in vals):
                continue
            f.write(f"{st['name']}\t{int(st['bound'])}\t{st['score']}\t{st['signal']}\t"
                    + "\t".join("" if v is None else f"{v:.4f}" for v in vals) + "\n")
    json.dump(results, open(out_p, "w"), indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
