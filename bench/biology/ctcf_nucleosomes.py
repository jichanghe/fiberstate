"""Single-molecule CTCF binding and the nucleosomes around it.

At ChIP-bound and unbound CTCF motif sites, every molecule is classed by what fiberstate calls
over the motif (footprint calls made at a 1% false-call rate):
    footprinted   a footprint covers >= 80% of the motif           (CTCF bound on this molecule)
    occluded      a nucleosome (or unresolved protected stretch) covers the motif centre
    open          neither: the motif is accessible but not footprinted
For each class, nucleosome occupancy is averaged at every position from -FLANK to +FLANK bp around
the motif centre, oriented by motif strand. A classic expectation is that phased nucleosome
arrays flank CTCF only on molecules where CTCF is bound; this tests it molecule by molecule.

    python ctcf_nucleosomes.py SITES.bed ALIGNED.bam CALLS.bed OUT_PREFIX [FLANK=1000]

SITES.bed  site table from select_sites.py (chrom start end name score strand bound/unbound signal)
ALIGNED.bam  the indexed BAM the calls were made from (for each molecule's aligned span)
CALLS.bed  `fiberstate call ... --false-call-rate 0.01 -o CALLS.bed` on that BAM
Writes OUT_PREFIX.profiles.tsv, OUT_PREFIX.summary.json and OUT_PREFIX.png/pdf.
"""
import json
import sys
from collections import defaultdict

import numpy as np
import pysam

CLASSES = ("footprinted", "open", "occluded")
MIN_MOL = 10


def load_sites(p):
    out = []
    for line in open(p):
        if line.startswith("#") or not line.strip():
            continue
        c = line.rstrip("\n").split("\t")
        out.append(dict(chrom=c[0], s=int(c[1]), e=int(c[2]), strand=c[5], bound=c[6] == "bound",
                        signal=float(c[7]) if len(c) > 7 and c[7] not in ("", ".") else float("nan")))
    return out


def load_calls(p, regions):
    """read name -> list of (type, start, end) for segments on the site chromosomes."""
    chroms = {r[0] for r in regions}
    calls = defaultdict(list)
    for line in open(p):
        if line.startswith("#"):
            continue
        c = line.split("\t", 6)
        if c[0] not in chroms:
            continue
        calls[c[4]].append((c[3], int(c[1]), int(c[2])))
    return calls


def classify(segs, s, e, centre):
    L = e - s
    for t, a, b in segs:
        if t == "footprint" and min(b, e) - max(a, s) >= 0.8 * L:
            return "footprinted"
    for t, a, b in segs:
        if t in ("nucleosome", "protected_unresolved") and a <= centre < b:
            return "occluded"
    return "open"


def main(sites_p, bam_p, calls_p, out, flank=1000):
    flank = int(flank)
    sites = load_sites(sites_p)
    calls = load_calls(calls_p, [(st["chrom"],) for st in sites])
    bam = pysam.AlignmentFile(bam_p)
    W = 2 * flank + 1
    cov = {(lab, k): np.zeros(W) for lab in ("bound", "unbound") for k in CLASSES}
    nuc = {key: np.zeros(W) for key in cov}
    per_site = []          # per site: counts by class, and per-class (coverage, nucleosome) near +/-1 nucleosomes
    n_without_calls = 0
    xs = np.arange(-flank, flank + 1)
    BAND = {"centre": np.abs(xs) <= 50, "flank": (np.abs(xs) >= 150) & (np.abs(xs) <= 250)}
    for st in sites:
        centre = (st["s"] + st["e"]) // 2
        lab = "bound" if st["bound"] else "unbound"
        counts = dict.fromkeys(CLASSES, 0)
        bands = {(k, b): [0.0, 0.0] for k in CLASSES for b in BAND}   # (nucleosome bases, covered bases)
        seen = set()
        for r in bam.fetch(st["chrom"], centre, centre + 1):
            if r.is_secondary or r.is_supplementary or r.query_name in seen:
                continue
            if r.reference_start > st["s"] or r.reference_end < st["e"]:
                continue
            seen.add(r.query_name)
            segs = calls.get(r.query_name)
            if segs is None:           # a molecule with no protected segments at all is fully open
                segs = []
                n_without_calls += 1
            k = classify(segs, st["s"], st["e"], centre)
            counts[k] += 1
            lo, hi = max(r.reference_start, centre - flank), min(r.reference_end, centre + flank + 1)
            c = np.zeros(W); c[lo - (centre - flank):hi - (centre - flank)] = 1
            n = np.zeros(W)
            for t, a, b in segs:
                if t == "nucleosome":
                    a2, b2 = max(a, lo), min(b, hi)
                    if a2 < b2:
                        n[a2 - (centre - flank):b2 - (centre - flank)] = 1
            if st["strand"] == "-":
                c, n = c[::-1], n[::-1]
            cov[(lab, k)] += c
            nuc[(lab, k)] += n
            for bn, m in BAND.items():
                bands[(k, bn)][0] += n[m].sum(); bands[(k, bn)][1] += c[m].sum()
        tot = sum(counts.values())
        if tot >= MIN_MOL:
            per_site.append(dict(bound=st["bound"], signal=st["signal"], n=tot, bands=bands, **counts))

    prof = {key: np.divide(nuc[key], cov[key], out=np.full(W, np.nan), where=cov[key] > 0) for key in cov}
    x = np.arange(-flank, flank + 1)
    with open(out + ".profiles.tsv", "w") as f:
        f.write("pos\t" + "\t".join(f"{lab}_{k}" for lab, k in cov) + "\t" + "\t".join(f"n_{lab}_{k}" for lab, k in cov) + "\n")
        for i in range(W):
            f.write(f"{x[i]}\t" + "\t".join(f"{prof[key][i]:.4f}" for key in cov) + "\t" + "\t".join(f"{int(cov[key][i])}" for key in cov) + "\n")

    def band(p, a, b):  # mean occupancy over |pos| in [a, b]
        m = (np.abs(x) >= a) & (np.abs(x) <= b)
        v = p[m]
        return float(np.nanmean(v)) if np.any(np.isfinite(v)) else float("nan")

    summ = dict(sites_scored=len(per_site), molecules_without_calls=n_without_calls, by_label={})
    for lab in ("bound", "unbound"):
        rows = [p for p in per_site if p["bound"] == (lab == "bound")]
        n = sum(p["n"] for p in rows)
        d = dict(sites=len(rows), molecules=n,
                 share=({k: sum(p[k] for p in rows) / n for k in CLASSES} if n else {}),
                 occupancy={})
        for k in CLASSES:
            p = prof[(lab, k)]
            d["occupancy"][k] = dict(centre_0_50=band(p, 0, 50), flank_150_250=band(p, 150, 250),
                                     far_600_1000=band(p, 600, 1000),
                                     molecules_at_centre=int(cov[(lab, k)][flank]))
        summ["by_label"][lab] = d
    # per-site footprinted share vs ChIP signal at bound sites
    b = [p for p in per_site if p["bound"] and np.isfinite(p["signal"])]
    if len(b) >= 10:
        from scipy.stats import spearmanr
        rho, pv = spearmanr([p["footprinted"] / p["n"] for p in b], [p["signal"] for p in b])
        summ["bound_sites_footprinted_share_vs_chip_signal"] = dict(spearman=float(rho), p=float(pv), sites=len(b))
    # site bootstrap: nucleosome occupancy 150-250 bp from the motif, footprinted minus open molecules (bound sites)
    rows = [p for p in per_site if p["bound"]]
    def diff(idx):
        f = np.array([rows[i]["bands"][("footprinted", "flank")] for i in idx]).sum(0)
        o = np.array([rows[i]["bands"][("open", "flank")] for i in idx]).sum(0)
        return f[0] / f[1] - o[0] / o[1] if f[1] and o[1] else np.nan
    if len(rows) >= 10:
        rng = np.random.default_rng(0)
        point = diff(range(len(rows)))
        bs = np.array([diff(rng.integers(0, len(rows), len(rows))) for _ in range(1000)])
        bs = bs[np.isfinite(bs)]
        summ["flank_150_250_footprinted_minus_open_bound"] = dict(
            diff=float(point), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            p_two_sided=float(min(1.0, 2 * min(np.mean(bs <= 0), np.mean(bs >= 0)))))
    for p_ in per_site:
        p_.pop("bands")
    json.dump(summ, open(out + ".summary.json", "w"), indent=1)
    plot(x, prof, out)
    print(json.dumps(summ, indent=1))


def plot(x, prof, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"footprinted": "#0072B2", "open": "#009E73", "occluded": "#7f7f7f"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, lab in zip(axes, ("bound", "unbound")):
        for k in CLASSES:
            p = prof[(lab, k)]
            if np.all(np.isnan(p)):
                continue
            ax.plot(x, p, color=col[k], lw=1.2, label=k)
        ax.set_title(f"ChIP-{lab} CTCF sites")
        ax.set_xlabel("Distance from motif centre (bp)")
        ax.axvline(0, color="#bbbbbb", lw=0.6, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Nucleosome occupancy")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300)


if __name__ == "__main__":
    main(*sys.argv[1:])
