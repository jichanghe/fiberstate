"""Run HiddenFoot (Molina lab, bioRxiv 2025.05.13.653852) on Fiber-seq molecules at motif sites.

HiddenFoot works one region at a time. For each site this writes, in OUTDIR/<site>/:
  seq.txt       reference window (motif centre +/- HALF bp), one plain line
  met.txt       header = 0-based window positions of A/T bases; one row per molecule:
                1 = m6A (accessible), 0 = A/T without m6A, 2 = no aligned base (missing)
  reads.txt     read names in row order
  motif.txt     the TF motif in HiddenFoot's format ("> NAME LEN", then "i A C G T" rows)
and a run list for `hiddenfoot run`.

    python -I bench/hiddenfoot_adapter.py prepare SITES.bed RAW.bam REF.fa OUTDIR [MOTIF.jaspar] [HALF=150]
    python -I bench/hiddenfoot_adapter.py collect SITES.bed OUTDIR OUT_states.tsv [THRESH=0.5] [TAG=hf]

Run HiddenFoot with -cutoffwms 1e-6 or lower: its default (1e-3, i.e. within ln(1000) of the
consensus score) rejects most motif matches that pass the JASPAR 85% relative-score rule.

`collect` reads OUTDIR/<site>/hf.hfprofile.<NAME>.txt (molecules x positions, posterior that the
TF covers each base) and calls a molecule footprinted when the mean posterior over the motif
exceeds THRESH; nucleosome-covered motif centres (hf.hfprofile.nucleosome.txt > 0.5) are 'nuc'.
"""
import os
import sys

import numpy as np
import pysam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fiberstate.io import fiber_arrays  # noqa: E402

CTCF_MA0139_2 = [  # JASPAR MA0139.2, rows A C G T
    [281, 56, 8, 744, 40, 107, 851, 5, 333, 54, 12, 56, 104, 372, 82],
    [49, 800, 903, 13, 528, 433, 11, 0, 3, 12, 0, 8, 733, 13, 482],
    [449, 21, 0, 65, 334, 48, 32, 903, 566, 504, 890, 775, 5, 507, 307],
    [134, 36, 2, 91, 11, 324, 18, 3, 9, 341, 8, 71, 67, 17, 37],
]


def read_jaspar(p):
    """JASPAR .jaspar file -> (name, 4 x L counts)."""
    name, rows = "TF", []
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[-1] if len(line[1:].split()) > 1 else line[1:].strip()
            continue
        if not line:
            continue
        vals = line.split("[")[1].split("]")[0].split() if "[" in line else line.split()[1:]
        rows.append([float(v) for v in vals])
    return name.replace(" ", "_"), rows


def load_sites(p):
    out = []
    for line in open(p):
        c = line.rstrip("\n").split("\t")
        out.append(dict(chrom=c[0], s=int(c[1]), e=int(c[2]), name=c[3]))
    return out


def prepare(sites_p, bam_p, ref_p, outdir, motif_p=None, half=150):
    half = int(half)
    name, pfm = ("CTCF", CTCF_MA0139_2) if not motif_p else read_jaspar(motif_p)
    L = len(pfm[0])
    os.makedirs(outdir, exist_ok=True)
    fa = pysam.FastaFile(ref_p)
    bam = pysam.AlignmentFile(bam_p)
    runlist = []
    n_written = 0
    for st in load_sites(sites_p):
        ctr = (st["s"] + st["e"]) // 2
        lo, hi = ctr - half, ctr + half + 1
        ref = fa.fetch(st["chrom"], lo, hi).upper()
        at = [i for i, b in enumerate(ref) if b in "AT"]
        rows, names = [], []
        for r in bam.fetch(st["chrom"], lo, hi):
            if r.is_secondary or r.is_supplementary or r.reference_start > lo or r.reference_end < hi:
                continue
            s, m = fiber_arrays(r)
            r2q = {}
            for qp, rp in r.get_aligned_pairs(matches_only=True):
                if lo <= rp < hi:
                    r2q[rp - lo] = qp
            row = []
            for i in at:
                qp = r2q.get(i)
                if qp is None or s[qp] not in (65, 84):
                    row.append(2)
                else:
                    row.append(1 if m[qp] > 0 else 0)
            rows.append(row)
            names.append(r.query_name)
        if len(rows) < 10:
            continue
        d = os.path.join(outdir, st["name"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "seq.txt"), "w") as f:
            f.write(ref + "\n")
        with open(os.path.join(d, "met.txt"), "w") as f:
            f.write(" ".join(map(str, at)) + "\n")
            for row in rows:
                f.write(" ".join(map(str, row)) + "\n")
        with open(os.path.join(d, "reads.txt"), "w") as f:
            f.write("\n".join(names) + "\n")
        with open(os.path.join(d, "motif.txt"), "w") as f:
            f.write(f"> {name} {L}\n")
            for j in range(L):
                f.write(f"{j} " + " ".join(str(pfm[k][j]) for k in range(4)) + "\n")
        with open(os.path.join(d, "window.txt"), "w") as f:
            f.write(f"{st['chrom']}\t{lo}\t{hi}\t{st['s'] - lo}\t{st['e'] - lo}\t{name}\n")
        runlist.append(st["name"])
        n_written += 1
    with open(os.path.join(outdir, "sites.txt"), "w") as f:
        f.write("\n".join(runlist) + "\n")
    print(f"prepared {n_written} sites in {outdir} (motif {name}, {L} bp, window {2 * half + 1} bp)")


def collect(sites_p, outdir, out_p, thresh=0.5, tag="hf"):
    thresh = float(thresh)
    n_sites = n_missing = 0
    with open(out_p, "w") as out:
        out.write("site\tread\tstate\tmotif_posterior\n")
        for st in load_sites(sites_p):
            d = os.path.join(outdir, st["name"])
            if not os.path.exists(os.path.join(d, "window.txt")):
                continue
            chrom, lo, hi, ms, me, name = open(os.path.join(d, "window.txt")).read().split()
            ms, me = int(ms), int(me)
            prof = os.path.join(d, f"{tag}.hfprofile.{name}.txt")
            nucp = os.path.join(d, f"{tag}.hfprofile.nucleosome.txt")
            reads = open(os.path.join(d, "reads.txt")).read().split()
            if not os.path.exists(prof):
                # no motif instance passed HiddenFoot's PWM cutoff: no TF can be called here
                n_missing += 1
                tf = np.zeros((len(reads), int(hi) - int(lo)))
            else:
                tf = np.loadtxt(prof, ndmin=2)
            nuc = np.loadtxt(nucp, ndmin=2) if os.path.exists(nucp) else np.zeros_like(tf)
            if tf.shape[0] != len(reads):
                print(f"warning: {st['name']}: {tf.shape[0]} rows vs {len(reads)} reads, skipped", file=sys.stderr)
                continue
            n_sites += 1
            c = (ms + me) // 2
            for i, rn in enumerate(reads):
                p = float(tf[i, ms:me].mean())
                state = "foot" if p > thresh else ("nuc" if nuc[i, c] > 0.5 else "open")
                out.write(f"{st['name']}\t{rn}\t{state}\t{p:.4f}\n")
    print(f"collected {n_sites} sites ({n_missing} without a motif profile) -> {out_p}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"prepare": prepare, "collect": collect}[cmd](*args)
