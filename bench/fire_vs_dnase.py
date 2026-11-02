"""Score FIRE's own accessibility track against DNase-seq peaks, on the same 100 bp bins.

    python -I bench/fire_vs_dnase.py FIRE_PILEUP_REGION.bed DNASE_PEAKS.bed[.gz] CHROM START END OUT.json

FIRE_PILEUP_REGION.bed is `tabix -h GM12878-fire-v0.1-pileup.bed.gz CHROM:START-END`.
FIRE's per-position accessibility = fire_coverage / coverage (fraction of molecules with a
FIRE element over the base). Bins need >= 20 molecules, as for the other tools.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.metrics import auroc  # noqa: E402
from bench.score_annotated_bam import read_intervals  # noqa: E402


def main(pile_p, dnase_p, chrom, start, end, out_p):
    lo, hi = int(start), int(end)
    n = hi - lo
    cov = np.zeros(n); fire = np.zeros(n)
    with open(pile_p) as f:
        header = None
        for line in f:
            if line.startswith("#"):
                header = line.lstrip("#").rstrip("\n").split("\t")
                continue
            c = line.rstrip("\n").split("\t")
            if header is None:
                raise SystemExit("pileup header missing: run tabix with -h")
            col = {h: j for j, h in enumerate(header)}
            a, b = max(lo, int(c[1])), min(hi, int(c[2]))
            if a >= b:
                continue
            cov[a - lo:b - lo] = float(c[col["coverage"]])
            fire[a - lo:b - lo] = float(c[col["fire_coverage"]])
    dm = np.zeros(n, bool)
    for a, b in read_intervals(dnase_p, chrom):
        dm[max(0, a - lo):max(0, min(hi, b) - lo)] = True
    B = 100; nb = n // B
    binned = lambda x: x[:nb * B].reshape(nb, B).mean(1)  # noqa: E731
    good = binned((cov >= 20).astype(float)) >= 0.9
    score = binned(fire / np.maximum(cov, 1))
    lab = binned(dm.astype(float)) >= 0.5
    res = dict(label="FIRE pileup", dnase_auroc=auroc(score[good], lab[good]),
               dnase_bins_positive=int((lab & good).sum()), bins_scored=int(good.sum()),
               pileup_columns=header)
    json.dump(res, open(out_p, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
