"""AUROC of bound vs unbound CTCF sites split by ChIP-seq signal tertile of the bound sites.

    python -I bench/ctcf_chip/by_chip_strength.py PER_SITE.tsv OUT.json
Columns occ_<tool> in PER_SITE.tsv are the per-site footprinted fraction.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_sites import auroc  # noqa: E402


def main(p, out_p):
    rows = [l.rstrip("\n").split("\t") for l in open(p)]
    h, rows = rows[0], rows[1:]
    col = {k: i for i, k in enumerate(h)}
    tools = [k[4:] for k in h if k.startswith("occ_")]
    b = [r for r in rows if r[col["bound"]] == "1"]
    u = [r for r in rows if r[col["bound"]] == "0"]
    sig = np.array([float(r[col["chip_signal"]]) for r in b])
    q1, q2 = np.quantile(sig, [1 / 3, 2 / 3])
    groups = {"weak ChIP": sig <= q1, "medium": (sig > q1) & (sig <= q2), "strong": sig > q2}
    res = {}
    for g, m in groups.items():
        bb = [r for r, k in zip(b, m) if k]
        res[g] = {}
        res[g]["mean_occ_bound"] = {}
        for t in tools:   # a tool can lack a value where it scored fewer than MIN_FIBERS molecules
            k = col["occ_" + t]
            xb = [float(r[k]) for r in bb if r[k] != ""]
            xu = [float(r[k]) for r in u if r[k] != ""]
            y = np.array([True] * len(xb) + [False] * len(xu))
            res[g][t] = float(auroc(np.array(xb + xu), y))
            res[g]["mean_occ_bound"][t] = float(np.mean(xb)) if xb else float("nan")
        res[g]["n_bound"] = len(bb)
    json.dump(res, open(out_p, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
