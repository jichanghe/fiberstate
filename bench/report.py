"""Aggregate benchmark JSON (mean and sd over seeds) into results/summary.json."""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def agg(rows, extra=None):
    out = {}
    by = {}
    for r in rows:
        for m, v in r["metrics"].items():
            by.setdefault((m, r["value"]), []).append(v)
    if extra:
        for r in extra:
            by.setdefault(("fibertools", r["value"]), []).append(r["metrics"])
    for (m, x), lst in by.items():
        keys = lst[0].keys()
        out.setdefault(m, {})[x] = {k: (float(np.nanmean([d[k] for d in lst])),
                                        float(np.nanstd([d[k] for d in lst])))
                                    for k in keys}
    return {m: [dict(x=x, **{k: v for k, v in d.items()}) for x, d in sorted(xs.items())]
            for m, xs in out.items()}


def main(d="results"):
    r = json.load(open(os.path.join(d, "results.json")))
    ft = None
    p = os.path.join(d, "ft_results.json")
    if os.path.exists(p):
        ft = json.load(open(p))
    s = dict(
        tuned=r["tuned"],
        sensitivity=agg(r["sensitivity"], ft["sensitivity"] if ft else None),
        fp_rate=agg(r["fp_rate"], ft["fp_rate"] if ft else None),
        cooccupancy=r["cooccupancy"],
    )
    json.dump(s, open(os.path.join(d, "summary.json"), "w"), indent=1)
    sens = s["sensitivity"]
    print("method, sens=0.6: nucF1 footF1 dyad falseIsland occMAE ms")
    for m, rows in sens.items():
        row = [x for x in rows if abs(x["x"] - 0.6) < 1e-9][0]
        g = lambda k: row.get(k, (float("nan"), 0))[0]
        print(f"{m:18s} {g('nuc_f1'):.3f} {g('foot_f1'):.3f} {g('dyad_median_err'):.1f} "
              f"{g('false_nuc_per_10kb_island'):.2f} {g('occupancy_mae'):.3f} {g('ms_per_read'):.1f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
