"""Learn the Hia5 context table from a Fiber-seq BAM.

    python -I bench/learn_ctx_real.py IN.bam OUT.tsv K [max_fibers]
"""
import json
import os
import sys

import numpy as np
import pysam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fiberstate.context import kmer_string  # noqa: E402
from fiberstate.io import fiber_arrays  # noqa: E402
from fiberstate.learn import learn_context  # noqa: E402


def main(bam_p, out_p, k, maxf=None):
    k = int(k)
    fibers = []
    with pysam.AlignmentFile(bam_p, check_sq=False) as b:
        for r in b:
            if r.is_secondary or r.is_supplementary:
                continue
            s, m = fiber_arrays(r)
            if s is not None and len(s) >= 1000:
                fibers.append((s, m))
            if maxf and len(fibers) >= int(maxf):
                break
    ctx, stats = learn_context(fibers, k=k, iterations=2)
    ctx.save(out_p)
    order = np.argsort(ctx.weights)
    print(json.dumps(dict(n_fibers=len(fibers), stats=stats), indent=1))
    print("lowest:", [(kmer_string(c, k), round(ctx.weights[c], 2)) for c in order[:8]])
    print("highest:", [(kmer_string(c, k), round(ctx.weights[c], 2)) for c in order[::-1][:8]])


if __name__ == "__main__":
    main(*sys.argv[1:])
