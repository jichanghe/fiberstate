"""Print a short summary of a Fiber-seq BAM and BED files (paths given as arguments)."""
import gzip
import sys

import numpy as np
import pysam

bam, *beds = sys.argv[1:]
b = pysam.AlignmentFile(bam)
lens, n, first = [], 0, None
for r in b:
    if r.is_secondary or r.is_supplementary:
        continue
    n += 1
    lens.append(r.query_length)
    if first is None:
        first = r
print("reads", n, "median length", np.median(lens), "contigs", len(b.header.references))
print("first", first.reference_name, first.reference_start, first.reference_end)
print("tags", [t for t, _ in first.get_tags()])
print("MM head", first.get_tag("MM")[:60] if first.has_tag("MM") else None)
mods = first.modified_bases
print("mod keys", list(mods.keys()) if mods else None)
for p in beds:
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f:
        lines = f.readlines()
    print(p.split("/")[-1], len(lines), "lines")
    for line in lines[:3]:
        print("  ", line.rstrip()[:200])
