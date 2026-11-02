#!/usr/bin/env bash
# Fetch the two small reference files for the DNase-seq accessibility test (run in ~/fiberseq).
set -euo pipefail
# GM12878 DNase-seq peaks, ENCODE ENCSR000EMT, GRCh38 (released)
[ -s ENCFF784FOQ.bed.gz ] || wget -q -O ENCFF784FOQ.bed.gz https://www.encodeproject.org/files/ENCFF784FOQ/@@download/ENCFF784FOQ.bed.gz
zcat ENCFF784FOQ.bed.gz | awk '$1=="chr19"' > dnase_chr19.bed
# FIRE's own per-base accessibility track for the same 500 kb region
tabix -h https://s3.kopah.uw.edu/stergachis/public/FIRE/broadly-consented/GM12878/GM12878-fire-v0.1-pileup.bed.gz \
  chr19:45000000-45500000 > fire_pileup_region.bed
echo "dnase peaks on chr19: $(wc -l < dnase_chr19.bed)   pileup lines: $(wc -l < fire_pileup_region.bed)"
echo "Send back: dnase_chr19.bed fire_pileup_region.bed"
