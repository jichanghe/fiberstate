#!/usr/bin/env bash
# Biological analysis for the Genome Research manuscript: nucleosomes around CTCF sites on single
# molecules, split by whether fiberstate calls CTCF bound on each molecule (1% false-call rate).
#
# Run in ~/fiberseq with the fiberhmm env active, after the CTCF benchmarks (it reuses
# ctcf.sites.bed and ctcf_sites.bam in . and in K562_test/). About 10-15 minutes with 8 threads.
set -euo pipefail
THREADS=${THREADS:-8}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p bio
for L in GM12878 K562; do
  if [ $L = GM12878 ]; then D=.; else D=K562_test; fi
  echo "== $L: fiberstate call at 1% false calls"
  [ -s bio/$L.calls.bed ] || fiberstate call "$D/ctcf_sites.bam" -o bio/$L.calls.bed --false-call-rate 0.01 -t "$THREADS" 2>&1 | tail -1
  echo "== $L: nucleosome profiles"
  python -I "$HERE/ctcf_nucleosomes.py" "$D/ctcf.sites.bed" "$D/ctcf_sites.bam" bio/$L.calls.bed bio/$L 1000 \
    | python -c "import json,sys; d=json.load(sys.stdin); print(json.dumps({k: d[k] for k in d if k != 'by_label'}, indent=1)); [print(l, {k: round(v, 3) for k, v in d['by_label'][l]['share'].items()}) for l in d['by_label']]"
done
tar czf bio_results.tar.gz bio/*.summary.json bio/*.profiles.tsv bio/*.png
ls -lh bio_results.tar.gz; echo "Send back: bio_results.tar.gz"
