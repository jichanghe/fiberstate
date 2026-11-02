#!/usr/bin/env bash
# Where does fiberstate's time go on this computer? Run in ~/fiberseq with the fiberhmm env active.
set -euo pipefail
BAM=${1:-ctcf_sites.bam}
echo "== machine"
echo "nproc: $(nproc)"
grep -m1 "model name" /proc/cpuinfo || true
free -g | head -2
python -c "import numba, numpy; print('numba', numba.__version__, 'numpy', numpy.__version__)"

echo "== 2000-fiber subset"
samtools view -h -F 0x900 "$BAM" | awk '/^@/ || n++ < 2000' | samtools view -b -o speed_subset.bam -
samtools index speed_subset.bam
fiberstate call speed_subset.bam -o /dev/null -t 1 >/dev/null 2>&1 || true    # warm-up: compile once

for t in 1 4; do
  fiberstate call speed_subset.bam -o /dev/null -t "$t" --timing
done
s=$(date +%s.%N)
fiberhmm-call -i speed_subset.bam -o speed_fh.bam --enzyme hia5 --seq pacbio -c 4 > /dev/null 2>&1
echo "fiberhmm 4 cores: $(echo "$(date +%s.%N) - $s" | bc) s for 2000 fibers"
