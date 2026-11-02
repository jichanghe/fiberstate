#!/usr/bin/env bash
# Re-run fiberstate with posterior footprint calling at two thresholds on the existing CTCF
# site sets (GM12878 in ~/fiberseq, K562 in ~/fiberseq/K562_test) and re-score all tools.
# Run in ~/fiberseq with the fiberhmm env active. About 20-30 minutes with 8 threads.
set -euo pipefail
THREADS=${THREADS:-8}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_set () {   # dir  sites  fs_old  fh  ft_tsv  label
  local D=$1 SITES=$2 FS=$3 FH=$4 FT=$5 L=$6
  ( cd "$D"
    for t in 0.7 0.5; do
      out=ctcf_fs_p${t/./}.bam
      if [ ! -s $out.bai ]; then
        s=$(date +%s); fiberstate call ctcf_sites.bam -o /dev/null --out-bam $out -t "$THREADS" --foot-posterior $t
        samtools index $out; echo "$L fiberstate posterior>$t: $(( $(date +%s) - s )) s"
      fi
    done
    python "$HERE/score_sites.py" "$SITES" ctcf_sites.bam ${L}_posterior_results.json \
      fiberstate_viterbi=$FS fiberstate_p07=ctcf_fs_p07.bam fiberstate_p05=ctcf_fs_p05.bam \
      FiberHMM=$FH fibertools_footprint=ftfootprint:$FT > /dev/null
    python - ${L}_posterior_results.json "$L" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); print(f"\n== {sys.argv[2]}")
for k, v in d.items():
    if k != "paired":
        print(f"  {k:22s} AUROC {v['auroc_site']:.3f}  false footprints at unbound {100*v['rate_unbound']:.1f}%  at bound {100*v['rate_bound']:.1f}%")
for k, v in d["paired"].items():
    if "raw" not in k and ("p07" in k or "p05" in k):
        print(f"  {k:44s} diff {v['auroc_diff']:+.3f}  DeLong p={v['delong_p']:.3g}")
PY
  )
}
run_set . ctcf.sites.bed ctcf_fs.bam ctcf_fh.bam ctcf_ft_footprint.tsv GM12878
run_set K562_test ctcf.sites.bed ctcf_sites_fs.bam ctcf_sites_fh.bam ft_footprint.tsv K562
tar czf posterior_results.tar.gz GM12878_posterior_results.json K562_test/K562_posterior_results.json
echo "Send back: posterior_results.tar.gz"
