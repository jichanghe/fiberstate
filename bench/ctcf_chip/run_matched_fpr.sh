#!/usr/bin/env bash
# Matched false-positive comparison on all six TF site sets (CTCF, REST, NRF1 x GM12878, K562),
# plus a fiberstate cut-off chosen on CTCF GM12878 only and tested on the other five.
#
# Re-runs fiberstate 0.4 once per set with --posterior-tags (per-base footprint posterior);
# FiberHMM, fibertools and HiddenFoot outputs from the earlier runs are reused.
# Run in ~/fiberseq with the fiberhmm env active. Roughly 30-45 minutes with 8 threads.
set -euo pipefail
THREADS=${THREADS:-8}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP="$(pwd)"
: > matched_fpr_times.txt
one () {   # dir label sites bam fh ft hf
  local D=$1 L=$2 SITES=$3 BAM=$4 FH=$5 FT=$6 HF=$7
  ( cd "$D"
    if [ ! -s ${L}_fs_post.bam.bai ]; then
      s=$(date +%s)
      fiberstate call "$BAM" -o /dev/null --out-bam ${L}_fs_post.bam --posterior-tags -t "$THREADS" --timing 2>&1 | tail -2
      echo "$L fiberstate_0.4_seconds $(( $(date +%s) - s )) molecules $(samtools view -c ${L}_fs_post.bam)" | tee -a "$TOP/matched_fpr_times.txt"
      samtools index ${L}_fs_post.bam
    fi
    echo; echo "== $L"
    python "$HERE/matched_fpr.py" "$SITES" "$BAM" "$TOP/${L}_matched_fpr.json" \
      fiberstate=fsposterior:${L}_fs_post.bam FiberHMM=fiberhmm:$FH fibertools_footprint=ftfootprint:$FT HiddenFoot=hfscore:$HF
  )
}
one .                 CTCF_GM12878 ctcf.sites.bed ctcf_sites.bam ctcf_fh.bam       ctcf_ft_footprint.tsv hf_states_hf6.tsv
one K562_test         CTCF_K562    ctcf.sites.bed ctcf_sites.bam ctcf_sites_fh.bam ft_footprint.tsv      hf_states_hf6.tsv
one REST_test/GM12878 REST_GM12878 tf.sites.bed   sites.bam      sites_fh.bam      ft_footprint.tsv      hf_states_hf6.tsv
one REST_test/K562    REST_K562    tf.sites.bed   sites.bam      sites_fh.bam      ft_footprint.tsv      hf_states_hf6.tsv
one NRF1_test/GM12878 NRF1_GM12878 tf.sites.bed   sites.bam      sites_fh.bam      ft_footprint.tsv      hf_states.tsv
one NRF1_test/K562    NRF1_K562    tf.sites.bed   sites.bam      sites_fh.bam      ft_footprint.tsv      hf_states.tsv
python "$HERE/summarize_matched_fpr.py" matched_fpr_summary.json CTCF_GM12878 \
  $(for L in CTCF_GM12878 CTCF_K562 REST_GM12878 REST_K562 NRF1_GM12878 NRF1_K562; do echo "$L=${L}_matched_fpr.json"; done)
tar czf matched_fpr_results.tar.gz *_matched_fpr.json matched_fpr_summary.json matched_fpr_times.txt
ls -lh matched_fpr_results.tar.gz; echo "Send back: matched_fpr_results.tar.gz"
