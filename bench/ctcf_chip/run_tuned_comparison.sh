#!/usr/bin/env bash
# Tuned-competitor comparison: each tool's setting chosen on CTCF GM12878, applied unchanged to
# CTCF K562, REST and NRF1 in both cell lines. Reuses outputs of the earlier runs (no tool is re-run):
# fiberstate *_fs_post.bam from run_matched_fpr.sh, FiberHMM BAMs, ft footprint tables, HiddenFoot tables.
# Run in ~/fiberseq with the fiberhmm env active. About 1-2 hours.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP="$(pwd)"
mkdir -p tuned
one () {   # dir label sites fs fh ft hf
  local D=$1 L=$2 SITES=$3 FS=$4 FH=$5 FT=$6 HF=$7
  [ -s "tuned/$L.json" ] && { echo "$L: done"; return; }
  for f in "$SITES" "$FS" "$FH" "$FT"; do [ -s "$D/$f" ] || { echo "$L: missing $D/$f"; exit 1; }; done
  local HFARG=(); [ -s "$D/$HF" ] && HFARG=(hf="$D/$HF")
  python "$HERE/tuned_comparison.py" collect "$L" "$D/$SITES" "tuned/$L.json" \
    fs="$D/$FS" fh="$D/$FH" ft="$D/$FT" "${HFARG[@]}"
}
one .                 CTCF_GM12878 ctcf.sites.bed CTCF_GM12878_fs_post.bam ctcf_fh.bam       ctcf_ft_footprint.tsv hf_states_hf6.tsv
one K562_test         CTCF_K562    ctcf.sites.bed CTCF_K562_fs_post.bam    ctcf_sites_fh.bam ft_footprint.tsv      hf_states_hf6.tsv
one REST_test/GM12878 REST_GM12878 tf.sites.bed   REST_GM12878_fs_post.bam sites_fh.bam      ft_footprint.tsv      hf_states_hf6.tsv
one REST_test/K562    REST_K562    tf.sites.bed   REST_K562_fs_post.bam    sites_fh.bam      ft_footprint.tsv      hf_states_hf6.tsv
one NRF1_test/GM12878 NRF1_GM12878 tf.sites.bed   NRF1_GM12878_fs_post.bam sites_fh.bam      ft_footprint.tsv      hf_states.tsv
one NRF1_test/K562    NRF1_K562    tf.sites.bed   NRF1_K562_fs_post.bam    sites_fh.bam      ft_footprint.tsv      hf_states.tsv
python "$HERE/tuned_comparison.py" compare CTCF_GM12878 tuned/summary.json \
  tuned/CTCF_GM12878.json tuned/CTCF_K562.json tuned/REST_GM12878.json tuned/REST_K562.json tuned/NRF1_GM12878.json tuned/NRF1_K562.json \
  | tee tuned/summary.txt
tar czf tuned_results.tar.gz tuned/summary.json tuned/summary.txt
ls -lh tuned_results.tar.gz; echo "Send back: tuned_results.tar.gz (or just paste the printed table)"
