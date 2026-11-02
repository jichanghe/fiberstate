#!/usr/bin/env bash
# Rerun HiddenFoot with a motif cutoff that admits every selected site, then rescore all tools.
#
# Why: HiddenFoot only considers a TF where 0.5*exp(PWM score - best score) >= -cutoffwms
# (default 1e-3). 86% of the CTCF and 77% of the REST motif matches chosen by the JASPAR 85%
# rule fall below that, so the first run never let HiddenFoot bind at most test sites
# ("without a motif profile"). 1e-6 admits all of them (lowest seen 4e-6).
# Also fixes the ChIP-strength split crashing on sites where one tool had too few molecules.
#
# Run in ~/fiberseq with the fiberhmm env active, after run_hiddenfoot_ctcf.sh and run_tf_test.sh.
set -euo pipefail
THREADS=${THREADS:-8}
CUT=${CUT:-1e-6}
TAG=hf6
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HERE/.." && pwd)"
TOP="$(pwd)"
HF="$TOP/HiddenFoot/hiddenfoot"
[ -x "$HF" ] || { echo "HiddenFoot/hiddenfoot not found; run run_hiddenfoot_ctcf.sh first"; exit 1; }
REST_REF=$(ls "$TOP"/refs/hg38_*.fa | head -1)

one () {   # dir sites bam ref motif(or -) out_label fs fh ft
  local D=$1 SITES=$2 BAM=$3 REF=$4 MOTIF=$5 L=$6 FS=$7 FH=$8 FT=$9
  ( cd "$D"
    [ -s hf_sites/sites.txt ] || python "$BENCH/hiddenfoot_adapter.py" prepare "$SITES" "$BAM" "$REF" hf_sites $([ "$MOTIF" = - ] || echo "$MOTIF")
    s=$(date +%s)
    grep -v '^$' hf_sites/sites.txt | xargs -P "$THREADS" -I{} sh -c '
      d=hf_sites/{}; [ -s $d/'$TAG'.hfprofile.nucleosome.txt ] && exit 0
      "'"$HF"'" run -s $d/seq.txt -m $d/met.txt -w $d/motif.txt -o $d/'$TAG' -padlen 150 -nuclen 147 -cutoffwms '$CUT' -numthreads 1 > $d/'$TAG'.log 2>&1 \
        || echo "HiddenFoot failed at {} (see $d/'$TAG'.log)"'
    echo "hiddenfoot_${L}_seconds $(( $(date +%s) - s ))" > ${L}_hf_times.txt
    python "$BENCH/hiddenfoot_adapter.py" collect "$SITES" hf_sites hf_states_$TAG.tsv 0.5 $TAG
    python "$HERE/score_sites.py" "$SITES" "$BAM" ${L}.json \
      fiberstate=$FS FiberHMM=$FH fibertools_footprint=ftfootprint:$FT HiddenFoot=states:hf_states_$TAG.tsv > /dev/null
    python "$HERE/by_chip_strength.py" ${L}.per_site.tsv ${L}_by_chip_strength.json > /dev/null
    python - ${L}.json ${L}_by_chip_strength.json "$L" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); cs = json.load(open(sys.argv[2])); print(f"\n== {sys.argv[3]}")
for k, v in d.items():
    if k != "paired":
        print(f"  {k:22s} AUROC {v['auroc_site']:.3f}  false footprints at unbound {100*v['rate_unbound']:.1f}%  at bound {100*v['rate_bound']:.1f}%  (sites {v['n_bound']}+{v['n_unbound']})")
for k, v in d["paired"].items():
    if "raw" not in k:
        print(f"  {k:44s} diff {v['auroc_diff']:+.3f}  DeLong p={v['delong_p']:.3g}")
for g in ("weak ChIP", "strong"):
    print(f"  {g:10s} " + "  ".join(f"{t} {cs[g][t]:.3f}" for t in cs[g] if t not in ("n_bound", "mean_occ_bound", "raw")))
PY
  )
}
M=$HERE/motifs/MA0138.3.jaspar
one .                 ctcf.sites.bed ctcf_sites.bam "$TOP/chr19.fa" -  CTCF_GM12878 ctcf_fs_p05.bam ctcf_fh.bam       ctcf_ft_footprint.tsv
one K562_test         ctcf.sites.bed ctcf_sites.bam "$TOP/chr19.fa" -  CTCF_K562    ctcf_fs_p05.bam ctcf_sites_fh.bam ft_footprint.tsv
one REST_test/GM12878 tf.sites.bed   sites.bam      "$REST_REF"     "$M" REST_GM12878 sites_fs.bam  sites_fh.bam      ft_footprint.tsv
one REST_test/K562    tf.sites.bed   sites.bam      "$REST_REF"     "$M" REST_K562    sites_fs.bam  sites_fh.bam      ft_footprint.tsv
tar czf final_results.tar.gz CTCF_GM12878* K562_test/CTCF_K562* REST_test/GM12878/REST_GM12878* REST_test/K562/REST_K562* \
  REST_test/*/tf.sites.bed REST_test/*/times.txt REST_test/*/peaks_source.txt REST_test/*/sites_log.txt \
  $(ls hf_sites/*/hf6.log 2>/dev/null | head -3)
ls -lh final_results.tar.gz; echo "Send back: final_results.tar.gz"
