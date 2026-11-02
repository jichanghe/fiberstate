#!/usr/bin/env bash
# Confirmation test on a second transcription factor, with fiberstate 0.3 defaults and NO retuning.
# Default TF = REST (JASPAR MA0138.3, 20 bp), in GM12878 and K562. Same design as the CTCF test:
# strong motif matches, bound = within 100 bp of an ENCODE IDR peak summit, unbound = no peak
# within 2 kb, unbound drawn to match the bound motif scores, sites 2 kb apart.
# REST has far fewer motif matches than CTCF, so sites come from several chromosomes.
#
# Run in ~/fiberseq with the fiberhmm env active (needs samtools, wget, fiberstate, fiberhmm-call, ft).
# Optional: HiddenFoot is included when ~/fiberseq/HiddenFoot/hiddenfoot exists
# (built by run_hiddenfoot_ctcf.sh).
#   TF=REST CELLS="GM12878 K562" CHROMS=chr1,chr2,... bash fiberstate/bench/ctcf_chip/run_tf_test.sh
set -euo pipefail
TF=${TF:-REST}
MOTIF_ID=${MOTIF_ID:-MA0138.3}
CELLS=${CELLS:-GM12878 K562}
CHROMS=${CHROMS:-chr1,chr2,chr6,chr11,chr12,chr17,chr19,chr20}
N_PER_CLASS=${N_PER_CLASS:-250}
THREADS=${THREADS:-8}
S3=https://s3.kopah.uw.edu/stergachis
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HERE/.." && pwd)"
TOP="$(pwd)"
HF="$TOP/HiddenFoot/hiddenfoot"
mkdir -p refs "${TF}_test"

echo "== 1/7 reference chromosomes ($CHROMS) and the $TF motif"
REF="$TOP/refs/hg38_${CHROMS//,/_}.fa"
if [ ! -s "$REF.fai" ]; then
  : > "$REF.tmp"
  for c in ${CHROMS//,/ }; do
    if [ "$c" = chr19 ] && [ -s chr19.fa ]; then cat chr19.fa >> "$REF.tmp"; continue; fi
    [ -s refs/$c.fa.gz ] || wget -q -O refs/$c.fa.gz "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/$c.fa.gz"
    zcat refs/$c.fa.gz >> "$REF.tmp"
  done
  mv "$REF.tmp" "$REF"; samtools faidx "$REF"
fi
MOTIF="$HERE/motifs/$MOTIF_ID.jaspar"
[ -s "$MOTIF" ] || { mkdir -p "$HERE/motifs"; wget -q -O "$MOTIF" "https://jaspar.elixir.no/api/v1/matrix/$MOTIF_ID/?format=jaspar"; }
L=$(python -c "import sys; sys.path.insert(0,'$HERE'); from select_sites import read_jaspar; print(read_jaspar('$MOTIF').shape[1])")
echo "motif $MOTIF_ID, $L bp"

for CELL in $CELLS; do
  D="$TOP/${TF}_test/$CELL"; mkdir -p "$D"; cd "$D"
  echo; echo "######## $CELL"
  echo "== 2/7 $CELL: Fiber-seq CRAM and ENCODE $TF ChIP-seq peaks"
  PREFIX=public/FIRE/broadly-consented/$CELL
  if [ ! -s listing.txt ]; then
    if command -v aws >/dev/null; then
      aws s3 ls --no-sign-request --endpoint-url https://s3.kopah.uw.edu "s3://stergachis/$PREFIX/" > listing.txt
    else
      curl -s "$S3?list-type=2&prefix=$PREFIX/" | grep -oP '(?<=<Key>)[^<]+' > listing.txt
    fi
  fi
  CRAM_NAME=$(grep -oE '[^ /]+filtered\.cram$' listing.txt | head -1)
  [ -n "$CRAM_NAME" ] || { echo "no *filtered.cram under $PREFIX (see listing.txt)"; exit 1; }
  CRAM=$S3/$PREFIX/$CRAM_NAME
  if [ ! -s peaks.bed.gz ]; then
    VAR="PEAKS_$CELL"; URL=${!VAR:-}
    if [ -z "$URL" ]; then
      python "$HERE/encode_peaks.py" "$TF" "$CELL" | tee peaks_source.txt
      URL=$(head -1 peaks_source.txt | cut -f2)
    else
      echo "user-supplied $URL" > peaks_source.txt
    fi
    wget -q -O peaks.bed.gz "$URL"
  fi
  echo "cram: $CRAM_NAME | peaks: $(head -1 peaks_source.txt | cut -f1,3,4)"

  echo "== 3/7 $CELL: $TF motif sites, bound / unbound"
  [ -s tf.sites.bed ] || python "$HERE/select_sites.py" "$REF" "$CHROMS" peaks.bed.gz tf "$N_PER_CLASS" 1 "$MOTIF" | tee sites_log.txt

  echo "== 4/7 $CELL: stream molecules at the sites"
  if [ ! -s sites.bam.bai ]; then
    samtools view -u -M -L tf.regions.bed -T "$REF" --keep-tag MM,ML "$CRAM" | samtools sort -o sites.bam -
    samtools index sites.bam
  fi
  echo "molecules: $(samtools view -c -F 0x900 sites.bam)"

  echo "== 5/7 $CELL: run the tools (fiberstate defaults, no retuning)"
  : > times.txt
  if [ ! -s sites_fs.bam.bai ]; then
    s=$(date +%s); fiberstate call sites.bam -o sites_fs.bed --out-bam sites_fs.bam -t "$THREADS" --timing
    echo "fiberstate_seconds $(( $(date +%s) - s ))" | tee -a times.txt; samtools index sites_fs.bam
  fi
  if [ ! -s sites_fh.bam.bai ]; then
    s=$(date +%s); fiberhmm-call -i sites.bam -o sites_fh.bam --enzyme hia5 --seq pacbio -c "$THREADS" > fh.log 2>&1
    echo "fiberhmm_seconds $(( $(date +%s) - s ))" | tee -a times.txt; samtools index sites_fh.bam
  fi
  if [ ! -s sites_ft.bam.bai ]; then
    s=$(date +%s); ft add-nucleosomes -t "$THREADS" sites.bam sites_ft.bam
    echo "fibertools_seconds $(( $(date +%s) - s ))" | tee -a times.txt; samtools index sites_ft.bam
  fi
  awk 'BEGIN{OFS="\t"}{print $1,$2-10,$3+10,$4,$5,$6}' tf.sites.bed > ft_sites.bed
  printf 'modules:\n  - [0, 10]\n  - [10, %d]\n  - [%d, %d]\n' $((10 + L)) $((10 + L)) $((20 + L)) > ft_modules.yaml
  [ -s ft_footprint.tsv ] || ft footprint --bed ft_sites.bed --yaml ft_modules.yaml -o ft_footprint.tsv sites_ft.bam
  EXTRA=()
  if [ -x "$HF" ]; then
    [ -s hf_sites/sites.txt ] || python "$BENCH/hiddenfoot_adapter.py" prepare tf.sites.bed sites.bam "$REF" hf_sites "$MOTIF"
    s=$(date +%s)
    grep -v '^$' hf_sites/sites.txt | xargs -P "$THREADS" -I{} sh -c '
      d=hf_sites/{}; [ -s $d/hf6.hfprofile.nucleosome.txt ] && exit 0
      "'"$HF"'" run -s $d/seq.txt -m $d/met.txt -w $d/motif.txt -o $d/hf6 -padlen 150 -nuclen 147 -cutoffwms 1e-6 -numthreads 1 > $d/hf6.log 2>&1 \
        || echo "HiddenFoot failed at {}"'
    echo "hiddenfoot_seconds $(( $(date +%s) - s ))" | tee -a times.txt
    python "$BENCH/hiddenfoot_adapter.py" collect tf.sites.bed hf_sites hf_states.tsv 0.5 hf6
    EXTRA=(HiddenFoot=states:hf_states.tsv)
  fi
  echo "versions: ft $(ft --version 2>&1 | head -1); fiberhmm $(python -c 'import fiberhmm; print(fiberhmm.__version__)' 2>/dev/null); fiberstate $(python -c 'import fiberstate; print(fiberstate.__version__)')" >> times.txt

  echo "== 6/7 $CELL: score"
  python "$HERE/score_sites.py" tf.sites.bed sites.bam ${TF}_${CELL}.json \
    fiberstate=sites_fs.bam FiberHMM=sites_fh.bam fibertools_footprint=ftfootprint:ft_footprint.tsv "${EXTRA[@]}" > /dev/null
  python "$HERE/by_chip_strength.py" ${TF}_${CELL}.per_site.tsv ${TF}_${CELL}_by_chip_strength.json > /dev/null
  python - ${TF}_${CELL}.json "$TF $CELL" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); print(f"\n== {sys.argv[2]}")
for k, v in d.items():
    if k != "paired":
        print(f"  {k:22s} AUROC {v['auroc_site']:.3f}  false footprints at unbound {100*v['rate_unbound']:.1f}%  at bound {100*v['rate_bound']:.1f}%  (n bound {v.get('n_bound', '?')}, unbound {v.get('n_unbound', '?')})")
for k, v in d["paired"].items():
    if "raw" not in k:
        print(f"  {k:44s} diff {v['auroc_diff']:+.3f}  DeLong p={v['delong_p']:.3g}")
PY
  cd "$TOP"
done

echo "== 7/7 pack results"
tar czf "${TF}_results.tar.gz" "${TF}_test"/*/*.json "${TF}_test"/*/*.per_site.tsv "${TF}_test"/*/tf.sites.bed \
  "${TF}_test"/*/times.txt "${TF}_test"/*/peaks_source.txt "${TF}_test"/*/sites_log.txt
ls -lh "${TF}_results.tar.gz"; echo "Send back: ${TF}_results.tar.gz"
