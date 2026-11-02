#!/usr/bin/env bash
# Same CTCF ChIP-seq and DNase-seq tests as for GM12878, on another cell line (default K562).
# K562 matters because FIRE's model was trained on GM12878 (DNase + CTCF labels) and fiberstate's
# default enzyme table was learned on GM12878, so K562 is held-out data for both.
#
# Run in ~/fiberseq with the fiberhmm env active (needs samtools, tabix, wget, aws or curl,
# fiberstate, fiberhmm-call, ft). Writes everything to ./<CELL>_test/ and a small
# <CELL>_results.tar.gz to send back.
set -euo pipefail
CELL=${CELL:-K562}
THREADS=${THREADS:-8}
N_PER_CLASS=${N_PER_CLASS:-250}
REGION=${REGION:-chr19:45000000-45500000}
# ENCODE, GRCh38, released: K562 CTCF ChIP-seq IDR peaks (ENCSR000EGM) and K562 DNase-seq peaks (ENCSR000EOT)
CTCF_PEAKS=${CTCF_PEAKS:-https://www.encodeproject.org/files/ENCFF660GHM/@@download/ENCFF660GHM.bed.gz}
DNASE_PEAKS=${DNASE_PEAKS:-https://www.encodeproject.org/files/ENCFF185XRG/@@download/ENCFF185XRG.bed.gz}
S3=https://s3.kopah.uw.edu/stergachis
PREFIX=public/FIRE/broadly-consented/$CELL
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HERE/.." && pwd)"
[ -f chr19.fa ] || { echo "run in the folder that holds chr19.fa"; exit 1; }
REF="$(pwd)/chr19.fa"
mkdir -p "${CELL}_test" && cd "${CELL}_test"

echo "== 1/8 find $CELL Fiber-seq files"
if command -v aws >/dev/null; then
  aws s3 ls --no-sign-request --endpoint-url https://s3.kopah.uw.edu "s3://stergachis/$PREFIX/" > listing.txt
else
  curl -s "$S3?list-type=2&prefix=$PREFIX/" | grep -oP '(?<=<Key>)[^<]+' > listing.txt
fi
CRAM_NAME=$(grep -oE '[^ /]+filtered\.cram$' listing.txt | head -1)
PEAK_NAME=$(grep -oE '[^ /]+-fire-[^ /]*-peaks\.bed\.gz$' listing.txt | head -1)
PILE_NAME=$(grep -oE '[^ /]+-fire-[^ /]*-pileup\.bed\.gz$' listing.txt | head -1)
echo "cram: $CRAM_NAME | peaks: $PEAK_NAME | pileup: $PILE_NAME"
[ -n "$CRAM_NAME" ] || { echo "no *filtered.cram under $PREFIX; listing saved in listing.txt"; exit 1; }
CRAM=$S3/$PREFIX/$CRAM_NAME

echo "== 2/8 ENCODE references"
[ -s ctcf_peaks.bed.gz ] || wget -q -O ctcf_peaks.bed.gz "$CTCF_PEAKS"
[ -s dnase_peaks.bed.gz ] || wget -q -O dnase_peaks.bed.gz "$DNASE_PEAKS"
zcat dnase_peaks.bed.gz | awk '$1=="chr19"' > dnase_chr19.bed

echo "== 3/8 CTCF motif sites on chr19, bound / unbound in $CELL"
python "$HERE/select_sites.py" "$REF" chr19 ctcf_peaks.bed.gz ctcf "$N_PER_CLASS" 1

echo "== 4/8 stream molecules (CTCF sites, then the $REGION accessibility region)"
if [ ! -s ctcf_sites.bam.bai ]; then
  samtools view -u -M -L ctcf.regions.bed -T "$REF" --keep-tag MM,ML "$CRAM" | samtools sort -o ctcf_sites.bam -
  samtools index ctcf_sites.bam
fi
if [ ! -s region.bam.bai ]; then
  samtools view -b -T "$REF" --keep-tag MM,ML -o region.bam "$CRAM" "$REGION"
  samtools index region.bam
fi
[ -s fire_peaks_region.bed ] || tabix -h "$S3/$PREFIX/$PEAK_NAME" "$REGION" > fire_peaks_region.bed
[ -s fire_pileup_region.bed ] || tabix -h "$S3/$PREFIX/$PILE_NAME" "$REGION" > fire_pileup_region.bed
echo "molecules: sites $(samtools view -c -F 0x900 ctcf_sites.bam), region $(samtools view -c -F 0x900 region.bam)"

: > times.txt
echo "== 5/8 fiberstate"
for b in region ctcf_sites; do
  if [ ! -s ${b}_fs.bam.bai ]; then
    s=$(date +%s); fiberstate call $b.bam -o ${b}_fs.bed --out-bam ${b}_fs.bam -t "$THREADS" --timing
    echo "fiberstate_${b}_seconds $(( $(date +%s) - s ))" | tee -a times.txt; samtools index ${b}_fs.bam
  fi
done

echo "== 6/8 FiberHMM and fibertools"
for b in region ctcf_sites; do
  if [ ! -s ${b}_fh.bam.bai ]; then
    s=$(date +%s); fiberhmm-call -i $b.bam -o ${b}_fh.bam --enzyme hia5 --seq pacbio -c "$THREADS" > ${b}_fh.log 2>&1
    echo "fiberhmm_${b}_seconds $(( $(date +%s) - s ))" | tee -a times.txt; samtools index ${b}_fh.bam
  fi
  if [ ! -s ${b}_ft.bam.bai ]; then
    s=$(date +%s); ft add-nucleosomes -t "$THREADS" $b.bam ${b}_ft.bam
    echo "fibertools_${b}_seconds $(( $(date +%s) - s ))" | tee -a times.txt; samtools index ${b}_ft.bam
  fi
done
awk 'BEGIN{OFS="\t"}{print $1,$2-10,$3+10,$4,$5,$6}' ctcf.sites.bed > ft_sites.bed
printf 'modules:\n  - [0, 10]\n  - [10, 25]\n  - [25, 35]\n' > ft_modules.yaml
ft footprint --bed ft_sites.bed --yaml ft_modules.yaml -o ft_footprint.tsv ctcf_sites_ft.bam
echo "versions: ft $(ft --version 2>&1 | head -1); fiberhmm $(python -c 'import fiberhmm; print(fiberhmm.__version__)' 2>/dev/null); fiberstate $(python -c 'import fiberstate; print(fiberstate.__version__)')" >> times.txt

echo "== 7/8 score"
python "$HERE/score_sites.py" ctcf.sites.bed ctcf_sites.bam ctcf_chip_results.json \
  fiberstate=ctcf_sites_fs.bam FiberHMM=ctcf_sites_fh.bam fibertools_footprint=ftfootprint:ft_footprint.tsv > /dev/null
C=${REGION%%:*}; R=${REGION#*:}; LO=${R%-*}; HI=${R#*-}
for t in fiberstate:region_fs.bam fibertools:region_ft.bam FiberHMM:region_fh.bam; do
  python -I "$BENCH/score_annotated_bam.py" "${t#*:}" fire_peaks_region.bed "$REF" "$C" "$LO" "$HI" "acc_${t%%:*}.json" "${t%%:*}" dnase_chr19.bed > /dev/null
done
python -I "$BENCH/fire_vs_dnase.py" fire_pileup_region.bed dnase_chr19.bed "$C" "$LO" "$HI" acc_FIRE.json > /dev/null

echo "== 8/8 does fiberstate's GM12878 enzyme table transfer to $CELL?"
fiberstate learn-context region.bam -o context_${CELL}.tsv -t "$THREADS"
python - "$CELL" <<'PY'
import sys, numpy as np, os, fiberstate
from fiberstate.context import ContextTable
cell = sys.argv[1]
a = ContextTable.load(os.path.join(os.path.dirname(fiberstate.__file__), "data", "hia5_k5_gm12878.tsv")).weights
b = ContextTable.load(f"context_{cell}.tsv").weights
print(f"enzyme table GM12878 vs {cell}: r(log weights) = {np.corrcoef(np.log(a), np.log(b))[0,1]:.3f}")
PY

python - <<'PY'
import json
d = json.load(open("ctcf_chip_results.json"))
print("\nCTCF ChIP-seq (AUROC, false-footprint rate at unbound sites):")
for k, v in d.items():
    if k != "paired":
        print(f"  {k:22s} {v['auroc_site']:.3f}   unbound {v['rate_unbound']:.3f}")
for k, v in d["paired"].items():
    if "raw" not in k:
        print(f"  {k:40s} diff {v['auroc_diff']:+.3f} DeLong p={v['delong_p']:.3g}")
print("\nAccessibility vs DNase-seq (AUROC):")
for t in ("fiberstate", "fibertools", "FiberHMM", "FIRE"):
    a = json.load(open(f"acc_{t}.json"))
    extra = f"  (open stretches >=100 bp {a['dnase_auroc_long_patches']:.3f})" if "dnase_auroc_long_patches" in a else ""
    print(f"  {t:11s} {a['dnase_auroc']:.3f}{extra}")
PY
cd ..
tar czf "${CELL}_results.tar.gz" "${CELL}_test"/*.json "${CELL}_test"/*.tsv "${CELL}_test"/times.txt \
  "${CELL}_test"/ctcf.sites.bed "${CELL}_test"/listing.txt "${CELL}_test"/*_fh.log
ls -lh "${CELL}_results.tar.gz"; echo "Send back: ${CELL}_results.tar.gz"
