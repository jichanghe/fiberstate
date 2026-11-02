#!/usr/bin/env bash
# 1) Calibrate fiberstate's footprint cut-off to false-call rates on the six ChIP-labelled site sets
#    (CTCF, REST, NRF1 x GM12878, K562), and pool them into the built-in calibration table.
# 2) Co-binding on real data: all CTCF, REST and NRF1 motif sites near the benchmark sites, per cell
#    line, tested pair by pair with and without fiberstate's accessibility conditioning.
#
# Run in ~/fiberseq with the fiberhmm env active, after the TF benchmarks (run_tf_test.sh for REST
# and NRF1, rerun_hiddenfoot.sh). About 1.5 hours with 8 threads: both parts run the model once
# per molecule (about 200,000 molecules each).
set -euo pipefail
THREADS=${THREADS:-8}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CT="$HERE/../ctcf_chip"
TOP="$(pwd)"
REF=$(ls "$TOP"/refs/hg38_*.fa | head -1)
mkdir -p calib cobind
: > calib/times.txt

echo "== 1/2 calibration"
cal () {   # dir label sites raw_bam
  local D=$1 L=$2 SITES=$3 BAM=$4
  [ -s calib/$L.tsv ] && return
  awk -F'\t' '$7=="unbound"' "$D/$SITES" > calib/$L.neg.bed
  awk -F'\t' '$7=="bound"'   "$D/$SITES" > calib/$L.pos.bed
  s=$(date +%s)
  fiberstate calibrate "$D/$BAM" --negatives calib/$L.neg.bed --positives calib/$L.pos.bed -o calib/$L.tsv -t "$THREADS" 2>&1 | sed "s/^/$L: /"
  echo "$L calibrate_seconds $(( $(date +%s) - s ))" >> calib/times.txt
}
cal .                 CTCF_GM12878 ctcf.sites.bed ctcf_sites.bam
cal K562_test         CTCF_K562    ctcf.sites.bed ctcf_sites.bam
cal REST_test/GM12878 REST_GM12878 tf.sites.bed   sites.bam
cal REST_test/K562    REST_K562    tf.sites.bed   sites.bam
cal NRF1_test/GM12878 NRF1_GM12878 tf.sites.bed   sites.bam
cal NRF1_test/K562    NRF1_K562    tf.sites.bed   sites.bam
fiberstate calibrate --merge calib/{CTCF,REST,NRF1}_{GM12878,K562}.tsv -o calib/pooled.tsv
python - <<'PY'
import glob
print("\nfalse-call rate (share of molecules at unbound sites called) by cut-off")
tabs = {p.split("/")[-1][:-4]: [l.split("\t") for l in open(p).read().splitlines()[1:]] for p in sorted(glob.glob("calib/*.tsv"))}
cut = [r[0] for r in tabs["pooled"]]
print("cut-off " + "".join(f"{k[:12]:>13s}" for k in tabs))
for i, c in enumerate(cut):
    if c in ("0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90"):
        print(f"{c:7s} " + "".join(f"{100 * float(t[i][1]):12.2f}%" for t in tabs.values()))
PY

echo; echo "== 2/2 co-binding (re-runs fiberstate with --posterior-tags: fp and np)"
post () {   # dir label raw_bam
  local D=$1 L=$2 BAM=$3
  if [ ! -s "$D/${L}_fs_post2.bam.bai" ]; then
    fiberstate call "$D/$BAM" -o /dev/null --out-bam "$D/${L}_fs_post2.bam" --posterior-tags -t "$THREADS" 2>&1 | tail -1
    samtools index "$D/${L}_fs_post2.bam"
  fi
}
post .                 CTCF_GM12878 ctcf_sites.bam
post REST_test/GM12878 REST_GM12878 sites.bam
post NRF1_test/GM12878 NRF1_GM12878 sites.bam
post K562_test         CTCF_K562    ctcf_sites.bam
post REST_test/K562    REST_K562    sites.bam
post NRF1_test/K562    NRF1_K562    sites.bam
for CELL in GM12878 K562; do
  if [ $CELL = GM12878 ]; then CD=.; CP=ENCFF797SDL.bed.gz; else CD=K562_test; CP=K562_test/ctcf_peaks.bed.gz; fi
  echo; echo "######## co-binding, $CELL"
  python "$HERE/cobind.py" cobind/$CELL "$REF" \
    --bams "$CD/CTCF_${CELL}_fs_post2.bam,REST_test/$CELL/REST_${CELL}_fs_post2.bam,NRF1_test/$CELL/NRF1_${CELL}_fs_post2.bam" \
    --windows "$CD/ctcf.sites.bed,REST_test/$CELL/tf.sites.bed,NRF1_test/$CELL/tf.sites.bed" \
    --tf "CTCF:-:$CP" --tf "REST:$CT/motifs/MA0138.3.jaspar:REST_test/$CELL/peaks.bed.gz" \
    --tf "NRF1:$CT/motifs/MA0506.1.jaspar:NRF1_test/$CELL/peaks.bed.gz"
done
tar czf calib_cobind_results.tar.gz calib/*.tsv calib/times.txt cobind/*.summary.json cobind/*.pairs.tsv
ls -lh calib_cobind_results.tar.gz; echo "Send back: calib_cobind_results.tar.gz"
