#!/usr/bin/env bash
# CTCF ChIP-seq test for Fiber-seq footprint callers (GM12878, chr19).
#
# Run from the folder that holds chr19.fa (and its .fai), with the `fiberhmm` conda env active
# and fiberstate installed (pip install ./fiberstate). Uses about 1-2 GB of download and
# ~1 GB of disk. Writes ctcf_chip_results.json and ctcf_times.txt to send back.
set -euo pipefail

N_PER_CLASS=${N_PER_CLASS:-250}
THREADS=${THREADS:-4}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRAM=https://s3.kopah.uw.edu/stergachis/public/FIRE/broadly-consented/GM12878/GM12878-fire-v0.1-filtered.cram
PEAKS=https://www.encodeproject.org/files/ENCFF797SDL/@@download/ENCFF797SDL.bed.gz   # GM12878 CTCF, ENCSR000AKB, GRCh38 IDR peaks

if ! command -v ft >/dev/null; then
  echo "== installing fibertools (ft) from bioconda into the current env"
  conda install -y -c conda-forge -c bioconda fibertools-rs >/dev/null || { echo "could not install fibertools-rs; see https://github.com/fiberseq/fibertools-rs"; exit 1; }
fi
for tool in samtools python fiberstate fiberhmm-call ft; do
  command -v "$tool" >/dev/null || { echo "missing: $tool (activate the fiberhmm env and install fiberstate)"; exit 1; }
done
[ -f chr19.fa ] || { echo "chr19.fa not found in $(pwd)"; exit 1; }
[ -f chr19.fa.fai ] || samtools faidx chr19.fa

echo "== 1/6 CTCF ChIP-seq peaks (ENCODE ENCFF797SDL)"
[ -s ENCFF797SDL.bed.gz ] || wget -q -O ENCFF797SDL.bed.gz "$PEAKS"

echo "== 2/6 scan chr19 for CTCF motifs and pick bound / unbound sites"
python "$HERE/select_sites.py" chr19.fa chr19 ENCFF797SDL.bed.gz ctcf "$N_PER_CLASS" 1

echo "== 3/6 stream Fiber-seq molecules over those sites (this is the slow, network-bound step)"
if [ ! -s ctcf_sites.bam ]; then
  samtools view -u -M -L ctcf.regions.bed -T chr19.fa --keep-tag MM,ML "$CRAM" \
    | samtools sort -o ctcf_sites.bam -
  samtools index ctcf_sites.bam
fi
echo "molecules: $(samtools view -c -F 0x900 ctcf_sites.bam)"

touch ctcf_times.txt
echo "== 4/6 fiberstate"
if [ ! -s ctcf_fs.bam.bai ]; then
  start=$(date +%s)
  fiberstate call ctcf_sites.bam -o ctcf_fs.bed --out-bam ctcf_fs.bam -t "$THREADS" --timing
  echo "fiberstate_seconds $(( $(date +%s) - start ))" | tee -a ctcf_times.txt
  samtools index ctcf_fs.bam
else echo "(reusing ctcf_fs.bam)"; fi

echo "== 5/6 FiberHMM and fibertools"
if [ ! -s ctcf_fh.bam.bai ]; then
  start=$(date +%s)
  fiberhmm-call -i ctcf_sites.bam -o ctcf_fh.bam --enzyme hia5 --seq pacbio -c "$THREADS"
  echo "fiberhmm_seconds $(( $(date +%s) - start ))" | tee -a ctcf_times.txt
  samtools index ctcf_fh.bam
else echo "(reusing ctcf_fh.bam)"; fi
# fibertools footprinting: needs its own nucleosome/MSP calls first; window = motif +/- 10 bp
start=$(date +%s)
ft add-nucleosomes -t "$THREADS" ctcf_sites.bam ctcf_ft.bam
samtools index ctcf_ft.bam
awk 'BEGIN{OFS="\t"}{print $1,$2-10,$3+10,$4,$5,$6}' ctcf.sites.bed > ctcf_ft_sites.bed
printf 'modules:\n  - [0, 10]\n  - [10, 25]\n  - [25, 35]\n' > ctcf_ft_modules.yaml
ft footprint --bed ctcf_ft_sites.bed --yaml ctcf_ft_modules.yaml -o ctcf_ft_footprint.tsv ctcf_ft.bam
echo "fibertools_seconds $(( $(date +%s) - start ))" | tee -a ctcf_times.txt
echo "fibertools_version $(ft --version)" >> ctcf_times.txt
echo "molecules $(samtools view -c -F 0x900 ctcf_sites.bam) threads $THREADS" >> ctcf_times.txt

echo "== 6/6 score against ChIP-seq"
python "$HERE/score_sites.py" ctcf.sites.bed ctcf_sites.bam ctcf_chip_results.json \
  fiberstate=ctcf_fs.bam FiberHMM=ctcf_fh.bam fibertools_footprint=ftfootprint:ctcf_ft_footprint.tsv > /dev/null
python - <<'PY'
import json
d = json.load(open("ctcf_chip_results.json"))
for k, v in d.items():
    if k == "paired":
        continue
    extra = "  vs raw %+.3f" % v["auroc_minus_raw"] if "auroc_minus_raw" in v else ""
    print(f"{k:22s} sites {v['n_sites']:4d}  AUROC {v['auroc_site']:.3f} "
          f"[{v['auroc_site_ci95'][0]:.3f}-{v['auroc_site_ci95'][1]:.3f}]  bound/unbound footprint ratio {v['ratio']:.2f}{extra}")
print("\npaired (A minus B):")
for k, v in d["paired"].items():
    print(f"  {k:40s} AUROC diff {v['auroc_diff']:+.3f}  DeLong p={v['delong_p']:.3g}  "
          f"false-footprint rate at unbound sites {v['unbound_mean_occ'][0]:.3f} vs {v['unbound_mean_occ'][1]:.3f}")
PY
echo "Done. Send back: ctcf_chip_results.json ctcf_chip_results.per_site.tsv ctcf_times.txt"
