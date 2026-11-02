#!/usr/bin/env bash
# Genome-wide CTCF footprint benchmark on GM12878: every strong CTCF motif on the chosen
# chromosomes, labelled by ENCODE CTCF ChIP-seq, scored for fiberstate, FiberHMM and fibertools.
#
# Works in 20 Mb windows: stream the window's molecules from the public FIRE CRAM, run the three
# tools, score, keep only small tables, delete the BAMs. Resumable: rerun the same command and
# finished windows are skipped. Disk stays around 2-3 GB.
#
# Defaults: autosomes chr1-chr22, molecules subsampled to 30% (~30x; the data are ~100x).
# Roughly 10 minutes per window with 8 threads -> about a day for all autosomes.
# Try one small chromosome first:   CHROMS=chr22 bash .../run_genome_ctcf.sh
# Run in ~/fiberseq with the fiberhmm env active.
set -euo pipefail
THREADS=${THREADS:-8}
CHROMS=${CHROMS:-chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22}
FRACTION=${FRACTION:-0.3}
WIN=${WIN:-20000000}
CRAM=${CRAM:-https://s3.kopah.uw.edu/stergachis/public/FIRE/broadly-consented/GM12878/GM12878-fire-v0.1-filtered.cram}
PEAKS=${PEAKS:-ENCFF797SDL.bed.gz}     # GM12878 CTCF IDR peaks (downloaded by run_ctcf_chip_test.sh)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP="$(pwd)"
G="$TOP/genome_ctcf"
mkdir -p "$G/sites" "$G/out" "$G/tmp" refs
[ -s "$PEAKS" ] || wget -q -O "$PEAKS" https://www.encodeproject.org/files/ENCFF797SDL/@@download/ENCFF797SDL.bed.gz
printf 'modules:\n  - [0, 10]\n  - [10, 25]\n  - [25, 35]\n' > "$G/ft_modules.yaml"
SEED_FRAC="7${FRACTION#0}"              # samtools -s SEED.FRACTION, seed 7
echo "fiberstate $(python -c 'import fiberstate; print(fiberstate.__version__)'), FiberHMM $(python -c 'import fiberhmm; print(fiberhmm.__version__)' 2>/dev/null), $(ft --version 2>&1 | head -1); subsample $FRACTION" | tee "$G/versions.txt"

for c in $CHROMS; do
  REF="$TOP/refs/$c.fa"
  if [ ! -s "$REF.fai" ]; then
    if [ "$c" = chr19 ] && [ -s "$TOP/chr19.fa" ]; then cp "$TOP/chr19.fa" "$REF"
    else
      [ -s "refs/$c.fa.gz" ] || wget -q -O "refs/$c.fa.gz" "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/$c.fa.gz"
      zcat "refs/$c.fa.gz" > "$REF"
    fi
    samtools faidx "$REF"
  fi
  SITES="$G/sites/$c.sites.bed"
  [ -s "$SITES" ] || python "$HERE/genome_ctcf.py" sites "$REF" "$c" "$PEAKS" "$SITES"
  LEN=$(cut -f2 "$REF.fai" | head -1)
  for (( s=0; s<LEN; s+=WIN )); do
    e=$(( s + WIN < LEN ? s + WIN : LEN ))
    W="${c}_${s}"
    [ -s "$G/out/$W.hist.json" ] && continue
    t0=$(date +%s)
    T="$G/tmp"; rm -rf "${T:?}"/*
    ok=0
    for attempt in 1 2 3 4 5 6; do      # the public server occasionally drops long downloads
      if samtools view -b -s "$SEED_FRAC" -T "$REF" --keep-tag MM,ML -o "$T/win.bam" "$CRAM" "$c:$((s + 1))-$e" \
         && samtools quickcheck "$T/win.bam"; then ok=1; break; fi
      echo "$W: download failed (attempt $attempt), retrying in $(( attempt * 60 )) s" >&2
      rm -f "$T/win.bam"; sleep $(( attempt * 60 ))
    done
    [ $ok = 1 ] || { echo "$W: download failed 6 times; rerun the same command later to resume"; exit 1; }
    samtools index "$T/win.bam"
    N=$(samtools view -c -F 0x900 "$T/win.bam")
    if [ "$N" -lt 50 ]; then        # centromere, gap or end of chromosome
      echo "{\"window\":\"$W\",\"chrom\":\"$c\",\"start\":$s,\"end\":$e,\"molecules\":0,\"calls\":{\"fiberstate\":0,\"FiberHMM\":0,\"fibertools\":0},\"hist\":{\"fiberstate\":{\"bound\":$(python -c 'print([0]*256)'),\"unbound\":$(python -c 'print([0]*256)')},\"FiberHMM\":{\"bound\":$(python -c 'print([0]*256)'),\"unbound\":$(python -c 'print([0]*256)')},\"fibertools\":{\"bound\":$(python -c 'print([0]*256)'),\"unbound\":$(python -c 'print([0]*256)')}}}" > "$G/out/$W.hist.json"
      echo "$W: $N molecules, skipped"; continue
    fi
    fiberstate call "$T/win.bam" -o /dev/null --out-bam "$T/fs.bam" --posterior-tags -t "$THREADS" 2>> "$G/fiberstate.log"
    samtools index "$T/fs.bam"
    fiberhmm-call -i "$T/win.bam" -o "$T/fh.bam" --enzyme hia5 --seq pacbio -c "$THREADS" >> "$G/fiberhmm.log" 2>&1
    samtools index "$T/fh.bam"
    ft add-nucleosomes -t "$THREADS" "$T/win.bam" "$T/ft.bam"
    samtools index "$T/ft.bam"
    awk -F'\t' -v lo=$(( s - 60000 )) -v hi=$(( e + 60000 )) 'BEGIN{OFS="\t"} $2>=lo && $3<=hi {print $1,$2-10,$3+10,$4,$5,$6}' "$SITES" > "$T/ft_sites.bed"
    ft footprint --bed "$T/ft_sites.bed" --yaml "$G/ft_modules.yaml" -o "$T/ft.tsv" "$T/ft.bam"
    python "$HERE/genome_ctcf.py" scan "$SITES" "$W" "$s" "$e" "$T/fs.bam" "$T/fh.bam" "$T/ft.tsv" "$G/out/$W"
    echo "$W molecules $N seconds $(( $(date +%s) - t0 ))" | tee -a "$G/times.txt"
    rm -rf "${T:?}"/*
  done
done
python "$HERE/genome_ctcf.py" summarize "$G/out" "$G/summary.json" | tee "$G/summary.txt"
tar czf genome_ctcf_results.tar.gz -C "$G" summary.json summary.txt times.txt versions.txt out
ls -lh genome_ctcf_results.tar.gz; echo "Send back: genome_ctcf_results.tar.gz"
