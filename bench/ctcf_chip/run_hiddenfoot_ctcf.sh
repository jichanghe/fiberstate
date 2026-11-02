#!/usr/bin/env bash
# Add HiddenFoot (MolinaLab-IGBMC, per-molecule thermodynamic TF + nucleosome model) to the
# CTCF ChIP-seq test, on the same 462 sites in GM12878 (~/fiberseq) and K562 (~/fiberseq/K562_test).
# Re-scores every tool together (fiberstate 0.3 default = ctcf_fs_p05.bam from rescore_posterior.sh)
# and splits the result by ChIP-seq strength, so all CTCF numbers come from fiberstate 0.3.
#
# Run in ~/fiberseq with the fiberhmm env active, after rescore_posterior.sh. Needs git, g++ with OpenMP.
# HiddenFoot fits its model separately for every site; expect roughly 0.5-2 h with 8 threads.
set -euo pipefail
THREADS=${THREADS:-8}
HF_COMMIT=${HF_COMMIT:-d1499e812bee6ae026ef4d1f5e215b7c30ae99bb}   # main, 2025-05-18
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HERE/.." && pwd)"
[ -f chr19.fa ] || { echo "run in ~/fiberseq (the folder that holds chr19.fa)"; exit 1; }
REF="$(pwd)/chr19.fa"

echo "== 1/4 build HiddenFoot"
if [ ! -x HiddenFoot/hiddenfoot ]; then
  [ -d HiddenFoot ] || git clone -q https://github.com/MolinaLab-IGBMC/HiddenFoot.git
  ( cd HiddenFoot && git checkout -q "$HF_COMMIT" && g++ -O3 -fopenmp source/hiddenfoot.cpp -o hiddenfoot )
fi
HF="$(pwd)/HiddenFoot/hiddenfoot"
"$HF" 2>&1 | head -3 || true

run_cell () {   # dir  label  fh_bam  ft_tsv
  local D=$1 L=$2 FH=$3 FT=$4
  ( cd "$D"
    echo "== $L: prepare per-site inputs"
    [ -s hf_sites/sites.txt ] || python "$BENCH/hiddenfoot_adapter.py" prepare ctcf.sites.bed ctcf_sites.bam "$REF" hf_sites
    echo "== $L: HiddenFoot on $(wc -l < hf_sites/sites.txt) sites, $THREADS at a time"
    s=$(date +%s)
    grep -v '^$' hf_sites/sites.txt | xargs -P "$THREADS" -I{} sh -c '
      d=hf_sites/{}; [ -s $d/hf6.hfprofile.nucleosome.txt ] && exit 0
      "'"$HF"'" run -s $d/seq.txt -m $d/met.txt -w $d/motif.txt -o $d/hf6 -padlen 150 -nuclen 147 -cutoffwms 1e-6 -numthreads 1 > $d/hf6.log 2>&1 \
        || echo "HiddenFoot failed at {} (see $d/hf.log)"'
    echo "hiddenfoot_${L}_seconds $(( $(date +%s) - s ))" | tee hf_times.txt
    python "$BENCH/hiddenfoot_adapter.py" collect ctcf.sites.bed hf_sites hf_states.tsv 0.5 hf6
    echo "== $L: score all tools"
    python "$HERE/score_sites.py" ctcf.sites.bed ctcf_sites.bam ${L}_with_hiddenfoot.json \
      fiberstate=ctcf_fs_p05.bam FiberHMM=$FH fibertools_footprint=ftfootprint:$FT HiddenFoot=states:hf_states.tsv > /dev/null
    python "$HERE/by_chip_strength.py" ${L}_with_hiddenfoot.per_site.tsv ${L}_by_chip_strength.json > /dev/null
    python - ${L}_with_hiddenfoot.json ${L}_by_chip_strength.json "$L" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); cs = json.load(open(sys.argv[2])); print(f"\n== {sys.argv[3]}")
for k, v in d.items():
    if k != "paired":
        print(f"  {k:22s} AUROC {v['auroc_site']:.3f}  false footprints at unbound {100*v['rate_unbound']:.1f}%  at bound {100*v['rate_bound']:.1f}%")
for k, v in d["paired"].items():
    if "HiddenFoot" in k or k.startswith("fiberstate vs"):
        print(f"  {k:44s} diff {v['auroc_diff']:+.3f}  DeLong p={v['delong_p']:.3g}")
for g in ("weak ChIP", "strong"):
    print(f"  {g:10s} " + "  ".join(f"{t} {cs[g][t]:.3f}" for t in cs[g] if t not in ("n_bound", "mean_occ_bound", "raw")))
PY
  )
}
run_cell . GM12878 ctcf_fh.bam ctcf_ft_footprint.tsv
run_cell K562_test K562 ctcf_sites_fh.bam ft_footprint.tsv
tar czf hiddenfoot_results.tar.gz GM12878_with_hiddenfoot.json GM12878_with_hiddenfoot.per_site.tsv GM12878_by_chip_strength.json \
  hf_times.txt K562_test/K562_with_hiddenfoot.json K562_test/K562_with_hiddenfoot.per_site.tsv K562_test/K562_by_chip_strength.json \
  K562_test/hf_times.txt hf_states.tsv K562_test/hf_states.tsv
ls -lh hiddenfoot_results.tar.gz; echo "Send back: hiddenfoot_results.tar.gz"
