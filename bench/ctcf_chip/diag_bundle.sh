#!/usr/bin/env bash
# Extract a small diagnostic bundle for CTCF sites where the tools disagree (run in ~/fiberseq,
# after run_ctcf_chip_test.sh). Produces ctcf_diag_bundle.tar.gz (a few MB) to send back.
set -euo pipefail
SITES=${SITES:-"bound_51 bound_70 bound_75 bound_110 bound_31 bound_109 bound_82 bound_69 bound_218 bound_58 bound_202 bound_91 unbound_3 unbound_75 unbound_192"}
#   first four: strong ChIP, fiberstate finds ~nothing; next five: fibertools >> others;
#   then three strong sites all tools agree on (controls); last three: false-positive candidates.
mkdir -p ctcf_diag
# exact name match on column 4 ("bound_51" must not match "unbound_51")
awk -v L="$SITES" 'BEGIN{n=split(L,a," "); for(i=1;i<=n;i++) w[a[i]]=1} ($4 in w)' ctcf.sites.bed > ctcf_diag/sites.bed
awk 'BEGIN{OFS="\t"}{print $1,$2-1500,$3+1500,$4}' ctcf_diag/sites.bed > ctcf_diag/regions.bed
echo "sites found: $(wc -l < ctcf_diag/sites.bed) of $(echo $SITES | wc -w)"
for b in ctcf_sites ctcf_fs ctcf_fh ctcf_ft; do
  samtools view -b -M -L ctcf_diag/regions.bed -o ctcf_diag/$b.bam $b.bam
  samtools index ctcf_diag/$b.bam
done
awk -v L="$SITES" 'BEGIN{n=split(L,a," "); for(i=1;i<=n;i++) w[a[i]]=1} ($4 in w)' ctcf_ft_sites.bed > ctcf_diag/ft_sites.bed
cp ctcf_ft_modules.yaml ctcf_diag/ 2>/dev/null || true
awk -F'\t' 'NR==1 || $1 ~ /^(bound|unbound)_/' ctcf_chip_results.per_site.tsv > ctcf_diag/per_site.tsv
tar czf ctcf_diag_bundle.tar.gz ctcf_diag
ls -lh ctcf_diag_bundle.tar.gz
echo "Send back: ctcf_diag_bundle.tar.gz"
