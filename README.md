# fiberstate

Per-molecule chromatin state calling for **Fiber-seq** and other m6A-footprinting long-read assays.

For every base of every DNA molecule, fiberstate gives the probability that it is **open
(linker)**, wrapped in a **nucleosome**, or covered by a **transcription-factor footprint**.
It then calls segments, and can report each footprint with an **estimated false-call rate**.
From many molecules it summarises per-site **occupancy** and tests whether two sites are
**bound together on the same molecule** more often than chance, after accounting for
shared accessibility.

It reads standard PacBio BAMs with `MM`/`ML` m6A tags and writes BED plus fibertools-compatible
BAM tags.

## Why use it

fiberstate is an explicit-duration hidden semi-Markov model (HSMM) that decodes nucleosomes
(~147 bp) and footprints (8–90 bp) jointly. Four modelling choices matter on real data:

- **G/C bases are "no evidence", not "unmarked".** Hia5 only methylates adenines, so GC-rich
  DNA has few places to mark whether it is open or not.
- **The enzyme's sequence preference is modelled.** A-tracts are marked 15–25× less often than
  average; the default table was learned from GM12878 Fiber-seq and transfers to K562 (r = 0.986).
- **Marking efficiency is re-estimated for every molecule** (a short EM with a prior), so a
  weakly marked molecule does not turn into long false nucleosomes.
- **Footprints are called from posteriors**, not only from the single best path, because a
  footprint's probability is spread over many possible lengths.

## How it compares

All numbers below come from public data: GM12878 and K562 Fiber-seq (FIRE release), labelled
with ENCODE ChIP-seq and DNase-seq. Tools are FiberHMM 3.0.2, fibertools-rs 0.13.1
(`ft footprint`, given the motif positions) and HiddenFoot (given the motif matrix). The full
report with every table is `results/report.html`.

**Genome-wide CTCF** (every strong CTCF motif on all 22 autosomes of GM12878: 920,000 sites
covered by at least 10 molecules, 6.75 million molecules). Held to 1% false calls, fiberstate
recovers 19.0% of molecules at ChIP-bound sites, against 10.9% for fibertools and 6.8% for
FiberHMM, and leads on all 22 chromosomes. By default it makes 0.71% false calls (fibertools
1.92%, FiberHMM 2.93%). Site AUROC is 0.834, 0.832 and 0.822.

**TF footprints at motif sites** (CTCF, REST and NRF1, each in GM12878 and K562; about 250 bound
and 250 unbound sites per set):

- **At a matched false-call rate** (every tool held to 1% of molecules called footprinted at
  unbound sites), fiberstate footprints more molecules at bound sites than FiberHMM and
  fibertools in **all six sets**, significantly in five (+0.8 to +11.7 percentage points; for
  example 17.1% vs 8.2% and 5.4% for CTCF in GM12878). It ties HiddenFoot, which is given the
  motif.
- **Its default calls are the most specific of the four tools.** It has the lowest false-call
  rate at unbound sites in all six sets (0.3–1.5%, against 1.3–4.9% for the others), and
  correspondingly calls fewer molecules at bound sites.
- **Ranking sites by footprinted share** (AUROC), the four tools are close. fiberstate is
  highest in five of six sets and lowest for NRF1 in K562.

**Accessibility**: against ENCODE DNase-seq peaks, the three per-molecule callers tie on open
stretches of at least 100 bp (AUROC 0.91–0.92). FiberHMM is slightly better when every open base
counts. FIRE, an accessibility caller rather than a per-molecule segmenter, is best (0.95).

**Speed**: about 26 ms of model time per 17 kb molecule on one core, against 1.5 ms for
fibertools. On an 8-thread laptop it calls 44,614 molecules in 9 minutes, against 1.4 minutes
for FiberHMM on 4 threads. A 30× human genome is roughly 40 core-hours.

## Install

```bash
pip install fiberstate-0.5.1-py3-none-any.whl      # or, from a source checkout: pip install .
```

It requires Python ≥ 3.9 and depends on numpy, numba, scipy, pysam and threadpoolctl.

## Use

```bash
# per-molecule calls: BED plus a BAM with tags (aligned or unaligned input; MM/ML m6A tags)
fiberstate call fibers.bam -o calls.bed --out-bam fibers.fs.bam -t 8

# footprints at a chosen false-call rate, each with its estimated false-call rate
fiberstate call fibers.bam -o calls.bed --out-bam fibers.fs.bam -t 8 --false-call-rate 0.01

# calibrate on your own data: motif sites with and without ChIP-seq peaks (aligned, indexed BAM)
fiberstate calibrate aligned.bam --negatives unbound_sites.bed --positives bound_sites.bed -o calib.tsv -t 8
fiberstate call fibers.bam -o calls.bed --false-call-rate 0.01 --calibration calib.tsv

# per-site occupancy and pairwise co-binding, conditioned on accessibility
fiberstate sites aligned.bam motifs.bed -o occupancy.tsv --pairs pairs.tsv --max-dist 1000

# learn the Hia5 sequence-context table from your own data (another enzyme, chemistry or m6A caller)
fiberstate learn-context aligned.bam -o context.tsv -t 8
fiberstate call fibers.bam -o calls.bed --context context.tsv
```

Useful options for `call`:

- `--foot-posterior T` sets the footprint cut-off directly. The default is 0.5; lower is more
  sensitive and less specific.
- `--posterior-tags` writes the per-base posteriors (see Output).
- `--em-iters 2` is about 1.5× faster, with less precise per-molecule rates.
- `--soft` uses ML probabilities instead of a 0.5 threshold.
- `--keep-inferred` keeps guessed nucleosome positions inside long unmarked stretches.

### What "false-call rate" means

It is the share of molecules at **ChIP-unbound motif sites** that fiberstate calls footprinted
at the motif. `fiberstate calibrate` measures it at every cut-off by running the model once per
molecule and applying the exact calling steps of `call`. With `--false-call-rate R`, `call`
uses the lowest cut-off whose rate is at most R. Each footprint is reported with the rate at a
cut-off equal to its own mean posterior.

The built-in table pools ChIP-labelled CTCF, REST and NRF1 sites in GM12878 and K562
(94,211 molecules at unbound sites). Its false-call rate is 2.5% at cut-off 0.1, 1.1% at 0.5 and
0.6% at 0.9, so `--false-call-rate 0.01` selects 0.6.

Rates differ up to sixfold between factors and cell types at the same cut-off: 0.26% for REST in
K562 against 1.51% for CTCF in K562, both at 0.5. Calibrate on your own data when you can.

### Output

BED columns, one row per segment:

```
chrom  start  end  type  read  confidence  q_start  q_end  read_p_access  read_p_protect  [est_false_call_rate]
```

- `type` is `nucleosome`, `footprint` or `protected_unresolved`. The last is a protected stretch
  with no m6A that would separate nucleosomes, so positions inside it are not identifiable.
- `confidence` is the mean posterior over the segment.
- The last column appears only with `--false-call-rate`.

BAM tags (positions in the stored SEQ orientation):

| tag | content |
|---|---|
| `ns`/`nl`, `nq` | nucleosome starts / lengths (fibertools layout), confidence × 255 |
| `as`/`al` | accessible stretches |
| `fs`/`fl`, `fq` | footprint starts / lengths, confidence × 255 |
| `us`/`ul` | unresolved protected stretches |
| `pa` | the molecule's estimated m6A rate in open DNA |
| `MA` | molecular-annotation string (`nuc.`, `tf.`, `unr.`), stored-SEQ frame (header says `coord=seq`) |
| `fe` | estimated false-call rate per footprint (with `--false-call-rate`) |
| `fp`, `np` | per-base footprint / nucleosome posterior × 255 (with `--posterior-tags`) |

### Co-binding

`fiberstate sites --pairs` builds, for each pair of sites, a 2×2 table of molecules (bound or not
at each site). By default it weights each molecule by the chance that both sites are accessible,
so sites that merely share an open region do not look coupled. It reports the log odds ratio, a
p-value from Fisher's exact test and an `informative` flag.

A pair is informative only when each site is bound, and unbound, on at least 3 molecules. A
rarely bound site gives a near-empty table, so uninformative pairs should not be interpreted.

On real data (5,612 informative pairs of CTCF, REST and NRF1 sites in GM12878), the test was
conservative: 0.4% of control pairs of two unbound sites reached p < 0.05. Conditioning on
accessibility removed about half of the raw association. No pair reached q < 0.1.

## How it works

Each molecule is a sequence of observations: m6A or not at each A/T, and no observation at G/C.
Observations are emitted by three segment types:

| segment | length | marking |
|---|---|---|
| linker | geometric, mean 45 bp | rate `p_access` × the base's sequence-context weight |
| nucleosome | mixture of ~147 bp and ~110 bp, 90–200 bp | low rate `p_protect` |
| footprint | ~28 bp, 8–90 bp | low rate `p_protect` |

Per molecule, a few EM rounds re-estimate the two rates with a Beta prior. Forward–backward then
gives per-base posteriors, and Viterbi gives the segmentation. Footprints are added inside linker
runs wherever the footprint posterior exceeds the cut-off. Protected calls that no m6A separates
are merged into one unresolved stretch.

The kernels are numba. They run as dot products over a contiguous history, which works because
a nucleosome lasts at least 90 bp. Posteriors match a straightforward reference implementation
to about 1e-14.

## Reproducing the benchmarks

The scripts are in `bench/`. The real-data runs stream reads from the public FIRE data and
ENCODE.

| script | what it does |
|---|---|
| `python -m bench.run --out results/` | simulated benchmark and co-binding simulation |
| `bench/ctcf_chip/run_ctcf_chip_test.sh`, `run_cell_line_test.sh` | CTCF ChIP-seq and DNase-seq tests, GM12878 and K562 |
| `bench/ctcf_chip/run_tf_test.sh` (`TF=REST`, `TF=NRF1 MOTIF_ID=MA0506.1`) | other factors, no retuning |
| `bench/ctcf_chip/run_hiddenfoot_ctcf.sh`, `rerun_hiddenfoot.sh` | HiddenFoot at the same sites |
| `bench/ctcf_chip/run_matched_fpr.sh` | matched false-call comparison |
| `bench/cobinding/run_calibration_and_cobinding.sh` | built-in calibration table and real-data co-binding |
| `python bench/build_report.py results` | rebuild `results/report.html` |

## Limitations

- **Simulator.** It assumed the enzyme prefers A/T-rich DNA; real Hia5 marks A-tracts less.
  The simulated footprint numbers in the report are therefore optimistic.
- **Validation scope.** Three factors at selected ChIP-labelled sites in two cell lines, and CTCF at
  every strong motif genome-wide in one cell line. Other factors were not tested genome-wide.
- **Speed.** fiberstate is slower than fibertools and FiberHMM.
- **Enzyme table.** The context table was learned with one m6A caller and chemistry; use
  `learn-context` for others.

## Citation and licence

Manuscript in preparation. MIT licence (see `LICENSE`).
