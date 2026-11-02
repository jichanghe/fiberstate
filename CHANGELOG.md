# Changelog

## 0.5.2 (2026-11-02)

- Zenodo archiving metadata (`.zenodo.json`). No code changes; results are identical to 0.5.1.

## 0.5.1 (2026-11-01)

- **Built-in calibration table** (`fiberstate/data/footprint_calibration.tsv`).
  - It pools `fiberstate calibrate` over ChIP-labelled CTCF, REST and NRF1 sites in GM12878 and
    K562 (94,211 molecules at unbound sites).
  - The false-call rate is 1.09% at the default cut-off, and `--false-call-rate 0.01` selects
    cut-off 0.6.
  - Per-set rates at 0.5 range from 0.26% to 1.51%.

## 0.5.0 (2026-10-31)

- **Calibrated footprint calling.**
  - New `fiberstate calibrate` measures the false-call rate of every footprint cut-off at
    motif sites known to be unbound. It runs the model once per molecule and applies the exact
    calling steps of `call` at each cut-off.
  - `fiberstate calibrate --merge` pools calibration tables.
  - `fiberstate call --false-call-rate R [--calibration TABLE]` picks the cut-off for a chosen
    false-call rate. Each footprint gets an estimated false-call rate: BED column 11 and BAM
    tag `fe`.
- **`--posterior-tags`** now writes the nucleosome posterior (`np`) as well as the footprint
  posterior (`fp`).
- **Co-binding statistic corrected.**
  - p-values now come from Fisher's exact test on the (soft, accessibility-weighted) 2×2 table.
  - Pairs carry an `informative` flag: each site bound, and unbound, on at least 3 molecules.
  - The previous normal approximation added 0.5 to every cell, which made pairs of rarely bound
    sites look coupled.
  - In the simulated co-binding benchmark, the accessibility-conditioned test finds 77% of
    coupled pairs at a 2.7% false discovery rate (previously 73% and 2.8%).
- `bench` is no longer installed as a top-level package.

## 0.4.1

- Reading, m6A decoding, BED formatting and BAM tagging moved from the main process into the
  workers, and BAM (de)compression is multi-threaded. Output is byte-identical to 0.4.0.
- New `--posterior-tags`: writes the per-base footprint posterior.

## 0.4.0

- New kernels (`hsmm_lin`): dot products over a contiguous history replace ring buffers.
  - Forward–backward is 2.2× faster and Viterbi 1.4×; one molecule takes 26 ms instead of 52.
  - Posteriors match 0.3 to 4e-14, and every benchmark number is unchanged to four decimals.
  - 1% of nucleosome edges move by 1–2 bp, at near-ties caused by rounding in the per-molecule
    rate fit.

## 0.3.0

- Footprints are also called wherever their posterior exceeds 0.5 inside linker runs
  (`--foot-posterior`). Viterbi alone under-calls them, because a footprint's probability is
  spread over many lengths.
  - CTCF ChIP-seq AUROC rose from 0.877 to 0.896 (GM12878) and from 0.822 to 0.888 (K562).

## 0.2.0

- Hia5 sequence-context table: a 5-mer marking weight, learned on GM12878. New
  `learn-context` command.
- Protected stretches with no separating m6A are merged into one "unresolved" segment
  (`--keep-inferred` restores the old behaviour).

## 0.1.1

- BLAS limited to one thread per worker; multi-threaded runs now scale.
- Fast kernels re-anchor their running totals every 32 bases (fixes NaN on molecules over 25 kb).

## 0.1.0

- First version: sequence-aware explicit-duration HSMM, per-molecule EM, occupancy and
  co-occupancy, simulator and benchmarks.
