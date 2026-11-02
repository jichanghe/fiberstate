"""Render results/summary.json (+ results/realdata/*.json) into a self-contained HTML report."""
import json, os, sys

CAVEATS = """
<p><b>Every fiberstate number on this page comes from versions 0.3 and 0.4.x with default settings</b> (posterior footprint calling at 0.5, GM12878 enzyme table), except the simulated benchmark at the top and the historical notes at the end. Version 0.4 only changes speed and reproduces 0.3's results (see Speed). Rows that existed only for older versions (a small CTCF set from the fibertools-rs test data) were removed.</p>
<p><b>TF-site tests: at a matched false-positive rate, fiberstate recovers the most binding of the motif-free callers.</b> Held to 1% false calls at unbound sites, fiberstate footprints more molecules at bound sites than FiberHMM and fibertools in all six sets (CTCF, REST and NRF1 in GM12878 and K562), significantly in five (+0.8 to +11.7 percentage points; REST in K562 p = 0.09 to 0.10). It ties HiddenFoot, which is given the motif matrix (within 2.4 points in five sets; HiddenFoot ahead for REST in GM12878, 0.9 points, p = 0.004; fiberstate ahead for CTCF in K562, 6.8 points, p = 0.15). The lower sensitivity seen with default calls is an operating point: fiberstate's default makes about half as many false calls as the others, and at equal false calls it finds more. On site AUROC, which ignores false-call rates, the four tools are close: fiberstate is highest in five of six sets and lowest for NRF1 in K562.</p>
<p><b>Genome-wide CTCF: the result holds everywhere.</b> Every strong CTCF motif on all 22 autosomes of GM12878 was scored, with no site selection or score matching: 920,549 sites covered by at least 10 molecules (39,537 ChIP-bound), and 6.75 million molecules subsampled to 30%.
<ul><li>Held to 1% false calls, fiberstate recovered 19.0% of bound molecules, against 10.9% for fibertools and 6.8% for FiberHMM (95% window-bootstrap CIs of the differences: 7.9&ndash;8.2 and 11.9&ndash;12.4 points). It led on all 22 chromosomes.</li>
<li>Default false calls were 0.71%, 1.92% and 2.93%; site AUROC was 0.834, 0.832 and 0.822.</li>
<li>For the weakest motifs fiberstate tied fibertools (0.732 vs 0.733); for the strongest it led both (0.915 vs 0.909 and 0.906). The lead over fibertools narrows to 0.7 points at 2% false calls.</li></ul></p>
<p><b>Calibration.</b> <span class="mono">fiberstate calibrate</span> applies the exact calling steps at every cut-off. It reproduces the benchmark's false-call rates (CTCF in GM12878: 1.40% at 0.5). Pooled over the six sets (94,211 molecules at unbound sites), the false-call rate is 2.53% at cut-off 0.1, 1.09% at 0.5 and 0.61% at 0.9. This is the built-in table behind <span class="mono">--false-call-rate</span>. Per set, rates at 0.5 range sixfold, from 0.26% (REST, K562) to 1.51% (CTCF, K562), so calibrating on one's own sites is preferable.</p>
<p><b>Co-binding on real molecules: no cooperative pairs, and the test is conservative.</b> We tested pairs of CTCF, REST and NRF1 motif sites 60–1000 bp apart near the benchmark sites. In GM12878, 5,612 pairs were informative.
<ul><li>Control pairs of two unbound sites reached p &lt; 0.05 in 0.4% of cases.</li>
<li>Conditioning on accessibility roughly halved the raw association (median log odds ratio +0.66 to +0.28 for controls, +0.55 to +0.37 for bound pairs).</li>
<li>No pair reached q &lt; 0.1. In K562 only 101 pairs were testable, with the same pattern.</li></ul>
Most pairs (89,127) were not testable, because one site was footprinted on fewer than 3 molecules. Detecting weak cooperativity needs deeper coverage.</p>
<p><b>The default cut-off stays.</b> Choosing fiberstate's cut-off on CTCF in GM12878 alone picks 0.35 instead of 0.5, which barely changes AUROC there (0.890 vs 0.888) and does not help on the other five sets (changes of -0.012 to +0.016). The cut-off only trades false calls for recovered binding; users who prefer more sensitivity can lower <span class="mono">--foot-posterior</span>.</p>
<p><b>REST is hard for every tool; NRF1 less so.</b> NRF1 binds a GC-rich motif with few A/T bases for Hia5 to mark, yet all four tools reach AUROC 0.82 to 0.84 in GM12878. For REST only 3 to 7% of molecules are footprinted at ChIP-bound REST sites, and in K562 no tool exceeds AUROC 0.6. REST binding in these cells may be weak or transient, ChIP peaks may include indirect binding, or REST may protect too little DNA from Hia5 to be seen. In GM12878 an unmarked motif is actually more common at unbound sites (AUROC 0.287), because bound sites lie in accessible DNA, so a footprint must be called inside open DNA to count.</p>
<p><b>HiddenFoot needed a lower motif cutoff.</b> It only considers a TF where the motif score is within ln(1000) of the consensus; 86% of the CTCF and 77% of the REST test sites fall below that, so the first run could not bind at most of them (CTCF AUROC 0.664). With <span class="mono">-cutoffwms 1e-6</span> it is competitive (CTCF GM12878 0.875, REST GM12878 0.735) but weaker in K562 (0.828, 0.550). It fits its model separately at every 301 bp site, which may be too little data at sites with few molecules; that setting was not tuned.</p>
<p><b>Why 0.3 calls footprints from posteriors.</b> At strongly bound CTCF sites that 0.2 missed, the motif carried no m6A inside a median 36 bp unmarked stretch and fiberstate's own footprint posterior there was a median 0.67, but Viterbi decoding chose linker: footprint lengths vary from 8 to 90 bp, so the footprint state's probability is split across many paths. Calling a footprint wherever the posterior exceeds 0.5 raised CTCF AUROC from 0.877 to 0.896 (GM12878, p = 0.03) and from 0.822 to 0.888 (K562, p = 9e-9).</p>
<p><b>The cost: twice as many footprint calls genome-wide, less enriched.</b> In the 0.5 Mb GM12878 region, 0.3 calls 21,798 footprints against 10,006 for 0.2, and the share inside FIRE peaks falls from 21% (2.4×) to 18% (2.0×). At equal call numbers its TF-sized calls (15–50 bp) remain more enriched than FiberHMM's (1.96× vs 1.64×). Nucleosome and accessibility results barely moved (DNase AUROC 0.836 in both versions).</p>
<p><b>Independent accessibility reference.</b> Against ENCODE DNase-seq peaks, counting every base outside a nucleosome or footprint as open, FiberHMM leads the per-molecule callers (GM12878 0.865, K562 0.906) over fibertools (0.838, 0.886) and fiberstate (0.836, 0.878). On open stretches of at least 100 bp the three tie (0.909 to 0.921). FIRE scores 0.951 in both cell lines; it was trained on GM12878, but scores the same on K562, so it is genuinely the best accessibility caller here.</p>
<p><b>Unresolved stretches.</b> fiberstate reports long unmarked stretches as one protected segment with an estimated nucleosome count instead of guessing positions (16% of protected DNA on real data); <span class="mono">--keep-inferred</span> restores the guesses. In simulation this lowers nucleosome recall at unchanged precision.</p>
<p><b>The simulator had the enzyme backwards.</b> It assumed A/T-rich context is marked more; real data shows A-tracts are marked 15–25× less. The simulated benchmark at the top overstates footprint accuracy.</p>
<p><b>Speed.</b> Version 0.4 replaces the ring-buffer kernels with dot products over a plain history, which the CPU can vectorise; a nucleosome lasts at least 90 bp, so the sum at each base only needs entries made 90 or more bases earlier. Forward-backward is 2.2× faster and Viterbi 1.4×; a whole molecule takes 26 ms of model time on one core here instead of 52 (fibertools: 1.5 ms). Posteriors agree with 0.3 to 4e-14 and every benchmark number on this page is unchanged to four decimals. Because the per-molecule rates are refitted with differently rounded sums, 1% of nucleosome edges move by 1–2 bp where two positions score almost exactly the same; no segment is gained or lost. On the user's laptop the gain was smaller than here: the 44,614-fiber CTCF set took 9.0 minutes on 8 threads with 0.4.1 (while also writing per-base posteriors) against 12 with 0.3, and FiberHMM needs 1.4 minutes on 4 threads. Version 0.4.1 moved reading, m6A decoding and tag writing from the main process into the workers (identical output); the laptop is now limited by worker CPU, with 8 workers sharing a 12-core laptop chip. A 30× human genome (about 5.5 million fibers) needs roughly 40 core-hours. Most of the remaining time is the four EM passes that refit each molecule's marking rates; <span class="mono">--em-iters 2</span> trades rate precision for about 1.5× more speed.</p>
<p><b>Other tools.</b> The workspace may not execute third-party programs, so HiddenFoot (commit d1499e8, built with g++ -O3 -fopenmp) and FiberHMM (v3.0.2, <span class="mono">--enzyme hia5 --seq pacbio</span>, defaults), is run on the user's computer. FiberHMM's default m6A threshold (ML 12 of 255) is far looser than the 0.5 used by fiberstate and fibertools. fibertools' <span class="mono">fiber-hmm</span> subcommand is a placeholder in 0.13.1.</p>
<p><b>History, fixed bugs.</b> (1) fibertools' annotation tag was first read as 0-based and in stored-read orientation, which mirrored its nucleosomes on reverse reads and gave fiberstate two advantages it does not have. (2) An early CTCF comparison counted marks in the flank and did not flip minus-strand motifs. (3) fiberstate 0.1.0 did not scale with threads because numpy's BLAS started its own thread pool in every worker; 0.1.1 limits BLAS to one thread per worker. (4) The fast kernel drifted to NaN on molecules over 25 kb; it now re-anchors every 32 bases.</p>
<p><b>Not yet tested:</b> genome-wide data, MNase comparisons, other m6A callers (the enzyme table may be caller-specific), soft ML-probability input.</p>
"""

SETUP = """
<li>Simulated 12 kb loci with 42% background GC and three GC islands each (80% GC, 0.6–2 kb), four regulatory elements per locus with 1–3 TF sites (15–40 bp) and per-element open probability 0.5–0.95.</li>
<li>Robustness sweeps: 12 loci × 30 fibers, 3 seeds per condition. Sensitivity sweep at false-positive rate 0.02; false-positive sweep at mean sensitivity 0.65.</li>
<li>A call counts as correct when it overlaps a true segment by at least 50% of both lengths (one-to-one matching). Dyad error is the median center offset of matched nucleosomes.</li>
<li>fibertools-rs v0.13.1 compiled from crates.io, run as <span class="mono">ft add-nucleosomes</span> with defaults on the same m6A calls.</li>
<li>Gap threshold: m6A-free runs ≥ 75 bp are nucleosomes, 20–74 bp are footprints (grid-searched). Per-base HMM: 3-state, geometric durations, every base an observation (grid-searched rates).</li>
<li>Co-occupancy: 50 loci × 60 fibers, two sites per element; coupled pairs share one latent binding draw. Score is the z of the log odds ratio; a call needs z > 1.645.</li>
<li>Tools considered (searched 31 October 2026: GitHub, PyPI, Bioconductor, bioRxiv and journals). Compared: FiberHMM 3.0.2 and fibertools-rs 0.13.1 (<span class="mono">add-nucleosomes</span>; motif-guided <span class="mono">footprint</span> in the CTCF test); FIRE as the accessibility reference and as a competitor against DNase-seq. <b>Compared at TF sites:</b> HiddenFoot (Molina lab, bioRxiv 2025; per-molecule TF, nucleosome and Pol II posteriors from motifs and per-region methylation matrices; run per site through <span class="mono">bench/hiddenfoot_adapter.py</span>). <b>Not yet compared, but relevant:</b> the Ramani lab SAMOSA / SMRT-Tag HMM (PacBio m6A, two states, research code); nomeR (FMI Basel; per-molecule footprint probabilities for single-molecule footprinting matrices, motif-free). Not comparable: FootprintCharter and dinoR (cluster- or window-level calls on bisulfite SMF), NP-SMLR and nanoNOMe (nanopore GpC), smftools (HMM stage not documented for m6A), dimelo-toolkit and methylscaper (visualisation), IDLI (classifies existing footprints), bonFIRE (compares FIRE peaks), fiberFold (predicts 3D contacts from footprints), FiberBrowser (uses FiberHMM's calls), fiberseq-qc, UCSC hub. fibertools' <span class="mono">fiber-hmm</span> subcommand is a placeholder in 0.13.1.</li>

<li>Real data: GM12878 FIRE v0.1 CRAM streamed for chr19:45.0–45.5 Mb (m6A tags only), FIRE peaks with FDR ≤ 0.05; GC-rich = ≥65% GC over 200 bp of hg38. CTCF set from the fibertools-rs test data.</li>
"""

d = sys.argv[1] if len(sys.argv) > 1 else "results"
s = open(os.path.join(d, "summary.json")).read()
rd0 = json.load(open(os.path.join(d, "realdata", "gm12878_chr19.json")))
c0 = json.load(open(os.path.join(d, "realdata", "ctcf_core_none.json")))
c1 = json.load(open(os.path.join(d, "realdata", "ctcf_core_gm.json")))
ctv = json.load(open(os.path.join(d, "realdata", "ctcf_validation.json")))
FS = json.load(open(os.path.join(d, "realdata", "score_fiberstate03.json")))   # fiberstate 0.3 defaults
FT = json.load(open(os.path.join(d, "realdata", "score_fibertools.json")))
_fh = os.path.join(d, "realdata", "score_fiberhmm.json")
FM = json.load(open(os.path.join(d, "realdata", "footprints_matched_v03.json")))["length_15_50"]
FH = json.load(open(_fh)) if os.path.exists(_fh) else None
prof = ctv["profile"]; z = prof["offsets"].index(0)
CC = json.load(open(os.path.join(d, "realdata", "ctcf_chip_summary.json")))
DN = {k: json.load(open(os.path.join(d, "realdata", f"dnase_{k}.json"))) for k in ("fibertools", "FiberHMM", "FIRE")}
DN["fiberstate"] = FS
_cs = os.path.join(d, "realdata", "GM12878_by_chip_strength.json")   # from run_hiddenfoot_ctcf.sh (fiberstate 0.3)
CS = json.load(open(_cs)) if os.path.exists(_cs) else None
K5 = json.load(open(os.path.join(d, "realdata", "k562_summary.json")))
PS = json.load(open(os.path.join(d, "realdata", "posterior_summary.json")))
TFS = json.load(open(os.path.join(d, "realdata", "tf_sites_summary.json")))
MFS = json.load(open(os.path.join(d, "realdata", "matched_fpr_summary.json")))
prot1 = c1["both_protected"]["n"] + c1["fs_only"]["n"]
bad1 = c1["fs_only"]["m6a_in_core_dist"]["2"] + c1["fs_only"]["m6a_in_core_dist"]["3+"]
ftp = c1["both_protected"]["n"] + c1["ft_only"]["n"]
pct = lambda x: "n/a" if x is None else f"{100*x:.0f}%"  # noqa: E731
pct1 = lambda x: "n/a" if x is None else f"{100*x:.1f}%"  # noqa: E731
fh = lambda key, fmt: ("run pending" if FH is None else fmt(FH.get(key)))  # noqa: E731
def foot(X):
    if X is None:
        return "run pending"
    if not X.get("n_footprints"):
        return "not called"
    return f"{pct(X['footprint_frac_in_peaks'])} ({X['footprint_enrichment']:.1f}×)"
# [check, fiberstate, fibertools, FiberHMM, pill class, reading]
REAL = [
    ["K562, accessibility vs DNase-seq, every open base", f"{K5['dnase']['auroc_all']['fiberstate']:.3f}", f"{K5['dnase']['auroc_all']['fibertools']:.3f}", f"{K5['dnase']['auroc_all']['FiberHMM']:.3f}", "mid", "FIRE {:.3f} on held-out data too".format(K5['dnase']['auroc_all']['FIRE'])],
    ["Accessibility vs ENCODE DNase-seq (ENCFF784FOQ), every open base", f"{DN['fiberstate']['dnase_auroc']:.3f}", f"{DN['fibertools']['dnase_auroc']:.3f}", f"{DN['FiberHMM']['dnase_auroc']:.3f}", "mid", "FIRE {:.3f}, best; same on held-out K562".format(DN['FIRE']['dnase_auroc'])],
    ["Accessibility vs DNase-seq, open stretches ≥ 100 bp only", f"{DN['fiberstate']['dnase_auroc_long_patches']:.3f}", f"{DN['fibertools']['dnase_auroc_long_patches']:.3f}", f"{DN['FiberHMM']['dnase_auroc_long_patches']:.3f}", "mid", "three-way tie"],
    ["Nucleosome length, most common (expected ~147 bp)", f"{FS['nuc_len_mode']:.0f} bp", f"{FT['nuc_len_mode']:.0f} bp", fh("nuc_len_mode", lambda v: f"{v:.0f} bp"), "good", "fiberstate closest (partly its prior)"],
    ["Nucleosome repeat length, most common (human ~185–190 bp)", f"{FS['repeat_mode']:.0f} bp", f"{FT['repeat_mode']:.0f} bp", fh("repeat_mode", lambda v: f"{v:.0f} bp"), "good", "all in range"],
    ["A/T marked inside called nucleosomes (lower = stricter)", pct1(FS["m6a_per_AT_inside_nucleosomes"]), pct1(FT["m6a_per_AT_inside_nucleosomes"]), fh("m6a_per_AT_inside_nucleosomes", pct1), "mid", "fibertools strictest by design"],
    ["Neighbouring nucleosomes with linker under 10 bp", pct(FS["back_to_back"]), pct(FT["back_to_back"]), fh("back_to_back", pct), "mid", "fiberstate and FiberHMM tie, fewer than fibertools"],
    ["Nucleosome occupancy, GC-rich DNA inside FIRE peaks", pct(FS["nuc_occ_gc_rich_in_peaks"]), pct(FT["nuc_occ_gc_rich_in_peaks"]), fh("nuc_occ_gc_rich_in_peaks", pct), "mid", "three-way tie (earlier gap was a parsing bug)"],
    ["Nucleosome occupancy, ordinary DNA outside peaks", pct(FS["nuc_occ_normal_outside"]), pct(FT["nuc_occ_normal_outside"]), fh("nuc_occ_normal_outside", pct), "mid", "fibertools calls fewer"],
    ["TF-sized footprints (15–50 bp): enrichment in FIRE peaks at equal call numbers", f"{FM['fiberstate_enrichment']:.2f}× ({FM['n_fiberstate']:,} calls)", "not called", f"{FM['fiberhmm_top_enrichment']:.2f}× (top {FM['matched_n']:,} of {FM['n_fiberhmm']:,} by its score)", "good", "fiberstate more enriched"],
]
REAL.insert(0, ["Wall time, 8 threads (user's computer): GM12878 CTCF set, 44,614 fibers / K562, 14,976 fibers",
                f"0.4.1: 9.0 min / 3.1 min (0.3: {PS['seconds_8_threads']['GM12878_p05']/60:.0f} min / {PS['seconds_8_threads']['K562_p05']/60:.0f} min)", "not timed at 8 threads",
                f"{CC['FiberHMM']['seconds']/60:.1f} min (GM12878, 4 threads)", "bad", "fiberstate ~6× slower than FiberHMM despite twice the threads"])
REAL.insert(1, ["Model time per fiber, one core (this workspace, 3,125 fibers)", "26 ms (0.4; 52 ms in 0.3)", "1.5 ms", "not timed", "bad", "fibertools about 17× faster"])
t = open(os.path.join(os.path.dirname(__file__), "report_template.html")).read()
out = (t.replace("__SUMMARY__", s).replace("__REAL__", json.dumps(REAL)).replace("__TF__", json.dumps(TFS)).replace("__MFPR__", json.dumps(MFS))
       .replace("__CAVEATS__", json.dumps(CAVEATS)).replace("__SETUP__", json.dumps(SETUP)))
open(os.path.join(d, "report.html"), "w").write(out)
print("wrote", os.path.join(d, "report.html"))
