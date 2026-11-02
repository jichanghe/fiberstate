"""Ground-truth simulator for Fiber-seq style single-molecule chromatin reads.

Each simulated read carries:
  * its DNA sequence (with realistic GC heterogeneity, including GC-rich islands
    that contain few methylatable A/T bases),
  * the true chromatin state of every base (linker / nucleosome / TF footprint),
  * the m6A calls a Fiber-seq caller would produce (Hia5 methylates accessible
    A on either strand; calls are incomplete and noisy).

The simulator deliberately includes effects the fiberstate model does NOT
assume, so the benchmark is not a pure "model fits its own generator" test:
  * per-read methylation sensitivity drawn from a Beta distribution,
  * AT-dinucleotide preference of Hia5 (context-dependent methylation rate),
  * nucleosome "breathing": elevated methylation in the outer 10 bp,
  * linker lengths with 10-bp periodicity (not geometric),
  * a minority of sub-nucleosomal particles (hexasomes, ~110 bp).
"""
from __future__ import annotations

import dataclasses
import numpy as np

LINKER, NUC, FOOT = 0, 1, 2
STATE_NAMES = {LINKER: "linker", NUC: "nucleosome", FOOT: "footprint"}
_BASES = np.frombuffer(b"ACGT", dtype=np.uint8)


@dataclasses.dataclass
class SimConfig:
    n_loci: int = 20
    locus_len: int = 12_000
    reads_per_locus: int = 40
    bg_gc: float = 0.42
    gc_islands_per_locus: int = 3
    island_gc: float = 0.80
    island_len: tuple = (600, 2000)
    ndrs_per_locus: int = 4
    sites_per_ndr: tuple = (1, 3)
    coupled_frac: float = 0.5        # fraction of within-NDR site pairs that are coupled
    sensitivity_mean: float = 0.65   # mean per-base m6A detection prob in accessible DNA
    sensitivity_conc: float = 12.0   # Beta concentration for per-read sensitivity
    fp_rate: float = 0.02            # m6A false-positive rate inside protected DNA
    breathing: bool = True
    at_preference: bool = True
    hexasome_frac: float = 0.05
    seed: int = 0


@dataclasses.dataclass
class Site:
    locus: int
    start: int
    end: int
    occupancy: float
    pair_id: int = -1      # sites sharing a pair_id >= 0 are a candidate pair
    coupled: bool = False


@dataclasses.dataclass
class SimRead:
    name: str
    locus: int
    seq: np.ndarray            # uint8 ASCII
    state: np.ndarray          # uint8 per-base true state
    m6a: np.ndarray            # bool per-base m6A call
    sensitivity: float
    segments: list             # list of (state, start, end) true protected segments
    bound: dict                # site index -> bool (true binding on this molecule)


def _random_seq(rng, n, gc):
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
    return _BASES[rng.choice(4, size=n, p=p)]


def make_locus(rng, cfg: SimConfig, locus_id: int):
    L = cfg.locus_len
    seq = _random_seq(rng, L, cfg.bg_gc)
    islands = []
    for _ in range(cfg.gc_islands_per_locus):
        ln = int(rng.integers(*cfg.island_len))
        s = int(rng.integers(0, L - ln))
        seq[s:s + ln] = _random_seq(rng, ln, cfg.island_gc)
        islands.append((s, s + ln))
    # nucleosome-depleted regions (regulatory elements), spaced apart
    ndrs = []
    slots = np.linspace(800, L - 800, cfg.ndrs_per_locus + 2)[1:-1]
    for c in slots:
        c = int(c + rng.integers(-300, 300))
        w = int(rng.integers(150, 350))
        ndrs.append((c - w // 2, c + w // 2, float(rng.uniform(0.5, 0.95))))
    sites = []
    pair_counter = locus_id * 1000
    for (a, b, _) in ndrs:
        k = int(rng.integers(cfg.sites_per_ndr[0], cfg.sites_per_ndr[1] + 1))
        width = b - a
        # place non-overlapping footprints
        placed = []
        for _ in range(k):
            for _try in range(20):
                fl = int(rng.integers(15, 41))
                s = int(rng.integers(a + 10, max(a + 11, b - fl - 10)))
                if all(s + fl + 8 < ps or s > pe + 8 for ps, pe in placed):
                    placed.append((s, s + fl))
                    break
        site_objs = [Site(locus_id, s, e, float(rng.uniform(0.15, 0.85))) for s, e in sorted(placed)]
        if len(site_objs) >= 2:
            coupled = rng.random() < cfg.coupled_frac
            site_objs[0].pair_id = site_objs[1].pair_id = pair_counter
            site_objs[0].coupled = site_objs[1].coupled = coupled
            pair_counter += 1
        sites.extend(site_objs)
    return dict(seq=seq, islands=islands, ndrs=ndrs, sites=sites)


def _linker_len(rng):
    # 10-bp periodic linker lengths (mode ~ 20-50 bp) - not geometric
    base = 10 * int(rng.choice([1, 2, 3, 4, 5, 6], p=[0.10, 0.22, 0.28, 0.2, 0.12, 0.08]))
    return max(5, base + int(rng.normal(0, 3)) + 2)


def _nuc_len(rng, cfg):
    if rng.random() < cfg.hexasome_frac:
        return int(rng.integers(100, 120))
    return int(np.clip(rng.normal(147, 4), 138, 160))


def simulate_read(rng, cfg, locus, locus_id, read_idx, site_index_offset):
    L = len(locus["seq"])
    state = np.zeros(L, dtype=np.uint8)
    segments = []
    bound = {}
    # decide which NDRs are open on this molecule
    open_ndrs = []
    for (a, b, p_open) in locus["ndrs"]:
        if rng.random() < p_open:
            open_ndrs.append((a, b))
    blocked = np.zeros(L, dtype=bool)
    for a, b in open_ndrs:
        blocked[max(0, a):min(L, b)] = True
    # TF binding inside open NDRs (with coupling for coupled pairs)
    sites = locus["sites"]
    pair_draw = {}
    for j, st in enumerate(sites):
        gi = site_index_offset + j
        in_open = any(a <= st.start and st.end <= b for a, b in open_ndrs)
        if not in_open:
            bound[gi] = False
            continue
        if st.pair_id >= 0 and st.coupled:
            # shared latent variable -> strong positive coupling
            if st.pair_id not in pair_draw:
                pair_draw[st.pair_id] = rng.random()
            u = pair_draw[st.pair_id]
            b_ = u < st.occupancy
        else:
            b_ = rng.random() < st.occupancy
        bound[gi] = bool(b_)
        if b_:
            state[st.start:st.end] = FOOT
            segments.append((FOOT, st.start, st.end))
    # nucleosome array: start with random phase, skip blocked (open NDR) regions.
    pos = int(rng.integers(0, 120))
    while pos < L:
        nl = _nuc_len(rng, cfg)
        end = pos + nl
        if end > L:
            break
        if blocked[pos:end].any():
            # jump past the NDR; +1 nucleosome positioned close to NDR edge
            nxt = np.argmax(~blocked[pos:]) if (~blocked[pos:]).any() else L
            idx = pos + np.flatnonzero(blocked[pos:end])[-1] + 1
            # move to end of the blocked block
            while idx < L and blocked[idx]:
                idx += 1
            pos = idx + int(rng.integers(0, 15))
            continue
        state[pos:end] = NUC
        segments.append((NUC, pos, end))
        pos = end + _linker_len(rng)
    segments.sort(key=lambda x: x[1])

    # ---- methylation ----
    seq = locus["seq"]
    is_a = seq == ord("A")
    is_t = seq == ord("T")
    methylatable = is_a | is_t
    sens = float(rng.beta(cfg.sensitivity_mean * cfg.sensitivity_conc,
                          (1 - cfg.sensitivity_mean) * cfg.sensitivity_conc))
    p = np.full(L, cfg.fp_rate)
    acc = state == LINKER
    p[acc] = sens
    if cfg.at_preference:
        # Hia5 prefers A in AT / TA dinucleotide context
        at_ctx = np.zeros(L, dtype=bool)
        at_ctx[:-1] |= methylatable[1:]
        at_ctx[1:] |= methylatable[:-1]
        mult = np.where(at_ctx, 1.12, 0.70)
        p[acc] = np.clip(p[acc] * mult[acc], 0, 0.98)
    if cfg.breathing:
        for st_, s, e in segments:
            if st_ == NUC:
                for edge in (slice(s, min(e, s + 10)), slice(max(s, e - 10), e)):
                    p[edge] = np.maximum(p[edge], 0.3 * sens)
    m6a = methylatable & (rng.random(L) < p)
    name = f"locus{locus_id}_read{read_idx}"
    return SimRead(name, locus_id, seq.copy(), state, m6a, sens,
                   [s for s in segments], bound)


def simulate(cfg: SimConfig):
    rng = np.random.default_rng(cfg.seed)
    loci, reads, all_sites = [], [], []
    for li in range(cfg.n_loci):
        loc = make_locus(rng, cfg, li)
        off = len(all_sites)
        loci.append(loc)
        all_sites.extend(loc["sites"])
        for ri in range(cfg.reads_per_locus):
            reads.append(simulate_read(rng, cfg, loc, li, ri, off))
        loc["site_offset"] = off
    return loci, reads, all_sites


if __name__ == "__main__":
    cfg = SimConfig(n_loci=2, reads_per_locus=3)
    loci, reads, sites = simulate(cfg)
    r = reads[0]
    print(r.name, len(r.seq), "sens=%.2f" % r.sensitivity, "m6A=%d" % r.m6a.sum(),
          "nucs=%d" % sum(1 for s in r.segments if s[0] == NUC),
          "foots=%d" % sum(1 for s in r.segments if s[0] == FOOT))
    print("sites:", len(sites), "coupled pairs:",
          len({s.pair_id for s in sites if s.coupled}))
