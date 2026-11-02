import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.metrics import prf, seg_prf  # noqa: E402
from bench.simulate import FOOT, NUC, SimConfig, simulate  # noqa: E402
from fiberstate.model import Model, Params  # noqa: E402
from fiberstate.sites import co_occupancy, occupancy  # noqa: E402


@pytest.fixture(scope="module")
def sim():
    return simulate(SimConfig(n_loci=2, reads_per_locus=6, seed=11))


def test_posteriors_sum_to_one(sim):
    _, reads, _ = sim
    c = Model().call_read(reads[0].seq, reads[0].m6a)
    tot = c.post_linker + c.post_nuc + c.post_foot
    assert np.allclose(tot, 1.0, atol=1e-6)
    assert np.isfinite(c.loglik)


def test_recovers_nucleosomes(sim):
    _, reads, _ = sim
    m = Model()
    tp = nt = np_ = 0
    for r in reads:
        c = m.call_read(r.seq, r.m6a)
        a, b, d, _ = seg_prf(r.segments, c.segments, NUC)
        tp += a; nt += b; np_ += d
    assert prf(tp, nt, np_)[2] > 0.95


def test_per_read_em_tracks_sensitivity(sim):
    _, reads, _ = sim
    m = Model()
    est = [m.call_read(r.seq, r.m6a).p_access for r in reads]
    true = [r.sensitivity for r in reads]
    assert np.corrcoef(est, true)[0, 1] > 0.8


def test_gc_island_gives_no_confident_nucleosome():
    rng = np.random.default_rng(0)
    L = 1200
    seq = np.frombuffer(b"".join(rng.choice([b"A", b"T"], L)), dtype=np.uint8).copy()
    seq[400:800] = np.frombuffer(b"G" * 400, dtype=np.uint8)
    m6a = np.zeros(L, bool)
    acc = np.ones(L, bool)
    m6a[acc] = rng.random(L) < 0.6
    m6a[400:800] = False                     # G cannot be methylated anyway
    c = Model().call_read(seq, m6a)
    # inside the island the model must stay uncertain rather than confidently protected
    assert c.post_nuc[450:750].mean() < 0.9
    for k, s, e, conf in c.segments:
        if k == NUC and s >= 400 and e <= 800:
            assert conf < 0.9


def test_cooccupancy_sign():
    rng = np.random.default_rng(1)
    u = rng.random(400)
    a = [(1 - x, 0.0, x) for x in (u < 0.5).astype(float)]
    b_pos = [(1 - x, 0.0, x) for x in (u < 0.5).astype(float)]
    b_ind = [(1 - x, 0.0, x) for x in (rng.random(400) < 0.5).astype(float)]
    assert co_occupancy(a, b_pos)["z"] > 5
    assert abs(co_occupancy(a, b_ind)["z"]) < 3.5
    o = occupancy(a)
    assert o["ci_low"] < o["occupancy"] < o["ci_high"]


def test_bam_roundtrip(tmp_path):
    pysam = pytest.importorskip("pysam")
    from fiberstate.io import fiber_arrays, write_sim_bam
    _, reads, _ = simulate(SimConfig(n_loci=1, reads_per_locus=3, seed=2))
    p = tmp_path / "x.bam"
    write_sim_bam(str(p), reads)
    with pysam.AlignmentFile(str(p), "rb", check_sq=False) as f:
        for rd, r in zip(f, reads):
            s, m = fiber_arrays(rd)
            assert (s == r.seq).all()
            assert (m.astype(bool) == r.m6a).all()


def test_unmarked_boundaries_are_merged():
    from fiberstate.model import UNRESOLVED, merge_unsupported
    L = 1000
    obs = np.zeros(L)
    obs[0:50] = 1.0                       # marked linker
    obs[700:760] = 1.0                    # marked linker
    path = np.zeros(L, dtype=np.uint8)
    segs = [(1, 50, 197, 0.9), (1, 200, 347, 0.9), (1, 350, 497, 0.9), (1, 760, 907, 0.9)]
    for k, s, e, _ in segs:
        path[s:e] = k
    g = np.full(L, 0.5)
    p2, s2 = merge_unsupported(path, segs, obs, g, g)
    # first three are separated only by unmarked 3 bp gaps -> one unresolved run
    assert s2[0][0] == UNRESOLVED and (s2[0][1], s2[0][2]) == (50, 497)
    # the last one has a marked linker before it? (497-760 contains marks at 700-760) -> kept
    assert s2[1] == segs[3]
    assert (p2[50:497] == UNRESOLVED).all()


def test_linear_kernels_match_ring_kernels():
    """hsmm_lin (default since 0.4) must reproduce hsmm_fast: posteriors to rounding, same Viterbi path."""
    import numpy as np
    from fiberstate import hsmm_fast, hsmm_lin
    from fiberstate.model import Model, Params
    M = Model(Params())
    rng = np.random.default_rng(7)
    for T in (1, 8, 89, 90, 201, 1500, 26000):
        for rate in (0.0, 0.05, 0.3):
            obs = (rng.random(T) < rate).astype(float)
            obs[rng.random(T) < 0.45] = -1
            w = np.exp(rng.normal(0, 0.4, T))
            a = (obs, w, 0.6, 0.03, 0.03, M.q, M.p.frac_nuc, M.dN, M.dF, M.initN, M.initF, *M.pi)
            f, l = hsmm_fast.fb(*a), hsmm_lin.fb(*a)
            for x, y in zip(f[:3], l[:3]):
                assert np.max(np.abs(x - y)) < 1e-10
            assert abs(f[3] - l[3]) < 1e-8 * max(1.0, abs(f[3]))
            assert np.array_equal(hsmm_fast.viterbi(*a), hsmm_lin.viterbi(*a))


def test_footprints_at_cutoffs_match_call_read():
    """Calibration must describe exactly what `call` reports at the cut-off in use."""
    import numpy as np
    from fiberstate.model import Model, Params, FOOT
    rng = np.random.default_rng(3)
    T = 3000
    seq = rng.choice(np.frombuffer(b"ACGT", dtype=np.uint8), T)
    m6a = ((rng.random(T) < 0.25) & np.isin(seq, [65, 84])).astype(np.uint8)
    for cut in (0.3, 0.5, 0.7):
        M = Model(Params(foot_posterior=cut))
        c = M.call_read(seq, m6a)
        feet = [(s, e) for k, s, e, _ in c.segments if k == FOOT]
        assert M.footprints_at_cutoffs(seq, m6a, (cut,))[cut] == feet


def test_co_occupancy_null_is_calibrated():
    """Independent sites: Fisher p < 0.05 in at most ~5% of testable pairs; rarely bound pairs not testable."""
    import numpy as np
    from fiberstate.sites import co_occupancy
    rng = np.random.default_rng(0)
    ps, inf = [], 0
    for _ in range(300):
        acc = rng.random(60) < 0.7
        bi = (rng.random(60) < 0.3) & acc
        bj = (rng.random(60) < 0.3) & acc
        mk = lambda b: [(0.0 if x else (1.0 if a else 0.0), 0.0 if a else 1.0, 1.0 if x else 0.0) for x, a in zip(b, acc)]  # noqa: E731
        r = co_occupancy(mk(bi), mk(bj))
        if r["informative"]:
            inf += 1
            ps.append(r["p"])
    assert inf > 250 and np.mean(np.array(ps) < 0.05) <= 0.07
    rare = [(1.0, 0.0, 0.0)] * 59 + [(0.0, 0.0, 1.0)]
    assert not co_occupancy(rare, rare)["informative"]
