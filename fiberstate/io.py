"""BAM input (Fiber-seq MM/ML tags) and BED / BAM-tag output."""
from __future__ import annotations

import array

import numpy as np

try:
    import pysam
except ImportError:  # pragma: no cover
    pysam = None

_COMP = bytes.maketrans(b"ACGTNacgtn", b"TGCANtgcan")


def fiber_arrays(read, ml_threshold=0.5, soft=False):
    """Return (seq uint8, m6a float array) in the read's query (as-stored) orientation.

    Fiber-seq m6A appears as A+a (top strand) and T-a (bottom strand) in MM/ML.
    """
    seq = read.query_sequence
    if seq is None:
        return None, None
    s = np.frombuffer(seq.encode(), dtype=np.uint8).copy()
    m = np.zeros(len(s), dtype=np.float64)
    try:
        mods = read.modified_bases
    except (ValueError, KeyError):
        mods = None
    if mods:
        for (base, strand, code), lst in mods.items():
            if code not in ("a", 21839):  # 'a' = 6mA; 21839 = ChEBI code for 6mA
                continue
            for pos, q in lst:
                if pos < 0 or pos >= len(m):
                    continue
                pq = (q + 0.5) / 256.0
                if soft:
                    m[pos] = max(m[pos], pq)
                elif pq >= ml_threshold:
                    m[pos] = 1.0
    return s, m


def query_to_ref(read):
    """Array mapping query position -> reference position (-1 if unaligned), built from the CIGAR."""
    n = read.query_length
    out = np.full(n, -1, dtype=np.int64)
    if read.is_unmapped or not read.cigartuples:
        return out
    q, r = 0, read.reference_start
    for op, ln in read.cigartuples:
        if op in (0, 7, 8):          # M, =, X: both advance
            out[q:q + ln] = np.arange(r, r + ln)
            q += ln; r += ln
        elif op in (1, 4):           # I, S: query only
            q += ln
        elif op in (2, 3):           # D, N: reference only
            r += ln
    return out


def ref_to_query(read):
    if read.is_unmapped:
        return {}
    return {rp: qp for qp, rp in read.get_aligned_pairs(matches_only=True)}


def segments_to_bed_rows(read, call, q2r=None):
    names = {1: "nucleosome", 2: "footprint", 3: "protected_unresolved"}
    rows = []
    for k, s, e, conf in call.segments:
        if q2r is not None and not read.is_unmapped:
            # nearest aligned positions inside the segment
            rs = q2r[s:e]
            rs = rs[rs >= 0]
            if len(rs) == 0:
                continue
            chrom, a, b = read.reference_name, int(rs.min()), int(rs.max()) + 1
        else:
            chrom, a, b = read.query_name, s, e
        rows.append((chrom, a, b, names[k], read.query_name, conf, s, e,
                     call.p_access, call.p_protect))
    return rows


def add_tags(read, call):
    """Write fibertools-compatible ns/nl (nucleosomes) and as/al (accessible patches),
    plus fiberstate-specific fs/fl (footprints) and nq/fq (confidence 0-255)."""
    nuc = [(s, e, c) for k, s, e, c in call.segments if k == 1]
    foot = [(s, e, c) for k, s, e, c in call.segments if k == 2]
    prot = sorted([(s, e) for k, s, e, _ in call.segments])
    acc = []
    prev = 0
    for s, e in prot:
        if s > prev:
            acc.append((prev, s))
        prev = e
    length = getattr(call, "length", None) or len(call.path)
    if prev < length:
        acc.append((prev, length))
    read.set_tag("ns", array.array("I", [int(x) for x in [s for s, _, _ in nuc]]))
    read.set_tag("nl", array.array("I", [int(x) for x in [e - s for s, e, _ in nuc]]))
    read.set_tag("nq", array.array("B", [int(x) for x in [int(round(255 * c)) for *_, c in nuc]]))
    read.set_tag("as", array.array("I", [int(x) for x in [s for s, _ in acc]]))
    read.set_tag("al", array.array("I", [int(x) for x in [e - s for s, e in acc]]))
    read.set_tag("fs", array.array("I", [int(x) for x in [s for s, _, _ in foot]]))
    read.set_tag("fl", array.array("I", [int(x) for x in [e - s for s, e, _ in foot]]))
    read.set_tag("fq", array.array("B", [int(x) for x in [int(round(255 * c)) for *_, c in foot]]))
    unr = [(s, e) for k, s, e, _ in call.segments if k == 3]
    read.set_tag("us", array.array("I", [int(s) for s, _ in unr]))
    read.set_tag("ul", array.array("I", [int(e - s) for s, e in unr]))
    read.set_tag("pa", float(call.p_access), value_type="f")
    # Molecular-annotation tag (same spec as fibertools/FiberHMM), in stored-SEQ frame.
    parts = [str(length)]
    for name, kk in (("nuc", 1), ("tf", 2), ("unr", 3)):
        iv = [(s_, e_) for k, s_, e_, _ in call.segments if k == kk]
        if iv:
            parts.append(f"{name}.:" + ",".join(f"{s_ + 1}-{e_ - s_}" for s_, e_ in iv))
    read.set_tag("MA", ";".join(parts), value_type="Z")


def write_sim_bam(path, reads, chrom_prefix="sim"):
    """Write simulated reads as an unaligned BAM with Fiber-seq style MM/ML tags."""
    header = {"HD": {"VN": "1.6", "SO": "unknown"}}
    with pysam.AlignmentFile(path, "wb", header=header) as out:
        for r in reads:
            a = pysam.AlignedSegment(out.header)
            seq = r.seq.tobytes().decode()
            a.query_name = r.name
            a.query_sequence = seq
            a.flag = 4
            a.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
            # A+a over all A (explicit '?'-free implicit mode: unlisted A = unmodified)
            a_pos = np.flatnonzero(r.seq == ord("A"))
            t_pos = np.flatnonzero(r.seq == ord("T"))
            mm, ml = [], []
            for base, pos in (("A", a_pos), ("T", t_pos)):
                called = pos[r.m6a[pos]]
                # MM skip counts are relative to occurrences of `base`
                idx = np.searchsorted(pos, called)
                skips = np.diff(np.concatenate(([-1], idx))) - 1
                strand = "+" if base == "A" else "-"
                mm.append(f"{base}{strand}a." + "".join("," + str(int(x)) for x in skips) + ";")
                ml.extend([230] * len(called))
            a.set_tag("MM", "".join(mm), value_type="Z")
            a.set_tag("ML", array.array("B", [int(x) for x in ml]))
            out.write(a)
