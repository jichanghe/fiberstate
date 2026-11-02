"""Real-data validation on Fiber-seq reads around CTCF motifs.

    python -I bench/realdata.py CTCF.bam CTCF_motifs.bed.gz FT_FOOTPRINTS.bed.gz FT_BIN OUT.json

1. Runs fiberstate on every fiber and fibertools `ft add-nucleosomes` on the same BAM.
2. Nucleosome length distributions and per-fiber methylation rates (sanity checks).
3. Aggregate state profile +/-600 bp around CTCF motifs, oriented by motif strand
   (expected biology: a footprint at the motif and phased nucleosomes on both sides).
4. Per-fiber agreement with fibertools' own CTCF footprint calls (`ft footprint`).
5. GC-rich windows: how often each tool calls nucleosomes where few A/T bases exist.
"""
import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pysam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fiberstate.io import fiber_arrays  # noqa: E402
from fiberstate.model import Model  # noqa: E402

FLANK = 600


def read_bed(p):
    op = gzip.open if p.endswith(".gz") else open
    out = []
    with op(p, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            out.append(line.rstrip("\n").split("\t"))
    return out


def _call(args):
    name, seq, m6a = args
    c = Model().call_read(seq, m6a)
    return name, c.path, c.post_linker.astype(np.float32), c.post_nuc.astype(np.float32), \
        c.post_foot.astype(np.float32), c.segments, c.p_access, c.p_protect


def ft_nucs(ft, bam):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "o.bam")
        t0 = time.perf_counter()
        subprocess.run([ft, "add-nucleosomes", bam, out], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dt = time.perf_counter() - t0
        res = {}
        with pysam.AlignmentFile(out, "rb", check_sq=False) as f:
            for rd in f:
                if rd.is_secondary or rd.is_supplementary:
                    continue
                from bench.ftparse import parse_annotations
                segs = parse_annotations(rd)["nuc"]
                res[rd.query_name] = segs
    return res, dt


def main(bam_p, motif_p, ftfp_p, ft, out_p):
    bam = pysam.AlignmentFile(bam_p)
    reads, payload = {}, []
    for r in bam:
        if r.is_secondary or r.is_supplementary or r.is_unmapped:
            continue
        s, m = fiber_arrays(r)
        if s is None:
            continue
        reads[r.query_name] = r
        payload.append((r.query_name, s, m))
    # sanity: m6A only on A/T
    at = sum(int(np.isin(s[m > 0], [65, 84]).sum()) for _, s, m in payload)
    tot = sum(int((m > 0).sum()) for _, s, m in payload)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(2) as ex:
        calls = {c[0]: c for c in ex.map(_call, payload, chunksize=8)}
    fs_time = time.perf_counter() - t0
    ftn, ft_time = ft_nucs(ft, bam_p)
    seqs = {n: s for n, s, _ in payload}
    m6as = {n: m for n, _, m in payload}

    # 2. nucleosome lengths, per-fiber rates
    fs_len = [e - s for c in calls.values() for k, s, e, _ in c[5] if k == 1]
    fs_foot_len = [e - s for c in calls.values() for k, s, e, _ in c[5] if k == 2]
    ft_len = [e - s for v in ftn.values() for s, e in v]
    pa = [c[6] for c in calls.values()]
    raw_rate = [float(m6as[n][np.isin(seqs[n], [65, 84])].mean()) for n in calls]

    # 3. aggregate profile around motifs
    motifs = read_bed(motif_p)
    offs = np.arange(-FLANK, FLANK + 1)
    prof = {k: np.zeros(len(offs)) for k in ("fs_nuc", "fs_foot", "fs_link", "ft_nuc", "m6a", "at")}
    cnt = np.zeros(len(offs))
    site_rows = []
    per_fiber = {}
    for mf in motifs:
        chrom, s, e, name, strand = mf[0], int(mf[1]), int(mf[2]), mf[3], mf[5]
        ctr = (s + e) // 2
        occ = []
        for r in bam.fetch(chrom, ctr - FLANK, ctr + FLANK):
            if r.query_name not in calls or r.is_secondary or r.is_supplementary:
                continue
            r2q = {rp: qp for qp, rp in r.get_aligned_pairs(matches_only=True)}
            c = calls[r.query_name]
            ftmask = np.zeros(len(c[1]), bool)
            for a, b in ftn.get(r.query_name, []):
                ftmask[a:b] = True
            for i, o in enumerate(offs):
                rp = ctr + (o if strand == "+" else -o)
                qp = r2q.get(rp)
                if qp is None:
                    continue
                cnt[i] += 1
                prof["fs_link"][i] += c[2][qp]
                prof["fs_nuc"][i] += c[3][qp]
                prof["fs_foot"][i] += c[4][qp]
                prof["ft_nuc"][i] += ftmask[qp]
                b_ = seqs[r.query_name][qp]
                if b_ in (65, 84):
                    prof["at"][i] += 1
                    prof["m6a"][i] += m6as[r.query_name][qp] > 0
            qc = r2q.get(ctr)
            if qc is not None:
                v = (float(c[2][qc]), float(c[3][qc]), float(c[4][qc]))
                occ.append(v)
                per_fiber[(name, r.query_name)] = v + (bool(ftmask[qc]),)
        P = np.array(occ) if occ else np.zeros((0, 3))
        site_rows.append(dict(name=name, chrom=chrom, start=s, n=len(occ),
                              fs_foot=float(P[:, 2].mean()) if len(P) else None,
                              fs_nuc=float(P[:, 1].mean()) if len(P) else None,
                              fs_link=float(P[:, 0].mean()) if len(P) else None))
    profile = {k: (v / np.maximum(cnt, 1)).tolist() for k, v in prof.items() if k not in ("m6a", "at")}
    profile["m6a_per_AT"] = (prof["m6a"] / np.maximum(prof["at"], 1)).tolist()
    profile["offsets"] = offs.tolist()

    # 4. agreement with ft footprint calls
    ftfp = read_bed(ftfp_p)
    name_by_coord = {(m[0], int(m[1])): m[3] for m in motifs}
    agree = dict(protected_both=0, ft_only=0, fs_only=0, open_both=0)
    split = dict(fs_foot=0, fs_nuc=0, fs_link=0)
    for row in ftfp:
        nm = name_by_coord.get((row[0], int(row[1])))
        codes = [int(x) for x in row[-3].split(",")]
        fibers = row[-1].split(",")
        for code, fb in zip(codes, fibers):
            v = per_fiber.get((nm, fb))
            if v is None:
                continue
            # ft: core CTCF modules 1b-4 (bits 2..6) all m6A-free = protected at the motif
            ft_prot = (code & 0b1111100) == 0b1111100
            fs_prot = (v[1] + v[2]) > 0.5
            key = ("protected_both" if ft_prot and fs_prot else "ft_only" if ft_prot
                   else "fs_only" if fs_prot else "open_both")
            agree[key] += 1
            if ft_prot:
                split[["fs_link", "fs_nuc", "fs_foot"][int(np.argmax(v[:3]))]] += 1

    # 5. GC-rich windows
    gc = dict(fs_nuc_bp=0, ft_nuc_bp=0, fs_conf_nuc_bp=0, win_bp=0)
    allw = dict(fs_nuc_bp=0, ft_nuc_bp=0, win_bp=0)
    for n, c in calls.items():
        s = seqs[n]
        atb = np.isin(s, [65, 84]).astype(float)
        k = np.ones(200) / 200
        frac_at = np.convolve(atb, k, mode="same")
        rich = frac_at < 0.35   # >65% GC
        fsn = c[1] == 1
        ftm = np.zeros(len(s), bool)
        for a, b in ftn.get(n, []):
            ftm[a:b] = True
        conf = np.zeros(len(s), bool)
        for kk, a, b, q in c[5]:
            if kk == 1 and q > 0.9:
                conf[a:b] = True
        gc["win_bp"] += int(rich.sum()); gc["fs_nuc_bp"] += int((fsn & rich).sum())
        gc["ft_nuc_bp"] += int((ftm & rich).sum()); gc["fs_conf_nuc_bp"] += int((conf & rich).sum())
        allw["win_bp"] += len(s); allw["fs_nuc_bp"] += int(fsn.sum()); allw["ft_nuc_bp"] += int(ftm.sum())

    res = dict(
        n_fibers=len(calls), median_len=float(np.median([len(s) for s in seqs.values()])),
        m6a_on_AT_frac=at / max(1, tot),
        fs_sec=fs_time, ft_sec=ft_time,
        fs_nuc_len_hist=np.histogram(fs_len, bins=np.arange(0, 305, 5))[0].tolist(),
        ft_nuc_len_hist=np.histogram(ft_len, bins=np.arange(0, 305, 5))[0].tolist(),
        fs_nuc_len_median=float(np.median(fs_len)), ft_nuc_len_median=float(np.median(ft_len)),
        fs_n_nuc=len(fs_len), ft_n_nuc=len(ft_len), fs_n_foot=len(fs_foot_len),
        fs_foot_len_median=float(np.median(fs_foot_len)) if fs_foot_len else None,
        p_access=dict(q10=float(np.quantile(pa, .1)), median=float(np.median(pa)), q90=float(np.quantile(pa, .9))),
        raw_rate_corr=float(np.corrcoef(pa, raw_rate)[0, 1]),
        profile=profile, sites=site_rows, agreement=agree, ft_protected_split=split,
        gc_rich=gc, overall=allw,
    )
    json.dump(res, open(out_p, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("profile", "fs_nuc_len_hist", "ft_nuc_len_hist")}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
