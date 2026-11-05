"""Tuned-competitor comparison: every tool's calling setting is chosen on one tuning set
(CTCF in GM12878) and applied unchanged to the other site sets.

NAR's methods guidelines ask that alternatives be run with optimized parameters, not defaults.
fiberstate's footprint cut-off was chosen on CTCF in GM12878, so each competitor gets the same
chance: a grid of settings, the one with the best site AUROC on the tuning set is kept.

Settings
  fiberstate   cut-off on the mean footprint posterior over the motif (0.05 ... 0.95)
  FiberHMM     cut-off on the quality of a footprint covering >= 80% of the motif
               ("any covering footprint" = its default, then 0.05 ... 0.95 of the 0-255 scale)
  HiddenFoot   cut-off on its motif posterior (0.05 ... 0.95; 0.5 = the benchmark's rule)
  fibertools   yes/no calls, so four calling rules built from its footprint codes:
               default (patch spans the window and core unmarked), core unmarked only,
               core unmarked with at least one flank marked, core unmarked with both flanks marked

    python tuned_comparison.py collect LABEL SITES.bed OUT.json fs=FS_POST.bam fh=FH.bam ft=FT.tsv [hf=HF.tsv]
    python tuned_comparison.py compare TUNE_LABEL OUT.json A.json B.json ...
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_sites import MIN_FIBERS, _delong_paired, auroc, load_sites  # noqa: E402
from matched_fpr import fh_scores, fs_scores, hf_scores  # noqa: E402

GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
FT_RULES = {
    "default: patch spans window and core unmarked": lambda c: bool(c & 1) and bool(c & 4),
    "core unmarked": lambda c: bool(c & 4),
    "core unmarked, at least one flank marked": lambda c: bool(c & 4) and not (c & 2 and c & 8),
    "core unmarked, both flanks marked": lambda c: bool(c & 4) and not (c & 2) and not (c & 8),
}


def ft_codes(tsv_p, sites):
    """(site index, read) -> fibertools footprint code (bit 0 patch spans; bit i+1 module i unmarked)."""
    key = {(st["chrom"], st["s"] - 10): i for i, st in enumerate(sites)}
    out = {}
    with open(tsv_p) as f:
        header = f.readline().lstrip("#").rstrip("\n").split("\t")
        col = {h: j for j, h in enumerate(header)}
        for line in f:
            c = line.rstrip("\n").split("\t")
            i = key.get((c[col["chrom"]], int(c[col["start"]])))
            if i is None or not c[col["fiber_names"]]:
                continue
            codes = [int(x) for x in c[col["footprint_codes"]].split(",") if x != ""]
            for code, nm in zip(codes, c[col["fiber_names"]].split(",")):
                out[(i, nm)] = code
    return out


def _per_site(values):
    per = {}
    for (i, _), v in values.items():
        per.setdefault(i, []).append(v)
    return {i: np.array(v) for i, v in per.items() if len(v) >= MIN_FIBERS}


def cmd_collect(label, sites_p, out_p, *specs):
    sites = load_sites(sites_p)
    paths = dict(s.split("=", 1) for s in specs)
    tools = {}
    if "fs" in paths:
        per = _per_site(fs_scores(paths["fs"], sites))
        tools["fiberstate"] = {str(t): {str(i): float(np.mean(v > t)) for i, v in per.items()} for t in GRID}
    if "fh" in paths:
        per = _per_site(fh_scores(paths["fh"], sites))
        st = {"any covering footprint (default)": {str(i): float(np.mean(v > 0)) for i, v in per.items()}}
        st.update({str(t): {str(i): float(np.mean(v > t)) for i, v in per.items()} for t in GRID})
        tools["FiberHMM"] = st
    if "hf" in paths:
        per = _per_site(hf_scores(paths["hf"], sites))
        tools["HiddenFoot"] = {str(t): {str(i): float(np.mean(v > t)) for i, v in per.items()} for t in GRID}
    if "ft" in paths:
        per = _per_site(ft_codes(paths["ft"], sites))
        tools["fibertools"] = {name: {str(i): float(np.mean([rule(int(c)) for c in v])) for i, v in per.items()}
                               for name, rule in FT_RULES.items()}
    json.dump(dict(label=label, bound={str(i): bool(st["bound"]) for i, st in enumerate(sites)}, tools=tools),
              open(out_p, "w"))
    print(f"{label}: " + ", ".join(f"{t} {len(next(iter(v.values())))} sites x {len(v)} settings" for t, v in tools.items()))


def _auc(occ, bound):
    ids = sorted(occ, key=int)
    return auroc([occ[i] for i in ids], [bound[i] for i in ids])


def _rates(occ, bound):
    b = [occ[i] for i in occ if bound[i]]; u = [occ[i] for i in occ if not bound[i]]
    return float(np.mean(u)) if u else float("nan"), float(np.mean(b)) if b else float("nan")


def cmd_compare(tune, out_p, *paths):
    sets = {d["label"]: d for d in (json.load(open(p)) for p in paths)}
    T = sets[tune]
    tools = [t for t in ("fiberstate", "fibertools", "FiberHMM", "HiddenFoot") if t in T["tools"]]
    chosen = {}
    for t in tools:
        aucs = {s: _auc(o, T["bound"]) for s, o in T["tools"][t].items()}
        chosen[t] = max(aucs, key=lambda s: (aucs[s], s))
    res = dict(tuning_set=tune, chosen_setting=chosen, sets={})
    for lab, d in sets.items():
        row = {}
        for t in tools:
            if t not in d["tools"]:
                continue
            o = d["tools"][t][chosen[t]]
            fcr, rec = _rates(o, d["bound"])
            best = max(_auc(oo, d["bound"]) for oo in d["tools"][t].values())
            row[t] = dict(auroc_tuned=_auc(o, d["bound"]), auroc_best_on_this_set=best, false_calls=fcr, bound_footprinted=rec)
        # paired DeLong, fiberstate (tuned) vs each tool (tuned), on sites both score
        fs = d["tools"]["fiberstate"][chosen["fiberstate"]]
        for t in tools[1:]:
            if t not in d["tools"]:
                continue
            o = d["tools"][t][chosen[t]]
            ids = sorted(set(fs) & set(o), key=int)
            y = np.array([d["bound"][i] for i in ids])
            diff, se, z, p = _delong_paired([fs[i] for i in ids], [o[i] for i in ids], y)
            row[t]["fiberstate_minus_auroc"] = diff; row[t]["delong_p"] = p
        res["sets"][lab] = row
    json.dump(res, open(out_p, "w"), indent=1)

    print(f"\nsettings chosen on {tune} (best site AUROC):")
    for t, s in chosen.items():
        print(f"  {t:11s} {s}")
    print(f"\nsite AUROC with tuned settings (p: paired DeLong vs fiberstate; [best possible on that set])")
    print(f"{'set':16s}" + "".join(f"{t:>26s}" for t in tools))
    for lab, row in res["sets"].items():
        cells = []
        for t in tools:
            if t not in row:
                cells.append(f"{'-':>26s}"); continue
            r = row[t]
            p = f" p={r['delong_p']:.2g}" if "delong_p" in r else ""
            cells.append(f"{r['auroc_tuned']:.3f}{p} [{r['auroc_best_on_this_set']:.3f}]".rjust(26))
        print(f"{lab + (' *' if lab == tune else ''):16s}" + "".join(cells))
    print("  * tuning set")
    for t in tools[1:]:
        held = [l for l in res["sets"] if l != tune and t in res["sets"][l]]
        wins = sum(res["sets"][l]["fiberstate"]["auroc_tuned"] > res["sets"][l][t]["auroc_tuned"] for l in held)
        sig = sum(res["sets"][l][t].get("delong_p", 1) < 0.05 and res["sets"][l][t]["fiberstate_minus_auroc"] > 0 for l in held)
        sigl = sum(res["sets"][l][t].get("delong_p", 1) < 0.05 and res["sets"][l][t]["fiberstate_minus_auroc"] < 0 for l in held)
        print(f"held-out sets: fiberstate higher than {t} in {wins}/{len(held)} (significantly higher in {sig}, significantly lower in {sigl})")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"collect": cmd_collect, "compare": cmd_compare}[cmd](*args)
