"""Summarise matched_fpr.py results over several site sets, and test a fiberstate cut-off
chosen on one tuning set (default CTCF_GM12878) on the others.

    python summarize_matched_fpr.py OUT.json TUNE_LABEL label=path.json ...
"""
import json
import sys


def main(out_p, tune, *specs):
    sets = {}
    for sp in specs:
        k, p = sp.split("=", 1)
        sets[k] = json.load(open(p))
    tools = [t for t in sets[next(iter(sets))] if not t.startswith("bootstrap")]
    print("\nBound molecules footprinted when every tool is held to 1% false calls at unbound sites")
    print(f"{'set':16s}" + "".join(f"{t[:14]:>16s}" for t in tools) + "   fiberstate vs others (points, p)")
    summary = {}
    for k, d in sets.items():
        row = {t: d[t]["bound_rate_interpolated"]["0.01"] for t in tools if t in d}
        boots = d.get("bootstrap_bound_rate_diff_at_1pct", {})
        summary[k] = dict(bound_rate_at_1pct=row, pauc=({t: d[t]["partial_auc_fpr_le_0.05"] for t in tools if t in d}), vs=boots)
        vs = "  ".join(f"{o[:10]} {100 * b['diff']:+.1f} ({'<0.001' if b['p_two_sided'] == 0 else '%.2g' % b['p_two_sided']})" for o, b in boots.items())
        print(f"{k:16s}" + "".join(f"{100 * row.get(t, float('nan')):15.1f}%" for t in tools) + "   " + vs)
    # cut-off chosen on the tuning set by site AUROC, applied unchanged elsewhere
    sw = sets[tune]["fiberstate"]["threshold_sweep"]
    best = max(sw, key=lambda t: sw[t]["site_auroc"])
    print(f"\nfiberstate cut-off chosen on {tune} by site AUROC: {best} (default call rule is 0.5)")
    held = {}
    for k, d in sets.items():
        s = d["fiberstate"]["threshold_sweep"]
        held[k] = dict(tuned=s[best], at_0_5=s["0.5"])
        tag = " (tuning set)" if k == tune else ""
        print(f"  {k:16s} site AUROC {s[best]['site_auroc']:.3f} (0.5: {s['0.5']['site_auroc']:.3f}); "
              f"false calls {100 * s[best]['unbound_rate']:.1f}% (0.5: {100 * s['0.5']['unbound_rate']:.1f}%); "
              f"bound {100 * s[best]['bound_rate']:.1f}% (0.5: {100 * s['0.5']['bound_rate']:.1f}%){tag}")
    json.dump(dict(per_set=summary, tuned_cutoff=best, tune_set=tune, held_out=held), open(out_p, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
