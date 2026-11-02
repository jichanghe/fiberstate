"""Find the ENCODE GRCh38 IDR thresholded peak file for a TF ChIP-seq target in a cell line.

    python encode_peaks.py TARGET CELL            -> prints "ACCESSION<TAB>download URL<TAB>experiment"
Prefers files ENCODE marks as the default ('preferred_default'), then conservative over optimal
IDR peaks, then the most recent. Uses the public ENCODE REST API (no login).
"""
import json
import sys
import urllib.parse
import urllib.request

BASE = "https://www.encodeproject.org"


def main(target, cell):
    q = urllib.parse.urlencode([
        ("type", "File"), ("file_format", "bed"), ("assembly", "GRCh38"), ("status", "released"),
        ("target.label", target), ("biosample_ontology.term_name", cell),
        ("assay_title", "TF ChIP-seq"), ("format", "json"), ("limit", "all"),
        ("field", "accession"), ("field", "href"), ("field", "output_type"), ("field", "dataset"),
        ("field", "preferred_default"), ("field", "date_created"),
    ])
    req = urllib.request.Request(f"{BASE}/search/?{q}", headers={"Accept": "application/json"})
    files = json.load(urllib.request.urlopen(req, timeout=60)).get("@graph", [])
    files = [f for f in files if "IDR thresholded peaks" in f.get("output_type", "")]
    if not files:
        sys.exit(f"no released GRCh38 IDR thresholded peaks for {target} in {cell}")
    files.sort(key=lambda f: (bool(f.get("preferred_default")), f["output_type"].startswith("conservative"),
                              f.get("date_created", "")), reverse=True)
    f = files[0]
    print(f"{f['accession']}\t{BASE}{f['href']}\t{f['dataset']}\t{f['output_type']}")
    for g in files[1:]:
        print(f"  (also {g['accession']} {g['output_type']} {g['dataset']})", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:])
