"""Read fibertools / FiberHMM nucleosome and footprint annotations into BAM-record coordinates.

The Molecular-annotation tags (fibertools-rs >= 0.10 `Ma`, FiberHMM `MA`) store
'start-length' with a 1-based start, in the *molecular* frame (the orientation the
molecule was sequenced in). pysam's query_sequence is in reference orientation, so for
reverse-strand alignments the intervals must be mirrored. Legacy `ns/nl` tags are
treated the same way.
"""


def _to_query(intervals, L, is_reverse):
    if not is_reverse:
        return intervals
    return sorted((L - b, L - a) for a, b in intervals)


def ma_frame(header):
    """Frame of FiberHMM's MA tags: 'molecular' only when the header declares coord=molecular."""
    h = header.to_dict() if hasattr(header, "to_dict") else dict(header or {})
    texts = [str(c) for c in h.get("CO", [])] + [" ".join(str(v) for v in pg.values()) for pg in h.get("PG", [])]
    return "molecular" if any("coord=molecular" in t.lower() for t in texts) else "seq"


def parse_annotations(rd, features=("nuc",), tag_order=("MA", "Ma"), MA_frame="seq"):
    """Return {feature: [(start, end), ...]} in query (BAM record) coordinates, 0-based half-open.

    fibertools `Ma` is always molecular frame; FiberHMM `MA` follows ``MA_frame``
    (see :func:`ma_frame`)."""
    L = rd.query_length or len(rd.query_sequence or "")
    out = {f: [] for f in features}
    for tag in tag_order:
        if not rd.has_tag(tag):
            continue
        for part in str(rd.get_tag(tag)).split(";")[1:]:
            if ":" not in part:
                continue
            head, data = part.split(":", 1)
            name = head.rstrip("+-.QqLlRr")  # 'nuc.Q' -> 'nuc', 'tf.QQQ' -> 'tf'
            for i, ch in enumerate(head):
                if ch in "+-.":
                    name = head[:i]
                    break
            if name not in out:
                continue
            for tok in filter(None, data.split(",")):
                s, l = tok.split("-", 1)
                s0 = int(s) - 1
                out[name].append((s0, s0 + int(l)))
        flip = rd.is_reverse and (tag == "Ma" or MA_frame == "molecular")
        for f in out:
            out[f] = _to_query(sorted(out[f]), L, flip)
        return out
    if "nuc" in out and rd.has_tag("ns"):
        iv = [(int(s), int(s) + int(l)) for s, l in zip(rd.get_tag("ns"), rd.get_tag("nl"))]
        out["nuc"] = _to_query(iv, L, rd.is_reverse)
    return out
