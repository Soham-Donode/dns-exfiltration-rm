"""
PCAP -> DNS feature extractor (CIC-Bell-DNS-EXF-2021 re-extraction)
====================================================================
Rebuilds a per-query dataset straight from the raw captures, because the released
CSVs lose the exfiltration traffic (the PCAPs were captured with snaplen=96, so every
DNS name is cut off after ~36 characters and the CSV extractor never sees the chunks).

What it does
------------
* Pure standard library parsing (classic pcap LE/BE, usec/nsec, and pcapng), Ethernet /
  VLAN / Linux-cooked / raw-IP link types, IPv4 + IPv6, UDP port 53 queries.
* Recovers the TRUE DNS message length from the UDP length field even when the packet
  was truncated by the capture (`dns_len`, `truncated`, `name_complete`).
* Stateless features of the paper's Table 1 (F17-F30) computed on the visible name,
  plus content-only ratios that do not depend on name length.
* Per-query label by query SHAPE (DNSExfiltrator "<seq>.<base64url>" chunks, "init"
  handshake, attacker domain) instead of by capture time, and the paper-style file label
  next to it so both can be compared.
* Window features (default 100 queries per source, non-overlapping) for the stateful
  layer. These are computed from the capture itself (no DNSDB / ASN / geo lookups).

Usage
-----
    python src/pcap_features.py PATH [PATH ...] --out-dir data/processed/pcap
    PATH = .pcap / .pcapng / .zip (nested zips ok) / directory

Outputs (in --out-dir): queries.csv, windows.csv, label_audit.csv
"""

import argparse
import csv
import io
import math
import re
import struct
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- pcap reading
_CLASSIC = {
    b"\xd4\xc3\xb2\xa1": ("<", 1e6),
    b"\xa1\xb2\xc3\xd4": (">", 1e6),
    b"\x4d\x3c\xb2\xa1": ("<", 1e9),
    b"\xa1\xb2\x3c\x4d": (">", 1e9),
}


def iter_packets(fh):
    """Yield (timestamp, linktype, data, orig_len) from a classic pcap or pcapng stream."""
    head = fh.read(4)
    if head == b"\x0a\x0d\x0d\x0a":
        yield from _iter_pcapng(fh, head)
        return
    if head not in _CLASSIC:
        raise ValueError(f"not a pcap/pcapng file (magic {head.hex()})")
    end, div = _CLASSIC[head]
    rest = fh.read(20)
    link = struct.unpack(end + "I", rest[16:20])[0]
    ph = struct.Struct(end + "IIII")
    read = fh.read
    while True:
        h = read(16)
        if len(h) < 16:
            return
        sec, frac, incl, orig = ph.unpack(h)
        data = read(incl)
        if len(data) < incl:
            return
        yield sec + frac / div, link, data, orig


def _iter_pcapng(fh, btype):
    end, ifaces = "<", []
    while True:
        if btype is None:
            btype = fh.read(4)
            if len(btype) < 4:
                return
        lenb = fh.read(4)
        if len(lenb) < 4:
            return
        if btype == b"\x0a\x0d\x0d\x0a":                       # section header
            bom = fh.read(4)
            end = "<" if bom == b"\x4d\x3c\x2b\x1a" else ">"
            fh.read(struct.unpack(end + "I", lenb)[0] - 12)
            ifaces, btype = [], None
            continue
        total = struct.unpack(end + "I", lenb)[0]
        body = fh.read(total - 12)
        fh.read(4)
        kind = struct.unpack(end + "I", btype)[0]
        btype = None
        if kind == 1 and len(body) >= 8:                        # interface description
            link, div, i = struct.unpack(end + "H", body[:2])[0], 1e6, 8
            while i + 4 <= len(body):
                code, olen = struct.unpack(end + "HH", body[i:i + 4])
                if code == 0:
                    break
                if code == 9 and olen >= 1:
                    b = body[i + 4]
                    div = float(2 ** (b & 0x7F)) if b & 0x80 else float(10 ** b)
                i += 4 + ((olen + 3) & ~3)
            ifaces.append((link, div))
        elif kind == 6 and len(body) >= 20:                     # enhanced packet
            ifc, hi, lo, cap, orig = struct.unpack(end + "IIIII", body[:20])
            link, div = ifaces[ifc] if ifc < len(ifaces) else (1, 1e6)
            yield ((hi << 32) | lo) / div, link, body[20:20 + cap], orig
        elif kind == 3 and len(body) >= 4:                      # simple packet
            orig = struct.unpack(end + "I", body[:4])[0]
            link = ifaces[0][0] if ifaces else 1
            yield 0.0, link, body[4:4 + min(orig, len(body) - 4)], orig


# --------------------------------------------------------------------------- L3 / L4 / DNS
def udp_info(link, d):
    """-> (src_ip, sport, dport, udp_len, captured_payload) for IPv4/IPv6 UDP, else None."""
    n = len(d)
    if link == 1:
        if n < 14:
            return None
        et, off = (d[12] << 8) | d[13], 14
        while et in (0x8100, 0x88A8, 0x9100):
            if n < off + 4:
                return None
            et, off = (d[off + 2] << 8) | d[off + 3], off + 4
    elif link == 113:
        if n < 16:
            return None
        et, off = (d[14] << 8) | d[15], 16
    elif link in (101, 12, 14):
        if n < 1:
            return None
        et, off = (0x0800 if d[0] >> 4 == 4 else 0x86DD), 0
    else:
        return None
    if et == 0x0800:
        if n < off + 20 or d[off + 9] != 17:
            return None
        if (((d[off + 6] & 0x1F) << 8) | d[off + 7]) != 0:      # non-first fragment
            return None
        src, l4 = ".".join(map(str, d[off + 12:off + 16])), off + (d[off] & 15) * 4
    elif et == 0x86DD:
        if n < off + 40:
            return None
        nh, l4, src = d[off + 6], off + 40, d[off + 8:off + 24].hex()
        while nh in (0, 43, 60):
            if n < l4 + 8:
                return None
            nh, l4 = d[l4], l4 + (d[l4 + 1] + 1) * 8
        if nh != 17:
            return None
    else:
        return None
    if n < l4 + 8:
        return None
    sport, dport, ulen = struct.unpack_from("!HHH", d, l4)
    return src, sport, dport, ulen, d[l4 + 8:l4 + ulen]


def parse_query(p):
    """Truncation-tolerant parse of the first question. -> (flags, labels, name_complete, qtype)"""
    if len(p) < 12:
        return None
    flags = (p[2] << 8) | p[3]
    if flags & 0x8000 or ((p[4] << 8) | p[5]) < 1:             # response, or no question
        return None
    i, n, labels, complete = 12, len(p), [], False
    while i < n:
        c = p[i]
        if c == 0:
            complete, i = True, i + 1
            break
        if c & 0xC0:
            break
        seg = p[i + 1:i + 1 + c]
        labels.append(seg.decode("latin-1"))
        if len(seg) < c:
            break
        i += 1 + c
    qtype = ((p[i] << 8) | p[i + 1]) if complete and i + 4 <= n else None
    return flags, labels, complete, qtype


# --------------------------------------------------------------------------- features
QTYPE = {1: "A", 2: "NS", 5: "CNAME", 6: "SOA", 10: "NULL", 12: "PTR", 15: "MX", 16: "TXT",
         28: "AAAA", 33: "SRV", 65: "HTTPS", 255: "ANY"}
SLD2 = {"co", "com", "org", "net", "gov", "edu", "ac", "or", "ne", "go"}
B64URL = re.compile(r"[A-Za-z0-9_-]+")
VOWELS = set("aeiouAEIOU")
ATTACK_TYPES = ["audio", "compressed", "exe", "image", "text", "video"]
ATTACKER_DOMAIN = "cicresearch"


def shannon(s):
    if not s:
        return 0.0
    c, n = Counter(s), len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def longest_alpha_run(s):
    best = cur = 0
    for ch in s:
        cur = cur + 1 if ch.isalpha() else 0
        best = max(best, cur)
    return best


def split_name(labels, complete):
    """-> (sld, subdomain_str, has_subdomain). Truncated names have no visible domain."""
    if not complete:
        return "<trunc>", ".".join(labels), 1
    suffix = 2 if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in SLD2 else 1
    idx = len(labels) - suffix - 1
    if idx < 0:
        return (labels[0] if labels else ""), "", 0
    return labels[idx], ".".join(labels[:idx]), int(idx > 0)


def exfil_shape(labels, name):
    """Label by query shape (DNSExfiltrator chunk / init handshake / attacker domain)."""
    if ATTACKER_DOMAIN in name.lower():
        return True
    if len(labels) >= 2:
        if labels[0].isdigit() and len(labels[1]) >= 20 and B64URL.fullmatch(labels[1]):
            return True
        if labels[0] == "init":
            return True
    return False


def query_features(labels, complete, qtype, dns_len, truncated):
    name = ".".join(labels)
    sld, sub, has_sub = split_name(labels, complete)
    chars = name.replace(".", "")
    nchar = max(len(chars), 1)
    lens = [len(x) for x in labels] or [0]
    upper = sum(c.isupper() for c in chars)
    lower = sum(c.islower() for c in chars)
    digit = sum(c.isdigit() for c in chars)
    special = len(chars) - upper - lower - digit
    # name length on the wire = len(".".join(labels)) + 2 ; DNS msg = 12 hdr + name + 4 (qtype/qclass)
    est = max(len(name), dns_len - 16) if truncated else len(name)
    return {
        "FQDN_count": len(name), "subdomain_length": len(sub), "upper": upper, "lower": lower,
        "numeric": digit, "entropy": round(shannon(name), 6), "special": special,
        "labels": len(labels), "labels_max": max(lens), "labels_average": round(sum(lens) / len(lens), 4),
        "longest_word": longest_alpha_run(chars), "sld": sld,
        "len": len(sld) + len(sub) + (1 if sub else 0), "subdomain": has_sub,
        # capture-aware extras
        "dns_len": dns_len, "fqdn_len_est": est, "truncated": int(truncated), "name_complete": int(complete),
        "qtype": QTYPE.get(qtype, "OTHER" if qtype is not None else "UNK"),
        # content-only (independent of name length)
        "upper_ratio": round(upper / nchar, 4), "lower_ratio": round(lower / nchar, 4),
        "digit_ratio": round(digit / nchar, 4), "special_ratio": round(special / nchar, 4),
        "vowel_ratio": round(sum(c in VOWELS for c in chars) / nchar, 4),
        "unique_char_ratio": round(len(set(chars)) / nchar, 4),
        "first_label_numeric": int(bool(labels) and labels[0].isdigit()),
    }


QUERY_FIELDS = [
    "file", "scenario", "file_kind", "attack_type", "pkt_index", "ts", "src",
    "label", "label_paper", "name",
    "FQDN_count", "subdomain_length", "upper", "lower", "numeric", "entropy", "special", "labels",
    "labels_max", "labels_average", "longest_word", "sld", "len", "subdomain",
    "dns_len", "fqdn_len_est", "truncated", "name_complete", "qtype",
    "upper_ratio", "lower_ratio", "digit_ratio", "special_ratio", "vowel_ratio",
    "unique_char_ratio", "first_label_numeric",
]


# --------------------------------------------------------------------------- windows
def window_rows(meta, recs, size):
    """recs: list of (ts, dns_len, entropy, qtype, name, sld, truncated, is_exfil, upper_ratio)"""
    out = []
    for w, start in enumerate(range(0, len(recs), size)):
        ch = recs[start:start + size]
        if len(ch) < max(size // 2, 2):
            continue                                            # drop a tiny tail window
        n = len(ch)
        ts = [r[0] for r in ch]
        lens = [r[1] for r in ch]
        mean_len = sum(lens) / n
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        dur = max(ts[-1] - ts[0], 1e-6)
        qt = Counter(r[3] for r in ch)
        exf = sum(r[7] for r in ch) / n
        row = dict(meta)
        row.update({
            "window": w, "n": n, "t_start": ts[0], "duration": round(dur, 6),
            "qps": round(n / dur, 6), "gap_mean": round(sum(gaps) / len(gaps), 6) if gaps else 0.0,
            "len_mean": round(mean_len, 4),
            "len_std": round(math.sqrt(sum((x - mean_len) ** 2 for x in lens) / n), 4),
            "len_max": max(lens), "entropy_mean": round(sum(r[2] for r in ch) / n, 6),
            "upper_ratio_mean": round(sum(r[8] for r in ch) / n, 6),
            "unique_name_ratio": round(len({r[4] for r in ch}) / n, 4),
            "unique_sld": len({r[5] for r in ch}),
            "truncated_frac": round(sum(r[6] for r in ch) / n, 4),
            "long_frac": round(sum(x > 100 for x in lens) / n, 4),
        })
        for q in ("A", "AAAA", "TXT", "NULL", "CNAME", "MX", "PTR"):
            row[f"frac_{q}"] = round(qt.get(q, 0) / n, 4)
        row["frac_other"] = round(1 - sum(row[f"frac_{q}"] for q in ("A", "AAAA", "TXT", "NULL", "CNAME", "MX", "PTR")), 4)
        row["exfil_frac"] = round(exf, 4)
        row["label"] = int(exf >= 0.5)
        out.append(row)
    return out


WINDOW_FIELDS = [
    "file", "scenario", "file_kind", "attack_type", "src", "window", "n", "t_start", "duration", "qps",
    "gap_mean", "len_mean", "len_std", "len_max", "entropy_mean", "upper_ratio_mean",
    "unique_name_ratio", "unique_sld", "truncated_frac", "long_frac",
    "frac_A", "frac_AAAA", "frac_TXT", "frac_NULL", "frac_CNAME", "frac_MX", "frac_PTR", "frac_other",
    "exfil_frac", "label",
]


# --------------------------------------------------------------------------- inputs
def _walk(path, shown=None):
    """Yield (display_path, opener) for every capture below path (dirs, zips, nested zips)."""
    path = Path(path) if not isinstance(path, Path) else path
    shown = shown or str(path)
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".pcap", ".pcapng", ".cap", ".zip"):
                yield from _walk(p)
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            yield from _walk_zip(zf, shown)
    else:
        yield shown, (lambda p=path: open(p, "rb"))


def _walk_zip(zf, shown):
    for info in zf.infolist():
        if info.is_dir():
            continue
        low = info.filename.lower()
        disp = f"{shown}/{info.filename}"
        if low.endswith((".pcap", ".pcapng", ".cap")):
            yield disp, (lambda z=zf, i=info: z.open(i))
        elif low.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(zf.read(info))) as inner:
                yield from _walk_zip(inner, disp)


def classify_file(display):
    """-> (scenario, file_kind, attack_type). Kind comes only from the last two path parts
    (folders like 'Attack_Light_Benign' contain both words)."""
    parts = display.replace("\\", "/").split("/")
    tail = "/".join(parts[-2:]).lower()
    base = parts[-1].lower()
    kind = "benign" if "benign" in tail else "attack"
    full = display.lower()
    scenario = "light" if "light" in full else "heavy" if "heavy" in full else "unknown"
    atype = "benign" if kind == "benign" else next((t for t in ATTACK_TYPES if t in base), "unknown")
    return scenario, kind, atype


# --------------------------------------------------------------------------- main routine
def extract_file(display, opener, qwriter, ports=(53,), window=100, scenario_override=None, max_packets=None):
    scenario, kind, atype = classify_file(display)
    if scenario_override:
        scenario = scenario_override
    meta = {"file": display.replace("\\", "/").split("/")[-1], "scenario": scenario,
            "file_kind": kind, "attack_type": atype}
    per_src, stats = defaultdict(list), Counter()
    with opener() as fh:
        for idx, (ts, link, data, _orig) in enumerate(iter_packets(fh)):
            if max_packets and idx >= max_packets:
                break
            stats["packets"] += 1
            u = udp_info(link, data)
            if not u or u[2] not in ports:
                continue
            src, _sp, _dp, ulen, payload = u
            q = parse_query(payload)
            if not q:
                continue
            _flags, labels, complete, qtype = q
            dns_len = max(ulen - 8, len(payload))
            truncated = len(payload) < ulen - 8
            name = ".".join(labels)
            f = query_features(labels, complete, qtype, dns_len, truncated)
            exfil = int(kind == "attack" and exfil_shape(labels, name))
            row = dict(meta)
            row.update(f)
            row.update({"pkt_index": idx, "ts": round(ts, 6), "src": src, "name": name,
                        "label": exfil, "label_paper": int(kind == "attack")})
            qwriter.writerow(row)
            per_src[src].append((ts, dns_len, f["entropy"], f["qtype"], name, f["sld"],
                                 f["truncated"], exfil, f["upper_ratio"]))
            stats["queries"] += 1
            stats["exfil_shape"] += exfil
            stats["long_msg"] += dns_len > 100
            stats["long_msg_not_labelled"] += (dns_len > 100) and not exfil
            stats["shape_but_short"] += bool(exfil) and dns_len <= 100
            stats["rule_hits_in_benign_file"] += int(kind == "benign" and exfil_shape(labels, name))
            stats["truncated"] += int(truncated)
    wins = []
    for src, recs in per_src.items():
        recs.sort(key=lambda r: r[0])
        m = dict(meta, src=src)
        wins.extend(window_rows(m, recs, window))
    stats = {**dict(stats), **meta}                             # Counter.update() would add, not set
    return stats, wins


def run(inputs, out_dir="data/processed/pcap", window=100, scenario=None, ports=(53,), max_packets=None, verbose=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit, n_files = [], 0
    with open(out / "queries.csv", "w", newline="", encoding="utf-8") as qf, \
         open(out / "windows.csv", "w", newline="", encoding="utf-8") as wf:
        qw = csv.DictWriter(qf, fieldnames=QUERY_FIELDS)
        ww = csv.DictWriter(wf, fieldnames=WINDOW_FIELDS)
        qw.writeheader()
        ww.writeheader()
        for inp in inputs:
            for display, opener in _walk(inp):
                n_files += 1
                stats, wins = extract_file(display, opener, qw, ports, window, scenario, max_packets)
                ww.writerows(wins)
                audit.append(stats)
                if verbose:
                    print(f"  {stats['file']:28s} {stats['scenario']:7s} {stats['file_kind']:7s} "
                          f"pkts {stats['packets']:>9,} queries {stats['queries']:>8,} "
                          f"exfil-shape {stats['exfil_shape']:>7,} windows {len(wins):>5,}")
    cols = ["file", "scenario", "file_kind", "attack_type", "packets", "queries", "exfil_shape", "long_msg",
            "long_msg_not_labelled", "shape_but_short", "rule_hits_in_benign_file", "truncated"]
    with open(out / "label_audit.csv", "w", newline="", encoding="utf-8") as af:
        aw = csv.DictWriter(af, fieldnames=cols, extrasaction="ignore")
        aw.writeheader()
        for s in audit:
            aw.writerow({c: s.get(c, 0) for c in cols})
    if verbose:
        print(f"[OK] {n_files} capture(s) -> {out}/queries.csv, windows.csv, label_audit.csv")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help=".pcap/.pcapng/.zip files or directories")
    ap.add_argument("--out-dir", default="data/processed/pcap")
    ap.add_argument("--window", type=int, default=100, help="queries per window (paper: 100)")
    ap.add_argument("--scenario", choices=["light", "heavy"], help="force scenario if not in the path")
    ap.add_argument("--ports", type=int, nargs="+", default=[53])
    ap.add_argument("--max-packets", type=int, help="debug: stop after N packets per file")
    a = ap.parse_args()
    run(a.inputs, a.out_dir, a.window, a.scenario, tuple(a.ports), a.max_packets)


if __name__ == "__main__":
    sys.exit(main())
