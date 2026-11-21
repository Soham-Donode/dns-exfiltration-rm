"""
Synthetic test traffic: evasive exfiltration + hard benign DNS
================================================================
The real CIC-Bell-DNS-EXF-2021 traffic is one tool (DNSExfiltrator) with one fixed shape and
benign traffic that is plain website lookups, so every classifier scores ~100% on it. This module
builds the two missing test sets as classic PCAPs (+ a labels sidecar), so they go through exactly
the same extractor (`pcap_features.py`) as the real captures:

1. EVASIVE EXFILTRATION  - the project's DNSExfiltrationSimulator (base32 / hex / base64, label
   length 12-63) plus variants it cannot produce: no sequence tag, word-encoding (low entropy),
   TXT / NULL / AAAA query types.
2. HARD BENIGN           - legitimate names that LOOK like exfiltration: CDN and cloud hostnames,
   hashed / GUID labels, telemetry, DKIM / DNSBL / PTR, DNS-0x20 case randomisation, punycode.
3. MIXED STREAMS         - one host's normal traffic with ONE attacker (one family, one attacker
   domain) diluted in at 1-50% of its queries, plus a clean reference stream (0%).
Pure evasive streams use three timings (fast 0.05 s, medium 0.5 s, slow 5 s mean gap), so a model
cannot rely on "fast = attack".

IMPORTANT: everything here is SYNTHETIC. The hard-benign patterns are approximations of real
naming schemes, not captured traffic; treat false-positive rates on them as indicative only.

    python src/evasion_data.py --out-dir data/processed/synth [--domains-file domains.txt]
"""

import argparse
import csv
import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dns_simulator import DNSExfiltrationSimulator  # noqa: E402

QTYPES = {"A": 1, "NS": 2, "CNAME": 5, "NULL": 10, "PTR": 12, "MX": 15, "TXT": 16, "AAAA": 28, "HTTPS": 65}

# name: (encoding, max label length, sequence tag, qtype)
FAMILIES = {
    "b32_l63_seq":        ("base32", 63, True, "A"),
    "b32_l32_seq":        ("base32", 32, True, "A"),
    "b32_l16_seq":        ("base32", 16, True, "A"),
    "b32_l20_noseq":      ("base32", 20, False, "A"),
    "b32_l12_noseq_null": ("base32", 12, False, "NULL"),
    "b32_l32_txt":        ("base32", 32, True, "TXT"),
    "hex_l63_seq":        ("hex", 63, True, "A"),
    "hex_l32_noseq":      ("hex", 32, False, "AAAA"),
    "hex_l16_seq":        ("hex", 16, True, "A"),
    "b64_l63_txt":        ("base64", 63, True, "TXT"),
    "b64_l24_noseq":      ("base64", 24, False, "A"),
    "words":              ("words", 0, False, "A"),
}
ATTACKER_DOMAINS = ["cloud-sync.net", "upd-cdn.org", "telemetry-hub.com", "static-assets.io",
                    "mailgate.biz", "api-edge.co", "update.fast-dl.com", "sync.pixelhost.net"]
WORDS = ("amber apple arrow beach bell berry birch blue bone brave bread brick brook cabin candy cedar chair "
         "cliff cloud clover coast coral cotton crane daisy delta dune eagle ember fern field flame forest "
         "frost garden gold grain grape green harbor hazel honey iron island ivory jade lemon lilac maple "
         "meadow mint moon moss ocean olive pearl pine plum river rose sage sand silver stone sun tulip "
         "willow wind").split()
assert len(WORDS) >= 64
WORDS = WORDS[:64]

MIXED_FAMILIES = tuple(FAMILIES)                                  # every family is tested diluted
PURE_GAPS = (0.05, 0.5, 5.0)

FALLBACK_DOMAINS = ["wikipedia.org", "github.com", "bbc.co.uk", "nytimes.com", "stackoverflow.com", "reddit.com",
                    "amazon.com", "netflix.com", "apple.com", "microsoft.com", "yahoo.com", "ebay.com",
                    "cnn.com", "twitch.tv", "spotify.com", "dropbox.com", "slack.com", "zoom.us", "mozilla.org",
                    "adobe.com", "paypal.com", "linkedin.com", "weather.com", "imdb.com"]


# --------------------------------------------------------------------------- exfiltration
def _words_queries(n, domain, rng):
    out = []
    for _ in range(n):
        k = rng.randint(6, 9)
        out.append("-".join(rng.choice(WORDS) for _ in range(k)) + "." + domain)
    return out


def exfil_family(family, n, rng):
    """-> list of (qname, qtype_name) for one evasive exfiltration family."""
    enc, label_len, seq, qtype = FAMILIES[family]
    domain = rng.choice(ATTACKER_DOMAINS)
    if enc == "words":
        return [(q, qtype) for q in _words_queries(n, domain, rng)]
    sim = DNSExfiltrationSimulator(target_domain=domain, encoding=enc, max_label_length=label_len)
    qs = sim.generate_queries(rng.randbytes(n * label_len))[:n]
    if not seq:
        qs = [q.split(".", 1)[1] for q in qs]                   # drop the "sq0001" tag
    return [(q, qtype) for q in qs]


# --------------------------------------------------------------------------- hard benign
_B36 = "abcdefghijklmnopqrstuvwxyz0123456789"


def _b36(r, n):
    return "".join(r.choice(_B36) for _ in range(n))


def _hex(r, n):
    return "".join(r.choice("0123456789abcdef") for _ in range(n))


def _uuid(r):
    h = _hex(r, 32)
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def _case0x20(name, r):
    return "".join(c.upper() if (c.isalpha() and r.random() < 0.5) else c for c in name)


def hard_benign(n, rng, domains=None):
    """-> list of (qname, qtype_name) of legitimate names that resemble exfiltration."""
    domains = domains or FALLBACK_DOMAINS
    r = rng

    def dom():
        return r.choice(domains)

    patterns = [
        ("cloudfront", lambda: (f"d{_b36(r, 13)}.cloudfront.net", "A")),
        ("akamai", lambda: (f"e{r.randint(1000, 99999)}.dscb.akamaiedge.net", "A")),
        ("googlevideo", lambda: (f"r{r.randint(1, 9)}---sn-{_b36(r, 8)}.googlevideo.com", "A")),
        ("ec2", lambda: (f"ec2-{r.randint(1, 223)}-{r.randint(0, 255)}-{r.randint(0, 255)}-{r.randint(1, 254)}.compute-1.amazonaws.com", "A")),
        ("execute-api", lambda: (f"{_b36(r, 10)}.execute-api.us-east-1.amazonaws.com", "A")),
        ("azure-blob", lambda: (f"{_b36(r, 24)}.blob.core.windows.net", "A")),
        ("azure-app", lambda: (f"{_b36(r, 12)}.cloudapp.azure.com", "AAAA")),
        ("sentry", lambda: (f"o{r.randint(100000, 999999)}.ingest.sentry.io", "A")),
        ("telemetry-guid", lambda: (f"{_uuid(r)}.events.data.microsoft.com", "A")),
        ("hash-reputation", lambda: (f"{_hex(r, 40)}.rep.example-av.net", "A")),
        ("cdn-hash-sub", lambda: (f"{_hex(r, 16)}.{dom()}", "A")),
        ("dkim", lambda: (f"s{r.randint(1, 2)}._domainkey.{dom()}", "TXT")),
        ("dnsbl", lambda: (f"{r.randint(1, 254)}.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}.zen.spamhaus.org", "A")),
        ("ptr", lambda: (f"{r.randint(1, 254)}.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}.in-addr.arpa", "PTR")),
        ("case0x20", lambda: (_case0x20(("www." if r.random() < 0.6 else "") + dom(), r), "A")),
        ("punycode", lambda: (f"xn--{_b36(r, 8)}-{_b36(r, 3)}.com", "A")),
        ("hyphen-words", lambda: ("-".join(r.choice(WORDS) for _ in range(r.randint(3, 5))) + "." + dom(), "A")),
        ("tracking", lambda: (f"{_b36(r, 20)}.click.{dom()}", "A")),
    ]
    return [patterns[r.randrange(len(patterns))][1]() for _ in range(n)]


def plain_benign(n, rng, domains=None):
    domains = domains or FALLBACK_DOMAINS
    return [((rng.choice(["www.", "", "cdn.", "mail.", "api."]) + rng.choice(domains)), rng.choice(["A", "A", "A", "AAAA"]))
            for _ in range(n)]


# --------------------------------------------------------------------------- pcap writing
def _dns_wire(name, qtype):
    labels = name.split(".")
    assert all(0 < len(l) <= 63 for l in labels) and len(name) <= 253, f"invalid DNS name: {name!r}"
    wire = b"".join(bytes([len(l)]) + l.encode("ascii") for l in labels) + b"\x00"
    return struct.pack("!HHHHHH", random.getrandbits(16), 0x0100, 1, 0, 0, 0) + wire + struct.pack("!HH", QTYPES[qtype], 1)


def _frame(payload, src_ip):
    udp = struct.pack("!HHHH", random.randint(1025, 65000), 53, 8 + len(payload), 0) + payload
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(udp), random.getrandbits(16), 0, 64, 17, 0,
                     bytes(src_ip), bytes([8, 8, 8, 8])) + udp
    return b"\xaa" * 6 + b"\xbb" * 6 + b"\x08\x00" + ip


def write_stream(path, events, src_ip=(192, 168, 50, 10), snaplen=96):
    """events: list of (ts, name, qtype_name, label, family). Writes path (.pcap) and path.labels.csv."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    events = sorted(events, key=lambda e: e[0])
    out = bytearray(b"\xd4\xc3\xb2\xa1" + struct.pack("<HHiIII", 2, 4, 0, 0, snaplen, 1))
    for ts, name, qtype, _label, _fam in events:
        fr = _frame(_dns_wire(name, qtype), src_ip)
        cap = fr[:snaplen]
        out += struct.pack("<IIII", int(ts), int((ts % 1) * 1e6), len(cap), len(fr)) + cap
    path.write_bytes(bytes(out))
    with open(path.with_suffix(".labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pkt_index", "label", "family", "name", "qtype"])
        for i, (_ts, name, qtype, label, fam) in enumerate(events):
            w.writerow([i, label, fam, name, qtype])
    return path


def _timeline(n, mean_gap, rng, t0=1_700_000_000.0):
    t, out = t0, []
    for _ in range(n):
        t += rng.expovariate(1.0 / mean_gap)
        out.append(t)
    return out


# --------------------------------------------------------------------------- generate
def generate(out_dir, domains=None, seed=7, n_family=1500, n_hard=8000, n_mixed=6000,
             mixed_rates=(0.01, 0.05, 0.2, 0.5), mixed_families=MIXED_FAMILIES, snaplen=96,
             train_mixed_rates=(0.05, 0.2), n_train_mixed=3000):
    """Write all synthetic captures under out_dir. Returns a list of manifest dicts
    (kind, file, family, rate, mean_gap, queries, exfil).

    kind = evasive | hard_benign | mixed (test streams) | train_mixed (diluted streams for EVERY
    family, meant for augmented training; generated last so the test streams do not depend on them)."""
    rng = random.Random(seed)
    random.seed(seed)
    out, manifest = Path(out_dir), []
    domains = domains or FALLBACK_DOMAINS

    def add(kind, path, events, family="", rate=0.0, gap=0.0):
        write_stream(path, events, src_ip=(192, 168, 50, 10 + len(manifest) % 240), snaplen=snaplen)
        manifest.append({"kind": kind, "file": Path(path).name, "family": family, "rate": rate, "mean_gap": gap,
                         "queries": len(events), "exfil": sum(e[3] for e in events)})

    for k, fam in enumerate(FAMILIES):                           # pure evasive streams, varied timing
        gap = PURE_GAPS[k % len(PURE_GAPS)]
        qs = exfil_family(fam, n_family, rng)
        ts = _timeline(len(qs), gap, rng)
        add("evasive", out / "evasive" / f"ev_{fam}.pcap", [(t, q, qt, 1, fam) for t, (q, qt) in zip(ts, qs)], fam, 1.0, gap)

    hb = hard_benign(n_hard, rng, domains)                       # hard benign only
    ts = _timeline(len(hb), 0.5, rng)
    add("hard_benign", out / "hardneg" / "hard_benign.pcap",
        [(t, q, qt, 0, "hard_benign") for t, (q, qt) in zip(ts, hb)], "hard_benign", 0.0, 0.5)

    def background(n):
        plain, hard = plain_benign(n, rng, domains), hard_benign(n, rng, domains)
        return [hard[i] if rng.random() < 0.3 else plain[i] for i in range(n)]

    ts = _timeline(n_mixed, 0.5, rng)                            # clean reference stream (rate 0)
    add("mixed", out / "mixed" / "mixed_none_r00.pcap",
        [(t, q, qt, 0, "background") for t, (q, qt) in zip(ts, background(n_mixed))], "none", 0.0, 0.5)

    def mixed_events(fam, rate, n):
        ts, bg = _timeline(n, 0.5, rng), background(n)
        pool = exfil_family(fam, int(n * rate * 1.5) + 10, rng)       # one call = one attacker domain
        events = []
        for i, t in enumerate(ts):
            if rng.random() < rate and pool:
                q, qt = pool.pop(0)
                events.append((t, q, qt, 1, fam))
            else:
                q, qt = bg[i]
                events.append((t, q, qt, 0, "background"))
        return events

    for fam in mixed_families:                                   # one attacker per stream
        for rate in mixed_rates:
            add("mixed", out / "mixed" / f"mixed_{fam}_r{int(round(rate * 100)):02d}.pcap",
                mixed_events(fam, rate, n_mixed), fam, rate, 0.5)

    for fam in FAMILIES:                                         # training-only diluted streams
        for rate in train_mixed_rates:
            add("train_mixed", out / "train_mixed" / f"trainmix_{fam}_r{int(round(rate * 100)):02d}.pcap",
                mixed_events(fam, rate, n_train_mixed), fam, rate, 0.5)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/processed/synth")
    ap.add_argument("--domains-file", help="text file with one benign base domain per line")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--snaplen", type=int, default=96, help="96 mimics the real captures; 65535 = full packets")
    a = ap.parse_args()
    doms = [l.strip() for l in open(a.domains_file)] if a.domains_file else None
    for m in generate(a.out_dir, doms, a.seed, snaplen=a.snaplen):
        print(f"  {m['kind']:12s} {m['file']:34s} queries {m['queries']:>6,} exfil {m['exfil']:>6,}")
    print(f"[OK] wrote captures + .labels.csv sidecars to {a.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
