"""Tests for src/pcap_features.py using synthetic captures (no real data needed).

Run:  python -m unittest tests/test_pcap_features.py
"""
import csv
import io
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pcap_features as pf  # noqa: E402


def dns_query(name, qtype=1, response=False):
    wire = b"".join(bytes([len(l)]) + l.encode() for l in name.split(".")) + b"\x00"
    return struct.pack("!HHHHHH", 0x1234, 0x8180 if response else 0x0100, 1, 0, 0, 0) + wire + struct.pack("!HH", qtype, 1)


def eth_ipv4_udp(payload, sport=40000, dport=53, vlan=False, src=(192, 168, 0, 7)):
    udp = struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(udp), 1, 0, 64, 17, 0, bytes(src), bytes([8, 8, 8, 8])) + udp
    eth = b"\xaa" * 6 + b"\xbb" * 6
    return eth + (b"\x81\x00\x00\x05" if vlan else b"") + b"\x08\x00" + ip


def eth_ipv6_udp(payload, dport=53):
    udp = struct.pack("!HHHH", 40000, dport, 8 + len(payload), 0) + payload
    ip6 = struct.pack("!IHBB16s16s", 0x60000000, len(udp), 17, 64, b"\x20\x01" + b"\x00" * 14, b"\x20\x02" + b"\x00" * 14) + udp
    return b"\xaa" * 6 + b"\xbb" * 6 + b"\x86\xdd" + ip6


def classic_pcap(frames, snaplen=65535, big_endian=False, nano=False):
    e = ">" if big_endian else "<"
    magic = {(False, False): b"\xd4\xc3\xb2\xa1", (True, False): b"\xa1\xb2\xc3\xd4",
             (False, True): b"\x4d\x3c\xb2\xa1", (True, True): b"\xa1\xb2\x3c\x4d"}[(big_endian, nano)]
    out = magic + struct.pack(e + "HHiIII", 2, 4, 0, 0, snaplen, 1)
    for i, f in enumerate(frames):
        cap = f[:snaplen]
        out += struct.pack(e + "IIII", 1000 + i, 500000 if not nano else 500000000, len(cap), len(f)) + cap
    return out


def pcapng(frames):
    def block(t, body):
        pad = (-len(body)) % 4
        total = 12 + len(body) + pad
        return struct.pack("<II", t, total) + body + b"\x00" * pad + struct.pack("<I", total)
    out = block(0x0A0D0D0A, struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1))
    out += block(1, struct.pack("<HHI", 1, 0, 0))
    for i, f in enumerate(frames):
        ts = 1_700_000_000_000_000 + i * 1000
        out += block(6, struct.pack("<IIIII", 0, ts >> 32, ts & 0xFFFFFFFF, len(f), len(f)) + f)
    return out


_L = "p2Zf2ooUVhW3dRbJATgHoMssqgCXpgQ3XCGDV-abcdefghijklmn_pqrstuvwxyz0123456789"[:60]   # DNS labels are <= 63 chars
CHUNK = f"448.{_L}.{_L}.{_L}.cicresearch.ca"                       # ~200+ byte query like DNSExfiltrator


class TestParsing(unittest.TestCase):
    def packets(self, blob):
        return list(pf.iter_packets(io.BytesIO(blob)))

    def test_classic_all_endianness_and_resolution(self):
        frame = eth_ipv4_udp(dns_query("example.com"))
        for be in (False, True):
            for nano in (False, True):
                pk = self.packets(classic_pcap([frame], big_endian=be, nano=nano))
                self.assertEqual(len(pk), 1)
                self.assertAlmostEqual(pk[0][0], 1000.5, places=3)

    def test_pcapng(self):
        frames = [eth_ipv4_udp(dns_query("a.example.com")), eth_ipv4_udp(dns_query("b.example.com"))]
        pk = self.packets(pcapng(frames))
        self.assertEqual(len(pk), 2)
        self.assertAlmostEqual(pk[1][0] - pk[0][0], 0.001, places=6)
        self.assertEqual(pk[0][2], frames[0])

    def test_plain_query_vlan_and_ipv6(self):
        for frame in (eth_ipv4_udp(dns_query("www.example.com")),
                      eth_ipv4_udp(dns_query("www.example.com"), vlan=True),
                      eth_ipv6_udp(dns_query("www.example.com"))):
            _, link, data, _ = self.packets(classic_pcap([frame]))[0]
            src, _sp, dport, ulen, payload = pf.udp_info(link, data)
            self.assertEqual(dport, 53)
            _fl, labels, complete, qtype = pf.parse_query(payload)
            self.assertEqual(labels, ["www", "example", "com"])
            self.assertTrue(complete)
            self.assertEqual(qtype, 1)

    def test_responses_and_other_ports_ignored(self):
        self.assertIsNone(pf.parse_query(dns_query("example.com", response=True)))
        self.assertIsNone(pf.udp_info(1, eth_ipv4_udp(b"x" * 20, dport=123)[:10]))

    def test_truncated_capture_recovers_true_length(self):
        frame = eth_ipv4_udp(dns_query(CHUNK))
        _, link, data, orig = self.packets(classic_pcap([frame], snaplen=96))[0]
        self.assertEqual(len(data), 96)
        self.assertGreater(orig, 96)
        _src, _sp, _dp, ulen, payload = pf.udp_info(link, data)
        _fl, labels, complete, _qt = pf.parse_query(payload)
        self.assertFalse(complete)
        self.assertEqual(ulen - 8, len(dns_query(CHUNK)))           # true DNS length survives
        self.assertTrue(len(payload) < ulen - 8)
        self.assertEqual(labels[0], "448")
        feats = pf.query_features(labels, complete, None, ulen - 8, True)
        self.assertEqual(feats["sld"], "<trunc>")
        self.assertEqual(feats["truncated"], 1)
        self.assertGreaterEqual(feats["fqdn_len_est"], len(CHUNK) - 5)   # EDNS/rounding slack only


class TestFeatures(unittest.TestCase):
    def test_name_features(self):
        f = pf.query_features(["www", "Ab1-x", "example", "com"], True, 1, 40, False)
        self.assertEqual(f["FQDN_count"], len("www.Ab1-x.example.com"))
        self.assertEqual((f["upper"], f["numeric"], f["special"]), (1, 1, 1))
        self.assertEqual(f["labels"], 4)
        self.assertEqual(f["labels_max"], 7)
        self.assertEqual(f["sld"], "example")
        self.assertEqual(f["subdomain_length"], len("www.Ab1-x"))
        self.assertEqual(f["subdomain"], 1)
        self.assertEqual(f["qtype"], "A")

    def test_cc_sld_and_bare_domain(self):
        self.assertEqual(pf.split_name(["shop", "bbc", "co", "uk"], True), ("bbc", "shop", 1))
        self.assertEqual(pf.split_name(["google", "com"], True), ("google", "", 0))

    def test_shannon(self):
        self.assertAlmostEqual(pf.shannon("aabb"), 1.0)
        self.assertEqual(pf.shannon(""), 0.0)

    def test_exfil_shape(self):
        ex = lambda n: pf.exfil_shape(n.split("."), n)
        self.assertTrue(ex("448.p2Zf2ooUVhW3dRbJATgHoMssqgCXpg"))
        self.assertTrue(ex("init.MF2WI2LPL4ZDAMJYFUYDCLJRGB"))
        self.assertTrue(ex("x.cicresearch.ca"))
        self.assertFalse(ex("www.bing.com"))
        self.assertFalse(ex("4.3.2.1.in-addr.arpa"))                # digits but short 2nd label
        self.assertFalse(ex("v10.events.data.microsoft.com"))

    def test_classify_file(self):
        self.assertEqual(pf.classify_file("D:/PCAP_Light/Attacks.zip/Attacks/light_audio.pcap"), ("light", "attack", "audio"))
        self.assertEqual(pf.classify_file("D:/PCAP_Light/Benign.zip/Benign/benign.pcap")[1], "benign")
        # folder 'Attack_heavy_Benign' must not turn an attack file into benign
        self.assertEqual(pf.classify_file("Attack_heavy_Benign/Attacks.zip/Attacks/heavy_exe.pcap"), ("heavy", "attack", "exe"))


class TestEndToEnd(unittest.TestCase):
    @staticmethod
    def read(path):
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def build(self, tmp):
        attack = [eth_ipv4_udp(dns_query("www.bing.com"))]
        attack += [eth_ipv4_udp(dns_query(f"{i}.{CHUNK.split('.', 1)[1]}")) for i in range(1, 11)]  # seq 1..10
        attack += [eth_ipv4_udp(dns_query("init.MF2WI2LPL4ZDAMJYFUYDCLJRGBPTCMJNGU2C.base64.cicresearch.ca"))]
        benign = [eth_ipv4_udp(dns_query(f"site{i}.example.org")) for i in range(30)]
        zp = tmp / "PCAP_Light.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("Attacks/light_text.pcap", classic_pcap(attack, snaplen=96))
            zf.writestr("Benign/benign.pcap", classic_pcap(benign, snaplen=96))
        return zp

    def test_run_over_zip(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            out = pf.run([self.build(t)], out_dir=t / "out", window=10, verbose=False)
            rows = self.read(out / "queries.csv")
            self.assertEqual(len(rows), 42)
            atk = [r for r in rows if r["file_kind"] == "attack"]
            self.assertEqual(len(atk), 12)
            self.assertEqual(sum(int(r["label"]) for r in atk), 11)         # all but bing
            self.assertTrue(all(r["label_paper"] == "1" for r in atk))      # paper-style: whole file
            self.assertTrue(all(r["label"] == "0" for r in rows if r["file_kind"] == "benign"))
            chunks = [r for r in atk if r["label"] == "1" and r["first_label_numeric"] == "1"]
            self.assertTrue(all(int(r["dns_len"]) > 100 and r["truncated"] == "1" for r in chunks))
            self.assertTrue(all(r["scenario"] == "light" and r["attack_type"] == "text" for r in atk))
            audit = self.read(out / "label_audit.csv")
            a = next(x for x in audit if x["file_kind"] == "attack")
            self.assertEqual(a["long_msg_not_labelled"], "0")
            self.assertEqual(a["shape_but_short"], "1")                      # init handshake is short
            self.assertEqual(next(x for x in audit if x["file_kind"] == "benign")["rule_hits_in_benign_file"], "0")
            wins = self.read(out / "windows.csv")
            self.assertEqual(len(wins), 4)                                   # 12 attack -> 1 window, 30 benign -> 3
            self.assertEqual(sum(int(w["label"]) for w in wins), 1)

    def test_nested_zip_and_directory(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            zp = self.build(t)
            outer = t / "data" / "wrapper.zip"
            outer.parent.mkdir()
            with zipfile.ZipFile(outer, "w") as zf:
                zf.write(zp, "PCAP_Light.zip")
            found = [d for d, _ in pf._walk(t / "data")]
            self.assertEqual(len(found), 2)
            self.assertTrue(all(d.endswith(".pcap") for d in found))


if __name__ == "__main__":
    unittest.main()
