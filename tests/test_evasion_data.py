"""Tests for src/evasion_data.py.   Run:  python -m unittest tests/test_evasion_data.py"""
import csv
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evasion_data as ev  # noqa: E402
import pcap_features as pf  # noqa: E402


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestGenerators(unittest.TestCase):
    def test_every_family_is_rfc_valid_and_distinct(self):
        rng = random.Random(1)
        seen = set()
        for fam, (enc, label_len, seq, qtype) in ev.FAMILIES.items():
            qs = ev.exfil_family(fam, 50, rng)
            self.assertEqual(len(qs), 50, fam)
            for name, qt in qs:
                self.assertEqual(qt, qtype)
                self.assertLessEqual(len(name), 253)
                self.assertTrue(all(0 < len(l) <= 63 for l in name.split(".")), name)
            first = qs[0][0].split(".")
            if label_len:
                self.assertLessEqual(max(len(l) for l in first[:-2] or first), label_len, fam)
            self.assertEqual(first[0].startswith("sq"), seq, fam)
            seen.add(qs[0][0].split(".", 1)[-1])
        self.assertGreater(len(seen), 1)                          # several attacker domains in use

    def test_hard_benign_valid_and_never_matches_exfil_shape(self):
        names = ev.hard_benign(2000, random.Random(2))
        for n, qt in names:
            self.assertTrue(all(0 < len(l) <= 63 for l in n.split(".")), n)
            self.assertIn(qt, ev.QTYPES)
            self.assertFalse(pf.exfil_shape(n.split("."), n), n)
        self.assertGreater(len({n.split(".")[-2] if len(n.split(".")) > 1 else n for n, _ in names}), 10)

    def test_case_randomisation_present(self):
        names = [n for n, _ in ev.hard_benign(3000, random.Random(3))]
        self.assertTrue(any(any(c.isupper() for c in n) for n in names))


class TestStreams(unittest.TestCase):
    def test_generate_and_extract_round_trip(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            man = ev.generate(t / "synth", n_family=40, n_hard=200, n_mixed=400, mixed_rates=(0.2,),
                              mixed_families=("b32_l16_seq", "hex_l16_seq"), train_mixed_rates=(0.2,), n_train_mixed=100)
            self.assertEqual(len(man), 2 * len(ev.FAMILIES) + 1 + 1 + 2)      # pure + hard + clean + mixed + train_mixed
            self.assertEqual(sum(m["kind"] == "train_mixed" for m in man), len(ev.FAMILIES))
            pf.run([t / "synth"], out_dir=t / "feat", window=100, scenario="synthetic", verbose=False)
            q = read(t / "feat" / "queries.csv")
            self.assertEqual(len(q), sum(m["queries"] for m in man))
            labels = {}
            for p in (t / "synth").rglob("*.labels.csv"):
                for r in read(p):
                    labels[(p.name.replace(".labels.csv", ".pcap"), int(r["pkt_index"]))] = (int(r["label"]), r["name"], r["family"])
            # every extracted query maps to a sidecar row; short names survive intact, long ones are cut by snaplen 96
            for row in q:
                lab, name, _fam = labels[(row["file"], int(row["pkt_index"]))]
                if row["name_complete"] == "1":
                    self.assertEqual(row["name"], name)
                else:
                    self.assertTrue(name.startswith(row["name"]))
                    self.assertEqual(row["truncated"], "1")
            self.assertEqual(sum(v[0] for v in labels.values()), sum(m["exfil"] for m in man))
            mixed = next(m for m in man if m["file"] == "mixed_b32_l16_seq_r20.pcap")
            self.assertAlmostEqual(mixed["exfil"] / mixed["queries"], 0.2, delta=0.06)
            self.assertEqual(next(m for m in man if m["file"] == "mixed_none_r00.pcap")["exfil"], 0)
            # one attacker per mixed stream: all its exfil queries share one domain
            doms = {n.split(".", 1)[1] if not n.startswith("sq") else n.split(".", 2)[2]
                    for (f, _), (lab, n, _fam) in labels.items() if f == "mixed_b32_l16_seq_r20.pcap" and lab == 1}
            self.assertEqual(len(doms), 1)
            gaps = {m["mean_gap"] for m in man if m["kind"] == "evasive"}
            self.assertEqual(gaps, set(ev.PURE_GAPS))

    def test_full_capture_keeps_names_complete(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            ev.generate(t / "s", n_family=10, n_hard=20, n_mixed=30, mixed_rates=(0.5,), mixed_families=("words",), snaplen=65535)
            pf.run([t / "s"], out_dir=t / "f", scenario="synthetic", verbose=False)
            self.assertTrue(all(r["truncated"] == "0" and r["name_complete"] == "1" for r in read(t / "f" / "queries.csv")))


if __name__ == "__main__":
    unittest.main()
