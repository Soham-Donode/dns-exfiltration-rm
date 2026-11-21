"""End-to-end smoke test of src/experiment.py on tiny fake captures (~30 s).

Run:  python -m unittest tests/test_experiment.py
"""
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evasion_data as ev  # noqa: E402
import experiment as ex  # noqa: E402


def fake_real_captures(root):
    """A light 'attack' capture with DNSExfiltrator-shaped chunks + background, and a benign capture."""
    r = random.Random(0)
    b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    t, attack = 1.6e9, []
    for i in range(400):
        t += 0.5
        if i % 20 == 0:
            attack.append((t, "www.bing.com", "A", 0, "bg"))
        else:
            chunk = ".".join("".join(r.choice(b64) for _ in range(60)) for _ in range(3))
            attack.append((t, f"{i}.{chunk}.cicresearch.ca", "A", 1, "x"))
    benign = []
    for i in range(600):
        t += 1.0
        benign.append((t, f"{r.choice(['www.', '', 'cdn.'])}site{r.randint(0, 150)}.{r.choice(['com', 'org', 'net'])}", "A", 0, "b"))
    ev.write_stream(root / "PCAP_Light" / "Attacks" / "light_text.pcap", attack)
    ev.write_stream(root / "PCAP_Light" / "Benign" / "benign.pcap", benign)
    return [str(root / "PCAP_Light")]


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cfg = dict(n_family=30, n_hard=200, n_mixed=300, mixed_rates=(0.2,), mixed_families=("b32_l16_seq", "words"),
                   train_mixed_rates=(0.2,), n_train_mixed=100, n_estimators=15)
        cls.D = ex.prepare(fake_real_captures(root), root / "work", cfg, verbose=False)
        cls.preds, cls.infos = ex.run_detectors(cls.D, ["baseline", "step1", "step2", "step3a", "step3b", "step4"],
                                                baseline_models=("RF", "LR"), verbose=False)
        cls.summary, cls.detail = ex.evaluate(cls.D, cls.preds)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_roles_and_labels(self):
        A = self.D["ALL"]
        self.assertEqual(set(A.role), {"real_train", "real_test", "evasive", "hard_train", "hard_test", "mixed",
                                       "mixed_train"})
        self.assertTrue((A[A.role == "evasive"].label == 1).all())
        self.assertTrue((A[A.role.isin(["hard_train", "hard_test"])].label == 0).all())
        self.assertEqual(int(A[(A.origin == "real")].label.sum()), 380)        # shape rule on the fake capture
        for c in ex.FEATURES["step2"] + ex.CHAR + ex.FEATURES["step4_behaviour"]:
            self.assertFalse(A[c].isna().any(), c)

    def test_every_detector_scores_every_test_query(self):
        n_test = int(self.D["ALL"].role.isin(ex.TEST_ROLES).sum())
        for name, p in self.preds.items():
            self.assertEqual(len(p), n_test, name)
            self.assertTrue(((p >= 0) & (p <= 1)).all(), name)
        self.assertEqual(set(self.summary.detector),
                         {"baseline", "baseline (LR)", "baseline+window", "step1", "step2", "step3a", "step3b",
                          "step4 behaviour", "step4"})
        self.assertTrue((self.preds["step4"] >= self.preds["step3a"]).all())          # OR never lowers a score
        b = self.preds["step4 behaviour"]
        trunc = (self.D["ALL"].loc[b.index, "base"] == "<trunc>").values
        self.assertTrue(trunc.any())
        self.assertTrue((b[trunc] == 0).all())                                         # abstains on cut-off names
        for c in ex.FEATURES["step4_behaviour"]:
            self.assertNotIn(c, ex.CHAR + ex.PAPER14)                                  # truly content-agnostic

    def test_metrics_in_range(self):
        k = ex.key_table(self.summary)
        vals = k.to_numpy(dtype=float)
        self.assertTrue(np.all((vals[~np.isnan(vals)] >= 0) & (vals[~np.isnan(vals)] <= 100)))
        self.assertGreater(k.loc["baseline", "real_f1"], 80)                  # the fake real data is easy
        self.assertEqual(set(self.detail.test), {"evasive (pure stream)", "mixed 20%"})

    def test_leave_one_encoding_out_never_trains_on_heldout(self):
        A = self.D["ALL"]
        for e, info in self.infos["step3a"].items():
            n_feats = info["model"].n_features_in_
            self.assertEqual(n_feats, len(ex.FEATURES["step2"]))
        # the fold for encoding e is trained without any evasive row of e
        mask = A.role.isin(["real_train", "hard_train"]) | (A.role.isin(["evasive", "mixed_train"]) & (A.enc != "words"))
        self.assertFalse(((A.enc == "words") & mask).any())
        self.assertFalse((mask & A.role.isin(["mixed", "real_test", "hard_test"])).any())      # test-only rows never train

    def test_cache_round_trip(self):
        root = Path(self.tmp.name)
        D2 = ex.prepare([str(root / "PCAP_Light")], root / "work", self.D["cfg"], verbose=False)
        self.assertEqual(len(D2["ALL"]), len(self.D["ALL"]))


class TestSeeds(unittest.TestCase):
    def test_run_seeds_and_aggregate(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            cfg = dict(n_family=20, n_hard=100, n_mixed=200, mixed_rates=(0.2,), mixed_families=("b32_l16_seq",),
                       train_mixed_rates=(0.2,), n_train_mixed=60, n_estimators=10)
            sums, dets = ex.run_seeds(fake_real_captures(root), root / "w", seeds=(1, 2), steps=("baseline", "step2"),
                                      cfg=cfg, verbose=False)
            self.assertEqual(sorted(sums.seed.unique()), [1, 2])
            mean, std, table = ex.aggregate(sums)
            self.assertEqual(list(mean.index), ["baseline", "baseline+window", "step2"])
            self.assertTrue(table.loc["baseline", "real_f1"].count(" ± ") == 1)
            self.assertFalse(any((root / "w" / "seed_1" / "synthetic").glob("**/*.pcap")))   # cleaned up


if __name__ == "__main__":
    unittest.main()
