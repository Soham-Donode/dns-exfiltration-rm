"""Tests for src/stateful_features.py.   Run:  python -m unittest tests/test_stateful_features.py"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import stateful_features as sf  # noqa: E402


def frame(rows):
    base = dict(scenario="x", file="f", src="h", dns_len=40, entropy=2.0, name_complete=1)
    return pd.DataFrame([{**base, **r} for r in rows])


class TestBaseAndSub(unittest.TestCase):
    def test_split(self):
        self.assertEqual(sf.base_and_sub("WwW.Example.COM", 1), ("example.com", "www"))
        self.assertEqual(sf.base_and_sub("shop.bbc.co.uk", 1), ("bbc.co.uk", "shop"))
        self.assertEqual(sf.base_and_sub("example.com", 1), ("example.com", ""))
        self.assertEqual(sf.base_and_sub("448.p2Zf2ooUVhW3", 0), ("<trunc>", "448.p2zf2oouvhw3"))


class TestHost(unittest.TestCase):
    def test_window_rate_and_diversity(self):
        df = sf.add_stateful(frame([{"ts": float(t), "name": f"n{t}.example.com"} for t in range(10)]), host_n=4)
        last = df.iloc[-1]
        self.assertEqual(last.h_n, 4)                                    # capped at host_n
        self.assertAlmostEqual(last.h_rate, 1.0)                         # 3 gaps over 3 s
        self.assertAlmostEqual(last.h_gap_mean, 1.0)
        self.assertAlmostEqual(last.h_gap_cv, 0.0)                       # perfectly regular
        self.assertEqual(last.h_unique_name_ratio, 1.0)
        self.assertEqual(last.h_unique_base_ratio, 0.25)                 # one base domain
        self.assertEqual(df.iloc[0].h_n, 1)

    def test_causal_and_per_host(self):
        rows = [{"ts": float(t), "name": "a.example.com"} for t in range(5)]
        rows += [{"ts": float(t), "name": "b.other.org", "src": "h2"} for t in range(3)]
        df = sf.add_stateful(frame(rows), host_n=100)
        h1 = df[df.src == "h"].sort_values("ts")
        self.assertEqual(list(h1.h_n), [1, 2, 3, 4, 5])                  # never sees the future
        self.assertEqual(df[df.src == "h2"].h_n.max(), 3)                # hosts do not mix

    def test_row_order_does_not_matter(self):
        rows = [{"ts": float(t), "name": f"x{t % 3}.example.com"} for t in range(12)]
        a = sf.add_stateful(frame(rows))
        b = sf.add_stateful(frame(rows).sample(frac=1, random_state=1)).sort_values("ts")
        np.testing.assert_allclose(a[sf.HOST_FEATURES + sf.DOMAIN_FEATURES].to_numpy(),
                                   b[sf.HOST_FEATURES + sf.DOMAIN_FEATURES].to_numpy())


class TestDomain(unittest.TestCase):
    def test_demo_separates_exfil_domain(self):
        df = sf.demo()
        ex = df[df.base == "evil.net"].iloc[-1]
        ok = df[df.base == "example.com"].iloc[-1]
        self.assertEqual((ex.d_n, ex.d_unique_sub), (5, 5))              # every chunk is a new subdomain
        self.assertEqual((ok.d_n, ok.d_unique_sub), (5, 1))              # same site looked up again
        self.assertAlmostEqual(ex.d_share, 0.5)

    def test_time_eviction_and_bytes(self):
        rows = [{"ts": 0.0, "name": "a.evil.net", "dns_len": 100}, {"ts": 5.0, "name": "b.evil.net", "dns_len": 100},
                {"ts": 20.0, "name": "c.evil.net", "dns_len": 100}]
        last = sf.add_stateful(frame(rows), dom_seconds=10).iloc[-1]
        self.assertEqual(last.d_n, 1)                                    # the two old ones fell out
        self.assertEqual(last.d_bytes, 100)
        self.assertEqual(last.d_gap_mean, 10)                            # no previous query -> cap

    def test_truncated_names_share_a_bucket(self):
        rows = [{"ts": float(t), "name": f"{t}.chunk{t}", "name_complete": 0} for t in range(4)]
        last = sf.add_stateful(frame(rows)).iloc[-1]
        self.assertEqual(last.base, "<trunc>")
        self.assertEqual(last.d_unique_sub, 4)


class TestBehaviour(unittest.TestCase):
    def test_long_horizon_and_novelty(self):
        rows = [{"ts": float(t * 10), "name": f"c{t}.evil.net", "dns_len": 50} for t in range(6)]   # every 10 s, new sub
        rows += [{"ts": float(t * 10 + 5), "name": "www.example.com"} for t in range(6)]            # same name again
        df = sf.add_behaviour(sf.add_stateful(frame(rows)), long_seconds=30)
        ex = df[df.base == "evil.net"].sort_values("ts")
        ok = df[df.base == "example.com"].sort_values("ts")
        self.assertEqual(list(ex.dl_n), [1, 2, 3, 4, 4, 4])                  # 30 s horizon holds 4 queries
        self.assertEqual(ex.d_new_frac.iloc[-1], 1.0)                        # every subdomain is new
        self.assertAlmostEqual(ex.d_cum_unique.iloc[-1], np.log1p(6))        # cumulative never forgets
        self.assertAlmostEqual(ex.dl_gap_cv.iloc[-1], 0.0)                   # perfectly regular
        self.assertAlmostEqual(ex.dl_bytes_rate.iloc[-1], 4 * 50 / 30)
        self.assertEqual(ok.d_new_frac.iloc[0], 1.0)                         # first lookup is new ...
        self.assertAlmostEqual(ok.d_new_frac.iloc[2], 1 / 3)                 # ... the repeats are not
        self.assertEqual(ok.d_new_frac.iloc[-1], 0.0)                        # the new one has left the 30 s horizon
        self.assertAlmostEqual(ok.d_cum_unique.iloc[-1], np.log1p(1))


class TestScores(unittest.TestCase):
    def test_rolling_scores(self):
        df = sf.add_stateful(frame([{"ts": float(t), "name": f"s{t}.evil.net"} for t in range(4)]), host_n=2)
        df["p"] = [0.0, 1.0, 1.0, 0.0]
        r = sf.add_rolling_scores(df, "p", host_n=2, dom_seconds=100)
        self.assertEqual(list(r.h_score_mean), [0.0, 0.5, 1.0, 0.5])
        self.assertEqual(list(r.d_score_max), [0.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(r.d_score_mean.iloc[-1], 0.5)


if __name__ == "__main__":
    unittest.main()
