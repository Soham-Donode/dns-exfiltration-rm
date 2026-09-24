"""
Unit Tests for DNS Exfiltration Detection System
Covers Weeks 1 through 5:
- Week 1: Dataset acquisition & verification
- Week 2: Scapy simulation engine & RFC 1035 validation
- Week 3: Traffic synthesis & PCAP/CSV parsing
- Week 4: Feature engineering (Entropy, Lengths, Ratios, Frequency)
- Week 5: Model training, tuning, and evaluation (DT, RF, XGBoost, 5-Fold CV)
"""

import unittest
import os
import tempfile
import pandas as pd
from pathlib import Path

from src.download_data import load_config, generate_cic_sample, verify_environment
from src.dns_simulator import (
    validate_rfc_compliance,
    DNSExfiltrationSimulator
)
from src.capture_traffic import (
    load_benign_domains,
    generate_synthesized_traffic,
    label_dns_records,
    save_labeled_csv
)
from src.features import (
    shannon_entropy,
    compute_entropy,
    vowel_digit_ratio,
    split_domain_components,
    extract_domain_features,
    calculate_query_frequency,
    process_traffic_dataset,
    process_domains_to_dataframe
)
from src.train import (
    generate_synthetic_dataset,
    get_base_model,
    evaluate_model,
    train_pipeline
)


class TestWeek1DatasetAcquisition(unittest.TestCase):
    """Test Week 1 modules."""

    def test_load_config(self):
        config = load_config("configs/config.yaml")
        self.assertIn("paths", config)
        self.assertIn("simulation", config)
        self.assertIn("training", config)

    def test_generate_cic_sample(self):
        config = load_config("configs/config.yaml")
        sample_path = generate_cic_sample(config, n_samples=20)
        self.assertTrue(Path(sample_path).exists())
        df = pd.read_csv(sample_path)
        self.assertEqual(len(df), 20)
        self.assertIn("qname", df.columns)
        self.assertIn("tool", df.columns)
        self.assertEqual(df["label"].iloc[0], 1)

    def test_verify_environment(self):
        report = verify_environment()
        self.assertIn("python_version", report)
        self.assertIn("scapy", report)


class TestWeek2SimulationEngine(unittest.TestCase):
    """Test Week 2 simulation and RFC compliance."""

    def test_validate_rfc_compliance(self):
        # Valid domains
        self.assertTrue(validate_rfc_compliance("google.com"))
        self.assertTrue(validate_rfc_compliance("sq001.krsxg5bamn.tunnel.example.com"))
        self.assertTrue(validate_rfc_compliance("a.b.c.d.com"))

        # Invalid domains
        self.assertFalse(validate_rfc_compliance(""))
        self.assertFalse(validate_rfc_compliance("a" * 64 + ".com"))  # Label > 63 chars
        self.assertFalse(validate_rfc_compliance("-invalid.com"))      # Leading hyphen
        self.assertFalse(validate_rfc_compliance("invalid-.com"))      # Trailing hyphen
        self.assertFalse(validate_rfc_compliance("inv@lid.com"))       # Invalid char

    def test_base32_encoding_and_chunking(self):
        sim = DNSExfiltrationSimulator(target_domain="tunnel.example.com", encoding="base32")
        payload = b"Top secret classified data 123456789"
        queries = sim.generate_queries(payload)
        self.assertGreaterEqual(len(queries), 1)
        for q in queries:
            self.assertTrue(validate_rfc_compliance(q))
            self.assertIn("tunnel.example.com", q)
            self.assertNotIn("=", q)  # Padding must be stripped

    def test_hex_encoding_and_chunking(self):
        sim = DNSExfiltrationSimulator(target_domain="tunnel.corp.org", encoding="hex")
        payload = b"Hex encoded confidential exfiltration message"
        queries = sim.generate_queries(payload)
        self.assertGreaterEqual(len(queries), 1)
        for q in queries:
            self.assertTrue(validate_rfc_compliance(q))
            self.assertIn("tunnel.corp.org", q)

    def test_packet_crafting(self):
        sim = DNSExfiltrationSimulator(target_domain="tunnel.example.com")
        pkt = sim.craft_packet("sq0001.abcd.tunnel.example.com", qtype="TXT")
        self.assertEqual(pkt["DNS"].qd.qtype, 16)  # TXT record


class TestWeek3CaptureAndSynthesis(unittest.TestCase):
    """Test Week 3 traffic synthesis and PCAP/CSV labeling."""

    def test_benign_domains_loading(self):
        domains = load_benign_domains(limit=10)
        self.assertGreaterEqual(len(domains), 10)
        self.assertTrue(all("." in d for d in domains))

    def test_traffic_synthesis(self):
        records, packets = generate_synthesized_traffic(benign_count=30, burst_count=20, low_slow_count=10)
        self.assertEqual(len(records), 60)
        self.assertEqual(len(packets), 60)

        df = pd.DataFrame(records)
        self.assertIn("attack_type", df.columns)
        self.assertIn("label", df.columns)
        self.assertEqual(set(df["attack_type"].unique()), {"benign", "burst_exfil", "low_and_slow_exfil"})
        self.assertEqual(sum(df["label"] == 0), 30)
        self.assertEqual(sum(df["label"] == 1), 30)

    def test_label_dns_records(self):
        records = [
            {"qname": "google.com"},
            {"qname": "sq0001.krsxg5bamnxw4.tunnel.corp.net"}
        ]
        labeled = label_dns_records(records, tunnel_domain="tunnel.")
        self.assertEqual(labeled[0]["label"], 0)
        self.assertEqual(labeled[1]["label"], 1)


class TestWeek4FeatureEngineering(unittest.TestCase):
    """Test Week 4 feature engineering functions."""

    def test_shannon_entropy(self):
        # Empty string
        self.assertEqual(shannon_entropy(""), 0.0)
        # Uniform single char
        self.assertEqual(shannon_entropy("aaaaaa"), 0.0)
        # Diverse string
        ent = compute_entropy("abcdefghijklmnop")
        self.assertAlmostEqual(ent, 4.0, places=1)

    def test_vowel_digit_ratio(self):
        v_ratio, d_ratio = vowel_digit_ratio("abcde123")
        self.assertAlmostEqual(v_ratio, 2 / 8)
        self.assertAlmostEqual(d_ratio, 3 / 8)

    def test_extract_domain_features(self):
        domain = "sq0001.krsxg5bamnxw4ztj.tunnel.example.com"
        feats = extract_domain_features(domain)
        self.assertIn("domain_len", feats)
        self.assertIn("entropy", feats)
        self.assertIn("subdomain_entropy", feats)
        self.assertIn("max_label_len", feats)
        self.assertIn("vowel_ratio", feats)
        self.assertIn("digit_ratio", feats)
        self.assertIn("hex_ratio", feats)
        self.assertGreater(feats["entropy"], 3.0)

    def test_query_frequency(self):
        records = pd.DataFrame({
            "qname": ["exfil.org"] * 4,
            "timestamp": [100.0, 101.0, 102.0, 103.0]
        })
        result = calculate_query_frequency(records, window="10s", rate_unit="second")
        self.assertIn("queries_per_second", result.columns)
        self.assertGreater(result["queries_per_second"].iloc[-1], 0.0)


class TestWeek5ModelTraining(unittest.TestCase):
    """Test Week 5 model training, cross-validation, and tuning."""

    def setUp(self):
        self.config = load_config("configs/config.yaml")
        self.df = generate_synthetic_dataset(n_samples=200, seed=42)

    def test_synthetic_dataset_generation(self):
        self.assertEqual(len(self.df), 200)
        self.assertIn("label", self.df.columns)
        self.assertIn("entropy", self.df.columns)
        self.assertEqual(sum(self.df["label"] == 0), 100)
        self.assertEqual(sum(self.df["label"] == 1), 100)

    def test_base_models_instantiation(self):
        dt = get_base_model("decision_tree", self.config)
        rf = get_base_model("random_forest", self.config)
        xgb = get_base_model("xgboost", self.config)
        self.assertIsNotNone(dt)
        self.assertIsNotNone(rf)
        self.assertIsNotNone(xgb)

    def test_stratified_evaluation(self):
        dt = get_base_model("decision_tree", self.config)
        X = self.df.drop(columns=["label"])
        y = self.df["label"]
        X_train, X_test, y_train, y_test = X.iloc[:160], X.iloc[160:], y.iloc[:160], y.iloc[160:]
        metrics = evaluate_model(dt, X_train, y_train, X_test, y_test, cv_folds=3)
        self.assertIn("cv_f1_mean", metrics)
        self.assertIn("test_accuracy", metrics)
        self.assertIn("test_f1", metrics)
        self.assertGreaterEqual(metrics["test_f1"], 0.8)


if __name__ == "__main__":
    unittest.main()
