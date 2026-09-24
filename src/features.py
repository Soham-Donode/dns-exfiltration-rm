"""
Week 4: DNS Feature Engineering Engine
======================================
Comprehensive extraction of lexical, information-theoretic, structural,
and temporal/frequency features from DNS queries:
1. Information Theory: Shannon entropy (full domain, subdomain, max label).
2. Label Lengths: total length, subdomain length, label count, max/avg/min label lengths.
3. Ratios: vowel ratio, digit ratio, consonant ratio, vowel-to-consonant, hex ratio, special chars.
4. Query Frequency: rolling window query rate per second/minute for burst vs slow detection.
5. Optimized for Colab, Kaggle, and production inference.
"""

import math
import re
import argparse
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd


def shannon_entropy(text: str) -> float:
    """
    Calculate Shannon entropy in bits for a DNS query or label:
    H(X) = - sum(p(x) * log2(p(x)))
    """
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return float(-sum((count / length) * math.log2(count / length) for count in counts.values()))


def compute_entropy(text: str) -> float:
    """Convenience alias for Shannon entropy rounded to 4 decimals."""
    return round(shannon_entropy(text), 4)


def vowel_digit_ratio(label: str) -> Tuple[float, float]:
    """
    Calculate the vowel ratio and digit ratio for a DNS label.
    Returns: (vowel_ratio, digit_ratio)
    """
    if not label:
        return 0.0, 0.0
    vowels = set("aeiou")
    length = len(label)
    vowel_count = sum(c.lower() in vowels for c in label)
    digit_count = sum(c.isdigit() for c in label)
    return round(vowel_count / length, 4), round(digit_count / length, 4)


def split_domain_components(fqdn: str) -> Dict[str, str]:
    """
    Split FQDN into subdomain, second-level domain, and suffix/TLD.
    Uses tldextract if available; otherwise falls back to robust regex heuristic.
    """
    cleaned = fqdn.strip(".").lower()
    try:
        import tldextract
        ext = tldextract.extract(cleaned)
        subdomain = ext.subdomain
        domain = ext.domain
        suffix = ext.suffix
    except ImportError:
        parts = cleaned.split(".")
        if len(parts) > 2:
            subdomain = ".".join(parts[:-2])
            domain = parts[-2]
            suffix = parts[-1]
        elif len(parts) == 2:
            subdomain = ""
            domain = parts[0]
            suffix = parts[1]
        else:
            subdomain = ""
            domain = cleaned
            suffix = ""

    return {
        "subdomain": subdomain,
        "domain": domain,
        "suffix": suffix,
        "full": cleaned
    }


def max_consecutive_consonants(text: str) -> int:
    """Compute the longest streak of consecutive consonants in the text."""
    vowels = set("aeiou")
    max_streak = 0
    current_streak = 0
    for char in text.lower():
        if char.isalpha() and char not in vowels:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 0
    return max_streak


def extract_domain_features(fqdn: str) -> Dict[str, Any]:
    """
    Extract a comprehensive feature vector from a DNS query domain.

    Features computed:
    - Length metrics: domain_len, subdomain_len, num_labels, max_label_len, avg_label_len, min_label_len
    - Entropy metrics: entropy (full domain), subdomain_entropy, max_label_entropy
    - Character distribution: digit_ratio, alpha_ratio, hex_ratio, vowel_ratio, consonant_ratio,
      vowel_consonant_ratio, special_ratio, unique_char_ratio
    - Subdomain metrics: subdomain_digit_ratio, subdomain_hex_ratio
    - Phonetic patterns: max_consecutive_consonants
    """
    parts = split_domain_components(fqdn)
    full_str = parts["full"]
    subdomain_str = parts["subdomain"]
    labels = [lbl for lbl in full_str.split(".") if lbl]

    full_len = len(full_str)
    sub_len = len(subdomain_str)
    num_labels = len(labels)

    if full_len == 0:
        return {}

    label_lens = [len(l) for l in labels] if labels else [0]
    max_label_len = max(label_lens)
    min_label_len = min(label_lens)
    avg_label_len = sum(label_lens) / max(num_labels, 1)

    # Label-level entropies
    label_entropies = [compute_entropy(l) for l in labels] if labels else [0.0]
    max_label_entropy = max(label_entropies)

    # Character distributions
    num_digits = sum(c.isdigit() for c in full_str)
    num_alpha = sum(c.isalpha() for c in full_str)
    num_hex = len(re.findall(r"[0-9a-fA-F]", full_str))
    num_vowels = len(re.findall(r"[aeiouAEIOU]", full_str))
    num_consonants = num_alpha - num_vowels
    num_special = sum(not c.isalnum() and c != "." for c in full_str)

    # Subdomain specific stats
    sub_digits = sum(c.isdigit() for c in subdomain_str) if sub_len > 0 else 0
    sub_hex = len(re.findall(r"[0-9a-fA-F]", subdomain_str)) if sub_len > 0 else 0

    # Ratios
    vowel_ratio = num_vowels / max(num_alpha, 1)
    consonant_ratio = num_consonants / max(num_alpha, 1)
    vowel_consonant_ratio = num_vowels / max(num_consonants, 1)

    features = {
        # 1. Length features
        "domain_len": full_len,
        "subdomain_len": sub_len,
        "num_labels": num_labels,
        "max_label_len": max_label_len,
        "min_label_len": min_label_len,
        "avg_label_len": round(avg_label_len, 2),

        # 2. Entropy features
        "entropy": compute_entropy(full_str),
        "subdomain_entropy": compute_entropy(subdomain_str) if sub_len > 0 else 0.0,
        "max_label_entropy": round(max_label_entropy, 4),

        # 3. Character & Vowel/Digit ratios
        "digit_ratio": round(num_digits / full_len, 4),
        "alpha_ratio": round(num_alpha / full_len, 4),
        "hex_ratio": round(num_hex / full_len, 4),
        "vowel_ratio": round(vowel_ratio, 4),
        "consonant_ratio": round(consonant_ratio, 4),
        "vowel_consonant_ratio": round(vowel_consonant_ratio, 4),
        "special_ratio": round(num_special / full_len, 4),
        "subdomain_digit_ratio": round(sub_digits / max(sub_len, 1), 4),
        "subdomain_hex_ratio": round(sub_hex / max(sub_len, 1), 4),
        "unique_char_ratio": round(len(set(full_str)) / full_len, 4),
        "max_consecutive_consonants": max_consecutive_consonants(full_str),
    }

    return features


def calculate_query_frequency(
    records: pd.DataFrame,
    domain_column: str = "qname",
    timestamp_column: str = "timestamp",
    window: str = "60s",
    rate_unit: str = "minute",
) -> pd.DataFrame:
    """
    Calculate rolling query frequency per domain or source IP over a time window.
    Supports timestamps as Unix seconds float or datetime strings.
    """
    if domain_column not in records or timestamp_column not in records:
        raise KeyError(f"records must contain {domain_column!r} and {timestamp_column!r}")
    if rate_unit not in {"second", "minute"}:
        raise ValueError("rate_unit must be 'second' or 'minute'")

    result = records.copy()
    timestamps = result[timestamp_column]
    if pd.api.types.is_numeric_dtype(timestamps):
        result["_dt"] = pd.to_datetime(timestamps, unit="s")
    else:
        result["_dt"] = pd.to_datetime(timestamps)

    result = result.sort_values([domain_column, "_dt"]).reset_index(drop=True)
    window_seconds = pd.Timedelta(window).total_seconds()
    divisor = 60.0 if rate_unit == "minute" else 1.0

    rates = []
    for _, group in result.groupby(domain_column, sort=False):
        time_index = group.set_index("_dt").index
        counts = pd.Series(1.0, index=time_index).rolling(window).sum()
        rates.extend((counts.to_numpy() / (window_seconds / divisor)).tolist())

    col_name = f"queries_per_{rate_unit}"
    result[col_name] = [round(r, 2) for r in rates]
    result.drop(columns=["_dt"], inplace=True)
    return result


def process_traffic_dataset(
    df: pd.DataFrame,
    window: str = "60s"
) -> pd.DataFrame:
    """
    Process an entire traffic DataFrame (e.g. from Week 3 PCAP extraction or CSV),
    calculating lexical, structural, information-theoretic, and temporal frequency features.
    """
    domain_col = "qname" if "qname" in df.columns else ("domain" if "domain" in df.columns else df.columns[0])
    domains = df[domain_col].astype(str).tolist()

    # 1. Lexical and entropy features
    feature_rows = []
    for d in domains:
        feats = extract_domain_features(d)
        feature_rows.append(feats)

    feats_df = pd.DataFrame(feature_rows)

    # 2. Add domain identifier and original columns
    feats_df["domain"] = domains
    if "label" in df.columns:
        feats_df["label"] = df["label"].values
    if "attack_type" in df.columns:
        feats_df["attack_type"] = df["attack_type"].values

    # 3. Calculate temporal frequency features if timestamps present
    if "timestamp" in df.columns and len(df) > 1:
        temp_df = pd.DataFrame({
            "qname": domains,
            "timestamp": df["timestamp"].values
        })
        try:
            freq_df = calculate_query_frequency(temp_df, domain_column="qname", timestamp_column="timestamp", window=window)
            feats_df["queries_per_minute"] = freq_df["queries_per_minute"].values
        except Exception as e:
            feats_df["queries_per_minute"] = 1.0
    else:
        feats_df["queries_per_minute"] = 1.0

    return feats_df


def process_domains_to_dataframe(
    domain_list: List[str],
    labels: Optional[List[int]] = None
) -> pd.DataFrame:
    """Convert a list of domains and optional labels (0: benign, 1: exfil) to a DataFrame."""
    feature_rows = []
    for d in domain_list:
        feat = extract_domain_features(d)
        feat["domain"] = d
        feat["queries_per_minute"] = 1.0
        feature_rows.append(feat)

    df = pd.DataFrame(feature_rows)
    if labels is not None and len(labels) == len(df):
        df["label"] = labels
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Week 4: Feature Engineering Engine (Shannon Entropy, Label Lengths, Ratios, Frequency)"
    )
    parser.add_argument("--domain", help="Inspect feature vector for a single domain/query")
    parser.add_argument("--input-csv", default="data/processed/labeled_traffic.csv", help="Input labeled CSV or domain CSV")
    parser.add_argument("--output-csv", default="data/processed/dns_features.csv", help="Output feature matrix CSV")
    parser.add_argument("--window", default="60s", help="Rolling time window for frequency computation (e.g. 60s, 5m)")

    args = parser.parse_args()

    if args.domain:
        feats = extract_domain_features(args.domain)
        v_ratio, d_ratio = vowel_digit_ratio(args.domain)
        print("=" * 65)
        print(f"Feature Extraction for: {args.domain}")
        print("=" * 65)
        for k, v in feats.items():
            print(f"  {k:28}: {v}")
        print(f"  vowel_digit_ratio tuple     : ({v_ratio}, {d_ratio})")
        print("=" * 65)
        return

    in_path = Path(args.input_csv)
    if not in_path.exists():
        print(f"[!] Input CSV '{in_path}' not found.")
        print("    Running demonstration feature extraction on benchmark examples:")
        demo_domains = [
            "google.com",
            "cdn.cloudflare.net",
            "sq0001.5465737420636f6e666964656e7469616c.tunnel.example.com",
            "sq0002.krsxg5bamnxw4ztjmrsw45djmfwcazdborqsazl.tunnel.example.com",
            "v1.89fa012b3c4d5e6f7a8b9c0d1e2f.tunnel.iodine-cic.net"
        ]
        df_demo = process_domains_to_dataframe(demo_domains, labels=[0, 0, 1, 1, 1])
        print(df_demo[["domain", "domain_len", "entropy", "max_label_len", "digit_ratio", "vowel_ratio", "label"]])
        return

    print(f"[*] Loading raw dataset from {in_path}...")
    df_raw = pd.read_csv(in_path)
    print(f"[+] Loaded {len(df_raw)} records. Extracting lexical, entropy, ratio, and frequency features...")

    df_feats = process_traffic_dataset(df_raw, window=args.window)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_feats.to_csv(out_path, index=False)

    print(f"[✓] Feature engineering completed successfully!")
    print(f"[✓] Saved {len(df_feats)} records x {df_feats.shape[1]} features to {out_path}")
    print("\nFeature Summary Preview:")
    cols_to_preview = ["domain", "domain_len", "entropy", "max_label_len", "digit_ratio", "vowel_ratio", "queries_per_minute"]
    if "label" in df_feats.columns:
        cols_to_preview.append("label")
    print(df_feats[cols_to_preview].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
