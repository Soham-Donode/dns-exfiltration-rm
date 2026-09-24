"""
Week 4: DNS Feature Engineering Engine
=======================================
Extraction of lexical, statistical, and information-theoretic features from
DNS query names (FQDNs), including Shannon entropy, character distribution
ratios, label counts, and n-gram characteristics.
"""

import math
import re
from collections import Counter
from typing import Dict, Any, List, Optional
import argparse
from pathlib import Path
import pandas as pd


def shannon_entropy(label: str) -> float:
    """Calculate Shannon entropy in bits for a DNS label or query string."""
    if not label:
        return 0.0
    counts = Counter(label)
    length = len(label)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def compute_entropy(text: str) -> float:
    """Calculate Shannon entropy of a given string: H(X) = -sum(p(x) * log2(p(x)))."""
    return round(shannon_entropy(text), 4)


def vowel_digit_ratio(label: str) -> tuple[float, float]:
    """Return the vowel and digit ratios for a DNS label."""
    vowels = set("aeiou")
    length = len(label) or 1
    vowel_count = sum(char.lower() in vowels for char in label)
    digit_count = sum(char.isdigit() for char in label)
    return vowel_count / length, digit_count / length


def calculate_query_frequency(
    records: pd.DataFrame,
    domain_column: str = "qname",
    timestamp_column: str = "timestamp",
    window: str = "60s",
    rate_unit: str = "minute",
) -> pd.DataFrame:
    """Add a per-domain rolling query rate to DNS records.

    ``timestamp`` may contain Unix seconds or datetime-like values. The result
    is sorted by domain and time, and the current query is included in its
    rolling window. ``rate_unit`` may be ``second`` or ``minute``.
    """
    if domain_column not in records or timestamp_column not in records:
        raise KeyError(f"records must contain {domain_column!r} and {timestamp_column!r}")
    if rate_unit not in {"second", "minute"}:
        raise ValueError("rate_unit must be 'second' or 'minute'")

    result = records.copy()
    timestamps = result[timestamp_column]
    if pd.api.types.is_numeric_dtype(timestamps):
        result[timestamp_column] = pd.to_datetime(timestamps, unit="s")
    else:
        result[timestamp_column] = pd.to_datetime(timestamps)
    result = result.sort_values([domain_column, timestamp_column]).reset_index(drop=True)

    window_seconds = pd.Timedelta(window).total_seconds()
    divisor = 60.0 if rate_unit == "minute" else 1.0
    rates = []
    for _, group in result.groupby(domain_column, sort=False):
        time_index = group.set_index(timestamp_column).index
        counts = pd.Series(1.0, index=time_index).rolling(window).sum()
        rates.extend((counts.to_numpy() / (window_seconds / divisor)).tolist())
    result[f"queries_per_{rate_unit}"] = rates
    return result


def split_domain_components(fqdn: str) -> Dict[str, str]:
    """
    Split FQDN into subdomain, domain, and suffix.
    Uses tldextract if installed, otherwise uses robust regex/split fallback.
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
    """Compute the maximum sequence of consecutive consonants."""
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
    
    Features include:
    - Length metrics (total length, subdomain length, average label length)
    - Structural metrics (subdomain count, max label length)
    - Information theory (Shannon entropy of full domain and subdomain)
    - Character ratios (digits, alphabets, hex, vowels, consonants, hyphens)
    - Phonetic/lexical metrics (max consecutive consonants, unique char ratio)
    """
    parts = split_domain_components(fqdn)
    full_str = parts["full"]
    subdomain_str = parts["subdomain"]
    labels = [lbl for lbl in full_str.split(".") if lbl]

    full_len = len(full_str)
    sub_len = len(subdomain_str)
    num_labels = len(labels)
    label_lens = [len(l) for l in labels] if labels else [0]
    max_label_len = max(label_lens)
    avg_label_len = sum(label_lens) / max(num_labels, 1)

    if full_len == 0:
        return {}

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

    features = {
        "domain_len": full_len,
        "subdomain_len": sub_len,
        "num_labels": num_labels,
        "max_label_len": max_label_len,
        "avg_label_len": round(avg_label_len, 2),
        "entropy": compute_entropy(full_str),
        "subdomain_entropy": compute_entropy(subdomain_str) if sub_len > 0 else 0.0,
        "digit_ratio": round(num_digits / full_len, 4),
        "alpha_ratio": round(num_alpha / full_len, 4),
        "hex_ratio": round(num_hex / full_len, 4),
        "vowel_ratio": round(num_vowels / max(num_alpha, 1), 4),
        "consonant_ratio": round(num_consonants / max(num_alpha, 1), 4),
        "special_ratio": round(num_special / full_len, 4),
        "subdomain_digit_ratio": round(sub_digits / max(sub_len, 1), 4),
        "subdomain_hex_ratio": round(sub_hex / max(sub_len, 1), 4),
        "unique_char_ratio": round(len(set(full_str)) / full_len, 4),
        "max_consecutive_consonants": max_consecutive_consonants(full_str),
    }

    return features


def process_domains_to_dataframe(
    domain_list: List[str],
    labels: Optional[List[int]] = None
) -> pd.DataFrame:
    """Convert a list of domains and optional labels (0: benign, 1: exfil) to a DataFrame."""
    feature_rows = []
    for d in domain_list:
        feat = extract_domain_features(d)
        feat["domain"] = d
        feature_rows.append(feat)

    df = pd.DataFrame(feature_rows)
    if labels is not None and len(labels) == len(df):
        df["label"] = labels
    return df


def main():
    parser = argparse.ArgumentParser(description="Extract features from DNS domains or PCAP.")
    parser.add_argument("--domain", help="Single domain to evaluate")
    parser.add_argument("--input-csv", help="Input CSV with domain column")
    parser.add_argument("--output-csv", default="data/processed/dns_features.csv", help="Output path for feature CSV")

    args = parser.parse_args()

    if args.domain:
        feats = extract_domain_features(args.domain)
        print(f"Features for '{args.domain}':")
        for k, v in feats.items():
            print(f"  {k:28}: {v}")
    elif args.input_csv:
        in_path = Path(args.input_csv)
        if not in_path.exists():
            print(f"[!] Input file {in_path} does not exist.")
            return

        df_in = pd.read_csv(in_path)
        col = "domain" if "domain" in df_in.columns else df_in.columns[0]
        domains = df_in[col].astype(str).tolist()
        label_col = "label" if "label" in df_in.columns else None
        labels = df_in[label_col].tolist() if label_col else None

        df_feats = process_domains_to_dataframe(domains, labels=labels)
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_feats.to_csv(out_path, index=False)
        print(f"[+] Extracted features for {len(df_feats)} domains saved to {out_path}")
    else:
        # Quick demonstration
        demo_domains = [
            "google.com",
            "cdn.cloudflare.net",
            "a4f91b72e04d.tunnel.attacker-domain.org",
            "mzxw6ytboi======.secret-leak.net"
        ]
        print("[*] Demonstration Feature Extraction on Test Domains:")
        df_demo = process_domains_to_dataframe(demo_domains, labels=[0, 0, 1, 1])
        print(df_demo[["domain", "domain_len", "entropy", "hex_ratio", "label"]])


if __name__ == "__main__":
    main()
