"""
Week 1: Scoping & Dataset Acquisition
=====================================
Acquisition and verification script for DNS exfiltration detection datasets:
1. Benign: Tranco Top 1M domain ranking list (evasion-resistant).
2. Malicious: CIC-Bell-DNS-EXF-2021 dataset (University of New Brunswick) or
   reproducible benchmark sample.
3. Environment verification for GitHub, Google Colab, and Kaggle.
"""

import os
import sys
import shutil
import zipfile
import argparse
import subprocess
from pathlib import Path
import urllib.request
import yaml
import pandas as pd


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_file(url: str, dest_path: Path) -> Path:
    """Download a remote file with progress reporting."""
    print(f"[+] Downloading: {url}")
    print(f"    Destination: {dest_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    def _reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded / total_size * 100)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r    Progress: {percent:5.1f}% ({mb:6.2f} MB / {total_mb:6.2f} MB)", end="")
        else:
            mb = downloaded / (1024 * 1024)
            print(f"\r    Downloaded: {mb:6.2f} MB", end="")

    urllib.request.urlretrieve(url, dest_path, reporthook=_reporthook)
    print("\n[+] Download complete.")
    return dest_path


def acquire_tranco(config: dict, sample_size: int = None) -> Path:
    """
    Acquire and extract the Tranco Top 1M list.
    Optionally create a truncated subset for rapid prototyping.
    """
    raw_dir = Path(config["paths"]["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    target_csv = Path(config["paths"]["tranco_file"])
    zip_path = raw_dir / "top-1m.csv.zip"
    tranco_url = config["paths"]["tranco_url"]

    if target_csv.exists() and target_csv.stat().st_size > 1000:
        print(f"[✓] Tranco dataset already present at {target_csv} ({target_csv.stat().st_size / (1024*1024):.2f} MB)")
    else:
        download_file(tranco_url, zip_path)
        print(f"[+] Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            csv_names = [f for f in zip_ref.namelist() if f.endswith(".csv")]
            if not csv_names:
                raise FileNotFoundError("No CSV file found in downloaded Tranco archive.")
            extracted_name = csv_names[0]
            zip_ref.extract(extracted_name, raw_dir)

            extracted_path = raw_dir / extracted_name
            if extracted_path != target_csv:
                extracted_path.rename(target_csv)

        if zip_path.exists():
            zip_path.unlink()
        print(f"[✓] Tranco list successfully saved to {target_csv}")

    if sample_size:
        sample_path = raw_dir / f"tranco_top_{sample_size}.csv"
        print(f"[*] Creating sample subset ({sample_size} records) at {sample_path}...")
        df = pd.read_csv(target_csv, header=None, nrows=sample_size, names=["rank", "domain"])
        df.to_csv(sample_path, index=False)
        print(f"[✓] Sample saved to {sample_path}")
        return sample_path

    return target_csv


def check_cic_dataset(config: dict) -> bool:
    """Check status of CIC-Bell-DNS-EXF-2021 dataset in data/raw/cic_bell."""
    cic_dir = Path(config["paths"]["cic_bell_dir"])
    print(f"[*] Checking CIC-Bell-DNS-EXF-2021 directory: {cic_dir}")

    if cic_dir.exists() and any(cic_dir.iterdir()):
        files = list(cic_dir.glob("*"))
        print(f"[✓] Found {len(files)} files in {cic_dir}:")
        for f in files[:8]:
            print(f"    - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
        if len(files) > 8:
            print(f"    ... and {len(files) - 8} more files.")
        return True
    else:
        print(f"[!] CIC-Bell dataset not found in {cic_dir}.")
        print("    Instructions for acquiring the official CIC-Bell-DNS-EXF-2021 dataset:")
        print("    1. Visit: https://www.unb.ca/cic/datasets/dns-exf-2021.html")
        print("    2. Request access and download the PCAP or CSV archives.")
        print(f"    3. Extract contents into: {cic_dir}/")
        print("    Alternatively, run with '--generate-cic-sample' to create a synthetic benchmark dataset.")
        return False


def generate_cic_sample(config: dict, n_samples: int = 1000) -> Path:
    """
    Generate a synthetic CIC-Bell style dataset for offline testing and benchmarking.
    Includes simulated traffic patterns from iodine, dnscat2, and dns2tcp.
    """
    cic_dir = Path(config["paths"]["cic_bell_dir"])
    cic_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = cic_dir / "cic_bell_sample.csv"

    print(f"[*] Generating {n_samples} synthetic CIC-Bell benchmark queries...")
    tools = ["iodine", "dnscat2", "dns2tcp"]
    records = []

    for i in range(n_samples):
        tool = tools[i % len(tools)]
        if tool == "iodine":
            # Base32/Base64 high-density query labels
            hex_payload = os.urandom(24).hex()
            qname = f"v1.{hex_payload[:28]}.{hex_payload[28:]}.tunnel.iodine-cic.net"
        elif tool == "dnscat2":
            # Hex-encoded chunked commands/data
            hex_data = os.urandom(18).hex()
            qname = f"dnscat.{i:04x}.{hex_data}.c2.dnscat2-cic.org"
        else:  # dns2tcp
            payload = os.urandom(20).hex()
            qname = f"data-{i:03d}.{payload}.auth.dns2tcp-cic.com"

        records.append({
            "timestamp": 1710000000.0 + i * 0.1,
            "src_ip": "192.168.10.15",
            "dst_ip": "192.168.10.1",
            "qname": qname,
            "qtype": 1 if tool == "dnscat2" else 16,
            "tool": tool,
            "label": 1,
        })

    df = pd.DataFrame(records)
    df.to_csv(sample_csv, index=False)
    print(f"[✓] Synthetic CIC-Bell benchmark dataset created at {sample_csv}")
    return sample_csv


def download_from_kaggle(dataset_slug: str, target_dir: Path) -> bool:
    """Download a dataset using the Kaggle CLI if configured."""
    print(f"[*] Attempting Kaggle download for '{dataset_slug}'...")
    if not shutil.which("kaggle"):
        print("[!] Kaggle CLI ('kaggle') not found. Please install via: pip install kaggle")
        return False

    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(target_dir), "--unzip"]
    try:
        subprocess.run(cmd, check=True)
        print(f"[✓] Kaggle dataset downloaded to {target_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] Kaggle CLI download failed: {e}")
        return False


def verify_environment() -> dict:
    """Verify runtime environment dependencies and capabilities."""
    print("=" * 60)
    print("         ENVIRONMENT VERIFICATION & CAPABILITY CHECK         ")
    print("=" * 60)

    report = {}

    # Python version
    report["python_version"] = sys.version.split()[0]
    print(f"[*] Python:       {report['python_version']}")

    # Check key packages
    packages = [
        "scapy", "sklearn", "xgboost", "lightgbm", "pandas",
        "numpy", "yaml", "joblib", "matplotlib", "seaborn"
    ]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            report[pkg] = ver
            print(f"[✓] {pkg:14}: {ver}")
        except ImportError:
            report[pkg] = "missing"
            print(f"[!] {pkg:14}: NOT INSTALLED")

    # Check tcpdump
    tcpdump_bin = shutil.which("tcpdump")
    report["tcpdump"] = tcpdump_bin or "missing"
    print(f"[*] tcpdump:      {tcpdump_bin or 'Not found in PATH'}")

    # Check Colab / Kaggle environment
    is_colab = "google.colab" in sys.modules or "COLAB_GPU" in os.environ
    is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle")
    report["is_colab"] = is_colab
    report["is_kaggle"] = is_kaggle

    env_type = "Google Colab" if is_colab else ("Kaggle Notebook" if is_kaggle else "Local Machine")
    print(f"[*] Environment:  {env_type}")
    print("=" * 60)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Week 1: Scoping & Dataset Acquisition (Tranco 1M & CIC-Bell-DNS-EXF-2021)"
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--tranco", action="store_true", help="Download and extract Tranco top-1M benign list")
    parser.add_argument("--sample-size", type=int, help="Create a truncated Tranco sample (e.g. 50000)")
    parser.add_argument("--check-cic", action="store_true", help="Check CIC-Bell dataset presence")
    parser.add_argument("--generate-cic-sample", action="store_true", help="Generate synthetic CIC-Bell benchmark data")
    parser.add_argument("--kaggle-dataset", type=str, help="Download a specific Kaggle dataset slug (requires kaggle CLI)")
    parser.add_argument("--check-env", action="store_true", help="Verify runtime environment and dependencies")
    parser.add_argument("--all", action="store_true", help="Run full Week 1 dataset acquisition pipeline")

    args = parser.parse_args()
    config = load_config(args.config)

    if not any([args.tranco, args.sample_size, args.check_cic, args.generate_cic_sample,
                args.kaggle_dataset, args.check_env, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.check_env or args.all:
        verify_environment()

    if args.tranco or args.all:
        acquire_tranco(config, sample_size=args.sample_size)

    if args.check_cic or args.all:
        has_cic = check_cic_dataset(config)
        if not has_cic and args.all:
            generate_cic_sample(config)

    if args.generate_cic_sample:
        generate_cic_sample(config)

    if args.kaggle_dataset:
        download_from_kaggle(args.kaggle_dataset, Path(config["paths"]["raw_data_dir"]))


if __name__ == "__main__":
    main()
