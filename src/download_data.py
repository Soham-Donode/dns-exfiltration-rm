"""
Week 1: Dataset Acquisition
============================
Script for downloading and setting up raw datasets for DNS exfiltration detection,
including the Tranco top 1M benign domain list and setup instructions/verification
for the CIC-Bell-DNS-EXF-2021 dataset.
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path
import urllib.request
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load project YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_file(url: str, dest_path: Path) -> Path:
    """Download file from url to dest_path with download feedback."""
    print(f"[+] Downloading from {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    def _reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            print(f"\r    Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.2f} MB)", end="")
        else:
            print(f"\r    Downloaded: {downloaded / 1024 / 1024:.2f} MB", end="")

    urllib.request.urlretrieve(url, dest_path, reporthook=_reporthook)
    print("\n[+] Download complete.")
    return dest_path


def acquire_tranco(config: dict) -> Path:
    """Acquire and extract Tranco top-1M domain list."""
    raw_dir = Path(config["paths"]["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "top-1m.csv.zip"
    target_csv = Path(config["paths"]["tranco_file"])
    tranco_url = config["paths"]["tranco_url"]

    if target_csv.exists():
        print(f"[!] Tranco file already exists at {target_csv}. Skipping download.")
        return target_csv

    download_file(tranco_url, zip_path)

    print(f"[+] Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        csv_names = [f for f in zip_ref.namelist() if f.endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError("No CSV file found in downloaded Tranco archive.")
        extracted_name = csv_names[0]
        zip_ref.extract(extracted_name, raw_dir)

        # Standardize target filename
        extracted_path = raw_dir / extracted_name
        if extracted_path != target_csv:
            extracted_path.rename(target_csv)

    if zip_path.exists():
        zip_path.unlink()

    print(f"[+] Tranco list successfully saved to {target_csv}")
    return target_csv


def check_cic_dataset(config: dict) -> None:
    """Check status of CIC-Bell-DNS-EXF-2021 dataset in data/raw/."""
    cic_dir = Path(config["paths"]["cic_bell_dir"])
    print(f"[*] Checking CIC-Bell dataset directory: {cic_dir}")
    if cic_dir.exists() and any(cic_dir.iterdir()):
        files = list(cic_dir.glob("*"))
        print(f"[+] Found {len(files)} files in {cic_dir}:")
        for f in files[:5]:
            print(f"    - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
        if len(files) > 5:
            print(f"    ... and {len(files) - 5} more files.")
    else:
        print(f"[!] CIC-Bell dataset not found in {cic_dir}.")
        print("    Please refer to data/README.md for download instructions:")
        print("    https://www.unb.ca/cic/datasets/dns-exf-2021.html")


def main():
    parser = argparse.ArgumentParser(description="Acquire DNS datasets for analysis and training.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to configuration file")
    parser.add_argument("--tranco", action="store_true", help="Download Tranco top-1M domain list")
    parser.add_argument("--check-cic", action="store_true", help="Check CIC-Bell dataset status")
    parser.add_argument("--all", action="store_true", help="Run all dataset acquisition tasks")

    args = parser.parse_args()
    config = load_config(args.config)

    if not (args.tranco or args.check_cic or args.all):
        parser.print_help()
        sys.exit(0)

    if args.tranco or args.all:
        acquire_tranco(config)

    if args.check_cic or args.all:
        check_cic_dataset(config)


if __name__ == "__main__":
    main()
