# DNS Exfiltration Detection & Tunneling Analysis

A modular, production-ready machine learning framework for simulating, capturing, engineering features, and detecting DNS-based data exfiltration and tunneling attacks.

---

## Project Structure

```text
dns-exfiltration-detection/
│
├── .gitignore
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml           # Store paths, hyperparameters, and constants
│
├── data/
│   ├── raw/                  # Place Tranco and CIC-Bell datasets here (gitignored)
│   ├── processed/            # Cleaned CSVs/PCAPs ready for ML (gitignored)
│   └── README.md             # Instructions on how to download the datasets
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── download_data.py      # Week 1: Dataset acquisition
│   ├── dns_simulator.py      # Week 2: Scapy simulation engine
│   ├── capture_traffic.py    # Week 3: Tcpdump/Scapy packet capture
│   ├── features.py           # Week 4: Entropy, ratios, frequency calculations
│   └── train.py              # Week 5: Model training pipeline
│
└── models/
    └── .gitkeep              # Save trained models here (.pkl, .json)
```

---

## 5-Week Implementation Roadmap

| Phase | Module | Description |
|---|---|---|
| **Week 1** | `src/download_data.py` | Acquire benign (Tranco top 1M) and malicious (CIC-Bell-DNS-EXF-2021) datasets. |
| **Week 2** | `src/dns_simulator.py` | Scapy simulation engine with chunking, multiple encodings (Base32, Base64, Hex), and query types (A, TXT, AAAA, CNAME). |
| **Week 3** | `src/capture_traffic.py` | Live DNS packet sniffer (`udp port 53`) and PCAP query extraction engine. |
| **Week 4** | `src/features.py` | Extract information-theoretic features (Shannon entropy), character ratios, and structural properties. |
| **Week 5** | `src/train.py` | Train, cross-validate, and evaluate ML models (Random Forest, XGBoost, LightGBM) and persist model artifacts. |

---

## Installation & Setup

### 1. Clone & Set Up Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration
All hyperparameters, network settings, and directory paths are maintained in [`configs/config.yaml`](file:///home/hyperghost/github/dns-exfiltration-rm/configs/config.yaml).

---

## Usage Guide

### Week 1: Dataset Acquisition
```bash
# Download Tranco top 1M benign list
python3 src/download_data.py --tranco

# Check status of downloaded/extracted datasets
python3 src/download_data.py --check-cic
```

### Week 2: Simulate Exfiltration Traffic
```bash
# Generate exfiltration DNS queries in dry-run mode
python3 src/dns_simulator.py --domain tunnel.corp.net --text "ConfidentialDataToken123" --encoding base32 --dry-run

# Craft and send live packets to target resolver and save to PCAP
python3 src/dns_simulator.py --server 127.0.0.1 --pcap-out data/raw/simulated_exfil.pcap
```

### Week 3: Packet Capture & PCAP Parsing
```bash
# Parse DNS queries from an existing PCAP capture
python3 src/capture_traffic.py --pcap data/raw/sample_capture.pcap

# Sniff live DNS traffic on local interface
python3 src/capture_traffic.py --interface lo --count 50 --out-pcap data/raw/live_capture.pcap
```

### Week 4: Feature Extraction
```bash
# Inspect features of a single query
python3 src/features.py --domain "dGhpc2lzYXNlY3JldGRhdGE.tunnel.attacker.com"

# Extract features from a list of domains and export to CSV
python3 src/features.py --input-csv data/raw/domains.csv --output-csv data/processed/dns_features.csv
```

### Week 5: Model Training & Evaluation
```bash
# Train Random Forest classifier using configs/config.yaml
python3 src/train.py --algo random_forest

# Train using XGBoost
python3 src/train.py --algo xgboost
```

---

## Jupyter Notebooks

Interactive notebooks for EDA, prototyping features, and evaluating models:
- [`notebooks/01_eda.ipynb`](file:///home/hyperghost/github/dns-exfiltration-rm/notebooks/01_eda.ipynb): Distribution analysis of query lengths, Shannon entropy, and character frequencies.
- [`notebooks/02_feature_engineering.ipynb`](file:///home/hyperghost/github/dns-exfiltration-rm/notebooks/02_feature_engineering.ipynb): Prototyping lexical and structural DNS query features.
- [`notebooks/03_model_training.ipynb`](file:///home/hyperghost/github/dns-exfiltration-rm/notebooks/03_model_training.ipynb): Model benchmarking, ROC curves, confusion matrices, and feature importances.
