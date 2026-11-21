# Data

Nothing in `data/raw/` or `data/processed/` is committed (see `.gitignore`). This file says what goes where.

## Layout

```text
data/
├── raw/
│   ├── pcap/                      CIC-Bell-DNS-EXF-2021 raw captures  -> notebooks 02-07
│   │   ├── PCAP_Light/            Attacks.zip, Benign.zip   (folder/zip name must contain "light")
│   │   └── PCAP_Heavy/            Attacks.zip, Benign.zip   (optional, name must contain "heavy")
│   └── cic_exf_csv/               CIC-Bell-DNS-EXF-2021 released CSVs -> notebook 01 only
│       ├── Attack_Light_Benign/   Attacks.zip, Benign.zip
│       ├── Attack_heavy_Benign/   Attacks.zip, Benign.zip
│       └── Benign.zip             day-1 benign
└── processed/
    └── work/                      created by the notebooks: extracted features, synthetic
                                   captures, prepared.pkl cache (safe to delete)
```

## Source

**CIC-Bell-DNS-EXF-2021** (Canadian Institute for Cybersecurity + Bell Canada):
https://www.unb.ca/cic/datasets/dns-exf-2021.html
Base paper: Mahdavifar et al., *Lightweight Hybrid Detection of Data Exfiltration using DNS based on Machine Learning*, ICCNS 2021, https://doi.org/10.1145/3507509.3507520

Do not confuse it with **CIC-Bell-DNS-2021** (benign / malware / phishing / spam *domains*). That is a different dataset and is not used here.

## Known problems with the released data (found in this project)

1. **The released CSVs contain almost no exfiltration.** 1 query to the attacker domain in 294,353 "attack" rows; the attack rows are background Windows traffic (PTR, mDNS, NetBIOS) labelled by capture time. Notebook 01 documents this.
2. **The PCAPs were captured with snaplen = 96.** Every packet is cut to 96 bytes, so DNS names stop after ~36 characters. The exfiltration queries *are* in the PCAPs; their true size (~265 bytes vs <= 72 for benign) is recovered from the UDP length field. `src/pcap_features.py` handles this.
3. One file name has a typo (`stateful_features-_light_benign`); the loaders tolerate it.

## Synthetic data

Notebooks 02-07 generate their synthetic test traffic themselves (`src/evasion_data.py`, fixed seed), so nothing needs to be downloaded for it.
