# DNS Exfiltration Detection: reproducing and improving a hybrid ML detector

This project reproduces the detector from

> S. Mahdavifar et al., **"Lightweight Hybrid Detection of Data Exfiltration using DNS based on Machine Learning"**, ICCNS 2021. https://doi.org/10.1145/3507509.3507520

on its own dataset (**CIC-Bell-DNS-EXF-2021**), checks how it behaves on traffic the paper never tested, and improves it step by step with temporal and per-domain statistical features.

## The story so far

| # | What we did | What we found |
|---|---|---|
| 1 | Reproduced the paper on the **released CSVs** (notebook 01) | RF reached 75–91% instead of the paper's 99.9%. A data audit showed why: the CSVs contain **1 exfiltration query in 294,353 "attack" rows**. The rest is background Windows traffic labelled by capture time. |
| 2 | Went back to the **raw PCAPs** | The captures use **snaplen 96**, so every DNS name is cut after ~36 characters. The exfiltration queries are there; their true size (~265 bytes vs ≤ 72 for benign) survives in the UDP header. We wrote our own extractor that handles this and labels queries by their shape, not by capture time. |
| 3 | **Baseline** on the corrected data (notebook 02) | The paper's model scores ~100% on the real data, matching the paper. |
| 4 | Tested the baseline on **synthetic evasive traffic** (notebook 02) | It collapses: other encodings, short labels and diluted (low-and-slow) attacks are barely detected, and the paper's window rule misses any attack below ~40% of a host's traffic. |
| 5 | **Improvements**, one per notebook (03–06) | Host temporal features → per-domain statistical features → augmented training + two-layer model → behaviour model, each compared with the baseline on identical data and metrics. |
| 6 | **Behaviour model** (notebook 06) | A second model with no character features (per-domain volume, novelty, timing) catches encodings that do not look random. |
| 7 | **Stabilised the numbers** (notebook 07) | Every model over 5 seeds, all 12 attack families, cut and full-packet traffic; results reported as mean ± std. |

## Repository layout

```text
dns-exfiltration-rm/
├── notebooks/                       run in order; each is self-contained for Google Colab
│   ├── 01_paper_csv_audit.ipynb     paper reproduction on the released CSVs + data audit
│   ├── 02_baseline.ipynb            paper's model (5 classifiers) on real + synthetic test sets
│   ├── 03_step1_temporal.ipynb      + host temporal features
│   ├── 04_step2_domain.ipynb        + per-domain statistical features
│   ├── 05_step3_two_layer.ipynb     + augmented training, two-layer model
│   ├── 06_step4_behaviour.ipynb     + content-agnostic behaviour model
│   └── 07_multiseed_comparison.ipynb every model, 5 seeds, cut and full packets -> mean ± std (numbers to report)
├── src/
│   ├── pcap_features.py             PCAP -> per-query features (snaplen-aware, stdlib only)
│   ├── stateful_features.py         causal host-temporal and per-domain features
│   ├── evasion_data.py              synthetic evasive exfiltration, hard benign, mixed streams
│   ├── dns_simulator.py             DNS exfiltration chunking/encoding engine (used by evasion_data)
│   └── experiment.py                shared data prep, detectors, metrics for notebooks 02-07
├── scripts/
│   ├── build_notebooks.py           regenerates all notebooks from src/ (run after editing src/)
│   ├── build_csv_audit_notebook.py  notebook 01
│   └── nbtools.py
├── tests/                           python -m unittest discover tests
├── data/                            datasets (not committed) - see data/README.md
├── results/                         notebook outputs (not committed); colab_runs/ = earlier Colab result zips
└── docs/                            base paper PDF (local only)
```

## How to run

**On Google Colab (recommended).**
1. Zip the PCAP archives into one file whose name contains the scenario, e.g. `PCAP_Light.zip` with `Attacks.zip` + `Benign.zip` inside (same for `PCAP_Heavy.zip`).
2. Open a notebook (File → Upload notebook) → Runtime → Run all → choose the zip when asked.
3. Each notebook downloads a results zip at the end (`summary.csv` holds the key metrics).

**Locally.**
```bash
pip install -r requirements.txt
# put the PCAP zips under data/raw/pcap/PCAP_Light/ (see data/README.md), then:
jupyter notebook notebooks/02_baseline.ipynb
python -m unittest discover tests        # 32 tests, ~15 s
python scripts/build_notebooks.py        # after changing anything in src/
```

## Evaluation (identical in notebooks 02–07)

| Test set | Content | Metric |
|---|---|---|
| real test | last 30% (by time) of every real capture; models train on the first 70% | F1, false-positive rate |
| evasive | 12 synthetic families: base32 / hex / base64 / word encoding, 12–63-char labels, with/without sequence tags, A/AAAA/TXT/NULL, fast/medium/slow | recall per encoding |
| hard benign | synthetic legitimate names that look suspicious (CDN/cloud hashes, GUIDs, telemetry, DKIM, DNSBL, 0x20 case, punycode) | false-positive rate |
| mixed | one host's normal traffic with one attacker diluted in at 1 / 5 / 20 / 50%, plus a clean stream | recall per rate, false-positive rate |

Models that train on synthetic data (step 3) are evaluated **leave-one-encoding-out**: the model that scores, e.g., hex traffic never saw any hex family during training.

## Results

**Mean ± standard deviation over 5 seeds** (notebook 07; **light + heavy PCAPs**, 341,101 real queries of which 251,568 exfiltration; Random Forest; % of queries). Each seed draws new synthetic traffic and new model randomness; the real split is fixed. Diluted-attack rows cover all 12 attack families. Step 3 and 4 models are evaluated leave-one-encoding-out (never trained on the encoding they are scored on). Light-only results are kept in `results/light_only/`. Adding the heavy data left the conclusions unchanged: step 3a / 4 moved by at most ~11 points (mostly upward, e.g. step 3a at 5% dilution with full packets 72.5 → 80.0), while steps 1 and 2, trained on real data only, got worse (they fit the single real tool more closely).

Two settings:
- **As captured**: synthetic traffic cut at 96 bytes like the real PCAPs, so long names lose their domain.
- **Full packets**: synthetic traffic kept whole, as a deployed sensor would see it (the real data only exists cut).

### As captured (snaplen 96)

| Metric | baseline | + paper window rule | step 1 temporal | step 2 domain | step 3a augmented | step 3b two-layer | **step 4 = 3a OR behaviour** |
|---|---|---|---|---|---|---|---|
| real test F1 | 100.0 ± 0.0 | 99.8 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| real test FPR | 0.0 ± 0.0 | 1.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.2 ± 0.1 |
| diluted attack recall, 1% | 0.5 ± 1.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 25.6 ± 6.9 | **32.6 ± 0.7** | 25.6 ± 6.9 |
| diluted attack recall, 5% | 0.4 ± 0.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 42.6 ± 10.9 | 33.3 ± 0.1 | **43.4 ± 11.0** |
| diluted attack recall, 20% | 0.5 ± 0.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.1 ± 0.1 | 54.0 ± 13.4 | 33.3 ± 0.0 | **68.9 ± 8.9** |
| diluted attack recall, 50% | 0.4 ± 0.7 | 0.0 ± 0.0 | 9.8 ± 4.7 | 23.8 ± 5.7 | 40.1 ± 7.7 | 34.1 ± 1.8 | **60.5 ± 11.1** |
| pure attack streams, 12 families\* | 0.6 ± 1.1 | 0.0 ± 0.0 | 30.5 ± 19.3 | 46.4 ± 9.6 | 92.9 ± 3.4 | 34.2 ± 1.8 | **99.9 ± 0.1** |
| false alarms, normal traffic | 0.1 ± 0.1 | 0.0 ± 0.0 | 0.1 ± 0.0 | 0.2 ± 0.1 | 0.1 ± 0.0 | 0.1 ± 0.1 | 0.1 ± 0.0 |
| false alarms, hard benign | 0.2 ± 0.4 | 0.0 ± 0.0 | 0.1 ± 0.1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |

### Full packets (domain always visible)

| Metric | baseline | step 3a augmented | **step 4 = 3a OR behaviour** |
|---|---|---|---|
| real test F1 / FPR | 100.0 / 0.0 | 100.0 / 0.0 | 100.0 / 0.2 |
| diluted attack recall, 1% | 0.2 ± 0.4 | 23.9 ± 2.7 | 23.9 ± 2.7 |
| diluted attack recall, 5% | 0.0 ± 0.0 | 80.0 ± 3.5 | **81.2 ± 3.2** |
| diluted attack recall, 20% | 0.0 ± 0.0 | 90.6 ± 1.2 | **99.9 ± 0.1** |
| diluted attack recall, 50% | 0.0 ± 0.0 | 91.6 ± 0.1 | **100.0 ± 0.0** |
| pure attack streams, 12 families\* | 0.0 ± 0.0 | 93.0 ± 3.2 | **100.0 ± 0.0** |
| false alarms, normal traffic | 0.1 ± 0.1 | 0.0 ± 0.0 | 0.1 ± 0.1 |
| false alarms, hard benign | 0.2 ± 0.4 | 0.0 ± 0.0 | 0.0 ± 0.0 |

\* pure-stream recall of augmented models is optimistic (they learn the *shape* of a pure attack stream); the diluted rows are the realistic test.

### What worked and what did not

| Step | Verdict | Evidence |
|---|---|---|
| Baseline (paper) | Matches the paper on its data, fails off it | F1 100 on 341k real queries; ≤ 1% of evasive or diluted attacks. The other four classifiers (GNB, MLP, SVM, LR) catch 10–49% but at 3–4% false alarms on normal traffic and 11–12% on hard benign (notebook 02). |
| Paper window rule | Does not help | Never fires below its 40% threshold; adds 1.0% false alarms on real data. |
| Step 1: host temporal | Not enough alone | Helps only at 50% dilution (10%); with more real-only training it overfits the one real tool further (worse than on light data). |
| Step 2: per-domain stats | Not enough alone | Same pattern: 24% at 50% dilution, ~0% below. Real-data-only training gives the model no reason to use these features. |
| **Step 3a: augmented training** | **Works** | Training on diverse *and* diluted simulated attacks: 43–54% (cut) / 80–92% (full packets) at 5–50% dilution on unseen encodings, ~0% false alarms. |
| Step 3b: two-layer | Narrow | Always catches the same 4 long-label families (33%); best at 1% on cut captures, weakest elsewhere. |
| **Step 4: + behaviour model** | **Works where domains are visible** | Full packets: 99.9% at 20%, 100% at 50%, 81% at 5% dilution, 0.1% false alarms. Cut captures: +15–20 points at 20–50% dilution. |
| Behaviour on cut-off names | Failed, fixed by abstaining | Long names share one `<trunc>` bucket that looks like an attacker domain: 2.7% false alarms until the model abstained on them. |

**Remaining blind spots:** 1% dilution (≤ 33%) and short base32 labels without a sequence tag (12–20 chars, < 10% even with full packets). Per-family numbers are in notebook 07.

### Why the behaviour model abstains on cut-off names
The captures cut names at 96 bytes, so a long name's domain is unknown and all of a host's long names fall into one `<trunc>` bucket. That bucket looks like a single attacker domain (hundreds of unique subdomains) even when it holds legitimate telemetry or CDN names; without abstaining, step 4 produced 2.7% false alarms on normal traffic. With full packets the problem does not exist.

Single-seed runs of every step are in notebooks 02–06 (seed 42); use the tables above when reporting.

## Future scope
- **Real traffic:** test against independent captures of real tunnelling tools (iodine, dnscat2, dns2tcp) and measure false alarms on real enterprise / home DNS instead of synthetic hard-benign names.
- **More kinds of attacks in training:** more tools, encodings and timing patterns (the model never saw the word encoding in training and still caught it through behaviour; more diversity should help the remaining blind spots).
- **Full-packet captures:** the released dataset is cut at 96 bytes; a re-capture without truncation would let per-domain features work on all traffic.
- **Low-rate attacks:** longer time horizons (hours to days) for the 1% case.

## Limitations

- Synthetic traffic approximates real naming schemes; it is not captured traffic. Use it to compare models, not to predict deployment numbers.
- The real captures hide the domain of long queries (snaplen 96); per-domain features group those queries into one `<trunc>` bucket per host.
- Only one exfiltration tool (DNSExfiltrator) exists in the real data.
