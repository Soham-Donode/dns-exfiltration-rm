"""
Builds every notebook in notebooks/:

    01_paper_csv_audit.ipynb      (scripts/build_csv_audit_notebook.py)
    02_baseline.ipynb             paper's model, trained on real data, tested on real + synthetic
    03_step1_temporal.ipynb       + host temporal features
    04_step2_domain.ipynb         + per-domain statistical features
    05_step3_two_layer.ipynb      + augmented training and the two-layer model
    06_step4_behaviour.ipynb      + content-agnostic behaviour model (targets the families step 3 misses)
    07_multiseed_comparison.ipynb every model over several seeds -> mean +/- std (the numbers to report)

Notebooks 02-07 are self-contained for Colab: they embed the five modules from src/ with %%writefile.

    python scripts/build_notebooks.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbtools import ROOT, code, embed, make_notebook, md  # noqa: E402

MODULES = ["pcap_features.py", "dns_simulator.py", "evasion_data.py", "stateful_features.py", "experiment.py"]

COMMON_INTRO = r"""
### Data used by every notebook (identical across 02-07 for a given seed)
| Test set | What it is | Metric |
|---|---|---|
| **real test** | last 30% (by time) of every CIC-Bell-DNS-EXF-2021 capture; training uses the first 70% | F1, FPR |
| **evasive** | 12 synthetic exfiltration families (base32 / hex / base64 / word encoding, labels 12-63 chars, with and without sequence tags, A / AAAA / TXT / NULL, fast / medium / slow) | recall per encoding |
| **hard benign** | synthetic legitimate names that look like exfiltration (CDN / cloud hashes, GUIDs, telemetry, DKIM, DNSBL, 0x20 case, punycode) | FPR |
| **mixed** | one host's normal traffic with ONE attacker diluted in at 1 / 5 / 20 / 50% of its queries, plus a clean stream | recall per rate, FPR |

Synthetic captures use **snaplen 96 like the real ones** and go through the same extractor. All numbers are per query, in %.

### How to run on Colab
1. Zip your PCAP archives into **one** file whose name says the scenario, e.g. `PCAP_Light.zip` (containing `Attacks.zip` and `Benign.zip`). Add `PCAP_Heavy.zip` the same way if you have it.
2. **File -> Upload notebook**, then **Runtime -> Run all**, and pick the zip(s) when the file picker appears.
3. Results download as a zip at the end.

Tip: set `CONFIG["work_dir"]` to a Google Drive folder (and `data_source="drive"`) to reuse the prepared data across notebooks instead of rebuilding it each time.

### Caveats
- Synthetic traffic is an **approximation** of real naming schemes, not captured traffic; use its numbers to compare models, not as deployment estimates.
- The captures cut names at 96 bytes, so the domain of long queries is invisible; those queries share one `<trunc>` bucket per host in the per-domain features.
"""

STEPS = {
    "02_baseline": dict(
        title="02. Baseline: the paper's model",
        steps=["baseline"], models=["RF", "GNB", "MLP", "SVM", "LR"], focus="baseline",
        intro=r"""
The base paper's first layer: **14 stateless features per query** (Table 1, F17-F30) and the five classifiers it compares (GNB, RF, MLP, SVM, LR), trained on the **real** captures only. `baseline+window` adds the paper's window rule: a window of 100 queries is flagged when more than 40% of its queries are flagged.

**Expected:** ~100% on the real test set (the real attack is easy), much lower on evasive and diluted traffic. This is the number every later step is compared against.
""",
        outro=r"""
- **real test** near 100% reproduces the paper's result once the exfiltration is actually in the data.
- **evasive / mixed** recall is the weakness the next steps target. The window rule (`baseline+window`) cannot help a diluted attack: at 1-20% dilution a window never reaches 40% flagged queries.
"""),
    "03_step1_temporal": dict(
        title="03. Step 1: + host temporal features",
        steps=["baseline", "step1"], models=["RF"], focus="step1",
        intro=r"""
Adds **causal host-level temporal features** over the host's last 100 queries (the paper's tau): query rate, mean inter-arrival time and its coefficient of variation (regularity), share of distinct names and base domains, mean / spread of DNS message length, mean entropy. This is the frequency feature the original project computed incorrectly (it was always 1.0), done properly.

Model: Random Forest on the 14 paper features + 9 host features, trained on the real captures only.

**Hypothesis:** helps when an attacker dominates a host's traffic; cannot isolate an attacker diluted among normal lookups, because host-level statistics mix both.
""",
        outro=r"""
- Compare **step1** with **baseline** row by row in the delta table. Gains on pure evasive streams with fast timing are expected; slow streams and low dilution rates are the hard cases.
- Feature importance shows whether the model leans on the new temporal features or still on the per-query ones.
"""),
    "04_step2_domain": dict(
        title="04. Step 2: + per-domain statistical features",
        steps=["baseline", "step1", "step2"], models=["RF"], focus="step2",
        intro=r"""
Adds **causal per-domain statistics** over the last 10 minutes of traffic from the same host to the same base domain: number of queries and of distinct subdomains, share of unique subdomains, subdomain length / entropy, mean gap, share of the host's traffic, bytes sent. An exfiltration domain receives many *unique* subdomains; a normal site is looked up with the same few names.

Model: Random Forest on paper features + host features + 9 domain features, trained on the real captures only.

**Hypothesis:** separates a diluted attacker domain from the host's normal traffic (what step 1 cannot), and catches short-label families that look like CDN hashes one query at a time.
""",
        outro=r"""
- The **mixed** columns are the test for this step: does recall at 1-20% dilution improve over step 1?
- Watch **hard_fpr**: CDN / cloud domains also receive many unique subdomains, so per-domain features can raise false positives there.
"""),
    "05_step3_two_layer": dict(
        title="05. Step 3: augmented training + two-layer model",
        steps=["baseline", "step1", "step2", "step3a", "step3b"], models=["RF"], focus="step3b",
        intro=r"""
Two changes, reported separately so their effects can be told apart:
- **step3a - augmented training.** The step-2 model also trains on synthetic exfiltration - pure attack streams *and* diluted streams where the attacker hides in normal traffic (5% and 20%, separate from the test streams) - and on the first half of the hard-benign stream. To stay honest it is evaluated **leave-one-encoding-out**: the model scoring base32 traffic never saw any base32 family in training, and so on for hex, base64 and word encoding.
- **step3b - two-layer model** (the paper's hybrid idea with real stateful signals). Layer 1: a per-query classifier on character-model features (bigram likelihood, hex / digit fractions, transitions, entropy of the longest label). Layer 2: step-2 features + the layer-1 score + its causal host / domain averages. Same augmentation and leave-one-encoding-out evaluation; layer-1 scores for training rows are out-of-fold.

Single seed; notebook 07 repeats everything over several seeds.
""",
        outro=r"""
- **Final table**: one row per model, same metrics as every earlier notebook. A real improvement keeps `real_f1` / `real_fpr` at baseline level and raises evasive / mixed recall without raising `hard_fpr` / `mixed_fpr`.
- step3a vs step2 is the effect of more diverse training data; step3b vs step3a is the effect of the two-layer design.
- The word encoding has low entropy and English-like bigrams; if it stays low, character statistics cannot catch it and only behaviour (step 1-2 features) can.
- **Read pure-stream recall for step 3 with care.** Once a model has trained on pure attack streams of other encodings, it can recognise the *shape* of a pure stream (one domain, every name new) regardless of encoding. The **mixed** columns are the realistic test: the attacker shares the host with normal traffic.
"""),
    "06_step4_behaviour": dict(
        title="06. Step 4: content-agnostic behaviour model",
        steps=["baseline", "step1", "step2", "step3a", "step3b", "step4"], models=["RF"], focus="step4 behaviour",
        intro=r"""
**Why step 3 still misses some families.** In training, character features (entropy, letter mix, bigram likelihood) separate every *other* encoding perfectly, so the model never needs the per-domain volume pattern. An unseen encoding that does not look random - the word encoding most of all - slips through.

**Step 4 adds a second, behaviour-only model** that sees *no character content*: per host -> domain volume, subdomain diversity, novelty (share of never-seen subdomains, distinct subdomains ever sent), timing regularity and bytes, over 10 minutes and 1 hour. It cannot take the "looks random" shortcut.
- **step4 behaviour** - the behaviour-only model on its own
- **step4** - final detector: a query is flagged if step 3a **or** the behaviour model flags it

Same augmentation and leave-one-encoding-out evaluation as step 3.

**Cut-off names.** The captures cut names at 96 bytes, so the domain of long names is unknown and all of a host's long names share one `<trunc>` bucket. That bucket looks like one busy attacker domain (hundreds of unique subdomains) even for legitimate telemetry / CDN names, so the behaviour model **abstains on cut-off names** and leaves them to step 3a. Notebook 07 also reports the full-packet setting, where every domain is visible.
""",
        outro=r"""
- **step4 behaviour** shows what behaviour alone can catch; **step4** shows what it adds to step 3a. Check the per-family table for the word encoding, hex with 63-char labels and short base32 without a sequence tag.
- An OR can only add detections, so it can also only add false positives: compare `mixed_fpr` and `hard_fpr` with step 3a.
- On these captures most attack families have names longer than the cut-off, where the behaviour model abstains, so its gain here is limited; notebook 07's full-packet results show what it does when domains are visible.
- At 1% dilution an attacker sends fewer unique subdomains per hour than a busy CDN does; behaviour alone cannot separate those, so expect the gain at 5% and above.
"""),
}


def notebook(name, spec):
    cells = [md(f"# {spec['title']}\n" + spec["intro"] + COMMON_INTRO)]
    cells.append(code(r'''
import os, sys, json, time, shutil, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
pd.set_option("display.max_columns", 40); pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)

NOTEBOOK = "__NAME__"
CONFIG = {
    "data_source": "upload" if IN_COLAB else "local",      # upload | drive | local  (real PCAP zips)
    "drive_path": "/content/drive/MyDrive/datasets/CIC-Bell-DNS-EXF-2021-PCAP",
    "local_paths": [p for p in os.environ.get("DNS_PCAP", "../data/raw/pcap").split(os.pathsep) if p],
    "work_dir": "/content/work" if IN_COLAB else os.environ.get("DNS_WORK", "../data/processed/work"),
    "output_dir": f"/content/results/{NOTEBOOK}" if IN_COLAB else os.environ.get("DNS_OUT", f"../results/{NOTEBOOK}"),
    "steps": __STEPS__,
    "baseline_models": __MODELS__,
    "experiment": {},          # overrides for experiment.DEFAULTS, e.g. {"n_mixed": 3000} for a faster run
}
print(f"{NOTEBOOK} | Colab: {IN_COLAB} | steps: {CONFIG['steps']}")
'''.replace("__NAME__", name).replace("__STEPS__", repr(spec["steps"])).replace("__MODELS__", repr(spec["models"]))))
    cells.append(md("## 1. Modules (embedded copies of `src/`)"))
    cells += [embed(m) for m in MODULES]
    cells.append(md("## 2. Data: real PCAPs + synthetic test traffic"))
    cells.append(code(r'''
work = Path(CONFIG["work_dir"]); work.mkdir(parents=True, exist_ok=True)
if CONFIG["data_source"] == "upload":
    from google.colab import files
    inputs = []
    for fname, blob in files.upload().items():
        p = work / "uploads" / fname; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(blob); inputs.append(str(p))
elif CONFIG["data_source"] == "drive":
    from google.colab import drive
    drive.mount("/content/drive"); inputs = [CONFIG["drive_path"]]
else:
    inputs = CONFIG["local_paths"]
inputs = [i for i in inputs if Path(i).exists()]
assert inputs, "No PCAP input found - check CONFIG"
print("Inputs:", inputs)

sys.path.insert(0, os.getcwd())
import importlib
import pcap_features, evasion_data, stateful_features, experiment as ex
for m in (pcap_features, evasion_data, stateful_features, ex):
    importlib.reload(m)
D = ex.prepare(inputs, work, CONFIG["experiment"])
'''))
    cells.append(code(r'''
print("Real captures - label audit (long_msg_not_labelled and rule_hits_in_benign_file must be 0):")
display(D["real_audit"])
print("Synthetic captures:")
display(D["manifest"].groupby("kind")[["queries", "exfil"]].sum())
'''))
    cells.append(md("## 3. Train and evaluate"))
    cells.append(code(r'''
preds, infos = ex.run_detectors(D, CONFIG["steps"], baseline_models=CONFIG["baseline_models"])
summary, detail = ex.evaluate(D, preds)
print("Key metrics (%). Rows = metrics, columns = models.")
display(ex.key_table(summary).T)
'''))
    if len(spec["steps"]) > 1:
        cells.append(code(r'''
print("Change versus the baseline (percentage points; + is better for recall/F1, - is better for FPR):")
display(ex.delta_vs(summary, "baseline").T)
'''))
    cells.append(code(r'''
print("Recall (%) per family: pure evasive streams and mixed streams")
display(detail.pivot_table(index=["test", "family"], columns="detector", values="value").round(1))
'''))
    cells.append(code(r'''
focus = "__FOCUS__"
print(f"Top features of '{focus}' (Random Forest importance):")
display(ex.feature_importance(infos[focus]).to_frame("importance"))
'''.replace("__FOCUS__", spec["focus"])))
    cells.append(md("## 4. Save"))
    cells.append(code(r'''
out = ex.save(CONFIG["output_dir"], summary, detail, D, extra={"notebook": NOTEBOOK, "steps": CONFIG["steps"]})
print("Saved to", out)
for f in sorted(out.iterdir()):
    print(f"  {f.name:28s} {f.stat().st_size / 1e3:8.1f} KB")
if IN_COLAB:
    shutil.make_archive(f"/content/{NOTEBOOK}_results", "zip", out)
    from google.colab import files
    files.download(f"/content/{NOTEBOOK}_results.zip")
'''))
    cells.append(md("## 5. How to read this\n" + spec["outro"]))
    return make_notebook(cells)


MULTISEED_INTRO = r"""
# 07. Multi-seed comparison: the numbers to report

Notebooks 02-06 use one seed, so one draw of synthetic traffic. A single draw can be lucky or unlucky: a family is either caught or missed, and with few families one draw moves the averages a lot. This notebook repeats the **whole experiment for several seeds**: every seed generates new payloads, new attacker domains, new background traffic and new timing, and retrains every model with a new random state. The real-data split stays fixed (it is time-ordered).

It reports every metric as **mean ± standard deviation over seeds**. Report these numbers, not the single-seed ones, and treat a difference smaller than about two standard deviations as noise.

It runs **two settings**:
- **as captured** - synthetic traffic cut at 96 bytes like the real PCAPs, so long names lose their domain;
- **full packets** - synthetic traffic kept whole, as a deployed sensor would see it. The real data is unchanged (it only exists cut). This is the setting that shows what per-domain behaviour features can do.
""" + COMMON_INTRO


def multiseed_notebook():
    cells = [md(MULTISEED_INTRO)]
    cells.append(code(r'''
import os, sys, json, time, shutil, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
pd.set_option("display.max_columns", 40); pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)

NOTEBOOK = "07_multiseed_comparison"
CONFIG = {
    "data_source": "upload" if IN_COLAB else "local",
    "drive_path": "/content/drive/MyDrive/datasets/CIC-Bell-DNS-EXF-2021-PCAP",
    "local_paths": [p for p in os.environ.get("DNS_PCAP", "../data/raw/pcap").split(os.pathsep) if p],
    "work_dir": "/content/work_seeds" if IN_COLAB else os.environ.get("DNS_WORK", "../data/processed/work_seeds"),
    "output_dir": f"/content/results/{NOTEBOOK}" if IN_COLAB else os.environ.get("DNS_OUT", f"../results/{NOTEBOOK}"),
    "seeds": [1, 2, 3, 4, 5],
    "steps": ["baseline", "step1", "step2", "step3a", "step3b", "step4"],
    "settings": {"as captured": {}, "full packets": {"snaplen": 65535}},
    "experiment": {},
}
print(f"{NOTEBOOK} | Colab: {IN_COLAB} | seeds {CONFIG['seeds']}")
'''))
    cells.append(md("## 1. Modules (embedded copies of `src/`)"))
    cells += [embed(m) for m in MODULES]
    cells.append(md("## 2. Run every step for every seed"))
    cells.append(code(r'''
work = Path(CONFIG["work_dir"]); work.mkdir(parents=True, exist_ok=True)
if CONFIG["data_source"] == "upload":
    from google.colab import files
    inputs = []
    for fname, blob in files.upload().items():
        p = work / "uploads" / fname; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(blob); inputs.append(str(p))
elif CONFIG["data_source"] == "drive":
    from google.colab import drive
    drive.mount("/content/drive"); inputs = [CONFIG["drive_path"]]
else:
    inputs = CONFIG["local_paths"]
inputs = [i for i in inputs if Path(i).exists()]
assert inputs, "No PCAP input found - check CONFIG"

sys.path.insert(0, os.getcwd())
import importlib
import pcap_features, evasion_data, stateful_features, experiment as ex
for m in (pcap_features, evasion_data, stateful_features, ex):
    importlib.reload(m)
RUNS = {}
for setting, overrides in CONFIG["settings"].items():
    t0 = time.time()
    print(f"=== {setting}")
    RUNS[setting] = ex.run_seeds(inputs, work / setting.replace(" ", "_"), CONFIG["seeds"], CONFIG["steps"],
                                 {**CONFIG["experiment"], **overrides})
    print(f"{setting}: {(time.time() - t0) / 60:.1f} min")
'''))
    cells.append(md("## 3. Results: mean ± std over seeds (%)"))
    cells.append(code(r'''
AGG = {s: ex.aggregate(sums) for s, (sums, dets) in RUNS.items()}
for s, (MEAN, STD, TABLE) in AGG.items():
    print(f"\n##### {s}")
    display(TABLE.T)
'''))
    cells.append(code(r'''
for s, (MEAN, STD, TABLE) in AGG.items():
    print(f"\n##### {s}: mean change versus the baseline (percentage points) and in units of the step's std")
    delta = (MEAN - MEAN.loc["baseline"]).drop(index="baseline")
    z = (delta / STD.drop(index="baseline").replace(0, np.nan)).round(1)
    display(pd.concat({"delta": delta.round(1), "delta / std": z}, axis=1).T.sort_index(level=1, sort_remaining=False))
'''))
    cells.append(code(r'''
for s, (sums, dets) in RUNS.items():
    print(f"\n##### {s}: recall per family, mean over seeds (%)")
    fam = dets.groupby(["test", "family", "detector"]).value.mean().unstack("detector")
    display(fam[[c for c in AGG[s][0].index if c in fam.columns]].round(1))
'''))
    cells.append(code(r'''
for s, (sums, dets) in RUNS.items():
    print(f"\n##### {s}: every seed separately (diluted-attack recall, mean over 1/5/20/50%)")
    mix_cols = [c for c in sums.columns if c.startswith("mixed_recall_")]
    display(sums.assign(diluted=sums[mix_cols].mean(axis=1)).pivot_table(index="seed", columns="detector", values="diluted").round(1))
'''))
    cells.append(md("## 4. Save"))
    cells.append(code(r'''
out = Path(CONFIG["output_dir"]); out.mkdir(parents=True, exist_ok=True)
for s, (sums, dets) in RUNS.items():
    tag = s.replace(" ", "_")
    MEAN, STD, TABLE = AGG[s]
    sums.to_csv(out / f"{tag}_summary_per_seed.csv", index=False)
    dets.to_csv(out / f"{tag}_detail_per_seed.csv", index=False)
    MEAN.round(3).to_csv(out / f"{tag}_summary_mean.csv"); STD.round(3).to_csv(out / f"{tag}_summary_std.csv")
    TABLE.to_csv(out / f"{tag}_summary_mean_std.csv")
(out / "run_config.json").write_text(json.dumps({"seeds": CONFIG["seeds"], "steps": CONFIG["steps"],
                                                  "settings": CONFIG["settings"], "experiment": CONFIG["experiment"]}, indent=2))
print("Saved to", out)
if IN_COLAB:
    shutil.make_archive(f"/content/{NOTEBOOK}_results", "zip", out)
    from google.colab import files
    files.download(f"/content/{NOTEBOOK}_results.zip")
'''))
    cells.append(md(r"""
## 5. How to read this
- A step is better than the baseline on a metric when the mean difference is clearly larger than the spread (`delta / std` well above 2).
- Families that stay near 0% in every seed are a real blind spot, not bad luck.
- `real_f1` / `real_fpr` must stay at baseline level; an improvement that costs false positives on real or hard-benign traffic is not an improvement.
- Compare the two settings for step 4: on cut captures the behaviour model abstains on most attack traffic; with full packets it sees every domain.
"""))
    return make_notebook(cells)


def main():
    out_dir = ROOT / "notebooks"
    out_dir.mkdir(exist_ok=True)
    for name, spec in STEPS.items():
        path = out_dir / f"{name}.ipynb"
        path.write_text(json.dumps(notebook(name, spec), indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] {path.relative_to(ROOT)}")
    path = out_dir / "07_multiseed_comparison.ipynb"
    path.write_text(json.dumps(multiseed_notebook(), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {path.relative_to(ROOT)}")
    import build_csv_audit_notebook
    build_csv_audit_notebook.main()


if __name__ == "__main__":
    main()
