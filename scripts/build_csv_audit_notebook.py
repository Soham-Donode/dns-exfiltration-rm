"""
Builds notebooks/01_paper_csv_audit.ipynb

A self-contained, Colab-ready reproduction of the base paper:
  Mahdavifar et al., "Lightweight Hybrid Detection of Data Exfiltration using DNS
  based on Machine Learning", ICCNS 2021. https://doi.org/10.1145/3507509.3507520

Run from the repo root:
    python scripts/build_baseline_notebook.py
"""

import json
from pathlib import Path

from nbtools import make_notebook, md, code


CELLS = []


def M(s):
    CELLS.append(md(s))


def C(s):
    CELLS.append(code(s))


# ---------------------------------------------------------------------------
M(r"""
# 01. Paper reproduction on the released CSVs + data audit

Reproduces the base paper **Mahdavifar et al., "Lightweight Hybrid Detection of Data Exfiltration using DNS based on Machine Learning", ICCNS 2021** ([doi:10.1145/3507509.3507520](https://doi.org/10.1145/3507509.3507520)) on the real **CIC-Bell-DNS-EXF-2021** dataset.

**What the paper does (and what this notebook replicates)**

| Step | Paper (Sec. 6) | Here |
|---|---|---|
| Data | Light attack (+ light-benign), Heavy attack (+ heavy-benign); day-1 benign injected into both | Same grouping, built from the CSV file names |
| Features | 14 stateless + 16 stateful (Table 1) | Whatever the official CSVs contain, used as-is |
| Preprocessing | drop timestamp, NaN → 0, label-encode categoricals, `unique_ttl` → mean | Same |
| Split | shuffled 70 / 30 | Same (`shuffle=True`, not stratified) |
| Models | GNB, RF, MLP, SVM, LR (scikit-learn, no hyperparameters reported) | Same, scikit-learn defaults |
| Hybrid | stateless RF → bins [0,0.4) / [0.4,0.7) / [0.7,1]; if suspicious ratio in window τ=100 > δ=0.4 → stateful classifier | Simulated on the held-out split |

**Targets to match (paper Table 3, %):** RF light 99.94 acc, RF heavy 99.97 acc.

### How to run on Google Colab
1. Download the **CSV** archive from the CIC-Bell-DNS-EXF-2021 page (https://www.unb.ca/cic/datasets/dns-exf-2021.html).
2. Zip the whole dataset folder (`Attack_Light_Benign/`, `Attack_heavy_Benign/`, `Benign.zip`) into **one** zip, about 12 MB.
3. In Colab: **File → Upload notebook** → pick this file → **Runtime → Run all**. When the file picker appears (section 2), choose a single zip of the whole dataset folder.

Full run on the full dataset takes roughly 15–30 min on a free Colab CPU runtime (MLP and SVM dominate). Set `quick_mode=True` for a ~2 min smoke run.
""")

C(r"""
import os, re, json, time, zipfile, shutil, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)

warnings.filterwarnings("ignore")  # silences MLP/LR ConvergenceWarnings from default settings

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

SEED = 42
np.random.seed(SEED)
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 220)
print(f"Colab: {IN_COLAB} | pandas {pd.__version__} | scikit-learn {sklearn.__version__}")
""")

M(r"""
## 1. Configuration
Edit this cell only. `data_source`:
- `"upload"` (Colab default) – a file picker appears: choose **one zip of the whole dataset folder** (on Windows: right-click the folder → *Send to → Compressed (zipped) folder*). Don't upload the inner zips one by one: three of them are called `Benign.zip` and would overwrite each other.
- `"local"` – a folder **or a .zip** on this machine (`local_dir`), e.g. `"/content/data.zip"` if you dropped it into Colab's Files panel
- `"drive"` – mount Google Drive and read `drive_path` (a folder, or a `.zip`); survives runtime resets
- `"url"` – download from `data_url` (a direct download link)
""")

C(r"""
CONFIG = {
    "data_source": "upload" if IN_COLAB else "local",
    "drive_path":  "/content/drive/MyDrive/datasets/CIC-Bell-DNS-EXF-2021",
    "data_url":    "",
    "local_dir":   os.environ.get("CIC_EXF_DIR", "../data/raw/cic_exf_csv"),
    "work_dir":    "/content/cic_exf" if IN_COLAB else os.environ.get("CIC_EXF_WORK", "../data/processed/csv_audit_work"),
    "output_dir":  "/content/drive/MyDrive/dns_exfil_results" if IN_COLAB else os.environ.get("CIC_EXF_OUT", "../results/01_paper_csv_audit"),

    # Paper protocol
    "test_size": 0.30,
    "models": ["GNB", "RF", "MLP", "SVM", "LR"],
    "scale_features": False,       # paper does not mention scaling -> False for a faithful baseline

    # SVC is O(n^2); the paper does not say how it was trained on ~650k rows.
    # We fit it on a random subsample and score it on a random test subsample.
    "svm_max_train": 20_000,
    "svm_max_test": 50_000,

    # Hybrid layer (paper Sec. 5 / 6.3.1)
    "window": 100, "step": 100, "delta": 0.4, "bins": (0.4, 0.7),

    # Speed
    "quick_mode": os.environ.get("CIC_EXF_QUICK", "0") == "1",
    "quick_rows_per_file": 3_000,
    "save_models": True,
}
print(json.dumps(CONFIG, indent=2))
""")

M(r"""
## 2. Get the dataset
""")

C(r"""
WORK = Path(CONFIG["work_dir"]); WORK.mkdir(parents=True, exist_ok=True)


def extract_all_zips(roots, dest):
    # Extracts every zip under `roots` (and nested zips) into dest/<relative folder>/<zip-stem>/,
    # keeping the parent folders: Attack_heavy_Benign/Benign.zip and Benign.zip must not collide.
    dest = Path(dest); seen = set()
    while True:
        zips = [(Path(r), z) for r in roots + [dest] if Path(r).exists()
                for z in (Path(r).rglob("*.zip") if Path(r).is_dir() else [Path(r)])
                if z.suffix.lower() == ".zip" and z.resolve() not in seen]
        if not zips:
            return
        for root, z in zips:
            seen.add(z.resolve())
            rel = z.relative_to(root).parent if root.is_dir() else Path()
            target = dest / rel / z.stem
            if target.exists() and any(target.iterdir()):
                print(f"  already extracted: {z.name}")
                continue
            print(f"  extracting {z.name} -> {target}")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(target)


src = CONFIG["data_source"]
if src == "drive":
    from google.colab import drive
    drive.mount("/content/drive")
    data_root = Path(CONFIG["drive_path"])
elif src == "upload":
    from google.colab import files
    for name, blob in files.upload().items():
        (WORK / name).write_bytes(blob)
    data_root = WORK
elif src == "url":
    import urllib.request
    assert CONFIG["data_url"], "Set CONFIG['data_url']"
    target = WORK / "dataset.zip"
    if not target.exists():
        print("Downloading ...")
        urllib.request.urlretrieve(CONFIG["data_url"], target)
    data_root = WORK
else:
    data_root = Path(CONFIG["local_dir"])

assert data_root.exists(), f"Dataset path not found: {data_root}"
extract_all_zips([data_root], WORK)

search_roots = [WORK] if data_root.is_file() else [data_root, WORK]
CSV_FILES = sorted({p.resolve() for r in search_roots for p in Path(r).rglob("*.csv")})
print(f"Found {len(CSV_FILES)} CSV files")
assert CSV_FILES, "No CSV files found - check the path / archive contents."
""")

M(r"""
## 3. File inventory
Each CSV is mapped to **feature set** (stateless / stateful), **scenario** (light / heavy / day-1 benign) and **label** from its file name and parent folder. Check the table: every row must be resolved. If the official archive names files differently, fix them in `OVERRIDES`.
""")

C(r"""
ATTACK_TYPES = ["audio", "compressed", "exe", "image", "text", "video"]

# Example: OVERRIDES = {"some_file.csv": {"feature_set": "stateless", "scenario": "light", "label": 1}}
OVERRIDES = {}


def file_key(path):
    # Shared key between the stateless and stateful CSV of the same pcap
    n = Path(path).name.lower()
    n = re.sub(r"(stateless|stateful)(_features)?", "", n)
    n = re.sub(r"(\.pcap)?\.csv$", "", n)
    return n.strip("-_ ")          # dataset typo: 'stateful_features-_light_benign'


def classify(path):
    p = Path(path)
    name, parent = p.name.lower(), p.parent.name.lower()
    full = str(p).replace("\\", "/").lower()

    feature_set = ("stateless" if "stateless" in name else "stateful" if "stateful" in name else
                   "stateless" if "stateless" in full else "stateful" if "stateful" in full else None)

    if "benign" in name:
        label = 0
    elif "attack" in name or any(t in name for t in ATTACK_TYPES):
        label = 1
    elif "benign" in parent and "attack" not in parent:
        label = 0
    elif "attack" in parent and "benign" not in parent:
        label = 1
    else:
        label = None

    if "light" in name or "light" in full:
        scenario = "light"
    elif "heavy" in name or "heavy" in full:
        scenario = "heavy"
    elif label == 0:
        scenario = "benign_day1"   # injected into both light and heavy (paper Sec. 6.2)
    else:
        scenario = None

    attack_type = "benign" if label == 0 else next((t for t in ATTACK_TYPES if t in name), "unknown")
    rec = dict(path=str(path), file=p.name, key=file_key(path), feature_set=feature_set,
               scenario=scenario, label=label, attack_type=attack_type)
    rec.update(OVERRIDES.get(p.name, {}))
    return rec


INV = pd.DataFrame([classify(f) for f in CSV_FILES])
unresolved = INV[INV[["feature_set", "scenario", "label"]].isna().any(axis=1)]
display(INV.drop(columns="path"))
if len(unresolved):
    print("\nWARNING - unresolved files (ignored unless fixed in OVERRIDES):")
    display(unresolved.drop(columns="path"))
INV = INV.drop(unresolved.index).reset_index(drop=True)
INV["label"] = INV["label"].astype(int)
print(INV.groupby(["feature_set", "scenario", "label"]).size().rename("files"))
""")

M(r"""
## 4. Load and preprocess (paper Sec. 6.3.1)
- timestamp removed from the features (kept aside only to order packets for the hybrid window)
- `unique_ttl` lists → mean TTL
- object columns → label-encoded (`longest_word`, `sld`, `rr_type`, `distinct_ip`, `unique_country`, `unique_asn`, `distinct_domains`, `reverse_dns`, ...)
- NaN / inf → 0
""")

C(r"""
TS_NAMES = {"timestamp", "time", "ts", "frame.time"}
LABEL_NAMES = {"label", "class", "target", "attack"}
META = ["_ts", "_pos", "_key", "_attack_type"]


def read_csv(path):
    nrows = CONFIG["quick_rows_per_file"] if CONFIG["quick_mode"] else None
    df = pd.read_csv(path, low_memory=False, on_bad_lines="skip", nrows=nrows, encoding_errors="replace")
    df.columns = [str(c).strip() for c in df.columns]
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def parse_ts(s):
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        return num.fillna(0).astype(float)
    dt = pd.to_datetime(s, errors="coerce")
    return (dt.astype("int64") / 1e9).where(dt.notna(), 0.0)


def list_mean(v):
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(v))
    return float(np.mean([float(n) for n in nums])) if nums else 0.0


def load_group(feature_set, scenario):
    rows = INV[(INV.feature_set == feature_set) & INV.scenario.isin([scenario, "benign_day1"])]
    frames = []
    for r in rows.itertuples():
        df = read_csv(r.path)
        ts_col = next((c for c in df.columns if c.lower() in TS_NAMES), None)
        df["_ts"] = parse_ts(df[ts_col]) if ts_col else np.arange(len(df), dtype=float)
        df["_pos"] = np.arange(len(df)) / max(len(df), 1)   # relative position in the capture (0..1)
        df = df.drop(columns=[c for c in df.columns if c.lower() in TS_NAMES | LABEL_NAMES])
        df["_key"], df["_attack_type"], df["label"] = r.key, r.attack_type, r.label
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else None


def preprocess(df):
    meta = df[META].reset_index(drop=True)
    y = df["label"].astype(int).reset_index(drop=True)
    X = df.drop(columns=META + ["label"]).reset_index(drop=True)
    if "unique_ttl" in X:
        X["unique_ttl"] = X["unique_ttl"].map(list_mean)
    cat_cols = []
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])):
            conv = pd.to_numeric(X[c], errors="coerce")
            if X[c].notna().any() and conv.notna().sum() >= 0.99 * X[c].notna().sum():
                X[c] = conv                                   # numbers stored as text
            else:
                X[c] = pd.factorize(X[c].fillna("0").astype(str))[0]   # label encoding
                cat_cols.append(c)
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    return X, y, meta, cat_cols


DATASETS = {}
for scen in ["light", "heavy"]:
    for fs in ["stateless", "stateful"]:
        raw = load_group(fs, scen)
        if raw is None or raw["label"].nunique() < 2:
            print(f"skip {scen}/{fs}: missing files or only one class")
            continue
        X, y, meta, cats = preprocess(raw)
        DATASETS[(scen, fs)] = dict(X=X, y=y, meta=meta, cat_cols=cats)
        print(f"{scen:5s} {fs:9s}: {len(X):>8,} rows | {X.shape[1]} features | attack {y.mean():.1%} | categorical: {cats}")
        del raw
""")

C(r"""
# Row counts vs paper Table 2 (benign includes the injected day-1 benign)
PAPER_T2 = {("heavy", "stateful"): (72_028, 156_014), ("heavy", "stateless"): (251_670, 402_767),
            ("light", "stateful"): (11_295, 109_766), ("light", "stateless"): (42_683, 281_164)}
t2 = []
for k, d in DATASETS.items():
    pa, pb = PAPER_T2[k]
    t2.append(dict(scenario=k[0], feature_set=k[1], attack=int(d["y"].sum()), paper_attack=pa,
                   benign=int((d["y"] == 0).sum()), paper_benign=pb))
display(pd.DataFrame(t2))
print("Features per set:")
for k, d in DATASETS.items():
    print(f"  {k}: {list(d['X'].columns)}")
""")

M(r"""
## 4b. Data audit: does the "attack" traffic contain exfiltration?
The paper's attacker used DNSExfiltrator with base64URL chunks of up to **63 characters per label** sent to **`cicresearch.ca`** (Sec. 6.1). If the attack CSVs really contain that traffic, we should see that domain and long subdomains. We also count distinct feature vectors: thousands of rows with only a few dozen distinct vectors means the file is dominated by repeated background lookups (mDNS, reverse PTR, NetBIOS), not exfiltration.
""")

C(r"""
audit = []
for r in INV[INV.feature_set == "stateless"].itertuples():
    d = read_csv(r.path)
    feats = d.drop(columns=[c for c in d.columns if c.lower() in TS_NAMES])
    sld = d["sld"].astype(str) if "sld" in d else pd.Series(dtype=str)
    audit.append(dict(file=r.file, scenario=r.scenario, label=r.label, rows=len(d),
                      distinct_vectors=len(feats.drop_duplicates()),
                      cicresearch_rows=int(sld.str.contains("cicresearch", case=False).sum()),
                      max_subdomain_len=int(d["subdomain_length"].max()) if "subdomain_length" in d else None,
                      top_sld=", ".join(sld.value_counts().head(4).index)))
AUDIT = pd.DataFrame(audit).sort_values(["label", "scenario", "file"])
display(AUDIT)
atk = AUDIT[AUDIT.label == 1]
print(f"Attack rows: {atk.rows.sum():,} | rows querying cicresearch.ca: {atk.cicresearch_rows.sum():,} "
      f"| longest subdomain in any attack row: {atk.max_subdomain_len.max()} chars (DNSExfiltrator uses up to 63)")
""")

M(r"""
## 5. Train and evaluate the five classifiers (paper Sec. 6.3.2)
We report **both**:
- **weighted** precision / F1 + accuracy (this matches how Table 3 numbers behave, e.g. GNB light: acc 69.4% but precision 91.4%)
- **attack-class** precision / recall / F1, false-positive rate and ROC-AUC, which are the honest numbers on imbalanced data (light is ~87% benign).
""")

C(r"""
def make_model(name):
    m = {"GNB": GaussianNB(),
         "RF":  RandomForestClassifier(n_jobs=-1, random_state=SEED),
         "MLP": MLPClassifier(random_state=SEED),
         "SVM": SVC(random_state=SEED),
         "LR":  LogisticRegression(random_state=SEED)}[name]
    return make_pipeline(StandardScaler(), m) if CONFIG["scale_features"] else m


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def evaluate(y_true, y_pred, y_score=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    pw, _, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    pa, ra, fa, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, y_score) if y_score is not None and len(np.unique(y_true)) == 2 else np.nan
    return {"accuracy": 100 * accuracy_score(y_true, y_pred),
            "precision_w": 100 * pw, "f1_w": 100 * fw,
            "precision_atk": 100 * pa, "recall_atk": 100 * ra, "f1_atk": 100 * fa,
            "fpr": 100 * fp / max(fp + tn, 1), "roc_auc": 100 * auc,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


rng = np.random.default_rng(SEED)
SPLITS, MODELS, rows = {}, {}, []
for (scen, fs), d in DATASETS.items():
    X_tr, X_te, y_tr, y_te, m_tr, m_te = train_test_split(
        d["X"], d["y"], d["meta"], test_size=CONFIG["test_size"], shuffle=True, random_state=SEED)
    SPLITS[(scen, fs)] = dict(X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te, m_tr=m_tr, m_te=m_te)
    print(f"\n=== {scen} / {fs}: train {len(X_tr):,}  test {len(X_te):,} ===")
    for name in CONFIG["models"]:
        Xa, ya, Xb, yb = X_tr, y_tr, X_te, y_te
        if name == "SVM":
            if len(Xa) > CONFIG["svm_max_train"]:
                i = rng.choice(len(Xa), CONFIG["svm_max_train"], replace=False); Xa, ya = Xa.iloc[i], ya.iloc[i]
            if len(Xb) > CONFIG["svm_max_test"]:
                i = rng.choice(len(Xb), CONFIG["svm_max_test"], replace=False); Xb, yb = Xb.iloc[i], yb.iloc[i]
        model = make_model(name)
        t0 = time.time(); model.fit(Xa, ya); fit_s = time.time() - t0
        t0 = time.time(); pred = model.predict(Xb); s = scores(model, Xb); pred_s = time.time() - t0
        r = dict(scenario=scen, feature_set=fs, model=name, n_train=len(Xa), n_test=len(Xb),
                 fit_s=round(fit_s, 2), predict_s=round(pred_s, 2), **evaluate(yb, pred, s))
        rows.append(r); MODELS[(scen, fs, name)] = model
        print(f"  {name:4s} acc {r['accuracy']:6.2f} | F1(w) {r['f1_w']:6.2f} | F1(atk) {r['f1_atk']:6.2f} "
              f"| FPR {r['fpr']:5.2f} | fit {fit_s:6.1f}s")

RESULTS = pd.DataFrame(rows)
display(RESULTS.round(2))
""")

M(r"""
## 6. Comparison with paper Table 3
Table 3 does not state whether the numbers come from stateless or stateful features, so both are shown. `diff_*` = ours − paper (percentage points).
""")

C(r"""
PAPER_T3 = pd.DataFrame([
    ("GNB", "light", 91.41, 74.94, 69.43), ("RF", "light", 99.94, 99.94, 99.94),
    ("MLP", "light", 91.13, 93.12, 95.32), ("SVM", "light", 78.53, 83.25, 88.60),
    ("LR",  "light", 92.97, 93.39, 95.13),
    ("GNB", "heavy", 83.46, 71.24, 71.01), ("RF", "heavy", 99.97, 99.97, 99.97),
    ("MLP", "heavy", 99.84, 99.84, 99.84), ("SVM", "heavy", 65.31, 64.90, 71.36),
    ("LR",  "heavy", 92.07, 89.59, 89.22)],
    columns=["model", "scenario", "paper_precision", "paper_f1", "paper_accuracy"])

CMP = RESULTS.merge(PAPER_T3, on=["model", "scenario"], how="left")
CMP["diff_accuracy"] = CMP["accuracy"] - CMP["paper_accuracy"]
CMP["diff_f1"] = CMP["f1_w"] - CMP["paper_f1"]
cols = ["scenario", "feature_set", "model", "accuracy", "paper_accuracy", "diff_accuracy",
        "precision_w", "paper_precision", "f1_w", "paper_f1", "diff_f1", "f1_atk", "fpr"]
for fs in CMP.feature_set.unique():
    print(f"\n##### {fs.upper()} features #####")
    display(CMP[CMP.feature_set == fs][cols].round(2).reset_index(drop=True))

fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, scen in zip(axes, ["light", "heavy"]):
    sub = CMP[CMP.scenario == scen]
    if sub.empty: continue
    piv = sub.pivot_table(index="model", columns="feature_set", values="accuracy").reindex(CONFIG["models"])
    piv["paper"] = PAPER_T3[PAPER_T3.scenario == scen].set_index("model")["paper_accuracy"]
    piv.plot.bar(ax=ax, rot=0); ax.set_title(f"{scen} attack - accuracy (%)"); ax.set_ylim(50, 101)
plt.tight_layout(); plt.show()
""")

M(r"""
## 7. Two-layer hybrid detector (paper Sec. 5)
Layer 1 (stateless RF) scores every packet in a window of τ = 100 packets: `[0,0.4)` benign, `[0.4,0.7)` suspicious, `[0.7,1]` attack.
- If the suspicious ratio `r_sus > δ = 0.4` → layer 2: the stateful RF scores the stateful records that fall in the same time span of the same capture, and decides for the whole window.
- Otherwise the window is flagged if any packet is scored as attack (the paper "dumps the whole window" on a single malicious packet).

Windows are built per capture file from the **held-out** rows only, in capture order. The stateful CSVs have no timestamp column, so a window is matched to the stateful records at the same relative position in the same capture. Since each capture is either all-benign or all-attack, the window label is unambiguous.
""")

C(r"""
def hybrid_eval(scen):
    if (scen, "stateless") not in SPLITS or (scen, "stateful") not in SPLITS:
        print(f"{scen}: need both stateless and stateful data"); return None
    W, step, delta = CONFIG["window"], CONFIG["step"], CONFIG["delta"]
    lo, hi = CONFIG["bins"]
    sl, sf = SPLITS[(scen, "stateless")], SPLITS[(scen, "stateful")]
    clf1, clf2 = MODELS[(scen, "stateless", "RF")], MODELS[(scen, "stateful", "RF")]

    # Stateful CSVs ship without a timestamp, so windows are aligned on the relative
    # position of each record inside its capture file (both CSVs come from the same pcap, in order).
    pk = pd.DataFrame({"key": sl["m_te"]["_key"].values, "ts": sl["m_te"]["_pos"].values,
                       "y": sl["y_te"].values, "p": clf1.predict_proba(sl["X_te"])[:, 1]})
    st = pd.DataFrame({"key": sf["m_te"]["_key"].values, "ts": sf["m_te"]["_pos"].values,
                       "p": clf2.predict_proba(sf["X_te"])[:, 1]})
    st_by_key = {k: g for k, g in st.groupby("key")}

    out = []
    for key, g in pk.groupby("key"):
        g = g.sort_values("ts")
        s = st_by_key.get(key)
        for start in range(0, len(g), step):
            w = g.iloc[start:start + W]
            r_sus = ((w.p >= lo) & (w.p < hi)).mean()
            layer1 = int((w.p >= hi).any())
            layer, pred = 1, layer1
            if r_sus > delta:
                sel = s[(s.ts >= w.ts.min()) & (s.ts <= w.ts.max())] if s is not None else None
                if sel is not None and len(sel):
                    layer, pred = 2, int(sel.p.mean() >= 0.5)
                else:
                    layer = "2 (no stateful rows, kept layer-1)"
            out.append(dict(key=key, y=int(w.y.iloc[0]), n=len(w), r_sus=r_sus,
                            stateless_only=layer1, hybrid=pred, layer=str(layer)))
    win = pd.DataFrame(out)
    res = pd.DataFrame([dict(scenario=scen, detector=c, windows=len(win), **evaluate(win.y, win[c]))
                        for c in ["stateless_only", "hybrid"]])
    print(f"\n{scen}: {len(win)} windows | escalated to layer 2: {(win.layer != '1').mean():.2%}")
    print(win.layer.value_counts().to_string())
    return res


HYBRID = pd.concat([r for r in (hybrid_eval("light"), hybrid_eval("heavy")) if r is not None], ignore_index=True) \
    if SPLITS else pd.DataFrame()
display(HYBRID[["scenario", "detector", "windows", "accuracy", "precision_atk", "recall_atk", "f1_atk", "fpr",
                "tn", "fp", "fn", "tp"]].round(2) if len(HYBRID) else HYBRID)
""")

M(r"""
## 8. Sanity checks: is the baseline score real?
Near-perfect scores on a random split can be inflated. These checks tell us how much headroom there is for our own model:
1. **Train/test duplicates**: share of test rows whose exact feature vector is also in train (exfil queries are highly repetitive).
2. **Attacker-domain shortcut**: every attack query ends in the same attacker domain (`cicresearch.ca`), so `sld` alone can identify attacks. Retrain RF without `sld` / `longest_word`.
3. **Cross-scenario**: train on heavy, test on light (and the reverse) - does it generalise to a different attack volume?
4. **Unseen file type**: train without one exfiltrated file type (audio, exe, ...), test on it.

Checks 3–4 use only numeric features (no label-encoded columns) because label codes are not comparable across datasets.
""")

C(r"""
checks = []

# 1. exact duplicates between train and test
for (scen, fs), s in SPLITS.items():
    h_tr = set(pd.util.hash_pandas_object(s["X_tr"], index=False).values)
    h_te = pd.util.hash_pandas_object(s["X_te"], index=False)
    dup = h_te.isin(h_tr).values
    atk = s["y_te"].values == 1
    print(f"{scen:5s} {fs:9s}: test rows also in train = {dup.mean():6.1%} "
          f"(attack {dup[atk].mean():6.1%} | benign {dup[~atk].mean():6.1%})")


def rf_eval(X_tr, y_tr, X_te, y_te):
    m = RandomForestClassifier(n_jobs=-1, random_state=SEED).fit(X_tr, y_tr)
    return evaluate(y_te, m.predict(X_te), m.predict_proba(X_te)[:, 1])


# 2. without attacker-domain features
SHORTCUT = ["sld", "longest_word"]
for (scen, fs), s in SPLITS.items():
    drop = [c for c in SHORTCUT if c in s["X_tr"].columns]
    if not drop: continue
    r = rf_eval(s["X_tr"].drop(columns=drop), s["y_tr"], s["X_te"].drop(columns=drop), s["y_te"])
    checks.append(dict(check=f"RF without {drop}", scenario=scen, feature_set=fs, **r))

# 3. cross-scenario (numeric features only)
def numeric_cols(scen, fs):
    d = DATASETS[(scen, fs)]
    return [c for c in d["X"].columns if c not in d["cat_cols"] + SHORTCUT]

for fs in ["stateless", "stateful"]:
    for a, b in [("heavy", "light"), ("light", "heavy")]:
        if (a, fs) not in SPLITS or (b, fs) not in SPLITS: continue
        cols = sorted(set(numeric_cols(a, fs)) & set(numeric_cols(b, fs)))
        sa, sb = SPLITS[(a, fs)], SPLITS[(b, fs)]
        r = rf_eval(sa["X_tr"][cols], sa["y_tr"], sb["X_te"][cols], sb["y_te"])
        checks.append(dict(check=f"train {a} -> test {b}", scenario=b, feature_set=fs, **r))

# 4. leave-one-file-type-out (stateless, numeric features)
for scen in ["light", "heavy"]:
    if (scen, "stateless") not in SPLITS: continue
    s = SPLITS[(scen, "stateless")]; cols = numeric_cols(scen, "stateless")
    for t in sorted(set(s["m_tr"]["_attack_type"]) - {"benign", "unknown"}):
        tr = (s["m_tr"]["_attack_type"] != t).values
        te = ((s["m_te"]["_attack_type"] == t) | (s["y_te"] == 0)).values
        r = rf_eval(s["X_tr"][cols][tr], s["y_tr"][tr], s["X_te"][cols][te], s["y_te"][te])
        checks.append(dict(check=f"unseen file type: {t}", scenario=scen, feature_set="stateless", **r))

CHECKS = pd.DataFrame(checks)
if len(CHECKS):
    display(CHECKS[["check", "scenario", "feature_set", "accuracy", "precision_atk", "recall_atk",
                    "f1_atk", "fpr", "roc_auc"]].round(2))
""")

C(r"""
# Random Forest feature importances (what the baseline relies on)
keys = [k for k in MODELS if k[2] == "RF"]
fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.5), squeeze=False)
for ax, k in zip(axes[0], keys):
    m = MODELS[k]; m = m[-1] if hasattr(m, "steps") else m
    imp = pd.Series(m.feature_importances_, index=SPLITS[k[:2]]["X_tr"].columns).sort_values().tail(12)
    imp.plot.barh(ax=ax); ax.set_title(f"RF {k[0]} / {k[1]}")
plt.tight_layout(); plt.show()
""")

M(r"""
## 9. Save results
""")

C(r"""
OUT = Path(CONFIG["output_dir"]); OUT.mkdir(parents=True, exist_ok=True)
RESULTS.to_csv(OUT / "baseline_results.csv", index=False)
CMP.to_csv(OUT / "baseline_vs_paper.csv", index=False)
if len(HYBRID): HYBRID.to_csv(OUT / "hybrid_results.csv", index=False)
if len(CHECKS): CHECKS.to_csv(OUT / "sanity_checks.csv", index=False)
(OUT / "run_config.json").write_text(json.dumps(
    {**CONFIG, "sklearn": sklearn.__version__, "pandas": pd.__version__,
     "features": {f"{k[0]}/{k[1]}": list(d["X"].columns) for k, d in DATASETS.items()}}, indent=2, default=str))

if CONFIG["save_models"]:
    for k, m in MODELS.items():
        if k[2] == "RF":
            joblib.dump(m, OUT / f"rf_{k[0]}_{k[1]}.joblib", compress=3)
print("Saved to", OUT)
for f in sorted(OUT.iterdir()):
    print(f"  {f.name:40s} {f.stat().st_size / 1e6:8.2f} MB")
""")

M(r"""
## 10. Reading the results
- **Baseline reproduced** if RF lands within ~0.1 pp of 99.94 (light) / 99.97 (heavy). Small differences are expected: the split seed, sklearn version and SVM subsampling are not specified in the paper.
- **The sanity checks show where an improvement can come from.** If accuracy stays ~99.9% only when `sld` is present, or drops on unseen file types / cross-scenario, the paper's number is optimistic, and a model that is robust under those protocols is a stronger, defensible contribution than another +0.01% on a random split.
- Compare models with **attack-class F1 and FPR**, not weighted accuracy, especially for the light (~13% attack) scenario.
""")


def main():
    nb = make_notebook(CELLS)
    nb["metadata"]["colab"] = {"provenance": [], "toc_visible": True}
    out = Path(__file__).resolve().parents[1] / "notebooks" / "01_paper_csv_audit.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Generated {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
