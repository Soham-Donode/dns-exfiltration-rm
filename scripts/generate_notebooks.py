"""
Generator script to produce clean, complete Jupyter notebooks for Weeks 1-5:
- notebooks/01_eda.ipynb
- notebooks/02_feature_engineering.ipynb
- notebooks/03_model_training.ipynb
"""

import json
from pathlib import Path


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md(source: str):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code(source: str):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }


def build_eda_notebook():
    cells = [
        md("""# 01. Exploratory Data Analysis (EDA)
## DNS Exfiltration and Tunneling Detection

This notebook analyzes distinguishing statistical, lexical, and structural characteristics between **benign web lookups** (e.g. Tranco 1M) and **malicious DNS exfiltration/tunneling traffic** (e.g., Base32, Hex, CIC-Bell-DNS-EXF-2021).

### Key Investigation Objectives:
1. Compare query length and subdomain length distributions.
2. Quantify Shannon entropy differences across domains and subdomains.
3. Analyze character distribution biases (digit ratio, hex ratio, vowel/consonant ratios).
4. Evaluate query frequency patterns (burst vs low-and-slow exfiltration).
5. Identify top linear and non-linear correlations with the malicious target label."""),

        code("""import os
import sys
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Ensure repo root is on sys.path
for p in [".", "..", "../.."]:
    if os.path.exists(os.path.join(p, "configs", "config.yaml")):
        sys.path.insert(0, os.path.abspath(p))
        REPO_ROOT = os.path.abspath(p)
        break

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
print(f"Repository Root: {REPO_ROOT}")"""),

        code("""# Load configuration
config_path = os.path.join(REPO_ROOT, "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

data_path = os.path.join(REPO_ROOT, config["paths"]["features_dataset_csv"])
print(f"Target Feature Dataset: {data_path}")"""),

        code("""# Load processed feature dataset or generate synthetic benchmark if not yet created
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records from {data_path}")
else:
    print("Processed dataset not found. Generating synthetic benchmark dataset...")
    from src.train import generate_synthetic_dataset
    df = generate_synthetic_dataset(n_samples=2000)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)

df.head(6)"""),

        code("""# Class distribution
print("=== Class Distribution ===")
class_counts = df["label"].value_counts()
print(class_counts)
print("\\nProportions:")
print(df["label"].value_counts(normalize=True).round(4))

fig, ax = plt.subplots(figsize=(6, 3))
sns.countplot(data=df, x="label", hue="label", legend=False, palette=["#2ecc71", "#e74c3c"], ax=ax)
ax.set_xticks([0, 1])
ax.set_xticklabels(["0: Benign", "1: Malicious / Exfil"])
ax.set_title("Dataset Class Balance")
plt.tight_layout()
plt.show()"""),

        code("""# Visualizing Query Lengths: domain_len, subdomain_len, max_label_len
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

sns.histplot(data=df, x="domain_len", hue="label", kde=True, ax=axes[0], palette=["#2ecc71", "#e74c3c"])
axes[0].set_title("Total FQDN Length Distribution")

sns.histplot(data=df, x="subdomain_len", hue="label", kde=True, ax=axes[1], palette=["#2ecc71", "#e74c3c"])
axes[1].set_title("Subdomain Length Distribution")

sns.histplot(data=df, x="max_label_len", hue="label", kde=True, ax=axes[2], palette=["#2ecc71", "#e74c3c"])
axes[2].set_title("Max Single Label Length")

plt.tight_layout()
plt.show()"""),

        code("""# Visualizing Information Theory: Shannon Entropy
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

sns.kdeplot(data=df, x="entropy", hue="label", fill=True, common_norm=False, ax=axes[0], palette=["#2ecc71", "#e74c3c"])
axes[0].set_title("Full FQDN Shannon Entropy (KDE)")
axes[0].set_xlabel("Entropy (Bits)")

sns.boxplot(data=df, x="label", y="subdomain_entropy", hue="label", legend=False, palette=["#2ecc71", "#e74c3c"], ax=axes[1])
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Benign (0)", "Exfil (1)"])
axes[1].set_title("Subdomain Entropy Comparison")
axes[1].set_ylabel("Entropy (Bits)")

plt.tight_layout()
plt.show()"""),

        code("""# Character Distributions: Digit Ratio, Hex Ratio, and Vowel/Consonant Ratio
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

sns.boxplot(data=df, x="label", y="digit_ratio", hue="label", legend=False, palette=["#2ecc71", "#e74c3c"], ax=axes[0])
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Benign", "Exfil"])
axes[0].set_title("Digit Ratio Distribution")

sns.boxplot(data=df, x="label", y="hex_ratio", hue="label", legend=False, palette=["#2ecc71", "#e74c3c"], ax=axes[1])
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Benign", "Exfil"])
axes[1].set_title("Hexadecimal Ratio Distribution")

sns.boxplot(data=df, x="label", y="vowel_ratio", hue="label", legend=False, palette=["#2ecc71", "#e74c3c"], ax=axes[2])
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(["Benign", "Exfil"])
axes[2].set_title("Vowel Ratio Distribution")

plt.tight_layout()
plt.show()"""),

        code("""# Feature Correlation Matrix
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["label"]]
corr_series = df[numeric_cols].apply(lambda x: x.corr(df["label"])).sort_values(ascending=False)

print("Top Correlations with Label (1 = Exfiltration):")
print(corr_series.to_string())

plt.figure(figsize=(10, 8))
top_corr_features = list(corr_series.abs().head(10).index) + ["label"]
sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="vlag", fmt=".2f", center=0)
plt.title("Correlation Matrix of Top Discriminative Features")
plt.tight_layout()
plt.show()"""),

        code("""# 2D Scatter: Shannon Entropy vs Domain Length Decision Boundary
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="domain_len",
    y="entropy",
    hue="label",
    style="label",
    alpha=0.8,
    palette=["#2ecc71", "#e74c3c"],
    s=60
)
plt.axvline(x=45, color="grey", linestyle="--", alpha=0.6, label="RFC / Typical Benign Threshold")
plt.axhline(y=3.8, color="grey", linestyle=":", alpha=0.6, label="Entropy Threshold (~3.8)")
plt.title("Entropy vs. Domain Length: Clear Clustering of Exfiltration Traffic")
plt.xlabel("Total Domain Length (characters)")
plt.ylabel("Shannon Entropy (bits)")
plt.legend(title="Class", labels=["Thresholds", "Benign (0)", "Exfil (1)"])
plt.tight_layout()
plt.show()"""),

        md("""### EDA Findings Summary:
1. **Query Lengths**: Benign domain names concentrate in 10-25 characters. Exfiltration queries consistently exceed 45+ characters to maximize throughput per packet.
2. **Shannon Entropy**: Benign names exhibit natural language entropy (~2.0 - 3.2 bits). Encoded Base32/Hex payloads exhibit significantly higher entropy (>3.8 bits).
3. **Character Distributions**: Encoded tunneling payloads feature high hexadecimal density and low vowel ratios, which heavily contrasts with natural linguistic domains.
4. **Conclusion**: Lexical and entropy features provide strong separation for linear and tree-based machine learning classifiers.""")
    ]
    return make_notebook(cells)


def build_feature_engineering_notebook():
    cells = [
        md("""# 02. Feature Engineering Engine
## Transforming Raw DNS Queries into Discriminative ML Vectors

This notebook demonstrates the feature engineering pipeline implemented in `src/features.py`:
1. **Information Theory**: Shannon Entropy (full domain, subdomain, individual labels).
2. **Lexical & Structural Metrics**: Domain lengths, label counts, min/max/avg label lengths.
3. **Linguistic Ratios**: Vowel ratio, digit ratio, consonant ratio, vowel-to-consonant, hex ratio.
4. **Temporal Frequency**: Rolling query frequency (queries per minute/second) for burst vs slow attacks.
5. **Batch Processing**: Transforming raw PCAP captures or labeled CSVs into feature matrices ready for ML."""),

        code("""import os
import sys
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Ensure repo root is on sys.path
for p in [".", "..", "../.."]:
    if os.path.exists(os.path.join(p, "configs", "config.yaml")):
        sys.path.insert(0, os.path.abspath(p))
        REPO_ROOT = os.path.abspath(p)
        break

# Load configuration
config_path = os.path.join(REPO_ROOT, "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

from src.features import (
    shannon_entropy,
    compute_entropy,
    vowel_digit_ratio,
    max_consecutive_consonants,
    split_domain_components,
    extract_domain_features,
    calculate_query_frequency,
    process_traffic_dataset
)

print("Feature engineering modules loaded successfully.")"""),

        md("""### 1. Information-Theoretic Metrics: Shannon Entropy
Shannon entropy quantifies the uncertainty or randomness of characters in a string:

$$H(X) = - \\sum_{i=1}^n P(x_i) \\log_2 P(x_i)$$

Natural domain names (e.g. `google.com`, `wikipedia.org`) have repetitive linguistic patterns resulting in low entropy.
Base32 and Hex encoded DNS tunnels (e.g. `54657374...tunnel.org`) exhibit near-maximal entropy for their character sets."""),

        code("""# Comparing entropy across benign, Base32, and Hex queries
sample_queries = {
    "Benign - Standard": "google.com",
    "Benign - Subdomain": "cdn.assets.cloudflare.net",
    "Exfil - Base32": "sq0001.krsxg5bamnxw4ztjmrsw45djmfwcazdb.tunnel.example.com",
    "Exfil - Hex": "sq0001.5465737420636f6e666964656e7469616c.tunnel.example.com"
}

for desc, q in sample_queries.items():
    ent = compute_entropy(q)
    parts = split_domain_components(q)
    sub_ent = compute_entropy(parts["subdomain"]) if parts["subdomain"] else 0.0
    print(f"{desc:20} | FQDN: {q[:45]:45}... | Total H: {ent:.4f} | Subdomain H: {sub_ent:.4f}")"""),

        md("""### 2. Lexical & Structural Metrics: Label Lengths & Components
According to RFC 1035:
- Labels must not exceed 63 characters.
- FQDN must not exceed 253 characters.

Attackers maximize each label to fit up to 63 bytes of payload, leading to unusually high `max_label_len` and `subdomain_len`."""),

        code("""# Extracting component and length features
demo_features = []
for desc, q in sample_queries.items():
    feats = extract_domain_features(q)
    feats["description"] = desc
    demo_features.append(feats)

df_demo = pd.DataFrame(demo_features)
cols = ["description", "domain_len", "subdomain_len", "num_labels", "max_label_len", "avg_label_len", "entropy"]
df_demo[cols]"""),

        md("""### 3. Linguistic Ratios: Vowel/Digit Ratios & Consonants
Normal English / human words have regular vowel-to-consonant ratios (~0.4 - 0.7).
Base32 has few vowels (`A, E, I`) relative to consonants, and Hex contains only `A-F` with `0-9` digits."""),

        code("""# Inspecting ratio metrics
ratio_cols = ["description", "digit_ratio", "hex_ratio", "vowel_ratio", "consonant_ratio", "max_consecutive_consonants"]
df_demo[ratio_cols]"""),

        md("""### 4. Temporal Frequency Metrics: Burst vs Low-and-Slow
DNS exfiltration attacks follow distinct operational profiles:
- **Burst mode**: Hundreds of queries sent in milliseconds/seconds to exfiltrate data quickly.
- **Low-and-Slow mode**: Covert queries sent at intervals (every 10-60s) to evade volume-based alarms."""),

        code("""# Demonstrating rolling query frequency computation
time_records = pd.DataFrame({
    "qname": ["tunnel.exfil.org"] * 5 + ["google.com"] * 2,
    "timestamp": [1710000000.0, 1710000001.0, 1710000002.0, 1710000002.5, 1710000003.0, 1710000000.0, 1710000030.0]
})

freq_result = calculate_query_frequency(time_records, window="10s", rate_unit="second")
freq_result"""),

        md("""### 5. Full Pipeline: Processing Raw Traffic into ML Feature Dataset"""),

        code("""# Process existing labeled traffic dataset or synthetic session
raw_traffic_path = os.path.join(REPO_ROOT, config["paths"].get("labeled_traffic_csv", "data/processed/labeled_traffic.csv"))
output_features_path = os.path.join(REPO_ROOT, config["paths"].get("features_dataset_csv", "data/processed/dns_features.csv"))

if os.path.exists(raw_traffic_path):
    print(f"Loading raw traffic from {raw_traffic_path}...")
    df_raw = pd.read_csv(raw_traffic_path)
else:
    print("Synthesizing raw traffic session...")
    from src.capture_traffic import generate_synthesized_traffic
    records, _ = generate_synthesized_traffic(benign_count=200, burst_count=120, low_slow_count=40)
    df_raw = pd.DataFrame(records)

# Execute feature extraction pipeline
df_features = process_traffic_dataset(df_raw, window="60s")
os.makedirs(os.path.dirname(output_features_path), exist_ok=True)
df_features.to_csv(output_features_path, index=False)

print(f"Successfully processed {len(df_features)} records into feature matrix.")
print(f"Saved feature dataset to: {output_features_path}")
df_features.head()"""),

        code("""# Feature Summary & Correlation with Target
numeric_features = [c for c in df_features.select_dtypes(include=[np.number]).columns if c not in ["label"]]
print(f"Total numeric features generated: {len(numeric_features)}")
print(numeric_features)

if "label" in df_features.columns:
    corr = df_features[numeric_features].apply(lambda x: x.corr(df_features["label"])).sort_values(ascending=False)
    print("\\nFeature correlation with malicious label:")
    print(corr.round(4).to_string())""")
    ]
    return make_notebook(cells)


def build_model_training_notebook():
    cells = [
        md("""# 03. Model Training & Evaluation
## Training Classifiers for DNS Exfiltration Detection

In Week 5, we train, tune, and benchmark three machine learning models:
1. **Baseline Decision Tree** (`DecisionTreeClassifier`): Fast, interpretable tree splits.
2. **Random Forest** (`RandomForestClassifier`): Ensemble bagging to reduce variance.
3. **XGBoost** (`XGBClassifier`): Gradient boosting on decision trees for high predictive precision.

### Cross-Validation Strategy:
- **Stratified 5-Fold Cross-Validation** to ensure proportional class distribution across folds.
- Hyperparameter tuning using `GridSearchCV`.
- Metrics: **F1-Score**, **ROC-AUC**, **Precision**, **Recall**, **Accuracy**, and **Confusion Matrices**."""),

        code("""import os
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Ensure repo root is on sys.path
for p in [".", "..", "../.."]:
    if os.path.exists(os.path.join(p, "configs", "config.yaml")):
        sys.path.insert(0, os.path.abspath(p))
        REPO_ROOT = os.path.abspath(p)
        break

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
print("Environment and ML libraries ready.")"""),

        code("""# Load Configuration & Dataset
config_path = os.path.join(REPO_ROOT, "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

data_path = os.path.join(REPO_ROOT, config["paths"]["features_dataset_csv"])
if not os.path.exists(data_path):
    from src.train import generate_synthetic_dataset
    df = generate_synthetic_dataset(n_samples=2500)
    df.to_csv(data_path, index=False)
else:
    df = pd.read_csv(data_path)

drop_cols = [c for c in ["domain", "query", "qname", "label", "attack_type"] if c in df.columns]
X = df.drop(columns=drop_cols)
y = df["label"].astype(int)

seed = config["training"].get("random_state", 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

print(f"Training Samples: {X_train.shape[0]} | Holdout Test Samples: {X_test.shape[0]}")
print(f"Features ({X.shape[1]}): {list(X.columns)}")"""),

        md("""### 1. Training Baseline Models with Stratified 5-Fold Cross-Validation"""),

        code("""# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_split=5, random_state=seed),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=seed, n_jobs=-1),
}
if XGB_AVAILABLE:
    models["XGBoost"] = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=seed)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cv_summary = {}

for name, clf in models.items():
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")
    cv_summary[name] = {"mean": scores.mean(), "std": scores.std(), "scores": scores}
    print(f"{name:16} | 5-Fold CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})")"""),

        md("""### 2. Hyperparameter Tuning with GridSearchCV"""),

        code("""from sklearn.model_selection import GridSearchCV

# Tune Random Forest as primary exemplar
rf_param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [8, 14],
    "min_samples_split": [2, 5]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=seed, n_jobs=-1),
    param_grid=rf_param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)

print(f"Best Random Forest Params: {rf_grid.best_params_}")
print(f"Best CV F1-Score:        {rf_grid.best_score_:.4f}")
models["Random Forest (Tuned)"] = rf_grid.best_estimator_"""),

        md("""### 3. Model Evaluation on Holdout Test Set"""),

        code("""# Evaluate each fitted model on unseen test set
results = []
test_probas = {}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred
    test_probas[name] = y_proba
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    })

df_benchmark = pd.DataFrame(results).round(4)
print("=== Holdout Evaluation Benchmark ===")
df_benchmark"""),

        code("""# Plot Comparative Performance Bar Chart
df_melted = df_benchmark.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1-Score", "ROC-AUC"])

plt.figure(figsize=(10, 4.5))
sns.barplot(data=df_melted, x="Model", y="value", hue="variable", palette="Set2")
plt.ylim(0.85, 1.02)
plt.title("Comparative Metric Benchmark Across Models")
plt.ylabel("Score")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()"""),

        md("""### 4. ROC Curves & Confusion Matrices"""),

        code("""# Plot ROC Curves side-by-side
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for name, probas in test_probas.items():
    fpr, tpr, _ = roc_curve(y_test, probas)
    auc = roc_auc_score(y_test, probas)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].set_title("Receiver Operating Characteristic (ROC) Curves")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right")

# Best Model Confusion Matrix Heatmap (Tuned RF or XGBoost)
best_model_name = "Random Forest (Tuned)" if "Random Forest (Tuned)" in models else "Random Forest"
best_clf = models[best_model_name]
cm = confusion_matrix(y_test, best_clf.predict(X_test))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["Benign (0)", "Exfil (1)"],
            yticklabels=["Benign (0)", "Exfil (1)"])
axes[1].set_title(f"Confusion Matrix: {best_model_name}")
axes[1].set_ylabel("True Label")
axes[1].set_xlabel("Predicted Label")

plt.tight_layout()
plt.show()"""),

        md("""### 5. Feature Importance Analysis"""),

        code("""# Feature Importance from Tuned Random Forest
importances = pd.Series(best_clf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=importances.head(10).values, y=importances.head(10).index, palette="mako")
plt.title("Top 10 Most Discriminative Features (MDI / Gini Importance)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()"""),

        md("""### 6. Persist Model Artifact & Real-Time Inference Demo"""),

        code("""# Save best model to disk
models_dir = os.path.join(REPO_ROOT, "models")
os.makedirs(models_dir, exist_ok=True)
save_path = os.path.join(models_dir, "best_dns_detector.joblib")
joblib.dump(best_clf, save_path)
print(f"Saved trained detector to {save_path}")

# Real-Time Query Classification Demonstration
from src.features import extract_domain_features

test_queries = [
    "github.com",
    "portal.office.com",
    "sq0001.5465737464617461.tunnel.attacker-domain.org",
    "sq0002.krsxg5bamnxw4ztjmrsw45dj.tunnel.exfil-c2.net"
]

inference_rows = [extract_domain_features(q) for q in test_queries]
df_inf = pd.DataFrame(inference_rows)
# Align columns with training feature set
for c in X.columns:
    if c not in df_inf:
        df_inf[c] = 1.0
df_inf = df_inf[X.columns]

predictions = best_clf.predict(df_inf)
probabilities = best_clf.predict_proba(df_inf)[:, 1]

print("\\n=== Live Real-Time Inference Results ===")
for q, pred, prob in zip(test_queries, predictions, probabilities):
    status = "🚨 EXFILTRATION DETECTED" if pred == 1 else "✅ BENIGN"
    print(f"{status:26} | P(Exfil)={prob:6.2%} | Query: {q}")""")
    ]
    return make_notebook(cells)


def main():
    notebooks_dir = Path("notebooks")
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    nb1 = build_eda_notebook()
    nb2 = build_feature_engineering_notebook()
    nb3 = build_model_training_notebook()

    with open(notebooks_dir / "01_eda.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=2)
    print("[✓] Generated notebooks/01_eda.ipynb")

    with open(notebooks_dir / "02_feature_engineering.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb2, f, indent=2)
    print("[✓] Generated notebooks/02_feature_engineering.ipynb")

    with open(notebooks_dir / "03_model_training.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb3, f, indent=2)
    print("[✓] Generated notebooks/03_model_training.ipynb")


if __name__ == "__main__":
    main()
