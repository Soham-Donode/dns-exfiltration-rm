"""
Week 5: Model Training Pipeline
================================
End-to-end model training, validation, and evaluation pipeline for detecting
DNS exfiltration and tunneling attacks using machine learning classifiers
(Random Forest, XGBoost, LightGBM).
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import joblib


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load training parameters and paths from configuration YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_synthetic_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """Generate a realistic synthetic feature dataset for testing the pipeline."""
    print("[*] Generating synthetic benchmark dataset...")
    np.random.seed(42)
    half = n_samples // 2

    # Benign DNS queries: shorter lengths, standard entropy, low digit/hex ratios
    benign = pd.DataFrame({
        "domain_len": np.random.normal(16, 5, half).clip(5, 45),
        "subdomain_len": np.random.normal(4, 3, half).clip(0, 25),
        "num_labels": np.random.choice([2, 3, 4], half, p=[0.7, 0.25, 0.05]),
        "max_label_len": np.random.normal(10, 4, half).clip(3, 30),
        "avg_label_len": np.random.normal(7, 2, half).clip(2, 20),
        "entropy": np.random.normal(2.7, 0.4, half).clip(1.0, 4.0),
        "subdomain_entropy": np.random.normal(1.5, 0.8, half).clip(0.0, 3.5),
        "digit_ratio": np.random.beta(1, 25, half),
        "alpha_ratio": np.random.beta(20, 2, half),
        "hex_ratio": np.random.beta(2, 10, half),
        "vowel_ratio": np.random.beta(4, 6, half),
        "consonant_ratio": np.random.beta(6, 4, half),
        "special_ratio": np.random.beta(1, 40, half),
        "subdomain_digit_ratio": np.random.beta(1, 20, half),
        "subdomain_hex_ratio": np.random.beta(1, 15, half),
        "unique_char_ratio": np.random.normal(0.65, 0.1, half).clip(0.3, 1.0),
        "max_consecutive_consonants": np.random.choice([1, 2, 3, 4], half, p=[0.4, 0.4, 0.15, 0.05]),
        "label": 0
    })

    # Exfiltration queries: long domains, high entropy, high digit/hex ratios, encoded chunks
    exfil = pd.DataFrame({
        "domain_len": np.random.normal(58, 12, half).clip(25, 120),
        "subdomain_len": np.random.normal(42, 12, half).clip(15, 95),
        "num_labels": np.random.choice([3, 4, 5], half, p=[0.4, 0.4, 0.2]),
        "max_label_len": np.random.normal(35, 10, half).clip(15, 63),
        "avg_label_len": np.random.normal(18, 5, half).clip(8, 40),
        "entropy": np.random.normal(4.3, 0.3, half).clip(3.2, 5.2),
        "subdomain_entropy": np.random.normal(4.1, 0.4, half).clip(3.0, 5.0),
        "digit_ratio": np.random.beta(5, 10, half),
        "alpha_ratio": np.random.beta(12, 5, half),
        "hex_ratio": np.random.beta(8, 4, half),
        "vowel_ratio": np.random.beta(2, 8, half),
        "consonant_ratio": np.random.beta(8, 2, half),
        "special_ratio": np.random.beta(2, 20, half),
        "subdomain_digit_ratio": np.random.beta(4, 8, half),
        "subdomain_hex_ratio": np.random.beta(6, 4, half),
        "unique_char_ratio": np.random.normal(0.85, 0.08, half).clip(0.6, 1.0),
        "max_consecutive_consonants": np.random.choice([3, 4, 5, 6, 7], half, p=[0.1, 0.2, 0.3, 0.25, 0.15]),
        "label": 1
    })

    df = pd.concat([benign, exfil], ignore_index=True)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def get_model(algo_name: str, config: dict):
    """Instantiate chosen classifier based on config."""
    algo = algo_name.lower()
    training_cfg = config.get("training", {})
    hp = training_cfg.get("hyperparameters", {})

    if algo == "random_forest":
        rf_params = hp.get("random_forest", {})
        return RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 150),
            max_depth=rf_params.get("max_depth", 20),
            min_samples_split=rf_params.get("min_samples_split", 5),
            n_jobs=rf_params.get("n_jobs", -1),
            random_state=training_cfg.get("random_state", 42)
        )
    elif algo == "xgboost":
        try:
            from xgboost import XGBClassifier
            xgb_params = hp.get("xgboost", {})
            return XGBClassifier(
                n_estimators=xgb_params.get("n_estimators", 150),
                max_depth=xgb_params.get("max_depth", 8),
                learning_rate=xgb_params.get("learning_rate", 0.1),
                eval_metric=xgb_params.get("eval_metric", "logloss"),
                random_state=training_cfg.get("random_state", 42)
            )
        except ImportError:
            print("[!] XGBoost not installed. Falling back to GradientBoostingClassifier.")
            return GradientBoostingClassifier(random_state=training_cfg.get("random_state", 42))
    elif algo == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(random_state=training_cfg.get("random_state", 42))
        except ImportError:
            print("[!] LightGBM not installed. Falling back to RandomForestClassifier.")
            return RandomForestClassifier(random_state=training_cfg.get("random_state", 42))
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def train_pipeline(
    config_path: str = "configs/config.yaml",
    dataset_path: str = None,
    algorithm: str = None
) -> Tuple[Dict[str, float], Any]:
    """Execute complete training, cross-validation, and evaluation pipeline."""
    config = load_config(config_path)
    data_file = dataset_path or config["paths"].get("features_dataset_csv")
    algo = algorithm or config["training"].get("algorithm", "random_forest")

    # Load dataset or generate synthetic if absent
    if data_file and os.path.exists(data_file):
        print(f"[+] Loading processed dataset from {data_file}...")
        df = pd.read_csv(data_file)
    else:
        print(f"[!] Dataset '{data_file}' not found.")
        df = generate_synthetic_dataset()
        # Save synthetic sample so subsequent runs have data
        synthetic_path = Path("data/processed/dns_features.csv")
        synthetic_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(synthetic_path, index=False)
        print(f"[+] Saved baseline dataset to {synthetic_path}")

    # Separate features and target
    drop_cols = [c for c in ["domain", "query", "label"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["label"].astype(int)

    test_size = config["training"].get("test_size", 0.2)
    seed = config["training"].get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    print(f"[*] Training Set: {X_train.shape[0]} samples | Test Set: {X_test.shape[0]} samples")
    print(f"[*] Features ({X.shape[1]}): {list(X.columns)}")

    # Initialize model
    model = get_model(algo, config)
    print(f"[*] Training model using '{algo}' algorithm...")

    # Cross-validation
    cv_folds = config["training"].get("cross_validation_folds", 5)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1")
    print(f"[+] {cv_folds}-Fold Cross-Validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Fit final model on training set
    model.fit(X_train, y_train)

    # Evaluation on holdout test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 55)
    print("                EVALUATION RESULTS               ")
    print("=" * 55)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print("-" * 55)
    print(f"  Confusion Matrix:\n{cm}")
    print("=" * 55)

    # Feature Importance (if tree-based)
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("\nTop 7 Most Informative Features:")
        for feat, score in importances.head(7).items():
            print(f"  {feat:28}: {score:.4f}")

    # Save model artifact
    model_output_path = Path(config["paths"].get("model_output_path", "models/dns_exfil_detector.joblib"))
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"\n[+] Saved trained model artifact to: {model_output_path}")

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc
    }
    return metrics, model


def main():
    parser = argparse.ArgumentParser(description="Train DNS Exfiltration detection model.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--data", help="Path to processed features CSV")
    parser.add_argument("--algo", choices=["random_forest", "xgboost", "lightgbm"], help="Model algorithm")

    args = parser.parse_args()
    train_pipeline(config_path=args.config, dataset_path=args.data, algorithm=args.algo)


if __name__ == "__main__":
    main()
