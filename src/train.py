"""
Week 5: Model Training & Tuning Pipeline
========================================
Training, hyperparameter tuning, and comparative evaluation for:
1. Baseline Decision Trees (DecisionTreeClassifier)
2. Random Forest (RandomForestClassifier)
3. Gradient Boosted Trees (XGBoost / XGBClassifier)

Evaluated with Stratified 5-Fold Cross-Validation, reporting Accuracy,
Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, and Feature Importances.
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import yaml
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
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

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load training configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_synthetic_dataset(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic feature dataset for testing and benchmarking."""
    print(f"[*] Generating synthetic feature dataset ({n_samples} samples)...")
    np.random.seed(seed)
    half = n_samples // 2

    # Benign DNS queries: low length, standard entropy, low digit/hex ratios
    benign = pd.DataFrame({
        "domain_len": np.random.normal(16, 5, half).clip(5, 45),
        "subdomain_len": np.random.normal(4, 3, half).clip(0, 25),
        "num_labels": np.random.choice([2, 3, 4], half, p=[0.7, 0.25, 0.05]),
        "max_label_len": np.random.normal(10, 4, half).clip(3, 30),
        "min_label_len": np.random.normal(3, 1, half).clip(1, 10),
        "avg_label_len": np.random.normal(7, 2, half).clip(2, 20),
        "entropy": np.random.normal(2.7, 0.4, half).clip(1.0, 4.0),
        "subdomain_entropy": np.random.normal(1.5, 0.8, half).clip(0.0, 3.5),
        "max_label_entropy": np.random.normal(2.3, 0.4, half).clip(1.0, 3.8),
        "digit_ratio": np.random.beta(1, 25, half),
        "alpha_ratio": np.random.beta(20, 2, half),
        "hex_ratio": np.random.beta(2, 10, half),
        "vowel_ratio": np.random.beta(4, 6, half),
        "consonant_ratio": np.random.beta(6, 4, half),
        "vowel_consonant_ratio": np.random.beta(4, 6, half),
        "special_ratio": np.random.beta(1, 40, half),
        "subdomain_digit_ratio": np.random.beta(1, 20, half),
        "subdomain_hex_ratio": np.random.beta(1, 15, half),
        "unique_char_ratio": np.random.normal(0.65, 0.1, half).clip(0.3, 1.0),
        "max_consecutive_consonants": np.random.choice([1, 2, 3, 4], half, p=[0.4, 0.4, 0.15, 0.05]),
        "queries_per_minute": np.random.exponential(1.5, half).clip(0.5, 15.0),
        "label": 0
    })

    # Malicious exfiltration: long labels, high entropy, dense hex/base32 chars, frequency bursts
    exfil = pd.DataFrame({
        "domain_len": np.random.normal(62, 14, half).clip(28, 140),
        "subdomain_len": np.random.normal(48, 12, half).clip(18, 110),
        "num_labels": np.random.choice([3, 4, 5], half, p=[0.35, 0.45, 0.20]),
        "max_label_len": np.random.normal(38, 10, half).clip(18, 63),
        "min_label_len": np.random.normal(3, 1, half).clip(1, 10),
        "avg_label_len": np.random.normal(19, 5, half).clip(8, 40),
        "entropy": np.random.normal(4.35, 0.3, half).clip(3.2, 5.2),
        "subdomain_entropy": np.random.normal(4.15, 0.35, half).clip(3.0, 5.0),
        "max_label_entropy": np.random.normal(3.8, 0.4, half).clip(2.5, 4.8),
        "digit_ratio": np.random.beta(5, 10, half),
        "alpha_ratio": np.random.beta(12, 5, half),
        "hex_ratio": np.random.beta(8, 4, half),
        "vowel_ratio": np.random.beta(2, 8, half),
        "consonant_ratio": np.random.beta(8, 2, half),
        "vowel_consonant_ratio": np.random.beta(2, 8, half),
        "special_ratio": np.random.beta(2, 20, half),
        "subdomain_digit_ratio": np.random.beta(5, 7, half),
        "subdomain_hex_ratio": np.random.beta(7, 4, half),
        "unique_char_ratio": np.random.normal(0.85, 0.08, half).clip(0.6, 1.0),
        "max_consecutive_consonants": np.random.choice([3, 4, 5, 6, 7], half, p=[0.1, 0.2, 0.3, 0.25, 0.15]),
        "queries_per_minute": np.random.exponential(12.0, half).clip(0.1, 80.0),
        "label": 1
    })

    df = pd.concat([benign, exfil], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def get_base_model(algo_name: str, config: dict, random_state: int = 42):
    """Instantiate base classifier from algorithm name and configuration."""
    algo = algo_name.lower()
    training_cfg = config.get("training", {})
    hp = training_cfg.get("hyperparameters", {})

    if algo in ("decision_tree", "dt"):
        dt_params = hp.get("decision_tree", {})
        return DecisionTreeClassifier(
            criterion=dt_params.get("criterion", "gini"),
            max_depth=dt_params.get("max_depth", 10),
            min_samples_split=dt_params.get("min_samples_split", 5),
            min_samples_leaf=dt_params.get("min_samples_leaf", 2),
            random_state=random_state,
        )

    elif algo in ("random_forest", "rf"):
        rf_params = hp.get("random_forest", {})
        return RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 100),
            max_depth=rf_params.get("max_depth", 15),
            min_samples_split=rf_params.get("min_samples_split", 5),
            min_samples_leaf=rf_params.get("min_samples_leaf", 2),
            n_jobs=rf_params.get("n_jobs", -1),
            random_state=random_state,
        )

    elif algo in ("xgboost", "xgb"):
        if XGB_AVAILABLE:
            xgb_params = hp.get("xgboost", {})
            return XGBClassifier(
                n_estimators=xgb_params.get("n_estimators", 100),
                max_depth=xgb_params.get("max_depth", 6),
                learning_rate=xgb_params.get("learning_rate", 0.1),
                eval_metric=xgb_params.get("eval_metric", "logloss"),
                random_state=random_state,
            )
        else:
            print("[!] XGBoost not installed. Falling back to GradientBoostingClassifier.")
            return GradientBoostingClassifier(random_state=random_state)

    else:
        raise ValueError(f"Unsupported algorithm '{algo_name}'. Choose from 'decision_tree', 'random_forest', 'xgboost'.")


def tune_model(
    algo_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform hyperparameter tuning with GridSearchCV using Stratified 5-Fold Cross Validation.
    """
    algo = algo_name.lower()
    base_model = get_base_model(algo, config, random_state=random_state)
    tuning_grids = config.get("training", {}).get("tuning_grids", {})
    param_grid = tuning_grids.get(algo, {})

    if not param_grid:
        # Default tuning grid fallbacks
        if algo in ("decision_tree", "dt"):
            param_grid = {
                "max_depth": [5, 10, 15],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
        elif algo in ("random_forest", "rf"):
            param_grid = {
                "n_estimators": [50, 100, 150],
                "max_depth": [10, 15, 20],
                "min_samples_split": [2, 5],
            }
        elif algo in ("xgboost", "xgb"):
            param_grid = {
                "n_estimators": [50, 100],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.2],
            }

    print(f"[*] Hyperparameter tuning for '{algo_name}' using Stratified {cv_folds}-Fold CV...")
    print(f"    Search grid: {param_grid}")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=skf,
        n_jobs=-1,
        verbose=0
    )

    t0 = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"[✓] Best hyperparameters found in {elapsed:.2f}s:")
    for param, val in grid_search.best_params_.items():
        print(f"    - {param}: {val}")
    print(f"    - Best Cross-Validation F1-Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform stratified 5-fold cross-validation and evaluate on holdout test set.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv_results = cross_validate(model, X_train, y_train, cv=skf, scoring=cv_scoring)

    # Train on full training split
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    # Predictions on holdout test split
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "cv_f1_mean": float(np.mean(cv_results["test_f1"])),
        "cv_f1_std": float(np.std(cv_results["test_f1"])),
        "cv_roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "fit_time_seconds": round(fit_time, 3),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return metrics


def train_pipeline(
    config_path: str = "configs/config.yaml",
    dataset_path: Optional[str] = None,
    algorithm: Optional[str] = None,
    tune: bool = False,
    save_model: bool = True
) -> Tuple[Dict[str, Any], Any]:
    """Execute end-to-end model training, validation, and evaluation pipeline."""
    config = load_config(config_path)
    data_file = dataset_path or config["paths"].get("features_dataset_csv")
    algo = algorithm or config["training"].get("algorithm", "random_forest")
    seed = config["training"].get("random_state", 42)
    cv_folds = config["training"].get("cross_validation_folds", 5)

    # Load dataset
    if data_file and os.path.exists(data_file):
        print(f"[+] Loading feature dataset from {data_file}...")
        df = pd.read_csv(data_file)
    else:
        print(f"[!] Dataset '{data_file}' not found. Generating synthetic benchmark dataset...")
        df = generate_synthetic_dataset(n_samples=2500, seed=seed)
        fallback_path = Path("data/processed/dns_features.csv")
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(fallback_path, index=False)
        print(f"[✓] Saved benchmark dataset to {fallback_path}")

    # Prepare features and labels
    drop_cols = [c for c in ["domain", "query", "qname", "label", "attack_type"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["label"].astype(int)

    test_size = config["training"].get("test_size", 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    print(f"[*] Training Data: {X_train.shape[0]} samples | Test Data: {X_test.shape[0]} samples")
    print(f"[*] Total Features ({X.shape[1]}): {list(X.columns)}")

    # Model training or tuning
    if tune:
        model, best_params = tune_model(algo, X_train, y_train, config, cv_folds=cv_folds, random_state=seed)
    else:
        model = get_base_model(algo, config, random_state=seed)

    print(f"[*] Evaluating '{algo}' with Stratified {cv_folds}-Fold Cross-Validation...")
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, cv_folds=cv_folds, random_state=seed)

    print("\n" + "=" * 65)
    print(f"             MODEL EVALUATION REPORT: {algo.upper()}             ")
    print("=" * 65)
    print(f"  5-Fold CV F1-Score:    {metrics['cv_f1_mean']:.4f} (+/- {metrics['cv_f1_std']:.4f})")
    print(f"  5-Fold CV ROC-AUC:     {metrics['cv_roc_auc_mean']:.4f} (+/- {metrics['cv_roc_auc_std']:.4f})")
    print("-" * 65)
    print(f"  Holdout Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  Holdout Test Precision:{metrics['test_precision']:.4f}")
    print(f"  Holdout Test Recall:   {metrics['test_recall']:.4f}")
    print(f"  Holdout Test F1-Score: {metrics['test_f1']:.4f}")
    print(f"  Holdout Test ROC-AUC:  {metrics['test_roc_auc']:.4f}")
    print(f"  Training Fit Time:     {metrics['fit_time_seconds']:.3f}s")
    print("-" * 65)
    cm = metrics["confusion_matrix"]
    print(f"  Confusion Matrix: [[TN={cm[0][0]}, FP={cm[0][1]}], [FN={cm[1][0]}, TP={cm[1][1]}]]")
    print("=" * 65)

    # Top feature importances
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("\nTop 7 Most Informative Features:")
        for feat, score in importances.head(7).items():
            print(f"  {feat:28}: {score:.4f}")

    # Persist model
    if save_model:
        models_dir = Path(config["paths"].get("models_dir", "models"))
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{algo}.joblib"
        joblib.dump(model, model_path)
        print(f"\n[✓] Saved trained model artifact to: {model_path}")

        # Also save default model path if specified
        default_out = Path(config["paths"].get("model_output_path", "models/dns_exfil_detector.joblib"))
        joblib.dump(model, default_out)

    return metrics, model


def benchmark_all_models(
    config_path: str = "configs/config.yaml",
    dataset_path: Optional[str] = None,
    tune: bool = False
) -> pd.DataFrame:
    """
    Train and benchmark all three models (Decision Tree, Random Forest, XGBoost)
    using Stratified 5-Fold Cross-Validation, reporting comparative metrics table.
    """
    algorithms = ["decision_tree", "random_forest", "xgboost"]
    config = load_config(config_path)
    results = []

    print("=" * 70)
    print("      BENCHMARKING DECISION TREE, RANDOM FOREST, AND XGBOOST       ")
    print("=" * 70)

    for algo in algorithms:
        print(f"\n>>> Running pipeline for: {algo.upper()} (Tune={tune}) <<<")
        metrics, _ = train_pipeline(
            config_path=config_path,
            dataset_path=dataset_path,
            algorithm=algo,
            tune=tune,
            save_model=True
        )
        row = {
            "Algorithm": algo,
            "CV F1 (mean)": round(metrics["cv_f1_mean"], 4),
            "CV F1 (std)": round(metrics["cv_f1_std"], 4),
            "CV ROC-AUC": round(metrics["cv_roc_auc_mean"], 4),
            "Test Accuracy": round(metrics["test_accuracy"], 4),
            "Test Precision": round(metrics["test_precision"], 4),
            "Test Recall": round(metrics["test_recall"], 4),
            "Test F1": round(metrics["test_f1"], 4),
            "Test ROC-AUC": round(metrics["test_roc_auc"], 4),
            "Fit Time (s)": metrics["fit_time_seconds"]
        }
        results.append(row)

    df_results = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("                     FINAL BENCHMARK COMPARISON TABLE                    ")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)

    # Save benchmark report to models/benchmark_report.json and .csv
    models_dir = Path(config["paths"].get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    report_json = models_dir / "benchmark_report.json"
    report_csv = models_dir / "benchmark_report.csv"

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    df_results.to_csv(report_csv, index=False)

    print(f"[✓] Benchmark reports saved to:\n    - {report_json}\n    - {report_csv}")
    return df_results


def main():
    parser = argparse.ArgumentParser(
        description="Week 5: Train and Tune Baseline Decision Trees, Random Forest, and XGBoost"
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--data", help="Path to processed features CSV dataset")
    parser.add_argument(
        "--algo",
        choices=["decision_tree", "random_forest", "xgboost", "all"],
        default="random_forest",
        help="Model algorithm to train (or 'all' for comparative benchmark)"
    )
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter grid search tuning")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark all 3 models side-by-side")

    args = parser.parse_args()

    if args.benchmark or args.algo == "all":
        benchmark_all_models(config_path=args.config, dataset_path=args.data, tune=args.tune)
    else:
        train_pipeline(
            config_path=args.config,
            dataset_path=args.data,
            algorithm=args.algo,
            tune=args.tune,
            save_model=True
        )


if __name__ == "__main__":
    main()
