"""
DNS Exfiltration Detection — Visual Simulation Runner
======================================================
Orchestrates the full detection pipeline with a rich terminal UI:

  Stage 1 ▸ Simulate  – craft RFC-compliant DNS exfiltration queries
  Stage 2 ▸ Capture   – synthesize realistic mixed traffic (benign + attack)
  Stage 3 ▸ Features  – extract lexical / entropy / frequency features
  Stage 4 ▸ Train     – train & evaluate ML classifiers

Usage:
    python -m src.simulate [options]
    python src/simulate.py   --payload "secret=token123" --mode burst
    python src/simulate.py   --demo
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Rich terminal UI ──────────────────────────────────────────────────────────
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

# ── Project modules ──────────────────────────────────────────────────────────
try:
    from src.dns_simulator import DNSExfiltrationSimulator, validate_rfc_compliance
    from src.capture_traffic import generate_synthesized_traffic
    from src.features import process_traffic_dataset, extract_domain_features
except ImportError:
    from dns_simulator import DNSExfiltrationSimulator, validate_rfc_compliance
    from capture_traffic import generate_synthesized_traffic
    from features import process_traffic_dataset, extract_domain_features

import pandas as pd
import numpy as np

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
C_GOOD    = "bold green"
C_BAD     = "bold red"
C_WARN    = "bold yellow"
C_INFO    = "bold cyan"
C_DIM     = "dim"
C_HEADER  = "bold magenta"
C_ACCENT  = "bold blue"
C_NEUTRAL = "white"

BANNER = r"""
 ██████╗ ███╗   ██╗███████╗    ███████╗██╗  ██╗███████╗██╗██╗
 ██╔══██╗████╗  ██║██╔════╝    ██╔════╝╚██╗██╔╝██╔════╝██║██║
 ██║  ██║██╔██╗ ██║███████╗    █████╗   ╚███╔╝ █████╗  ██║██║
 ██║  ██║██║╚██╗██║╚════██║    ██╔══╝   ██╔██╗ ██╔══╝  ██║██║
 ██████╔╝██║ ╚████║███████║    ███████╗██╔╝ ██╗██║     ██║███████╗
 ╚═════╝ ╚═╝  ╚═══╝╚══════╝    ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝
          DNS  Exfiltration  Detection  ·  Visual  Simulation
"""


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stage_rule(title: str, color: str = "cyan") -> None:
    console.print()
    console.print(Rule(f"[bold {color}]  {title}  [/]", style=color))
    console.print()


def _sparkline(values: List[float], width: int = 40) -> str:
    """Render a Unicode block-element spark-line for a sequence of values."""
    BLOCKS = " ▁▂▃▄▅▆▇█"
    if not values:
        return " " * width
    mn, mx = min(values), max(values)
    span = mx - mn if mx != mn else 1.0
    chars = [BLOCKS[min(8, int((v - mn) / span * 8))] for v in values]
    # Resize to target width
    ratio = len(chars) / max(width, 1)
    line = "".join(chars[int(i * ratio)] for i in range(width))
    return line


def _ascii_bar_chart(
    data: Dict[str, float],
    width: int = 35,
    title: str = "",
    color_map: Optional[Dict[str, str]] = None,
) -> Panel:
    """Return a Rich Panel containing a horizontal ASCII bar chart."""
    if not data:
        return Panel("No data", title=title)
    max_val = max(data.values()) if data.values() else 1.0
    grid = Table.grid(padding=(0, 1))
    grid.add_column(justify="right", min_width=22)
    grid.add_column(justify="left")
    grid.add_column(justify="right", min_width=7)
    for label, val in data.items():
        bar_len = max(1, int(val / max_val * width))
        color = (color_map or {}).get(label, "cyan")
        bar = Text("█" * bar_len, style=color)
        grid.add_row(
            Text(label, style="bold white"),
            bar,
            Text(f"{val:.2f}", style="dim"),
        )
    return Panel(grid, title=f"[bold]{title}[/]", border_style="cyan", padding=(0, 1))


def _feature_heatmap(features: Dict[str, Any], title: str = "Feature Vector") -> Panel:
    """Display a feature dict as a two-column colour-coded table."""
    t = Table(show_header=True, header_style="bold cyan", box=box.MINIMAL_DOUBLE_HEAD, padding=(0, 1))
    t.add_column("Feature", style="bold white", no_wrap=True)
    t.add_column("Value", justify="right")
    t.add_column("Indicator", justify="center")

    HIGH_ALERT = {"entropy", "subdomain_entropy", "max_label_entropy",
                  "max_label_len", "domain_len", "hex_ratio", "digit_ratio"}
    for k, v in features.items():
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        if k in HIGH_ALERT and isinstance(v, (int, float)):
            indicator = "[red]▲[/]" if v > 3.5 else "[green]●[/]"
        else:
            indicator = "[dim]·[/]"
        t.add_row(k, val_str, indicator)
    return Panel(t, title=f"[bold magenta]{title}[/]", border_style="magenta")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Exfiltration Simulation
# ─────────────────────────────────────────────────────────────────────────────

def stage_simulate(
    payload: bytes,
    domain: str = "tunnel.example.com",
    encoding: str = "base32",
    mode: str = "burst",
    qtype: str = "TXT",
    dry_run: bool = True,
) -> Tuple[List[str], "DNSExfiltrationSimulator"]:
    _stage_rule("STAGE 1 — DNS Exfiltration Simulation", "red")

    delay, jitter = (0.01, 0.002) if mode == "burst" else (2.0, 0.5)

    sim = DNSExfiltrationSimulator(
        target_domain=domain,
        encoding=encoding,
        delay=delay,
        jitter=jitter,
    )

    queries = sim.generate_queries(payload)

    # ── Summary panel ──────────────────────────────────────────────────────
    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold cyan")
    info.add_column(style="white")
    rows = [
        ("Payload size",  f"{len(payload)} bytes"),
        ("Encoding",      f"[yellow]{encoding.upper()}[/]"),
        ("Mode",          f"[{'red' if mode == 'burst' else 'blue'}]{mode.upper()}[/]"),
        ("Query type",    qtype),
        ("Target domain", f"[green]{domain}[/]"),
        ("Total queries", f"[bold white]{len(queries)}[/]"),
        ("Delay / jitter",f"{delay}s ± {jitter}s"),
    ]
    for k, v in rows:
        info.add_row(k, v)

    console.print(Panel(info, title="[bold red]Exfiltration Config[/]", border_style="red"))

    # ── Query preview ──────────────────────────────────────────────────────
    preview = Table(
        "Idx", "RFC?", "FQDN (truncated)", "Label Length",
        title="[bold]Query Preview[/]",
        box=box.ROUNDED,
        show_lines=False,
        border_style="dim red",
        header_style="bold red",
    )
    preview.add_column("Idx", style="dim", width=5, justify="right")
    preview.add_column("RFC?", width=5, justify="center")
    preview.add_column("FQDN (truncated)", style="green", no_wrap=False)
    preview.add_column("Len", justify="right", style="cyan")

    sample = queries[:8] + (["…"] if len(queries) > 8 else [])
    for i, q in enumerate(queries[:8], 1):
        valid = "✔" if validate_rfc_compliance(q) else "✘"
        color = C_GOOD if valid == "✔" else C_BAD
        preview.add_row(
            str(i),
            f"[{color}]{valid}[/]",
            q[:80] + ("…" if len(q) > 80 else ""),
            str(len(q)),
        )
    if len(queries) > 8:
        preview.add_row("…", "…", f"[dim]…and {len(queries) - 8} more queries[/]", "")
    console.print(preview)

    # ── Animated progress (dry-run) ────────────────────────────────────────
    progress = Progress(
        SpinnerColumn("dots2", style="red"),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=40, style="red", complete_style="bright_red"),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("[red]Encoding & chunking payload…", total=len(queries))
        for _ in queries:
            time.sleep(0.003)
            progress.advance(task)

    rfc_pass = sum(1 for q in queries if validate_rfc_compliance(q))
    console.print(
        f"\n  [bold green]✔ {rfc_pass}/{len(queries)} queries passed RFC 1035 validation[/]"
    )
    if dry_run:
        console.print("  [dim yellow]DRY-RUN mode — no packets sent to network.[/]\n")

    return queries, sim


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – Traffic Synthesis & Capture
# ─────────────────────────────────────────────────────────────────────────────

def stage_capture(
    benign_count: int = 200,
    burst_count: int = 150,
    low_slow_count: int = 50,
    output_csv: str = "data/processed/labeled_traffic.csv",
) -> pd.DataFrame:
    _stage_rule("STAGE 2 — Traffic Synthesis & Capture", "yellow")

    console.print(
        Panel(
            f"  Benign queries    : [green]{benign_count}[/]\n"
            f"  Burst exfil       : [red]{burst_count}[/]\n"
            f"  Low-and-slow exfil: [yellow]{low_slow_count}[/]\n"
            f"  Output CSV        : [dim]{output_csv}[/]",
            title="[bold yellow]Synthesis Parameters[/]",
            border_style="yellow",
        )
    )

    with Progress(
        SpinnerColumn("earth", style="yellow"),
        TextColumn("[bold yellow]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:
        task = prog.add_task("Synthesizing mixed DNS traffic session…", total=None)
        records, _ = generate_synthesized_traffic(
            benign_count=benign_count,
            burst_count=burst_count,
            low_slow_count=low_slow_count,
            target_tunnel="tunnel.corp-exfil.net",
        )
        prog.update(task, description="[bold green]Traffic synthesis complete.")

    df = pd.DataFrame(records)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # ── Traffic timeline mini-chart ────────────────────────────────────────
    label_counts = df["label"].value_counts()
    attack_counts = df["attack_type"].value_counts()

    summary = Table(
        title="[bold yellow]Traffic Session Summary[/]",
        box=box.ROUNDED,
        border_style="yellow",
        header_style="bold yellow",
    )
    summary.add_column("Class", style="bold white")
    summary.add_column("Count", justify="right", style="cyan")
    summary.add_column("Pct", justify="right")

    total = len(df)
    for cls, cnt in attack_counts.items():
        color = "green" if cls == "benign" else "red"
        summary.add_row(
            f"[{color}]{cls}[/]",
            str(cnt),
            f"[dim]{cnt / total * 100:.1f}%[/]",
        )
    console.print(summary)

    # ── Sparkline timeline ─────────────────────────────────────────────────
    df_sorted = df.sort_values("timestamp")
    ts_arr = df_sorted["timestamp"].values
    label_arr = df_sorted["label"].values
    n_bins = 60
    bins = np.array_split(label_arr, n_bins)
    exfil_rate = [b.mean() for b in bins if len(b)]

    spark = _sparkline(exfil_rate, width=58)
    console.print(
        Panel(
            f"  [dim]Low ──────────────────────────── High[/]\n"
            f"  [red]{spark}[/]\n"
            f"  [dim]Exfiltration intensity over session timeline (60 bins)[/]",
            title="[bold yellow]Attack Timeline Sparkline[/]",
            border_style="dim yellow",
        )
    )

    console.print(f"\n  [bold green]✔ Saved {total} records → [white]{out_path}[/][/]\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def stage_features(
    df_traffic: pd.DataFrame,
    output_csv: str = "data/processed/dns_features.csv",
) -> pd.DataFrame:
    _stage_rule("STAGE 3 — Feature Engineering", "cyan")

    n = len(df_traffic)
    console.print(f"  Extracting features from [bold]{n}[/] DNS records…\n")

    # Progress with batch processing
    batch_size = max(1, n // 80)
    records_out: List[Dict[str, Any]] = []

    progress = Progress(
        SpinnerColumn("toggle", style="cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=45, style="cyan", complete_style="bright_cyan"),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Computing lexical / entropy / frequency features…", total=n)
        for i in range(0, n, batch_size):
            batch = df_traffic.iloc[i : i + batch_size]
            for _, row in batch.iterrows():
                feats = extract_domain_features(str(row.get("qname", "")))
                feats["domain"] = str(row.get("qname", ""))
                feats["label"] = int(row.get("label", 0))
                feats["attack_type"] = str(row.get("attack_type", "unknown"))
                feats["queries_per_minute"] = 1.0
                records_out.append(feats)
            progress.advance(task, len(batch))

    df_feats = pd.DataFrame(records_out)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_feats.to_csv(out_path, index=False)

    # ── Benign vs Exfil feature comparison ────────────────────────────────
    numeric_cols = [c for c in df_feats.columns
                    if c not in ("domain", "label", "attack_type")
                    and pd.api.types.is_numeric_dtype(df_feats[c])]

    benign_mean = df_feats[df_feats["label"] == 0][numeric_cols].mean()
    exfil_mean  = df_feats[df_feats["label"] == 1][numeric_cols].mean()

    top_feats = ["domain_len", "subdomain_len", "max_label_len",
                 "entropy", "subdomain_entropy", "hex_ratio",
                 "digit_ratio", "unique_char_ratio", "max_consecutive_consonants"]
    top_feats = [f for f in top_feats if f in numeric_cols]

    comp = Table(
        title="[bold cyan]Feature Contrast: Benign vs Exfiltration[/]",
        box=box.DOUBLE_EDGE,
        border_style="cyan",
        header_style="bold cyan",
        show_lines=True,
    )
    comp.add_column("Feature",       style="bold white", no_wrap=True)
    comp.add_column("Benign (mean)", justify="right", style="green")
    comp.add_column("Exfil (mean)",  justify="right", style="red")
    comp.add_column("Δ",             justify="right", style="yellow")

    for feat in top_feats:
        b = benign_mean.get(feat, 0.0)
        e = exfil_mean.get(feat, 0.0)
        delta = e - b
        sign = "▲" if delta > 0 else "▼"
        comp.add_row(feat, f"{b:.3f}", f"{e:.3f}", f"{sign} {abs(delta):.3f}")
    console.print(comp)

    # ── Sample feature vector for one exfil query ─────────────────────────
    exfil_sample = df_feats[df_feats["label"] == 1].head(1)
    if not exfil_sample.empty:
        sample_feats = {
            k: v for k, v in exfil_sample.iloc[0].to_dict().items()
            if k not in ("domain", "label", "attack_type")
            and isinstance(v, (int, float))
        }
        console.print(
            _feature_heatmap(
                {k: round(float(v), 4) for k, v in sample_feats.items()},
                title="Sample Exfil Query Feature Vector"
            )
        )

    console.print(
        f"\n  [bold green]✔ Feature matrix: {df_feats.shape[0]} rows × {df_feats.shape[1]} cols → [white]{out_path}[/][/]\n"
    )
    return df_feats


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 – ML Training & Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _confusion_matrix_panel(cm: List[List[int]], title: str = "Confusion Matrix") -> Panel:
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    t = Table(show_header=True, header_style="bold white", box=box.SIMPLE_HEAD)
    t.add_column("",            style="bold dim", justify="center")
    t.add_column("Pred Benign", justify="center")
    t.add_column("Pred Exfil",  justify="center")
    t.add_row("Actual Benign", f"[green]TN={tn}[/]", f"[yellow]FP={fp}[/]")
    t.add_row("Actual Exfil",  f"[red]FN={fn}[/]",   f"[green]TP={tp}[/]")
    return Panel(t, title=f"[bold]{title}[/]", border_style="magenta", padding=(0, 1))


def stage_train(
    df_feats: pd.DataFrame,
    algorithms: Optional[List[str]] = None,
) -> None:
    _stage_rule("STAGE 4 — ML Model Training & Evaluation", "magenta")

    algorithms = algorithms or ["decision_tree", "random_forest"]

    try:
        from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix as sk_cm,
        )
    except ImportError:
        console.print("[bold red]scikit-learn not installed — skipping training stage.[/]")
        return

    drop_cols = [c for c in ["domain", "qname", "label", "attack_type"] if c in df_feats.columns]
    X = df_feats.drop(columns=drop_cols)
    y = df_feats["label"].astype(int)

    # Remove any non-numeric or all-NaN columns
    X = X.select_dtypes(include=[np.number]).fillna(0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    console.print(
        f"  Train: [bold]{len(X_train)}[/] samples  |  Test: [bold]{len(X_test)}[/] samples  |"
        f"  Features: [bold]{X.shape[1]}[/]\n"
    )

    MODEL_MAP = {
        "decision_tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    }

    try:
        from xgboost import XGBClassifier
        MODEL_MAP["xgboost"] = XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42)
    except ImportError:
        pass

    all_results: List[Dict[str, Any]] = []

    for algo in algorithms:
        if algo not in MODEL_MAP:
            console.print(f"  [yellow]Skipping unknown algorithm: {algo}[/]")
            continue

        model = MODEL_MAP[algo]
        console.print(f"  [bold magenta]▶ Training {algo.upper()}…[/]")

        with Progress(
            SpinnerColumn("star", style="magenta"),
            TextColumn(f"[magenta]{{task.description}}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as prog:
            task = prog.add_task(f"Fitting {algo}…", total=None)
            t0 = time.time()
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_res = cross_validate(
                model, X_train, y_train, cv=skf,
                scoring={"f1": "f1", "roc_auc": "roc_auc"},
            )
            model.fit(X_train, y_train)
            fit_time = time.time() - t0
            prog.update(task, description=f"[green]Done in {fit_time:.2f}s")

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        cm      = sk_cm(y_test, y_pred).tolist()

        metrics = {
            "algorithm":    algo,
            "cv_f1_mean":   float(np.mean(cv_res["test_f1"])),
            "cv_f1_std":    float(np.std(cv_res["test_f1"])),
            "cv_auc_mean":  float(np.mean(cv_res["test_roc_auc"])),
            "test_accuracy":  float(accuracy_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "test_recall":    float(recall_score(y_test, y_pred, zero_division=0)),
            "test_f1":        float(f1_score(y_test, y_pred, zero_division=0)),
            "test_roc_auc":   float(roc_auc_score(y_test, y_proba)),
            "confusion_matrix": cm,
            "fit_time": fit_time,
        }
        all_results.append(metrics)

        # Per-model result panel
        res_table = Table.grid(padding=(0, 2))
        res_table.add_column(style="bold cyan")
        res_table.add_column(style="white")
        res_rows = [
            ("5-Fold CV F1",      f"{metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}"),
            ("5-Fold CV ROC-AUC", f"{metrics['cv_auc_mean']:.4f}"),
            ("Test Accuracy",     f"[bold]{metrics['test_accuracy']:.4f}[/]"),
            ("Test Precision",    f"{metrics['test_precision']:.4f}"),
            ("Test Recall",       f"{metrics['test_recall']:.4f}"),
            ("Test F1",           f"[bold green]{metrics['test_f1']:.4f}[/]"),
            ("Test ROC-AUC",      f"[bold green]{metrics['test_roc_auc']:.4f}[/]"),
            ("Fit Time",          f"{metrics['fit_time']:.3f}s"),
        ]
        for k, v in res_rows:
            res_table.add_row(k, v)

        console.print(
            Columns([
                Panel(res_table, title=f"[bold magenta]{algo.upper()} Metrics[/]",
                      border_style="magenta", padding=(0, 1)),
                _confusion_matrix_panel(cm, title="Confusion Matrix"),
            ])
        )

        # Feature importances
        if hasattr(model, "feature_importances_"):
            importances = sorted(
                zip(X.columns, model.feature_importances_),
                key=lambda x: x[1], reverse=True
            )[:8]
            fi_data = {k: float(v) for k, v in importances}
            console.print(
                _ascii_bar_chart(
                    fi_data,
                    title=f"{algo.upper()} — Top Feature Importances",
                    color_map={k: "magenta" for k in fi_data},
                )
            )

    # ── Comparative benchmark table ────────────────────────────────────────
    if len(all_results) > 1:
        _stage_rule("Benchmark Comparison", "white")
        bench = Table(
            title="[bold white]Model Comparison[/]",
            box=box.DOUBLE_EDGE,
            border_style="white",
            header_style="bold white",
            show_lines=True,
        )
        for col in ("Algorithm", "CV F1", "CV AUC", "Test Acc", "Test F1", "Test AUC", "Time(s)"):
            bench.add_column(col, justify="center")
        for r in all_results:
            bench.add_row(
                r["algorithm"],
                f"{r['cv_f1_mean']:.4f}",
                f"{r['cv_auc_mean']:.4f}",
                f"{r['test_accuracy']:.4f}",
                f"[bold green]{r['test_f1']:.4f}[/]",
                f"[bold green]{r['test_roc_auc']:.4f}[/]",
                f"{r['fit_time']:.2f}",
            )
        console.print(bench)


# ─────────────────────────────────────────────────────────────────────────────
# Live packet ticker (visual effect during simulation)
# ─────────────────────────────────────────────────────────────────────────────

def _live_packet_feed(queries: List[str], max_show: int = 18, speed: float = 0.06) -> None:
    """Show a scrolling live packet feed using Rich Live."""
    console.print()
    console.print(Rule("[bold red]  Live DNS Exfiltration Feed  [/]", style="red"))
    console.print()

    feed: deque = deque(maxlen=max_show)
    icons = ["📡", "🔴", "🟠", "⚠️ ", "🔺"]

    def _render() -> Panel:
        text = Text()
        for ts, fqdn in feed:
            text.append(f" {random.choice(icons)}  ", style="red")
            text.append(f"{ts}  ", style="dim")
            text.append(fqdn[:85] + ("…" if len(fqdn) > 85 else "") + "\n",
                        style="bold green" if len(fqdn) < 60 else "bold red")
        return Panel(
            text,
            title=f"[red]DNS Queries — {len(queries)} total[/]",
            border_style="red",
            padding=(0, 1),
        )

    sample = queries[:60]
    with Live(_render(), console=console, refresh_per_second=12) as live:
        for i, q in enumerate(sample):
            t = time.strftime("%H:%M:%S")
            feed.append((t, q))
            live.update(_render())
            time.sleep(speed)

    if len(queries) > 60:
        console.print(f"  [dim]…and {len(queries) - 60} more queries (truncated for display)[/]")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DNS Exfiltration Detection — Full Visual Simulation"
    )
    parser.add_argument("--payload",   default=None,       help="Text payload to exfiltrate (default: demo secret)")
    parser.add_argument("--domain",    default="tunnel.example.com", help="Attacker-controlled tunnel domain")
    parser.add_argument("--encoding",  choices=["base32", "hex", "base64"], default="base32")
    parser.add_argument("--mode",      choices=["burst", "low-and-slow"], default="burst")
    parser.add_argument("--qtype",     choices=["A", "TXT", "AAAA", "CNAME"], default="TXT")
    parser.add_argument("--benign",    type=int, default=200, help="Benign queries in synthesis")
    parser.add_argument("--burst",     type=int, default=120, help="Burst exfil queries in synthesis")
    parser.add_argument("--slow",      type=int, default=40,  help="Low-and-slow exfil queries")
    parser.add_argument("--algos",     nargs="+", default=["decision_tree", "random_forest"],
                        help="ML algorithms to train (decision_tree / random_forest / xgboost)")
    parser.add_argument("--skip-train",  action="store_true", help="Skip ML training stage")
    parser.add_argument("--skip-feed",   action="store_true", help="Skip the live packet ticker")
    parser.add_argument("--demo",        action="store_true", help="Quick demo with small counts")
    args = parser.parse_args()

    if args.demo:
        args.benign, args.burst, args.slow = 60, 40, 15

    # ── Banner ────────────────────────────────────────────────────────────
    console.print(Text(BANNER, style="bold red"), justify="center")
    console.print(
        Align.center(
            Panel(
                "[dim]Educational simulation — no real packets are transmitted[/]",
                style="dim",
                border_style="dim",
            )
        )
    )
    console.print()

    # ── Payload ───────────────────────────────────────────────────────────
    if args.payload:
        payload = args.payload.encode()
    else:
        payload = (
            b"CRITICAL_EXFIL: user=admin token=ghp_x89Fa9c01B2dE "
            b"db_pass=Sup3rS3cr3t! timestamp=1710000000 target=192.168.10.50"
        )
        console.print(
            Panel(
                f"[dim]Using default demo payload ({len(payload)} bytes)[/]",
                style="dim yellow", border_style="dim yellow",
            )
        )

    # ── Stage 1 ──────────────────────────────────────────────────────────
    queries, sim = stage_simulate(
        payload=payload,
        domain=args.domain,
        encoding=args.encoding,
        mode=args.mode,
        qtype=args.qtype,
        dry_run=True,
    )

    # ── Live feed ─────────────────────────────────────────────────────────
    if not args.skip_feed:
        _live_packet_feed(queries, max_show=16, speed=0.04)

    # ── Stage 2 ──────────────────────────────────────────────────────────
    df_traffic = stage_capture(
        benign_count=args.benign,
        burst_count=args.burst,
        low_slow_count=args.slow,
    )

    # ── Stage 3 ──────────────────────────────────────────────────────────
    df_feats = stage_features(df_traffic)

    # ── Stage 4 ──────────────────────────────────────────────────────────
    if not args.skip_train:
        stage_train(df_feats, algorithms=args.algos)

    # ── Final summary ─────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold green]  Simulation Complete  [/]", style="green"))
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")
    summary.add_row("Total queries simulated",  str(len(queries)))
    summary.add_row("Traffic records synthesized", str(len(df_traffic)))
    summary.add_row("Feature vectors extracted", str(len(df_feats)))
    summary.add_row("Models trained", str(len(args.algos)) if not args.skip_train else "skipped")
    summary.add_row("Artifacts saved to", "[dim]data/processed/[/]")
    console.print(
        Panel(summary, title="[bold green]Pipeline Summary[/]", border_style="green")
    )


if __name__ == "__main__":
    main()
