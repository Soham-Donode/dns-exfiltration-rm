"""
Shared experiment pipeline for notebooks 02-05
=============================================
One place that builds the data, the splits, the detectors and the metrics, so the baseline and
every improvement step are evaluated on IDENTICAL data with IDENTICAL metrics.

Data (prepare):
    real       CIC-Bell-DNS-EXF-2021 PCAPs -> pcap_features (labels by query shape)
               split per capture by time: first 70% = real_train, last 30% = real_test
    synthetic  evasion_data: 12 evasive families (pure streams, varied timing), hard-benign
               stream (first half = hard_train, second half = hard_test), mixed streams
               (one attacker diluted into normal traffic at 1/5/20/50%, plus a clean stream),
               train-only mixed streams for every family (5/20%) used by the augmented steps

Detectors (run_detectors):
    baseline          paper's 14 stateless features, trained on real_train      (any model)
    baseline+window   baseline RF + the paper's window rule (flag window if > 40% flagged)
    step1             + host temporal features                                  (real_train)
    step2             + per-domain statistical features                         (real_train)
    step3a            step2 features + augmented training with synthetic families (pure AND diluted
                      streams) and hard negatives, evaluated leave-one-encoding-out
    step3b            two-layer: char-model query classifier -> host/domain score aggregates
                      + step2 features, same augmentation and leave-one-encoding-out
    step4 behaviour   content-agnostic model: per-domain volume / diversity / novelty / timing /
                      bytes over 10 min and 1 h, NO character features; augmented, leave-one-encoding-out.
                      It ABSTAINS on names the capture cut off (base "<trunc>"): their domain is unknown,
                      so all of a host's long names share one bucket and look like one busy attacker
                      domain (cfg behaviour_on_trunc=True disables this).
    step4             step3a OR step4 behaviour (flag a query if either model flags it)

Every detector returns one score in [0, 1] per TEST query. For leave-one-encoding-out models a
query that belongs to no encoding (real_test, hard_test, clean stream) gets the mean over the
four fold models, so its false-positive rate is the average over folds.

Metrics (evaluate): query-level recall / FPR / F1 on each test set, recall per encoding and per
dilution rate. Everything is in percent.
"""

import glob
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import evasion_data as ev
import pcap_features as pf
import stateful_features as sf

DEFAULTS = dict(seed=42, window=100, dom_seconds=600.0, train_frac=0.7,
                n_family=1500, n_hard=8000, n_mixed=5000, mixed_rates=(0.01, 0.05, 0.2, 0.5),
                mixed_families=ev.MIXED_FAMILIES, snaplen=96, train_mixed_rates=(0.05, 0.2), n_train_mixed=3000,
                svm_max_train=10_000, n_estimators=100,
                paper_delta=0.4, long_seconds=3600.0, behaviour_on_trunc=False)

PAPER14 = ["FQDN_count", "subdomain_length", "upper", "lower", "numeric", "entropy", "special", "labels",
           "labels_max", "labels_average", "longest_word", "sld_code", "len", "subdomain"]
CHAR = ["entropy", "upper_ratio", "lower_ratio", "digit_ratio", "special_ratio", "vowel_ratio", "unique_char_ratio",
        "longest_word", "first_label_numeric", "ll_all", "ll_long", "long_len", "long_entropy", "hex_frac_long",
        "trans_long", "digit_frac_long"]
FEATURES = {
    "baseline": PAPER14,
    "step1": PAPER14 + sf.HOST_FEATURES,
    "step2": PAPER14 + sf.HOST_FEATURES + sf.DOMAIN_FEATURES,
    "step3_layer1": CHAR,
    "step3_layer2": PAPER14 + sf.HOST_FEATURES + sf.DOMAIN_FEATURES + ["l1_score"] + sf.SCORE_FEATURES,
    # behaviour only: no per-query lexical features, no subdomain length / entropy, no host features
    "step4_behaviour": ["d_n", "d_unique_sub", "d_unique_ratio", "d_gap_mean", "d_share", "d_bytes"]
                       + sf.BEHAVIOUR_FEATURES,
}
ENCODINGS = ["base32", "hex", "base64", "words"]
TEST_ROLES = ["real_test", "evasive", "hard_test", "mixed"]
KEYS = list(sf.KEYS)


# --------------------------------------------------------------------------- character model
_ALPHA = "abcdefghijklmnopqrstuvwxyz0123456789-_."
_IDX = {c: i for i, c in enumerate(_ALPHA)}
_K = len(_ALPHA) + 1


def _enc(s):
    return [_IDX.get(c, _K - 1) for c in s.lower()]


def train_bigram_lm(names):
    cnt = np.ones((_K, _K))
    for n in names:
        e = _enc(n)
        for a, b in zip(e, e[1:]):
            cnt[a, b] += 1
    return np.log(cnt / cnt.sum(1, keepdims=True))


def char_features(df, lm):
    def ll(s):
        e = _enc(s)
        return float(np.mean([lm[a, b] for a, b in zip(e, e[1:])])) if len(e) > 1 else 0.0
    cls = lambda c: 0 if c.isdigit() else 1 if c.isalpha() else 2
    rows = []
    for n in df["name"].fillna("").astype(str):
        labs = [l for l in n.split(".") if l] or [""]
        L = max(labs, key=len)
        m = max(len(L), 1)
        rows.append((ll(n), ll(L), len(L), pf.shannon(L), sum(c in "0123456789abcdefABCDEF" for c in L) / m,
                     sum(cls(a) != cls(b) for a, b in zip(L, L[1:])) / max(len(L) - 1, 1),
                     sum(c.isdigit() for c in L) / m))
    cols = ["ll_all", "ll_long", "long_len", "long_entropy", "hex_frac_long", "trans_long", "digit_frac_long"]
    return pd.DataFrame(rows, columns=cols, index=df.index)


# --------------------------------------------------------------------------- data
def _fingerprint(inputs, cfg):
    parts = [json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(cfg.items())}, default=str)]
    for i in inputs:
        p = Path(i)
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        parts += [f"{f.name}:{f.stat().st_size}" for f in files if f.is_file()]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def prepare(inputs, work_dir, cfg=None, verbose=True, use_cache=True, keep_synthetic=True):
    """Build the full dataset. Returns dict(ALL=DataFrame, manifest=DataFrame, real_audit=DataFrame, cfg=dict).
    keep_synthetic=False deletes the generated synthetic PCAPs once their features are extracted."""
    cfg = {**DEFAULTS, **(cfg or {})}
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    key = _fingerprint(inputs, cfg)
    cache = work / "prepared.pkl"
    if use_cache and cache.exists():
        D = pd.read_pickle(cache)
        if D.get("key") == key:
            if verbose:
                print(f"[cache] loaded prepared data from {cache}")
            return D
    t0 = time.time()

    # ---- real captures
    if verbose:
        print("Extracting real PCAPs ...")
    pf.run(inputs, out_dir=work / "real", window=cfg["window"], verbose=verbose)
    Q = pd.read_csv(work / "real" / "queries.csv", low_memory=False)
    real_audit = pd.read_csv(work / "real" / "label_audit.csv")
    Q = Q.sort_values(["scenario", "file", "ts"]).reset_index(drop=True)
    pos = Q.groupby(["scenario", "file"]).cumcount() / Q.groupby(["scenario", "file"])["ts"].transform("size")
    Q["role"] = np.where(pos < cfg["train_frac"], "real_train", "real_test")
    Q["origin"], Q["enc"], Q["rate"] = "real", "real", np.nan
    Q["family"] = np.where(Q.label == 1, "real_exfil", "real_benign")
    Q["stream_family"] = "real"

    # ---- synthetic captures
    ben = Q[(Q.label == 0) & (Q.name_complete == 1) & (Q.role == "real_train")].name.dropna()
    domains = sorted({".".join(n.split(".")[-2:]).lower() for n in ben
                      if len(n.split(".")) >= 2 and "arpa" not in n and len(n) < 40})
    syn_dir = work / "synthetic"
    shutil.rmtree(syn_dir, ignore_errors=True)
    if verbose:
        print(f"Generating synthetic traffic ({len(domains):,} benign base domains from the real capture) ...")
    man = pd.DataFrame(ev.generate(syn_dir, domains or None, seed=cfg["seed"], n_family=cfg["n_family"],
                                   n_hard=cfg["n_hard"], n_mixed=cfg["n_mixed"], mixed_rates=cfg["mixed_rates"],
                                   mixed_families=cfg["mixed_families"], snaplen=cfg["snaplen"],
                                   train_mixed_rates=cfg["train_mixed_rates"], n_train_mixed=cfg["n_train_mixed"]))
    pf.run([syn_dir], out_dir=work / "syn", window=cfg["window"], scenario="synthetic", verbose=False)
    S = pd.read_csv(work / "syn" / "queries.csv", low_memory=False)
    side = pd.concat([pd.read_csv(p).assign(file=Path(p).name.replace(".labels.csv", ".pcap"))
                      for p in glob.glob(str(syn_dir / "**" / "*.labels.csv"), recursive=True)])
    S = S.drop(columns=["label"]).merge(side[["file", "pkt_index", "label", "family"]], on=["file", "pkt_index"], how="left")
    if S.label.isna().any():
        raise RuntimeError("synthetic queries without a label")
    S["label"] = S.label.astype(int)
    S = S.merge(man[["file", "kind", "family", "rate"]].rename(columns={"family": "stream_family"}), on="file")
    S = S.sort_values(["file", "ts"]).reset_index(drop=True)
    spos = S.groupby("file").cumcount() / S.groupby("file")["ts"].transform("size")
    S["role"] = np.select([S.kind == "evasive", (S.kind == "hard_benign") & (spos < 0.5), S.kind == "hard_benign",
                           S.kind == "train_mixed"], ["evasive", "hard_train", "hard_test", "mixed_train"], "mixed")
    S["enc"] = S.stream_family.map(lambda f: ev.FAMILIES[f][0] if f in ev.FAMILIES else "none")
    S["origin"] = "synthetic"
    S = S.drop(columns=["kind"])

    # ---- features
    ALL = pd.concat([Q, S], ignore_index=True, sort=False)
    ALL["sld_code"] = pd.factorize(ALL["sld"].fillna(""))[0]
    lm = train_bigram_lm(ALL[(ALL.role == "real_train") & (ALL.label == 0)].name.fillna(""))
    ALL = pd.concat([ALL, char_features(ALL, lm)], axis=1)
    if verbose:
        print(f"Computing stateful features for {len(ALL):,} queries ...")
    ALL = sf.add_stateful(ALL, host_n=cfg["window"], dom_seconds=cfg["dom_seconds"], keys=KEYS)
    ALL = sf.add_behaviour(ALL, long_seconds=cfg["long_seconds"], keys=KEYS)
    D = {"ALL": ALL, "manifest": man, "real_audit": real_audit, "cfg": cfg, "key": key}
    pd.to_pickle(D, cache)
    if not keep_synthetic:
        shutil.rmtree(syn_dir, ignore_errors=True)
        shutil.rmtree(work / "syn", ignore_errors=True)
    if verbose:
        print(f"[OK] prepared {len(ALL):,} queries in {time.time() - t0:.0f}s")
        print(ALL.groupby(["origin", "role"]).label.agg(queries="size", exfil="sum").to_string())
    return D


# --------------------------------------------------------------------------- models
def make_model(name, cfg):
    seed = cfg["seed"]
    m = {"RF": RandomForestClassifier(n_estimators=cfg["n_estimators"], n_jobs=-1, random_state=seed),
         "GNB": GaussianNB(),
         "LR": LogisticRegression(max_iter=1000, random_state=seed),
         "MLP": MLPClassifier(max_iter=300, random_state=seed),
         "SVM": SVC(random_state=seed)}[name]
    return make_pipeline(StandardScaler(), m) if name in ("LR", "MLP", "SVM") else m


def fit(name, X, y, cfg):
    if name == "SVM" and len(X) > cfg["svm_max_train"]:
        i = np.random.default_rng(cfg["seed"]).choice(len(X), cfg["svm_max_train"], replace=False)
        X, y = X.iloc[i], y.iloc[i]
    return make_model(name, cfg).fit(X, y)


def _test(A):
    return A[A.role.isin(TEST_ROLES)]


def _single(D, features, model="RF"):
    A, cfg = D["ALL"], D["cfg"]
    tr = A[A.role == "real_train"]
    m = fit(model, tr[features], tr.label, cfg)
    t = _test(A)
    return pd.Series(m.predict(t[features]).astype(float), index=t.index), {"model": m, "features": features}


def paper_window_rule(D, pred):
    """Paper Sec. 5: flag a whole window of `window` queries when > delta of them are flagged."""
    A, cfg = D["ALL"], D["cfg"]
    t = A.loc[pred.index, KEYS + ["ts"]].assign(p=pred.values).sort_values(KEYS + ["ts"])
    t["win"] = t.groupby(KEYS).cumcount() // cfg["window"]
    flag = t.groupby(KEYS + ["win"])["p"].transform("mean") > cfg["paper_delta"]
    return flag.astype(float).reindex(pred.index)


def _loeo(D, train_and_predict):
    """Leave-one-encoding-out: fold e trains without encoding e; held-out rows take their fold's score,
    rows of no encoding take the mean over folds."""
    A = D["ALL"]
    t = _test(A)
    out = pd.Series(0.0, index=t.index)
    neutral = ~t.enc.isin(ENCODINGS)
    infos = {}
    for e in ENCODINGS:
        train_mask = A.role.isin(["real_train", "hard_train"]) | (A.role.isin(["evasive", "mixed_train"]) & (A.enc != e))
        p, info = train_and_predict(A, train_mask, t)
        own = t.enc == e
        out[own] = p[own]
        out[neutral] += p[neutral] / len(ENCODINGS)
        infos[e] = info
    return out, infos


def step3a(D):
    cfg, F = D["cfg"], FEATURES["step2"]

    def tp(A, mask, t):
        m = fit("RF", A.loc[mask, F], A.loc[mask, "label"], cfg)
        return pd.Series(m.predict(t[F]).astype(float), index=t.index), {"model": m, "features": F}
    return _loeo(D, tp)


def step3b(D):
    cfg = D["cfg"]
    F1, F2 = FEATURES["step3_layer1"], FEATURES["step3_layer2"]

    def tp(A, mask, t):
        l1 = make_model("RF", cfg)
        tr = A[mask]
        cv = StratifiedKFold(3, shuffle=True, random_state=cfg["seed"])
        oof = cross_val_predict(l1, tr[F1], tr.label, cv=cv, method="predict_proba")[:, 1]
        l1.fit(tr[F1], tr.label)
        score = pd.Series(l1.predict_proba(A[F1])[:, 1], index=A.index)
        score[mask] = oof                                           # out-of-fold for training rows
        A2 = sf.add_rolling_scores(A[KEYS + ["ts", "base"]].assign(l1_score=score.values), "l1_score",
                                   host_n=cfg["window"], dom_seconds=cfg["dom_seconds"], keys=KEYS)
        X = pd.concat([A[FEATURES["step2"]], A2[["l1_score"] + sf.SCORE_FEATURES]], axis=1)
        l2 = fit("RF", X.loc[mask, F2], A.loc[mask, "label"], cfg)
        return pd.Series(l2.predict(X.loc[t.index, F2]).astype(float), index=t.index), \
            {"model": l2, "features": F2, "layer1": l1}
    return _loeo(D, tp)


def step4_behaviour(D):
    cfg, F = D["cfg"], FEATURES["step4_behaviour"]
    use_trunc = cfg.get("behaviour_on_trunc", False)

    def tp(A, mask, t):
        if not use_trunc:                                        # train only where the domain is known
            mask = mask & (A.base != "<trunc>")
        m = fit("RF", A.loc[mask, F], A.loc[mask, "label"], cfg)
        p = pd.Series(m.predict(t[F]).astype(float), index=t.index)
        if not use_trunc:
            p[(t.base == "<trunc>").values] = 0.0                # abstain: leave these to step 3a
        return p, {"model": m, "features": F}
    return _loeo(D, tp)


DETECTORS = {
    "baseline": lambda D, model="RF": _single(D, FEATURES["baseline"], model),
    "step1": lambda D: _single(D, FEATURES["step1"]),
    "step2": lambda D: _single(D, FEATURES["step2"]),
    "step3a": step3a,
    "step3b": step3b,
    "step4 behaviour": step4_behaviour,
}
DESCRIPTION = {
    "baseline": "paper's 14 stateless features, real data only",
    "baseline+window": "baseline RF + paper window rule (>40% of 100 queries)",
    "step1": "+ host temporal features",
    "step2": "+ per-domain statistical features",
    "step3a": "step2 + augmented training (leave-one-encoding-out)",
    "step3b": "two-layer: char model -> score aggregates + step2, augmented",
    "step4 behaviour": "behaviour-only model (no character features), augmented; abstains on cut-off names",
    "step4": "step3a OR step4 behaviour",
}


def run_detectors(D, steps, baseline_models=("RF",), verbose=True):
    """steps: subset of ['baseline', 'step1', 'step2', 'step3a', 'step3b', 'step4'] (baseline always adds the
    paper window rule; step4 also reports its behaviour-only model). Returns (preds, infos)."""
    preds, infos = {}, {}
    for s in steps:
        t0 = time.time()
        if s == "baseline":
            for m in baseline_models:
                name = "baseline" if m == "RF" else f"baseline ({m})"
                preds[name], infos[name] = DETECTORS["baseline"](D, m)
            if "RF" in baseline_models:
                preds["baseline+window"] = paper_window_rule(D, preds["baseline"])
                infos["baseline+window"] = infos["baseline"]
        elif s == "step4":
            for part in ("step3a", "step4 behaviour"):
                if part not in preds:
                    preds[part], infos[part] = DETECTORS[part](D)
            preds["step4"] = np.maximum(preds["step3a"], preds["step4 behaviour"])
            infos["step4"] = infos["step4 behaviour"]
        else:
            preds[s], infos[s] = DETECTORS[s](D)
        if verbose:
            print(f"  {s:10s} done in {time.time() - t0:5.0f}s")
    return preds, infos


# --------------------------------------------------------------------------- metrics
def _rates(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    tp, fp = (p * y).sum(), (p * (1 - y)).sum()
    fn, tn = ((1 - p) * y).sum(), ((1 - p) * (1 - y)).sum()
    rec = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and not (math.isnan(prec) or math.isnan(rec)) else 0.0
    return rec, fpr, f1


def evaluate(D, preds):
    """-> (summary: one row per detector, detail: recall per family / dilution rate)."""
    A, cfg = D["ALL"], D["cfg"]
    rows, detail = [], []
    for name, p in preds.items():
        t = A.loc[p.index].assign(p=p.values)
        r = {"detector": name, "what": DESCRIPTION.get(name.split(" (")[0], name)}
        rt = t[t.role == "real_test"]
        rec, fpr, f1 = _rates(rt.label, rt.p)
        r.update(real_f1=100 * f1, real_recall=100 * rec, real_fpr=100 * fpr)
        evs = t[t.role == "evasive"]
        fam_rec = evs.groupby("stream_family").p.mean() * 100
        for f, v in fam_rec.items():
            detail.append(dict(detector=name, test="evasive (pure stream)", family=f, encoding=ev.FAMILIES[f][0], value=v))
        r["evasive_recall"] = fam_rec.mean()
        for e in ENCODINGS:
            fs = [f for f in fam_rec.index if ev.FAMILIES[f][0] == e]
            r[f"evasive_{e}"] = fam_rec[fs].mean() if fs else np.nan
        r["hard_fpr"] = 100 * t[t.role == "hard_test"].p.mean()
        mx = t[t.role == "mixed"]
        for rate in cfg["mixed_rates"]:
            g = mx[(mx.rate == rate) & (mx.label == 1)].groupby("stream_family").p.mean() * 100
            r[f"mixed_recall_{int(round(rate * 100))}%"] = g.mean()
            for f, v in g.items():
                detail.append(dict(detector=name, test=f"mixed {int(round(rate * 100))}%", family=f,
                                   encoding=ev.FAMILIES[f][0], value=v))
        r["mixed_fpr"] = 100 * mx[mx.label == 0].p.mean()
        rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(detail)


KEY_COLUMNS = ["detector", "real_f1", "real_fpr", "evasive_recall", "evasive_base32", "evasive_hex", "evasive_base64",
               "evasive_words", "hard_fpr", "mixed_recall_1%", "mixed_recall_5%", "mixed_recall_20%",
               "mixed_recall_50%", "mixed_fpr"]


def key_table(summary):
    return summary[[c for c in KEY_COLUMNS if c in summary.columns]].set_index("detector").round(2)


def delta_vs(summary, ref="baseline"):
    k = key_table(summary)
    return (k - k.loc[ref]).drop(index=ref).round(2) if ref in k.index else k


def feature_importance(info, top=15):
    if isinstance(info, dict) and "model" not in info:           # leave-one-encoding-out: average folds
        imps = [pd.Series(i["model"].feature_importances_, index=i["features"]) for i in info.values()]
        return pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False).head(top).round(4)
    m = info["model"]
    m = m[-1] if hasattr(m, "steps") else m
    if not hasattr(m, "feature_importances_"):
        return pd.Series(dtype=float)
    return pd.Series(m.feature_importances_, index=info["features"]).sort_values(ascending=False).head(top).round(4)


def run_seeds(inputs, work_dir, seeds=(1, 2, 3, 4, 5), steps=("baseline", "step1", "step2", "step3a", "step3b", "step4"),
              cfg=None, verbose=True):
    """Repeat prepare -> run_detectors -> evaluate for several seeds. Each seed draws new synthetic traffic
    (payloads, attacker domains, background, timing) and new model randomness; the real split is fixed.
    Returns (summaries, details) with a `seed` column."""
    sums, dets = [], []
    for s in seeds:
        t0 = time.time()
        D = prepare(inputs, Path(work_dir) / f"seed_{s}", {**(cfg or {}), "seed": s}, verbose=False,
                    keep_synthetic=False)
        preds, _ = run_detectors(D, list(steps), verbose=False)
        summ, det = evaluate(D, preds)
        sums.append(summ.assign(seed=s))
        dets.append(det.assign(seed=s))
        if verbose:
            k = key_table(summ)
            print(f"seed {s}: {len(D['ALL']):,} queries, {time.time() - t0:.0f}s | diluted recall 1/5/20/50% "
                  + " | ".join(f"{d}: " + "/".join(f"{k.loc[d, c]:.0f}" for c in KEY_COLUMNS[9:13]) for d in k.index))
        del D
    return pd.concat(sums, ignore_index=True), pd.concat(dets, ignore_index=True)


def aggregate(summaries):
    """-> (mean, std, display table 'mean ± std') per detector over seeds, detectors in run order."""
    cols = [c for c in KEY_COLUMNS[1:] if c in summaries.columns]
    order = list(dict.fromkeys(summaries.detector))
    g = summaries.groupby("detector")[cols]
    mean, std = g.mean().reindex(order), g.std(ddof=1).reindex(order).fillna(0)
    table = mean.round(1).astype(str) + " ± " + std.round(1).astype(str)
    return mean, std, table


def save(out_dir, summary, detail, D, extra=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    detail.to_csv(out / "detail.csv", index=False)
    D["manifest"].to_csv(out / "synthetic_manifest.csv", index=False)
    D["real_audit"].to_csv(out / "real_label_audit.csv", index=False)
    cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in D["cfg"].items()}
    (out / "run_config.json").write_text(json.dumps({"config": cfg, "features": FEATURES, **(extra or {})},
                                                    indent=2, default=str))
    return out
