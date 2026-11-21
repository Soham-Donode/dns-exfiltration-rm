"""
Stateful (temporal + per-domain statistical) features for DNS queries
=====================================================================
Every feature is CAUSAL: it is computed from the current query and the traffic that came
BEFORE it from the same host (and to the same base domain), exactly what a streaming detector
could compute at that moment. Nothing looks into the future, so a time-ordered train/test split
stays clean.

Step 1 - host temporal features, over the host's last `host_n` queries (paper: tau = 100):
    h_n                 queries in the window
    h_rate              queries per second
    h_gap_mean, h_gap_cv  mean inter-arrival time and its coefficient of variation (regularity)
    h_unique_name_ratio   distinct names / queries
    h_unique_base_ratio   distinct base domains / queries
    h_len_mean, h_len_std true DNS message length
    h_entropy_mean        mean name entropy

Step 2 - per-domain statistical features, over the last `dom_seconds` of traffic from the same
host to the same base domain (the domain is "<trunc>" when the capture cut the name, so long
truncated queries of one host fall back to one shared bucket):
    d_n, d_unique_sub, d_unique_ratio   volume and subdomain diversity
    d_sub_len_mean, d_sub_len_std       subdomain length
    d_sub_entropy_mean                  subdomain entropy
    d_gap_mean                          mean inter-arrival time to this domain
    d_share                             share of the host's queries (same time span) going here
    d_bytes                             total DNS bytes sent to this domain

Step 3 helper - add_rolling_scores(): host / domain rolling mean (and domain max) of a per-query
score, used by the two-layer model.

Step 4 - add_behaviour(): content-agnostic behaviour of a host -> domain pair over a long horizon
(`long_seconds`, default 1 h) and over the whole history. No character statistics at all, so a model
on these features cannot take the "looks random" shortcut:
    dl_n, dl_unique_sub, dl_unique_ratio   volume and subdomain diversity over the long horizon
    dl_bytes_rate                          DNS bytes per second sent to the domain (long horizon)
    dl_gap_cv                              regularity of queries to the domain (long horizon)
    d_new_frac                             share of recent queries that used a never-seen subdomain
    d_cum_unique                           log(1 + distinct subdomains ever sent to the domain)
"""

import math
from collections import Counter, deque

import numpy as np
import pandas as pd

import pcap_features as pf

HOST_FEATURES = ["h_n", "h_rate", "h_gap_mean", "h_gap_cv", "h_unique_name_ratio", "h_unique_base_ratio",
                 "h_len_mean", "h_len_std", "h_entropy_mean"]
DOMAIN_FEATURES = ["d_n", "d_unique_sub", "d_unique_ratio", "d_sub_len_mean", "d_sub_len_std",
                   "d_sub_entropy_mean", "d_gap_mean", "d_share", "d_bytes"]
SCORE_FEATURES = ["h_score_mean", "d_score_mean", "d_score_max"]
BEHAVIOUR_FEATURES = ["dl_n", "dl_unique_sub", "dl_unique_ratio", "dl_bytes_rate", "dl_gap_cv",
                      "d_new_frac", "d_cum_unique"]
KEYS = ("scenario", "file", "src")


def base_and_sub(name, complete):
    """-> (base domain, subdomain), both lower-case (DNS is case-insensitive; 0x20 randomisation)."""
    name = (name or "").lower()
    if not complete:
        return "<trunc>", name
    labels = name.split(".") if name else []
    _sld, sub, _has = pf.split_name(labels, True)
    return (name[len(sub) + 1:] if sub else name), sub


def _order(df, keys):
    g = df.groupby(list(keys), sort=False, dropna=False).ngroup().to_numpy()
    return g, np.lexsort((df["ts"].to_numpy(float), g))


def _std(s, s2, n):
    return math.sqrt(max(s2 / n - (s / n) ** 2, 0.0)) if n else 0.0


def add_stateful(df, host_n=100, dom_seconds=600.0, keys=KEYS):
    """Return a copy of df with `base`, `sub` and the HOST_FEATURES + DOMAIN_FEATURES columns.
    Needs columns: keys, ts, dns_len, entropy, name, name_complete."""
    df = df.copy()
    bs = [base_and_sub(n, c) for n, c in zip(df["name"].fillna("").astype(str), df["name_complete"].astype(int))]
    df["base"] = [b for b, _ in bs]
    df["sub"] = [s for _, s in bs]
    n_rows = len(df)
    out = {c: np.zeros(n_rows) for c in HOST_FEATURES + DOMAIN_FEATURES}
    ts, ln, ent = (df["ts"].to_numpy(float), df["dns_len"].to_numpy(float), df["entropy"].to_numpy(float))
    names = [s.lower() for s in df["name"].fillna("").astype(str)]
    base, sub = df["base"].to_numpy(), df["sub"].to_numpy()
    sub_len = np.array([len(s) for s in sub], dtype=float)
    sub_ent = np.array([pf.shannon(s) for s in sub], dtype=float)
    grp, order = _order(df, keys)

    cur = None
    for i in order:
        if grp[i] != cur:                                    # new host stream: reset all state
            cur = grp[i]
            hq = deque()                                     # (ts, len, ent, name, base)
            h = {"len": 0.0, "len2": 0.0, "ent": 0.0, "gap": 0.0, "gap2": 0.0}
            hgaps, hnames, hbases = deque(), Counter(), Counter()
            htime = deque()                                  # host timestamps within dom_seconds
            doms = {}
        t = ts[i]
        # ---------------- host window (last host_n queries)
        if hq:
            g = t - hq[-1][0]
            hgaps.append(g); h["gap"] += g; h["gap2"] += g * g
        hq.append((t, ln[i], ent[i], names[i], base[i]))
        h["len"] += ln[i]; h["len2"] += ln[i] ** 2; h["ent"] += ent[i]
        hnames[names[i]] += 1; hbases[base[i]] += 1
        if len(hq) > host_n:
            _t, l0, e0, n0, b0 = hq.popleft()
            h["len"] -= l0; h["len2"] -= l0 ** 2; h["ent"] -= e0
            hnames[n0] -= 1
            if not hnames[n0]: del hnames[n0]
            hbases[b0] -= 1
            if not hbases[b0]: del hbases[b0]
            g0 = hgaps.popleft(); h["gap"] -= g0; h["gap2"] -= g0 * g0
        n = len(hq)
        span = hq[-1][0] - hq[0][0]
        ng = len(hgaps)
        gmean = h["gap"] / ng if ng else dom_seconds
        out["h_n"][i] = n
        out["h_rate"][i] = (n - 1) / span if span > 0 else 0.0
        out["h_gap_mean"][i] = gmean
        out["h_gap_cv"][i] = _std(h["gap"], h["gap2"], ng) / gmean if ng and gmean > 0 else 0.0
        out["h_unique_name_ratio"][i] = len(hnames) / n
        out["h_unique_base_ratio"][i] = len(hbases) / n
        out["h_len_mean"][i] = h["len"] / n
        out["h_len_std"][i] = _std(h["len"], h["len2"], n)
        out["h_entropy_mean"][i] = h["ent"] / n
        # ---------------- host time window (for d_share)
        htime.append(t)
        while htime[0] < t - dom_seconds:
            htime.popleft()
        # ---------------- domain window (last dom_seconds to the same base)
        d = doms.get(base[i])
        if d is None:
            d = doms[base[i]] = {"q": deque(), "subs": Counter(), "sl": 0.0, "sl2": 0.0, "se": 0.0, "by": 0.0}
        q = d["q"]
        while q and q[0][0] < t - dom_seconds:
            _t, s0, l0, e0, b0 = q.popleft()
            d["subs"][s0] -= 1
            if not d["subs"][s0]: del d["subs"][s0]
            d["sl"] -= l0; d["sl2"] -= l0 ** 2; d["se"] -= e0; d["by"] -= b0
        q.append((t, sub[i], sub_len[i], sub_ent[i], ln[i]))
        d["subs"][sub[i]] += 1
        d["sl"] += sub_len[i]; d["sl2"] += sub_len[i] ** 2; d["se"] += sub_ent[i]; d["by"] += ln[i]
        dn = len(q)
        out["d_n"][i] = dn
        out["d_unique_sub"][i] = len(d["subs"])
        out["d_unique_ratio"][i] = len(d["subs"]) / dn
        out["d_sub_len_mean"][i] = d["sl"] / dn
        out["d_sub_len_std"][i] = _std(d["sl"], d["sl2"], dn)
        out["d_sub_entropy_mean"][i] = d["se"] / dn
        out["d_gap_mean"][i] = (t - q[0][0]) / (dn - 1) if dn > 1 else dom_seconds
        out["d_share"][i] = dn / len(htime)
        out["d_bytes"][i] = d["by"]
    for c, v in out.items():
        df[c] = v
    return df


def add_rolling_scores(df, score_col, host_n=100, dom_seconds=600.0, keys=KEYS):
    """Causal host / domain aggregates of a per-query score (two-layer model). Needs `base`."""
    s = df[score_col].to_numpy(float)
    ts, base = df["ts"].to_numpy(float), df["base"].to_numpy()
    res = {c: np.zeros(len(df)) for c in SCORE_FEATURES}
    grp, order = _order(df, keys)
    cur = None
    for i in order:
        if grp[i] != cur:
            cur, hq, hsum, doms = grp[i], deque(), 0.0, {}
        hq.append(s[i]); hsum += s[i]
        if len(hq) > host_n:
            hsum -= hq.popleft()
        res["h_score_mean"][i] = hsum / len(hq)
        d = doms.setdefault(base[i], deque())
        while d and d[0][0] < ts[i] - dom_seconds:
            d.popleft()
        d.append((ts[i], s[i]))
        vals = [v for _, v in d]
        res["d_score_mean"][i] = sum(vals) / len(vals)
        res["d_score_max"][i] = max(vals)
    out = df.copy()
    for c, v in res.items():
        out[c] = v
    return out


def add_behaviour(df, long_seconds=3600.0, keys=KEYS):
    """Causal long-horizon behaviour per host -> base domain (needs `base` and `sub` from add_stateful)."""
    ts, ln = df["ts"].to_numpy(float), df["dns_len"].to_numpy(float)
    base, sub = df["base"].to_numpy(), df["sub"].to_numpy()
    res = {c: np.zeros(len(df)) for c in BEHAVIOUR_FEATURES}
    grp, order = _order(df, keys)
    cur = None
    for i in order:
        if grp[i] != cur:
            cur, doms = grp[i], {}
        t = ts[i]
        d = doms.get(base[i])
        if d is None:
            d = doms[base[i]] = {"q": deque(), "subs": Counter(), "by": 0.0, "g": 0.0, "g2": 0.0, "ng": 0,
                                 "new": 0, "seen": set(), "last": None}
        q = d["q"]
        while q and q[0][0] < t - long_seconds:
            _t, s0, l0, g0, n0 = q.popleft()
            d["subs"][s0] -= 1
            if not d["subs"][s0]:
                del d["subs"][s0]
            d["by"] -= l0
            d["new"] -= n0
            if g0 is not None:
                d["g"] -= g0; d["g2"] -= g0 * g0; d["ng"] -= 1
        gap = None if d["last"] is None else t - d["last"]
        d["last"] = t
        is_new = int(sub[i] not in d["seen"])
        d["seen"].add(sub[i])
        q.append((t, sub[i], ln[i], gap, is_new))
        d["subs"][sub[i]] += 1
        d["by"] += ln[i]
        d["new"] += is_new
        if gap is not None:
            d["g"] += gap; d["g2"] += gap * gap; d["ng"] += 1
        n = len(q)
        res["dl_n"][i] = n
        res["dl_unique_sub"][i] = len(d["subs"])
        res["dl_unique_ratio"][i] = len(d["subs"]) / n
        res["dl_bytes_rate"][i] = d["by"] / long_seconds
        gm = d["g"] / d["ng"] if d["ng"] else 0.0
        res["dl_gap_cv"][i] = _std(d["g"], d["g2"], d["ng"]) / gm if d["ng"] and gm > 0 else 0.0
        res["d_new_frac"][i] = d["new"] / n
        res["d_cum_unique"][i] = math.log1p(len(d["seen"]))
    out = df.copy()
    for c, v in res.items():
        out[c] = v
    return out


def demo():
    """Tiny worked example (also used by the tests)."""
    rows = []
    for k in range(5):                                       # one host: 5 lookups of one site, 5 exfil chunks
        rows.append(dict(scenario="x", file="f", src="h", ts=float(k), dns_len=40, entropy=2.5,
                         name="www.example.com", name_complete=1))
        rows.append(dict(scenario="x", file="f", src="h", ts=k + 0.5, dns_len=80, entropy=4.0,
                         name=f"a{k}x9q2z7k.evil.net", name_complete=1))
    return add_stateful(pd.DataFrame(rows), host_n=4, dom_seconds=10)
