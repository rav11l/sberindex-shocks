"""Полусинтетическая проверка детекторов и оценка на реальных событиях.

Истинных дат шоков по МО нет. Поэтому основная проверка — инъекция: в случайную половину
рядов с полной историей в случайный месяц τ окна оценки вносится ступенчатый сдвиг уровня
y_t ← y_t·(1+δ) для t ≥ τ. Вторая половина остаётся нетронутой и даёт оценку ложных тревог.
Ограничение метода: в «чистых» рядах есть собственные настоящие шоки, и тревоги по ним
засчитываются как ложные — оценка точности консервативна.

Новости моделируются как зашумлённый сигнал: для доли recall_news инъекций «новость»
выходит за lead месяцев до τ; ещё false_news_rate рядов получают новость без шока.
Это позволяет измерить, сколько даёт априорная информация, не выдавая симуляцию за факт.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .detectors import REGISTRY, transform


def inject(wide: pd.DataFrame, eval_cols, deltas, share: float = 0.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    w = wide.copy()
    ids = w.index.to_numpy()
    shocked = rng.choice(ids, int(len(ids) * share), replace=False)
    truth = []
    for sid in shocked:
        tau = eval_cols[rng.integers(0, len(eval_cols))]
        delta = float(rng.choice(deltas))
        w.loc[sid, w.columns >= tau] *= (1 + delta)
        truth.append({"series_id": sid, "tau": tau, "delta": delta})
    return w, pd.DataFrame(truth)


def simulate_news(truth: pd.DataFrame, index: pd.Index, columns, eval_cols, recall_news: float = 0.6,
                  false_news_rate: float = 0.05, lead: int = 1, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    E = pd.DataFrame(0, index=index, columns=columns)
    for _, r in truth.iterrows():
        if rng.random() < recall_news:
            # новость известна заранее; в детектор она входит на месяц ожидаемого вступления
            E.loc[r["series_id"], r["tau"]] = 1
    clean = index.difference(truth["series_id"])
    for sid in rng.choice(clean, int(len(clean) * false_news_rate), replace=False):
        E.loc[sid, eval_cols[rng.integers(0, len(eval_cols))]] = 1
    return E


def score(alarms: pd.DataFrame, truth: pd.DataFrame, max_delay: int = 2) -> dict:
    eval_cols = list(alarms.columns)
    pos = {c: i for i, c in enumerate(eval_cols)}
    t_by = truth.set_index("series_id")
    tp = fn = fp = 0
    delays = []
    for sid, row in alarms.iterrows():
        hits = [pos[c] for c, v in row.items() if v]
        if sid in t_by.index:
            ti = pos[t_by.at[sid, "tau"]]
            ok = [a for a in hits if ti <= a <= ti + max_delay]
            # тревоги до шока — ложные; тревоги после окна max_delay не штрафуются: прирост г/г
            # несёт эхо ступеньки ещё 12 месяцев, и повторная тревога по тому же шоку не ошибка
            fp += sum(1 for a in hits if a < ti)
            if ok:
                tp += 1
                delays.append(ok[0] - ti)
            else:
                fn += 1
        else:
            fp += len(hits)
    n_clean_months = (~alarms.index.isin(truth["series_id"])).sum() * len(eval_cols)
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    return {
        "TP": tp, "FP": fp, "FN": fn,
        "precision": prec, "recall": rec,
        "F1": 2 * prec * rec / (prec + rec) if prec and rec else 0.0,
        "mean_delay": float(np.mean(delays)) if delays else np.nan,
        "share_delay0": float(np.mean(np.array(delays) == 0)) if delays else np.nan,
        "false_alarms_per_100_clean_months":
            100 * alarms.loc[~alarms.index.isin(truth["series_id"])].to_numpy().sum() / max(n_clean_months, 1),
    }


def run(wide: pd.DataFrame, cfg: dict):
    c = cfg["changepoint"]
    eval_cols = [pd.Timestamp(x) for x in pd.date_range(c["eval_start"], c["eval_end"], freq="MS")]
    full = wide[wide.notna().all(axis=1)]
    if c.get("max_series"):
        full = full.sample(n=min(c["max_series"], len(full)), random_state=c.get("seed", 0))
    rows, by_delta = [], []
    for rep in range(c.get("repeats", 1)):
        seed = c.get("seed", 0) + rep
        w_inj, truth = inject(full, eval_cols, c["deltas"], c.get("share_shocked", 0.5), seed)
        d = transform(w_inj)
        news = simulate_news(truth, d.index, d.columns, eval_cols, **c.get("news_simulation", {}), seed=seed)
        for name, spec in c["detectors"].items():
            if not spec.get("enabled", True):
                continue
            params = {k: v for k, v in spec.items() if k not in {"enabled", "fn", "use_news"}}
            fn = REGISTRY[spec.get("fn", name)]
            variants = [("", None)]
            if spec.get("use_news"):
                variants.append(("+news", news))
            for suffix, eh in variants:
                kw = dict(params)
                if eh is not None:
                    kw["event_hazard"] = eh
                A = fn(d, eval_cols, **kw)
                rows.append({"detector": name + suffix, "repeat": rep, **score(A, truth, c.get("max_delay", 2))})
                for delta, tsub in truth.groupby("delta"):
                    keep = A.index.difference(truth["series_id"]).union(pd.Index(tsub["series_id"]))
                    s = score(A.loc[keep], tsub, c.get("max_delay", 2))
                    by_delta.append({"detector": name + suffix, "repeat": rep, "delta": delta,
                                     "recall": s["recall"], "mean_delay": s["mean_delay"]})
    res = pd.DataFrame(rows).groupby("detector").mean(numeric_only=True).drop(columns="repeat")
    bd = pd.DataFrame(by_delta).groupby(["detector", "delta"]).mean(numeric_only=True).drop(columns="repeat")
    return res.sort_values("F1", ascending=False).reset_index(), bd.reset_index()


def real_events(wide: pd.DataFrame, truth: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Детекторы на реальных рядах против размеченных локальных событий: месяц первой тревоги."""
    c = cfg["changepoint"]
    eval_cols = [pd.Timestamp(x) for x in pd.date_range(c["eval_start"], c["eval_end"], freq="MS")]
    if truth.empty:
        return pd.DataFrame()
    sub = wide.loc[wide.index.intersection(truth["series_id"].unique())]
    d_all = transform(wide)
    d = d_all.loc[sub.index]
    out = []
    for name, spec in c["detectors"].items():
        if not spec.get("enabled", True):
            continue
        params = {k: v for k, v in spec.items() if k not in {"enabled", "fn", "use_news"}}
        fn = REGISTRY[spec.get("fn", name)]
        A = fn(d_all, eval_cols, **params).loc[d.index] if spec.get("fn", name) == "cross_section" else fn(d, eval_cols, **params)
        for _, t in truth.iterrows():
            if t["series_id"] not in A.index:
                continue
            row = A.loc[t["series_id"]]
            after = [c_ for c_, v in row.items() if v and c_ >= t["tau"]]
            before = [c_ for c_, v in row.items() if v and c_ < t["tau"]]
            out.append({"detector": name, "series_id": t["series_id"], "event_id": t["event_id"],
                        "tau": t["tau"].date(), "first_alarm_after": after[0].date() if after else None,
                        "delay_months": (after[0].to_period("M") - t["tau"].to_period("M")).n if after else None,
                        "alarms_before": len(before)})
    return pd.DataFrame(out)
