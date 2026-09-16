"""Скользящий бэктест прогнозов и метрики.

Точки прогноза (origins) задаются в конфиге. Для каждой точки T модель видит столбцы до T
включительно и прогнозирует T+h. Метрики считаются по парам (ряд, T, h), где факт известен.

MAE      — в рублях, основная метрика конкурса.
R2_level — 1 − SSE/SST по уровням. Почти всегда близок к 1, потому что межрядовый разброс
           уровней огромен; приводится для полноты, но выводы на нём не строятся.
R2_growth— тот же R² на прогнозе прироста г/г: log ŷ_{T+h} − log y_{T+h−12} против факта.
           Именно он показывает, объясняет ли модель что-то сверх уровня МО.
rel_MAE  — MAE модели / MAE snaive на тех же парах.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .models import baselines, foundation, global_gbm
from .models.foundation import Unavailable

log = logging.getLogger(__name__)

MODELS = {**baselines.REGISTRY, **foundation.REGISTRY, "lgbm": global_gbm.fit_predict}


def run_model(name: str, spec: dict, wide: pd.DataFrame, origin: pd.Timestamp, horizons, reg,
              panel: pd.DataFrame | None = None):
    fn = MODELS[spec.get("fn", name)]
    params = {k: v for k, v in spec.items() if k not in {"fn", "enabled"}}
    if spec.get("fn", name) == "lgbm":
        params["reg"] = reg if spec.get("use_events", True) else None
    if panel is not None:
        # общий фактор считается по всей панели, даже если прогноз строится для выборки МО
        params["panel"] = panel.loc[:, :origin]
    train = wide.loc[:, :origin]
    return fn(train, horizons, **params)


def backtest(wide: pd.DataFrame, cfg: dict, reg: pd.DataFrame | None = None, series: pd.Index | None = None):
    horizons = cfg["forecast"]["horizons"]
    origins = [pd.Timestamp(o) for o in cfg["forecast"]["origins"]]
    panel = wide
    if series is not None:
        wide = wide.loc[series]
    rows, status = [], []
    for name, spec in cfg["forecast"]["models"].items():
        if not spec.get("enabled", True):
            continue
        for o in origins:
            t0 = time.time()
            try:
                fc = run_model(name, spec, wide, o, horizons, reg, panel)
            except Unavailable as e:
                log.warning("%s пропущена: %s", name, e)
                status.append({"model": name, "origin": o, "status": "skipped", "note": str(e)})
                break
            dt = time.time() - t0
            status.append({"model": name, "origin": o, "status": "ok", "seconds": round(dt, 1)})
            for h in horizons:
                tgt = o + pd.DateOffset(months=h)
                if tgt not in wide.columns:
                    continue
                prev_year = tgt - pd.DateOffset(months=12)
                rows.append(pd.DataFrame({
                    "model": name, "origin": o, "h": h, "series_id": fc.index,
                    "yhat": fc[h].to_numpy(), "y": wide[tgt].reindex(fc.index).to_numpy(),
                    "y_prev_year": wide[prev_year].reindex(fc.index).to_numpy(),
                }))
            log.info("%s @ %s: %.1f c", name, o.date(), dt)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(status)


def _r2(y, yhat):
    sst = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - yhat) ** 2).sum() / sst if sst > 0 else np.nan


def metrics(pred: pd.DataFrame, by=("model",)) -> pd.DataFrame:
    p = pred.dropna(subset=["y", "yhat", "y_prev_year"]).copy()
    # сравниваем модели только на общих парах
    common = p.groupby(["origin", "h", "series_id"])["model"].transform("nunique") == p["model"].nunique()
    p = p[common]
    p["ae"] = (p["y"] - p["yhat"]).abs()
    p["g"] = np.log(p["y"] / p["y_prev_year"])
    p["ghat"] = np.log(p["yhat"] / p["y_prev_year"])
    key = ["origin", "h", "series_id"]
    base = p[p["model"] == "snaive"].set_index(key)["ae"]
    rows = []
    for gkey, g in p.groupby(list(by)):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        b = base.reindex(pd.MultiIndex.from_frame(g[key])).sum() if len(base) else np.nan
        rows.append({**dict(zip(by, gkey)),
                     "n": len(g),
                     "MAE": g["ae"].mean(),
                     "MedAE": g["ae"].median(),
                     "R2_level": _r2(g["y"], g["yhat"]),
                     "R2_growth": _r2(g["g"], g["ghat"]),
                     "rel_MAE_vs_snaive": g["ae"].sum() / b if b else np.nan})
    return pd.DataFrame(rows)


def dm_test(pred: pd.DataFrame, a: str, b: str) -> dict:
    """Тест Диболда–Мариано на разности абсолютных ошибок, усреднённых по рядам в каждой (origin, h).

    Усреднение по рядам снимает межрядовую корреляцию ошибок в одном месяце: единица наблюдения —
    пара (точка прогноза, горизонт). При малом числе пар тест слабый, это оговаривается в отчёте.
    """
    from scipy import stats

    p = pred.dropna(subset=["y", "yhat"])
    p = p.assign(ae=(p["y"] - p["yhat"]).abs())
    wa = p[p.model == a].groupby(["origin", "h"])["ae"].mean()
    wb = p[p.model == b].groupby(["origin", "h"])["ae"].mean()
    d = (wa - wb).dropna()
    if len(d) < 3:
        return {"a": a, "b": b, "n": len(d), "mean_diff": d.mean(), "t": np.nan, "p": np.nan}
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return {"a": a, "b": b, "n": len(d), "mean_diff": d.mean(), "t": t, "p": 2 * stats.t.sf(abs(t), len(d) - 1)}
