"""Глобальная модель градиентного бустинга: одна модель на все МО и горизонты.

Цель — поправка к сезонному наивному прогнозу с приростом:
    target = log y_{T+h} − log ŷ^{snaive_growth}_{T+h|T}
Обучающие примеры строятся по всем прошлым точкам прогноза o < T, для которых y_{o+h}
уже наблюдён к моменту T. Признаки используют только данные до o включительно.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..events import event_features
from .baselines import snaive_growth


def _features(wide_to_o: pd.DataFrame, h: int, reg: pd.DataFrame | None) -> pd.DataFrame:
    lg = np.log(wide_to_o)
    g = lg - lg.shift(12, axis=1)
    o = wide_to_o.columns[-1]
    target_month = o + pd.DateOffset(months=h)
    g_last = g.iloc[:, -1]
    med = g_last.median()
    f = pd.DataFrame(index=wide_to_o.index)
    f["h"] = h
    f["target_month"] = target_month.month
    f["g0"] = g_last
    f["g1"] = g.iloc[:, -2]
    f["g2"] = g.iloc[:, -3]
    f["g_mean3"] = g.iloc[:, -3:].mean(axis=1)
    f["g_std3"] = g.iloc[:, -3:].std(axis=1)
    f["g_peer_dev"] = g_last - med
    f["g_peer_med"] = med
    f["mom"] = lg.iloc[:, -1] - lg.iloc[:, -2]
    f["seas_step"] = lg.iloc[:, -12 + h - 1] - lg.iloc[:, -13]  # сезонный шаг прошлого года от T-12 к T+h-12
    f["log_level"] = lg.iloc[:, -1]
    f["rank_level"] = f["log_level"].rank(pct=True)
    f["same_name"] = wide_to_o.index.str.split("|").str[2].astype(int) > 0
    if reg is not None and len(reg):
        f = f.join(event_features(reg, wide_to_o.index, o, target_month))
    return f


def fit_predict(wide: pd.DataFrame, horizons, reg: pd.DataFrame | None = None, min_growth_obs: int = 3,
                base_window: int = 3, params: dict | None = None, seed: int = 0, **_):
    import lightgbm as lgb

    params = {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 40,
              "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0,
              "random_state": seed, "verbose": -1, **(params or {})}
    cols = wide.columns
    T = len(cols) - 1
    X, y = [], []
    first_o = 12 + max(min_growth_obs, base_window) - 1  # нужно минимум min_growth_obs значений прироста г/г
    for h in horizons:
        for oi in range(first_o, T - h + 1):
            w = wide.iloc[:, : oi + 1]
            base = snaive_growth(w, [h], window=base_window)[h]
            actual = wide.iloc[:, oi + h]
            fx = _features(w, h, reg)
            X.append(fx)
            y.append(np.log(actual) - np.log(base))
    if not X:
        raise ValueError("Недостаточно истории для обучения глобальной модели")
    X = pd.concat(X)
    y = pd.concat(y)
    ok = y.notna() & np.isfinite(y)
    model = lgb.LGBMRegressor(objective="l1", **params)
    model.fit(X[ok], y[ok])
    out = {}
    for h in horizons:
        fx = _features(wide, h, reg)[X.columns]
        base = snaive_growth(wide, [h], window=base_window)[h]
        out[h] = base * np.exp(model.predict(fx))
    res = pd.DataFrame(out, index=wide.index)
    res.attrs["feature_importance"] = dict(zip(X.columns, model.feature_importances_.tolist()))
    return res
