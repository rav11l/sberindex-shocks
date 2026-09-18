"""Разборы поверх готовых прогнозов: ансамбли и разрез ошибок по размеру МО.

Ансамбль строится на таблице прогнозов (outputs/forecast_predictions.parquet), а не заново,
поэтому в него можно включать модели из разных прогонов (облачный и локальный Chronos).
Размер МО берётся по среднему уровню расходов за 2023 год: он известен в каждой точке прогноза,
поэтому деление на группы не заглядывает вперёд.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest

KEY = ["origin", "h", "series_id"]


def combine(pred: pd.DataFrame, members: list[str], name: str, how: str = "mean") -> pd.DataFrame:
    """Строка ансамбля из перечисленных моделей: среднее или медиана прогнозов."""
    sub = pred[pred["model"].isin(members)]
    wide = sub.pivot_table(index=KEY, columns="model", values="yhat")
    wide = wide.dropna(subset=members)
    yhat = wide[members].mean(axis=1) if how == "mean" else wide[members].median(axis=1)
    ref = sub.drop_duplicates(KEY).set_index(KEY)[["y", "y_prev_year"]].reindex(yhat.index)
    out = yhat.rename("yhat").to_frame().join(ref).assign(model=name).reset_index()
    return out[["model", "origin", "h", "series_id", "yhat", "y", "y_prev_year"]]


def switch(pred: pd.DataFrame, big_model: str, small_model: str, size: pd.Series,
           q: float = 0.8, name: str = "switch") -> pd.DataFrame:
    """Крупные МО считает одна модель, остальные — другая. Порог — квантиль уровня расходов."""
    thr = size.quantile(q)
    big = set(size[size >= thr].index)
    a = pred[(pred["model"] == big_model) & (pred["series_id"].isin(big))]
    b = pred[(pred["model"] == small_model) & (~pred["series_id"].isin(big))]
    return pd.concat([a, b], ignore_index=True).assign(model=name)


def size_groups(wide: pd.DataFrame, year: str = "2023", n: int = 5) -> pd.Series:
    """Средний уровень расходов за год -> номер группы (1 — самые малые МО)."""
    lvl = wide.loc[:, wide.columns.year == int(year)].mean(axis=1)
    return lvl


def metrics_by_size(pred: pd.DataFrame, level: pd.Series, n: int = 5) -> pd.DataFrame:
    grp = pd.qcut(level, n, labels=[f"Q{i+1}" for i in range(n)])
    p = pred.assign(size_group=pred["series_id"].map(grp))
    rows = []
    for g, part in p.groupby("size_group", observed=True):
        m = backtest.metrics(part)
        m.insert(0, "size_group", g)
        m["n_series"] = part["series_id"].nunique()
        m["mean_level"] = float(level[grp[grp == g].index].mean())
        rows.append(m)
    out = pd.concat(rows, ignore_index=True)
    # относительная ошибка внутри группы: MAE к уровню группы, чтобы группы были сопоставимы
    out["MAE_pct_of_level"] = out["MAE"] / out["mean_level"] * 100
    return out


def best_by_size(by_size: pd.DataFrame) -> pd.DataFrame:
    return (by_size.sort_values("MAE").groupby("size_group", observed=True).head(1)
            .sort_values("size_group")[["size_group", "model", "MAE", "MAE_pct_of_level", "n_series"]])
