"""Локальные модели: сезонный наивный, сезонный наивный с годовым приростом, ETS на
относительном ряду, Prophet (базовая модель жюри).

Единый интерфейс: fit_predict(wide, horizons, **params) -> DataFrame [series_id × h] уровней.
wide — ряды × месяцы, последний столбец — точка прогноза T.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _target_cols(wide: pd.DataFrame, horizons):
    last = wide.columns[-1]
    return {h: last + pd.DateOffset(months=h) for h in horizons}


def snaive(wide: pd.DataFrame, horizons, **_):
    """y_{T+h} = y_{T+h-12}."""
    out = {}
    for h in horizons:
        out[h] = wide.iloc[:, -12 + h - 1] if h <= 12 else np.nan
    return pd.DataFrame(out, index=wide.index)


def snaive_growth(wide: pd.DataFrame, horizons, window: int = 1, **_):
    """y_{T+h} = y_{T+h-12} · exp(средний log-прирост г/г за последние `window` месяцев)."""
    lg = np.log(wide)
    g = (lg - lg.shift(12, axis=1)).iloc[:, -window:].mean(axis=1)
    out = {h: wide.iloc[:, -12 + h - 1] * np.exp(g) for h in horizons}
    return pd.DataFrame(out, index=wide.index)


def common_factor(wide: pd.DataFrame, panel: pd.DataFrame | None = None) -> pd.Series:
    """Общий фактор: медиана log-уровня по рядам в каждом месяце (по всей панели, если передана)."""
    src = panel if panel is not None else wide
    return np.log(src).median(axis=0).reindex(wide.columns)


def ets_relative(wide: pd.DataFrame, horizons, panel: pd.DataFrame | None = None, **_):
    """ETS без сезонности на отклонении ряда от общего фактора; фактор — snaive_growth по медиане.

    Короткие ряды (24 месяца) не позволяют оценить сезонность внутри ряда; её несёт общий
    фактор, который оценивается по двум тысячам рядов сразу.
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS

    cf = common_factor(wide, panel)
    rel = np.log(wide).sub(cf, axis=1)
    H = max(horizons)
    long = rel.stack().rename("y").reset_index()
    long.columns = ["unique_id", "ds", "y"]
    sf = StatsForecast(models=[AutoETS(season_length=1)], freq="MS", n_jobs=1)
    fc = sf.forecast(df=long.dropna(), h=H)
    fc = fc.reset_index() if "unique_id" not in fc.columns else fc
    fc["h"] = fc.groupby("unique_id").cumcount() + 1
    rel_fc = fc.pivot(index="unique_id", columns="h", values="AutoETS")
    cf_wide = pd.DataFrame([np.exp(cf.values)], columns=wide.columns)
    cf_fc = snaive_growth(cf_wide, horizons).iloc[0]
    out = {h: np.exp(rel_fc[h]) * cf_fc[h] for h in horizons}
    return pd.DataFrame(out).reindex(wide.index)


def prophet(wide: pd.DataFrame, horizons, max_series: int | None = None, yearly_fourier: int = 3,
            seed: int = 0, **_):
    """Prophet на каждом ряду отдельно, как базовая модель в условиях конкурса."""
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    idx = wide.index
    if max_series is not None and len(idx) > max_series:
        idx = pd.Index(np.random.default_rng(seed).choice(idx, max_series, replace=False))
    H = max(horizons)
    res = {}
    for sid in idx:
        s = wide.loc[sid].dropna()
        df = pd.DataFrame({"ds": s.index, "y": s.values})
        m = Prophet(yearly_seasonality=yearly_fourier, weekly_seasonality=False, daily_seasonality=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(df)
        fut = m.make_future_dataframe(periods=H, freq="MS", include_history=False)
        yhat = m.predict(fut)["yhat"].to_numpy()
        res[sid] = {h: yhat[h - 1] for h in horizons}
    return pd.DataFrame(res).T.reindex(wide.index)


REGISTRY = {
    "snaive": snaive,
    "snaive_growth": snaive_growth,
    "ets_relative": ets_relative,
    "prophet": prophet,
}
