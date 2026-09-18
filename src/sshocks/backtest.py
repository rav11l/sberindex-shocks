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


def paired_summary(pred: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Парные сравнения двух моделей, не опирающиеся на 12 ячеек.

    share_series_better — доля МО, где средняя ошибка модели a меньше;
    wilcoxon_p          — знаково-ранговый тест по МО (ряды коррелированы между собой,
                          поэтому он говорит о типичном МО, а не о независимых наблюдениях);
    boot_ci             — блочный бутстрап по целевым месяцам: месяц — единица ресэмплинга,
                          что учитывает и межрядовую корреляцию, и перекрытие горизонтов.
    """
    from scipy import stats

    key = ["origin", "h", "series_id"]
    p = pred[pred["model"].isin([a, b])].dropna(subset=["y", "yhat"]).copy()
    p["ae"] = (p["y"] - p["yhat"]).abs()
    common = p.groupby(key)["model"].transform("nunique") == 2
    p = p[common]
    w = p.pivot_table(index=key, columns="model", values="ae")
    diff = (w[a] - w[b]).dropna()
    by_series = diff.groupby(level="series_id").mean()
    idx = diff.index.to_frame(index=False)
    target = _target_month(idx["origin"].to_numpy(), idx["h"].to_numpy())
    rng = np.random.default_rng(seed)
    months = pd.unique(target)
    blocks = [diff.to_numpy()[target == m] for m in months]
    sums = np.array([b.sum() for b in blocks])
    sizes = np.array([len(b) for b in blocks])
    pick = rng.integers(0, len(months), size=(n_boot, len(months)))
    means = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    stat, wp = stats.wilcoxon(by_series) if len(by_series) > 10 else (np.nan, np.nan)
    return {"a": a, "b": b, "n_series": len(by_series), "mean_diff": float(diff.mean()),
            "share_series_better": float((by_series < 0).mean()),
            "median_series_diff": float(by_series.median()), "wilcoxon_p": float(wp),
            "boot_ci_low": float(lo), "boot_ci_high": float(hi), "n_target_months": len(months)}


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


def _target_month(origin: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Целевой месяц ячейки: точка прогноза плюс горизонт. Прогнозы из разных точек на один
    месяц — перекрывающиеся наблюдения, это учитывается в тестах."""
    per = pd.PeriodIndex(pd.to_datetime(origin), freq="M") + h.astype(int)
    return per.to_timestamp().to_numpy()


def dm_test(pred: pd.DataFrame, a: str, b: str, hac: bool = True) -> dict:
    """Тест Диболда–Мариано на разности средних абсолютных ошибок в ячейках (точка, горизонт).

    Две поправки против наивной версии:
    1. Ячейки перекрываются по целевому месяцу: прогноз из июня на h = 3 и из августа на h = 1
       метят в один и тот же сентябрь. Дисперсия считается по Ньюи–Уэсту с лагом max(h) − 1,
       плюс поправка Харви–Лейборна–Ньюболда на малую выборку.
    2. Дополнительно приводится тест на непересекающихся наблюдениях: ошибки усредняются
       по целевому месяцу (6 месяцев вместо 12 ячеек).
    Обе модели сравниваются только на общих парах «МО–точка–горизонт».
    """
    from scipy import stats

    key = ["origin", "h", "series_id"]
    p = pred[pred["model"].isin([a, b])].dropna(subset=["y", "yhat"]).copy()
    p["ae"] = (p["y"] - p["yhat"]).abs()
    common = p.groupby(key)["model"].transform("nunique") == 2
    p = p[common]
    cell = p.pivot_table(index=["origin", "h"], columns="model", values="ae", aggfunc="mean")
    if len(cell) < 3 or {a, b} - set(cell.columns):
        return {"a": a, "b": b, "n": len(cell), "mean_diff": np.nan, "t": np.nan, "p": np.nan}
    d = (cell[a] - cell[b]).dropna()
    idx = d.index.to_frame(index=False)
    target = pd.Series(_target_month(idx["origin"].to_numpy(), idx["h"].to_numpy()))
    order = np.argsort(target.to_numpy())
    x = d.to_numpy()[order]
    n, mean = len(x), x.mean()
    lag = int(idx["h"].max()) - 1
    e = x - mean
    gamma = [float((e * e).mean())]
    for k in range(1, min(lag, n - 1) + 1):
        gamma.append(float((e[k:] * e[:-k]).mean()))
    var = gamma[0] + 2 * sum((1 - k / (lag + 1)) * g for k, g in enumerate(gamma[1:], start=1)) if lag else gamma[0]
    var = max(var, 1e-12)
    t = mean / np.sqrt(var / n)
    # поправка Харви–Лейборна–Ньюболда для малой выборки
    hln = np.sqrt((n + 1 - 2 * (lag + 1) + (lag + 1) * lag / n) / n)
    t_hac = t * hln
    p_hac = 2 * stats.t.sf(abs(t_hac), n - 1)
    t_plain = mean / (x.std(ddof=1) / np.sqrt(n))
    # непересекающиеся наблюдения: среднее по целевому месяцу
    by_target = pd.Series(d.to_numpy(), index=target.to_numpy()).groupby(level=0).mean()
    m = len(by_target)
    t_tgt = by_target.mean() / (by_target.std(ddof=1) / np.sqrt(m)) if m >= 3 else np.nan
    p_tgt = 2 * stats.t.sf(abs(t_tgt), m - 1) if m >= 3 else np.nan
    res = {"a": a, "b": b, "n": n, "mean_diff": mean, "t": t_hac if hac else t_plain,
           "p": p_hac if hac else 2 * stats.t.sf(abs(t_plain), n - 1),
           "t_naive": t_plain, "p_naive": 2 * stats.t.sf(abs(t_plain), n - 1),
           "n_targets": m, "t_targets": t_tgt, "p_targets": p_tgt}
    return res
