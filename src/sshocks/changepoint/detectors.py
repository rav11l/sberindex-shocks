"""Онлайн-детекторы точек структурных изменений для коротких панелей МО.

Все детекторы работают с одним и тем же преобразованным рядом
    d_{i,t} = log(y_{i,t}/y_{i,t-12}) − median_j log(y_{j,t}/y_{j,t-12}),
то есть с приростом г/г относительно медианного МО в том же месяце. Преобразование
убирает сезонность и общероссийские движения (инфляцию, ставку, праздники), которые
для отдельного МО шоком не являются.

Онлайн-режим: решение в месяце t принимается только по данным до t включительно.
Каждый детектор возвращает булеву матрицу тревог [ряд × месяц].
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def transform(wide: pd.DataFrame) -> pd.DataFrame:
    g = np.log(wide).diff(12, axis=1)
    return g.sub(g.median(axis=0), axis=1)


def _robust_scale(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    mad = np.median(np.abs(x - np.median(x))) * 1.4826
    return mad if mad > 1e-9 else np.std(x) + 1e-9


def zscore(d: pd.DataFrame, eval_cols, k: float = 3.0, min_hist: int = 3, **_):
    """|d_t − медиана прошлого| / MAD прошлого > k."""
    A = pd.DataFrame(False, index=d.index, columns=eval_cols)
    arr = d.to_numpy()
    cols = list(d.columns)
    for t in eval_cols:
        j = cols.index(t)
        hist = arr[:, :j]
        for i in range(arr.shape[0]):
            h = hist[i][~np.isnan(hist[i])]
            if len(h) < min_hist or np.isnan(arr[i, j]):
                continue
            s = _robust_scale(h)
            A.iat[i, eval_cols.index(t)] = abs(arr[i, j] - np.median(h)) / s > k
    return A


def cross_section(d: pd.DataFrame, eval_cols, k: float = 3.0, **_):
    """Панельный детектор: скачок Δd_t сравнивается с разбросом скачков всех МО в том же месяце.

    Не требует истории ряда — работает с первого месяца, когда определён прирост г/г.
    """
    dd = d.diff(axis=1)
    A = pd.DataFrame(False, index=d.index, columns=eval_cols)
    for t in eval_cols:
        x = dd[t]
        s = _robust_scale(x.to_numpy())
        A[t] = ((x - x.median()).abs() / s > k).fillna(False)
    return A


def cusum(d: pd.DataFrame, eval_cols, k: float = 0.5, h: float = 4.0, min_hist: int = 3, **_):
    """Двусторонний CUSUM по стандартизованному отклонению от медианы прошлого, со сбросом."""
    A = pd.DataFrame(False, index=d.index, columns=eval_cols)
    cols = list(d.columns)
    start = cols.index(eval_cols[0])
    arr = d.to_numpy()
    for i in range(arr.shape[0]):
        sp = sn = 0.0
        ref_start = None
        for j in range(start, len(cols)):
            hist = arr[i, :j] if ref_start is None else arr[i, ref_start:j]
            hist = hist[~np.isnan(hist)]
            if len(hist) < min_hist or np.isnan(arr[i, j]):
                continue
            z = (arr[i, j] - np.median(hist)) / _robust_scale(hist)
            sp = max(0.0, sp + z - k)
            sn = max(0.0, sn - z - k)
            if sp > h or sn > h:
                if cols[j] in eval_cols:
                    A.iat[i, eval_cols.index(cols[j])] = True
                sp = sn = 0.0
                ref_start = j
    return A


def bocpd(d: pd.DataFrame, eval_cols, hazard: float = 1 / 24, threshold: float = 0.5, min_hist: int = 3,
          event_hazard: pd.DataFrame | None = None, event_boost: float = 6.0, **_):
    """Байесовский онлайн-детектор (Adams & MacKay, 2007), гауссовская модель с известной дисперсией.

    Дисперсия наблюдения оценивается робастно по истории ряда до начала окна оценки.
    Тревога в t, если апостериорная вероятность того, что с x_t начался новый режим, превышает threshold.
    event_hazard — матрица [ряд × месяц] с 1 там, где по реестру/новостям ожидается событие;
    в эти месяцы априорная интенсивность смены режима умножается на event_boost. Так новости
    входят в детектор как априорная информация, а не как отдельный сигнал.
    """
    A = pd.DataFrame(False, index=d.index, columns=eval_cols)
    cols = list(d.columns)
    start = cols.index(eval_cols[0])
    arr = d.to_numpy()
    for i in range(arr.shape[0]):
        x = arr[i]
        first = np.where(~np.isnan(x))[0]
        if len(first) < min_hist:
            continue
        hist = x[first[0]:start]
        hist = hist[~np.isnan(hist)]
        if len(hist) < min_hist:
            continue
        sigma2 = _robust_scale(hist) ** 2
        mu0, tau2 = np.median(hist), max(np.var(hist), sigma2)
        # апостериорные параметры для каждой длины пробега
        logR = np.array([0.0])
        mu = np.array([mu0])
        prec = np.array([1 / tau2])
        for j in range(first[0], len(cols)):
            if np.isnan(x[j]):
                continue
            pvar = 1 / prec + sigma2
            # весь пересчёт в логарифмах: при сильном выбросе и узкой дисперсии произведение R * pred
            # обнуляется у всех длин пробега, и R становился NaN до конца ряда
            logp = -0.5 * (x[j] - mu) ** 2 / pvar - 0.5 * np.log(2 * np.pi * pvar)
            # cp-масса, рождённая на шаге j, оценивает x_{j+1} уже как новый режим, поэтому
            # ожидаемое событие месяца j+1 повышает интенсивность на шаге j
            hz = hazard
            nxt = cols[j + 1] if j + 1 < len(cols) else None
            if event_hazard is not None and nxt is not None and nxt in event_hazard.columns:
                if event_hazard.iat[i, event_hazard.columns.get_loc(nxt)] > 0:
                    hz = min(0.9, hazard * event_boost)
            lj = logR + logp
            lcp = np.logaddexp.reduce(lj) + np.log(hz)
            logR = np.append(lcp, lj + np.log1p(-hz))
            logR -= np.logaddexp.reduce(logR)
            R = np.exp(logR)
            new_prec = prec + 1 / sigma2
            new_mu = (mu * prec + x[j] / sigma2) / new_prec
            mu = np.append(mu0, new_mu)
            prec = np.append(1 / tau2, new_prec)
            if j >= start and cols[j] in eval_cols:
                # R[1] — вероятность того, что x_t первое наблюдение нового режима
                A.iat[i, eval_cols.index(cols[j])] = R[1] > threshold
    return A


def ruptures_online(d: pd.DataFrame, eval_cols, algo: str = "pelt", model: str = "l2", penalty: float = 3.0,
                    recent: int = 1, min_hist: int = 3, **_):
    """PELT/BinSeg из ruptures, перезапускаемые на истории до t: тревога, если найден разлом
    в последних `recent` точках."""
    import ruptures as rpt

    A = pd.DataFrame(False, index=d.index, columns=eval_cols)
    cols = list(d.columns)
    arr = d.to_numpy()
    for i in range(arr.shape[0]):
        for t in eval_cols:
            j = cols.index(t)
            x = arr[i, : j + 1]
            x = x[~np.isnan(x)]
            if len(x) < min_hist + 2:
                continue
            s = _robust_scale(x[:-1]) or 1.0
            sig = (x / s).reshape(-1, 1)
            est = rpt.Pelt(model=model, min_size=1, jump=1) if algo == "pelt" else rpt.Binseg(model=model, min_size=1, jump=1)
            bkps = est.fit(sig).predict(pen=penalty)[:-1]
            A.iat[i, eval_cols.index(t)] = any(b >= len(x) - recent for b in bkps)
    return A


REGISTRY = {
    "zscore": zscore,
    "cross_section": cross_section,
    "cusum": cusum,
    "bocpd": bocpd,
    "pelt": lambda d, e, **p: ruptures_online(d, e, algo="pelt", **p),
    "binseg": lambda d, e, **p: ruptures_online(d, e, algo="binseg", **p),
}
