"""Реестр событий и новостей: согласование с месячной сеткой СберИндекса без заглядывания вперёд.

Правила согласования (подробно в docs/research_design.md, раздел 5):
1. Время события — момент публикации (announce_date), а не вступления в силу. Признак
   для прогноза из точки T строится только по записям с announce_date <= конец месяца T.
2. Дата вступления в силу (effective_date) переводится в месяц СберИндекса (метка ds —
   первое число месяца). Событие «относится» к месяцу, в котором вступает в силу.
3. География: national — все ряды; region/mo — ряды, чьё название МО попадает под
   регулярное выражение mo_pattern. Идентификатора региона в выгрузке нет, поэтому
   сопоставление по названию проверяется вручную (колонка n_matched в отчёте).
4. Новости агрегируются в счётчики по (месяц публикации, география, тип) и сдвигаются
   на лаг, выбранный на обучающем окне.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REGISTRY_COLUMNS = [
    "event_id", "announce_date", "effective_date", "scope", "mo_pattern", "type",
    "affected_categories", "expected_sign", "source_url", "verified", "note",
]


def month_start(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()


def load_registry(path: str | Path, only_verified: bool = False) -> pd.DataFrame:
    reg = pd.read_csv(path, dtype=str).fillna("")
    missing = set(REGISTRY_COLUMNS) - set(reg.columns)
    if missing:
        raise ValueError(f"В реестре нет колонок: {sorted(missing)}")
    reg["announce_date"] = pd.to_datetime(reg["announce_date"].replace("", np.nan))
    reg["effective_date"] = pd.to_datetime(reg["effective_date"].replace("", np.nan))
    # если дата объявления неизвестна, считаем её равной дате вступления (консервативно)
    reg["announce_date"] = reg["announce_date"].fillna(reg["effective_date"])
    reg["announce_month"] = month_start(reg["announce_date"])
    reg["effective_month"] = month_start(reg["effective_date"])
    if only_verified:
        reg = reg[reg["verified"].str.lower() == "yes"]
    return reg.reset_index(drop=True)


def match_series(reg: pd.DataFrame, series_ids: pd.Index) -> dict[str, np.ndarray]:
    """event_id -> булев массив по рядам."""
    mo = pd.Series(series_ids, index=series_ids).str.split("|").str[1]
    out = {}
    for _, e in reg.iterrows():
        if e["scope"] == "national":
            out[e["event_id"]] = np.ones(len(series_ids), dtype=bool)
        else:
            pat = re.compile(e["mo_pattern"]) if e["mo_pattern"] else None
            out[e["event_id"]] = mo.str.contains(pat).to_numpy() if pat else np.zeros(len(series_ids), bool)
    return out


def event_features(reg: pd.DataFrame, series_ids: pd.Index, origin: pd.Timestamp,
                   target: pd.Timestamp) -> pd.DataFrame:
    """Признаки событий для прогноза из точки origin на месяц target.

    ev_local_target   — число локальных событий, вступающих в силу в месяце target;
    ev_nat_target     — то же для национальных;
    ev_local_recent   — локальные события, вступившие в силу за 3 месяца до target;
    Используются только события, объявленные не позже конца месяца origin.
    """
    known = reg[reg["announce_month"] <= origin]
    masks = match_series(known, series_ids)
    n = len(series_ids)
    f = {k: np.zeros(n) for k in ["ev_local_target", "ev_nat_target", "ev_local_recent", "ev_nat_recent"]}
    for _, e in known.iterrows():
        m = masks[e["event_id"]]
        sign = -1.0 if e["expected_sign"] == "negative" else 1.0
        kind = "nat" if e["scope"] == "national" else "local"
        if e["effective_month"] == target:
            f[f"ev_{kind}_target"] += m * sign
        lag = (target.to_period("M") - e["effective_month"].to_period("M")).n if pd.notna(e["effective_month"]) else -1
        if 1 <= lag <= 3:
            f[f"ev_{kind}_recent"] += m * sign
    return pd.DataFrame(f, index=series_ids)


def truth_changepoints(reg: pd.DataFrame, series_ids: pd.Index, only_local: bool = True) -> pd.DataFrame:
    """Размеченные сдвиги для проверки детекторов на реальных событиях."""
    rows = []
    sub = reg[reg["scope"] != "national"] if only_local else reg
    masks = match_series(sub, series_ids)
    for _, e in sub.iterrows():
        for sid in np.asarray(series_ids)[masks[e["event_id"]]]:
            rows.append({"series_id": sid, "event_id": e["event_id"], "tau": e["effective_month"],
                         "announce": e["announce_month"]})
    return pd.DataFrame(rows)


def aggregate_news(news: pd.DataFrame, series_ids: pd.Index, gazetteer: dict[str, str] | None = None,
                   lag_months: int = 0) -> pd.DataFrame:
    """Новости (published_at, geo, type[, weight]) -> счётчики по ряду и месяцу публикации.

    geo: 'RU' для федеральных, иначе регулярное выражение по названию МО или ключ gazetteer.
    Результат сдвинут на lag_months вперёд: новость месяца t влияет на месяц t+lag.
    """
    news = news.copy()
    news["month"] = month_start(news["published_at"]) + pd.offsets.MonthBegin(lag_months)
    news["weight"] = news.get("weight", 1.0)
    mo = pd.Series(series_ids, index=series_ids).str.split("|").str[1]
    parts = []
    for (month, geo, typ), g in news.groupby(["month", "geo", "type"]):
        w = g["weight"].sum()
        if geo == "RU":
            mask = np.ones(len(series_ids), bool)
        else:
            pat = gazetteer.get(geo, geo) if gazetteer else geo
            mask = mo.str.contains(pat).to_numpy()
        parts.append(pd.DataFrame({"series_id": np.asarray(series_ids)[mask], "ds": month,
                                   "type": typ, "w": w}))
    if not parts:
        return pd.DataFrame(columns=["series_id", "ds", "type", "w"])
    return pd.concat(parts).groupby(["series_id", "ds", "type"], as_index=False)["w"].sum()
