"""Загрузка и подготовка рядов СберИндекса «Потребительские безналичные расходы на уровне МО».

Выгрузка (csv.zip или parquet) не содержит идентификатора ряда: только название МО.
Одинаковые названия встречаются в разных регионах (294 пары «категория–МО» в выгрузке
от 25.11.2025). Ряды в файле идут сплошными блоками в порядке indicator_id API, поэтому
ряд восстанавливается как непрерывный блок строк с одинаковыми категорией и МО и
возрастающим периодом. Номер блока внутри одноимённых МО хранится в `dup_rank`.
"""
from __future__ import annotations

import hashlib
import io
import json
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_ID = "potrebitelskie-beznalicnye-rashody-na-urovne-munizipalnyh-obrazovanij"
API_BASE = "https://sberindex.ru/api/dataset/v1"
ALL = "Все категории"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes(raw_dir: str | Path, hashes_file: str | Path) -> dict:
    """Сверяет sha256 файлов с записанными. Возвращает {файл: 'ok'|'mismatch'|'missing'}."""
    raw_dir = Path(raw_dir)
    recorded = json.loads(Path(hashes_file).read_text(encoding="utf-8"))
    status = {}
    for name, meta in recorded["files"].items():
        p = raw_dir / name
        if not p.exists():
            status[name] = "missing"
        else:
            status[name] = "ok" if sha256(p) == meta["sha256"] else "mismatch"
    return status


def download(raw_dir: str | Path, fmt: str = "parquet") -> Path:
    """Скачивает выгрузку через публичный API сайта. Нужен заголовок RqUID (UUID)."""
    import urllib.request

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = f"{API_BASE}/download/{DATASET_ID}/{fmt}"
    req = urllib.request.Request(url, headers={"RqUID": uuid.uuid4().hex, "User-Agent": "sberindex-shocks"})
    name = "sberindex_mo_spending.parquet" if fmt == "parquet" else "sberindex_mo_spending.csv.zip"
    out = raw_dir / name
    with urllib.request.urlopen(req, timeout=120) as r:
        out.write_bytes(r.read())
    return out


def read_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.name.endswith(".csv.zip"):
        with zipfile.ZipFile(path) as z:
            inner = z.namelist()[0]
            df = pd.read_csv(io.BytesIO(z.read(inner)), sep=";", dtype={"decimals": str, "unit_mult": str})
    else:
        raise ValueError(f"Неизвестный формат: {path}")
    df["period"] = pd.to_datetime(df["period"])
    return df


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Сырые строки -> длинная таблица с series_id, сохраняющая порядок файла."""
    key = df["category_15"] + "|" + df["mo"]
    new_block = (key != key.shift()) | (df["period"] <= df["period"].shift())
    df = df.assign(block=new_block.cumsum())
    first = df.groupby("block", sort=False)[["category_15", "mo"]].first()
    first["dup_rank"] = first.groupby(["category_15", "mo"]).cumcount()
    first["n_same_name"] = first.groupby(["category_15", "mo"])["dup_rank"].transform("size")
    first["series_id"] = (
        first["category_15"] + "|" + first["mo"] + "|" + first["dup_rank"].astype(str)
    )
    out = df.merge(first[["series_id", "dup_rank", "n_same_name"]], left_on="block", right_index=True)
    out = out.rename(columns={"category_15": "category", "period": "ds", "value": "y"})
    return out[["series_id", "category", "mo", "dup_rank", "n_same_name", "ds", "y"]].reset_index(drop=True)


def panel(long: pd.DataFrame, category: str = ALL, min_obs: int = 24) -> pd.DataFrame:
    """Широкая таблица ряд × месяц для одной категории, только ряды с min_obs наблюдениями."""
    sub = long[long["category"] == category]
    wide = sub.pivot(index="series_id", columns="ds", values="y").sort_index(axis=1)
    return wide[wide.notna().sum(axis=1) >= min_obs]


def load(cfg: dict) -> pd.DataFrame:
    raw = Path(cfg["data"]["raw_dir"]) / cfg["data"]["file"]
    if not raw.exists():
        if cfg["data"].get("download_if_missing", False):
            download(cfg["data"]["raw_dir"], "parquet" if raw.suffix == ".parquet" else "csv")
        else:
            raise FileNotFoundError(f"{raw} не найден. Запустите: python -m sshocks.cli data --download")
    return to_long(read_raw(raw))


def seasonal_log_growth(wide: pd.DataFrame) -> pd.DataFrame:
    """log(y_t / y_{t-12})."""
    return np.log(wide).diff(12, axis=1)
