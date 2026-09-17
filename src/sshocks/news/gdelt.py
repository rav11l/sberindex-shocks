"""Новостной поток из GDELT 1.0 (дневные файлы событий) и разметка правилами.

Почему так (docs/research_design.md, §5):
- GDELT открыт и бесплатен, условия допускают свободное использование со ссылкой на проект;
  платных и проприетарных компонентов нет (п. 10.3 Положения конкурса).
- У каждой записи есть дата, геопривязка места события (ActionGeo_*) и ссылка на статью.
- Разметка — прозрачные правила (configs/news_rules.yaml) и ручная проверка выборки,
  без языковых моделей.

Файлы: http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip. Каждый скачанный архив
хешируется (sha256 в манифесте), из него сохраняются только записи с местом события в России,
сам архив удаляется.
"""
from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
import time
import warnings
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

BASE = "http://data.gdeltproject.org/events/{d}.export.CSV.zip"
# номера колонок в файлах GDELT 1.0 (58 колонок, формат с 01.04.2013)
COLS = {0: "GLOBALEVENTID", 1: "SQLDATE", 26: "EventCode", 28: "EventRootCode", 31: "NumMentions", 32: "NumSources",
        33: "NumArticles", 34: "AvgTone", 49: "ActionGeo_Type", 50: "ActionGeo_FullName",
        51: "ActionGeo_CountryCode", 52: "ActionGeo_ADM1Code", 53: "ActionGeo_Lat",
        54: "ActionGeo_Long", 57: "SOURCEURL"}


# ---------- загрузка ----------

def _get(url: str, retries: int = 3, timeout: int = 120) -> bytes | None:
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            log.warning("%s: HTTP %s, попытка %d", url, e.code, i + 1)
        except Exception as e:  # сеть
            log.warning("%s: %s, попытка %d", url, e, i + 1)
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"не удалось скачать {url}")


def _parse(blob: bytes, country: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, sep="\t", header=None, dtype=str, usecols=list(COLS),
                             quoting=csv.QUOTE_NONE, on_bad_lines="skip")
    df.columns = [COLS[c] for c in df.columns]
    return df[df["ActionGeo_CountryCode"] == country]


def fetch(start: str, end: str, raw_dir: str | Path, manifest: str | Path, country: str = "RS") -> pd.DataFrame:
    """Скачивает дни [start, end], пишет помесячные parquet и манифест. Повторный запуск докачивает."""
    raw_dir, manifest = Path(raw_dir), Path(manifest)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(manifest, dtype=str) if manifest.exists() else pd.DataFrame(
        columns=["date", "url", "sha256", "bytes", "rows_country"])
    have = set(done["date"])
    dates = pd.date_range(start, end, freq="D")
    for month, days in pd.Series(dates).groupby(dates.strftime("%Y-%m").to_numpy()):
        out = raw_dir / f"gdelt_events_{country}_{month}.parquet"
        todo = [d for d in days if d.strftime("%Y%m%d") not in have]
        if not todo and out.exists():
            continue
        parts = [pd.read_parquet(out)] if out.exists() else []
        for d in todo:
            ds = d.strftime("%Y%m%d")
            url = BASE.format(d=ds)
            blob = _get(url)
            if blob is None:
                log.warning("нет файла за %s", ds)
                row = {"date": ds, "url": url, "sha256": "", "bytes": "0", "rows_country": "0"}
            else:
                df = _parse(blob, country)
                parts.append(df)
                row = {"date": ds, "url": url, "sha256": hashlib.sha256(blob).hexdigest(),
                       "bytes": str(len(blob)), "rows_country": str(len(df))}
                log.info("%s: %d записей %s", ds, len(df), country)
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        if parts:
            pd.concat(parts, ignore_index=True).drop_duplicates("GLOBALEVENTID").to_parquet(out)
        done.sort_values("date").to_csv(manifest, index=False)  # сохраняем после каждого месяца
    files = sorted(raw_dir.glob(f"gdelt_events_{country}_*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True) if files else pd.DataFrame()


# ---------- география ----------

_TR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
               ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s",
                "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e", "yu", "ya"]))


def translit(s: str) -> str:
    return "".join(_TR.get(ch, ch) for ch in s.lower())


def _key(s: str) -> str:
    """Ключ сравнения латиницы: без дефисов, мягких вариантов и удвоений (Orsk, Novotroitsk, Yekaterinburg)."""
    s = translit(s) if re.search("[а-яё]", s.lower()) else s.lower()
    s = re.sub(r"[^a-z]", "", s)
    s = s.replace("kh", "h").replace("ts", "c").replace("iy", "i").replace("yy", "i").replace("y", "i").replace("j", "i")
    s = re.sub(r"^ie", "e", s)          # Yekaterinburg -> ekaterinburg
    return re.sub(r"(.)\1", r"\1", s)


_PREFIX = re.compile(r"^(городской округ|муниципальный округ|муниципальный район|городское поселение|"
                     r"закрытое административно-территориальное образование|зато|город|г\.)\s+", re.I)
_ADJ = re.compile(r"(ский|цкий|ской|цкой|ный|ной|ий|ый)$")


def city_of_mo(mo: str) -> str | None:
    """Город из названия МО: «городской округ город Орск» -> «Орск»; районы с прилагательным -> None."""
    s = mo.strip()
    if s.startswith("внутригородская"):          # районы Москвы и Санкт-Петербурга: GDELT их не различает
        return None
    for _ in range(3):
        s = _PREFIX.sub("", s)
    s = re.sub(r"\s+(городской округ|муниципальный округ|муниципальный район)$", "", s)
    if not s or _ADJ.search(s.split()[-1].lower()) or "район" in s:
        return None
    return s


def gazetteer(series_ids: pd.Index) -> pd.DataFrame:
    mo = pd.Series(pd.Index(series_ids).str.split("|").str[1]).drop_duplicates()
    g = pd.DataFrame({"mo": mo, "city": mo.map(city_of_mo)}).dropna()
    g["key"] = g["city"].map(_key)
    g["n_same_key"] = g.groupby("key")["mo"].transform("nunique")
    return g.reset_index(drop=True)


# ---------- разметка ----------

def load_rules(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def label(ev: pd.DataFrame, rules: dict, gaz: pd.DataFrame) -> pd.DataFrame:
    """Записи GDELT -> новости (published_at, type, geo, mo, source_url, rule) по правилам."""
    if ev.empty:
        return pd.DataFrame(columns=["published_at", "type", "scope", "mo", "geo", "expected_sign",
                                     "source_url", "rule", "n_articles"])
    df = ev.dropna(subset=["SOURCEURL"]).copy()
    df["published_at"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d")
    url = df["SOURCEURL"].str.lower()
    hits = []
    for typ, r in rules["types"].items():
        m = pd.Series(False, index=df.index)
        why = pd.Series("", index=df.index)
        if r.get("url_regex"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)   # группы в regex нужны только для чтения
                mu = url.str.contains(r["url_regex"], regex=True)
            m |= mu
            why = why.mask(mu, "url")
        if r.get("cameo_codes"):
            mc = df["EventCode"].isin([str(c) for c in r["cameo_codes"]])
            if r.get("cameo_requires_url"):
                mc &= mu
            m |= mc
            why = why.mask(mc & (why == ""), "cameo")
        sub = df[m].assign(type=typ, rule=why[m], expected_sign=r.get("expected_sign", "unclear"),
                           scope_rule=r.get("scope", "auto"))
        hits.append(sub)
    lab = pd.concat(hits, ignore_index=True)
    if lab.empty:
        return label(pd.DataFrame(), rules, gaz)
    # география: страна целиком -> national; город -> МО по газеттиру, неоднозначные отбрасываются
    city = lab["ActionGeo_FullName"].fillna("").str.split(",").str[0]
    lab["key"] = city.map(_key)
    uniq = gaz[gaz["n_same_key"] == 1].set_index("key")["mo"]
    lab["mo"] = lab["key"].map(uniq)
    is_country = lab["ActionGeo_Type"].eq("1")
    lab["scope"] = np.where(lab["scope_rule"].eq("national") | is_country, "national",
                            np.where(lab["mo"].notna(), "mo", "unmatched"))
    lab["geo"] = np.where(lab["scope"].eq("national"), "RU",
                          np.where(lab["scope"].eq("mo"), "^" + lab["mo"].fillna("").map(re.escape) + "$", ""))
    lab["n_articles"] = pd.to_numeric(lab["NumArticles"], errors="coerce").fillna(1)
    # одна статья = одна новость данного типа: дубликаты по ссылке схлопываются
    lab = (lab.sort_values("n_articles", ascending=False)
              .drop_duplicates(["SOURCEURL", "type"])
              .rename(columns={"SOURCEURL": "source_url", "ActionGeo_FullName": "place"}))
    cols = ["published_at", "type", "scope", "mo", "geo", "place", "expected_sign", "source_url", "rule",
            "EventCode", "n_articles"]
    return lab[cols].sort_values(["published_at", "type"]).reset_index(drop=True)


def review_sample(lab: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Выборка для ручной проверки, стратифицированная по типу. Колонку check заполняет человек:
    ok | wrong_type | wrong_geo | irrelevant | dead_link."""
    use = lab[lab["scope"] != "unmatched"]
    if use.empty:
        return use.assign(check="")
    k = max(1, n // use["type"].nunique())
    s = use.groupby("type", group_keys=False).apply(lambda g: g.sample(min(len(g), k), random_state=seed))
    return s.assign(check="", comment="").reset_index(drop=True)


def review_stats(rev: pd.DataFrame) -> pd.DataFrame:
    """Точность правил по проверенной выборке (полноту даёт сверка с реестром, см. registry_recall)."""
    r = rev[rev["check"].fillna("").str.len() > 0]
    if r.empty:
        return pd.DataFrame()
    return (r.assign(ok=r["check"].eq("ok"), geo_ok=~r["check"].isin(["wrong_geo"]))
             .groupby("type").agg(n=("ok", "size"), precision=("ok", "mean"), geo_precision=("geo_ok", "mean"))
             .reset_index())


def registry_recall(lab: pd.DataFrame, reg: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """Нашёл ли поток события из реестра: новость того же типа и географии в ±window_days от объявления."""
    rows = []
    for _, e in reg.iterrows():
        base = e["type"].replace("disaster_payments", "disaster")
        cand = lab[lab["type"].eq(base)]
        if e["scope"] != "national":
            cand = cand[cand["mo"].fillna("").str.contains(e["mo_pattern"], regex=True)]
        d = (cand["published_at"] - e["announce_date"]).dt.days
        near = cand[d.abs() <= window_days]
        rows.append({"event_id": e["event_id"], "type": e["type"], "announce_date": e["announce_date"].date(),
                     "found": len(near) > 0, "n_news": len(near),
                     "first_news": near["published_at"].min().date() if len(near) else None})
    return pd.DataFrame(rows)
