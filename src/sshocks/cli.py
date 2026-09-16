"""Точка входа: python -m sshocks.cli <команда> --config configs/default.yaml

Команды:
  data         проверка хешей, сборка длинной таблицы, паспорт данных
  forecast     скользящий бэктест прогнозных моделей
  changepoint  полусинтетическая проверка детекторов и реальные события
  figures      рисунки для отчёта
  all          всё по порядку
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import backtest, data, events
from .changepoint import benchmark

log = logging.getLogger("sshocks")


def load_cfg(path: str, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg


def out_dir(cfg) -> Path:
    p = Path(cfg["output_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def cmd_data(cfg, args):
    raw_dir = Path(cfg["data"]["raw_dir"])
    if args.download:
        path = data.download(raw_dir, "parquet" if cfg["data"]["file"].endswith("parquet") else "csv")
        log.info("скачано: %s", path)
    hf = Path(cfg["data"]["hashes"])
    if hf.exists():
        st = data.verify_hashes(raw_dir, hf)
        log.info("хеши: %s", st)
        if any(v == "mismatch" for v in st.values()):
            log.warning("файл отличается от зафиксированного — вероятно, СберИндекс обновил выгрузку")
    long = data.load(cfg)
    wide = data.panel(long, cfg["data"]["category"], min_obs=1)
    passport = {
        "rows": int(len(long)),
        "series_total": int(long["series_id"].nunique()),
        "categories": long.groupby("category")["series_id"].nunique().to_dict(),
        "period": [str(long["ds"].min().date()), str(long["ds"].max().date())],
        "target_category": cfg["data"]["category"],
        "series_in_category": int(len(wide)),
        "series_full_24": int(wide.notna().all(axis=1).sum()),
        "same_name_series": int((long.drop_duplicates("series_id")["n_same_name"] > 1).sum()),
    }
    (out_dir(cfg) / "data_passport.json").write_text(json.dumps(passport, ensure_ascii=False, indent=2), encoding="utf-8")
    long.to_parquet(out_dir(cfg) / "long.parquet")
    log.info("паспорт: %s", passport)
    return long


def _wide(cfg):
    p = out_dir(cfg) / "long.parquet"
    long = pd.read_parquet(p) if p.exists() else data.load(cfg)
    return data.panel(long, cfg["data"]["category"], min_obs=cfg["data"]["min_obs"])


def _registry(cfg):
    return events.load_registry(cfg["events"]["registry"], only_verified=cfg["events"]["only_verified"])


def cmd_forecast(cfg, args):
    wide = _wide(cfg)
    reg = _registry(cfg)
    series = None
    n = cfg["forecast"].get("max_series")
    if n and n < len(wide):
        series = wide.sample(n=n, random_state=cfg["seed"]).index
    pred, status = backtest.backtest(wide, cfg, reg, series)
    o = out_dir(cfg)
    pred.to_parquet(o / "forecast_predictions.parquet")
    status.to_csv(o / "forecast_status.csv", index=False)
    m = backtest.metrics(pred)
    mh = backtest.metrics(pred, by=("model", "h"))
    m.to_csv(o / "forecast_metrics.csv", index=False)
    mh.to_csv(o / "forecast_metrics_by_h.csv", index=False)
    models = m["model"].tolist()
    dms = [backtest.dm_test(pred, a, "prophet") for a in models if a != "prophet" and "prophet" in models]
    pd.DataFrame(dms).to_csv(o / "forecast_dm_vs_prophet.csv", index=False)
    print(m.sort_values("MAE").to_string(index=False))
    return m


def cmd_changepoint(cfg, args):
    wide = _wide(cfg)
    res, bd = benchmark.run(wide, cfg)
    o = out_dir(cfg)
    res.to_csv(o / "changepoint_benchmark.csv", index=False)
    bd.to_csv(o / "changepoint_by_delta.csv", index=False)
    print(res.to_string(index=False))
    reg = events.load_registry(cfg["events"]["registry"], only_verified=False)
    truth = events.truth_changepoints(reg, wide.index)
    real = benchmark.real_events(wide, truth, cfg)
    real.to_csv(o / "changepoint_real_events.csv", index=False)
    if len(real):
        print(real.to_string(index=False))
    return res


def cmd_figures(cfg, args):
    from . import figures
    figures.make_all(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sshocks")
    ap.add_argument("command", choices=["data", "forecast", "changepoint", "figures", "all"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="переопределение ключей: forecast.max_series=200")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_cfg(args.config, args.set)
    np.random.seed(cfg["seed"])
    steps = ["data", "forecast", "changepoint", "figures"] if args.command == "all" else [args.command]
    for s in steps:
        globals()[f"cmd_{s}"](cfg, args)


if __name__ == "__main__":
    main()
