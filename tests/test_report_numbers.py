"""Сверка чисел отчёта с таблицами прогона.

Тест защищает от расхождения текста и результатов: если прогон пересчитан, а отчёт нет
(или наоборот), тест падает. Если каталог прогона недоступен, тест пропускается —
на чистой машине без данных он не мешает.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_full"
REPORT = ROOT / "report" / "methodology.md"

pytestmark = pytest.mark.skipif(
    not (OUT / "analysis_metrics.csv").exists(), reason="нет каталога прогона outputs_full")


def _num(text: str) -> float:
    return float(text.replace(" ", "").replace(" ", "").replace(",", "."))


def _mae_from_report() -> dict[str, float]:
    """Таблица «Результаты» отчёта: строки вида | ens3_mean | 762 | ... |"""
    rows = {}
    for line in REPORT.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*([a-z_0-9]+)\s*\|\s*([\d\s ]+)\s*\|", line)
        if m:
            rows.setdefault(m.group(1), _num(m.group(2)))
    return rows


@pytest.mark.parametrize("model", ["ens3_mean", "chronos_base", "ets_relative", "factor_only",
                                   "prophet_relative", "snaive_growth", "prophet"])
def test_mae_in_report_matches_run(model):
    metrics = pd.read_csv(OUT / "analysis_metrics.csv").set_index("model")["MAE"]
    assert model in metrics.index, f"модель {model} отсутствует в прогоне"
    reported = _mae_from_report()
    assert model in reported, f"модель {model} не найдена в таблице отчёта"
    assert abs(reported[model] - metrics[model]) <= 1.0, (
        f"{model}: в отчёте {reported[model]}, в прогоне {metrics[model]:.1f}")


def test_pairs_count_matches():
    n = int(pd.read_csv(OUT / "analysis_metrics.csv")["n"].max())
    text = REPORT.read_text(encoding="utf-8")
    assert f"{n // 1000} {n % 1000:03d}" in text or f"{n // 1000} {n % 1000:03d}" in text, (
        f"число пар {n} не упоминается в отчёте")


def test_event_study_numbers():
    s = pd.read_csv(OUT / "event_study_summary.csv").iloc[0]
    text = REPORT.read_text(encoding="utf-8")
    assert f"{abs(s['mean_affected_pp']):.1f}".replace(".", ",") in text
    assert f"{int(s['n_affected'])}" in text


def test_detector_table_matches():
    b = pd.read_csv(OUT / "changepoint_benchmark.csv").set_index("detector")
    text = REPORT.read_text(encoding="utf-8")
    for det in ["cross_section", "pelt"]:
        rec = f"{b.loc[det, 'recall']:.2f}".replace(".", ",")
        assert rec in text, f"полнота {det} = {rec} не найдена в отчёте"
