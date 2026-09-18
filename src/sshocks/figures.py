"""Рисунки для отчёта. Читают только файлы из output_dir, ничего не пересчитывают."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data
from .changepoint.detectors import transform

INK, MUTED, ACCENT, GRID = "#1f2328", "#6e7781", "#0b6e4f", "#d0d7de"
SERIES = ["#0b6e4f", "#1f6feb", "#bc4c00", "#8250df", "#57606a", "#a40e26"]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)


def make_all(cfg):
    o = Path(cfg["output_dir"])
    fig_dir = o / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    long = pd.read_parquet(o / "long.parquet")
    wide = data.panel(long, cfg["data"]["category"], min_obs=cfg["data"]["min_obs"])

    # 1. Общий фактор и разброс МО
    lg = np.log(wide)
    q = lg.quantile([0.1, 0.5, 0.9], axis=0)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = wide.columns
    ax.fill_between(x, np.exp(q.loc[0.1]), np.exp(q.loc[0.9]), color=ACCENT, alpha=0.15, lw=0, label="10–90 % МО")
    ax.plot(x, np.exp(q.loc[0.5]), color=ACCENT, lw=2, label="медианное МО")
    ax.set_ylabel("руб. в месяц", color=MUTED, fontsize=8)
    ax.set_title(f"Безналичные расходы, {cfg['data']['category'].lower()}: {len(wide)} МО", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_panel.png", dpi=160)
    plt.close(fig)

    # 2. MAE по моделям и горизонтам
    p = o / "forecast_metrics_by_h.csv"
    if p.exists():
        m = pd.read_csv(p)
        order = m.groupby("model")["MAE"].mean().sort_values().index
        fig, ax = plt.subplots(figsize=(7, 3.2))
        hs = sorted(m["h"].unique())
        wbar = 0.8 / len(hs)
        for k, h in enumerate(hs):
            mm = m[m.h == h].set_index("model").reindex(order)
            ax.bar(np.arange(len(order)) + k * wbar, mm["MAE"], wbar, label=f"h={h}",
                   color=ACCENT, alpha=0.45 + 0.25 * k)
        ax.set_xticks(np.arange(len(order)) + wbar * (len(hs) - 1) / 2, order, rotation=20, ha="right")
        ax.set_ylabel("MAE, руб.", color=MUTED, fontsize=8)
        ax.set_title("Ошибка прогноза по моделям и горизонтам", loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
        fig.tight_layout()
        fig.savefig(fig_dir / "02_mae.png", dpi=160)
        plt.close(fig)

    # 3. Детекторы: точность и полнота
    p = o / "changepoint_benchmark.csv"
    if p.exists():
        b = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.scatter(b["recall"], b["precision"], color=ACCENT, s=30)
        for _, r in b.iterrows():
            ax.annotate(r["detector"], (r["recall"], r["precision"]), fontsize=7, color=INK,
                        xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel("полнота", color=MUTED, fontsize=8)
        ax.set_ylabel("точность", color=MUTED, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Детекторы на полусинтетических шоках", loc="left", fontsize=10, color=INK)
        _style(ax)
        fig.tight_layout()
        fig.savefig(fig_dir / "03_detectors.png", dpi=160)
        plt.close(fig)

    # 4. Пример: прирост г/г относительно медианы для МО из реестра локальных событий
    d = transform(wide).loc[:, "2024-01-01":]
    picks = [s for s in d.index if any(k in s for k in ["город Орск", "город Новотроицк", "город Оренбург"])]
    if picks:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(d.columns, d.quantile(0.9) * 100, color=GRID, lw=1, ls="--", label="90-й перцентиль МО")
        ax.plot(d.columns, d.quantile(0.1) * 100, color=GRID, lw=1, ls="--", label="10-й перцентиль МО")
        for s in picks:
            ax.plot(d.columns, d.loc[s] * 100, lw=1.8, label=s.split("|")[1].replace("городской округ ", ""))
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.axvline(pd.Timestamp("2024-04-01"), color=MUTED, lw=0.8)
        ax.set_ylabel("п. п. к медианному МО", color=MUTED, fontsize=8)
        ax.set_title("Прирост расходов г/г относительно медианы: Оренбургская область, 2024", loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        _style(ax)
        fig.tight_layout()
        fig.savefig(fig_dir / "04_orenburg_case.png", dpi=160)
        plt.close(fig)

    # 5. Кривые точность–полнота по порогам детекторов
    p = o / "changepoint_sweep.csv"
    if p.exists():
        sw = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        for i, (det, g) in enumerate(sw.groupby("detector")):
            g = g.sort_values("recall")
            ax.plot(g["recall"], g["precision"], marker="o", ms=3, lw=1.4,
                    color=SERIES[i % len(SERIES)], label=det)
        ax.set_xlabel("полнота", color=MUTED, fontsize=8)
        ax.set_ylabel("точность", color=MUTED, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Детекторы: кривые по порогу", loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=7)
        _style(ax)
        fig.tight_layout()
        fig.savefig(fig_dir / "05_detector_curves.png", dpi=160)
        plt.close(fig)

    # 6. Ошибка по группам размера МО: ансамбль против базовой модели
    p = o / "analysis_by_size.csv"
    if p.exists():
        by = pd.read_csv(p)
        show = [m for m in ["ens3_mean", "chronos_base", "ets_relative", "snaive_growth", "prophet"]
                if m in set(by["model"])]
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        groups = sorted(by["size_group"].unique())
        xs = np.arange(len(groups))
        wbar = 0.8 / max(1, len(show))
        for i, m in enumerate(show):
            v = by[by["model"] == m].set_index("size_group").reindex(groups)["MAE_pct_of_level"]
            ax.bar(xs + i * wbar, v, wbar, label=m, color=SERIES[i % len(SERIES)])
        ax.set_xticks(xs + wbar * (len(show) - 1) / 2, groups)
        ax.set_ylabel("MAE, % от уровня расходов группы", color=MUTED, fontsize=8)
        ax.set_title("Ошибка по группам МО: Q1 — малые, Q5 — крупные", loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        _style(ax)
        fig.tight_layout()
        fig.savefig(fig_dir / "06_by_size.png", dpi=160)
        plt.close(fig)

    # 7. Событийный анализ паводков: сдвиг затронутых МО на фоне остальных
    p = o / "event_study.csv"
    if p.exists():
        es = pd.read_csv(p)
        d_all = transform(wide) * 100
        pre = d_all.loc[:, "2024-01-01":"2024-03-01"].mean(axis=1)
        post = d_all.loc[:, "2024-04-01":"2024-06-01"].mean(axis=1)
        shift = (post - pre).drop(es["series_id"], errors="ignore")
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.hist(shift, bins=60, color=GRID, label="остальные МО")
        for _, r in es.iterrows():
            ax.axvline(r["shift_pp"], color=ACCENT, lw=1.2, alpha=0.85)
        ax.axvline(np.nan, color=ACCENT, lw=1.2, label="МО из реестра ЧС")
        ax.set_xlabel("сдвиг прироста г/г, п. п. (апр.–июнь минус янв.–март)", color=MUTED, fontsize=8)
        ax.set_title("Паводки весны 2024: затронутые МО на фоне остальных", loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=7)
        _style(ax)
        fig.tight_layout()
        fig.savefig(fig_dir / "07_event_study.png", dpi=160)
        plt.close(fig)
