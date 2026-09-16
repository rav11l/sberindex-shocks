"""Модели-основы для временных рядов (zero-shot). Необязательные зависимости.

chronos : пакет chronos-forecasting, чекпойнты amazon/chronos-bolt-{tiny,mini,small,base}
timesfm : пакет timesfm, чекпойнт google/timesfm-2.0-500m-pytorch

Если пакет или веса недоступны, модель пропускается с предупреждением, а в таблице
результатов появляется строка со статусом skipped — прогон остальных моделей не падает.

Два режима подачи ряда:
  level    — сам ряд (≤ 24 точки);
  relative — отклонение от общего фактора (медианы по МО), как в ets_relative: модель-основа
             видит ряд без общей сезонности, сезонность возвращает общий фактор.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .baselines import common_factor, snaive_growth

log = logging.getLogger(__name__)


class Unavailable(RuntimeError):
    pass


def _context(wide: pd.DataFrame, mode: str, panel: pd.DataFrame | None = None):
    if mode == "level":
        return wide, None
    cf = common_factor(wide, panel)
    return np.log(wide).sub(cf, axis=1), cf


def _restore(pred: pd.DataFrame, wide: pd.DataFrame, cf, mode: str, horizons):
    if mode == "level":
        return pred
    cf_wide = pd.DataFrame([np.exp(cf.values)], columns=wide.columns)
    cf_fc = snaive_growth(cf_wide, horizons).iloc[0]
    return pd.DataFrame({h: np.exp(pred[h]) * cf_fc[h] for h in horizons}, index=wide.index)


def chronos(wide: pd.DataFrame, horizons, checkpoint: str = "amazon/chronos-bolt-small",
            mode: str = "relative", device: str = "cpu", batch: int = 256, panel: pd.DataFrame | None = None, **_):
    try:
        import torch
        from chronos import BaseChronosPipeline
    except ImportError as e:
        raise Unavailable(f"chronos-forecasting не установлен: {e}")
    try:
        pipe = BaseChronosPipeline.from_pretrained(checkpoint, device_map=device, torch_dtype=torch.float32)
    except Exception as e:  # нет сети или весов
        raise Unavailable(f"не удалось загрузить {checkpoint}: {e}")
    ctx, cf = _context(wide, mode, panel)
    H = max(horizons)
    preds = []
    for i in range(0, len(ctx), batch):
        part = [torch.tensor(r[~np.isnan(r)], dtype=torch.float32) for r in ctx.iloc[i:i + batch].to_numpy()]
        q, mean = pipe.predict_quantiles(part, prediction_length=H, quantile_levels=[0.5])
        preds.append(q[..., 0].numpy())
    arr = np.vstack(preds)
    pred = pd.DataFrame({h: arr[:, h - 1] for h in horizons}, index=wide.index)
    return _restore(pred, wide, cf, mode, horizons)


def timesfm(wide: pd.DataFrame, horizons, checkpoint: str = "google/timesfm-2.0-500m-pytorch",
            mode: str = "relative", panel: pd.DataFrame | None = None, **_):
    try:
        import timesfm as tfm
    except ImportError as e:
        raise Unavailable(f"timesfm не установлен: {e}")
    try:
        model = tfm.TimesFm(
            hparams=tfm.TimesFmHparams(backend="cpu", per_core_batch_size=64, horizon_len=max(horizons),
                                       num_layers=50, use_positional_embedding=False, context_len=64),
            checkpoint=tfm.TimesFmCheckpoint(huggingface_repo_id=checkpoint))
    except Exception as e:
        raise Unavailable(f"не удалось загрузить {checkpoint}: {e}")
    ctx, cf = _context(wide, mode, panel)
    inputs = [r[~np.isnan(r)] for r in ctx.to_numpy()]
    point, _ = model.forecast(inputs, freq=[1] * len(inputs))
    pred = pd.DataFrame({h: point[:, h - 1] for h in horizons}, index=wide.index)
    return _restore(pred, wide, cf, mode, horizons)


REGISTRY = {"chronos": chronos, "timesfm": timesfm}
