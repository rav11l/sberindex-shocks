import numpy as np
import pandas as pd

from sshocks import data, events
from sshocks.changepoint import benchmark, detectors
from sshocks.models import baselines


def _raw():
    periods = pd.date_range("2023-01-01", periods=24, freq="MS")
    rows = []
    # два одноимённых МО подряд и один отдельный
    for mo, base in [("Первомайский муниципальный район", 100), ("Первомайский муниципальный район", 200), ("город N", 300)]:
        for i, p in enumerate(periods):
            rows.append({"period": p, "value": base * (1.01 ** i) * (1.2 if p.month == 12 else 1.0),
                         "category_15": "Все категории", "mo": mo})
    return pd.DataFrame(rows)


def test_same_name_series_are_split():
    long = data.to_long(_raw())
    assert long["series_id"].nunique() == 3
    assert set(long["dup_rank"]) == {0, 1}


def test_snaive_growth_exact_on_geometric_series():
    wide = data.panel(data.to_long(_raw()), min_obs=24)
    train = wide.iloc[:, :18]
    fc = baselines.snaive_growth(train, [1, 2, 3], window=1)
    actual = wide.iloc[:, 18:21].to_numpy()
    assert np.allclose(fc.to_numpy(), actual, rtol=1e-9)


def test_transform_removes_common_factor():
    wide = data.panel(data.to_long(_raw()), min_obs=24)
    d = detectors.transform(wide)
    assert np.allclose(d.iloc[:, 12:].to_numpy(), 0, atol=1e-12)


def test_injected_shift_is_detected_by_cross_section():
    rng = np.random.default_rng(0)
    cols = pd.date_range("2023-01-01", periods=24, freq="MS")
    wide = pd.DataFrame(np.exp(rng.normal(10, 0.01, (200, 24))), columns=cols,
                        index=[f"Все категории|МО{i}|0" for i in range(200)])
    eval_cols = list(pd.date_range("2024-04-01", "2024-12-01", freq="MS"))
    w, truth = benchmark.inject(wide, eval_cols, [0.3], share=0.1, seed=1)
    A = detectors.cross_section(detectors.transform(w), eval_cols, k=5)
    s = benchmark.score(A, truth)
    assert s["recall"] > 0.9


def test_event_features_respect_announce_date(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("event_id,announce_date,effective_date,scope,mo_pattern,type,affected_categories,expected_sign,source_url,verified,note\n"
                 "X,2024-08-10,2024-09-01,national,,t,all,negative,,yes,\n", encoding="utf-8")
    reg = events.load_registry(p)
    idx = pd.Index(["Все категории|A|0"])
    before = events.event_features(reg, idx, pd.Timestamp("2024-07-01"), pd.Timestamp("2024-09-01"))
    after = events.event_features(reg, idx, pd.Timestamp("2024-08-01"), pd.Timestamp("2024-09-01"))
    assert before["ev_nat_target"].iloc[0] == 0
    assert after["ev_nat_target"].iloc[0] == -1
