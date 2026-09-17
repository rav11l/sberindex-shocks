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


def _gdelt_zip(rows):
    import io, zipfile
    lines = []
    for r in rows:
        f = [""] * 58
        _gdelt_zip.n = getattr(_gdelt_zip, "n", 0) + 1
        f[0] = str(_gdelt_zip.n)
        f[1], f[26], f[28], f[33] = r["date"], r.get("code", "010"), r.get("code", "010")[:2], "3"
        f[49], f[50], f[51], f[57] = r["gtype"], r["place"], r["cc"], r["url"]
        lines.append("\t".join(f))
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("x.export.CSV", "\n".join(lines))
    return b.getvalue()


def test_gdelt_parse_label_recall():
    import pandas as pd
    from sshocks.news import gdelt
    from sshocks import events
    blob = _gdelt_zip([
        {"date": "20240406", "gtype": "4", "place": "Orsk, Orenburg, Russia", "cc": "RS",
         "url": "https://ria.ru/20240406/proryv-damby-orsk-1938.html"},
        {"date": "20240406", "gtype": "4", "place": "Orsk, Orenburg, Russia", "cc": "RS",
         "url": "https://ria.ru/20240406/proryv-damby-orsk-1938.html"},          # дубликат ссылки
        {"date": "20240726", "gtype": "1", "place": "Russia", "cc": "RS",
         "url": "https://www.interfax.ru/business/972/klyuchevuyu-stavku-povysili"},
        {"date": "20240410", "gtype": "4", "place": "Berlin, Germany", "cc": "GM", "url": "https://x.de/flood"},
        {"date": "20240410", "gtype": "4", "place": "Sovetsk, Russia", "cc": "RS", "url": "https://x.ru/navodnenie"},
    ])
    ev = gdelt._parse(blob, "RS")
    assert len(ev) == 4
    sids = pd.Index(["Все категории|городской округ город Орск|0", "Все категории|городской округ Советск|0",
                     "Все категории|Советский муниципальный район|0", "Все категории|городской округ Советск|1",
                     "Все категории|муниципальный округ город Советск|0"])
    gaz = gdelt.gazetteer(sids)
    assert gdelt.city_of_mo("городской округ город Орск") == "Орск"
    assert gdelt.city_of_mo("Советский муниципальный район") is None
    lab = gdelt.label(ev, gdelt.load_rules("configs/news_rules.yaml"), gaz)
    orsk = lab[lab["type"] == "disaster"]
    assert (orsk["scope"] == "mo").sum() == 1 and orsk["mo"].dropna().iloc[0] == "городской округ город Орск"
    assert lab[lab["place"].str.startswith("Sovetsk")]["scope"].eq("unmatched").all()   # одноимённые МО
    assert lab[lab["type"] == "key_rate"]["scope"].eq("national").all()
    reg = events.load_registry("data/events/registry.csv")
    rec = gdelt.registry_recall(lab, reg).set_index("event_id")
    assert rec.loc["E101", "found"] and rec.loc["E005", "found"] and not rec.loc["E102", "found"]


def test_gdelt_fetch_resumes(tmp_path, monkeypatch):
    import pandas as pd
    from sshocks.news import gdelt
    calls = []
    blob = _gdelt_zip([{"date": "20240101", "gtype": "4", "place": "Orsk, Russia", "cc": "RS", "url": "u"}])
    monkeypatch.setattr(gdelt, "_get", lambda url: calls.append(url) or (None if "20240102" in url else
                        _gdelt_zip([{"date": url[-25:-17], "gtype": "4", "place": "Orsk, Russia", "cc": "RS", "url": "u"}])))
    ev = gdelt.fetch("2024-01-01", "2024-02-01", tmp_path / "raw", tmp_path / "m.csv")
    assert len(calls) == 32 and len(ev) == 31
    man = pd.read_csv(tmp_path / "m.csv", dtype=str)
    assert len(man) == 32 and man["sha256"].str.len().fillna(0).astype(int).max() == 64
    gdelt.fetch("2024-01-01", "2024-02-01", tmp_path / "raw", tmp_path / "m.csv")
    assert len(calls) == 32            # повторный запуск ничего не качает
