import pandas as pd
from trading.data import sources


class FakeNDL:
    def __init__(self, frame): self._frame = frame
    def get_table(self, name, **kw): return self._frame.copy()


def test_latest_fundamentals_picks_max_datekey():
    frame = pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB"],
        "datekey": pd.to_datetime(["2025-05-01", "2026-02-01", "2026-01-15"]),
        "revenue": [10, 20, 5], "fcf": [1, 2, 1], "assets": [100, 110, 50],
        "price": [9, 11, 4], "sharesbas": [1000, 1000, 500],
    })
    out = sources.latest_fundamentals(["AAA", "BBB"], ndl=FakeNDL(frame))
    assert len(out) == 2
    assert out.set_index("ticker").loc["AAA", "revenue"] == 20  # the 2026-02-01 row


def test_latest_marketcap_picks_latest_on_or_before_asof():
    frame = pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB"],
        "date": pd.to_datetime(["2026-05-29", "2026-06-01", "2026-06-01"]),
        "marketcap": [100, 110, 50],
    })
    out = sources.latest_marketcap(["AAA", "BBB"], asof=pd.Timestamp("2026-06-02"), ndl=FakeNDL(frame))
    assert out.set_index("ticker").loc["AAA", "marketcap"] == 110


def test_fetch_close_history_extracts_and_ffills():
    import pandas as pd
    from trading.data import sources

    idx = pd.to_datetime(["2026-06-05", "2026-06-08", "2026-06-09"])
    # yfinance multi-ticker shape: columns are a (field, ticker) MultiIndex.
    cols = pd.MultiIndex.from_tuples(
        [("Close", "AAA"), ("Close", "BBB"), ("Open", "AAA"), ("Open", "BBB")]
    )
    raw = pd.DataFrame(
        [[10.0, 20.0, 1, 1], [float("nan"), 21.0, 1, 1], [12.0, 22.0, 1, 1]],
        index=idx, columns=cols,
    )

    def fake_download(tickers, **kw):
        return raw

    out = sources.fetch_close_history(["BBB", "AAA"], "2026-06-05", "2026-06-10",
                                      download=fake_download)
    assert list(out.columns) == ["AAA", "BBB"]            # sorted, Close-only
    assert out.loc[pd.Timestamp("2026-06-08"), "AAA"] == 10.0   # NaN forward-filled
    assert out.index[0] == pd.Timestamp("2026-06-05")


def test_fetch_close_history_empty_tickers():
    from trading.data import sources
    out = sources.fetch_close_history([], "2026-06-05", "2026-06-10",
                                      download=lambda *a, **k: None)
    assert out.empty


def test_fetch_close_history_single_ticker_flat_columns():
    import pandas as pd
    from trading.data import sources

    idx = pd.to_datetime(["2026-06-05", "2026-06-08"])
    raw = pd.DataFrame({"Open": [1, 1], "Close": [10.0, float("nan")]}, index=idx)

    def fake_download(tickers, **kw):
        return raw

    out = sources.fetch_close_history(["AAA"], "2026-06-05", "2026-06-09",
                                      download=fake_download)
    assert list(out.columns) == ["AAA"]
    assert out.loc[pd.Timestamp("2026-06-08"), "AAA"] == 10.0   # NaN forward-filled
