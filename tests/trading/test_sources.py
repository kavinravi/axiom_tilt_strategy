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
