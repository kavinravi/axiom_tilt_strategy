import pandas as pd
from trading.data.universe import current_members_from_sp500_table


def test_current_members_from_action_table():
    # 'current' is the explicit membership marker; reconstruction must agree.
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01", "2020-01-01"]),
        "action": ["added", "removed", "current", "current"],
        "ticker": ["AAA", "AAA", "BBB", "CCC"],
    })
    members = current_members_from_sp500_table(df)
    assert members == ["BBB", "CCC"]  # AAA added then removed; BBB/CCC current
