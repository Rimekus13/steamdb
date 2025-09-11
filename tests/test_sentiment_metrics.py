import pytest

def test_kpi_distribution_sums_100(df_small, vader_mock):
    from dashboard.analysis import compute_sentiment, classify_sentiment
    df = df_small.copy()
    df["sentiment"] = df["review"].apply(compute_sentiment)
    df["bucket"] = df["sentiment"].apply(classify_sentiment)
    total = len(df)
    pos = (df["bucket"] == "positive").mean() * 100.0
    neu = (df["bucket"] == "neutral").mean() * 100.0
    neg = (df["bucket"] == "negative").mean() * 100.0
    assert pytest.approx(pos + neu + neg, rel=1e-6, abs=1e-6) == 100.0

def test_language_grouping(df_small, vader_mock):
    from dashboard.analysis import compute_sentiment
    df = df_small.copy()
    df["sentiment"] = df["review"].apply(compute_sentiment)
    grouped = df.groupby("language")["sentiment"].mean()
    assert set(grouped.index) == {"english", "french", "german"}
    assert grouped["german"] > grouped["french"]
