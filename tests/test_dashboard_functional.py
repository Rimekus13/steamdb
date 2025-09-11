import pandas as pd

def test_apply_filters_basic(df_small, vader_mock):
    from dashboard.analysis import compute_sentiment
    from dashboard.app import apply_filters
    df = df_small.copy()
    df["sentiment"] = df["review"].apply(compute_sentiment)
    out = apply_filters(
        df,
        languages=["english"],
        date_range=(pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-30")),
        sentiment_range=(0.0, 1.0),
        search_terms=[]
    )
    assert (out["language"] == "english").all()
    assert (out["sentiment"] >= 0.0).all()
    assert out["timestamp"].min() >= pd.Timestamp("2025-06-01")
    assert out["timestamp"].max() <= pd.Timestamp("2025-06-30")

def test_before_after_kpi_consistency(df_small, vader_mock):
    from dashboard.analysis import compute_sentiment, before_after_delta
    df = df_small.copy()
    df["sentiment"] = df["review"].apply(compute_sentiment)
    pivot = pd.Timestamp("2025-06-10")
    res = before_after_delta(df, date_col="timestamp", value_col="sentiment", pivot_date=pivot)
    assert set(res.keys()) == {"mean_before", "mean_after", "delta"}
    for k in res:
        assert isinstance(res[k], float)
