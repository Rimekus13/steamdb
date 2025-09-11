import pytest

def test_pipeline_integration_minimal(df_small, vader_mock):
    from dashboard.analysis import compute_sentiment, classify_sentiment
    silver_df = df_small.copy()
    silver_df["sentiment"] = silver_df["review"].apply(compute_sentiment)
    silver_df["bucket"] = silver_df["sentiment"].apply(classify_sentiment)
    kpis = {
        "sentiment_mean": round(silver_df["sentiment"].mean(), 4),
        "pos_pct": round((silver_df["bucket"] == "positive").mean() * 100.0, 2),
        "neu_pct": round((silver_df["bucket"] == "neutral").mean() * 100.0, 2),
        "neg_pct": round((silver_df["bucket"] == "negative").mean() * 100.0, 2),
        "languages": int(silver_df["language"].nunique())
    }
    assert -1 <= kpis["sentiment_mean"] <= 1
    assert pytest.approx(kpis["pos_pct"] + kpis["neu_pct"] + kpis["neg_pct"], abs=0.01) == 100.0
    assert kpis["languages"] >= 2
