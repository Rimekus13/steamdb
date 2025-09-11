import pandas as pd
import numpy as np
import pytest

POS, NEG = 0.05, -0.05

def test_classify_sentiment_thresholds():
    from dashboard.analysis import classify_sentiment
    assert classify_sentiment(0.06) == "positive"
    assert classify_sentiment(-0.06) == "negative"
    assert classify_sentiment(0.0) == "neutral"
    assert classify_sentiment(POS) == "neutral"
    assert classify_sentiment(NEG) == "neutral"

def test_moving_average_basic():
    from dashboard.analysis import moving_avg
    s = pd.Series([1, 2, 3, 4, 5])
    out = moving_avg(s, window=3)
    assert len(out) == 5
    assert not np.isnan(out).any()

def test_compute_sentiment_vectorized(df_small, vader_mock):
    from dashboard.analysis import compute_sentiment
    scores = df_small["review"].apply(compute_sentiment)
    assert scores.between(-1, 1).all()
    assert scores.iloc[1] < 0
    assert scores.iloc[0] > 0 and scores.iloc[3] > 0
