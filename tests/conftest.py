import os, sys
import pandas as pd
import pytest
from datetime import datetime, timedelta

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

@pytest.fixture
def df_small():
    data = [
        {"app_id": 1250410, "language": "english", "review": "Great gameplay, smooth performance!", "timestamp": "2025-06-01", "playtime_forever": 120, "review_id": "r1"},
        {"app_id": 1250410, "language": "french",  "review": "Beaucoup de bugs et des lags.",      "timestamp": "2025-06-08", "playtime_forever": 15,  "review_id": "r2"},
        {"app_id": 1250410, "language": "english", "review": "Okay, but crashes sometimes.",       "timestamp": "2025-06-15", "playtime_forever": 55,  "review_id": "r3"},
        {"app_id": 1250410, "language": "german",  "review": "Perfekt!",                           "timestamp": "2025-06-22", "playtime_forever": 5,   "review_id": "r4"},
    ]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

@pytest.fixture
def df_weekly_series():
    base = datetime(2025, 5, 1)
    vals = [0.10, 0.12, -0.80, 0.15, 0.16, 0.18]
    return pd.DataFrame({
        "week": [base + timedelta(weeks=i) for i in range(len(vals))],
        "sentiment_mean": vals
    })

@pytest.fixture
def vader_mock(monkeypatch):
    # Patch dashboard.analysis.get_vader to avoid nltk dependency in tests
    import dashboard.analysis as analysis

    class FakeVader:
        def polarity_scores(self, text):
            t = (text or "").lower()
            if any(k in t for k in ["bug", "lag", "crash", "bad"]):
                return {"compound": -0.6}
            if any(k in t for k in ["great", "perfekt", "smooth", "amazing"]):
                return {"compound": 0.7}
            return {"compound": 0.05}

    monkeypatch.setattr(analysis, "get_vader", lambda: FakeVader())
    return True