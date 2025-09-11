# analysis.py — UNICODE friendly cleaning + sentiment helpers
import re
import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def ensure_nltk():
    try: nltk.data.find("corpora/stopwords")
    except LookupError: nltk.download("stopwords")
    try: nltk.data.find("tokenizers/punkt")
    except LookupError: nltk.download("punkt")

def get_stop_set():
    ensure_nltk()
    sw = set()
    for lg in ("english","french"):
        try: sw |= set(stopwords.words(lg))
        except: pass
    sw |= set("""game games play played playing steam je tu il elle nous vous ils elles plus tres très les des une un le la de du dans sur par pour est pas que avec sans ou mais donc car bien mal bug bugs crash crashe""".split())
    return sw

def get_vader():
    try: nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError: nltk.download("vader_lexicon")
    return SentimentIntensityAnalyzer()

def compute_sentiment(sia, text: str) -> float:
    return sia.polarity_scores(text)["compound"] if isinstance(text, str) else 0.0

def clean_text_series(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.lower()
    s = s.str.replace(r"http\S+", " ", regex=True)          # URLs
    s = s.str.replace("_", " ", regex=False)                # underscores -> espace
    s = s.str.replace(r"[^\w\s]+", " ", regex=True)         # garde \w (UNICODE) et espaces
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()  # espaces multiples
    return s

def contains_any(text, kws):
    t = str(text).lower()
    return any(k.lower() in t for k in kws)

def top_unigrams_bigrams(texts, n_top=15):
    sw = get_stop_set(); toks=[]
    for t in texts:
        toks += [w for w in re.findall(r"\w+", str(t).lower()) if len(w)>2 and w not in sw]
    uni = Counter(toks).most_common(n_top)
    big = Counter([" ".join(bg) for bg in ngrams(toks, 2) if all(len(w)>2 and w not in sw for w in bg)]).most_common(n_top)
    return uni, big

def pick_examples(df_sub, n=3):
    pos_examples = df_sub.sort_values("sentiment", ascending=False).head(n)["review_text"].astype(str).tolist()
    neg_examples = df_sub.sort_values("sentiment", ascending=True).head(n)["review_text"].astype(str).tolist()
    def cut(s, L=220):
        s = re.sub(r"\s+", " ", s).strip()
        return s if len(s)<=L else s[:L-1]+"…"
    return [cut(x) for x in pos_examples], [cut(x) for x in neg_examples]
