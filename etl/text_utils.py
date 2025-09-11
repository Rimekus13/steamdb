
import re
from unidecode import unidecode
from langdetect import detect, LangDetectException
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
from nltk.corpus import stopwords

try:
    download("vader_lexicon", quiet=True)
    download("stopwords", quiet=True)
except Exception:
    pass

_sia = SentimentIntensityAnalyzer()
_stop = set(stopwords.words("english")).union(stopwords.words("french"))

_url = re.compile(r"https?://\S+")
_ws = re.compile(r"\s+")
_nonword = re.compile(r"[^a-zA-Z0-9\s]")

def clean_text(s: str) -> str:
    if not isinstance(s, str) or not s:
        return ""
    s = unidecode(s)
    s = _url.sub(" ", s)
    s = s.lower()
    s = _nonword.sub(" ", s)
    s = _ws.sub(" ", s).strip()
    return s

def detect_lang(s: str) -> str:
    try:
        return detect(s)
    except LangDetectException:
        return "unknown"

def sentiment_scores(s: str) -> dict:
    if not s:
        return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
    return _sia.polarity_scores(s)

def tokenize_no_stop(s: str):
    toks = [t for t in s.split() if t and t not in _stop and len(t) > 2]
    return toks
