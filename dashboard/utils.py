# utils.py
import re
import matplotlib.dates as mdates

DEFAULT_FIGSIZE_WIDE = (5.0, 2.2)
DEFAULT_FIGSIZE_TALL = (6.0, 3.0)

def title(ax, txt): 
    ax.set_title(txt, fontsize=11, pad=6)

def compact_time_axis(ax, minticks=3, maxticks=6, rotate=0):
    loc = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
    ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.tick_params(axis="x", labelrotation=rotate); ax.tick_params(labelsize=9)

def clamp(d, lo, hi):
    if d < lo: return lo
    if d > hi: return hi
    return d

def tokenize(text):
    return [w for w in re.findall(r"[a-zA-ZÀ-ÖØ-öø-ÿ0-9]+", str(text).lower())]
