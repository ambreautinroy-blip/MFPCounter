"""Leave-one-mouse-out evaluation of the classical baseline (parameters tuned on the other mice)."""
import os
import pandas as pd
from .config import path
from .evaluate import load_cache, evaluate, summarize
from .classical import detect, grid

cache = load_cache()
mice = sorted({k[0] for k in cache})


def run(keys, params):
    return evaluate(cache, {k: [detect(p, **params) for p in cache[k]["paths"]] for k in keys})


def score(df):  # tuning objective: total relative count error + (1 - F1)
    tp, fp, fn = df.tp.sum(), df.fp.sum(), df.fn.sum()
    return (df.pred - df.ambre).abs().sum() / df.ambre.sum() + (fp + fn) / (2 * tp + fp + fn)


if __name__ == "__main__":
    allres = {}
    params = list(grid())
    table = {i: {m: run([k for k in cache if k[0] == m], p) for m in mice} for i, p in enumerate(params)}
    out = []
    for m in mice:
        best = min(range(len(params)), key=lambda i: score(pd.concat([table[i][o] for o in mice if o != m])))
        print(f"held-out MFP {m}: params {params[best]}")
        out.append(table[best][m])
    df = summarize(pd.concat(out), "classical")
    print(df.to_string())
    df.to_csv(path("cache", "classical_cv.csv"), index=False)
