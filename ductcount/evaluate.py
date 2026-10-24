"""Shared evaluation: dedup field detections in the cut mosaic and compare with Ambre.

Ambre marks each duct once per cut (in one of the overlapping fields), so the fair
comparison is at the cut level: unique ducts in mosaic coordinates."""
import os, pickle
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .config import path
from .stitch import to_mosaic, inside_fov, merge

CACHE = path("cache", "stitch.pkl")
DEDUP_R = 60     # px full-res: detections closer than this in the mosaic = same duct
MATCH_R = 120    # px full-res: a prediction matches a mark if within this distance (~ duct radius)


def load_cache():
    return pickle.load(open(CACHE, "rb"))


def match(pred, gt, r=MATCH_R):
    """Hungarian matching. Returns (tp_pairs, fp_idx, fn_idx)."""
    pred, gt = np.asarray(pred).reshape(-1, 2), np.asarray(gt).reshape(-1, 2)
    if not len(pred) or not len(gt):
        return [], list(range(len(pred))), list(range(len(gt)))
    D = np.hypot(*(pred[:, None] - gt[None]).transpose(2, 0, 1))
    i, j = linear_sum_assignment(D)
    ok = D[i, j] < r
    tp = list(zip(i[ok], j[ok]))
    fp = sorted(set(range(len(pred))) - set(i[ok]))
    fn = sorted(set(range(len(gt))) - set(j[ok]))
    return tp, fp, fn


def cut_mosaic_predictions(entry, field_preds):
    """field_preds: list (per field) of Nx2 full-res points. Keep only points inside each FOV,
    map to mosaic and merge duplicates from overlapping fields."""
    allm = []
    for T, m, p in zip(entry["T"], entry["masks"], field_preds):
        p = np.asarray(p, float).reshape(-1, 2)
        p = p[inside_fov(m, p)]
        allm.append(to_mosaic(T, p))
    return merge(np.vstack(allm) if allm else np.zeros((0, 2)), DEDUP_R)


def evaluate(cache, preds):
    """preds: dict cut_key -> list of per-field Nx2 arrays. Returns per-cut DataFrame."""
    rows = []
    for key, e in cache.items():
        if key not in preds:
            continue
        gt = e["unique_mosaic"]
        pm = cut_mosaic_predictions(e, preds[key])
        tp, fp, fn = match(pm, gt)
        rows.append(dict(mfp=key[0], section=key[1], cut=key[2], ambre=len(gt), pred=len(pm),
                         tp=len(tp), fp=len(fp), fn=len(fn)))
    df = pd.DataFrame(rows)
    return df


def summarize(df, name=""):
    tp, fp, fn = df.tp.sum(), df.fp.sum(), df.fn.sum()
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    err = df.pred - df.ambre
    print(f"{name:12s} cuts={len(df)} | count MAE/cut={err.abs().mean():.1f} bias={err.mean():+.1f} "
          f"rel.err={(err.abs().sum() / max(df.ambre.sum(), 1)):.1%} | precision={p:.2f} recall={r:.2f} F1={2*p*r/max(p+r,1e-9):.2f}")
    return df
