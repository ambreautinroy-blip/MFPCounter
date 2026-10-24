"""Out-of-fold threshold calibration for a set of leave-one-mouse-out models.

Peaks on training images are more confident than on unseen images, so a threshold chosen on the
training mice is too high for a new mouse. Here every mouse is predicted by the model that never
saw it (OOF), and:
  * the threshold used for mouse m is chosen on the OOF predictions of the OTHER mice
    (honest cross-validated estimate, and what the app shows for Ambre's data);
  * the deployment threshold (new mice) is chosen on all OOF predictions.

usage: uv run python -m ductcount.calibrate [TAG]     (TAG '' = v1, 'v2', ...)
writes cache/oof_<TAG>.pkl and, with --register NAME, the thresholds into the app registry.
"""
import argparse, json, os, pickle
import numpy as np
import pandas as pd
from .config import path
from .evaluate import load_cache, evaluate, summarize
from .unet import load, predict_heatmap, peaks, MODELS

HERE = os.path.dirname(os.path.abspath(__file__))
THRS = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.125, 0.14, 0.15, 0.175, 0.2, 0.25, 0.3]


def objective(df):
    """mean relative count error per cut (smoothed so 1-duct cuts don't dominate)"""
    return ((df.pred - df.ambre).abs() / (df.ambre + 5)).mean()


def oof_peaks(tag):
    """tag: one model set ('', 'v2') or an ensemble 'v2+v2s1' (heatmaps averaged per fold)."""
    out = path("cache", f"oof_{tag or 'v1'}.pkl")
    if os.path.exists(out):
        return pickle.load(open(out, "rb"))
    cache = load_cache()
    res = {}
    for m in sorted({k[0] for k in cache}):
        models = [load(f"{MODELS}/unet_{t + '_' if t else ''}holdout_mfp{m}.pt") for t in tag.split("+")]
        for k in [k for k in cache if k[0] == m]:
            hms = [np.mean([predict_heatmap(mo, p) for mo in models], 0) for p in cache[k]["paths"]]
            res[k] = {t: [peaks(h, t) for h in hms] for t in THRS}
        print("OOF heatmaps done for MFP", m, flush=True)
    pickle.dump(res, open(out, "wb"))
    return res


def calibrate(tag):
    cache = load_cache()
    P = oof_peaks(tag)
    mice = sorted({k[0] for k in P})
    ev = {t: evaluate(cache, {k: v[t] for k, v in P.items()}) for t in THRS}
    per_mouse, out = {}, []
    for m in mice:
        t = min(THRS, key=lambda t: objective(ev[t][ev[t].mfp != m]))
        per_mouse[m] = t
        df = ev[t][ev[t].mfp == m]
        summarize(df, f"MFP{m} thr{t}")
        out.append(df)
    cv = pd.concat(out)
    summarize(cv, f"OOF-CV {tag or 'v1'}")
    deploy = min(THRS, key=lambda t: objective(ev[t]))
    print(f"deployment threshold (all mice): {deploy}")
    return per_mouse, deploy, cv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", default="")
    ap.add_argument("--register", default=None, help="registry model name to update with these thresholds")
    a = ap.parse_args()
    per_mouse, deploy, cv = calibrate(a.tag)
    cv.to_csv(path("cache", f"oof_cv_{a.tag or 'v1'}.csv"), index=False)
    print(cv.to_string(index=False))
    if a.register:
        from .project import path
        reg = json.load(open(path("models", "registry.json")))
        reg["models"][a.register].update(thr=deploy, thr_oof={str(k): v for k, v in per_mouse.items()})
        json.dump(reg, open(path("models", "registry.json"), "w"), indent=1)
        print("registry updated:", a.register)
