"""Leave-one-mouse-out training/evaluation of the U-Net, then a final model on all mice."""
import os, pickle, sys
import numpy as np
import pandas as pd
from .config import path
from .evaluate import load_cache, evaluate, summarize
from .unet import train, load, predict_heatmap, peaks, MODELS

THRS = [0.04, 0.06, 0.08, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3]
cache = load_cache()
mice = sorted({k[0] for k in cache})
EPOCHS = int(os.environ.get("EPOCHS", 30))
NORM = os.environ.get("NORM", "0") == "1"  # Reinhard colour normalisation
SEED = int(os.environ.get("SEED", 0))
TAG = os.environ.get("TAG", "")  # e.g. TAG=v2 -> models/unet_v2_holdout_mfpN.pt


def items(keys):
    return [(p, pts) for k in keys for p, pts in zip(cache[k]["paths"], cache[k]["complete"])]


def heatmaps(model, keys):
    return {k: [predict_heatmap(model, p) for p in cache[k]["paths"]] for k in keys}


def at(hms, thr):
    return evaluate(cache, {k: [peaks(h, thr) for h in v] for k, v in hms.items()})


def total_err(df):
    return (df.pred - df.ambre).abs().sum()


if __name__ == "__main__":
    out, chosen = [], []
    for m in mice:
        tr = [k for k in cache if k[0] != m]
        te = [k for k in cache if k[0] == m]
        print(f"fold: hold out MFP {m} ({len(items(te))} fields), train on {len(items(tr))} fields", flush=True)
        path = f"{MODELS}/unet_{TAG + '_' if TAG else ''}holdout_mfp{m}.pt"
        model = load(path) if os.path.exists(path) else train(items(tr), EPOCHS, path, norm=NORM, seed=SEED)
        # threshold picked on the training mice only (no peeking at the held-out mouse)
        hm_tr = heatmaps(model, tr)
        thr = min(THRS, key=lambda t: total_err(at(hm_tr, t)))
        chosen.append(thr)
        df = at(heatmaps(model, te), thr)
        summarize(df, f"MFP{m} thr{thr}")
        out.append(df)
    df = summarize(pd.concat(out), "U-Net LOMO")
    print(df.to_string())
    df.to_csv(path("cache", f"unet{TAG}_cv.csv"), index=False)
    # production model = ensemble of the fold models (each trained on 3 of the 4 mice)
    print("threshold for the ensemble:", np.median(chosen))
    pickle.dump(dict(thr=float(np.median(chosen)), models=[f"unet_{TAG + '_' if TAG else ''}holdout_mfp{m}.pt" for m in mice]),
                open(f"{MODELS}/ensemble{TAG}.cfg.pkl", "wb"))
