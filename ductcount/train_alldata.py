"""Train the colour-normalised U-Net on all of Ambre's mice, for images of new mice (e.g. MFP 5).
Not used for MFP 1-4 (those keep out-of-fold predictions). Registered as registry["for_new_mice"]."""
import json, os, shutil, time
from .evaluate import load_cache
from .unet import train, MODELS
from .project import path

cache = load_cache()
items = [(p, pts) for k in cache for p, pts in zip(cache[k]["paths"], cache[k]["complete"])]
files = []
for seed in (0, 1):
    out = f"{MODELS}/unet_v2_alldata_s{seed}.pt"
    if not os.path.exists(out):
        print(f"training on all {len(items)} fields, seed {seed}", flush=True)
        train(items, 45, out, seed=seed, norm=True)
    shutil.copy(out, path("models", os.path.basename(out)))
    files.append(os.path.basename(out))
reg = json.load(open(path("models", "registry.json")))
# threshold: the v2 deployment threshold from out-of-fold calibration
reg["for_new_mice"] = dict(files=files, thr=0.175, created=time.strftime("%Y-%m-%d %H:%M"),
                           note="v2 config trained on all of MFP 1-4 (2 seeds); OOF-calibrated v2 threshold")
json.dump(reg, open(path("models", "registry.json"), "w"), indent=1)
print("registered for_new_mice:", files, flush=True)
