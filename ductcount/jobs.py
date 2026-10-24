"""Background jobs launched by the web app (also usable from the command line).

  uv run ductcount infer [--cuts 1.3.1,1.3.2]   predictions with the current model
  uv run ductcount train [--epochs N]           train on the reviewed fields, then predict

The model registry (data/models/registry.json) lists every trained model; "current" is used for
inference, "for_new_mice" (optional) for mice that have no cross-validation model."""
import argparse, json, os, re, time
import numpy as np
import torch
from .project import DATA, path, project, annotations, field_index
from .stitch import to_mosaic, from_mosaic, inside_fov, merge
from .evaluate import DEDUP_R
from .unet import load, predict_maps, peaks, train

REG = path("models", "registry.json")
DEFAULT_THR = 0.15          # peak threshold for a model trained from scratch (no calibration yet)
SCRATCH_EPOCHS = 40         # training from ImageNet weights
FINETUNE_EPOCHS = 6         # fine-tuning an existing model on corrections


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def registry():
    """The model registry, or None when no model has been trained or imported yet."""
    return json.load(open(REG)) if os.path.exists(REG) else None


def save_registry(reg):
    os.makedirs(path("models"), exist_ok=True)
    json.dump(reg, open(REG, "w"), indent=1)


# ---------------------------------------------------------------- inference

def infer(cuts=None):
    reg = registry()
    if not reg or not reg.get("current"):
        log("no model yet: annotate some fields, tick 'Field reviewed', then Retrain")
        return
    name = reg["current"]
    info = reg["models"][name]
    models = {f: load(path("models", f)) for f in info["files"]}
    # cross-validated ensembles: each mouse is predicted by the model that never saw it (honest
    # comparison with the manual counts); mice outside the training set use the whole ensemble
    oof = {}  # mouse -> models that never saw it (several when the entry is an ensemble of runs)
    for f in info["files"]:
        if m := re.search(r"holdout_mfp(\d+)", f):
            oof.setdefault(int(m.group(1)), []).append(f)
    new = reg.get("for_new_mice")  # models trained on all labelled mice, for mice outside the CV
    if new:
        for f in new["files"]:
            models.setdefault(f, load(path("models", f)))
    P = project()
    cuts = cuts or sorted(P["cuts"])
    os.makedirs(path("predictions"), exist_ok=True)
    for n, cid in enumerate(cuts):
        c = P["cuts"][cid]
        if c["mfp"] in oof:  # OOF threshold chosen on the other mice (calibrate.py)
            use, thr = [models[f] for f in oof[c["mfp"]]], info.get("thr_oof", {}).get(str(c["mfp"]), info["thr"])
        elif new:
            use, thr = [models[f] for f in new["files"]], new["thr"]
        else:
            use, thr = [models[f] for f in info["files"]], info["thr"]
        per_field = []
        for p, m in zip(c["paths"], c["masks"]):
            maps = np.mean([predict_maps(mo, p) for mo in use], 0)
            q = peaks(maps[0], thr)
            q = q[inside_fov(m, q)]
            iy, ix = (q[:, 1] * 0.25).astype(int), (q[:, 0] * 0.25).astype(int)
            score = maps[0][iy, ix]
            atyp = maps[1][iy, ix] if len(maps) > 1 else np.zeros(len(q))
            per_field.append((q, score, atyp))
        # same dedup as Ambre: a duct belongs to the first field whose FOV contains it
        allm = [to_mosaic(T, q) for T, (q, _, _) in zip(c["T"], per_field)]
        mos = merge(np.vstack(allm), DEDUP_R)
        owner = np.full(len(mos), -1)
        for i, (T, m) in enumerate(zip(c["T"], c["masks"])):
            owner[(owner < 0) & inside_fov(m, from_mosaic(T, mos))] = i
        out = {}
        for i, f in enumerate(c["fields"]):
            q, s, a = per_field[i]
            mine = from_mosaic(c["T"][i], mos[owner == i])
            pts = []
            for k in range(len(q)):
                d = np.hypot(*(mine - q[k]).T) if len(mine) else np.array([1e9])
                pts.append(dict(x=round(float(q[k, 0]), 1), y=round(float(q[k, 1]), 1), score=round(float(s[k]), 3),
                                # atypical = channel 1 carries at least half of the duct peak
                                cls="atypical" if a[k] >= 0.5 * s[k] else "normal", counted=bool(d.min() < DEDUP_R)))
            out[str(f)] = pts
        counts = {f: sum(p["counted"] for p in v) for f, v in out.items()}
        how = f"out-of-fold ({', '.join(oof[c['mfp']])})" if c["mfp"] in oof else f"new mouse: {', '.join(new['files']) if new else 'all models'}"
        json.dump(dict(model=name, how=how, thr=thr, time=time.strftime("%Y-%m-%d %H:%M"), fields=out),
                  open(path("predictions", cid + ".json"), "w"))
        log(f"[{n + 1}/{len(cuts)}] cut {cid}: {sum(counts.values())} ducts")
    log("inference done")


# ---------------------------------------------------------------- training

def training_items():
    """(mouse, (image, all points[, atypical points, classified])) for every reviewed field. Marks are propagated to
    the other fields of the cut that see the same duct (Ambre marks each duct only once)."""
    P = project()
    items, any_atyp = [], False
    for cid, c in P["cuts"].items():
        ann = annotations(cid)
        rev = [str(f) for f in c["fields"] if ann["reviewed"].get(str(f))]
        if not rev:
            continue
        mos, cls = [], []
        classified = {}  # a field with >= 1 atypical mark is taken as fully classified normal/atypical
        for f in rev:
            pts = ann["fields"].get(f, [])
            classified[f] = any(p["cls"] == "atypical" for p in pts)
            if pts:
                mos.append(to_mosaic(c["T"][field_index(cid, f)], [(p["x"], p["y"]) for p in pts]))
                cls += [p["cls"] == "atypical" for p in pts]
        mos = np.vstack(mos) if mos else np.zeros((0, 2))
        cls = np.array(cls, bool)
        any_atyp |= cls.any()
        for f in rev:
            i = field_index(cid, f)
            q = from_mosaic(c["T"][i], mos)
            ok = inside_fov(c["masks"][i], q)
            items.append((c["mfp"], (c["paths"][i], q[ok], q[ok & cls], classified[f])))
    if not any_atyp:
        items = [(m, it[:2]) for m, it in items]
    return items


def train_from_scratch(mice_items, epochs, cuts=None):
    """First model of a workspace: one model per held-out mouse when there are >= 2 annotated mice
    (so each mouse is predicted by a model that never saw it), else a single model."""
    mice = sorted({m for m, _ in mice_items})
    name = time.strftime("model-%Y%m%d-%H%M")
    files = []
    folds = mice if len(mice) >= 2 else [None]
    for m in folds:
        items = [it for mm, it in mice_items if m is None or mm != m]
        new = f"{name}_holdout_mfp{m}.pt" if m is not None else f"{name}.pt"
        log(f"training {new} from ImageNet weights on {len(items)} fields, {epochs} epochs")
        train(items, epochs, path("models", new), log=log, norm=True)
        files.append(new)
    reg = registry() or dict(models={})
    reg["models"][name] = dict(files=files, thr=DEFAULT_THR, norm=True, created=time.strftime("%Y-%m-%d %H:%M"),
                               note=f"trained from scratch on {len(mice_items)} reviewed fields ({len(mice)} mice), {epochs} epochs")
    reg["current"] = name
    save_registry(reg)
    log(f"saved {name}, now current model")
    infer(cuts)


def do_train(epochs=None, cuts=None):
    """Train on all reviewed fields: from scratch if there is no model yet, else fine-tune the
    current one. Cross-validated models (holdout_mfpN) are fine-tuned fold by fold, each WITHOUT
    its own mouse, so the predictions shown for annotated mice stay out-of-fold (honest)."""
    mice_items = training_items()
    if not mice_items or not sum(len(it[1]) for _, it in mice_items):
        log("nothing to train on: mark ducts in some fields and tick 'Field reviewed'")
        return
    reg = registry()
    if not reg or not reg.get("current"):
        return train_from_scratch(mice_items, epochs or SCRATCH_EPOCHS, cuts)
    epochs = epochs or FINETUNE_EPOCHS
    items_all = [it for _, it in mice_items]
    cur_name = reg["current"]
    cur = reg["models"][cur_name]
    norm = cur.get("norm", False)  # same preprocessing as the current (validated) model
    name = time.strftime("model-%Y%m%d-%H%M")
    n_at = sum(len(i[2]) for i in items_all) if len(items_all[0]) > 2 else 0
    log(f"training on {len(items_all)} reviewed fields, {sum(len(i[1]) for i in items_all)} duct marks, {n_at} atypical")
    files = []
    for j, f in enumerate(cur["files"]):
        m = re.search(r"holdout_mfp(\d+)", f)
        items = [it for mfp, it in mice_items if not m or mfp != int(m.group(1))]
        new = f"{name}_{j}_holdout_mfp{m.group(1)}.pt" if m else f"{name}_{j}.pt"
        log(f"fine-tuning {f} -> {new} on {len(items)} fields")
        ck = torch.load(path("models", f), map_location="cpu")  # keep each model's own preprocessing
        train(items, epochs, path("models", new), log=log, norm=ck.get("norm", False) if "state_dict" in ck else False,
              init=path("models", f), lr=3e-4)
        files.append(new)
    reg = registry()
    reg["models"][name] = dict(files=files, thr=cur["thr"], thr_oof=cur.get("thr_oof", {}), norm=norm,
                               created=time.strftime("%Y-%m-%d %H:%M"), parent=cur_name,
                               note=f"fine-tuned in the app on {len(items_all)} reviewed fields, {epochs} epochs")
    reg["current"] = name
    save_registry(reg)
    log(f"saved {name}, now current model")
    infer(cuts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["infer", "train", "realign"])
    ap.add_argument("--cuts", default=None)
    ap.add_argument("--epochs", type=int, default=None, help=f"default {FINETUNE_EPOCHS} (fine-tune) / {SCRATCH_EPOCHS} (from scratch)")
    a = ap.parse_args()
    if a.cmd == "realign":  # manual alignment saved by the app: data/alignment_pending.json
        from .project import realign
        pend = json.load(open(path("alignment_pending.json")))
        log(f"applying manual alignment of cut {pend['cut']}")
        realign(pend["cut"], pend["F"])
        infer([pend["cut"]])
    elif a.cmd == "infer":
        infer(a.cuts.split(",") if a.cuts else None)
    else:
        do_train(a.epochs, a.cuts.split(",") if a.cuts else None)
