"""Command line: uv run ductcount <command>

  setup IMAGES_DIR [--models DIR]   scan, stitch and align the images, prepare the app
  serve [--port 8765]               start the web app
  infer                             predict all cuts with the current model
  train [--epochs N]                train / fine-tune on the reviewed fields, then predict
  export-models OUT_DIR             copy the current model(s) + registry, to share them

All generated data goes to $DUCT_DATA (default ./data).
"""
import argparse, json, os, shutil, sys, time


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def setup(images, models=None):
    from .config import save_settings, path, DATA
    images = os.path.abspath(images)
    if not os.path.isdir(images):
        sys.exit(f"not a folder: {images}")
    save_settings(images=images)
    log(f"images: {images}")
    log(f"data:   {DATA}")

    from .gt import scan, fields
    t, skipped = scan(images)
    if not len(t):
        sys.exit("no images found. File names must contain '<mouse>.<section>.<cut> Field_<n>' "
                 "and end in .jpg/.tif/.png, e.g. 'T47D MFP 1.3.2 Field_0004.jpg'.")
    f = fields(images)
    f = f[~f.dup]
    cuts = f.groupby(["mfp", "section", "cut"]).size()
    log(f"found {len(t)} image files -> {len(f)} fields in {len(cuts)} cuts "
        f"({f.mfp.nunique()} {'mouse' if f.mfp.nunique() == 1 else 'mice'}); "
        f"{int(f.points.notna().sum())} fields have ImageJ marks "
        f"({int(f.n.fillna(0).sum())} marks)")
    for s in skipped[:10]:
        log(f"  skipped (name not recognised): {os.path.relpath(s, images)}")
    if len(skipped) > 10:
        log(f"  ... and {len(skipped) - 10} more skipped files")

    import cv2
    from .register import load_small
    for i, p in enumerate(f.path):
        im = load_small(p)
        if i == 0 and abs(im.shape[1] / im.shape[0] - 4032 / 3040) > 0.02:
            log(f"warning: images are {im.shape[1] * 4}x{im.shape[0] * 4}; the field-of-view model assumes 4032x3040")
        if (i + 1) % 20 == 0 or i + 1 == len(f):
            log(f"decoded {i + 1}/{len(f)} images")

    from . import build_cache, project
    log("stitching the fields of every cut (SIFT)")
    build_cache.build(log)
    log("building mosaics, annotations and cut-to-cut alignment")
    project.build(log)

    if models:
        import_models(models)
    from .jobs import registry, infer
    if registry():
        log("predicting all cuts with the current model")
        infer()
    else:
        log("no model: the app works for manual counting; mark ducts, tick 'Field reviewed' and click Retrain")

    from .xcut_check import flag_reliable
    from .xcut_refine import refine_all
    flag_reliable(log)
    refine_all(log)
    log("done. Start the app with:  uv run ductcount serve")


def import_models(src):
    from .config import path
    reg = json.load(open(os.path.join(src, "registry.json")))
    os.makedirs(path("models"), exist_ok=True)
    files = {f for m in reg["models"].values() for f in m["files"]} | set(reg.get("for_new_mice", {}).get("files", []))
    for f in sorted(files):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), path("models", f))
        else:
            log(f"warning: {f} listed in registry.json but missing in {src}")
    json.dump(reg, open(path("models", "registry.json"), "w"), indent=1)
    log(f"imported {len(files)} model files from {src} (current: {reg.get('current')})")


def export_models(out):
    from .config import path
    from .jobs import registry
    reg = registry()
    if not reg:
        sys.exit("no model to export")
    keep = {reg["current"]: reg["models"][reg["current"]]}
    out_reg = dict(current=reg["current"], models=keep)
    if "for_new_mice" in reg:
        out_reg["for_new_mice"] = reg["for_new_mice"]
    files = set(keep[reg["current"]]["files"]) | set(reg.get("for_new_mice", {}).get("files", []))
    os.makedirs(out, exist_ok=True)
    for f in sorted(files):
        shutil.copy(path("models", f), os.path.join(out, f))
    json.dump(out_reg, open(os.path.join(out, "registry.json"), "w"), indent=1)
    log(f"exported {len(files)} model files + registry.json to {out}")


def main():
    ap = argparse.ArgumentParser(prog="ductcount", description="Duct counting in stitched microscope fields")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="scan, stitch and align an images folder")
    s.add_argument("images")
    s.add_argument("--models", help="folder with registry.json + model weights to import")
    s = sub.add_parser("serve", help="start the web app")
    s.add_argument("--port", type=int, default=8765)
    sub.add_parser("infer", help="predict all cuts with the current model")
    s = sub.add_parser("train", help="train / fine-tune on the reviewed fields, then predict")
    s.add_argument("--epochs", type=int, default=None)
    s = sub.add_parser("export-models", help="copy the current model(s) + registry to a folder")
    s.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "setup":
        setup(a.images, a.models)
    elif a.cmd == "serve":
        from .server import run
        run(a.port)
    elif a.cmd == "infer":
        from .jobs import infer
        infer()
    elif a.cmd == "train":
        from .jobs import do_train
        do_train(a.epochs)
    elif a.cmd == "export-models":
        export_models(a.out)


if __name__ == "__main__":
    main()
