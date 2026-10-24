"""Workspace locations.

Everything generated (caches, stitching, mosaics, annotations, predictions, models, job logs) lives
in one data directory: $DUCT_DATA, default ./data. The images folder is recorded there by
`ductcount setup` (data/config.json), so the code never assumes where the images are.
"""
import json
import os

DATA = os.path.abspath(os.environ.get("DUCT_DATA", "data"))


def path(*p):
    return os.path.join(DATA, *p)


def settings():
    f = path("config.json")
    return json.load(open(f)) if os.path.exists(f) else {}


def save_settings(**kw):
    os.makedirs(DATA, exist_ok=True)
    s = settings()
    s.update(kw)
    json.dump(s, open(path("config.json"), "w"), indent=1)


def images_root():
    root = settings().get("images")
    if not root:
        raise SystemExit("No images folder configured: run `uv run ductcount setup /path/to/images` first.")
    return root
