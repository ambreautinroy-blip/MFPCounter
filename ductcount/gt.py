"""Scan the images folder: one row per image, joined with manual ImageJ point annotations if present.

File names must contain `<mouse>.<section>.<cut> Field_<n>` followed by an image extension,
e.g. `T47D MFP 1.3.2 Field_0004.jpg` (mouse 1, section 3, cut 2, field 4). Files may be in any
sub-folder. Manual counts are read from ImageJ multipoint exports `<image name>_coords.csv`
(columns Image, Index, X, Y) found anywhere under the same folder.
"""
import glob, os, re
import pandas as pd
from .config import images_root

NAME = re.compile(r"(\d+)\.(\d+)\.(\d+)\s*Field_0*(\d+)( 2)?\.(jpe?g|tiff?|png)$", re.I)
IMAGE_EXT = (".jpg", ".jpeg", ".tif", ".tiff", ".png")


def parse(name):
    m = NAME.search(name)
    if not m:
        return None
    mfp, sec, cut, field, dup, ext = m.groups()
    ext = {"jpeg": "jpg", "tiff": "tif"}.get(ext.lower(), ext.lower())
    return dict(mfp=int(mfp), section=int(sec), cut=int(cut), field=int(field), dup=bool(dup), ext=ext)


def key(d):
    return (d["mfp"], d["section"], d["cut"], d["field"], d["dup"], d["ext"])


def load_points(root=None):
    """All ImageJ `*_coords.csv` files -> dict key -> list of (x, y)."""
    pts = {}
    for f in glob.glob(f"{root or images_root()}/**/*_coords.csv", recursive=True):
        df = pd.read_csv(f)
        p = parse(df.Image.iloc[0]) if len(df) else parse(os.path.basename(f).replace("_coords.csv", ".jpg"))
        if p:
            pts[key(p)] = list(zip(df.X, df.Y)) if len(df) else []
    return pts


def scan(root=None):
    """(images DataFrame, list of skipped image files whose name does not follow the convention)"""
    root = root or images_root()
    pts = load_points(root)
    rows, skipped = [], []
    for d, _, files in os.walk(root):
        for f in sorted(files):
            if not f.lower().endswith(IMAGE_EXT):
                continue
            p = parse(f)
            if p is None:
                skipped.append(os.path.join(d, f))
                continue
            k = key(p)
            rows.append(dict(path=os.path.join(d, f), **p, n_points=len(pts[k]) if k in pts else None, points=pts.get(k)))
    return pd.DataFrame(rows), skipped


def table(root=None):
    return scan(root)[0]


def fields(root=None):
    """One row per physical field. A .tif and a .jpg of the same field (Ambre's .tif files are
    pixel-identical copies) are merged: image = jpg if present, points = whichever has coords.
    points is None when the field has no manual annotation."""
    t, pts = table(root), load_points(root)
    out = []
    for k, g in t.groupby(["mfp", "section", "cut", "field", "dup"]):
        g = g.sort_values("ext")  # jpg before png/tif
        p = next((pts[(*k, e)] for e in ("jpg", "tif", "png") if (*k, e) in pts), None)
        out.append(dict(zip(["mfp", "section", "cut", "field", "dup"], k), path=g.path.iloc[0], points=p))
    df = pd.DataFrame(out, columns=["mfp", "section", "cut", "field", "dup", "path", "points"])
    df["n"] = df.points.map(lambda p: None if p is None else len(p))
    return df
