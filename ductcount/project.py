"""Project state shared by the web app and the training/inference jobs.

data/
  project.pkl            stitching of every cut + cross-cut alignment (built once by `build`)
  mosaics/<cut>.jpg      stitched overview of each cut at MOSAIC_SCALE
  annotations/<cut>.json manual marks per field (seeded with Ambre's ImageJ points)
  predictions/<cut>.json model detections per field
  models/                trained models + registry.json
  jobs/                  job logs

A cut id is "mfp.section.cut" (e.g. "1.3.2"); a field id is "<cut>.<field>" (e.g. "1.3.2.5").
Coordinates are always full-resolution pixels of the field image.

Built by `uv run ductcount setup`.
"""
import json, os, pickle, sys, time, uuid
import cv2
import numpy as np
from .register import load_small, SCALE
from .stitch import to_mosaic, from_mosaic, inside_fov

HERE = os.path.dirname(os.path.abspath(__file__))
from .config import DATA, path  # noqa: E402  (workspace location)
MOSAIC_SCALE = 0.125


def cut_id(k):
    return ".".join(map(str, k))


# ---------------------------------------------------------------- build

def _mosaic(e, out):
    """Render the stitched cut at MOSAIC_SCALE. Groups of fields that could not be stitched to each
    other are laid side by side. Returns per-field 3x3 transforms field (full res) -> mosaic image px."""
    S = np.diag([MOSAIC_SCALE, MOSAIC_SCALE, 1])
    h, w = 3040, 4032
    box = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).T
    comps = {}
    for i, T in enumerate(e["T"]):
        comps.setdefault(int(round(T[0, 2] / 1e6)), []).append(i)
    F, x0, H = [None] * len(e["T"]), 0.0, 0
    for c in sorted(comps):
        corners = np.vstack([((S @ e["T"][i]) @ box)[:2].T for i in comps[c]])
        mn, mx = corners.min(0), corners.max(0)
        off = np.eye(3); off[:2, 2] = [x0 - mn[0], -mn[1]]
        for i in comps[c]:
            F[i] = off @ S @ e["T"][i]
        x0 += mx[0] - mn[0] + 40
        H = max(H, mx[1] - mn[1])
    W, H = int(x0 - 40) + 1, int(H) + 1
    canvas = np.full((H, W, 3), 255, np.uint8)
    for i in reversed(range(len(F))):  # field 1 drawn last = on top
        A = F[i] @ np.diag([1 / SCALE, 1 / SCALE, 1])  # source image is at SCALE
        im = load_small(e["paths"][i])
        wi = cv2.warpAffine(im, A[:2], (W, H))
        wm = cv2.warpAffine(e["masks"][i], A[:2], (W, H), flags=cv2.INTER_NEAREST)
        canvas[wm > 0] = wi[wm > 0]
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return F, [W, H]


def _darkness(img, f):
    """Tissue signal for registration: colour saturation (H&E stain), not brightness, so the
    grey vignetted field-of-view discs do not dominate."""
    im = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    sat = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    return cv2.GaussianBlur(sat, (0, 0), 2)


def _align_mosaics(A, B):
    """Similarity (rotation + translation) mapping mosaic image A onto mosaic image B.
    Serial cuts share tissue outline and large structures but not cells, so: brute-force rotation
    with FFT cross-correlation at low resolution, then ECC refinement. Returns (3x3, score)."""
    f = 0.25
    a, b = _darkness(A, f), _darkness(B, f)
    n = int(2 ** np.ceil(np.log2(max(*a.shape, *b.shape) * 2)))
    Fb = np.fft.rfft2(b - b.mean(), (n, n))
    best = (-1, None)
    ca = (a.shape[1] / 2, a.shape[0] / 2)
    for ang in range(0, 360, 3):
        R = cv2.getRotationMatrix2D(ca, ang, 1.0)
        c, s_ = abs(R[0, 0]), abs(R[0, 1])
        nw, nh = int(a.shape[0] * s_ + a.shape[1] * c), int(a.shape[0] * c + a.shape[1] * s_)
        R[0, 2] += nw / 2 - ca[0]; R[1, 2] += nh / 2 - ca[1]
        ar = cv2.warpAffine(a, R, (nw, nh))
        if max(nw, nh) > n:
            continue
        xc = np.fft.irfft2(np.conj(np.fft.rfft2(ar - ar.mean(), (n, n))) * Fb, (n, n))
        score = xc.max() / (np.linalg.norm(ar - ar.mean()) * np.linalg.norm(b - b.mean()) + 1e-9)
        if score > best[0]:
            dy, dx = np.unravel_index(xc.argmax(), xc.shape)
            dy, dx = (dy if dy < n // 2 else dy - n), (dx if dx < n // 2 else dx - n)
            M = np.vstack([R, [0, 0, 1]]); M[:2, 2] += [dx, dy]
            best = (score, M)
    score, M = best
    # refine at mosaic/2 resolution with ECC (euclidean)
    f2 = 0.5
    a2, b2 = _darkness(A, f2), _darkness(B, f2)
    K = np.diag([f2 / f, f2 / f, 1]) @ M @ np.diag([f / f2, f / f2, 1])
    W = K[:2].astype(np.float32)
    try:
        _, W = cv2.findTransformECC(b2, a2, cv2.invertAffineTransform(W).astype(np.float32), cv2.MOTION_EUCLIDEAN,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5), None, 5)
        K = np.vstack([cv2.invertAffineTransform(W), [0, 0, 1]])
    except cv2.error:
        pass
    D = np.diag([f2, f2, 1])
    return np.linalg.inv(D) @ K @ D, float(score)


def build(log=print):
    from .evaluate import load_cache
    from .gt import fields as gt_fields
    cache = load_cache()
    os.makedirs(path("mosaics"), exist_ok=True)
    os.makedirs(path("annotations"), exist_ok=True)
    proj = {"cuts": {}, "cross": {}}
    names = {r.path: os.path.basename(r.path) for r in gt_fields().itertuples()}
    ov = path("alignment_overrides.json")
    overrides = json.load(open(ov)) if os.path.exists(ov) else {}
    for k, e in cache.items():
        cid = cut_id(k)
        if cid in overrides:  # manual alignment made in the app
            e = dict(e, T=[np.asarray(T) for T in overrides[cid]])
        F, size = _mosaic(e, path("mosaics", cid + ".jpg"))
        proj["cuts"][cid] = dict(mfp=k[0], section=k[1], cut=k[2], fields=e["fields"], paths=e["paths"],
                                 names=[names.get(p, os.path.basename(p)) for p in e["paths"]],
                                 T=e["T"], masks=e["masks"], F=F, mosaic_size=size)
        ann = path("annotations", cid + ".json")
        if not os.path.exists(ann):  # seed with Ambre's ImageJ marks
            pts = {str(f): [dict(id=uuid.uuid4().hex[:8], x=float(x), y=float(y), cls="normal", src="ambre")
                            for x, y in own] for f, own in zip(e["fields"], e["own"])}
            # a field with manual ImageJ marks is a complete annotation (0 ducts is informative too);
            # fields without any manual file are left unreviewed so they are not used as negatives
            lab = e.get("labelled", [True] * len(e["fields"]))
            json.dump(dict(fields=pts, reviewed={str(f): bool(l) for f, l in zip(e["fields"], lab)}), open(ann, "w"))
        log(f"mosaic {cid}")
    for a in proj["cuts"]:
        for b in proj["cuts"]:
            ka, kb = proj["cuts"][a], proj["cuts"][b]
            if a < b and (ka["mfp"], ka["section"]) == (kb["mfp"], kb["section"]):
                M, score = _align_mosaics(cv2.imread(path("mosaics", a + ".jpg")), cv2.imread(path("mosaics", b + ".jpg")))
                proj["cross"][(a, b)] = M          # mosaic image a -> mosaic image b
                proj["cross"][(b, a)] = np.linalg.inv(M)
                proj.setdefault("cross_score", {})[(a, b)] = proj["cross_score"][(b, a)] = score
                log(f"aligned cut {a} with {b} (correlation {score:.2f})")
    pickle.dump(proj, open(path("project.pkl"), "wb"))


# ---------------------------------------------------------------- runtime helpers

_proj, _mtime = None, None


def project():
    """Project state; reloaded when project.pkl changes on disk (e.g. after a manual realignment)."""
    global _proj, _mtime
    m = os.path.getmtime(path("project.pkl"))
    if _proj is None or m != _mtime:
        _proj, _mtime = pickle.load(open(path("project.pkl"), "rb")), m
    return _proj


def realign(cid, F_new):
    """Apply a manual alignment of the fields of a cut (3x3 transforms field -> mosaic image px, as
    edited in the app): new stitching, mosaic and cut-to-cut alignment for this cut. The override is
    also kept in alignment_overrides.json."""
    from . import xcut_refine
    from .xcut_check import marks, hit
    P = project()
    c = P["cuts"][cid]
    S = np.diag([MOSAIC_SCALE, MOSAIC_SCALE, 1])
    c["T"] = [np.linalg.inv(S) @ np.asarray(F, float) for F in F_new]   # one common frame for all fields
    c["F"], c["mosaic_size"] = _mosaic(c, path("mosaics", cid + ".jpg"))
    ov = path("alignment_overrides.json")
    overrides = json.load(open(ov)) if os.path.exists(ov) else {}
    overrides[cid] = [T.tolist() for T in c["T"]]
    json.dump(overrides, open(ov, "w"))
    # cut-to-cut alignment involving this cut
    P["cross_field"] = {k: v for k, v in P.get("cross_field", {}).items() if cid not in (k[0], k[2])}
    r = 200 * MOSAIC_SCALE
    for o, oc in P["cuts"].items():
        if o == cid or (oc["mfp"], oc["section"]) != (c["mfp"], c["section"]):
            continue
        M, _ = _align_mosaics(cv2.imread(path("mosaics", cid + ".jpg")), cv2.imread(path("mosaics", o + ".jpg")))
        P["cross"][(cid, o)], P["cross"][(o, cid)] = M, np.linalg.inv(M)
        for a, b in ((cid, o), (o, cid)):
            for i, L in xcut_refine.refine(a, b, P["cross"][(a, b)]).items():
                P["cross_field"][(a, i, b)] = L
        h1 = hit(to_mosaic(P["cross"][(cid, o)], marks(cid)), marks(o), r)
        h2 = hit(to_mosaic(P["cross"][(o, cid)], marks(o)), marks(cid), r)
        ok = bool(len(marks(cid)) >= 5 and len(marks(o)) >= 5 and min(h1, h2) > 0.4)
        P.setdefault("cross_ok", {})[(cid, o)] = P["cross_ok"][(o, cid)] = ok
        print(f"re-aligned {cid} <-> {o}: marks agreement {h1:.0%}/{h2:.0%}, reliable={ok}", flush=True)
    pickle.dump(P, open(path("project.pkl"), "wb"))


def annotations(cid):
    return json.load(open(path("annotations", cid + ".json")))


def save_annotations(cid, data):
    tmp = path("annotations", cid + ".json.tmp")
    json.dump(data, open(tmp, "w"))
    os.replace(tmp, path("annotations", cid + ".json"))


def predictions(cid):
    p = path("predictions", cid + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def field_index(cid, field):
    return project()["cuts"][cid]["fields"].index(int(field))


def map_points(src_cid, src_field, dst_cid, dst_field, pts):
    """Map Nx2 full-res points from one field to another (same cut, or another cut of the same
    section via the mosaic alignment). Returns (mapped points, bool mask inside destination FOV)."""
    P = project()
    a, b = P["cuts"][src_cid], P["cuts"][dst_cid]
    pts = np.asarray(pts, float).reshape(-1, 2)
    if not len(pts):
        return pts, np.zeros(0, bool)
    i = field_index(src_cid, src_field)
    m = to_mosaic(a["F"][i], pts)
    if src_cid != dst_cid:  # per-field refined alignment when available (xcut_refine.py)
        m = to_mosaic(P.get("cross_field", {}).get((src_cid, i, dst_cid), P["cross"][(src_cid, dst_cid)]), m)
    j = field_index(dst_cid, dst_field)
    q = from_mosaic(b["F"][j], m)
    return q, inside_fov(b["masks"][j], q)

