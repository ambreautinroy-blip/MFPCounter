"""Refine the cut-to-cut alignment locally, one field at a time.

Serial slices deform slightly, so a single similarity per pair of cuts is only approximate.
For every field of cut A we re-run ECC (euclidean) on a window around that field, starting from
the global alignment, and keep the local transform if it improves the duct-mark agreement."""
import pickle
import cv2
import numpy as np
from . import project as pj
from .project import project, annotations
from .stitch import to_mosaic
from .xcut_check import marks, hit

FOV_C, FOV_R = np.array([2016.0, 1520.0]), 1450.0


def sat(img):
    s = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    return cv2.GaussianBlur(s, (0, 0), 1.5)


def field_marks(cid, i):
    c = project()["cuts"][cid]
    f = c["fields"][i]
    pts = annotations(cid)["fields"].get(str(f), [])
    return to_mosaic(c["F"][i], [(p["x"], p["y"]) for p in pts])


def refine(a, b, M):
    P = project()
    A, B = cv2.imread(pj.path("mosaics", a + ".jpg")), cv2.imread(pj.path("mosaics", b + ".jpg"))
    sA, sB = sat(A), sat(B)
    H, W = sB.shape
    Aw = cv2.warpAffine(sA, M[:2], (W, H))
    out = {}
    Bm = marks(b)
    r = 200 * pj.MOSAIC_SCALE
    for i, F in enumerate(P["cuts"][a]["F"]):
        c = to_mosaic(M, to_mosaic(F, FOV_C))[0]
        rad = FOV_R * pj.MOSAIC_SCALE * 1.3
        mask = np.zeros((H, W), np.uint8)
        cv2.circle(mask, (int(c[0]), int(c[1])), int(rad), 255, -1)
        if mask.sum() == 0:
            continue
        try:
            _, Wc = cv2.findTransformECC(sB, Aw, np.eye(2, 3, dtype=np.float32), cv2.MOTION_EUCLIDEAN,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), mask, 5)
        except cv2.error:
            continue
        # Wc maps B coords -> Aw coords; the local correction Aw -> B is its inverse
        L = np.vstack([cv2.invertAffineTransform(Wc), [0, 0, 1]]) @ M
        if np.hypot(*(L[:2, 2] - M[:2, 2])) > rad:      # implausible jump
            continue
        Am = field_marks(a, i)
        if len(Am) >= 3 and len(Bm) >= 3:
            if hit(to_mosaic(L, Am), Bm, r) < hit(to_mosaic(M, Am), Bm, r):
                continue
        out[i] = L
    return out


def refine_all(log=print):
    P = project()
    P["cross_field"] = {}
    n = 0
    for (a, b), M in sorted(P["cross"].items()):
        loc = refine(a, b, M)
        for i, L in loc.items():
            P["cross_field"][(a, i, b)] = L
        n += len(loc)
    pickle.dump(P, open(pj.path("project.pkl"), "wb"))
    log(f"cut-to-cut alignment refined locally for {n} fields")


if __name__ == "__main__":
    refine_all()
