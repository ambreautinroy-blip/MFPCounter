"""Register microscope fields (within a cut, and across serial cuts) with SIFT + RANSAC."""
import hashlib, os
import cv2
from .config import path
import numpy as np

SCALE = 0.25


SMALL = path("cache", "small")


def load_small(path, scale=SCALE):
    """Downscaled image, cached on disk as PNG (decoding the 12 MP originals is slow)."""
    c = os.path.join(SMALL, hashlib.md5(f"{os.path.abspath(path)}{scale}".encode()).hexdigest() + ".png")
    if os.path.exists(c):
        return cv2.imread(c, cv2.IMREAD_COLOR)
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    os.makedirs(SMALL, exist_ok=True)
    cv2.imwrite(c, im)
    return im


def fov_mask(im, erode=15):
    """Circular field of view: everything that is not the black vignette."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    m = (g > 40).astype(np.uint8)
    # dark tissue (dense nuclei) is also < 40: take the convex hull of the bright disc
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    hull = cv2.convexHull(np.column_stack(np.where(lab == big)[::-1]).astype(np.int32))
    m = np.zeros_like(m)
    cv2.fillConvexPoly(m, hull, 1)
    return cv2.erode(m, np.ones((erode, erode), np.uint8))


_sift = cv2.SIFT_create(nfeatures=4000)


def features(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    return _sift.detectAndCompute(g, fov_mask(im))


def register(a, b, fa=None, fb=None, ratio=0.75):
    """Similarity transform mapping points of image a into image b (both at SCALE).
    Returns (2x3 matrix or None, n_inliers)."""
    ka, da = fa or features(a)
    kb, db = fb or features(b)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return None, 0
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < ratio * x[1].distance]
    if len(good) < 8:
        return None, len(good)
    pa = np.float32([ka[g.queryIdx].pt for g in good])
    pb = np.float32([kb[g.trainIdx].pt for g in good])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None:
        return None, 0
    return M, int(inl.sum())


def apply(M, pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    return pts @ M[:, :2].T + M[:, 2]
