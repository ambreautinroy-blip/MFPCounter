"""Classical baseline: hematoxylin colour deconvolution + blob (connected component) detection.

Ducts are rings of densely packed epithelial nuclei, i.e. compact blobs of strong
hematoxylin; adipose tissue has only sparse nuclei. Touching ducts form a single
component, which matches the 'touching ducts count as one' rule."""
import itertools
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.color import rgb2hed
from skimage.measure import regionprops, label
from .register import load_small, fov_mask, SCALE

_cache = {}


def hematoxylin(path):
    if path not in _cache:
        im = load_small(path)
        h = rgb2hed(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))[..., 0]
        h = ndi.gaussian_filter(h, 2).astype(np.float32)
        h[fov_mask(im, 31) == 0] = 0
        _cache[path] = h
    return _cache[path]


def detect(path, thr=0.06, min_area=150, close=7):
    h = hematoxylin(path)
    m = (h > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    m = ndi.binary_fill_holes(m)
    pts = [r.centroid[::-1] for r in regionprops(label(m)) if r.area >= min_area]
    return np.array(pts).reshape(-1, 2) / SCALE


GRID = dict(thr=[0.03, 0.04, 0.05, 0.06, 0.08, 0.10], min_area=[50, 100, 200, 400], close=[3, 7, 11])


def grid():
    keys = list(GRID)
    for v in itertools.product(*GRID.values()):
        yield dict(zip(keys, v))
