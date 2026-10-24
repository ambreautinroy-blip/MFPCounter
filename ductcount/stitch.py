"""Stitch the fields of one cut into a common mosaic frame.

Fields overlap heavily and Ambre marks each duct in only one field, so counts must be
deduplicated in mosaic coordinates. All coordinates here are full resolution."""
import itertools
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from .register import load_small, features, register, fov_mask, SCALE

MIN_INLIERS = 20


def _ok(M):
    """same magnification and a shift smaller than the field of view (fields must overlap)"""
    s = np.hypot(*M[:, 0])
    return 0.9 < s < 1.1 and np.hypot(*M[:, 2]) < 700


def stitch(paths):
    """Return list of 3x3 full-res transforms field -> mosaic, plus the FOV masks at SCALE.
    Fields that do not register to any other field get their own isolated frame."""
    ims = [load_small(p) for p in paths]
    fs = [features(i) for i in ims]
    n = len(paths)
    W = np.zeros((n, n))
    pair = {}
    for i, j in itertools.combinations(range(n), 2):
        M, k = register(ims[i], ims[j], fs[i], fs[j])
        if M is not None and k >= MIN_INLIERS and _ok(M):
            W[i, j] = W[j, i] = 1.0 / k  # MST on 1/inliers -> keep strongest links
            pair[i, j] = M
    T = [None] * n
    ncomp, lab = connected_components(W > 0, directed=False)
    mst = minimum_spanning_tree(W).toarray()
    adj = (mst + mst.T) > 0
    for comp in range(ncomp):
        members = np.where(lab == comp)[0]
        root = members[0]
        # each unconnected group of fields gets its own frame, far from the others
        T[root] = np.eye(3); T[root][0, 2] = comp * 1e6
        stack = [root]
        while stack:
            i = stack.pop()
            for j in np.where(adj[i])[0]:
                if T[j] is not None:
                    continue
                # M maps i -> j (at SCALE); we need j -> mosaic = (i -> mosaic) o (j -> i)
                M = pair[(i, j)] if (i, j) in pair else None
                A = np.vstack([M, [0, 0, 1]]) if M is not None else np.linalg.inv(np.vstack([pair[(j, i)], [0, 0, 1]]))
                A_full = A.copy(); A_full[:2, 2] /= SCALE
                T[j] = T[i] @ np.linalg.inv(A_full)
                stack.append(j)
    masks = [fov_mask(im, 31) for im in ims]
    return T, masks


def to_mosaic(T, pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    return pts @ T[:2, :2].T + T[:2, 2]


def from_mosaic(T, pts):
    return to_mosaic(np.linalg.inv(T), pts)


def inside_fov(mask, pts):
    """pts full-res in field coords -> bool array"""
    pts = np.asarray(pts, float).reshape(-1, 2) * SCALE
    H, W = mask.shape
    x, y = pts[:, 0].astype(int), pts[:, 1].astype(int)
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    ok[ok] = mask[y[ok], x[ok]] > 0
    return ok


def merge(points, radius):
    """Greedy cluster of mosaic points within radius -> cluster centers."""
    pts = np.asarray(points, float).reshape(-1, 2)
    used = np.zeros(len(pts), bool)
    out = []
    for i in range(len(pts)):
        if used[i]:
            continue
        d = np.hypot(*(pts - pts[i]).T)
        grp = (d < radius) & ~used
        used |= grp
        out.append(pts[grp].mean(0))
    return np.array(out).reshape(-1, 2)
