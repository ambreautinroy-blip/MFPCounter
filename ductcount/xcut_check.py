"""Decide which pairs of serial cuts are reliably aligned (used by the app's 'other cuts' layer).

Check: after mapping cut A onto cut B, how many of A's ducts land near one of B's ducts, compared
with random shifts of the same point set. Ducts = manual marks when there are enough, else the
model's detections; without either, fall back to the image-correlation score of the alignment
(marked 'unverified')."""
import pickle
import numpy as np
from . import project as pj
from .project import project, annotations, predictions
from .stitch import to_mosaic

MIN_POINTS = 5
SCORE_OK = 0.6   # correlation score accepted when no ducts are available to verify the alignment


def marks(cid, source="manual"):
    """Ducts of a cut in mosaic-image pixels, from manual marks or model detections."""
    c = project()["cuts"][cid]
    if source == "manual":
        ann = annotations(cid)
        pts = [[(p["x"], p["y"]) for p in ann["fields"].get(str(f), [])] for f in c["fields"]]
    else:
        pred = predictions(cid)
        pts = [[(p["x"], p["y"]) for p in (pred["fields"].get(str(f), []) if pred else []) if p["counted"]]
               for f in c["fields"]]
    out = [to_mosaic(c["F"][i], q) for i, q in enumerate(pts)]
    return np.vstack(out) if out else np.zeros((0, 2))


def hit(a, b, r):
    return np.mean([np.min(np.hypot(*(b - p).T)) < r for p in a]) if len(a) and len(b) else np.nan


def flag_reliable(log=print):
    P = project()
    r = 200 * pj.MOSAIC_SCALE  # 200 px full resolution
    rng = np.random.default_rng(0)
    ok, how = {}, {}
    for (a, b), M in P["cross"].items():
        for source in ("manual", "model"):
            A, B = to_mosaic(M, marks(a, source)), marks(b, source)
            if len(A) >= MIN_POINTS and len(B) >= MIN_POINTS:
                h = hit(A, B, r)
                rand = np.nanmean([hit(A + rng.normal(0, 60, 2), B, r) for _ in range(20)])
                ok[(a, b)], how[(a, b)] = bool(h > 0.4 and h > 1.5 * rand), source
                break
        else:
            ok[(a, b)] = P.get("cross_score", {}).get((a, b), 0) >= SCORE_OK
            how[(a, b)] = "unverified"
    for (a, b) in list(ok):  # symmetric
        ok[(a, b)] = ok[(b, a)] = ok[(a, b)] and ok[(b, a)]
    P["cross_ok"], P["cross_how"] = ok, how
    pickle.dump(P, open(pj.path("project.pkl"), "wb"))
    good = sorted(f"{a}-{b}" for (a, b), v in ok.items() if v and a < b)
    log(f"cut-to-cut alignment reliable for {len(good)} of {len(ok) // 2} pairs: {', '.join(good) or '-'}")
    return ok


if __name__ == "__main__":
    flag_reliable()
