"""Stitch every cut once and cache transforms + completed labels (data/cache/stitch.pkl).

Completed labels: Ambre marks a duct in only one of the overlapping fields. For training we
propagate each mark into every other field whose FOV contains it."""
import os, pickle
import numpy as np
from .config import path
from .gt import fields
from .stitch import stitch, to_mosaic, from_mosaic, inside_fov, merge

OUT = path("cache", "stitch.pkl")
DEDUP_R = 60  # px full res; two marks closer than this in mosaic = same duct

def build(log=print):
    f = fields()
    f = f[~f.dup].sort_values(["mfp", "section", "cut", "field"]).reset_index(drop=True)
    res = {}
    for key, g in f.groupby(["mfp", "section", "cut"]):
        T, masks = stitch(list(g.path))
        allm = [to_mosaic(T[i], p) for i, p in enumerate(g.points) if T[i] is not None and p]
        allm = np.vstack(allm) if allm else np.zeros((0, 2))
        uniq = merge(allm, DEDUP_R)
        complete = []
        for i in range(len(g)):
            own = np.asarray(g.points.iloc[i] or [], float).reshape(-1, 2)
            if T[i] is None:
                complete.append(own); continue
            loc = from_mosaic(T[i], uniq)
            loc = loc[inside_fov(masks[i], loc)]
            # keep own marks + propagated marks not already near an own mark
            extra = [p for p in loc if not len(own) or np.min(np.hypot(*(own - p).T)) > DEDUP_R]
            complete.append(np.vstack([own, np.array(extra).reshape(-1, 2)]))
        res[key] = dict(fields=list(g.field), paths=list(g.path), T=T, masks=masks,
                        labelled=[p is not None for p in g.points],
                        own=[np.asarray(p or [], float).reshape(-1, 2) for p in g.points],
                        complete=complete, unique_mosaic=uniq)
        ngroups = len({int(round(t[0, 2] / 1e6)) for t in T})
        log(f"cut {'.'.join(map(str, key))}: {len(g)} fields stitched into {ngroups} group(s), "
            f"{int(g.n.fillna(0).sum())} manual marks ({len(uniq)} unique)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pickle.dump(res, open(OUT, "wb"))
    return res


if __name__ == "__main__":
    build()
