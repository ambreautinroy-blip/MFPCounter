"""Compare two sets of app predictions against the manual marks, per cut and per mouse.
usage: uv run python -m ductcount.compare_predictions BEFORE_DIR data/predictions"""
import json, os, sys
import pandas as pd
from .project import annotations, project

before, after = sys.argv[1], sys.argv[2]


def counted(d, cid):
    p = json.load(open(os.path.join(d, cid + ".json")))
    return sum(q["counted"] for v in p["fields"].values() for q in v)


rows = []
for cid in sorted(project()["cuts"], key=lambda s: tuple(map(int, s.split(".")))):
    man = sum(len(v) for v in annotations(cid)["fields"].values())
    rows.append(dict(cut=cid, mouse=int(cid[0]), manual=man, before=counted(before, cid), after=counted(after, cid)))
df = pd.DataFrame(rows)
df["err_before"] = (df.before - df.manual).abs()
df["err_after"] = (df.after - df.manual).abs()
print(df.to_string(index=False))
g = df.groupby("mouse")[["manual", "err_before", "err_after"]].sum()
g["rel_before"] = (g.err_before / g.manual).map("{:.1%}".format)
g["rel_after"] = (g.err_after / g.manual).map("{:.1%}".format)
print(g.to_string())
print(f"all: {df.err_before.sum() / df.manual.sum():.1%} -> {df.err_after.sum() / df.manual.sum():.1%}")
