"""Web app for browsing, counting and correcting duct marks.

usage: uv run ductcount serve [--port 8765]   ->  http://localhost:8765
"""
import csv, glob, io, json, os, subprocess, sys, time
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from . import project as pj
from .project import path, project, annotations, save_annotations, predictions, map_points, field_index
from .stitch import to_mosaic
from .evaluate import DEDUP_R

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    """index.html with versioned static URLs so the browser reloads CSS/JS after every change"""
    html = open(os.path.join(STATIC, "index.html")).read()
    for f in ("style.css", "app.js"):
        v = int(os.path.getmtime(os.path.join(STATIC, f)))
        html = html.replace(f"/static/{f}", f"/static/{f}?v={v}")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- data

def _cut_summary(cid):
    c = project()["cuts"][cid]
    ann = annotations(cid)
    pred = predictions(cid)
    fields = []
    for f, name in zip(c["fields"], c["names"]):
        pts = ann["fields"].get(str(f), [])
        mp = [p for p in (pred["fields"].get(str(f), []) if pred else []) if p["counted"]]
        fields.append(dict(field=f, name=name, manual=len(pts),
                           manual_atypical=sum(p["cls"] == "atypical" for p in pts),
                           model=len(mp) if pred else None,
                           model_atypical=sum(p["cls"] == "atypical" for p in mp) if pred else None,
                           reviewed=bool(ann["reviewed"].get(str(f)))))
    tot = lambda k: None if any(x[k] is None for x in fields) else sum(x[k] for x in fields)
    return dict(id=cid, mfp=c["mfp"], section=c["section"], cut=c["cut"], fields=fields,
                manual=tot("manual"), manual_atypical=tot("manual_atypical"),
                model=tot("model"), model_atypical=tot("model_atypical"),
                reviewed=sum(x["reviewed"] for x in fields), n_fields=len(fields),
                model_name=pred["model"] if pred else None)


@app.get("/api/summary")
def summary():
    return [_cut_summary(cid) for cid in sorted(project()["cuts"], key=lambda s: tuple(map(int, s.split("."))))]


@app.get("/api/cut/{cid}")
def cut(cid: str):
    if cid not in project()["cuts"]:
        raise HTTPException(404)
    c = project()["cuts"][cid]
    s = _cut_summary(cid)
    s["mosaic_size"] = c["mosaic_size"]
    s["mosaic_v"] = int(os.path.getmtime(path("mosaics", cid + ".jpg")))  # cache-busting after realignment
    s["F"] = [np.asarray(F).tolist() for F in c["F"]]
    s["others"] = sorted(k for k, v in project()["cuts"].items()
                         if (v["mfp"], v["section"]) == (c["mfp"], c["section"]) and k != cid)
    s["aligned"] = [k for k in s["others"] if project().get("cross_ok", {}).get((k, cid))]
    return s


@app.get("/api/field/{cid}/{field}")
def field(cid: str, field: int):
    """Marks of this field + marks from overlapping fields and from the other cuts, mapped here."""
    P = project()
    c = P["cuts"][cid]
    ann = annotations(cid)
    pred = predictions(cid)
    out = dict(points=ann["fields"].get(str(field), []), reviewed=bool(ann["reviewed"].get(str(field))),
               model=pred["fields"].get(str(field), []) if pred else [], same_cut=[], other_cuts=[])
    for f in c["fields"]:
        if f == field:
            continue
        pts = ann["fields"].get(str(f), [])
        q, ok = map_points(cid, f, cid, field, [(p["x"], p["y"]) for p in pts])
        out["same_cut"] += [dict(x=float(x), y=float(y), field=f, cls=p["cls"]) for (x, y), p, k in zip(q, pts, ok) if k]
    for o in P["cuts"]:
        oc = P["cuts"][o]
        if o == cid or (oc["mfp"], oc["section"]) != (c["mfp"], c["section"]) \
                or not P.get("cross_ok", {}).get((o, cid)):
            continue
        oann = annotations(o)
        for f in oc["fields"]:
            pts = oann["fields"].get(str(f), [])
            q, ok = map_points(o, f, cid, field, [(p["x"], p["y"]) for p in pts])
            out["other_cuts"] += [dict(x=float(x), y=float(y), cut=o, field=f, cls=p["cls"]) for (x, y), p, k in zip(q, pts, ok) if k]
    return out


@app.get("/api/corresponding/{cid}/{field}/{target}")
def corresponding(cid: str, field: int, target: str):
    """Field of cut `target` showing the same tissue as (cid, field): via the cut-to-cut alignment when
    it is reliable, else the same field number (clamped)."""
    P = project()
    t = P["cuts"][target]
    i = field_index(cid, field)
    if P.get("cross_ok", {}).get((cid, target)):
        c = np.array([[2016.0, 1520.0]])  # centre of the field of view, full res
        best = None
        for f in t["fields"]:
            q, ok = map_points(cid, field, target, f, c)
            if ok[0]:
                d = np.hypot(*(q[0] - c[0]))
                if best is None or d < best[0]:
                    best = (d, f)
        if best:
            return dict(field=best[1], index=t["fields"].index(best[1]), aligned=True)
    j = min(i, len(t["fields"]) - 1)
    return dict(field=t["fields"][j], index=j, aligned=False)


@app.put("/api/field/{cid}/{field}")
def save_field(cid: str, field: int, body: dict = Body(...)):
    ann = annotations(cid)
    if "points" in body:
        ann["fields"][str(field)] = [dict(id=str(p["id"]), x=float(p["x"]), y=float(p["y"]),
                                          cls=p.get("cls", "normal"), src=p.get("src", "manual")) for p in body["points"]]
    if "reviewed" in body:
        ann["reviewed"][str(field)] = bool(body["reviewed"])
    save_annotations(cid, ann)
    return _cut_summary(cid)


@app.put("/api/alignment/{cid}")
def save_alignment(cid: str, body: dict = Body(...)):
    """Manual field alignment from the overview editor; applied by a background job."""
    if cid not in project()["cuts"] or len(body["F"]) != len(project()["cuts"][cid]["fields"]):
        raise HTTPException(400)
    json.dump(dict(cut=cid, F=body["F"]), open(path("alignment_pending.json"), "w"))
    return start_job("realign")


@app.get("/api/mosaic_points/{cid}")
def mosaic_points(cid: str):
    """All manual and model marks of a cut in mosaic image pixels (for the overview)."""
    c = project()["cuts"][cid]
    ann, pred = annotations(cid), predictions(cid)
    man, mod = [], []
    for i, f in enumerate(c["fields"]):
        pts = ann["fields"].get(str(f), [])
        if pts:
            for (x, y), p in zip(to_mosaic(c["F"][i], [(p["x"], p["y"]) for p in pts]), pts):
                man.append(dict(x=float(x), y=float(y), field=f, cls=p["cls"]))
        mp = [p for p in (pred["fields"].get(str(f), []) if pred else []) if p["counted"]]
        if mp:
            for (x, y), p in zip(to_mosaic(c["F"][i], [(p["x"], p["y"]) for p in mp]), mp):
                mod.append(dict(x=float(x), y=float(y), field=f, cls=p["cls"], score=p["score"]))
    return dict(manual=man, model=mod)


# ---------------------------------------------------------------- images

@app.get("/img/mosaic/{cid}")
def img_mosaic(cid: str):
    return FileResponse(path("mosaics", cid + ".jpg"), headers={"Cache-Control": "max-age=3600"})


@app.get("/img/field/{cid}/{field}")
def img_field(cid: str, field: int, s: int = 2):
    """s=2: half resolution (2016x1520), s=8: thumbnail."""
    c = project()["cuts"][cid]
    os.makedirs(path("tiles"), exist_ok=True)
    out = path("tiles", f"{cid}.{field}.s{s}.jpg")
    if not os.path.exists(out):
        im = cv2.imread(c["paths"][field_index(cid, field)])
        im = cv2.resize(im, None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(out, im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return FileResponse(out, headers={"Cache-Control": "max-age=86400"})


# ---------------------------------------------------------------- training target / model heatmap

def _overlay_png(hm, rgb, out):
    """probability map (0..1, quarter resolution) -> transparent PNG in one colour"""
    a = np.clip(hm, 0, 1)
    img = np.zeros((*a.shape, 4), np.uint8)
    img[..., :3] = rgb[::-1]                       # BGR for OpenCV
    img[..., 3] = (a ** 0.7 * 230).astype(np.uint8)
    cv2.imwrite(out, img)


@app.get("/img/target/{cid}/{field}")
def img_target(cid: str, field: int):
    """What the model is trained on for this field: a Gaussian at every duct mark, including marks
    made in overlapping fields of the same cut (propagated), exactly as in jobs.training_items."""
    from .unet import heatmap
    from .register import SCALE
    c = project()["cuts"][cid]
    i = field_index(cid, field)
    ann = annotations(cid)
    pts = [map_points(cid, f, cid, field, [(p["x"], p["y"]) for p in ann["fields"].get(str(f), [])])
           for f in c["fields"] if ann["reviewed"].get(str(f))]
    q = np.vstack([p[ok] for p, ok in pts if len(p)]) if pts else np.zeros((0, 2))
    h, w = 760, 1008
    hm = heatmap((h, w), q * SCALE)
    os.makedirs(path("tiles"), exist_ok=True)
    out = path("tiles", f"target.{cid}.{field}.png")
    _overlay_png(np.clip(hm * 1.6, 0, 1), (0, 220, 255), out)  # cyan, distinct from the blue marks
    return FileResponse(out, headers={"Cache-Control": "no-store"})


_models = {}


@app.get("/img/heatmap/{cid}/{field}")
def img_heatmap(cid: str, field: int):
    """Model confidence map for this field (out-of-fold models for Ambre's mice), cached per model."""
    from .unet import load, predict_heatmap
    if not os.path.exists(path("models", "registry.json")):
        raise HTTPException(404, "no model yet")
    reg = json.load(open(path("models", "registry.json")))
    info = reg["models"][reg["current"]]
    out = path("tiles", f"heat.{reg['current']}.{cid}.{field}.png")
    if not os.path.exists(out):
        mfp = project()["cuts"][cid]["mfp"]
        files = [f for f in info["files"] if f"holdout_mfp{mfp}" in f] or \
                reg.get("for_new_mice", {}).get("files") or info["files"]
        for f in files:
            if f not in _models:
                _models[f] = load(path("models", f))
        p = project()["cuts"][cid]["paths"][field_index(cid, field)]
        hm = np.mean([predict_heatmap(_models[f], p) for f in files], 0)
        os.makedirs(path("tiles"), exist_ok=True)
        _overlay_png(hm / max(info.get("thr", 0.15) * 3, 1e-3), (230, 176, 0), out)
    return FileResponse(out, headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- export

@app.get("/api/export/fields.csv")
def export_fields():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["mouse", "section", "cut", "field", "image", "manual", "manual_atypical", "model", "model_atypical", "reviewed"])
    for s in summary():
        for f in s["fields"]:
            w.writerow([s["mfp"], s["section"], s["cut"], f["field"], f["name"], f["manual"], f["manual_atypical"],
                        f["model"], f["model_atypical"], f["reviewed"]])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=duct_counts_per_field.csv"})


@app.get("/api/export/sections.csv")
def export_sections():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["mouse", "section", "cut", "fields", "manual", "manual_atypical", "model", "model_atypical"])
    for s in summary():
        w.writerow([s["mfp"], s["section"], s["cut"], s["n_fields"], s["manual"], s["manual_atypical"], s["model"], s["model_atypical"]])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=duct_counts_per_cut.csv"})


# ---------------------------------------------------------------- jobs

_procs = {}


def _pid_alive(pidfile):
    """jobs survive a server restart: check the process recorded at launch"""
    try:
        os.kill(int(open(pidfile).read()), 0)
        return True
    except (OSError, ValueError):
        return False


def _jobs():
    os.makedirs(path("jobs"), exist_ok=True)
    out = []
    for f in sorted(glob.glob(path("jobs", "*.log")), reverse=True)[:10]:
        jid = os.path.basename(f)[:-4]
        p = _procs.get(jid)
        lines = open(f).read().splitlines()
        lines = [l for l in lines if "TIFF_Warning" not in l]
        if (p is not None and p.poll() is None) or (p is None and _pid_alive(f[:-4] + ".pid")):
            status = "running"
        elif any("Traceback" in l for l in lines) or (p is not None and p.returncode != 0):
            status = "failed"
        else:
            status = "done"
        out.append(dict(id=jid, kind=jid.split("-")[-1], status=status, log=lines[-12:]))
    return out


@app.get("/api/jobs")
def jobs():
    reg = json.load(open(path("models", "registry.json"))) if os.path.exists(path("models", "registry.json")) else None
    return dict(jobs=_jobs(), model=reg.get("current") if reg else None, models=reg["models"] if reg else {})


@app.post("/api/jobs/{kind}")
def start_job(kind: str):
    if kind not in ("infer", "train", "realign"):
        raise HTTPException(400)
    if any(p.poll() is None for p in _procs.values()) or any(j["status"] == "running" for j in _jobs()):
        raise HTTPException(409, "a job is already running")
    jid = time.strftime("%Y%m%d-%H%M%S-") + kind
    log = open(path("jobs", jid + ".log"), "w")
    # epochs: jobs.py picks fine-tuning (current model) or from-scratch (no model yet)
    _procs[jid] = subprocess.Popen([sys.executable, "-u", "-m", "ductcount.jobs", kind],
                                   stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "DUCT_DATA": pj.DATA})
    open(path("jobs", jid + ".pid"), "w").write(str(_procs[jid].pid))
    return dict(id=jid)


def run(port=8765, host="127.0.0.1"):
    import uvicorn
    os.makedirs(path("jobs"), exist_ok=True)
    print(f"Duct Counter: http://localhost:{port}  (data: {pj.DATA})", flush=True)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run(int(os.environ.get("PORT", 8765)))
