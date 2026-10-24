"""Headless UI smoke test: screenshots of overview and field views, plus a click/undo round-trip.
usage: uv run python -m ductcount.ui_check OUTDIR   (app running on :8765)"""
import sys, json
from playwright.sync_api import sync_playwright

out = sys.argv[1]
URL = "http://127.0.0.1:8765/"
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport={"width": 1600, "height": 950}, device_scale_factor=1)
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: m.type == "error" and errs.append(m.text))
    pg.goto(URL + "#3.2.1"); pg.wait_for_timeout(2500)
    pg.screenshot(path=f"{out}/ui_overview.png")
    pg.click("#view-switch button[data-view=summary]"); pg.wait_for_timeout(800)
    pg.screenshot(path=f"{out}/ui_summary.png", full_page=True)
    pg.click("#summary tr.cut >> nth=12"); pg.wait_for_timeout(2000)
    print("after summary click:", pg.evaluate("[S.view, S.cut.id]"))
    pg.click("#view-switch button[data-view=field]"); pg.wait_for_timeout(2000)
    pg.screenshot(path=f"{out}/ui_field.png")
    pg.keyboard.press("ArrowRight"); pg.keyboard.press("ArrowRight"); pg.wait_for_timeout(1500)
    pg.check("#l-cuts"); pg.wait_for_timeout(300)
    pg.screenshot(path=f"{out}/ui_field3.png")
    pg.keyboard.press("ArrowLeft"); pg.keyboard.press("ArrowLeft"); pg.wait_for_timeout(1500)
    # add a mark in the middle of the canvas, check count increments, then remove it
    before = pg.evaluate("S.field.points.length")
    box = pg.locator("#canvas").bounding_box()
    pg.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2); pg.wait_for_timeout(800)
    after = pg.evaluate("S.field.points.length")
    pg.screenshot(path=f"{out}/ui_field_added.png")
    pg.keyboard.press("Delete"); pg.wait_for_timeout(800)
    final = pg.evaluate("S.field.points.length")
    print("current field:", pg.evaluate("[S.fi, S.field.fieldNo, S.cut.fields[S.fi].manual]"))
    saved = json.loads(pg.evaluate("fetch(`/api/field/3.2.1/${S.field.fieldNo}`).then(r=>r.text())"))
    print("marks before/after add/after delete:", before, after, final, "| saved on server:", len(saved["points"]))
    print("errors:", errs)
    b.close()
