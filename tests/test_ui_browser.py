"""The web pages in a real (headless) browser: the main flows end to end. Skipped where Chromium or ffmpeg is missing."""
import shutil
import socket
import threading
import time

import numpy as np
import pytest

pytest.importorskip("starlette")
uvicorn = pytest.importorskip("uvicorn")
sync_api = pytest.importorskip("playwright.sync_api")
pytest.importorskip("cv2")
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not installed")

from vidlabel import Dataset  # noqa: E402
from vidlabel.probe import probe_video  # noqa: E402
from vidlabel.ui.app import create_app  # noqa: E402

from test_ui_server import make_video  # noqa: E402


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp("ui")
    vp = make_video(root / "clip.mp4", n=60)
    ds = Dataset.create(root / "d")
    ds.add_video(probe_video(vp, "clip"))
    ds.add_labels(["person", "car"])
    rows = [{"kind": "box", "label": lab, "start_frame": f, "box": (5 + t * 10, 10, 35 + t * 10, 40), "track_id": t, "origin": "model",
             "confidence": 0.5 + 0.1 * t} for t, lab in ((1, "person"), (2, "person"), (3, "car")) for f in range(5, 40)]
    m = np.zeros((64, 96), bool)
    m[30:50, 60:90] = True
    rows += [{"kind": "mask", "label": "car", "start_frame": 10, "mask": m, "origin": "human"}]
    ds.add_annotations(rows, "clip")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = create_app(root / "d", token="tk", port=port)
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    while not srv.started:
        time.sleep(0.05)
    yield ds, f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(5)


@pytest.fixture
def page(server):
    ds, base = server
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:                                   # browser not installed
            pytest.skip(f"no Chromium: {e}")
        pg = b.new_page(viewport={"width": 1280, "height": 800})
        pg.errors = []
        pg.on("pageerror", lambda e: pg.errors.append(str(e)))
        pg.on("console", lambda m: pg.errors.append(m.text) if m.type == "error" else None)
        pg.goto(f"{base}/?token=tk")
        pg.evaluate("localStorage.setItem('vl_author','ana')")
        pg.goto(f"{base}/#/overview")
        pg.reload()
        yield pg, ds, base
        b.close()


def test_overview_shows_counts(page):
    pg, ds, base = page
    pg.wait_for_selector(".figures")
    assert "Labels" in pg.inner_text("main") and "clip" in pg.inner_text("main")
    assert not pg.errors


def test_viewer_shows_frame_boxes_and_mask_and_steps(page):
    pg, ds, base = page
    pg.goto(f"{base}/#/video/clip?f=10")
    pg.reload()
    pg.wait_for_selector(".annrow")
    assert pg.locator(".annrow").count() == 4                    # three tracks and one mask on frame 10
    pixels = pg.evaluate("""() => { const c = document.querySelector('.stage canvas'); const x = c.getContext('2d');
        const d = x.getImageData(70, 40, 1, 1).data; return [c.width, c.height, d[0], d[1], d[2]]; }""")
    assert pixels[:2] == [96, 64] and pixels[2:] != [60, 60, 60]  # the mask tint changed the pixel at (70, 40)
    pg.keyboard.press("ArrowRight")
    pg.wait_for_function("document.body.innerText.includes('frame 11 /')")
    pg.keyboard.press("Shift+ArrowRight")
    pg.wait_for_function("document.body.innerText.includes('frame 21 /')")
    assert not pg.errors


def test_review_commit_history_revert_and_conflict(page):
    pg, ds, base = page
    before = ds._ann_path("clip").read_bytes()
    pg.goto(f"{base}/#/review/clip")
    pg.reload()
    pg.wait_for_selector("tbody tr")
    pg.click("tbody tr:first-child button:has-text('Reject')")
    pg.wait_for_function("document.querySelector('#changes-count').textContent !== '0'")
    assert ds._ann_path("clip").read_bytes() == before            # a draft changes nothing
    pg.click("#changes-btn")
    pg.click("#drawer button:has-text('Commit to dataset')")
    pg.wait_for_selector(".toast:has-text('Write a short message')")
    pg.fill("#drawer textarea", "reject first track")
    pg.click("#drawer button:has-text('Commit to dataset')")
    pg.wait_for_selector(".toast:has-text('Committed')")
    assert ds.annotations("clip", status="rejected").num_rows == 35
    pg.goto(f"{base}/#/history")
    pg.wait_for_selector("tbody tr:has-text('reject first track')")
    pg.click("tbody tr:first-child button:has-text('Revert')")
    pg.click(".modal button.primary")
    pg.wait_for_selector(".toast:has-text('Reverted')")
    assert ds.annotations("clip", status="rejected").num_rows == 0
    # a conflict: edit in the page, change the dataset behind its back, then commit
    pg.goto(f"{base}/#/review/clip")
    pg.reload()
    pg.wait_for_selector("tbody tr")
    pg.click("tbody tr:nth-child(2) button:has-text('Approve')")
    pg.wait_for_function("document.querySelector('#changes-count').textContent !== '0'")
    ds.update_track("clip", 1, status="approved")
    pg.click("#changes-btn")
    pg.fill("#drawer textarea", "approve second")
    pg.click("#drawer button:has-text('Commit to dataset')")
    pg.wait_for_selector("#drawer button:has-text('Apply my edits on top')")
    pg.click("#drawer button:has-text('Apply my edits on top')")
    pg.wait_for_selector(".toast:has-text('Committed')")
    assert ds.annotations("clip", track_id=2).column("status").unique().to_pylist() == ["approved"]
    assert [e for e in pg.errors if "409" not in e] == []             # the 409 is the conflict this test caused on purpose


def test_validation_and_release_pages(page):
    pg, ds, base = page
    pg.goto(f"{base}/#/validation")
    pg.reload()
    pg.wait_for_selector("text=quick check")
    pg.click("button:has-text('Run full check')")
    pg.wait_for_selector("text=full check, masks decoded", timeout=20000)
    pg.goto(f"{base}/#/releases")
    pg.click("button:has-text('Build release')")
    pg.wait_for_selector(".notice:has-text('Built')", timeout=20000)
    pg.wait_for_selector("tbody tr:has-text('latest')")
    assert not pg.errors


def test_builder_pages_labels_views_derive_and_add_video(page, tmp_path):
    pg, ds, base = page
    pg.goto(f"{base}/#/labels")
    pg.reload()
    pg.fill("input[type=text]", "bike, bus")
    pg.click("button:has-text('Add labels')")
    pg.wait_for_selector(".toast:has-text('Labels added')")
    pg.wait_for_selector("td:has-text('bike')")
    assert {"bike", "bus"} <= {l.name for l in ds.vocab.labels}
    # create a view by typing the region, then copy whole-frame rows into it with a preview first
    pg.goto(f"{base}/#/views")
    pg.reload()
    pg.wait_for_selector("text=Create a view")
    pg.fill("input[placeholder^='view name']", "left_half")
    pg.fill("input[type=number] >> nth=0", "0")
    pg.fill("input[type=number] >> nth=1", "0")
    pg.fill("input[type=number] >> nth=2", "48")
    pg.fill("input[type=number] >> nth=3", "64")
    pg.click("button:has-text('Create view')")
    pg.wait_for_selector(".toast:has-text('View created')")
    assert "left_half" in ds.view_ids()
    pg.click("button:has-text('Copy labels in')")
    pg.click(".modal button:has-text('Preview')")
    pg.wait_for_selector(".modal .notice:has-text('would be copied')")
    assert ds.annotations("clip", view_id=["left_half"]).num_rows == 0       # a preview writes nothing
    pg.click(".modal button:has-text('Copy rows')")
    pg.wait_for_selector(".modal .notice:text-matches('^Copied')")
    assert ds.annotations("clip", view_id=["left_half"]).num_rows > 0
    # add a second video from the Videos page, then check it
    vp = make_video(tmp_path / "second.mp4", n=30)
    pg.goto(f"{base}/#/browse")
    pg.reload()
    pg.fill("input[placeholder='path to a video file']", str(vp))
    pg.fill("input[placeholder^='name']", "second")
    pg.click("button:has-text('Add video')")
    pg.wait_for_selector(".notice:has-text('Added second')", timeout=30000)
    assert "second" in ds.video_ids()
    pg.click("tr:has-text('second') button:has-text('Check')")
    pg.click(".modal button:has-text('Run check')")
    pg.wait_for_selector(".modal :text('matches the dataset')", timeout=30000)
    assert not pg.errors


def test_overview_score_has_a_band_per_video(page):
    pg, ds, base = page
    pg.goto(f"{base}/#/overview")
    pg.reload()
    pg.wait_for_selector(".score-row canvas")
    assert pg.locator(".score-row").count() >= 1
    assert not pg.errors


def test_file_picker_lists_folders_and_videos_and_accepts_quoted_paths(page):
    pg, ds, base = page
    root = ds.path.parent
    pg.goto(f"{base}/#/browse")
    pg.reload()
    pg.fill("input[placeholder='path to a video file']", f'"{root}"')       # pasted with quotes, as Windows "Copy as path" does
    pg.click("button:has-text('Browse')")
    pg.wait_for_selector(".picker div:has-text('clip.mp4')")
    assert "[object" not in pg.inner_text(".modal")
    assert pg.locator(".picker div:has-text('d')").count() >= 1             # the dataset folder is listed too
    pg.click(".picker div:has-text('clip.mp4')")
    pg.click(".modal button:has-text('Use this file')")
    assert pg.input_value("input[placeholder='path to a video file']").endswith("clip.mp4")
    assert not pg.errors


# ---------------------------------------------------------------- drawing tools
def _viewer(pg, base, frame):
    pg.goto(f"{base}/#/video/clip?f={frame}")
    pg.reload()
    pg.wait_for_selector("canvas.ov")
    pg.wait_for_function("document.body.innerText.includes('frame %d /')" % frame)


def _ov(pg):
    b = pg.locator("canvas.ov").bounding_box()
    return lambda fx, fy: (b["x"] + b["width"] * fx, b["y"] + b["height"] * fy)


def _draft_count(pg):
    return int(pg.inner_text("#changes-count"))


def _commit(pg, message):
    pg.click("#commit-btn")
    pg.fill("#drawer textarea", message)
    pg.click("#drawer button:has-text('Commit to dataset')")
    pg.wait_for_selector(".toast:has-text('Committed')")


def test_draw_move_and_resize_a_box_then_commit(page):
    pg, ds, base = page
    _viewer(pg, base, 45)                                     # a frame with no boxes yet
    before = ds.annotations("clip", frames=(45, 45), kind="box").num_rows
    at = _ov(pg)
    pg.keyboard.press("b")
    pg.mouse.move(*at(0.10, 0.10))
    pg.mouse.down()
    pg.mouse.move(*at(0.30, 0.30), steps=4)
    pg.mouse.move(*at(0.50, 0.50), steps=4)
    pg.mouse.up()
    pg.wait_for_function("document.querySelector('#changes-count').textContent === '1'")
    assert ds.annotations("clip", frames=(45, 45), kind="box").num_rows == before          # a draft changes nothing
    pg.wait_for_selector(".annrow.pend")
    pg.keyboard.press("v")
    pg.mouse.move(*at(0.30, 0.30))                            # inside the new box: drag it right by 20% of the width
    pg.mouse.down()
    pg.mouse.move(*at(0.40, 0.30), steps=4)
    pg.mouse.move(*at(0.50, 0.30), steps=4)
    pg.mouse.up()
    pg.wait_for_function("V.rows.some(r => r.pending === 'added' && r.x0 > 25)")          # the page has the moved box
    pg.mouse.move(*at(0.70, 0.50))                            # the bottom-right corner handle of the moved box: pull it out to 90%
    pg.mouse.down()
    pg.mouse.move(*at(0.80, 0.50), steps=4)
    pg.mouse.move(*at(0.90, 0.50), steps=4)
    pg.mouse.up()
    pg.wait_for_function("V.rows.some(r => r.pending === 'added' && r.x1 > 80)")
    _commit(pg, "draw one box")
    t = ds.annotations("clip", frames=(45, 45), kind="box")
    assert t.num_rows == before + 1
    row = [r for r in t.to_pylist() if r["track_id"] is None][0]
    assert abs(row["x0"] - 96 * 0.30) < 2.5 and abs(row["y0"] - 64 * 0.10) < 2.5 and abs(row["y1"] - 64 * 0.50) < 2.5
    assert abs(row["x1"] - 96 * 0.90) < 2.5                    # moved right 20%, then its right edge pulled out
    assert row["origin"] == "human" and row["status"] == "unreviewed"


def test_label_a_span_on_the_timeline(page):
    pg, ds, base = page
    _viewer(pg, base, 2)
    before = ds.annotations("clip", kind="marker").num_rows
    pg.keyboard.press("s")
    b = pg.locator(".transport canvas").bounding_box()
    pg.mouse.move(b["x"] + b["width"] * 0.50, b["y"] + 30)
    pg.mouse.down()
    pg.mouse.move(b["x"] + b["width"] * 0.65, b["y"] + 30, steps=5)
    pg.mouse.move(b["x"] + b["width"] * 0.80, b["y"] + 30, steps=5)
    pg.mouse.up()
    pg.wait_for_selector(".toast:has-text('Labelled frames')")
    _commit(pg, "label a span")
    t = ds.annotations("clip", kind="marker")
    assert t.num_rows == before + 1
    new = [r for r in t.to_pylist() if r["start_frame"] > 20 and r["end_frame"] < 56 and r["end_frame"] - r["start_frame"] > 10]
    assert new and abs(new[-1]["start_frame"] - 59 * 0.50) < 3 and abs(new[-1]["end_frame"] - 59 * 0.80) < 3


def test_paint_polygon_erase_and_reshape_a_mask(page):
    pg, ds, base = page
    from vidlabel import masks
    _viewer(pg, base, 50)
    before = ds.annotations("clip", frames=(50, 50), kind="mask").num_rows
    at = _ov(pg)
    pg.keyboard.press("m")                                    # brush: a thick stroke
    pg.evaluate("V.brush = 6")
    pg.mouse.move(*at(0.20, 0.50))
    pg.mouse.down()
    pg.mouse.move(*at(0.40, 0.50), steps=6)
    pg.mouse.move(*at(0.60, 0.50), steps=6)
    pg.mouse.up()
    pg.keyboard.press("p")                                    # polygon: a triangle below it, closed with Enter
    for fx, fy in ((0.30, 0.70), (0.50, 0.70), (0.40, 0.90)):
        pg.mouse.click(*at(fx, fy))
    pg.keyboard.press("Enter")
    pg.click("button:has-text('Save mask')")
    pg.wait_for_function("document.querySelector('#changes-count').textContent === '1'")
    _commit(pg, "paint a mask")
    t = ds.annotations("clip", frames=(50, 50), kind="mask")
    assert t.num_rows == before + 1
    m = [r for r in t.to_pylist() if r["track_id"] is None][0]
    area1 = int(masks.decode(m["mask"]).sum())
    assert area1 > 100 and m["x0"] < 96 * 0.25 and m["y1"] > 64 * 0.8
    # select it, edit it, erase the left half of the stroke, apply
    _viewer(pg, base, 50)
    at = _ov(pg)
    pg.mouse.click(*at(0.40, 0.50))
    pg.click("button:has-text('Edit mask shape')")
    pg.wait_for_selector(".notice:has-text('Editing a mask')")
    pg.keyboard.press("e")
    pg.evaluate("V.brush = 10")
    pg.mouse.move(*at(0.12, 0.45))
    pg.mouse.down()
    pg.mouse.move(*at(0.30, 0.55), steps=6)
    pg.mouse.move(*at(0.34, 0.60), steps=6)
    pg.mouse.up()
    pg.click("button:has-text('Apply to the mask')")
    pg.wait_for_function("document.querySelector('#changes-count').textContent === '1'")
    _commit(pg, "trim the mask")
    m2 = [r for r in ds.annotations("clip", frames=(50, 50), kind="mask").to_pylist() if r["annotation_id"] == m["annotation_id"]][0]
    assert int(masks.decode(m2["mask"]).sum()) < area1 and m2["x0"] > m["x0"]


def test_drawing_is_off_inside_a_view(page):
    pg, ds, base = page
    pg.goto(f"{base}/#/video/clip?f=3&view=left_half")
    pg.reload()
    pg.wait_for_selector("canvas.ov")
    assert pg.locator(".tools button:has-text('Box')").is_disabled()
    assert pg.locator("text=Choose “Whole frame” to draw").count() == 1


def test_mask_panel_is_shown_once(page):
    pg, ds, base = page
    _viewer(pg, base, 52)
    pg.keyboard.press("m")
    pg.keyboard.press("p")
    pg.keyboard.press("b")
    pg.keyboard.press("m")
    assert pg.locator(".side .notice:has-text('New mask')").count() == 1
