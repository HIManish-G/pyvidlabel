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


def _settled(pg, frame):
    """Wait until the picture on the canvas, the frame box and the timeline all show `frame`."""
    pg.wait_for_function("f => V && V.painted && V.painted.n === f && V.painted.shown === f && V.painted.playhead === f "
                         "&& +document.querySelector('.framein').value === f", arg=frame)


def test_viewer_shows_frame_boxes_and_mask_and_steps(page):
    pg, ds, base = page
    pg.goto(f"{base}/#/video/clip?f=10")
    pg.reload()
    pg.wait_for_selector(".annrow")
    assert pg.locator(".annrow").count() == 4                    # three tracks and one mask on frame 10
    _settled(pg, 10)
    pg.wait_for_function("V.masks.size > 0 && [...V.masks.values()].every(e => e.ready)")      # the mask image has arrived
    pg.evaluate("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")      # and been painted
    pixels = pg.evaluate("""() => { const c = document.querySelector('.stage canvas'); const x = c.getContext('2d');
        const d = x.getImageData(70, 40, 1, 1).data; return [c.width, c.height, d[0], d[1], d[2]]; }""")
    assert pixels[:2] == [96, 64] and pixels[2:] != [60, 60, 60]  # the mask tint changed the pixel at (70, 40)
    pg.keyboard.press("ArrowRight")
    _settled(pg, 11)
    pg.keyboard.press("Shift+ArrowRight")
    _settled(pg, 21)
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
    try:
        pg.wait_for_selector("tbody tr:has-text('reject first track')", timeout=15000)
    except Exception as e:                                    # say what the page and the server held, so a failure on another machine can be read
        hist = pg.evaluate("fetch('/api/history').then(r => r.text())")
        raise AssertionError(f"history row missing. page text: {pg.inner_text('#main')[:400]!r}; /api/history: {hist[:600]}; errors: {pg.errors}") from e
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
    _settled(pg, frame)


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


def test_drawing_tools_are_available_inside_a_view_and_respect_its_labels(page):
    pg, ds, base = page
    pg.goto(f"{base}/#/video/clip?f=3&view=left_half")
    pg.reload()
    pg.wait_for_selector("canvas.ov")
    assert not pg.locator(".tools button:has-text('Box')").is_disabled()
    assert "drawing in view left_half" in pg.inner_text(".tools")
    ds.add_view({"view_id": "only_car", "video_id": "clip", "region": {"x": 0, "y": 0, "width": 48, "height": 32}, "allowed_labels": ["car"]})
    pg.goto(f"{base}/#/video/clip?f=3&view=only_car")
    pg.reload()
    pg.wait_for_selector("canvas.ov")
    assert pg.locator("select[aria-label='Label for new drawings'] option").all_inner_texts() == ["car"]


def test_mask_panel_is_shown_once(page):
    pg, ds, base = page
    _viewer(pg, base, 52)
    pg.keyboard.press("m")
    pg.keyboard.press("p")
    pg.keyboard.press("b")
    pg.keyboard.press("m")
    assert pg.locator(".side .notice:has-text('New mask')").count() == 1


# ---------------------------------------------------------------- the player stays in sync
def test_scrubbing_the_timeline_keeps_readout_playhead_and_picture_in_step(page):
    pg, ds, base = page
    _viewer(pg, base, 0)
    b = pg.locator(".transport canvas").bounding_box()
    pg.mouse.move(b["x"] + 3, b["y"] + 40)
    pg.mouse.down()
    bad = []
    for i in range(1, 61):                                    # a wild drag back and forth across the whole bar
        fx = abs(((i * 7) % 40) - 20) / 20 * 0.96 + 0.02
        pg.mouse.move(b["x"] + b["width"] * fx, b["y"] + 40)
        got = pg.evaluate("""async () => { await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
            const r = document.querySelector('.transport canvas').getBoundingClientRect();
            return { n: V.n, readout: +document.querySelector('.framein').value, painted: V.painted.n, playhead: V.painted.playhead,
                     w: r.width }; }""")
        want = int(fx * 60)
        if not (got["n"] == got["readout"] == got["painted"] == got["playhead"] and abs(got["n"] - want) <= 1):
            bad.append((i, fx, want, got))
    pg.mouse.up()
    assert not bad, bad[:3]
    n = pg.evaluate("V.n")
    _settled(pg, n)                                           # and the picture catches up with the same frame
    assert not pg.errors


def test_a_click_lands_on_exactly_the_frame_under_the_mouse_also_when_zoomed(page):
    pg, ds, base = page
    _viewer(pg, base, 0)
    c = pg.locator(".transport canvas")
    b = c.bounding_box()
    pg.mouse.move(b["x"] + b["width"] / 2, b["y"] + 40)
    for _ in range(3):
        pg.mouse.wheel(0, -300)                               # zoom in around the middle
    pg.wait_for_function("V.tv.t1 - V.tv.t0 < 40")
    t0, t1 = pg.evaluate("[V.tv.t0, V.tv.t1]")
    for fx in (0.05, 0.31, 0.5, 0.77, 0.95):
        x = b["x"] + b["width"] * fx
        pg.mouse.click(x, b["y"] + 40)
        want = min(59, max(0, int(t0 + fx * (t1 - t0))))
        pg.wait_for_function("w => V.n === w", arg=want)
        _settled(pg, want)
        # the playhead is painted at the middle of that frame's column
        cx = pg.evaluate("tlX(V.n + 0.5)")
        assert abs(cx - b["width"] * fx) <= b["width"] / (t1 - t0)
    pg.click("button[title='Fit: show the whole timeline']")
    pg.wait_for_function("V.tv.t0 === 0 && V.tv.t1 === 60")
    assert not pg.errors


def test_playing_moves_forward_frame_by_frame_and_stops_in_step(page):
    pg, ds, base = page
    _viewer(pg, base, 0)
    pg.evaluate("window.__seen = []; (function loop(){ if (V && V.playing) { const s = window.__seen; if (s[s.length-1] !== V.n) s.push(V.n); } requestAnimationFrame(loop); })()")
    pg.click(".transport .play")
    pg.wait_for_function("V.n >= 12")
    pg.click(".transport .play")                              # pause
    seen = pg.evaluate("window.__seen")
    assert seen == sorted(seen) and all(b - a == 1 for a, b in zip(seen, seen[1:])), seen      # never skips, never goes back
    _settled(pg, pg.evaluate("V.n"))
    assert not pg.errors


def test_zooming_the_picture_keeps_drawing_exact(page):
    pg, ds, base = page
    _viewer(pg, base, 46)
    stage = pg.locator(".stage").bounding_box()
    pg.mouse.move(stage["x"] + stage["width"] / 2, stage["y"] + stage["height"] / 2)
    pg.mouse.wheel(0, -600)
    pg.wait_for_function("V.zoom > 1.5")
    at = _ov(pg)                                              # the overlay's box now includes the zoom
    pg.keyboard.press("b")
    pg.mouse.move(*at(0.40, 0.40))
    pg.mouse.down()
    pg.mouse.move(*at(0.50, 0.50), steps=4)
    pg.mouse.move(*at(0.60, 0.60), steps=4)
    pg.mouse.up()
    pg.wait_for_function("V.rows.some(r => r.pending === 'added')")
    r = pg.evaluate("V.rows.find(r => r.pending === 'added')")
    assert abs(r["x0"] - 96 * 0.40) < 1.5 and abs(r["y1"] - 64 * 0.60) < 1.5     # picture coordinates, whatever the zoom
    pg.keyboard.press("0")
    pg.wait_for_function("V.zoom === 1")
    assert not pg.errors


def test_dragging_the_end_of_a_span_resizes_it(page):
    pg, ds, base = page
    _viewer(pg, base, 3)
    ds.add_annotations([{"kind": "marker", "label": "person", "start_frame": 10, "end_frame": 20, "origin": "human"}], "clip")
    pg.reload()
    _settled(pg, 3)
    pg.wait_for_function("V.tl && V.tl.markers.some(m => m.start_frame === 10 && m.end_frame === 20)")
    b = pg.locator(".transport canvas").bounding_box()
    lay = pg.evaluate("tlLayout(V)")
    y = b["y"] + lay["spans"]["y"] + 6
    x_end = b["x"] + b["width"] * 21 / 60 - 2                 # just inside the right end
    pg.mouse.move(x_end, y)
    pg.mouse.down()
    pg.mouse.move(b["x"] + b["width"] * 26 / 60, y, steps=5)
    pg.mouse.up()
    pg.wait_for_function("V.tl.markers.some(m => m.start_frame === 10 && m.end_frame >= 24 && m.end_frame <= 26 && m.pending)")
    assert not pg.errors


def test_jump_between_annotation_edges_and_copy_a_box_forward(page):
    pg, ds, base = page
    _viewer(pg, base, 45)                                     # tracks 1-3 run over frames 5 to 39
    pg.keyboard.press(",")
    _settled(pg, 39)
    pg.keyboard.press(",")
    pg.wait_for_function("V.n < 39")                          # the next earlier edge (a span or mask edge may sit between)
    _viewer(pg, base, 6)
    pg.wait_for_selector(".annrow")
    pg.evaluate("V.sel = V.rows.find(r => r.kind === 'box' && r.track_id === 1).annotation_id")
    before = ds.annotations("clip", frames=(7, 7), kind="box").num_rows
    pg.keyboard.press("c")                                    # frame 7 already has track 1: refused, nothing changes
    pg.wait_for_selector(".toast.bad:has-text('already has a box')")
    _viewer(pg, base, 39)
    pg.wait_for_selector(".annrow")
    pg.evaluate("V.sel = V.rows.find(r => r.kind === 'box' && r.track_id === 1).annotation_id")
    pg.keyboard.press("c")                                    # frame 40 is empty: the copy lands there and the viewer follows
    _settled(pg, 40)
    pg.wait_for_function("V.rows.some(r => r.pending === 'added' && r.track_id === 1 && r.start_frame === 40)")
    assert ds.annotations("clip", frames=(40, 40), kind="box").num_rows == 0            # still only a draft
    assert ds.annotations("clip", frames=(7, 7), kind="box").num_rows == before
    pg.evaluate("fetch('/api/draft/discard', {method: 'POST', headers: {'X-Requested-With': 'vidlabel', 'X-Vidlabel-Author': 'ana', 'Content-Type': 'application/json'}, body: '{}'})")
    assert not pg.errors


def test_pressing_c_quickly_never_makes_two_boxes_for_one_track_on_a_frame(page):
    pg, ds, base = page
    _viewer(pg, base, 39)
    pg.wait_for_selector(".annrow")
    pg.evaluate("V.sel = V.rows.find(r => r.kind === 'box' && r.track_id === 1).annotation_id")
    for _ in range(6):                                        # six quick presses, faster than the server answers
        pg.keyboard.press("c")
    pg.wait_for_timeout(1500)
    rows = pg.evaluate("fetch('/api/draft', {headers: {'X-Vidlabel-Author': 'ana'}}).then(r => r.json())")
    adds = [o for o in rows["ops"] if o["kind"] == "add" and not o["undone"]]
    labels = [o["label"] for o in adds]
    assert len(labels) == len(set(labels)), labels            # no frame got two copies
    _commit(pg, "copy forward")                               # and the commit is accepted
    t = ds.annotations("clip", kind="box", track_id=1).to_pylist()
    frames = [r["start_frame"] for r in t if r["view_id"] is None]
    assert len(frames) == len(set(frames))
    assert not pg.errors


def test_the_box_tool_refuses_a_second_box_for_a_track_on_the_same_frame(page):
    pg, ds, base = page
    _viewer(pg, base, 20)                                     # tracks 1 to 3 already have boxes here
    pg.evaluate("V.track = '2'")
    at = _ov(pg)
    pg.keyboard.press("b")
    pg.mouse.move(*at(0.10, 0.60))
    pg.mouse.down()
    pg.mouse.move(*at(0.20, 0.70), steps=3)
    pg.mouse.move(*at(0.30, 0.90), steps=3)
    pg.mouse.up()
    pg.wait_for_selector(".toast.bad:has-text('already has a box')")
    assert _draft_count(pg) == 0


def test_a_slow_page_never_overwrites_the_page_you_moved_to(page):
    pg, ds, base = page
    pg.evaluate("""() => { const f = window.fetch; window.fetch = (u, o) => (String(u).includes('/api/videos/clip')
        ? new Promise(r => setTimeout(r, 1500)).then(() => f(u, o)) : f(u, o)); }""")                 # the Review page's data arrives late
    pg.evaluate("location.hash = '#/review/clip'")
    pg.wait_for_timeout(300)
    pg.evaluate("location.hash = '#/history'")                # move on before the Review page has finished
    pg.wait_for_selector("h1:has-text('History')")
    pg.wait_for_timeout(2500)                                 # let the slow page finish
    assert pg.locator("h1").first.inner_text().strip().lower().startswith("history")
    assert not pg.errors


def test_proposals_page_takes_agent_rows_into_my_draft(page):
    pg, ds, base = page
    from vidlabel.agent import AgentSession
    AgentSession(ds.path, "scout").propose_boxes("clip", [{"frame": 55, "label": "person", "box": [2, 2, 20, 20], "confidence": 0.6}],
                                                  note="from the browser test")
    pg.goto(f"{base}/#/overview")
    pg.reload()
    pg.wait_for_selector("#nav a:has-text('Proposals') .badge")
    pg.click("#nav a:has-text('Proposals')")
    pg.wait_for_selector("h1:has-text('Proposals')")
    pg.wait_for_selector("text=from the browser test")
    assert _draft_count(pg) == 0
    pg.click("button:has-text('Take all into my draft')")
    pg.click(".modal button:has-text('OK')")
    pg.wait_for_selector(".toast:has-text('Taken into your draft')")
    pg.wait_for_function("document.querySelector('#changes-count').textContent === '1'")
    pg.wait_for_selector(".card.empty")                                     # nothing left waiting
    assert not pg.locator("#nav a:has-text('Proposals') .badge").count()
    _viewer(pg, base, 55)
    pg.wait_for_selector(".annrow.pend")                                    # the proposal is a pending row in the viewer
    assert pg.evaluate("V.rows.some(r => r.pending === 'added' && r.origin === 'model' && r.author === 'agent:scout')")
    pg.evaluate("fetch('/api/draft/discard', {method: 'POST', headers: {'X-Requested-With': 'vidlabel', 'X-Vidlabel-Author': 'ana', 'Content-Type': 'application/json'}, body: '{}'})")
    assert not pg.errors


def test_proposals_page_empty_state_hostile_text_per_video_and_conflicts(page):
    pg, ds, base = page
    from vidlabel.agent import AgentSession
    from vidlabel.ui import draft as Dr
    pg.goto(f"{base}/#/proposals")
    pg.reload()
    pg.wait_for_selector(".card.empty:has-text('Nothing is waiting')")
    assert not pg.locator("#nav a:has-text('Proposals') .badge").count()

    hostile = "<img src=x onerror=\"window.__pwned=1\"><script>window.__pwned=1</script>"
    ag = AgentSession(ds.path, "scout")
    ag.propose_boxes("clip", [{"frame": 56, "label": "person", "box": [2, 2, 20, 20], "track_id": 9}, {"frame": 57, "label": "person", "box": [2, 2, 20, 20], "track_id": 9}], note=hostile)
    ag.propose_spans("clip", [{"start_frame": 1, "end_frame": 4, "label": "car"}], note="span note")
    pg.goto(f"{base}/#/proposals")
    pg.reload()
    pg.wait_for_selector("h2:has-text('agent:scout')")
    assert pg.evaluate("window.__pwned") is None and pg.locator("table img, table script").count() == 0       # a note is text, never markup
    assert "<img" in pg.inner_text("table")
    assert pg.locator("tbody tr").count() == 3

    # someone else puts a box on track 9 in frame 56 into my draft: the take-in must be refused as a whole and say why
    Dr.Draft.load(ds, "ana").add("clip", {"kind": "box", "label": "person", "start_frame": 56, "box": (1, 1, 9, 9), "track_id": 9})
    pg.reload()
    pg.wait_for_selector("h2:has-text('agent:scout')")
    n = _draft_count(pg)
    pg.click("button:has-text('Take all into my draft')")
    pg.click(".modal button:has-text('OK')")
    pg.wait_for_selector(".toast:has-text('Nothing was taken in')")
    assert _draft_count(pg) == n and pg.locator("tbody tr").count() == 3                       # my draft and the list are as they were

    # remove the conflict, then take in only what is wanted; discard the rest
    Dr.Draft.load(ds, "ana").discard(0)
    pg.reload()
    pg.wait_for_selector("h2:has-text('agent:scout')")
    pg.click("button.small:has-text('Throw away')")
    pg.click(".modal button:has-text('OK')")
    pg.wait_for_selector(".toast:has-text('Discarded')")
    pg.wait_for_selector(".card.empty")
    assert Dr.proposals(ds, "ana") == [] and Dr.Draft.load(ds, "ana").summary()["added"] == 0
    assert [e for e in pg.errors if "status of 400" not in e] == [] and len(pg.errors) == 1       # only the refusal above, which is meant


def test_draw_a_box_a_mask_and_a_span_inside_a_view(page):
    pg, ds, base = page
    from vidlabel import masks
    ds.add_view({"view_id": "crop", "video_id": "clip", "region": {"x": 24, "y": 16, "width": 48, "height": 32}})
    pg.goto(f"{base}/#/video/clip?f=46&view=crop")
    pg.reload()
    pg.wait_for_selector("canvas.ov")
    _settled(pg, 46)
    assert pg.evaluate("V.size[0] === 48 && V.size[1] === 32")                    # the picture is the view's
    assert not pg.locator(".tools button:has-text('Box')[disabled]").count()      # tools are available in a view
    assert "drawing in view crop" in pg.inner_text(".tools")
    at = _ov(pg)
    pg.keyboard.press("b")
    pg.mouse.move(*at(0.10, 0.20))
    pg.mouse.down()
    pg.mouse.move(*at(0.30, 0.40), steps=4)
    pg.mouse.move(*at(0.50, 0.60), steps=4)
    pg.mouse.up()
    pg.wait_for_function("document.querySelector('#changes-count').textContent === '1'")
    pg.keyboard.press("p")
    for fx, fy in ((0.60, 0.20), (0.90, 0.20), (0.90, 0.80)):
        pg.mouse.click(*at(fx, fy))
    pg.mouse.dblclick(*at(0.90, 0.80))
    pg.click("button:has-text('Save mask')")
    pg.wait_for_function("document.querySelector('#changes-count').textContent === '2'")
    pg.wait_for_function("[...V.masks.values()].some(m => m.ready && !m.empty)")       # the pending mask is drawn in the picture, not only after the commit
    pg.keyboard.press("s")
    b = pg.locator(".transport canvas").bounding_box()
    pg.mouse.move(b["x"] + b["width"] * 0.25, b["y"] + 30)
    pg.mouse.down()
    pg.mouse.move(b["x"] + b["width"] * 0.30, b["y"] + 30, steps=5)
    pg.mouse.move(b["x"] + b["width"] * 0.35, b["y"] + 30, steps=5)
    pg.mouse.up()
    pg.wait_for_selector(".toast:has-text('Labelled frame')")
    _commit(pg, "draw in a view")
    rows = ds.annotations("clip", view_id=["crop"]).to_pylist()
    box = next(r for r in rows if r["kind"] == "box")
    assert abs(box["x0"] - (24 + 48 * 0.10)) < 2 and abs(box["y0"] - (16 + 32 * 0.20)) < 2 and abs(box["x1"] - (24 + 48 * 0.50)) < 2 and abs(box["y1"] - (16 + 32 * 0.60)) < 2
    mrow = next(r for r in rows if r["kind"] == "mask")
    full = masks.decode(mrow["mask"])
    assert full.shape == (64, 96) and 24 <= mrow["x0"] and mrow["x1"] <= 72 and 16 <= mrow["y0"] and mrow["y1"] <= 48 and full.sum() > 50
    assert any(r["kind"] == "marker" for r in rows)
    assert all(r["view_id"] == "crop" for r in rows) and not [r for r in ds.annotations("clip", frames=(46, 46), view_id=[None]).to_pylist() if r["kind"] != "marker" and r["start_frame"] == 46]
    # back on the whole frame the box is where it really is
    pg.goto(f"{base}/#/video/clip?f=46")
    pg.reload()
    pg.wait_for_selector("canvas.ov")
    _settled(pg, 46)
    assert pg.evaluate("V.rows.filter(r => r.kind === 'box').length") == 0         # a view's rows are not whole-frame rows
    assert not pg.errors


def test_sample_inspector_page_shows_the_targets_and_moves_between_samples(page):
    pg, ds, base = page
    pg.goto(f"{base}/#/inspector?video=clip&f=6&release=none")
    pg.reload()
    pg.wait_for_selector("h1:has-text('Sample inspector')")
    pg.wait_for_selector(".inspect-pic .ibox")
    assert pg.locator("h2:has-text('target')").count() == 1
    n = pg.locator(".inspect-pic .ibox").count()
    assert n == pg.locator("table tbody tr").count() - pg.locator("details table tbody tr").count() and n >= 1
    assert pg.locator(".inspect-pic").get_attribute("aria-label").startswith("Frame 6 of clip")
    pg.goto(f"{base}/#/inspector?video=clip&f=10&release=none")
    pg.wait_for_selector(".inspect-pic .imask")                              # a frame with a mask shows it, tinted with its class colour
    pg.goto(f"{base}/#/inspector?video=clip&f=6&release=none")
    pg.wait_for_selector(".inspect-pic .ibox")
    pg.click("button:has-text('Next sample')")
    pg.wait_for_function("location.hash.includes('f=7')")
    pg.wait_for_selector(".inspect-pic .ibox")
    pg.click("button[aria-label='Previous frame']")
    pg.wait_for_function("location.hash.includes('f=6')")
    # settings change what the loader would see: boxes as fractions in cxcywh
    pg.select_option("label:has-text('Box format') select", "cxcywh")
    pg.check("label:has-text('fractions') input")
    pg.wait_for_function("location.hash.includes('fmt=cxcywh') && location.hash.includes('norm=1')")
    pg.wait_for_selector("th:has-text('cxcywh, fractions')")
    assert all(0 <= float(v) <= 1 for v in pg.locator("table tbody tr td.mono").first.inner_text().split(","))
    pg.fill("input[aria-label='Frame number']", "500")
    pg.keyboard.press("Enter")
    pg.wait_for_selector(".toast.bad, .toast:has-text('no frame')")
    # the old inspector card on the Releases page is now a link to this page
    pg.goto(f"{base}/#/releases")
    pg.reload()
    pg.click("a:has-text('Inspect what a training sample contains')")
    pg.wait_for_selector("h1:has-text('Sample inspector')")
    assert [e for e in pg.errors if "status of 4" not in e] == []
