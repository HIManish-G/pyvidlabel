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
    pg.wait_for_selector("h1:has-text('Overview')")
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
