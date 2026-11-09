"""The web server: security, routes over a real dataset with a real video, jobs, and coverage of the CLI and library."""
import shutil
import subprocess
import time

import numpy as np
import pytest

pytest.importorskip("starlette")
pytest.importorskip("httpx")
cv2 = pytest.importorskip("cv2")

from starlette.testclient import TestClient  # noqa: E402
from vidlabel import Dataset, masks  # noqa: E402
from vidlabel.cli import app as cli_app  # noqa: E402
from vidlabel.probe import probe_video  # noqa: E402
from vidlabel.ui import app as uiapp  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")
H = {"X-Requested-With": "vidlabel", "X-Vidlabel-Author": "ana"}


def make_video(path, n=40, w=96, h=64):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25", "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        f = np.full((h, w, 3), 60, np.uint8)
        f[16:32, 0:16] = (255, 0, 0)
        p.stdin.write(f.tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return path


@pytest.fixture
def env(tmp_path):
    ds = Dataset.create(tmp_path / "d")
    vp = make_video(tmp_path / "clip.mp4") if HAVE_FFMPEG else None
    if vp:
        meta = probe_video(vp, "clip")
        ds.add_video(meta)
    else:
        ds.add_video({"video_id": "clip", "frame_count": 40, "fps": {"num": 25, "den": 1}, "width": 96, "height": 64})
    ds.add_labels(["thing", "other"])
    m = np.zeros((64, 96), bool)
    m[20:30, 30:50] = True
    ds.add_annotations([{"kind": "box", "label": "thing", "start_frame": f, "box": (5, 5, 40, 40), "track_id": 1,
                         "origin": "model", "confidence": 0.8} for f in range(0, 10)]
                       + [{"kind": "mask", "label": "other", "start_frame": 3, "mask": m, "origin": "human"},
                          {"kind": "marker", "label": "thing", "start_frame": 20, "end_frame": 30, "origin": "human"}], "clip")
    a = uiapp.create_app(tmp_path / "d", token="tok", port=8765)
    c = TestClient(a, base_url="http://127.0.0.1:8765")
    c.cookies.set("vl_token", "tok")
    return c, ds, a.state.ui


def post(c, path, body=None, **kw):
    return c.post(path, json=body or {}, headers=H, **kw)


# ---------------------------------------------------------------- security
def test_no_token_is_refused_but_the_page_explains(env):
    c, ds, st = env
    anon = TestClient(c.app, base_url="http://127.0.0.1:8765")
    assert anon.get("/api/dataset").status_code == 401
    r = anon.get("/")
    assert r.status_code == 401 and "token" in r.text


def test_token_in_address_sets_cookie(env):
    c, ds, st = env
    anon = TestClient(c.app, base_url="http://127.0.0.1:8765")
    r = anon.get("/?token=tok", follow_redirects=False)
    assert r.status_code in (302, 307) and "vl_token" in r.headers["set-cookie"]
    assert anon.get("/api/dataset").status_code == 200
    assert TestClient(c.app, base_url="http://127.0.0.1:8765").get("/?token=wrong").status_code == 401


def test_wrong_host_and_origin_refused(env):
    c, ds, st = env
    assert c.get("/api/dataset", headers={"Host": "evil.example:8765"}).status_code == 421
    assert c.get("/api/dataset", headers={"Origin": "http://evil.example"}).status_code == 403
    assert c.get("/api/dataset", headers={"Origin": "http://127.0.0.1:8765"}).status_code == 200


def test_changes_need_the_header(env):
    c, ds, st = env
    assert c.post("/api/draft/undo", json={}).status_code == 403
    assert post(c, "/api/draft/undo").status_code == 400            # nothing to undo is a clear error, not a crash
    assert post(c, "/api/draft/undo").json()["error"]


def test_shared_server_has_no_file_access(env, tmp_path):
    c, ds, st = env
    a = uiapp.create_app(tmp_path / "d", token="t", host="0.0.0.0", port=9000)
    s = TestClient(a, base_url="http://0.0.0.0:9000")
    s.cookies.set("vl_token", "t")
    r = s.get("/api/fs?path=/")
    assert r.status_code == 403
    assert s.get("/api/session").json()["can_browse_files"] is False


def test_fs_roots_confine_paths(tmp_path):
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (tmp_path / "secret").mkdir()
    a = uiapp.create_app(tmp_path / "d", token="t", fs_roots=[root])
    c = TestClient(a, base_url="http://127.0.0.1:8765")
    c.cookies.set("vl_token", "t")
    assert c.get("/api/fs", params={"path": str(root)}).status_code == 200
    assert c.get("/api/fs", params={"path": str(root / "sub")}).json()["parent"] == str(root.resolve())
    assert c.get("/api/fs", params={"path": str(tmp_path / "secret")}).status_code == 403
    assert c.get("/api/fs", params={"path": str(root / ".." / "secret")}).status_code == 403
    assert post(c, "/api/init", {"path": str(tmp_path / "secret" / "x")}).status_code == 403


def test_setup_mode_and_init(tmp_path):
    a = uiapp.create_app(tmp_path / "new", token="t")
    c = TestClient(a, base_url="http://127.0.0.1:8765")
    c.cookies.set("vl_token", "t")
    assert c.get("/api/session").json()["has_dataset"] is False
    assert c.get("/api/dataset").status_code == 409
    assert post(c, "/api/init", {"path": str(tmp_path / "new")}).status_code == 200
    assert c.get("/api/dataset").json()["stats"]["total"] == 0 if "total" in c.get("/api/dataset").json()["stats"] else True
    assert post(c, "/api/init", {"path": str(tmp_path / "new")}).status_code in (400, 409)


# ---------------------------------------------------------------- reading
def test_dataset_and_listing_routes(env):
    c, ds, st = env
    d = c.get("/api/dataset").json()
    assert d["videos"] == ["clip"] and {l["name"] for l in d["labels"]} == {"thing", "other"}
    assert c.get("/api/videos").json()["videos"][0]["video_id"] == "clip"
    assert c.get("/api/videos/clip").json()["metadata"]["video_id"] == "clip"
    assert c.get("/api/videos/nope").status_code == 404
    assert {l["name"]: l["count"] for l in c.get("/api/labels").json()["labels"]} == {"thing": 11, "other": 1}
    assert c.get("/api/validate").json()["ok"] is True
    assert c.get("/api/releases").json()["releases"] == []
    assert c.get("/api/schema").status_code == 200


def test_ids_are_strings_and_pagination_and_filters(env):
    c, ds, st = env
    r = c.get("/api/annotations", params={"video_id": "clip", "limit": 5}).json()
    assert r["total"] == 12 and len(r["rows"]) == 5
    assert all(isinstance(x["annotation_id"], str) for x in r["rows"])
    assert all("mask" not in x for x in r["rows"])
    assert c.get("/api/annotations", params={"video_id": "clip", "kind": "mask"}).json()["rows"][0]["has_mask"] is True
    assert c.get("/api/annotations", params={"video_id": "clip", "frames": "3-3"}).json()["total"] == 2
    assert c.get("/api/annotations", params={"video_id": "clip", "frames": "x"}).status_code == 400
    ids = [x["annotation_id"] for x in r["rows"]]
    assert all(int(i) == int(str(i)) for i in ids)


def test_tracks_and_timeline(env):
    c, ds, st = env
    t = c.get("/api/tracks", params={"video_id": "clip"}).json()["tracks"]
    assert t[0]["track_id"] == 1 and t[0]["n_rows"] == 10
    tl = c.get("/api/timeline/clip").json()
    assert tl["frame_count"] == 40 and len(tl["markers"]) == 1 and sum(tl["density"]) == 11


def test_frame_data_with_mask_image(env):
    c, ds, st = env
    d = c.get("/api/frame-data/clip/3").json()
    mask = [r for r in d["rows"] if r["kind"] == "mask"][0]
    assert mask["mask_png"].startswith("data:image/png;base64,") and (mask["mask_x"], mask["mask_y"]) == (30, 20)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg")
def test_frame_image(env):
    c, ds, st = env
    r = c.get("/api/frame/clip/5")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (64, 96)
    assert c.get("/api/frame/clip/999").status_code == 404
    small = c.get("/api/frame/clip/5", params={"max_side": 48}).content
    assert max(cv2.imdecode(np.frombuffer(small, np.uint8), 1).shape[:2]) == 48


# ---------------------------------------------------------------- the draft over HTTP
def test_draft_flow_overlay_commit_history_revert(env):
    c, ds, st = env
    before = ds._ann_path("clip").read_bytes()
    r = post(c, "/api/draft/update-track", {"video_id": "clip", "track_id": 1, "values": {"status": "rejected"}})
    assert r.json()["summary"]["modified"] == 10
    assert ds._ann_path("clip").read_bytes() == before                      # nothing written yet
    mine = c.get("/api/annotations", params={"video_id": "clip", "status": "rejected"}, headers={"X-Vidlabel-Author": "ana"}).json()
    assert mine["total"] == 10 and mine["rows"][0]["pending"] == "modified"
    other = c.get("/api/annotations", params={"video_id": "clip", "status": "rejected"}, headers={"X-Vidlabel-Author": "bob"}).json()
    assert other["total"] == 0                                              # per-author drafts
    assert post(c, "/api/draft/commit", {"message": ""}).status_code == 400  # message required
    done = post(c, "/api/draft/commit", {"message": "reject track 1"}).json()
    assert done["seq"] >= 1
    assert c.get("/api/annotations", params={"video_id": "clip", "status": "rejected", "draft": "0"}).json()["total"] == 10
    hist = c.get("/api/history").json()["commits"]
    assert hist[0]["message"] == "reject track 1" and hist[0]["author"] == "ana"
    assert post(c, f"/api/history/{hist[0]['seq']}/revert").status_code == 200
    assert ds.annotations("clip", status="rejected").num_rows == 0


def test_draft_add_undo_redo_discard(env):
    c, ds, st = env
    r = post(c, "/api/draft/add", {"video_id": "clip", "row": {"kind": "box", "label": "thing", "start_frame": 30, "box": [1, 1, 9, 9]}}).json()
    tmp = r["id"]
    assert tmp.startswith("p")
    assert post(c, "/api/draft/update", {"video_id": "clip", "ids": [tmp], "values": {"label": "other"}}).status_code == 200
    rows = c.get("/api/annotations", params={"video_id": "clip", "frames": "30-30"}, headers={"X-Vidlabel-Author": "ana"}).json()["rows"]
    assert [x["label"] for x in rows if x["annotation_id"] == tmp] == ["other"]
    assert post(c, "/api/draft/undo").json()["summary"]["added"] == 1
    assert post(c, "/api/draft/redo").status_code == 200
    assert post(c, "/api/draft/discard", {}).json()["summary"]["added"] == 0
    assert post(c, "/api/draft/update", {"video_id": "clip", "ids": ["abc"], "values": {}}).status_code == 400


def test_bad_edit_is_refused_with_message(env):
    c, ds, st = env
    r = post(c, "/api/draft/add", {"video_id": "clip", "row": {"kind": "box", "label": "nope", "start_frame": 1, "box": [1, 1, 9, 9]}})
    assert r.status_code in (400, 422) and r.json()["error"]


def test_conflict_is_409(env):
    c, ds, st = env
    post(c, "/api/draft/update-track", {"video_id": "clip", "track_id": 1, "values": {"status": "rejected"}})
    ds.update_track("clip", 1, status="approved")                             # someone else changes the dataset
    r = post(c, "/api/draft/commit", {"message": "x"})
    assert r.status_code == 409 and r.json()["conflict"] is True and r.json()["videos"] == ["clip"]


# ---------------------------------------------------------------- structural routes
def test_labels_and_views(env):
    c, ds, st = env
    assert post(c, "/api/labels", {"names": ["a, b", "c"]}).json()["added"] == ["a", "b", "c"]
    assert post(c, "/api/labels", {"names": []}).status_code == 400
    v = post(c, "/api/views", {"view_id": "left", "video_id": "clip", "region": {"x": 0, "y": 0, "width": 48, "height": 64}}).json()
    assert v["view_id"] == "left"
    assert c.get("/api/views").json()["views"][0]["view_id"] == "left"
    dry = post(c, "/api/views/derive", {"video_id": "clip", "view_id": "left", "dry_run": True}).json()
    assert dry["dry_run"] is True and dry["copied"] > 0
    assert ds.annotations("clip", view_id=["left"]).num_rows == 0
    real = post(c, "/api/views/derive", {"video_id": "clip", "view_id": "left"}).json()
    assert real["copied"] == dry["copied"] and ds.annotations("clip", view_id=["left"]).num_rows == real["copied"]


# ---------------------------------------------------------------- jobs
def wait(c, job, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_validate_and_build_jobs(env):
    c, ds, st = env
    j = wait(c, post(c, "/api/jobs/validate", {"deep": True}).json())
    assert j["status"] == "done" and j["result"]["ok"] is True
    b = wait(c, post(c, "/api/jobs/build", {}).json())
    assert b["status"] == "done", b
    rel = c.get("/api/releases").json()["releases"]
    assert len(rel) == 1 and rel[0]["is_latest"] is True
    assert any(x["id"] == b["id"] for x in c.get("/api/jobs").json()["jobs"])
    assert c.get("/api/jobs/nope").status_code == 404


def test_failed_job_reports_error(env):
    c, ds, st = env
    j = wait(c, post(c, "/api/jobs/check-video", {"video_id": "missing"}).json())
    assert j["status"] == "failed" and j["error"]


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg")
def test_add_video_index_check_and_preview_jobs(env, tmp_path):
    c, ds, st = env
    vp = make_video(tmp_path / "second.mp4", n=30)
    j = wait(c, post(c, "/api/jobs/add-video", {"path": str(vp), "video_id": "second"}).json())
    assert j["status"] == "done", j
    assert "second" in c.get("/api/dataset").json()["videos"]
    assert wait(c, post(c, "/api/jobs/check-video", {"video_id": "second"}).json())["status"] == "done"
    assert wait(c, post(c, "/api/jobs/index", {"video_ids": ["second"], "force": True}).json())["status"] == "done"
    p = wait(c, post(c, "/api/jobs/preview", {"video_id": "clip", "frames": "0-3"}).json())
    assert p["status"] == "done", p
    tag = p["result"]["export"]
    files = p["result"].get("files") or []
    if files:
        assert c.get(f"/api/exports/{tag}/{files[0]}").status_code == 200
    assert c.get(f"/api/exports/{tag}/..%2Fx").status_code in (404, 400)
    assert c.get("/api/exports/zz/.hidden").status_code == 404


def test_cache_job_and_sample_inspector(env, tmp_path):
    c, ds, st = env
    wait(c, post(c, "/api/jobs/build", {}).json())
    s = c.get("/api/sample", params={"video_id": "clip", "frame": 3, "kind": ["box", "mask"]})
    assert s.status_code == 200, s.text
    body = s.json()
    assert body["class_names"] and "boxes" in body["targets"]


# ---------------------------------------------------------------- coverage of the CLI and library
CLI_EXCLUDED = {"ui": "starts this server", "mcp": "serves AI agents over standard input and output"}
API_EXCLUDED = {
    "Dataset.add_video_from_probe": "not a method", "Dataset.path": "property", "Dataset.close": "no-op",
}


def test_every_cli_command_has_a_route():
    covered = {name for spec in uiapp.ROUTES for name in spec["covers"]}
    names = {(c.name or c.callback.__name__).replace("_", "-") for c in cli_app.registered_commands}
    missing = {n for n in names if n not in covered and n not in CLI_EXCLUDED}
    assert not missing, f"CLI commands without a route: {sorted(missing)}"


def test_every_public_dataset_method_is_covered_or_excluded():
    covered = {name for spec in uiapp.ROUTES for name in spec["covers"]}
    public = {f"Dataset.{n}" for n in dir(Dataset) if not n.startswith("_") and callable(getattr(Dataset, n))}
    missing = {m for m in public if m not in covered and m not in API_EXCLUDED}
    assert not missing, f"Dataset methods without a route (add to a route's covers=, or to API_EXCLUDED with a reason): {sorted(missing)}"


def test_shared_server_accepts_its_own_address_but_not_other_sites(tmp_path):
    Dataset.create(tmp_path / "d")
    a = uiapp.create_app(tmp_path / "d", token="t", host="0.0.0.0", port=9000)
    c = TestClient(a, base_url="http://192.168.1.5:9000")
    assert c.get("/api/session").status_code == 401                       # token still needed
    c.cookies.set("vl_token", "t")
    assert c.get("/api/session").status_code == 200
    assert c.get("/api/session", headers={"Origin": "http://evil.example"}).status_code == 403
    assert c.get("/api/session", headers={"Origin": "http://192.168.1.5:9000"}).status_code == 200


def test_timeline_has_status_bands_and_roots_find_moved_videos(env, tmp_path):
    c, ds, st = env
    tl = c.get("/api/timeline/clip").json()
    assert set(tl["by_status"]) >= {"approved", "unreviewed", "rejected"}
    assert [sum(x) for x in zip(*tl["by_status"].values())] == tl["density"]
    if HAVE_FFMPEG:
        moved = tmp_path / "elsewhere"
        moved.mkdir()
        (tmp_path / "clip.mp4").rename(moved / "clip.mp4")
        assert c.get("/api/videos").json()["videos"][0]["file_found"] is False
        assert c.get("/api/frame/clip/1").status_code == 404
        assert post(c, "/api/roots", {"roots": [str(moved)]}).json()["roots"] == [str(moved.resolve())]
        assert c.get("/api/videos").json()["videos"][0]["file_found"] is True
        assert c.get("/api/frame/clip/1").status_code == 200


def test_quoted_paths_and_file_paths_in_the_picker(env, tmp_path):
    c, ds, st = env
    folder = tmp_path / "pick"
    folder.mkdir()
    f = folder / "a.mp4"
    f.write_bytes(b"x")
    quoted = c.get("/api/fs", params={"path": f'"{folder}"'})            # Windows "Copy as path" adds quotes
    assert quoted.status_code == 200 and [e["name"] for e in quoted.json()["entries"]] == ["a.mp4"]
    given_file = c.get("/api/fs", params={"path": str(f)}).json()          # a file shows the folder it is in
    assert given_file["path"] == str(folder.resolve())
    assert c.get("/api/fs", params={"path": str(folder / "missing")}).status_code == 404


# ---------------------------------------------------------------- drawing: boxes, spans and masks through the draft
def _bits(mask):
    return uiapp.mask_to_bits(mask)


def test_mask_bits_round_trip():
    m = np.zeros((7, 13), bool)
    m[2:5, 3:9] = True
    assert (uiapp.bits_to_mask(_bits(m)) == m).all()
    with pytest.raises(ValueError):
        uiapp.bits_to_mask({"width": 3, "height": 3, "data": "AA=="}, None)
    with pytest.raises(ValueError):
        uiapp.bits_to_mask(_bits(m), (14, 7))


def test_draw_box_span_and_mask_then_commit_and_revert(env):
    c, ds, st = env
    n0 = ds.annotations("clip").num_rows
    box = post(c, "/api/draft/add", {"video_id": "clip", "row": {"kind": "box", "label": "other", "start_frame": 12, "box": [10, 8, 60, 50], "track_id": 9}})
    assert box.status_code == 200, box.text
    span = post(c, "/api/draft/add", {"video_id": "clip", "row": {"kind": "marker", "label": "thing", "start_frame": 5, "end_frame": 8}})
    assert span.status_code == 200, span.text
    m = np.zeros((64, 96), bool)
    m[10:30, 40:70] = True
    mk = post(c, "/api/draft/add", {"video_id": "clip", "row": {"kind": "mask", "label": "thing", "start_frame": 15, "mask_bits": _bits(m)}})
    assert mk.status_code == 200, mk.text
    shown = c.get("/api/frame-data/clip/15", headers={"X-Vidlabel-Author": "ana"}).json()["rows"]
    mrow = [r for r in shown if r["kind"] == "mask"][0]
    assert mrow["pending"] == "added" and (mrow["x0"], mrow["y0"], mrow["x1"], mrow["y1"]) == (40, 10, 70, 30) and mrow["mask_png"]
    got = c.get(f"/api/mask-bits/clip/{mk.json()['id']}", params={"frame": 15}, headers={"X-Vidlabel-Author": "ana"}).json()
    assert (uiapp.bits_to_mask(got) == m).all()                   # a pending mask can be loaded for editing
    # change the mask's shape: the box follows it, in one edit
    m2 = m.copy()
    m2[10:20, 40:50] = False
    m2[40:50, 5:20] = True
    assert post(c, "/api/draft/update", {"video_id": "clip", "ids": [mk.json()["id"]], "values": {"mask_bits": _bits(m2)}}).status_code == 200
    # the box can be moved and the span's frames changed
    assert post(c, "/api/draft/update", {"video_id": "clip", "ids": [box.json()["id"]], "values": {"box": [12, 10, 62, 52]}}).status_code == 200
    assert post(c, "/api/draft/update", {"video_id": "clip", "ids": [span.json()["id"]], "values": {"start_frame": 4, "end_frame": 9}}).status_code == 200
    done = post(c, "/api/draft/commit", {"message": "draw things"})
    assert done.status_code == 200, done.text
    assert ds.annotations("clip").num_rows == n0 + 3
    t = ds.annotations("clip", frames=(15, 15), kind="mask", label="thing")
    assert t.num_rows == 1 and (masks.decode(t.column("mask")[0].as_py()) == m2).all()
    assert [t.column(k)[0].as_py() for k in ("x0", "y0", "x1", "y1")] == [5, 10, 70, 50]
    assert ds.annotations("clip", kind="marker", label="thing", frames=(4, 4)).num_rows >= 1
    assert post(c, f"/api/history/{done.json()['seq']}/revert").status_code == 200
    assert ds.annotations("clip").num_rows == n0


def test_editing_a_committed_mask_and_reverting_restores_it(env):
    c, ds, st = env
    old = ds.annotations("clip", kind="mask")
    aid = old.column("annotation_id")[0].as_py()
    before = masks.decode(old.column("mask")[0].as_py())
    m = np.zeros((64, 96), bool)
    m[5:15, 5:15] = True
    r = post(c, "/api/draft/update", {"video_id": "clip", "ids": [str(aid)], "values": {"mask_bits": _bits(m)}})
    assert r.status_code == 200, r.text
    done = post(c, "/api/draft/commit", {"message": "reshape mask"}).json()
    assert (masks.decode(ds.annotations("clip", kind="mask").column("mask")[0].as_py()) == m).all()
    assert post(c, f"/api/history/{done['seq']}/revert").status_code == 200
    assert (masks.decode(ds.annotations("clip", kind="mask").column("mask")[0].as_py()) == before).all()


def test_wrong_size_or_empty_mask_is_refused(env):
    c, ds, st = env
    bad = post(c, "/api/draft/add", {"video_id": "clip", "row": {"kind": "mask", "label": "thing", "start_frame": 1, "mask_bits": _bits(np.ones((10, 10), bool))}})
    assert bad.status_code == 400 and "96" in bad.json()["error"]
    aid = ds.annotations("clip", kind="mask").column("annotation_id")[0].as_py()
    empty = post(c, "/api/draft/update", {"video_id": "clip", "ids": [str(aid)], "values": {"mask_bits": _bits(np.zeros((64, 96), bool))}})
    assert empty.status_code == 400 and "empty" in empty.json()["error"]


# ---------------------------------------------------------------- the player's routes
def test_annotations_range_has_no_mask_pixels_and_a_revision(env):
    c, ds, st = env
    d = c.get("/api/annotations-range/clip", params={"start": 0, "end": 20}).json()
    masks = [r for r in d["rows"] if r["kind"] == "mask"]
    assert masks and all("mask" not in r and len(r["mask_rev"]) == 8 for r in masks)
    assert {r["start_frame"] for r in d["rows"]} and c.get("/api/annotations-range/clip", params={"start": 0, "end": 5000}).status_code == 400


def test_mask_png_returns_the_corner_and_is_cached_by_revision(env):
    c, ds, st = env
    m = [r for r in c.get("/api/annotations-range/clip", params={"start": 3, "end": 3}).json()["rows"] if r["kind"] == "mask"][0]
    r = c.get(f"/api/mask-png/clip/{m['annotation_id']}", params={"frame": 3, "rev": m["mask_rev"]})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert (r.headers["x-mask-x"], r.headers["x-mask-y"]) == ("30", "20") and "immutable" in r.headers["cache-control"]
    assert c.get(f"/api/mask-png/clip/{m['annotation_id']}", params={"frame": 3}).headers["cache-control"] == "no-store"


def test_timeline_window_gets_finer_counts(env):
    c, ds, st = env
    whole = c.get("/api/timeline/clip").json()
    part = c.get("/api/timeline/clip", params={"start": 0, "end": 10, "bins": 10}).json()
    assert (part["start"], part["end"], part["bin_size"], len(part["density"])) == (0, 10, 1, 10)
    assert sum(part["density"]) > 0
    assert [sum(x) for x in zip(*part["by_status"].values())] == part["density"]


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg")
def test_old_frame_requests_are_dropped_and_cached_frames_are_reused(env):
    c, ds, st = env
    assert c.get("/api/frame/clip/7", params={"cid": "a", "seq": 5}).status_code == 200      # decoded and cached
    assert c.get("/api/frame/clip/8", params={"cid": "a", "seq": 6}).status_code == 200
    assert c.get("/api/frame/clip/9", params={"cid": "a", "seq": 4}).status_code == 204      # replaced by newer requests already
    assert c.get("/api/frame/clip/7", params={"cid": "a", "seq": 3}).status_code == 200      # but a cached frame is free to send


# ---------------------------------------------------------------- agent proposals over HTTP
def test_proposals_list_adopt_and_discard(env):
    c, ds, st = env
    from vidlabel.agent import AgentSession
    ag = AgentSession(ds.path, "scout")
    ag.propose_boxes("clip", [{"frame": 30, "label": "thing" if "thing" in {l.name for l in ds.vocab.labels} else ds.vocab.labels[0].name,
                               "box": [2, 2, 20, 20], "confidence": 0.6}], note="seen by the agent")
    info = c.get("/api/dataset").json()
    assert info["proposals"] == 1
    p = c.get("/api/proposals").json()["proposals"]
    assert p[0]["author"] == "agent:scout" and p[0]["rows"][0]["note"] == "seen by the agent"
    before = ds._ann_path("clip").read_bytes()
    r = post(c, "/api/proposals/adopt", {"author": "agent:scout"})
    assert r.status_code == 200 and r.json()["taken"] == 1 and r.json()["draft"]["summary"]["added"] == 1
    assert ds._ann_path("clip").read_bytes() == before                      # taking in changes nothing in the dataset
    assert c.get("/api/dataset").json()["proposals"] == 0
    assert post(c, "/api/proposals/adopt", {"author": "agent:scout"}).status_code == 400
    assert post(c, "/api/proposals/discard", {"author": "ana"}).status_code == 400       # a person's draft is not an agent's
    post(c, "/api/draft/discard", {})


def test_proposals_need_the_same_protections_as_other_changes(env):
    c, ds, st = env
    r = c.post("/api/proposals/adopt", json={"author": "agent:x"}, headers={"X-Vidlabel-Author": "ana"})       # no X-Requested-With
    assert r.status_code in (400, 403)
