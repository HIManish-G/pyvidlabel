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


def test_proposals_page_api_edge_cases(env):
    c, ds, st = env
    from vidlabel.agent import AgentSession
    anon = TestClient(c.app, base_url="http://127.0.0.1:8765")
    assert anon.get("/api/proposals").status_code == 401                           # token needed to even look
    assert c.get("/api/proposals").json()["proposals"] == []
    ag = AgentSession(ds.path, "scout")
    ag.propose_boxes("clip", [{"frame": 31, "label": "thing", "box": [2, 2, 20, 20], "track_id": 5},
                              {"frame": 32, "label": "thing", "box": [2, 2, 20, 20], "track_id": 5}])
    AgentSession(ds.path, "second").propose_spans("clip", [{"start_frame": 3, "end_frame": 8, "label": "other"}])
    assert c.get("/api/dataset").json()["proposals"] == 3
    assert post(c, "/api/proposals/adopt", {}).status_code == 400                           # which agent?
    assert post(c, "/api/proposals/adopt", {"author": "nobody"}).status_code == 400
    assert post(c, "/api/proposals/adopt", {"author": "agent:scout", "video_id": "no-such-video"}).status_code == 400
    assert post(c, "/api/proposals/discard", {"author": "agent:scout", "video_id": "no-such-video"}).status_code == 400
    assert c.get("/api/dataset").json()["proposals"] == 3                                   # nothing was lost by the refusals
    r = post(c, "/api/proposals/adopt", {"author": "agent:scout", "video_id": "clip"})
    assert r.status_code == 200 and r.json()["taken"] == 2
    d = c.get("/api/draft", headers=H).json()
    assert d["summary"]["added"] == 2 and len(d["ops"]) == 2
    assert post(c, "/api/draft/undo", {}).status_code == 200                                # the person can undo a take-in step by step
    assert post(c, "/api/draft/redo", {}).status_code == 200
    r = post(c, "/api/proposals/discard", {"author": "agent:second"})
    assert r.status_code == 200 and c.get("/api/dataset").json()["proposals"] == 0
    r = post(c, "/api/draft/commit", {"message": "agent pass"})
    assert r.status_code == 200
    after = Dataset.open(ds.path)
    assert len(after.annotations("clip", track_id=5)) == 2
    assert {r["author"] for r in after.annotations("clip", track_id=5).to_pylist()} == {"agent:scout"}


# ---------------------------------------------------------------- drawing inside a view
def _bits(mask):
    from vidlabel.ui.app import mask_to_bits
    return mask_to_bits(mask)


def test_boxes_drawn_in_a_view_are_stored_in_video_pixels_and_come_back_in_view_pixels(env):
    c, ds, st = env
    ds.add_view({"view_id": "crop", "video_id": "clip", "region": {"x": 20, "y": 10, "width": 40, "height": 30}, "allowed_labels": ["thing"]})
    ds.add_view({"view_id": "big", "video_id": "clip", "region": {"x": 20, "y": 10, "width": 40, "height": 30}, "output_size": [80, 60]})
    r = post(c, "/api/draft/add", {"video_id": "clip", "row": {"kind": "box", "label": "thing", "start_frame": 25, "box": [4, 6, 24, 16], "view_id": "crop", "track_id": 7}})
    assert r.status_code == 200, r.text
    r2 = post(c, "/api/draft/add", {"video_id": "clip", "row": {"kind": "box", "label": "thing", "start_frame": 25, "box": [8, 12, 48, 32], "view_id": "big", "track_id": 8}})
    assert r2.status_code == 200
    assert post(c, "/api/draft/commit", {"message": "in views"}).status_code == 200
    rows = {x["track_id"]: x for x in Dataset.open(ds.path).annotations("clip", frames=(25, 25), view_id=["crop", "big"]).to_pylist()}
    assert (rows[7]["x0"], rows[7]["y0"], rows[7]["x1"], rows[7]["y1"]) == (24, 16, 44, 26) and rows[7]["view_id"] == "crop"
    assert (rows[8]["x0"], rows[8]["y0"], rows[8]["x1"], rows[8]["y1"]) == (24, 16, 44, 26) and rows[8]["view_id"] == "big"     # 2x view: halved
    back = c.get("/api/annotations-range/clip", params={"start": 25, "end": 25, "view_id": "crop"}).json()
    row = next(x for x in back["rows"] if x["track_id"] == 7)
    assert [row["x0"], row["y0"], row["x1"], row["y1"]] == [4, 6, 24, 16]                          # what the person drew
    big = next(x for x in c.get("/api/annotations-range/clip", params={"start": 25, "end": 25, "view_id": "big"}).json()["rows"] if x["track_id"] == 8)
    assert [big["x0"], big["y0"], big["x1"], big["y1"]] == [8, 12, 48, 32]
    assert not [x for x in c.get("/api/annotations-range/clip", params={"start": 25, "end": 25, "view_id": ""}).json()["rows"] if x["track_id"] in (7, 8)]   # not on the whole frame


def test_moving_a_box_in_a_view_converts_the_new_position(env):
    c, ds, st = env
    ds.add_view({"view_id": "crop", "video_id": "clip", "region": {"x": 20, "y": 10, "width": 40, "height": 30}})
    ds.add_annotations([{"kind": "box", "label": "thing", "start_frame": 30, "box": (24, 16, 44, 26), "view_id": "crop", "origin": "human"}], "clip")
    aid = Dataset.open(ds.path).annotations("clip", frames=(30, 30), view_id=["crop"]).to_pylist()[0]["annotation_id"]
    assert post(c, "/api/draft/update", {"video_id": "clip", "ids": [str(aid)], "values": {"box": [10, 5, 30, 15]}, "view_id": "crop"}).status_code == 200
    post(c, "/api/draft/commit", {"message": "move"})
    r = Dataset.open(ds.path).annotations("clip", frames=(30, 30), view_id=["crop"]).to_pylist()[0]
    assert (r["x0"], r["y0"], r["x1"], r["y1"]) == (30, 15, 50, 25)


def test_masks_drawn_in_a_view_land_in_the_right_place_and_can_be_edited_there(env):
    c, ds, st = env
    from vidlabel import masks
    W, H = 96, 64
    ds.add_view({"view_id": "crop", "video_id": "clip", "region": {"x": 20, "y": 10, "width": 40, "height": 30}})
    m = np.zeros((30, 40), bool)
    m[5:15, 10:30] = True
    r = post(c, "/api/draft/add", {"video_id": "clip", "row": {"kind": "mask", "label": "thing", "start_frame": 26, "mask_bits": _bits(m), "view_id": "crop", "track_id": 3}})
    assert r.status_code == 200, r.text
    post(c, "/api/draft/commit", {"message": "mask in view"})
    row = Dataset.open(ds.path).annotations("clip", frames=(26, 26), kind="mask", view_id=["crop"]).to_pylist()[0]
    full = masks.decode(row["mask"])
    assert full.shape == (H, W) and full.sum() == m.sum() and full[15:25, 30:50].all() and (row["x0"], row["y0"], row["x1"], row["y1"]) == (30, 15, 50, 25)
    got = c.get(f"/api/mask-bits/clip/{row['annotation_id']}", params={"frame": 26, "view_id": "crop"}).json()
    assert (got["width"], got["height"]) == (40, 30) and got == _bits(m)                    # round trip is exact

    # editing it in the view replaces it (everything in a view lies inside the view, so nothing outside can be lost)
    edited = np.zeros((30, 40), bool)
    edited[0:5, 0:5] = True
    assert post(c, "/api/draft/update", {"video_id": "clip", "ids": [str(row["annotation_id"])], "values": {"mask_bits": _bits(edited)}, "frame": 26, "view_id": "crop"}).status_code == 200
    post(c, "/api/draft/commit", {"message": "edit"})
    new = masks.decode(Dataset.open(ds.path).annotations("clip", frames=(26, 26), kind="mask", view_id=["crop"]).to_pylist()[0]["mask"])
    assert new.sum() == 25 and new[10:15, 20:25].all() and (new[:10] == 0).all()


def test_a_resized_view_takes_new_masks_but_refuses_to_edit_old_ones(env):
    c, ds, st = env
    ds.add_view({"view_id": "big", "video_id": "clip", "region": {"x": 20, "y": 10, "width": 40, "height": 30}, "output_size": [80, 60]})
    m = np.zeros((60, 80), bool)
    m[10:30, 20:60] = True
    assert post(c, "/api/draft/add", {"video_id": "clip", "row": {"kind": "mask", "label": "thing", "start_frame": 28, "mask_bits": _bits(m), "view_id": "big"}}).status_code == 200
    post(c, "/api/draft/commit", {"message": "m"})
    row = Dataset.open(ds.path).annotations("clip", frames=(28, 28), kind="mask", view_id=["big"]).to_pylist()[0]
    assert (row["x0"], row["y0"], row["x1"], row["y1"]) == (30, 15, 50, 25)
    r = post(c, "/api/draft/update", {"video_id": "clip", "ids": [str(row["annotation_id"])], "values": {"mask_bits": _bits(m)}, "frame": 28, "view_id": "big"})
    assert r.status_code == 400 and "resized" in r.text and "whole frame" in r.text
    assert post(c, "/api/draft/discard", {}).status_code == 200


def test_view_rules_still_apply_to_drawn_rows(env):
    c, ds, st = env
    ds.add_view({"view_id": "crop", "video_id": "clip", "region": {"x": 20, "y": 10, "width": 40, "height": 30}, "allowed_labels": ["thing"], "frame_range": [10, 20]})
    ok = {"kind": "box", "label": "thing", "start_frame": 15, "box": [1, 1, 9, 9], "view_id": "crop"}
    assert post(c, "/api/draft/add", {"video_id": "clip", "row": {**ok, "label": "other"}}).status_code == 400          # label not allowed in this view
    assert post(c, "/api/draft/add", {"video_id": "clip", "row": {**ok, "start_frame": 30}}).status_code == 400         # frame outside the view's range
    assert post(c, "/api/draft/add", {"video_id": "clip", "row": {**ok, "view_id": "nope"}}).status_code in (400, 404)       # no such view
    ds.add_video({"video_id": "w", "frame_count": 5, "fps": {"num": 25, "den": 1}, "width": 32, "height": 32})
    assert post(c, "/api/draft/add", {"video_id": "w", "row": {**ok, "start_frame": 2}}).status_code == 400            # a view of another video
    assert post(c, "/api/draft/add", {"video_id": "clip", "row": ok}).status_code == 200
    assert post(c, "/api/draft/add", {"video_id": "clip", "row": {**ok, "track_id": 1}}).status_code == 200
    assert post(c, "/api/draft/add", {"video_id": "clip", "row": {**ok, "track_id": 1}}).status_code == 400            # the one-box-per-track rule holds in a view too
    assert post(c, "/api/draft/add", {"video_id": "clip", "row": {"kind": "marker", "label": "thing", "start_frame": 11, "end_frame": 14, "view_id": "crop"}}).status_code == 200
    post(c, "/api/draft/discard", {})


def test_sample_inspector_gives_objects_boxes_in_pixels_masks_and_neighbours(env):
    c, ds, st = env
    base = {"video_id": "clip", "release": "none", "kind": ["box", "mask"]}
    s = c.get("/api/sample", params={**base, "frame": 5}).json()
    assert s["image_size"] == [96, 64] and s["frame_count"] == 40 and s["in_samples"] is True
    assert s["objects"] and all({"index", "class_id", "class_name", "track_id", "annotation_id", "box", "target_box"} <= set(o) for o in s["objects"])
    o = s["objects"][0]
    assert o["class_name"] in ("thing", "other") and o["box"] == [5, 5, 40, 40] and isinstance(o["annotation_id"], str)
    assert s["previous_sample"] == 4 and s["next_sample"] == 6
    first = c.get("/api/sample", params={**base, "frame": 0}).json()
    assert first["previous_sample"] is None and first["next_sample"] == 1
    last = c.get("/api/sample", params={**base, "frame": 39}).json()
    assert last["next_sample"] is None or last["in_samples"] is False
    gap = c.get("/api/sample", params={**base, "frame": 20}).json()                                  # a frame with nothing on it
    assert gap["in_samples"] is False and gap["objects"] == [] and gap["previous_sample"] is not None
    # the box in the loader's format, and the same box back in pixels
    for fmt, norm, want in (("xywh", False, [5, 5, 35, 35]), ("cxcywh", False, [22.5, 22.5, 35, 35]), ("xyxy", True, [5 / 96, 5 / 64, 40 / 96, 40 / 64])):
        t = c.get("/api/sample", params={**base, "frame": 5, "box_format": fmt, "normalized": "1" if norm else ""}).json()["objects"][0]
        assert t["box"] == [5, 5, 40, 40] and np.allclose(t["target_box"], want, atol=1e-3), (fmt, norm, t)
    only_boxes = c.get("/api/sample", params={"video_id": "clip", "release": "none", "kind": ["box"], "frame": 5}).json()
    assert only_boxes["mask_pngs"] == [] and "masks" not in only_boxes["targets"]
    on3 = c.get("/api/sample", params={**base, "frame": 3}).json()                                   # frame 3 has a box and a mask
    assert len(on3["objects"]) == 2 and len(on3["mask_pngs"]) == 1 and on3["mask_pngs"][0]["png"].startswith("data:image/png;base64,")
    mp = on3["mask_pngs"][0]
    assert (mp["x"], mp["y"], mp["w"], mp["h"]) == (30, 20, 20, 10) and on3["objects"][mp["index"]]["class_name"] == "other"
    assert c.get("/api/sample", params={**base, "frame": 500}).status_code in (400, 404)
    assert c.get("/api/sample", params={**base, "frame": 5, "video_id": "nope"}).status_code in (400, 404)
    assert c.get("/api/sample", params={**base, "frame": 5, "box_format": "zzz"}).status_code == 400
    assert c.get("/api/sample", params={**base, "frame": 5, "release": "r-does-not-exist"}).status_code in (400, 404)
