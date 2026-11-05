"""The draft: pending edits, undo, overlay, commit (all or nothing), conflicts, history and revert."""
import hashlib
import json

import numpy as np
import pytest
from vidlabel import Dataset, VideoMetadata
from vidlabel.ui import draft as D
from vidlabel.validate import ValidationError


def _hash(ds, vid="v"):
    return hashlib.sha256(ds._ann_path(vid).read_bytes()).hexdigest()


@pytest.fixture
def ds(tmp_path):
    ds = Dataset.create(tmp_path / "d")
    for vid in ("v", "w"):
        ds.add_video(VideoMetadata(video_id=vid, frame_count=100, fps={"num": 25, "den": 1}, width=64, height=48))
    ds.add_labels(["thing", "other"])
    ds.add_annotations([{"kind": "box", "label": "thing", "start_frame": f, "box": (5, 5, 20, 20), "track_id": 1,
                         "origin": "model", "confidence": 0.9} for f in range(0, 20, 2)]
                       + [{"kind": "box", "label": "other", "start_frame": f, "box": (30, 10, 50, 30), "track_id": 2,
                           "origin": "model"} for f in range(0, 20, 2)]
                       + [{"kind": "marker", "label": "thing", "start_frame": 40, "end_frame": 50, "origin": "human"}], "v")
    return ds


def test_nothing_changes_until_commit(ds):
    before = _hash(ds)
    d = D.Draft.load(ds, "ana")
    d.update_track("v", 1, status="rejected")
    d.add("v", {"kind": "box", "label": "thing", "start_frame": 60, "box": (1, 1, 9, 9)})
    d.delete("v", [ds.annotations("v", track_id=2).column("annotation_id").to_pylist()[0]])
    assert _hash(ds) == before
    assert d.summary() == {"added": 1, "modified": 10, "deleted": 1, "videos": ["v"]}


def test_rows_show_the_draft_applied(ds):
    d = D.Draft.load(ds, "ana")
    tmp = d.add("v", {"kind": "box", "label": "thing", "start_frame": 60, "box": (1, 1, 9, 9), "track_id": 9})
    d.update_track("v", 1, status="rejected")
    rows = d.rows("v")
    assert {r["_pending"] for r in rows if r["track_id"] == 1} == {"modified"} and all(r["status"] == "rejected" for r in rows if r["track_id"] == 1)
    assert [r for r in rows if r["annotation_id"] == tmp][0]["_pending"] == "added"
    # filters follow the pending values: rows moved into "rejected" appear under that filter, and leave "unreviewed"
    assert len(d.rows("v", status="rejected")) == 10
    assert all(r["track_id"] != 1 for r in d.rows("v", status="unreviewed"))
    assert [r["annotation_id"] for r in d.rows("v", frames=(60, 60))] == [tmp]
    assert all("_pending" not in r for r in D.Draft.load(ds, "bob").rows("v"))      # another author sees the dataset as it is


def test_undo_redo_and_new_edit_ends_redo(ds):
    d = D.Draft.load(ds, "ana")
    d.update_track("v", 1, status="rejected")
    d.update_track("v", 2, status="approved")
    d.undo()
    assert d.summary()["modified"] == 10 and d.can_redo
    d.redo()
    assert d.summary()["modified"] == 20
    d.undo(); d.undo()
    assert d.summary() == {"added": 0, "modified": 0, "deleted": 0, "videos": []}
    d.redo()
    d.update_track("v", 2, status="rejected")           # a new edit drops the redo history
    assert not d.can_redo and len(d.ops) == 2
    with pytest.raises(D.DraftError, match="nothing to redo"):
        d.redo()


def test_draft_survives_reload_and_authors_are_separate(ds):
    D.Draft.load(ds, "ana").update_track("v", 1, status="approved")
    assert D.Draft.load(ds, "ana").summary()["modified"] == 10
    assert D.Draft.load(ds, "bob").summary()["modified"] == 0
    assert (ds.path / ".vidlabel-ui" / "drafts" / "ana.json").exists()


def test_commit_applies_everything_and_records_history(ds):
    d = D.Draft.load(ds, "ana")
    d.update_track("v", 1, status="rejected")
    gone = ds.annotations("v", track_id=2).column("annotation_id").to_pylist()[:2]
    d.delete("v", gone)
    d.add("v", {"kind": "box", "label": "other", "start_frame": 70, "box": (2, 2, 8, 8), "track_id": 7, "confidence": 0.5})
    assert d.check()["ok"]
    rec = d.commit("review round one")
    assert rec["seq"] == 1 and rec["author"] == "ana" and rec["summary"] == {"added": 1, "modified": 10, "deleted": 2, "videos": ["v"]}
    t = ds.annotations("v")
    assert t.num_rows == 21 - 2 + 1
    assert set(ds.annotations("v", track_id=1).column("status").to_pylist()) == {"rejected"}
    assert not set(gone) & set(t.column("annotation_id").to_pylist())
    new = ds.annotations("v", track_id=7).to_pylist()[0]
    assert new["author"] == "ana" and new["origin"] == "human" and new["annotation_id"] in rec["videos"]["v"]["added_ids"]
    assert not d.ops and not d.path.exists()                        # the draft is empty after a commit
    assert [c["message"] for c in D.history(ds)] == ["review round one"]
    assert ds.validate() == []


def test_commit_needs_a_message_and_something_to_commit(ds):
    d = D.Draft.load(ds, "ana")
    with pytest.raises(D.DraftError, match="nothing to commit"):
        d.commit("x")
    d.update_track("v", 1, status="approved")
    with pytest.raises(D.DraftError, match="commit message"):
        d.commit("  ")


def test_revert_restores_exact_rows_and_can_be_reverted(ds):
    original = {r["annotation_id"]: r for r in ds.annotations("v").to_pylist()}
    d = D.Draft.load(ds, "ana")
    d.update_track("v", 1, status="rejected")
    d.delete_track("v", 2)
    d.add("v", {"kind": "marker", "label": "thing", "start_frame": 80, "end_frame": 85})
    d.commit("big edit")
    assert ds.annotations("v").num_rows == 11 + 1
    rec = D.revert(ds, 1, "bob")
    assert rec["kind"] == "revert" and rec["reverts"] == 1 and rec["message"].startswith("Revert #1")
    now = {r["annotation_id"]: r for r in ds.annotations("v").to_pylist()}
    assert now == original                                        # same ids, same values, masks included
    again = D.revert(ds, 2, "bob")                                # reverting the revert redoes the edit
    assert ds.annotations("v").num_rows == 12 and again["reverts"] == 2
    assert ds.validate() == []


def test_revert_skips_rows_changed_since(ds):
    d = D.Draft.load(ds, "ana")
    d.update_track("v", 1, status="rejected")
    d.commit("reject track 1")
    some = ds.annotations("v", track_id=1).column("annotation_id").to_pylist()[0]
    ds.delete_annotations("v", [some])                            # changed outside
    rec = D.revert(ds, 1, "ana")
    assert any(str(some) in s for s in rec["skipped"])
    assert set(ds.annotations("v", track_id=1).column("status").to_pylist()) == {"unreviewed"}
    d2 = D.Draft.load(ds, "ana")
    d2.delete_track("v", 1)
    d2.commit("drop track 1")
    with pytest.raises(D.DraftError, match="nothing left to revert"):
        D.revert(ds, 1, "ana")


def test_conflict_when_the_video_changed_underneath(ds):
    d = D.Draft.load(ds, "ana")
    d.update_track("v", 1, status="rejected")
    ds.update_annotations("v", ds.annotations("v", track_id=2).column("annotation_id").to_pylist()[:1], status="approved")   # a CLI user
    with pytest.raises(D.Conflict) as e:
        d.commit("mine")
    assert e.value.videos == ["v"] and d.ops                       # nothing applied, draft kept
    rec = d.commit("mine again", reapply=True)                    # nothing it needs is gone, so it goes through
    assert rec["summary"]["modified"] == 10


def test_reapply_skips_edits_to_rows_that_no_longer_exist(ds):
    d = D.Draft.load(ds, "ana")
    ids = ds.annotations("v", track_id=1).column("annotation_id").to_pylist()
    d.update("v", ids, status="approved")
    ds.delete_annotations("v", ids[:3])
    with pytest.raises(D.Conflict):
        d.check()
    r = d.check(reapply=True)
    assert len(r["skipped"]) == 3 and r["modified"] == 7
    rec = d.commit("x", reapply=True)
    assert len(rec["skipped"]) == 3


def test_a_failed_commit_puts_everything_back_and_keeps_the_draft(ds, monkeypatch):
    ds.add_annotations([{"kind": "box", "label": "thing", "start_frame": 1, "box": (1, 1, 5, 5), "origin": "model"}], "w")
    before = {v: _hash(ds, v) for v in ("v", "w")}
    d = D.Draft.load(ds, "ana")
    d.update_track("v", 1, status="rejected")
    d.delete("w", ds.annotations("w").column("annotation_id").to_pylist())
    d.add("w", {"kind": "box", "label": "thing", "start_frame": 3, "box": (1, 1, 5, 5)})
    real = Dataset.add_annotations

    def boom(self, rows, video_id=None, **kw):
        if self.path == ds.path:                                  # fail on the real dataset (after the deletes ran), not the dry run
            raise RuntimeError("disk full")
        return real(self, rows, video_id, **kw)

    monkeypatch.setattr(Dataset, "add_annotations", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        d.commit("will fail")
    monkeypatch.undo()
    assert {v: _hash(ds, v) for v in ("v", "w")} == before         # both videos exactly as they were
    assert d.ops and D.history(ds) == []


def test_problems_are_found_before_anything_is_written(ds):
    d = D.Draft.load(ds, "ana")
    # the draft itself refuses a second box for track 1 on frame 0, so put one in the journal directly: the commit must still catch it
    d._record({"op": "add", "video_id": "v", "tmp": "p1", "label": "forced", "row": {"kind": "box", "label": "thing", "start_frame": 0, "end_frame": 0,
               "x0": 1, "y0": 1, "x1": 5, "y1": 5, "track_id": 1, "origin": "human", "author": "ana"}})
    before = _hash(ds)
    with pytest.raises(ValidationError, match="duplicate_track_in_frame|track"):
        d.commit("clash")
    assert _hash(ds) == before and d.ops and D.history(ds) == []


def test_bad_edits_are_refused_when_recorded(ds):
    d = D.Draft.load(ds, "ana")
    with pytest.raises(D.DraftError, match="vocabulary|label"):
        d.add("v", {"kind": "box", "label": "nope", "start_frame": 1, "box": (1, 1, 5, 5)})
    with pytest.raises(D.DraftError):
        d.add("v", {"kind": "box", "label": "thing", "start_frame": 999, "box": (1, 1, 5, 5)})
    with pytest.raises(D.DraftError, match="no such annotation"):
        d.update("v", [12345], status="approved")
    with pytest.raises(D.DraftError, match="status"):
        d.update_track("v", 1, status="maybe")
    with pytest.raises(D.DraftError, match="cannot change"):
        d.update_track("v", 1, annotation_id=5)
    with pytest.raises(D.DraftError, match="no rows for track 99"):
        d.delete_track("v", 99)
    with pytest.raises(KeyError):
        d.add("nope", {"kind": "marker", "label": "thing", "start_frame": 0})
    assert d.ops == []


def test_pending_rows_can_be_changed_and_dropped(ds):
    d = D.Draft.load(ds, "ana")
    t1 = d.add("v", {"kind": "box", "label": "thing", "start_frame": 60, "box": (1, 1, 9, 9)})
    t2 = d.add("v", {"kind": "box", "label": "thing", "start_frame": 61, "box": (1, 1, 9, 9)})
    d.update("v", [t1], box=(2, 2, 12, 12), status="approved")
    d.delete("v", [t2])
    n = d.net()["v"]
    assert len(n["adds"]) == 1 and n["adds"][0]["x0"] == 2 and n["adds"][0]["status"] == "approved"
    assert d.summary()["added"] == 1


def test_discard_one_edit_takes_its_dependents(ds):
    d = D.Draft.load(ds, "ana")
    t = d.add("v", {"kind": "box", "label": "thing", "start_frame": 60, "box": (1, 1, 9, 9)})
    d.update("v", [t], status="approved")
    d.update_track("v", 1, status="rejected")
    assert d.discard(0) == 2                                       # the add and the change that used it
    assert [o["kind"] for o in d.state()["ops"]] == ["update"] and d.summary()["added"] == 0
    assert d.summary()["modified"] == 10


def test_masks_round_trip_through_a_draft_and_history(ds):
    m = np.zeros((48, 64), bool)
    m[10:20, 5:25] = True
    d = D.Draft.load(ds, "ana")
    t = d.add("v", {"kind": "mask", "label": "thing", "start_frame": 70, "mask": m})
    pending = [r for r in d.rows("v") if r["annotation_id"] == t][0]
    assert (pending["x0"], pending["y0"], pending["x1"], pending["y1"]) == (5, 10, 25, 20)
    d2 = D.Draft.load(ds, "ana")                                   # reloaded from disk: the mask survived as base64
    assert d2.rows("v", kind="mask")[0]["mask"] == pending["mask"]
    d.commit("a mask")
    stored = ds.annotations("v", kind="mask").to_pylist()[0]
    from vidlabel import masks
    assert (masks.decode(stored["mask"]) == m).all()
    d3 = D.Draft.load(ds, "ana"); d3.delete("v", [stored["annotation_id"]]); d3.commit("remove it")
    D.revert(ds, 2, "ana")                                         # the mask comes back byte for byte
    back = ds.annotations("v", kind="mask").to_pylist()[0]
    assert back["mask"] == stored["mask"] and back["annotation_id"] == stored["annotation_id"]


def test_history_keeps_the_last_fifty(ds):
    for i in range(53):
        d = D.Draft.load(ds, "ana")
        d.add("v", {"kind": "marker", "label": "thing", "start_frame": i % 90})
        d.commit(f"commit {i}")
    h = D.history(ds)
    assert len(h) == 50 and h[0]["seq"] == 53 and h[-1]["seq"] == 4
    assert not (ds.path / ".vidlabel-ui" / "history" / "000001.json").exists()
    with pytest.raises(D.DraftError, match="no commit number 1"):
        D.revert(ds, 1)


def test_the_state_folder_does_not_upset_the_validator(ds):
    D.Draft.load(ds, "ana").update_track("v", 1, status="approved")
    assert ds.validate() == []
    assert json.loads((ds.path / ".vidlabel-ui" / "drafts" / "ana.json").read_text())["ops"][0]["op"] == "update"


def test_big_ids_are_kept_exactly(ds):
    big = max(ds.annotations("v").column("annotation_id").to_pylist())
    d = D.Draft.load(ds, "ana")
    d.delete("v", [big])
    rec = d.commit("big id")
    assert rec["videos"]["v"]["deleted_ids"] == [big]
    D.revert(ds, 1)
    assert big in ds.annotations("v").column("annotation_id").to_pylist()


def test_a_track_cannot_get_two_boxes_on_one_frame_in_the_draft(ds):
    d = D.Draft.load(ds, "ana")
    with pytest.raises(D.DraftError, match="already has a box on frame 4"):           # a committed box is there
        d.add("v", {"kind": "box", "label": "thing", "start_frame": 4, "box": (1, 1, 9, 9), "track_id": 1})
    d.add("v", {"kind": "box", "label": "thing", "start_frame": 5, "box": (1, 1, 9, 9), "track_id": 1})
    with pytest.raises(D.DraftError, match="already has a box on frame 5"):           # and so is a pending one
        d.add("v", {"kind": "box", "label": "thing", "start_frame": 5, "box": (2, 2, 9, 9), "track_id": 1})
    d.add("v", {"kind": "box", "label": "thing", "start_frame": 5, "box": (2, 2, 9, 9), "track_id": 7})     # another track is fine
    d.add("v", {"kind": "box", "label": "thing", "start_frame": 5, "box": (2, 2, 9, 9)})                    # so is no track at all
    d.add("v", {"kind": "mask", "label": "thing", "start_frame": 4, "mask": np.ones((48, 64), bool), "track_id": 1})   # a mask is another kind
    assert d.summary()["added"] == 4


def test_deleting_a_box_frees_its_place_and_track_changes_are_checked(ds):
    d = D.Draft.load(ds, "ana")
    rid = [r for r in ds.annotations("v", track_id=1).to_pylist() if r["start_frame"] == 4][0]["annotation_id"]
    d.delete("v", [rid])
    d.add("v", {"kind": "box", "label": "thing", "start_frame": 4, "box": (1, 1, 9, 9), "track_id": 1})      # the place is free again
    d2 = D.Draft.load(ds, "bob")
    other = ds.annotations("v", track_id=2).to_pylist()[0]["annotation_id"]
    with pytest.raises(D.DraftError, match="track 1 would have two boxes on frame 0"):                      # merging track 2 into track 1 collides
        d2.update("v", ds.annotations("v", track_id=2).column("annotation_id").to_pylist(), track_id=1)
    with pytest.raises(D.DraftError, match="two boxes"):
        d2.update("v", [other], track_id=1)
    d2.update("v", [other], track_id=5)                                                                      # a free number is fine
    assert d2.summary()["modified"] == 1
