"""What an agent may do: read, propose into its own draft, and nothing else. No MCP package is needed for these tests."""
import hashlib
import shutil

import numpy as np
import pytest
from vidlabel import Dataset, VideoMetadata
from vidlabel.agent import MAX_PENDING_ROWS, AgentError, AgentSession
from vidlabel.ui import draft as D

HAVE_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


def _hash(ds):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ds.path.rglob("*.parquet"))}


@pytest.fixture
def ds(tmp_path):
    ds = Dataset.create(tmp_path / "d")
    ds.add_video(VideoMetadata(video_id="v", frame_count=100, fps={"num": 25, "den": 1}, width=64, height=48))
    ds.add_labels(["thing", "other"])
    ds.add_annotations([{"kind": "box", "label": "thing", "start_frame": f, "box": (5, 5, 20, 20), "track_id": 1, "origin": "model",
                         "confidence": 0.9} for f in range(0, 10)], "v")
    return ds


@pytest.fixture
def ag(ds):
    return AgentSession(ds.path, "scout")


def test_overview_tells_the_agent_the_rules(ag):
    o = ag.overview()
    assert o["your_author_name"] == "agent:scout" and o["videos"][0]["width"] == 64 and "cannot approve" in o["rules"]
    assert {l["name"] for l in o["labels"]} == {"thing", "other"}


def test_proposals_go_to_the_agents_draft_only_as_model_and_unreviewed(ds, ag):
    before = _hash(ds)
    r = ag.propose_boxes("v", [{"frame": 20, "label": "thing", "box": [1, 1, 30, 30], "track_id": 2, "confidence": 0.7},
                               {"frame": 21, "label": "thing", "box": [2, 2, 30, 30], "track_id": 2}], note="found by the detector")
    assert r["recorded"] == 2 and r["ids"] == ["p1", "p2"] and r["pending"]["added"] == 2
    assert _hash(ds) == before                                                       # the dataset itself is untouched
    rows = D.Draft.load(ds, "agent:scout").rows("v", include_deleted=False, frames=(20, 21))
    assert {(x["origin"], x["status"], x["author"]) for x in rows} == {("model", "unreviewed", "agent:scout")}
    assert D.Draft.load(ds, "ana").summary()["added"] == 0                          # a person's own draft does not see them
    assert ag.draft_status()["edits"][0]["note"] == "found by the detector"


def test_an_agent_cannot_set_status_origin_or_author(ag):
    for extra in ({"status": "approved"}, {"origin": "human"}, {"author": "ana"}):
        with pytest.raises(AgentError, match="not allowed here"):
            ag.propose_boxes("v", [{"frame": 30, "label": "thing", "box": [1, 1, 9, 9], **extra}])
    assert ag.draft_status()["summary"]["added"] == 0


def test_a_batch_is_all_or_nothing(ds, ag):
    ag.propose_boxes("v", [{"frame": 40, "label": "thing", "box": [1, 1, 9, 9]}])
    bad = [{"frame": 41, "label": "thing", "box": [1, 1, 9, 9]},
           {"frame": 42, "label": "no-such-label", "box": [1, 1, 9, 9]}]
    with pytest.raises(AgentError, match="item 1.*Nothing from this call was recorded"):
        ag.propose_boxes("v", bad)
    assert ag.draft_status()["summary"]["added"] == 1                                # still only the first, earlier call
    for items in ([{"frame": 50, "label": "thing", "box": [1, 1, 900, 9]}],           # outside the frame
                  [{"frame": 500, "label": "thing", "box": [1, 1, 9, 9]}],            # no such frame
                  [{"frame": 51, "label": "thing", "box": [1, 1, 9, 9], "confidence": 3}]):
        with pytest.raises(AgentError):
            ag.propose_boxes("v", items)
    assert ag.draft_status()["summary"]["added"] == 1


def test_a_track_cannot_get_two_boxes_on_one_frame(ag):
    with pytest.raises(AgentError, match="already has a box on frame 3"):           # a committed box is there
        ag.propose_boxes("v", [{"frame": 3, "label": "thing", "box": [1, 1, 9, 9], "track_id": 1}])
    with pytest.raises(AgentError, match="already has a box on frame 60"):          # two in one batch
        ag.propose_boxes("v", [{"frame": 60, "label": "thing", "box": [1, 1, 9, 9], "track_id": 4},
                               {"frame": 60, "label": "thing", "box": [2, 2, 9, 9], "track_id": 4}])
    assert ag.draft_status()["summary"]["added"] == 0


def test_spans_and_polygon_masks(ds, ag):
    ag.propose_spans("v", [{"start_frame": 10, "end_frame": 20, "label": "thing", "confidence": 0.6}])
    ag.propose_masks("v", [{"frame": 30, "label": "other", "polygon": [[10, 10], [40, 10], [40, 30], [10, 30]]}])
    rows = {r["kind"]: r for r in D.Draft.load(ds, "agent:scout").rows("v", include_deleted=False) if r.get("_pending")}
    assert rows["marker"]["end_frame"] == 20 and rows["mask"]["mask"] is not None
    from vidlabel import masks
    m = masks.decode(bytes(rows["mask"]["mask"]))
    assert m.shape == (48, 64) and m[20, 25] and not m[2, 2]
    for poly in ([[1, 1], [2, 2]], [[1, 1], [2, 2], [999, 3]], "nonsense", [[5, 5], [5, 5], [5, 5]]):
        with pytest.raises(AgentError):
            ag.propose_masks("v", [{"frame": 31, "label": "other", "polygon": poly}])
    with pytest.raises(AgentError, match="end_frame < start_frame"):
        ag.propose_spans("v", [{"start_frame": 20, "end_frame": 10, "label": "thing"}])


def test_listing_shows_pending_rows_and_filters(ag):
    ag.propose_boxes("v", [{"frame": 70, "label": "other", "box": [1, 1, 9, 9]}])
    r = ag.list_annotations("v", label="other")
    assert r["total"] == 1 and r["rows"][0]["pending"] == "added" and isinstance(r["rows"][0]["annotation_id"], str) and "mask" not in r["rows"][0]
    r = ag.list_annotations("v", frame_start=0, frame_end=4, limit=2)
    assert r["total"] == 5 and r["returned"] == 2
    with pytest.raises(AgentError):
        ag.list_annotations("nope")
    assert ag.list_tracks("v")["tracks"][0]["track_id"] == 1


def test_retract_only_takes_back_your_own_pending_proposals(ds, ag):
    ids = ag.propose_boxes("v", [{"frame": 80, "label": "thing", "box": [1, 1, 9, 9]}, {"frame": 81, "label": "thing", "box": [1, 1, 9, 9]}])["ids"]
    committed = ds.annotations("v").column("annotation_id").to_pylist()[0]
    with pytest.raises(AgentError, match="not pending proposals of yours"):
        ag.retract([str(committed)])                                                 # a row that is in the dataset
    with pytest.raises(AgentError, match="not pending proposals of yours"):
        ag.retract([ids[0], "p99"])                                                  # one good id does not slip a bad one through
    assert ag.draft_status()["summary"]["added"] == 2
    assert ag.retract([ids[0]])["pending"]["added"] == 1


def test_one_agent_cannot_pile_up_unlimited_proposals(ag, monkeypatch):
    import vidlabel.agent as A
    monkeypatch.setattr(A, "MAX_PENDING_ROWS", 3)
    ag.propose_boxes("v", [{"frame": f, "label": "thing", "box": [1, 1, 9, 9]} for f in range(50, 53)])
    with pytest.raises(AgentError, match="limit is 3"):
        ag.propose_boxes("v", [{"frame": 55, "label": "thing", "box": [1, 1, 9, 9]}])
    assert MAX_PENDING_ROWS == 20000


def test_a_person_can_still_commit_what_an_agent_proposed(ds, ag):
    """The proposals are ordinary draft rows, so the normal commit path (checked, all or nothing, with history) accepts them."""
    ag.propose_boxes("v", [{"frame": 90, "label": "thing", "box": [1, 1, 9, 9], "track_id": 5, "confidence": 0.8}])
    D.Draft.load(ds, "agent:scout").commit("agent proposals")
    t = ds.annotations("v", track_id=5).to_pylist()
    assert len(t) == 1 and t[0]["origin"] == "model" and t[0]["status"] == "unreviewed" and t[0]["author"] == "agent:scout"


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg")
def test_frame_image_draws_the_annotations_including_pending_ones(tmp_path):
    import cv2
    from test_ui_server import make_video
    from vidlabel.probe import probe_video
    vp = make_video(tmp_path / "clip.mp4", n=20)
    ds = Dataset.create(tmp_path / "d2")
    ds.add_video(probe_video(vp, "clip"))
    ds.add_labels(["thing"])
    ag = AgentSession(ds.path, "scout")
    plain = cv2.imdecode(np.frombuffer(ag.frame_image("clip", 5, annotations=False), np.uint8), 1)
    assert plain.shape[:2] == (64, 96)
    ag.propose_boxes("clip", [{"frame": 5, "label": "thing", "box": [30, 20, 80, 55]}])
    drawn = cv2.imdecode(np.frombuffer(ag.frame_image("clip", 5), np.uint8), 1)
    assert np.abs(drawn.astype(int) - plain.astype(int)).sum() > 1000                # the pending box is on the picture
    assert max(cv2.imdecode(np.frombuffer(ag.frame_image("clip", 5, max_side=48), np.uint8), 1).shape[:2]) == 48
    with pytest.raises(AgentError, match="no frame 99"):
        ag.frame_image("clip", 99)


# ---------------------------------------------------------------- taking an agent's proposals into a person's draft
def _propose_some(ag):
    ag.propose_boxes("v", [{"frame": 40, "label": "thing", "box": [1, 1, 30, 30], "track_id": 3, "confidence": 0.7},
                           {"frame": 41, "label": "thing", "box": [2, 2, 30, 30], "track_id": 3}], note="detector pass 1")
    ag.propose_masks("v", [{"frame": 42, "label": "other", "polygon": [[10, 10], [40, 10], [40, 30]]}])
    ag.propose_spans("v", [{"start_frame": 50, "end_frame": 60, "label": "thing"}])


def test_a_person_sees_agent_proposals_but_not_other_peoples_drafts(ds, ag):
    _propose_some(ag)
    D.Draft.load(ds, "bob").add("v", {"kind": "box", "label": "thing", "start_frame": 70, "box": (1, 1, 9, 9)})
    p = D.proposals(ds, "ana")
    assert [x["author"] for x in p] == ["agent:scout"] and p[0]["is_agent"] and p[0]["adds"] == 4 and p[0]["by_video"] == {"v": 4}
    assert p[0]["rows"][0]["note"] == "detector pass 1" and {r["kind"] for r in p[0]["rows"]} == {"box", "marker", "mask"}
    assert D.proposals(ds, "agent:scout") == []                                  # an agent does not see itself
    with pytest.raises(D.DraftError, match="nothing waiting"):
        D.adopt(ds, "ana", "bob")                                                # a person's draft cannot be taken or discarded
    with pytest.raises(D.DraftError, match="nothing waiting"):
        D.discard_proposals(ds, "ana", "bob")


def test_adopting_moves_checked_rows_into_my_draft_and_commit_keeps_the_agents_name(ds, ag):
    before = _hash(ds)
    _propose_some(ag)
    r = D.adopt(ds, "ana", "agent:scout")
    assert r["taken"] == 4 and _hash(ds) == before                               # still nothing in the dataset
    mine = D.Draft.load(ds, "ana")
    rows = [x for x in mine.rows("v", include_deleted=False) if x.get("_pending") == "added"]
    assert {x["kind"] for x in rows} == {"box", "marker", "mask"} and all(x["origin"] == "model" and x["author"] == "agent:scout" for x in rows)
    assert next(x for x in rows if x["kind"] == "mask")["x0"] is not None        # the box of the mask was worked out again
    assert D.proposals(ds, "ana") == []                                          # they left the agent's draft
    mine.commit("take in agent proposals")
    t = ds.annotations("v", track_id=3).to_pylist()
    assert len(t) == 2 and {x["origin"] for x in t} == {"model"} and {x["status"] for x in t} == {"unreviewed"}


def test_adopting_is_all_or_nothing_and_rechecked_against_my_draft(ds, ag):
    _propose_some(ag)
    mine = D.Draft.load(ds, "ana")
    mine.add("v", {"kind": "box", "label": "thing", "start_frame": 41, "box": (1, 1, 9, 9), "track_id": 3})   # track 3 now has frame 41
    snap = (list(mine.ops), mine.cursor)
    with pytest.raises(D.DraftError, match="cannot be taken in.*already has a box on frame 41.*Nothing was taken in"):
        D.adopt(ds, "ana", "agent:scout")
    again = D.Draft.load(ds, "ana")
    assert (again.ops, again.cursor) == snap                                      # my draft is exactly as it was
    assert D.proposals(ds, "ana")[0]["adds"] == 4                                 # and the agent still has all of them


def test_adopt_one_video_only_and_discard(ds, ag):
    ds.add_video(VideoMetadata(video_id="w", frame_count=30, fps={"num": 25, "den": 1}, width=64, height=48))
    _propose_some(ag)
    ag.propose_boxes("w", [{"frame": 3, "label": "thing", "box": [1, 1, 9, 9]}])
    assert D.adopt(ds, "ana", "agent:scout", "w")["taken"] == 1
    assert D.proposals(ds, "ana")[0]["by_video"] == {"v": 4}
    assert D.discard_proposals(ds, "ana", "agent:scout")["discarded"] == 4
    assert D.proposals(ds, "ana") == []
    assert D.Draft.load(ds, "ana").summary()["added"] == 1                        # what was already taken is kept


# ---------------------------------------------------------------- strictness: nothing odd gets into a draft
NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize("item, why", [
    ({"frame": 1, "label": "thing", "box": [True, 0, 5, 5]}, "box value"),
    ({"frame": 1, "label": "thing", "box": [NAN, 1, 9, 9]}, "box value"),
    ({"frame": 1, "label": "thing", "box": [1, 1, INF, 9]}, "box value"),
    ({"frame": 1, "label": "thing", "box": [9, 9, 1, 1]}, "x1 > x0"),
    ({"frame": 1, "label": "thing", "box": [5, 5, 5, 5]}, "x1 > x0"),
    ({"frame": 1, "label": "thing", "box": [1, 1, 9]}, "must be"),
    ({"frame": 1, "label": "thing", "box": "1,1,9,9"}, "must be"),
    ({"frame": 1.5, "label": "thing", "box": [1, 1, 9, 9]}, "whole number"),
    ({"frame": "3", "label": "thing", "box": [1, 1, 9, 9]}, "whole number"),
    ({"frame": True, "label": "thing", "box": [1, 1, 9, 9]}, "whole number"),
    ({"frame": -1, "label": "thing", "box": [1, 1, 9, 9]}, "out of range"),
    ({"frame": 100, "label": "thing", "box": [1, 1, 9, 9]}, "does not exist"),
    ({"frame": 10 ** 30, "label": "thing", "box": [1, 1, 9, 9]}, "out of range"),
    ({"frame": 1, "label": "nope", "box": [1, 1, 9, 9]}, "vocab"),
    ({"frame": 1, "label": 5, "box": [1, 1, 9, 9]}, "label must be a text"),
    ({"frame": 1, "label": ["thing"], "box": [1, 1, 9, 9]}, "label must be a text"),
    ({"frame": 1, "label": "thing", "box": [1, 1, 9, 9], "confidence": 1.5}, "between 0 and 1"),
    ({"frame": 1, "label": "thing", "box": [1, 1, 9, 9], "confidence": -0.1}, "between 0 and 1"),
    ({"frame": 1, "label": "thing", "box": [1, 1, 9, 9], "confidence": NAN}, "finite"),
    ({"frame": 1, "label": "thing", "box": [1, 1, 9, 9], "confidence": True}, "finite"),
    ({"frame": 1, "label": "thing", "box": [1, 1, 9, 9], "confidence": "0.5"}, "finite"),
    ({"frame": 1, "label": "thing", "box": [1, 1, 9, 9], "track_id": -1}, "out of range"),
    ({"frame": 1, "label": "thing", "box": [1, 1, 9, 9], "track_id": 1.5}, "whole number"),
    ({"frame": 1, "label": "thing", "box": [1, 1, 9, 9], "track_id": "a"}, "whole number"),
    ({"frame": 1, "label": "thing", "box": [1, 1, 9, 9], "track_id": True}, "whole number"),
    ({"frame": 1, "label": "thing", "box": [1, 1, 9, 9], "track_id": 10 ** 30}, "out of range"),
])
def test_odd_values_are_refused_with_a_message_and_nothing_is_recorded(ds, ag, item, why):
    before = _hash(ds)
    with pytest.raises(AgentError, match=why):
        ag.propose_boxes("v", [item])
    assert ag.draft_status()["summary"]["added"] == 0 and _hash(ds) == before


def test_whole_numbers_given_as_floats_are_fine_and_come_out_as_integers(ds, ag):
    ag.propose_boxes("v", [{"frame": 3.0, "label": "thing", "box": [1, 1.5, 9, 9.25], "track_id": 4.0, "confidence": 1}])
    r = next(x for x in D.Draft.load(ds, "agent:scout").rows("v", include_deleted=False, frames=(3, 3)) if x.get("_pending"))
    assert r["start_frame"] == 3 and r["track_id"] == 4 and r["confidence"] == 1.0 and (r["x0"], r["y0"], r["x1"], r["y1"]) == (1, 1.5, 9, 9.25)


@pytest.mark.parametrize("item, why", [
    ({"start_frame": 9, "end_frame": 3, "label": "thing"}, "end_frame < start_frame"),
    ({"start_frame": 9, "end_frame": 300, "label": "thing"}, "does not exist"),
    ({"start_frame": "a", "end_frame": 3, "label": "thing"}, "whole number"),
    ({"start_frame": 1, "end_frame": 3, "label": "thing", "confidence": 7}, "between 0 and 1"),
    ({"start_frame": 1, "end_frame": 3, "label": "thing", "box": [1, 1, 2, 2]}, "not allowed"),
    ({"start_frame": 1, "end_frame": 3, "label": "thing", "track_id": 2}, "not allowed"),
    ({"start_frame": 1, "end_frame": 3}, "missing"),
])
def test_odd_spans_are_refused(ds, ag, item, why):
    with pytest.raises(AgentError, match=why):
        ag.propose_spans("v", [item])
    assert ag.draft_status()["summary"]["added"] == 0


@pytest.mark.parametrize("poly, why", [
    ("abc", "list of"), ([[0, 0], [5], [5, 0]], "list of"), ([[0, 0], [5, 5]], "at least 3"), ([[0, 0], [NAN, 5], [5, 0]], "at least 3"),
    ([[0, 0], [5, 5], [10, 10]], "no area"), ([[0, 0], [10, 10], [10, 0], [0, 10]], "no area"),
    ([[-1, 0], [5, 5], [5, 0]], "inside the frame"), ([[0, 0], [65, 5], [5, 0]], "inside the frame"), ([[0, 0], [5, 49], [5, 0]], "inside the frame"),
    ([[1, 1], [1.2, 1], [1, 1.2]], "no area"),
])
def test_odd_polygons_are_refused(ag, poly, why):
    with pytest.raises(AgentError, match=why):
        ag.propose_masks("v", [{"frame": 5, "label": "thing", "polygon": poly}])
    assert ag.draft_status()["summary"]["added"] == 0


def test_a_polygon_becomes_the_mask_it_describes(ds, ag):
    from vidlabel import masks
    ag.propose_masks("v", [{"frame": 60, "label": "thing", "polygon": [[10, 10], [30, 10], [30, 30], [10, 30]], "track_id": 8},
                           {"frame": 61, "label": "thing", "polygon": [[0, 0], [64, 0], [64, 48], [0, 48]]}])
    D.adopt(ds, "ana", "agent:scout")
    D.Draft.load(ds, "ana").commit("take in")
    rows = {r["start_frame"]: r for r in Dataset.open(ds.path).annotations("v", kind="mask").to_pylist()}
    small = masks.decode(rows[60]["mask"])
    assert small.shape == (48, 64) and small[20, 20] and not small[5, 5] and not small[35, 35]
    assert 19 * 19 <= small.sum() <= 21 * 21 and rows[60]["track_id"] == 8
    assert (rows[60]["x0"], rows[60]["y0"]) == (10, 10) and (rows[60]["x1"], rows[60]["y1"]) in {(30, 30), (31, 31)}
    assert masks.decode(rows[61]["mask"]).all()                                  # the whole frame, corners included


def test_notes_must_be_text_and_are_shortened(ds, ag):
    with pytest.raises(AgentError, match="note must be text"):
        ag.propose_boxes("v", [{"frame": 1, "label": "thing", "box": [1, 1, 9, 9]}], note=5)
    ag.propose_boxes("v", [{"frame": 2, "label": "thing", "box": [1, 1, 9, 9]}], note="x" * 10000)
    assert len(ag.draft_status()["edits"][0]["note"]) == 300
    ag.propose_boxes("v", [{"frame": 3, "label": "thing", "box": [1, 1, 9, 9]}], note="   ")
    assert "note" not in ag.draft_status()["edits"][1]


def test_the_size_of_a_call_is_limited(ds, ag):
    ok = [{"frame": i % 100, "label": "thing", "box": [1, 1, 9, 9], "track_id": 1000 + i} for i in range(500)]
    assert ag.propose_boxes("v", ok)["recorded"] == 500
    with pytest.raises(AgentError, match="at most 500"):
        ag.propose_boxes("v", [{"frame": 1, "label": "thing", "box": [1, 1, 9, 9]}] * 501)
    with pytest.raises(AgentError, match="non-empty"):
        ag.propose_boxes("v", [])
    with pytest.raises(AgentError, match="every item"):
        ag.propose_boxes("v", [1])
    assert ag.draft_status()["summary"]["added"] == 500


# ---------------------------------------------------------------- reading
def test_listing_validates_filters_and_pages(ds, ag):
    for kw, why in (({"kind": "zzz"}, "unknown kind"), ({"status": "zzz"}, "unknown status"), ({"label": "nope"}, "no label"),
                    ({"view_id": "nov"}, "no view"), ({"track_id": "a"}, "whole number"), ({"limit": "a"}, "whole number")):
        with pytest.raises(AgentError, match=why):
            ag.list_annotations("v", **kw)
    with pytest.raises(AgentError, match="no video"):
        ag.list_annotations("nov")
    with pytest.raises(AgentError, match="no view"):
        ag.list_tracks("v", "nov")
    a = ag.list_annotations("v", limit=4)
    assert a["total"] == 10 and a["returned"] == 4 and a["offset"] == 0
    b = ag.list_annotations("v", limit=4, offset=8)
    assert b["returned"] == 2 and {r["annotation_id"] for r in a["rows"]}.isdisjoint({r["annotation_id"] for r in b["rows"]})
    assert ag.list_annotations("v", limit=-5)["returned"] == 1 and ag.list_annotations("v", offset=-3)["offset"] == 0
    assert ag.list_annotations("v", offset=10 ** 9)["rows"] == []
    assert ag.list_annotations("v", frame_start=50, frame_end=10)["total"] == 0
    assert ag.list_annotations("v", frame_start=3, frame_end=5)["total"] == 3
    assert ag.list_annotations("v", frame_start=8)["total"] == 2 and ag.list_annotations("v", frame_end=1)["total"] == 2
    assert all(isinstance(r["annotation_id"], str) and r["has_mask"] is False for r in a["rows"])
    assert ag.list_annotations("v", limit=10 ** 6)["returned"] == 10


def test_listing_marks_pending_rows_and_hides_other_authors_drafts(ds, ag):
    ag.propose_boxes("v", [{"frame": 20, "label": "thing", "box": [1, 1, 9, 9]}])
    D.Draft.load(ds, "bob").add("v", {"kind": "box", "label": "thing", "start_frame": 21, "box": (1, 1, 9, 9)})
    rows = ag.list_annotations("v", frame_start=20, frame_end=21)["rows"]
    assert [(r["start_frame"], r["pending"]) for r in rows] == [(20, "added")]
    other = AgentSession(ds.path, "other")
    assert other.list_annotations("v", frame_start=20, frame_end=21)["total"] == 0


def test_the_tracks_listing_does_not_include_unadopted_proposals_but_listing_does(ds, ag):
    ag.propose_boxes("v", [{"frame": 20, "label": "thing", "box": [1, 1, 9, 9], "track_id": 9}])
    assert {t["track_id"] for t in ag.list_tracks("v")["tracks"]} == {1}
    assert ag.list_annotations("v", track_id=9)["total"] == 1


# ---------------------------------------------------------------- who is who
def test_agent_names_never_share_a_draft_or_escape_the_folder(ds):
    from vidlabel.agent import agent_author
    names = [None, "", "agent", "scout", "Scout", "..", ".", "../../x", "a:b", "ab", "a/b", "ab ", "名前", "日本", "x" * 80, "y" * 80 + "z"]
    authors = {n: agent_author(n) for n in names}
    assert authors[None] == authors[""] == authors["agent"] == "agent:agent" and authors["ab "] == authors["ab"]
    distinct = {a.lower() for n, a in authors.items() if n not in (None, "", "ab ")}
    assert len(distinct) == len([n for n in names if n not in (None, "", "ab ")])
    assert len({a.lower() for a in (authors["scout"], authors["Scout"], authors[".."], authors["."], authors["名前"], authors["日本"], authors["a:b"], authors["ab"])}) == 8
    for n in names:
        s = AgentSession(ds.path, n)
        s.propose_spans("v", [{"start_frame": 1, "end_frame": 2, "label": "thing"}])
    drafts = ds.path / ".vidlabel-ui" / "drafts"
    files = [p for p in drafts.iterdir()]
    assert all(p.parent == drafts and p.suffix == ".json" and "/" not in p.name for p in files)
    assert len({p.name.lower() for p in files}) == len(files)                      # no two share a file, even on a case-insensitive disk


def test_a_session_picks_up_its_own_pending_work_after_a_restart(ds):
    AgentSession(ds.path, "scout").propose_spans("v", [{"start_frame": 5, "end_frame": 9, "label": "thing"}])
    again = AgentSession(ds.path, "scout")
    assert again.draft_status()["summary"]["added"] == 1 and again.overview()["your_pending_changes"]["added"] == 1
    assert again.retract(["p1"])["retracted"] == 1 and AgentSession(ds.path, "scout").draft_status()["summary"]["added"] == 0


def test_retract_is_all_or_nothing(ds, ag):
    ag.propose_boxes("v", [{"frame": 70, "label": "thing", "box": [1, 1, 9, 9]}, {"frame": 71, "label": "thing", "box": [1, 1, 9, 9]}])
    with pytest.raises(AgentError, match="not pending"):
        ag.retract(["p1", "p99"])
    assert ag.draft_status()["summary"]["added"] == 2
    with pytest.raises(AgentError):
        ag.retract("p1")                                                         # a bare string is not a list of ids
    assert ag.retract(["p1", "p1"])["retracted"] == 1 and ag.draft_status()["summary"]["added"] == 1


# ---------------------------------------------------------------- two agents, a person, a changing dataset
def test_two_agents_propose_independently_and_the_second_take_in_is_refused_cleanly(ds):
    a, b = AgentSession(ds.path, "a"), AgentSession(ds.path, "b")
    a.propose_boxes("v", [{"frame": 80, "label": "thing", "box": [1, 1, 9, 9], "track_id": 5}])
    b.propose_boxes("v", [{"frame": 80, "label": "other", "box": [2, 2, 9, 9], "track_id": 5}])       # same track and frame
    assert {p["author"] for p in D.proposals(ds, "ana")} == {"agent:a", "agent:b"}
    assert D.adopt(ds, "ana", "agent:a")["taken"] == 1
    with pytest.raises(D.DraftError, match="already has a box on frame 80.*Nothing was taken in"):
        D.adopt(ds, "ana", "agent:b")
    assert D.proposals(ds, "ana")[0]["author"] == "agent:b" and D.proposals(ds, "ana")[0]["adds"] == 1


def test_proposals_survive_the_dataset_changing_and_are_rechecked_when_taken_in(ds, ag):
    ag.propose_boxes("v", [{"frame": 90, "label": "thing", "box": [1, 1, 9, 9], "track_id": 6}, {"frame": 91, "label": "thing", "box": [1, 1, 9, 9]}])
    ds.add_annotations([{"kind": "box", "label": "thing", "start_frame": 90, "box": (3, 3, 9, 9), "track_id": 6, "origin": "human"}], "v")   # someone commits the same track and frame
    assert ag.list_annotations("v", frame_start=90, frame_end=90)["total"] == 2                          # reading still works
    with pytest.raises(D.DraftError, match="Nothing was taken in"):
        D.adopt(ds, "ana", "agent:scout")
    assert D.Draft.load(ds, "ana").summary()["added"] == 0 and D.proposals(ds, "ana")[0]["adds"] == 2
    ag.retract(["p1"])
    assert D.adopt(ds, "ana", "agent:scout")["taken"] == 1


def test_a_label_added_later_can_be_used_and_a_removed_video_is_refused(ds, ag):
    with pytest.raises(AgentError, match="vocab"):
        ag.propose_spans("v", [{"start_frame": 1, "end_frame": 2, "label": "late"}])
    ds.add_labels(["late"])
    assert AgentSession(ds.path, "scout").propose_spans("v", [{"start_frame": 1, "end_frame": 2, "label": "late"}])["recorded"] == 1


def test_the_whole_road_from_proposal_to_a_valid_dataset(ds, ag):
    before = ds.stats()
    ag.propose_boxes("v", [{"frame": f, "label": "thing", "box": [2 + f % 5, 3, 30, 40], "track_id": 11, "confidence": 0.5 + f / 100} for f in range(30, 40)])
    ag.propose_spans("v", [{"start_frame": 30, "end_frame": 39, "label": "other", "confidence": 0.9}])
    ag.propose_masks("v", [{"frame": 35, "label": "thing", "polygon": [[10, 10], [40, 12], [30, 40]], "track_id": 12}])
    D.adopt(ds, "ana", "agent:scout")
    me = D.Draft.load(ds, "ana")
    me.commit("agent pass")
    after = Dataset.open(ds.path)
    new = [r for r in after.annotations("v").to_pylist() if r["author"] == "agent:scout"]
    assert len(new) == 12 and {(r["origin"], r["status"]) for r in new} == {("model", "unreviewed")}
    assert not [i for i in after.validate(deep=True) if i.level == "error"]
    assert D.proposals(after, "ana") == [] and AgentSession(ds.path, "scout").draft_status()["summary"]["added"] == 0
    assert after.stats() != before
    release, _ = after.build()
    assert release.n_rows == len(after.annotations("v"))
