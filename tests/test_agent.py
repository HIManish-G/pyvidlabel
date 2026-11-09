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
