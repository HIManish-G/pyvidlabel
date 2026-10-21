import pytest

from conftest import blob_mask
from vidlabel import ValidationError


def box(frame, track, label="car", **kw):
    return {"kind": "box", "label": label, "start_frame": frame, "box": (5, 5, 30, 30), "track_id": track,
            "origin": "human", **kw}


def codes(ds):
    return {(i.level, i.code) for i in ds.validate()}


def test_clean_tracks_pass(ds):
    ds.add_annotations([box(f, 1) for f in range(10)] + [box(f, 2, "person") for f in range(5, 15)], "vid1")
    ds.add_annotations([box(3, None), box(3, None), box(4, None)], "vid1", validate=False)   # no track: never checked
    assert ds.validate() == []


def test_label_change_on_a_track_is_a_warning_not_an_error(ds):
    ds.add_annotations([box(0, 7, "car"), box(1, 7, "car"), box(2, 7, "person")], "vid1", validate=False)
    assert codes(ds) == {("warning", "track_label_conflict")}
    ds.build()                                                  # warnings never block a build


def test_two_boxes_of_one_track_in_a_frame_is_an_error(ds):
    ds.add_annotations([box(3, 4), box(3, 4), box(4, 4)], "vid1", validate=False)
    assert ("error", "duplicate_track_in_frame") in codes(ds)
    with pytest.raises(ValidationError, match="more than once"):
        ds.build()
    with pytest.raises(ValidationError):                        # also caught when a single call adds both
        ds.add_annotations([box(9, 1), box(9, 1)], "vid2")


def test_a_box_and_a_mask_of_the_same_track_can_share_a_frame(ds):
    ds.add_annotations([box(3, 4), {"kind": "mask", "label": "car", "start_frame": 3, "mask": blob_mask(),
                                    "track_id": 4, "origin": "model"}], "vid1")
    assert ds.validate() == []


def test_rejected_rows_do_not_count(ds):
    """A reviewer rejects a box and re-draws it under the same track id: that is normal, not a conflict."""
    ds.add_annotations([box(3, 4, status="rejected"), box(3, 4), box(5, 4, "person", status="rejected"), box(5, 4)],
                       "vid1")
    assert ds.validate() == []


def test_tracks_in_different_views_are_separate(ds):
    ds.add_view({"view_id": "a", "video_id": "vid1", "region": {"x": 0, "y": 0, "width": 64, "height": 24}})
    ds.add_view({"view_id": "b", "video_id": "vid1", "region": {"x": 0, "y": 24, "width": 64, "height": 24}})
    ds.add_annotations([
        {"kind": "box", "label": "car", "start_frame": 1, "box": (1, 1, 9, 9), "track_id": 1, "view_id": "a", "origin": "model"},
        {"kind": "box", "label": "person", "start_frame": 1, "box": (1, 30, 9, 40), "track_id": 1, "view_id": "b", "origin": "model"},
        {"kind": "box", "label": "highlight", "start_frame": 1, "box": (20, 1, 40, 9), "track_id": 1, "view_id": "a", "origin": "model",
         "status": "rejected"},
    ], "vid1")
    assert ds.validate() == []                                  # same id, different label, different views: fine
    ds.add_annotations([{"kind": "box", "label": "car", "start_frame": 1, "box": (30, 1, 40, 9), "track_id": 1, "view_id": "a",
                         "origin": "model"}], "vid1", validate=False)
    assert ("error", "duplicate_track_in_frame") in codes(ds)   # but within one view it is a duplicate


def test_a_later_call_cannot_duplicate_a_track_in_a_frame(ds):
    ds.add_annotations([box(3, 4), box(4, 4)], "vid1")
    with pytest.raises(ValidationError, match="more than once"):
        ds.add_annotations([box(3, 4)], "vid1")                  # same track, same frame as an earlier call
    assert len(ds.annotations("vid1")) == 2                      # nothing was written
    ds.add_annotations([box(5, 4), box(3, 5)], "vid1")           # other frames / other tracks are fine
    ds.add_annotations([box(3, 4, status="rejected")], "vid1")   # rejected rows never count
    assert len(ds.annotations("vid1")) == 5


def test_old_problems_elsewhere_do_not_block_new_tracks(ds):
    ds.add_annotations([box(3, 9), box(3, 9)], "vid1", validate=False)    # an existing duplicate on track 9
    ds.add_annotations([box(8, 1)], "vid1")                                # adding track 1 is unaffected
    assert len(ds.annotations("vid1")) == 3
