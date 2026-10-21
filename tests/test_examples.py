import importlib.util
from pathlib import Path

import numpy as np
import pytest

from conftest import make_meta
from vidlabel import Dataset, ValidationError
from vidlabel.boxes import rotate_boxes

ROOT = Path(__file__).resolve().parents[1]


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_csv_converter_example(ds, tmp_path):
    conv = load_example("convert_csv")
    csv_file = tmp_path / "det.csv"
    csv_file.write_text("frame,track,class,x,y,w,h,score\n0,1,Car,10,5,20,10,0.9\n1,1,Car,12,5,20,10,\n3,,Person ,1,1,8,8,0.5\n",
                        encoding="utf-8")
    ids = conv.convert(ds, csv_file, "vid1")
    assert len(ids) == 3 and ds.validate() == []
    t = ds.annotations("vid1").sort_by("start_frame").to_pylist()
    assert [r["label"] for r in t] == ["car", "car", "person"]                       # names normalised, vocabulary extended
    assert (t[0]["x0"], t[0]["y0"], t[0]["x1"], t[0]["y1"]) == (10, 5, 30, 15)       # x,y,w,h -> corners
    assert t[0]["track_id"] == 1 and t[2]["track_id"] is None
    assert t[0]["confidence"] == pytest.approx(0.9) and t[1]["confidence"] is None
    assert {r["origin"] for r in t} == {"imported"} and {r["tool"] for r in t} == {"csv-converter"}
    assert t[2]["extra"] == '{"csv_line": 4}'
    again = conv.convert(ds, csv_file, "vid1")                                       # re-running replaces, never duplicates
    assert len(again) == 3 and len(ds.annotations("vid1")) == 3
    ds.add_annotations([{"kind": "marker", "label": "person", "start_frame": 0, "origin": "human"}], "vid1")
    conv.convert(ds, csv_file, "vid1")                                               # other people's rows are left alone
    assert len(ds.annotations("vid1")) == 4


def test_csv_converter_reports_bad_rows_and_writes_nothing(ds, tmp_path):
    conv = load_example("convert_csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("frame,track,class,x,y,w,h,score\n0,1,car,10,5,20,10,0.9\n9999,1,car,1,1,5,5,0.9\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="beyond video length"):
        conv.convert(ds, bad, "vid1")
    assert len(ds.annotations("vid1")) == 0


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_rotate_boxes_matches_rotating_the_pixels(rotation):
    W, H = 40, 24
    frame = np.zeros((H, W), bool)
    frame[3:11, 5:19] = True                                                         # stored box (5, 3, 19, 11)
    k = {0: 0, 90: -1, 180: 2, 270: 1}[rotation]                                      # np.rot90 is counter-clockwise
    rotated = np.rot90(frame, k)
    ys, xs = np.nonzero(rotated)
    want = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    assert rotate_boxes([(5, 3, 19, 11)], rotation, (W, H)).tolist() == [want]


def test_rotate_boxes_rejects_other_angles():
    with pytest.raises(ValueError):
        rotate_boxes([(0, 0, 1, 1)], 45, (10, 10))
