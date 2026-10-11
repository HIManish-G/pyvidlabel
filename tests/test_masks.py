import numpy as np
import pytest

from vidlabel import masks


@pytest.mark.parametrize("shape", [(1, 1), (7, 9), (48, 64), (101, 203)])
def test_roundtrip_random(shape):
    rng = np.random.default_rng(1)
    for density in (0.0, 0.01, 0.5, 1.0):
        m = rng.random(shape) < density
        assert np.array_equal(masks.decode(masks.encode(m)), m)


def test_edges_and_empty():
    m = np.zeros((20, 30), bool)
    m[0, 0] = m[19, 29] = True
    assert np.array_equal(masks.decode(masks.encode(m)), m)
    assert masks.bbox(masks.encode(m)) == (0, 0, 30, 20)
    e = masks.encode(np.zeros((5, 5), bool))
    assert masks.info(e)["empty"] and masks.bbox(e) is None
    assert not masks.decode(e).any()


def test_crop_and_bbox():
    m = np.zeros((40, 40), bool)
    m[10:15, 20:33] = True
    b = masks.encode(m)
    crop, y0, x0 = masks.decode_crop(b)
    assert (y0, x0, crop.shape) == (10, 20, (5, 13))
    assert masks.bbox(b) == (20, 10, 33, 15)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        masks.encode(np.zeros((2, 3, 4)))
    with pytest.raises(ValueError):
        masks.info(b"abc")


@pytest.mark.parametrize("compressed", [False, True])
def test_coco_rle_roundtrip(compressed):
    rng = np.random.default_rng(2)
    for m in [rng.random((17, 23)) > 0.5, np.zeros((6, 4), bool), np.ones((6, 4), bool)]:
        assert np.array_equal(masks.from_coco_rle(masks.to_coco_rle(m, compressed)), m)


def test_coco_rle_known_value():
    # column-major: column 0 = [0,1], column 1 = [1,1] -> runs 0:1, 1:3
    m = np.array([[0, 1], [1, 1]], bool)
    assert masks.to_coco_rle(m)["counts"] == [1, 3]
