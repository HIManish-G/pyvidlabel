# Performance

What was measured, on what, and how to read the numbers. They come from **one machine** (a 2-core cloud container), a few real clips and
one synthetic dataset, and are meant to show orders of magnitude and trade-offs, not to be quoted as benchmarks. Re-measure on your
hardware and data before depending on any figure.

## Annotation tables

Synthetic dataset: 100 videos, 1,000,000 annotations in total (boxes, masks and markers).

| operation | result |
|---|---|
| write (`add_annotations`) | about 20 s |
| `validate` (without `--deep`) | about 5.5 s |
| `build` | about 6 s |
| size on disk | about 83 MB |
| open `FrameSamples` on the release | about 0.6 s and 0.5 GB |

`FrameSamples` once held one Python dictionary per row (6.9 s and 1.6 GB for the same data); it now keeps NumPy columns, and a
regression test limits its memory. A small real dataset (5,265 annotations including masks) is 328 KB next to a 15 MB video.

Masks use `bitpack_zstd`: cropped to the mask's bounding box, bit-packed, zstd level 3. Decoding is about 0.1 ms for typical object
masks, and the blobs are smaller than COCO RLE on typical object masks (measured when the codec was chosen; see [masks](masks.md)).

## Reading video frames

Frames are decoded with OpenCV. Reading frame *N* means decoding from the keyframe before it, so what matters is the keyframe spacing.

| situation | rate |
|---|---|
| sequential reading, 720p H.264 | about 400 frames/s |
| random access, one frame at a time, 720p, keyframe every 250 | about 6 frames/s |
| random access, 1,100-frame 384x384 clip, 300 random frames | 33 frames/s one at a time |
| the same 300 frames with a `ReadPlan` | 370 frames/s |
| the same 300 frames from the JPEG cache | 911 frames/s |
| the OpenCV seek-and-read floor on three real clips | 25 to 30 ms per seek, whatever the keyframe spacing |

A `ReadPlan` groups requests by keyframe interval, reads each interval once in increasing order and decodes repeated frames once. The
gain is largest when keyframes are far apart and requests are many. Re-encoding with `-g 30` helped only when the original keyframes were far
apart (one clip went from 14 to 36 frames/s; the file became 2.2 times larger).

## Does a `ReadPlan` hurt training?

One small test (a real 1,100-frame clip, a tiny CNN regressing box centres, 4,800 training samples per run, validated on later frames):

| loading | validation error (mean absolute, mean over seeds) | whole run |
|---|---|---|
| plain random access (3 seeds) | 0.022 | 180 s |
| `ReadPlan`, no shuffle buffer (5 seeds) | 0.048 (0.012 to 0.079) | 7 s |
| `ReadPlan`, `shuffle_buffer=256` | 0.028 | 7 s |
| `ReadPlan`, `shuffle_buffer=1024` | 0.023 | 7 s |
| `ReadPlan`, `shuffle_buffer=3000` | 0.023 | 7 s |

Without a buffer, 99.7% of neighbouring frames in a batch were within 30 frames of each other and training was unstable. With a buffer of
about 1,000 samples or more it matched random access at roughly 25 times the speed. The buffer costs memory: 1,024 frames of 384x384
are about 450 MB per process. One model, one clip, a few seeds: treat as a hint to check on your own task.

## Views and masks

Landsat-style tiles (2048 px scene, 512x512 views) come back through crop, resize and release with an exact IoU of 1.0 for masks drawn in
view pixels at the view's own resolution. Masks drawn at a coarser resolution than the view and pasted up lose detail in the obvious way;
in another test, resized 256x256 crops came back with a mean IoU of about 0.94.

## Where time goes, and what to do

| symptom | likely cause | try |
|---|---|---|
| training starves the GPU on video input | random access pays a decode from the keyframe each time | `ReadPlan` + `shuffle_buffer` of about 1,000; the frame cache; more keyframes |
| `validate` slow on many masks | `--deep` decodes every mask | run it before release builds, not on every edit |
| high memory in many DataLoader workers | each worker holds a copy of what it opened | `FrameSamples` is columnar and small; the cost is usually decoded frames in shuffle buffers |
| first sample slow | opening a reader and seeking | keep readers open (`max_open` in `iter_plan`) |

## Not measured

Parquet against other table formats at large scale, several simultaneous writers, many-GPU training, real variable-frame-rate datasets,
and Windows and macOS timings. See [limitations](limitations.md).
