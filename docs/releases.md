# Releases

A release is the dataset compiled into one file for training: a snapshot you can name, pin in a config, share and reproduce. This page
covers what a release is, how to build one for a purpose, what its id does and does not promise, and how to use it.

## What `build` does

`vidlabel build my_dataset` (or `ds.build()`):

1. validates the dataset and refuses to continue if there are errors;
2. merges every video's annotation table into one, sorted by (`video_id`, `start_frame`, `annotation_id`);
3. leaves out `rejected` rows unless you ask;
4. writes `compiled/<release_id>/annotations.parquet`, a snapshot `vocab.json`, and `release.json` (rows, filters, spec version, time);
5. points `compiled/LATEST` at it.

```
compiled/
  LATEST                        the newest release id
  r3fa91c0b2d44/
    annotations.parquet
    vocab.json                  the labels as they were at build time
    release.json
```

Releases are small compared with the videos they describe, and old ones stay until you delete them.

## Choosing what goes in

All filters are optional and combine.

| argument (`ds.build(...)`) | meaning | default |
|---|---|---|
| `statuses` | which review states to include | `("unreviewed", "approved")` |
| `origins` | only rows from these origins, e.g. `["human"]` | all |
| `kinds` | only `"marker"`, `"box"`, `"mask"` rows | all |
| `video_ids` | only these videos (a train/validation split is two releases) | all |
| `view_ids` | only rows tagged with these views; `None` inside the list means whole-frame rows | all |

Command line: `vidlabel build my_dataset` and `--include-rejected`. Everything else is Python.

Examples:

```python
ds.build(statuses=["approved"])                       # only work a person confirmed
ds.build(origins=["human"])                           # no model output at all
ds.build(video_ids=train_videos)                      # a training split
ds.build(view_ids=[None])                             # whole-frame rows only, leaving out view copies
```

A release that includes rejected rows (`--include-rejected`, or `statuses` containing `"rejected"`) is meant for inspection: it is
written, but **`LATEST` is not moved to it**. Refer to it by its id.

## What the id means

The release id is `"r"` plus the first 12 hexadecimal characters of the SHA-256 of `annotations.parquet`.

- Building the **same dataset folder** twice with no changes gives the same id.
- Changing any annotation, or any filter that changes which rows are included, gives a different id.
- The id is a hash of the **file's bytes**. Those bytes include each row's random `annotation_id` and the Parquet writer's version.
  So *re-creating* a dataset from a script (which draws new random ids) gives a different id for the same content, and a very different
  PyArrow version could in principle change the bytes for the same rows.

What this means in practice: **share the dataset folder, including `compiled/`**. Anyone who has the folder has the release with the
id you pinned, and `FrameSamples(release="r3fa91c0b2d44")` reads exactly those rows. Do not expect to regenerate an id by re-running a
converter. Content-derived ids were considered and are not planned; see [design decisions](design-decisions.md).

## The vocabulary snapshot

Class indices (the `labels` in a sample) are positions in the vocabulary. A release carries a snapshot of `vocab.json` as it was when
built, so adding or reordering labels in the dataset later never changes the class indices of a pinned release. Releases made by a
version that did not save the snapshot fall back to the live vocabulary.

## Using a release

<!-- doctest -->
```python
from vidlabel import Dataset, VideoMetadata, FrameSamples
from vidlabel.build import Compiled

ds = Dataset.create("demo")
ds.add_video(VideoMetadata(video_id="v", frame_count=100, fps={"num": 25, "den": 1}, width=64, height=48))
ds.add_labels(["thing"])
ds.add_annotations([{"kind": "box", "label": "thing", "start_frame": f, "box": (5, 5, 30, 30), "origin": "human"}
                    for f in range(0, 100, 5)], "v")
release, path = ds.build()

samples = FrameSamples(ds, release=release.release_id)       # pinned: the same rows and class indices every time
print(len(samples), samples.describe()["release_id"])        # log describe() next to a training run

compiled = Compiled(path)                                     # read the release table directly
print(compiled.read(columns=["video_id", "start_frame", "label"]).num_rows)
print(compiled.clip("v", 10, 30).num_rows)                    # rows overlapping frames 10..30, reading only needed row groups
```

`FrameSamples(release="latest")` follows `LATEST`; `release=None` reads the live, unreviewed-or-approved annotations and can change
between runs, which is convenient while labelling and wrong for a result you want to reproduce.

## A pattern for experiments

1. Label and review in the dataset.
2. `build` with the filters for the experiment; write the id in your config and in the experiment log (`samples.describe()` gives
   `release_id`, view, kinds, class names).
3. Keep the `compiled/<id>/` folder with the dataset when you share or archive it.
4. When labels change, build again; the new id marks the new data, and the old release still reproduces the old run.
