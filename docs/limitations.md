# Limitations and roadmap

An honest list of what vidlabel does not do, what has not been verified, and what is deliberately left out.

## Deliberately out of scope

| not included | why, and what to do |
|---|---|
| importers and exporters for COCO, YOLO, MOT, CVAT, ... | formats vary too much for one library to cover well. Write a converter on `add_annotations` ([guide](converters.md)); a tested example is in `examples/convert_csv.py` |
| a labelling interface | vidlabel is the format and library between labelling tools and training code. A local web interface (`vidlabel ui`) can browse, review, validate and release; drawing boxes, spans and masks works on the whole frame and inside a view; accessibility was checked by tools, not by assistive-technology users; drawing masks needs a pointer (see [Web interface](ui.md)) |
| storing video pixels | datasets reference videos by path hint and content hash |
| sampling policy (class balancing, weighting, curriculum) | it is the user's decision; vidlabel arranges cheap reading of the frames you choose (`ReadPlan`) |
| keypoints, polygons, image-level classes as dedicated kinds | the three kinds are marker, box, mask. Keypoints and polygons can go in `extra` for now or be rasterised to masks; a new kind needs a spec change |
| a database or server | plain files by design |

## Known limitations of the design

- **One writer per video.** Writes are atomic but not locked; two people editing the same video can overwrite each other's changes.
  Different videos never conflict.
- **Release ids are not content-derived.** The id hashes the Parquet file's bytes, which include random annotation ids, so re-creating
  a dataset from a script gives a new id. Share the folder, including `compiled/` ([releases](releases.md#what-the-id-means)).
- **One table per video.** Many tiny videos (or images registered as one-frame videos) mean many small files. Millions of rows across
  hundreds of videos work; tens of thousands of videos have not been tried.
- **Labels cannot be renamed or deleted** by a command; see [how-to](howto.md#delete-or-relabel).
- **Single-frame boxes and masks.** A tracked object is one row per frame, linked by `track_id`; interpolation between keyframes is not
  built in.
- **The loader cannot see a small trim or a re-encode of the same length.** It warns when a file's frame count clearly differs from the
  metadata. Only `vidlabel check-video --hash` proves it is the same file.
- **Random access has a floor** of roughly 25 to 30 ms per seek in OpenCV, whatever the keyframes. `ReadPlan` and the cache are the answers.
- **No importer means no free ride from existing data.** Converting is a small script, but it is a script you write.

## Not verified yet

| area | status |
|---|---|
| Windows and macOS | tested in CI only (Python 3.12), not by hand; no timings |
| codecs and containers other than H.264 in MP4 | untested |
| real variable-frame-rate datasets | the logic is tested on constructed files only |
| Parquet against alternatives (Lance and others) at scale | no head-to-head benchmark |
| several simultaneous writers | unsupported |
| many-GPU training, very large datasets | untested |
| the effect of `ReadPlan` on training quality beyond one small test | see [performance](performance.md#does-a-readplan-hurt-training) |
| satellite data | tested with derived (not human) labels on real imagery; no labelled satellite video was available |
| PyPI release and the final project name | not done; "vidlabel" is a working name |

## Roadmap (not promises)

1. Testing the local web interface with assistive technology and on touch and pen screens; a keyboard way to draw masks ([design](ui-design.md)).
2. Publish to PyPI under a final name; add a changelog automation and a signed release process.
3. Decide on image collections (many images in one table) if people ask for them: single images already work as one-frame videos.
4. More annotation kinds (keypoints, polygons) if a clear need appears; each needs a spec change and validator rules.

Have a case that does not fit? The `extra` field and `x-` origins exist so that you do not have to wait for the format to change.
