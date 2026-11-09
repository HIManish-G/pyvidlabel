# Compatibility and versioning

## Versions

There are two version numbers, always released together while the project is young:

| name | where | meaning |
|---|---|---|
| **spec version** | `SPEC_VERSION` in `vidlabel-spec`, `manifest.json` of every dataset | the format of the files |
| **library version** | `vidlabel.__version__` | the code that reads and writes them |

The format (spec) is currently at **0.6.0**. The two packages are at **0.6.1** and pin each other exactly (`vidlabel` requires `vidlabel-spec==0.6.1`); the package patch version can move without the format changing.

## What a version promises

- The **major** version of a dataset's spec must match the library's. A different major version is reported as the error `spec_major`.
- Within a major version, a library opens datasets written by older minor versions of the spec. A dataset written by 0.4 or 0.5 opens in 0.6.
- A newer dataset may use features an older library cannot read. For example, a view using `frame_ranges` (added in 0.6.0) cannot be read by
  a 0.5 library. When a spec feature is added, the entry in [changelog](changelog.md) says so.
- Custom data in `extra` never needs a spec change and is preserved.
- While the version is below 1.0, minor versions can still tighten validation (a dataset that was accepted may report a new error) or
  change library behaviour. Each such change is in the changelog.

## Upgrading

1. Read the [changelog](changelog.md) entries between your version and the new one.
2. Update the packages (both together).
3. Run `vidlabel validate PATH` on your datasets. A newly reported error names the rule and the rows.
4. Rebuild releases you want to keep reproducible only if you need them in the new format; **old releases stay readable**, and a
   pinned release id keeps returning the same rows.

Datasets are not rewritten by upgrading. Only a write (adding annotations, editing, building) touches files.

## Platforms

| platform | status |
|---|---|
| Linux, Python 3.10 to 3.13 | the test suite is run here and passes |
| Windows, macOS | the CI workflow runs the test suite on them (Python 3.12) and it passes; they have not been exercised by hand; the code avoids platform-specific behaviour (explicit UTF-8, `pathlib`, process start methods tested with `fork` and `spawn`) |

Real-world use on Windows and macOS is the largest unverified area.

## External requirements and tested versions

| need | for | note |
|---|---|---|
| Python >= 3.10 | everything | |
| NumPy >= 1.24, PyArrow >= 14, zstandard >= 0.22, Typer >= 0.12, pydantic >= 2.5 | the core | |
| OpenCV (headless) >= 4.8 | frames, previews, cache, loaders | `pip install "vidlabel[video]"` |
| PyTorch >= 2.0 | the loaders in `torch_data` | `pip install "vidlabel[torch]"` |
| ffmpeg / ffprobe | registering videos, frame index, tests that encode video | any recent build |

Video decoding quality depends on your OpenCV/ffmpeg build. Frame reading and seek checks have been exercised on H.264 in MP4 only; other
codecs and containers are not verified. `vidlabel check-video --seek` tests that random access returns the right frames for a given file.

## File formats you can rely on

| file | stability |
|---|---|
| `manifest.json`, `vocab.json`, `metadata.json`, view files, `frame_index.json`, `release.json` | JSON with strict, documented fields |
| `annotations.parquet` | the columns in [`ANNOTATION_COLUMNS`](SPEC.md#annotation-table-one-row-per-annotation-all-kinds-together), stored with zstd |
| mask blobs | codec `bitpack_zstd` version 1, little-endian header; see [masks](masks.md) |

Any Parquet reader (pandas, DuckDB, Polars, Spark) can read the annotation tables; masks are opaque bytes unless decoded with
`vidlabel.masks`.
