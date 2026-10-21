# Contributing

## Setup and tests

- `pip install -e packages/vidlabel-spec -e packages/vidlabel pytest`, then `pytest`.
- Frame, preview, loader and seek tests need OpenCV and ffmpeg (`pip install -e "packages/vidlabel[video]"`); the loader tests
  also need PyTorch (`[torch]`). Each skips itself if something is missing. CI runs everything on Linux, Windows and macOS.

## The spec is the contract

Changing a column or metadata field means bumping `SPEC_VERSION` (and the pinned versions in the two `pyproject.toml` files),
updating `docs/SPEC.md`, and adding a test. Custom data never needs a spec change: put it in `extra`. A new validation check
needs a code, an entry in the table in `docs/SPEC.md` (a test fails if it is missing), and a test.

## Documentation is tested

- The reference pages `docs/cli.md` and `docs/python-api.md` are **generated** from the code. After changing a public
  function, its docstring, or a command or option, run `python docs/gen_reference.py`. A test fails if they are stale.
- In any page, a Python block preceded by `<!-- doctest -->` is executed by the tests (blocks on one page run in order and share
  variables, in an empty folder), and a shell block preceded by `<!-- clitest -->` runs each `vidlabel ...` line. Use them for
  anything that needs no video file or special install. Code that does is shown without the marker.
- Internal links and heading anchors are checked, and `docs/converters.md` must contain `examples/convert_csv.py` verbatim
  (re-copy it if you change the example).

## Other changes

- New mask codecs: add a value to `MaskCodec`, a module next to `masks.py`, and a lossless round-trip test.
- Converters to and from other formats are **not** part of this repository: they vary too much. Write them as separate packages on the
  public API; see [docs/converters.md](docs/converters.md). Rows only need the ordinary annotation columns (origin is a column, never a separate structure).
- Text files are read and written as UTF-8 explicitly, so datasets move between operating systems. Keep it that way.
