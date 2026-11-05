# Development

For people changing vidlabel itself. The short version is also in [CONTRIBUTING](https://github.com/OWNER/REPO/blob/main/CONTRIBUTING.md)
(replace the link with your repository's).

## Set up

```bash
git clone <your repository URL> && cd vidlabel
python -m venv .venv && source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e packages/vidlabel-spec -e "packages/vidlabel[torch]" pytest
pytest
```

Frame, preview, loader and seek tests need OpenCV, ffmpeg (they encode small test videos) and, for the loaders, PyTorch. Each test skips
itself when something is missing, so a core-only install runs a smaller suite. The suite is a few minutes at most (about 240 tests, under a
minute on a laptop-class machine).

## Repository layout

```
packages/vidlabel-spec/   the format: pydantic models, SPEC_VERSION, JSON Schema
packages/vidlabel/        the library and the `vidlabel` command
docs/                     all documentation (also published as the GitHub wiki)
examples/                 convert_csv.py (copied verbatim into docs/converters.md), make_sample.py
tests/                    the test suite
.github/workflows/ci.yml  Linux 3.10/3.12/3.13, Windows and macOS 3.12
```

## The spec is the contract

A change to a column or metadata field means: bump `SPEC_VERSION` in `vidlabel-spec`, update the pinned `vidlabel-spec==` in
`packages/vidlabel/pyproject.toml` and both package versions, update [`SPEC.md`](SPEC.md) (including its Versions section), and add a test.
Custom data never needs a spec change: it goes in `extra`.

A new validation check needs a code, a row in the validation table in `SPEC.md` (a test fails if it is missing), an entry in
[validation](validation.md) saying how to fix it, and a test.

## Documentation is tested

- [`cli.md`](cli.md) and [`python-api.md`](python-api.md) are generated from the code. After changing a public function, its docstring,
  or a command or option, run `python docs/gen_reference.py` (and `--check` to verify). A test fails if they are stale.
- A Python block preceded by `<!-- doctest -->` runs in the tests; blocks on one page run in order in an empty folder and share variables.
  A shell block preceded by `<!-- clitest -->` runs each `vidlabel ...` line. Use them for anything that needs no video file.
- Internal links and heading anchors are checked on every page.
- The wiki is generated from `docs/` by `python docs/build_wiki.py`; see below.

## The GitHub wiki

The pages in `docs/` are the single source of truth. `python docs/build_wiki.py OUTPUT_DIR` writes a wiki-ready folder: the home page, a sidebar, a footer,
and every page under its wiki name with links rewritten. Publishing is a copy and a push:

```bash
python docs/build_wiki.py wiki-out
git clone https://github.com/OWNER/REPO.wiki.git
cp wiki-out/*.md REPO.wiki/ && cd REPO.wiki && git add -A && git commit -m "Update wiki" && git push
```

(GitHub creates the `.wiki` repository when you save the first page in the wiki tab.) Never edit the wiki by hand; the next publish would overwrite it.

## Other changes

- **New mask codec:** add a value to `MaskCodec`, a module beside `masks.py`, and a lossless round-trip test.
- **Converters** to and from other formats are not part of this repository; write them as separate packages on the public API.
- **Text I/O** is UTF-8 explicitly everywhere; keep it that way so datasets move between operating systems.
- **Tests that need video** create small clips with ffmpeg in a temporary folder; no binary test assets are committed.

## Before opening a pull request

1. `pytest` passes.
2. `python docs/gen_reference.py --check` passes if you changed public signatures or help text.
3. Behaviour changes have a line in [changelog](changelog.md).
4. A bug fix has a regression test that fails without the fix.

## Making a release

Not yet exercised (nothing is published). The intended steps:

1. Update the changelog and bump versions (`SPEC_VERSION` only if the format changed) in the two `pyproject.toml` files and `__init__.py`.
2. Run the full suite on all three operating systems in CI.
3. Build both packages (`python -m build packages/vidlabel-spec packages/vidlabel`) and test installing the wheels in a clean environment.
4. Tag, publish the spec package first and then the library (it pins the spec exactly), then publish the wiki.
