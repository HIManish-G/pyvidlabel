# For AI agents

This file is for coding and annotation agents working with a vidlabel dataset or this repository.

**Using a dataset** (full guide: `docs/agents.md`)

- Read with the command line: `vidlabel info|validate|tracks|build PATH --json` and `vidlabel query PATH VIDEO ...` print one JSON document.
- Or connect over MCP: `pip install "vidlabel[mcp]"`, then run `vidlabel mcp PATH --name NAME`.
- You may **propose** (boxes, spans, polygon masks). Proposals land in your own draft as `origin=model`, `status=unreviewed`; a person takes them in
  under *Proposals* in `vidlabel ui PATH` and commits. You do not approve, reject, delete existing rows, or commit, and the tools do not let you.
- Coordinates are the video's own pixels, frames start at 0, a track has one box per frame, labels must already exist.
- Do not edit files inside a dataset folder by hand; use the library or the tools.

**Working on this repository**

- `CONTRIBUTING.md` has the layout and how to run the tests (`pytest -q`; the browser tests need Playwright's Chromium, and skip themselves without it).
- `docs/` is the single source for the documentation and the wiki; examples in it are executed by the tests. After changing a command or a public
  signature run `python docs/gen_reference.py`, and `python docs/gen_reference.py --check` must pass.
- The format spec is `docs/SPEC.md`; a change to the files needs a spec version change and a validator rule with a test.
- No importers or exporters for other formats belong in the library; `docs/converters.md` explains how to write one outside it.
