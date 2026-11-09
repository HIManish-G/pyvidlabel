# AI agents

vidlabel can be used by an AI agent in two ways, and in both the rule is the same: **an agent proposes, a person decides.**

- **Over MCP** (`vidlabel mcp`): the agent reads the dataset and proposes boxes, spans and masks. Its proposals wait in a draft until a
  person takes them in under *Proposals* in the [web interface](ui.md), reviews them and commits.
- **Through the command line**: every read command can print JSON (`--json`, and `vidlabel query`), so an agent that can run commands
  needs no special integration.

## What an agent can and cannot do

| can | cannot |
|---|---|
| read the dataset overview, annotations (with filters and paging), tracks and validation | approve or reject rows |
| see a frame as a picture with the annotations drawn on it | change or delete a row that is already in the dataset |
| propose boxes, spans and polygon masks, with a track, a confidence and a note | set `origin`, `status` or `author` (always `model`, `unreviewed`, `agent:<name>`) |
| take back its own pending proposals | commit anything |

Every proposal is checked by the library before it is recorded, like a row drawn by hand: the label must exist, the box must be inside the
frame, the frame must exist, numbers must be real numbers (not text, true/false, NaN or infinity), frames and track ids whole numbers, confidence must be between 0 and 1, and a track can have only one box (or mask) on a frame. A call is **all or
nothing**: if one item is refused, none of the call is recorded, and the agent is told which item was wrong and why. One agent can hold at most
20,000 pending rows.

Proposals go to the agent's own draft, the same kind of draft a person's web-interface session keeps (`.vidlabel-ui/drafts/agent_<name>.json`
inside the dataset folder). Nothing in the dataset changes until a person commits.

## Set it up

```bash
pip install "vidlabel[mcp]"
```

Point your agent's MCP configuration at `vidlabel mcp`. For clients that use a JSON file:

```json
{
  "mcpServers": {
    "vidlabel": {
      "command": "vidlabel",
      "args": ["mcp", "/path/to/dataset", "--name", "my-agent"]
    }
  }
}
```

For Claude Code: `claude mcp add vidlabel -- vidlabel mcp /path/to/dataset --name my-agent`. If your client cannot find the `vidlabel` command
(common on Windows), give the full path to it inside your virtual environment, or use `python -m vidlabel.mcp_server /path/to/dataset --name my-agent`.

`--name` becomes the author of the proposals (`agent:my-agent`). Use a different name for each agent or each run you want to review separately.
Names are kept to lower-case letters, digits, `-` and `_`; if a name has to be changed (capitals, spaces, other characters), a short code is
added so two different names never share a draft.
The server talks over standard input and output and runs only while the agent's client is connected.

## The tools

| tool | what it does |
|---|---|
| `overview` | videos (size, frames, fps), the labels the agent may use, views, counts, its author name and the rules |
| `list_annotations` | annotations of one video, filtered by kind, label, status, origin, track, frames or view, with `limit` and `offset`; its own pending proposals are included and marked |
| `list_tracks` | the tracks of one video |
| `get_frame` | a JPEG of one frame with the annotations drawn on it; `max_side` sets the size |
| `validate` | the format check of the committed dataset (`deep` also decodes masks) |
| `propose_boxes` | `{frame, label, box: [x0, y0, x1, y1], track_id?, confidence?}` items, in the video's own pixels |
| `propose_spans` | `{start_frame, end_frame, label, confidence?}` items (inclusive) |
| `propose_masks` | `{frame, label, polygon: [[x, y], ...], track_id?, confidence?}` items; the polygon is filled into a mask |
| `retract` | take back pending proposals by id (`p3`); rows in the dataset cannot be removed |
| `draft_status` | what the agent has proposed so far, and what to tell the person |

`propose_*` accept an optional `note`, a short reason that the reviewer sees next to the rows. A call takes at most 500 items.

## Review what an agent proposed

1. Run `vidlabel ui /path/to/dataset`. The **Proposals** entry shows a badge with the number of rows waiting.
2. On the Proposals page you see each agent, its rows (video, kind, label, frames, track, confidence, note) and buttons to **take in** all of
   them or one video's, or to **throw them away**.
3. Taking in copies the rows into *your* draft, where they are checked again against the dataset as your draft leaves it (all or nothing; if one
   cannot be taken, for example because you have drawn a box for the same track on that frame, nothing is taken and you are told which). They show as pending in the viewer
   and the Review page, marked `model` and `unreviewed`, with the agent as `author`.
4. Approve, reject, adjust or undo them there, then **Commit**. After the commit they are ordinary rows; reviewing them is the normal
   [Review](ui.md) workflow (sort by lowest confidence, approve or reject a track at once).

Only agents' drafts appear on that page. A person's draft is never visible to or changeable by anyone else.

## Through the command line

Every read command can print one JSON document:

```bash
vidlabel info DATASET --json
vidlabel validate DATASET --json          # exit code 1 when there are errors, as without --json
vidlabel tracks DATASET VIDEO --json
vidlabel build DATASET --json
vidlabel query DATASET VIDEO --kind box --label car --frames 100-200 --limit 50     # always JSON
```

With `--json` (and always for `query`), an error is also JSON, `{"ok": false, "error": "..."}`, with exit code 1. Annotation ids are strings (they can exceed what
JavaScript numbers hold), and `query` never prints mask pixels (`has_mask` tells you a row has one).

## Limits

- Agents can only add rows. Changing or removing existing rows is for people, in the web interface or the library.
- The picture tool needs the video file on the computer running the server and OpenCV (`vidlabel[mcp]` installs it).
- There is no model inside vidlabel; the agent brings its own. How good the proposals are depends on it, which is why they arrive unreviewed.
- Two drafts are independent: an agent that keeps proposing while you take its rows in may leave new rows waiting; take them in again.
