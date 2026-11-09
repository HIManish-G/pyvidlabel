"""The MCP server over a real stdio connection: what an agent's client actually sees."""
import asyncio
import base64
import hashlib
import json
import shutil
import sys

import pytest

pytest.importorskip("mcp")
from mcp import StdioServerParameters  # noqa: E402
from vidlabel import Dataset, VideoMetadata  # noqa: E402
from vidlabel.ui import draft as D  # noqa: E402


def _hash(ds):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ds.path.rglob("*.parquet"))}


@pytest.fixture
def ds(tmp_path):
    ds = Dataset.create(tmp_path / "d")
    ds.add_video(VideoMetadata(video_id="v", frame_count=100, fps={"num": 25, "den": 1}, width=64, height=48))
    ds.add_labels(["thing"])
    return ds


def _params(ds):
    return StdioServerParameters(command=sys.executable, args=["-m", "vidlabel.mcp_server", str(ds.path), "--name", "tester"])


async def _session(ds, fn):
    """Run `fn(call, tools)` with a connected client; `call(name, args)` returns (is_error, parsed result or text)."""
    try:
        from mcp.client.client import Client                      # mcp 2.x
        ctx = Client(_params(ds))
        async with ctx as c:
            async def call(name, args=None):
                r = await c.call_tool(name, args or {})
                return _unpack(r)
            return await fn(call, [t.name for t in (await c.list_tools()).tools])
    except ImportError:                                            # mcp 1.x
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client
        async with stdio_client(_params(ds)) as (rd, wr):
            async with ClientSession(rd, wr) as c:
                await c.initialize()
                async def call(name, args=None):
                    return _unpack(await c.call_tool(name, args or {}))
                return await fn(call, [t.name for t in (await c.list_tools()).tools])


def _unpack(r):
    kinds = [getattr(x, "type", "") for x in r.content]
    if "image" in kinds:
        return r.is_error if hasattr(r, "is_error") else r.isError, [x for x in r.content if x.type == "image"][0]
    text = "".join(x.text for x in r.content if x.type == "text")
    err = getattr(r, "is_error", None)
    err = r.isError if err is None else err
    if err:
        return True, text
    sc = getattr(r, "structured_content", None) or getattr(r, "structuredContent", None)
    return False, (sc if sc is not None else json.loads(text))


def test_tools_are_listed_and_there_is_no_way_to_commit_or_approve(ds):
    async def run(call, tools):
        return tools
    tools = asyncio.run(_session(ds, run))
    assert {"overview", "list_annotations", "list_tracks", "get_frame", "validate", "propose_boxes", "propose_spans", "propose_masks",
            "retract", "draft_status"} <= set(tools)
    assert not [t for t in tools if any(w in t for w in ("commit", "approve", "reject", "delete", "update", "set_status"))]


def test_an_agent_proposes_over_the_wire_and_the_dataset_stays_untouched(ds):
    before = _hash(ds)

    async def run(call, tools):
        err, o = await call("overview")
        assert not err and o["your_author_name"] == "agent:tester"
        err, r = await call("propose_boxes", {"video_id": "v", "boxes": [{"frame": 5, "label": "thing", "box": [1, 1, 20, 20], "track_id": 1, "confidence": 0.8}],
                                              "note": "seen by the agent"})
        assert not err and r["recorded"] == 1
        err, msg = await call("propose_boxes", {"video_id": "v", "boxes": [{"frame": 6, "label": "thing", "box": [1, 1, 20, 20], "status": "approved"}]})
        assert err and "not allowed here" in msg                            # refused, and the agent can read why
        err, msg = await call("propose_boxes", {"video_id": "v", "boxes": [{"frame": 6, "label": "nothing", "box": [1, 1, 20, 20]}]})
        assert err and "Nothing from this call was recorded" in msg
        err, st = await call("draft_status")
        assert st["summary"]["added"] == 1 and st["edits"][0]["note"] == "seen by the agent"
        err, lst = await call("list_annotations", {"video_id": "v"})
        assert lst["total"] == 1 and lst["rows"][0]["pending"] == "added"
    asyncio.run(_session(ds, run))
    assert _hash(ds) == before
    assert D.Draft.load(ds, "agent:tester").summary()["added"] == 1


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg")
def test_get_frame_returns_a_picture(tmp_path):
    from test_ui_server import make_video
    from vidlabel.probe import probe_video
    ds = Dataset.create(tmp_path / "d")
    ds.add_video(probe_video(make_video(tmp_path / "c.mp4", n=10), "clip"))
    ds.add_labels(["thing"])

    async def run(call, tools):
        err, img = await call("get_frame", {"video_id": "clip", "frame": 3, "max_side": 64})
        assert not err and (getattr(img, 'mime_type', None) or img.mimeType) == "image/jpeg" and base64.b64decode(img.data)[:2] == b"\xff\xd8"
        err, msg = await call("get_frame", {"video_id": "clip", "frame": 99})
        assert err and "no frame 99" in msg
    asyncio.run(_session(ds, run))
