"""`vidlabel mcp PATH`: a Model Context Protocol server, so an AI agent can read a dataset and propose annotations.

Needs `pip install "vidlabel[mcp]"`. It speaks MCP over standard input and output; point your agent's MCP configuration at
`vidlabel mcp PATH --name my-agent`. What the tools can and cannot do is set in `vidlabel.agent` (read, and propose into the agent's own
draft; never approve, delete, or commit). A person takes the proposals in under *Proposals* in `vidlabel ui`.
"""
from __future__ import annotations

from typing import Any

from .agent import AgentError, AgentSession

INSTRUCTIONS = """\
You are working with a vidlabel video-annotation dataset. Coordinates are always in the video's own pixels (x to the right, y down,
box = [x0, y0, x1, y1]); frames are counted from 0. Boxes and masks belong to one frame; spans (markers) cover frames start..end inclusive.

Start with `overview`. Look before you propose: `list_annotations`, `list_tracks`, and `get_frame` (a picture with the annotations drawn on it).
Use only labels that exist (`overview` lists them). Give a track_id to follow one object across frames, and a track can have only one
box per frame. Give a confidence between 0 and 1 when you are unsure, and a short note saying why.

You can only PROPOSE. `propose_boxes`, `propose_spans` and `propose_masks` put rows into your own draft as model-made and unreviewed; each
call is all or nothing, and a refused call tells you which item was wrong. You cannot approve, reject, change or delete rows already in the
dataset, and you cannot commit. When you are done, call `draft_status` and tell the person to open Proposals in `vidlabel ui`.
"""


def _tool_error_class():
    """The exception whose message the agent gets to read (anything else is reported without its text)."""
    try:
        from mcp.server.mcpserver.exceptions import ToolError
    except ImportError:
        from mcp.server.fastmcp.exceptions import ToolError
    return ToolError


def _speaking(fn):
    """Show an `AgentError` (a mistake the agent can fix: wrong label, box outside the frame, ...) to the agent with its full message."""
    import functools
    ToolError = _tool_error_class()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (AgentError, KeyError, ValueError, OSError) as e:
            raise ToolError(str(e.args[0]) if isinstance(e, KeyError) and e.args else (str(e) or type(e).__name__)) from None
    return wrapper


def _server_class():
    try:                                                      # mcp 2.x
        from mcp.server.mcpserver import MCPServer
        return MCPServer
    except ImportError:                                       # mcp 1.x
        from mcp.server.fastmcp import FastMCP
        return FastMCP


def build_server(path, name: str | None = None):
    """The MCP server object for the dataset at `path` (not started)."""
    try:
        from mcp.types import ToolAnnotations
        server_cls, Image = _server_class(), _image_class()
    except ImportError as e:
        raise RuntimeError(f'the MCP server needs the extra package: pip install "vidlabel[mcp]"  ({e})') from None
    session = AgentSession(path, name)
    mcp = server_cls("vidlabel", instructions=INSTRUCTIONS)
    read = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
    propose = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)

    def tool(annotations):
        return lambda fn: mcp.tool(annotations=annotations)(_speaking(fn))

    @tool(read)
    def overview() -> dict[str, Any]:
        """The dataset: videos (size, frames, fps), labels you may use, views, counts, your author name and the rules for agents. Call this first."""
        return session.overview()

    @tool(read)
    def list_annotations(video_id: str, kind: str | None = None, label: str | None = None, status: str | None = None,
                         origin: str | None = None, track_id: int | None = None, frame_start: int | None = None,
                         frame_end: int | None = None, view_id: str | None = None, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        """Annotations of one video (kind: box, mask or marker; status: unreviewed, approved or rejected; origin: human, model, ...), with
        your own pending proposals included and marked `pending`. Page through large results with `limit` and `offset`."""
        return session.list_annotations(video_id, kind=kind, label=label, status=status, origin=origin, track_id=track_id,
                                        frame_start=frame_start, frame_end=frame_end, view_id=view_id, limit=limit, offset=offset)

    @tool(read)
    def list_tracks(video_id: str, view_id: str | None = None) -> dict[str, Any]:
        """The tracks of one video: id, labels, first and last frame, number of rows, statuses and origins."""
        return session.list_tracks(video_id, view_id)

    @tool(read)
    def get_frame(video_id: str, frame: int, annotations: bool = True, max_side: int = 768):
        """A picture of one frame, with the annotations drawn on it. max_side is the longest edge in pixels."""
        return Image(data=session.frame_image(video_id, frame, annotations=annotations, max_side=max_side), format="jpeg")

    @tool(read)
    def validate(deep: bool = False) -> dict[str, Any]:
        """Check the committed dataset against the format rules. deep=True also decodes every mask (slow)."""
        return session.validate(deep)

    @tool(propose)
    def propose_boxes(video_id: str, boxes: list[dict[str, Any]], note: str | None = None) -> dict[str, Any]:
        """Propose bounding boxes. Each item: {frame, label, box: [x0, y0, x1, y1], track_id?, confidence?}. All or nothing (at most 500 per call).
        They go into your draft as model-made, unreviewed rows; a person reviews and commits them. `note` says why."""
        return session.propose_boxes(video_id, boxes, note)

    @tool(propose)
    def propose_spans(video_id: str, spans: list[dict[str, Any]], note: str | None = None) -> dict[str, Any]:
        """Propose labelled stretches of frames (events or states). Each item: {start_frame, end_frame (inclusive), label, confidence?}. All or nothing."""
        return session.propose_spans(video_id, spans, note)

    @tool(propose)
    def propose_masks(video_id: str, masks: list[dict[str, Any]], note: str | None = None) -> dict[str, Any]:
        """Propose masks drawn as polygons. Each item: {frame, label, polygon: [[x, y], ...] (3 or more corners inside the frame), track_id?, confidence?}. All or nothing."""
        return session.propose_masks(video_id, masks, note)

    @tool(ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False))
    def retract(ids: list[str]) -> dict[str, Any]:
        """Take back proposals of yours that are still only in your draft (ids like "p3"). Rows already in the dataset cannot be removed."""
        return session.retract(ids)

    @tool(read)
    def draft_status() -> dict[str, Any]:
        """What you have proposed so far and have not yet had taken in by a person, and what to tell them."""
        return session.draft_status()

    return mcp


def _image_class():
    try:
        from mcp.server.mcpserver import Image
    except ImportError:
        from mcp.server.fastmcp import Image
    return Image


def serve(path, name: str | None = None) -> None:
    build_server(path, name).run()


def main(argv: list[str] | None = None) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="python -m vidlabel.mcp_server", description="Serve a vidlabel dataset to an AI agent over MCP (stdio).")
    p.add_argument("path")
    p.add_argument("--name", default="agent", help="Who the agent is; proposals are made by 'agent:NAME'.")
    a = p.parse_args(argv)
    serve(a.path, a.name)


if __name__ == "__main__":
    main()
