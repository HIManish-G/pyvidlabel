"""Accessibility of the web interface: colour contrast of the design tokens, names for every control, dialogs and toasts that work with a
keyboard and a screen reader, drawing without a pointer. With axe-core available (VIDLABEL_AXE_JS, or node_modules/axe-core, or the
`axe-playwright-python` package) every page and state is also run through it."""
import os
import re
import shutil
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "packages" / "vidlabel" / "src" / "vidlabel" / "ui" / "static"


# ---------------------------------------------------------------- contrast of the colours (no browser needed)
def _lum(hex_):
    h = hex_.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _ratio(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _tokens():
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    light = dict(re.findall(r"--([a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", css.split("@media (prefers-color-scheme: dark)")[0]))
    dark_block = css.split("@media (prefers-color-scheme: dark)")[1].split("}")[0]
    dark = {**light, **dict(re.findall(r"--([a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", dark_block))}
    return {"light": light, "dark": dark}


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_text_colours_reach_4_5_to_1_on_every_surface(scheme):
    t = _tokens()[scheme]
    for fg in ("ink", "ink-2", "accent", "ok", "bad", "warn"):
        for bg in ("paper", "sheet", "wash"):
            assert _ratio(t[fg], t[bg]) >= 4.5, f"{scheme}: {fg} {t[fg]} on {bg} {t[bg]} is {_ratio(t[fg], t[bg]):.2f}"
    for bg in ("paper", "sheet"):
        assert _ratio(t["pending"], t[bg]) >= 4.5                           # the pink used as a text colour
    assert _ratio(t["on-pending"], t["pending"]) >= 4.5 and _ratio(t["on-bad"], t["bad"]) >= 4.5      # text on the pink badges and the red messages
    assert _ratio(t["tape-ink"], t["tape"]) >= 4.5 and _ratio(t["paper"], t["ink"]) >= 4.5


# ---------------------------------------------------------------- in a browser
pytest.importorskip("starlette")
pytest.importorskip("uvicorn")
pytest.importorskip("playwright.sync_api")
pytest.importorskip("cv2")
from test_ui_browser import _commit, _draft_count, _settled, _viewer, page, server  # noqa: E402,F401

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not installed")

NAMELESS = """() => {
  const name = e => (e.getAttribute('aria-label') || '').trim() || (e.labels && e.labels.length ? [...e.labels].map(l => l.textContent).join('').trim() : '')
                    || (e.getAttribute('aria-labelledby') ? document.getElementById(e.getAttribute('aria-labelledby'))?.textContent.trim() : '') || (e.tagName === 'BUTTON' || e.tagName === 'A' ? e.textContent.trim() : '') || (e.title || '').trim();
  const bad = [];
  for (const e of document.querySelectorAll('input:not([type=hidden]), select, textarea, button, a[href]'))
    if (e.offsetParent !== null && !name(e)) bad.push(e.outerHTML.slice(0, 90));
  for (const th of document.querySelectorAll('th')) if (!th.textContent.trim() && !th.querySelector('input, button')) bad.push('empty th');
  for (const b of document.querySelectorAll('[role=dialog]')) if (!b.getAttribute('aria-modal') || !b.getAttribute('aria-labelledby')) bad.push('dialog without name');
  return bad;
}"""


def _axe_source():
    p = os.environ.get("VIDLABEL_AXE_JS")
    cands = [Path(p)] if p else []
    cands.append(Path(__file__).resolve().parents[1] / "node_modules" / "axe-core" / "axe.min.js")
    try:
        import axe_playwright_python
        cands.append(Path(axe_playwright_python.__file__).parent / "axe.min.js")
    except ImportError:
        pass
    return next((c.read_text(encoding="utf-8") for c in cands if c.is_file()), None)


PAGES = ["overview", "browse", "labels", "views", "review/clip", "proposals", "validation", "releases", "inspector?video=clip&f=10&release=none", "history"]


def test_every_control_has_a_name_on_every_page_and_there_is_one_heading(page):
    pg, ds, base = page
    for scheme in ("light", "dark"):
        pg.emulate_media(color_scheme=scheme)
        for p in PAGES + ["video/clip?f=10"]:
            pg.goto(f"{base}/#/{p}")
            pg.reload()
            pg.wait_for_selector("h1")
            pg.wait_for_timeout(300)
            assert pg.evaluate(NAMELESS) == [], p
            assert pg.locator("h1").count() == 1, (p, pg.locator("h1").all_inner_texts())
            assert pg.title().endswith("· vidlabel") and pg.locator("#nav a[aria-current=page]").count() == 1, p


def test_the_current_page_is_announced_and_tool_buttons_say_whether_they_are_pressed(page):
    pg, ds, base = page
    _viewer(pg, base, 10)
    assert pg.inner_text("#announce") == "Videos" or pg.inner_text("#announce") == "Viewer"
    pressed = pg.evaluate("[...document.querySelectorAll('.tools button')].map(b => b.getAttribute('aria-pressed'))")
    assert set(pressed) == {"true", "false"} and pressed.count("true") == 1          # false is stated, not left out
    pg.keyboard.press("b")
    assert pg.get_attribute(".tools button:has-text('Box')", "aria-pressed") == "true"
    assert pg.get_attribute(".tools button:has-text('Select')", "aria-pressed") == "false"


def test_the_timeline_is_a_slider_that_tells_the_frame(page):
    pg, ds, base = page
    _viewer(pg, base, 12)
    tl = pg.locator("canvas.tlc")
    assert tl.get_attribute("role") == "slider" and tl.get_attribute("aria-valuemin") == "0" and tl.get_attribute("aria-valuemax") == "59"
    assert tl.get_attribute("aria-valuenow") == "12" and "frame 12 of 59" in tl.get_attribute("aria-valuetext")
    tl.focus()
    pg.keyboard.press("ArrowRight")
    pg.keyboard.press("Shift+ArrowRight")
    _settled(pg, 23)
    pg.wait_for_function("document.querySelector('canvas.tlc').getAttribute('aria-valuenow') === '23'")


def test_dialogs_work_with_the_keyboard_and_give_focus_back(page):
    pg, ds, base = page
    pg.goto(f"{base}/#/overview")
    pg.reload()
    pg.wait_for_selector("h1")
    pg.focus("#author-btn")
    pg.keyboard.press("Enter")
    pg.wait_for_selector(".modal[role=dialog][aria-modal=true]")
    pg.wait_for_function("document.activeElement.tagName === 'INPUT' && !!document.activeElement.closest('.modal')")      # focus moved into the dialog
    labelled = pg.evaluate("document.getElementById(document.querySelector('.modal').getAttribute('aria-labelledby')).textContent")
    assert "Who is editing" in labelled
    for _ in range(6):                                                                # Tab never leaves the dialog
        pg.keyboard.press("Tab")
        assert pg.evaluate("!!document.activeElement.closest('.modal')")
    pg.keyboard.press("Shift+Tab")
    assert pg.evaluate("!!document.activeElement.closest('.modal')")
    pg.keyboard.press("Escape")
    pg.wait_for_selector(".modal", state="detached")
    assert pg.evaluate("document.activeElement.id") == "author-btn"                   # focus is back where it was
    # a confirmation answers "no" on Escape and on a click outside, so nothing is left waiting
    pg.evaluate("(() => { window.__ans = 'pending'; confirmBox('Sure?', 'text').then(v => { window.__ans = v; }); })()")
    pg.wait_for_selector(".modal")
    pg.keyboard.press("Escape")
    pg.wait_for_function("window.__ans === false")
    pg.evaluate("(() => { window.__ans = 'pending'; confirmBox('Sure?', 'text').then(v => { window.__ans = v; }); })()")
    pg.wait_for_selector(".modal")
    pg.mouse.click(5, 5)
    pg.wait_for_function("window.__ans === false")
    pg.evaluate("(() => { window.__ans = 'pending'; confirmBox('Sure?', 'text').then(v => { window.__ans = v; }); })()")
    pg.wait_for_selector(".modal")
    pg.click(".modal button:has-text('OK')")
    pg.wait_for_function("window.__ans === true")


def test_messages_are_announced_errors_as_alerts(page):
    pg, ds, base = page
    pg.goto(f"{base}/#/overview")
    pg.reload()
    pg.wait_for_selector("h1")
    pg.evaluate("toast('Saved'); toast('That did not work', true)")
    assert pg.get_attribute(".toast:has-text('Saved')", "role") == "status"
    assert pg.get_attribute(".toast:has-text('That did not work')", "role") == "alert"


def test_the_skip_link_moves_to_the_page_without_leaving_it(page):
    pg, ds, base = page
    pg.goto(f"{base}/#/labels")
    pg.reload()
    pg.wait_for_selector("h1")
    pg.keyboard.press("Tab")
    assert pg.evaluate("document.activeElement.className") == "skip"
    pg.keyboard.press("Enter")
    assert pg.evaluate("document.activeElement.id") == "main" and pg.evaluate("location.hash") == "#/labels"


def test_the_drawer_button_says_whether_the_list_is_open(page):
    pg, ds, base = page
    pg.goto(f"{base}/#/overview")
    pg.reload()
    pg.wait_for_selector("h1")
    assert pg.get_attribute("#changes-btn", "aria-expanded") == "false"
    pg.click("#changes-btn")
    assert pg.get_attribute("#changes-btn", "aria-expanded") == "true" and pg.get_attribute("#drawer", "aria-label") == "Pending changes"
    pg.click("#changes-btn")


# ---------------------------------------------------------------- drawing without a pointer
def _open_numbers(pg):
    if pg.locator(".bynum").get_attribute("open") is None:
        pg.click(".bynum summary")


def test_boxes_and_spans_can_be_added_and_edited_by_typing_numbers(page):
    pg, ds, base = page
    _viewer(pg, base, 47)
    before = ds.annotations("clip", frames=(47, 47), kind="box").num_rows
    pg.keyboard.press("b")
    _open_numbers(pg)
    pg.fill(".bynum input[aria-label=Left]", "10")
    pg.fill(".bynum input[aria-label=Top]", "8")
    pg.fill(".bynum input[aria-label=Right]", "50")
    pg.fill(".bynum input[aria-label=Bottom]", "40")
    pg.click("button:has-text('Add box')")
    pg.wait_for_function("document.querySelector('#changes-count').textContent === '1'")
    pg.wait_for_selector(".annrow.pend")
    # the new box is selected: change it with the keyboard-friendly fields
    pg.fill("#main input[aria-label=Right] >> nth=-1", "60")
    pg.click("button:has-text('Apply box')")
    pg.wait_for_function("V.rows.some(r => r.pending && r.x1 === 60)")
    # mistakes are explained, not silently ignored
    pg.keyboard.press("b")
    _open_numbers(pg)
    pg.fill(".bynum input[aria-label=Left]", "70")
    pg.fill(".bynum input[aria-label=Right]", "20")
    pg.click("button:has-text('Add box')")
    pg.wait_for_selector(".toast:has-text('at least 2 pixels')")
    pg.fill(".bynum input[aria-label=Left]", "-5")
    pg.fill(".bynum input[aria-label=Right]", "20")
    pg.click("button:has-text('Add box')")
    pg.wait_for_selector(".toast:has-text('must lie inside')")
    assert _draft_count(pg) == 1                                                          # the edit of the pending box merged into its addition
    pg.keyboard.press("s")
    _open_numbers(pg)
    pg.fill(".bynum input[aria-label='From frame']", "20")
    pg.fill(".bynum input[aria-label='To frame']", "25")
    pg.click("button:has-text('Add span')")
    pg.wait_for_selector(".toast:has-text('Labelled frames 20 to 25')")
    pg.fill(".bynum input[aria-label='From frame']", "30")
    pg.fill(".bynum input[aria-label='To frame']", "99")
    pg.click("button:has-text('Add span')")
    pg.wait_for_selector(".toast:has-text('between 0 and 59')")
    _commit(pg, "by numbers")
    t = ds.annotations("clip", frames=(47, 47), kind="box")
    assert t.num_rows == before + 1
    row = [r for r in t.to_pylist() if r["track_id"] is None][0]
    assert (row["x0"], row["y0"], row["x1"], row["y1"]) == (10, 8, 60, 40)
    m = [r for r in ds.annotations("clip", kind="marker").to_pylist() if r["start_frame"] == 20]
    assert m and m[0]["end_frame"] == 25


# ---------------------------------------------------------------- axe-core, when it is available
def test_axe_finds_nothing_on_any_page_or_state(page):
    src = _axe_source()
    if src is None:
        pytest.skip("axe-core not available (set VIDLABEL_AXE_JS, or `npm install axe-core` at the repository root)")
    pg, ds, base = page
    from vidlabel.agent import AgentSession
    AgentSession(ds.path, "scout").propose_boxes("clip", [{"frame": 55, "label": "person", "box": [2, 2, 20, 20], "confidence": 0.6}], note="a note")
    run = ("axe.run(document, {runOnly: {type: 'tag', values: ['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa','best-practice']}})"
           ".then(r => r.violations.map(v => v.id + ': ' + v.nodes.slice(0, 3).map(n => n.target.join(' ')).join(' | ')))")
    problems = []

    def audit(name):
        pg.add_script_tag(content=src)
        problems.extend(f"{scheme} {name}: {v}" for v in pg.evaluate(run))
    for scheme in ("light", "dark"):
        pg.emulate_media(color_scheme=scheme)
        for p in PAGES:
            pg.goto(f"{base}/#/{p}")
            pg.reload()
            pg.wait_for_selector("h1")
            pg.wait_for_timeout(500)
            audit(p)
        _viewer(pg, base, 10)
        for tool in ("b", "s", "m", "p", "v"):
            pg.keyboard.press(tool)
            pg.wait_for_timeout(150)
            audit("viewer tool " + tool)
        pg.click("#author-btn")
        pg.wait_for_selector(".modal")
        audit("dialog")
        pg.keyboard.press("Escape")
        pg.click("#changes-btn")
        pg.wait_for_timeout(200)
        audit("changes drawer")
        pg.click("#changes-btn")
    assert problems == []
