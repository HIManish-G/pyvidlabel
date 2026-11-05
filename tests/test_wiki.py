"""The GitHub wiki is generated from docs/: every page is included, links resolve, and nothing from the tests leaks into it."""
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_wiki", ROOT / "docs" / "build_wiki.py")
build_wiki = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_wiki)


def test_wiki_builds_with_every_page_and_valid_links(tmp_path):
    files = build_wiki.build(tmp_path / "w", "acme/vidlabel")
    names = {p.name for p in files}
    assert {"Home.md", "Installation.md", "Releases.md", "File-Format-Specification.md"} <= names
    assert (tmp_path / "w" / "_Sidebar.md").exists() and (tmp_path / "w" / "_Footer.md").exists()
    assert build_wiki.check_links(tmp_path / "w") == []
    # every page in the sidebar exists and every page is in the sidebar
    side = (tmp_path / "w" / "_Sidebar.md").read_text()
    linked = set(re.findall(r"\]\(([^)]+)\)", side))
    assert linked == {p.stem for p in files}


def test_wiki_pages_have_no_test_markers_or_relative_md_links(tmp_path):
    build_wiki.build(tmp_path / "w", "acme/vidlabel")
    for p in (tmp_path / "w").glob("*.md"):
        text = re.sub(r"```.*?```", "", p.read_text(encoding="utf-8"), flags=re.S)
        assert not re.search(r"<!-- (doctest|clitest) -->\n```", p.read_text(encoding="utf-8")), p.name
        assert not re.search(r"\]\((?!https?://)[^)]*\.md[)#]", text), p.name
        assert "OWNER/REPO" not in p.read_text(encoding="utf-8")


def test_every_docs_page_is_in_the_wiki_or_deliberately_left_out():
    mapped = set(build_wiki.page_map())
    for page in (ROOT / "docs").glob("*.md"):
        assert page.name in mapped or page.name in build_wiki.EXCLUDED, page.name
