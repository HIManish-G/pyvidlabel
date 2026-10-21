"""The documentation is tested: runnable examples are executed, links resolve, and generated pages are current."""
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGES = sorted(DOCS.glob("*.md")) + [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]

BLOCK = re.compile(r"<!-- (doctest|clitest) -->\n```(?:python|bash)\n(.*?)```", re.S)


def blocks(path, kind):
    return [m.group(2) for m in BLOCK.finditer(path.read_text(encoding="utf-8")) if m.group(1) == kind]


def slug(heading: str) -> str:
    """GitHub's heading anchor: lowercase, drop punctuation except - and _, spaces become hyphens."""
    h = heading.strip().lower().replace("`", "")
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    out, in_code = set(), False
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"#{1,6} ", line):
            out.add(slug(line.lstrip("#")))
    return out


# ---------- runnable examples ----------
@pytest.mark.parametrize("page", [p for p in PAGES if blocks(p, "doctest")], ids=lambda p: p.name)
def test_python_examples_run(page, tmp_path, monkeypatch):
    """Blocks marked <!-- doctest --> run in order, sharing one namespace, in an empty working directory."""
    pytest.importorskip("numpy")
    monkeypatch.chdir(tmp_path)
    ns: dict = {"__name__": "__doc__"}
    for i, code in enumerate(blocks(page, "doctest")):
        try:
            exec(compile(code, f"{page.name}[block {i}]", "exec"), ns)
        except Exception as e:  # show which block failed
            pytest.fail(f"{page.name} runnable block {i} failed: {type(e).__name__}: {e}\n---\n{code}")


@pytest.mark.parametrize("page", [p for p in PAGES if blocks(p, "clitest")], ids=lambda p: p.name)
def test_cli_examples_run(page, tmp_path, monkeypatch):
    from vidlabel.cli import app

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    for code in blocks(page, "clitest"):
        for line in code.strip().split("\n"):
            cmd = line.split("#")[0].strip()
            if not cmd:
                continue
            assert cmd.startswith("vidlabel "), cmd
            res = runner.invoke(app, cmd.split()[1:])
            assert res.exit_code == 0, f"`{cmd}` failed:\n{res.output}"


# ---------- links ----------
@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_internal_links_and_anchors_resolve(page):
    text = re.sub(r"```.*?```", "", page.read_text(encoding="utf-8"), flags=re.S)
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if re.match(r"[a-z]+://|mailto:", target):
            continue
        file_part, _, frag = target.partition("#")
        dest = (page.parent / file_part).resolve() if file_part else page
        assert dest.exists(), f"{page.name}: link to missing file {target}"
        if frag and dest.suffix == ".md":
            assert frag in anchors(dest), f"{page.name}: no heading for #{frag} in {dest.name}"


# ---------- generated pages ----------
def test_generated_reference_pages_are_current():
    res = subprocess.run([sys.executable, str(DOCS / "gen_reference.py"), "--check"], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr


def test_every_public_name_is_in_the_api_reference():
    import vidlabel

    api = (DOCS / "python-api.md").read_text(encoding="utf-8")
    missing = [n for n in vidlabel.__all__ if n != "SPEC_VERSION" and n not in api]
    assert not missing, f"missing from docs/python-api.md: {missing}"


# ---------- spec and concepts stay in step with the code ----------
def test_every_validation_code_is_documented():
    src = (ROOT / "packages/vidlabel/src/vidlabel/validate.py").read_text(encoding="utf-8")
    codes = set(re.findall(r'(?:report|err|warn)\(\s*(?:"(?:error|warning)",\s*)?"([a-z_]+)"', src))
    codes |= set(re.findall(r'Issue\("(?:error|warning)",\s*"([a-z_]+)"', src))
    codes.discard("")
    spec = (DOCS / "SPEC.md").read_text(encoding="utf-8")
    missing = sorted(c for c in codes if f"`{c}`" not in spec)
    assert len(codes) > 30 and not missing, f"validation codes missing from docs/SPEC.md: {missing}"


def test_every_annotation_column_and_command_is_described():
    from vidlabel_spec import COLUMN_NAMES

    concepts = (DOCS / "concepts.md").read_text(encoding="utf-8")
    assert [c for c in COLUMN_NAMES if f"`{c}`" not in concepts] == []
    cli = (DOCS / "cli.md").read_text(encoding="utf-8")
    from vidlabel.cli import app
    import typer

    assert [n for n in typer.main.get_command(app).commands if f"## `{n}`" not in cli] == []


def test_converter_guide_contains_the_tested_example_verbatim():
    example = (ROOT / "examples" / "convert_csv.py").read_text(encoding="utf-8").rstrip("\n")
    guide = (DOCS / "converters.md").read_text(encoding="utf-8")
    assert example in guide, "docs/converters.md is out of date: re-copy examples/convert_csv.py into the guide"


def test_versions_in_docs_match_the_code():
    from vidlabel_spec import SPEC_VERSION

    assert f"spec {SPEC_VERSION}" in (DOCS / "SPEC.md").read_text(encoding="utf-8").split("\n")[0]
    assert f"version {SPEC_VERSION}" in (DOCS / "index.md").read_text(encoding="utf-8")
