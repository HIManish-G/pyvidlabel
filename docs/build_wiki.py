"""Build a GitHub-wiki-ready folder from the pages in docs/.

    python docs/build_wiki.py OUTPUT_DIR [--repo OWNER/REPO]

GitHub wikis are flat folders of Markdown files named after their page title, with `Home.md`, `_Sidebar.md` and `_Footer.md` special.
The pages in docs/ stay the single source of truth (their examples and links are tested); this script only renames them, rewrites
relative links to wiki links, and adds the sidebar and footer. It exits with an error if any link in the output does not resolve.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent

# (docs file, wiki page name, sidebar label). The order here is the order of the sidebar.
SECTIONS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("Start here", [
        ("installation.md", "Installation", "Installation"),
        ("getting-started.md", "Getting-Started", "Getting started"),
        ("concepts.md", "Concepts", "Concepts"),
        ("glossary.md", "Glossary", "Glossary"),
    ]),
    ("Tutorials", [
        ("tutorial-animal-tracking.md", "Tutorial-Animal-Tracking", "Animal tracking"),
        ("tutorial-satellite-tiles.md", "Tutorial-Satellite-Tiles", "Satellite tiles"),
    ]),
    ("Guides", [
        ("howto.md", "How-To-Recipes", "How-to recipes"),
        ("views.md", "Views", "Views"),
        ("masks.md", "Masks", "Masks"),
        ("releases.md", "Releases", "Releases"),
        ("validation.md", "Validation", "Validation"),
        ("ui.md", "Web-Interface", "Web interface"),
        ("agents.md", "AI-Agents", "AI agents"),
        ("training.md", "Training", "Training"),
        ("converters.md", "Writing-a-Converter", "Writing a converter"),
    ]),
    ("Reference", [
        ("cli.md", "Command-Line-Reference", "Command line"),
        ("python-api.md", "Python-API-Reference", "Python API"),
        ("SPEC.md", "File-Format-Specification", "File format specification"),
    ]),
    ("Background", [
        ("architecture.md", "Architecture", "Architecture"),
        ("design-decisions.md", "Design-Decisions", "Design decisions"),
        ("performance.md", "Performance", "Performance"),
        ("compatibility.md", "Compatibility-and-Versioning", "Compatibility and versioning"),
        ("limitations.md", "Limitations-and-Roadmap", "Limitations and roadmap"),
        ("security.md", "Security", "Security"),
        ("ui-design.md", "UI-Design", "Web interface design"),
    ]),
    ("Project", [
        ("faq.md", "Troubleshooting-and-FAQ", "Troubleshooting and FAQ"),
        ("development.md", "Development", "Development"),
        ("changelog.md", "Changelog", "Changelog"),
    ]),
]
HOME = ("home.md", "Home")
EXCLUDED = {"index.md"}          # the repository's docs landing page; the wiki has Home instead
LINK = re.compile(r"(\]\()([^)\s]+)(\))")
FENCE = re.compile(r"(```.*?```)", re.S)


def page_map() -> dict[str, str]:
    m = {HOME[0]: HOME[1]}
    for _, items in SECTIONS:
        for fname, page, _label in items:
            m[fname] = page
    return m


def rewrite_links(text: str, mapping: dict[str, str], repo: str, source: str) -> str:
    """Relative `.md` links become wiki links; links to repository files become absolute GitHub links. Code blocks are untouched."""
    def fix(match: re.Match) -> str:
        target = match.group(2)
        if re.match(r"[a-z]+://|mailto:|#", target):
            return match.group(0)
        path, _, frag = target.partition("#")
        name = Path(path).name
        if path.endswith(".md") and (DOCS / path).resolve().parent == DOCS and name in mapping:
            return f"{match.group(1)}{mapping[name]}{'#' + frag if frag else ''}{match.group(3)}"
        resolved = (DOCS / path).resolve()
        try:
            rel = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            raise SystemExit(f"{source}: link {target} points outside the repository")
        return f"{match.group(1)}https://github.com/{repo}/blob/main/{rel}{'#' + frag if frag else ''}{match.group(3)}"

    parts = FENCE.split(text)
    return "".join(p if p.startswith("```") else LINK.sub(fix, p) for p in parts)


def sidebar() -> str:
    lines = ["**[vidlabel](Home)**", ""]
    for title, items in SECTIONS:
        lines.append(f"**{title}**")
        lines.extend(f"- [{label}]({page})" for _f, page, label in items)
        lines.append("")
    return "\n".join(lines)


def footer(version: str) -> str:
    return (f"vidlabel {version} · pages are generated from the repository's `docs/` folder; "
            f"to change one, edit `docs/` and run `python docs/build_wiki.py`.")


def _slug(heading: str) -> str:
    """GitHub's heading anchor: lowercase, drop punctuation except - and _, spaces become hyphens."""
    h = heading.strip().lower().replace("`", "")
    return re.sub(r"[^\w\- ]", "", h).replace(" ", "-")


def _anchors(text: str) -> set[str]:
    out, in_code = set(), False
    for line in text.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"#{1,6} ", line):
            out.add(_slug(line.lstrip("#")))
    return out


def check_links(out: Path) -> list[str]:
    """Problems in the built wiki: a link naming a page that is not there, or a #fragment that is not a heading of that page."""
    pages = {p.stem: p.read_text(encoding="utf-8") for p in out.glob("*.md")}
    problems = []
    for name, text in pages.items():
        body = FENCE.sub("", text)
        for target in re.findall(r"\]\(([^)\s]+)\)", body):
            if re.match(r"[a-z]+://|mailto:", target):
                continue
            page, _, frag = target.partition("#")
            page = page or name
            if page not in pages:
                problems.append(f"{name}: link to missing page {target}")
            elif frag and frag not in _anchors(pages[page]):
                problems.append(f"{name}: no heading for #{frag} in {page}")
    return problems


def build(out: Path, repo: str = "OWNER/REPO") -> list[Path]:
    mapping = page_map()
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.md"):
        old.unlink()
    written: list[Path] = []
    missing = sorted(p.name for p in DOCS.glob("*.md") if p.name not in mapping and p.name not in EXCLUDED)
    if missing:
        raise SystemExit(f"docs pages with no wiki page (add them to SECTIONS or EXCLUDED in build_wiki.py): {', '.join(missing)}")
    for fname, page in mapping.items():
        src = DOCS / fname
        text = src.read_text(encoding="utf-8")
        text = rewrite_links(text, mapping, repo, fname).replace("OWNER/REPO", repo)
        text = re.sub(r"<!-- (doctest|clitest) -->\n", "", text)        # test markers mean nothing on a wiki
        if fname not in (HOME[0],):
            text = text.rstrip("\n") + f"\n\n---\n*Source: [`docs/{fname}`](https://github.com/{repo}/blob/main/docs/{fname})*\n"
        target = out / f"{page}.md"
        target.write_text(text, encoding="utf-8")
        written.append(target)
    (out / "_Sidebar.md").write_text(sidebar() + "\n", encoding="utf-8")
    sys.path.insert(0, str(ROOT / "packages" / "vidlabel" / "src"))
    try:
        from vidlabel import __version__
    except Exception:  # the library need not be importable just to build the wiki
        __version__ = "dev"
    (out / "_Footer.md").write_text(footer(__version__) + "\n", encoding="utf-8")
    problems = check_links(out)
    if problems:
        raise SystemExit("broken links in the wiki:\n  " + "\n  ".join(problems))
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path, help="folder to write the wiki pages into")
    ap.add_argument("--repo", default="OWNER/REPO", help="GitHub owner/name, used for links to files outside the wiki")
    args = ap.parse_args()
    files = build(args.out, args.repo)
    print(f"wrote {len(files)} pages + _Sidebar.md + _Footer.md to {args.out}")
