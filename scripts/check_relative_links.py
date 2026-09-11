"""Fail when a relative Markdown or HTML link points at a missing file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))(?:\s+[^)]*)?\)")
HTML_LINK = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']", re.IGNORECASE)
SKIP_SCHEMES = {"http", "https", "mailto", "data"}


def local_targets(markdown: Path) -> list[tuple[str, Path]]:
    text = markdown.read_text(encoding="utf-8")
    raw_targets = [
        match.group(1) or match.group(2) for match in MARKDOWN_LINK.finditer(text)
    ]
    raw_targets.extend(match.group(1) for match in HTML_LINK.finditer(text))
    targets: list[tuple[str, Path]] = []
    for raw in raw_targets:
        parsed = urlsplit(raw)
        if parsed.scheme.lower() in SKIP_SCHEMES or raw.startswith("#"):
            continue
        path_text = unquote(parsed.path)
        if not path_text:
            continue
        target = Path(path_text)
        if target.is_absolute():
            continue
        targets.append((raw, markdown.parent / target))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    missing: list[str] = []
    checked = 0
    for markdown in sorted(root.rglob("*.md")):
        if any(part in {".git", ".venv"} for part in markdown.parts):
            continue
        for raw, target in local_targets(markdown):
            checked += 1
            if not target.exists():
                missing.append(f"{markdown.relative_to(root)}: {raw}")
    if missing:
        raise SystemExit("Missing relative links:\n" + "\n".join(missing))
    print(f"PASS: {checked} relative Markdown/HTML links resolve")


if __name__ == "__main__":
    main()
