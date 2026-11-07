"""Markdown → chunks. Each chunk keeps its heading path, so a hit can be cited as "Page › Section"."""
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
ANCHOR_RE = re.compile(r"\s*\{\s*#[\w-]+\s*\}\s*$")  # "# First Steps { #first-steps }"
INCLUDE_RE = re.compile(r"^\s*\{\*\s*(\S+)(.*?)\*\}\s*$")  # "{* ../../docs_src/x.py hl[2,6] *}": code lives outside the docs
LINES_RE = re.compile(r"ln\[([\d:,]+)\]")
MAX_CODE_CHARS = 2000  # a longer example would crowd out the prose around it
ADMONITION_RE = re.compile(r"^///\s*(\w+)?.*$")  # "/// tip" ... "///"
TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class Chunk:
    path: str
    title: str
    heading: str
    ord: int
    text: str


Include = Callable[[str, str], str | None]  # (path from the directive, its options) -> code, or None


def code_include(src_root: Path) -> Include:
    """Resolve "{* ../../docs_src/a/b.py ln[1:11] *}" to the file under src_root, keeping only the listed lines.

    Without this every code example of the docs is missing from the index, and the model writes code
    from memory — the answer judge flagged exactly that (an `app.mount(...)` line no source contained).
    """
    def resolve(ref: str, options: str) -> str | None:
        rel = ref.split("docs_src/", 1)[-1]
        f = src_root / "docs_src" / rel
        if "docs_src/" not in ref or not f.is_file():
            return None
        lines = f.read_text(encoding="utf-8").splitlines()
        m = LINES_RE.search(options)
        if m:
            keep: list[str] = []
            for part in m.group(1).split(","):
                a, _, b = part.partition(":")
                keep += lines[int(a) - 1:int(b or a)]
            lines = keep
        # no blank lines: the example stays one paragraph, so packing never splits it
        code = "\n".join(l for l in lines if l.strip())[:MAX_CODE_CHARS]
        return f"```{f.suffix.lstrip('.') or 'text'}\n{code}\n```"
    return resolve


def clean(markdown: str, include: Include | None = None) -> str:
    """Drop what carries no meaning for search: HTML, admonition fences, heading anchors; inline code includes."""
    out = []
    for line in markdown.splitlines():
        m = INCLUDE_RE.match(line)
        if m:
            code = include(m.group(1), m.group(2)) if include else None
            if code:
                out += ["", code, ""]
            continue
        m = ADMONITION_RE.match(line)
        if m:
            if m.group(1):
                out.append(f"{m.group(1).capitalize()}:")
            continue
        line = ANCHOR_RE.sub("", TAG_RE.sub("", line))
        out.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def sections(text: str) -> list[tuple[list[str], str]]:
    """Split by headings: [(heading path, body)]; the path is ["Page", "Section", "Subsection"]."""
    path: list[str] = []
    body: list[str] = []
    out = []
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING_RE.match(line)
        if m:
            if body and "".join(body).strip():
                out.append((list(path), "\n".join(body).strip()))
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2)]
            body = []
        else:
            body.append(line)
    if "".join(body).strip():
        out.append((list(path), "\n".join(body).strip()))
    return out


def pack(body: str, limit: int) -> list[str]:
    """Greedy packing of paragraphs up to `limit` chars; a single long paragraph is cut on sentence ends."""
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n", body):
        pieces = [para] if len(para) <= limit or para.startswith("```") else re.split(r"(?<=[.!?])\s+", para)
        for p in pieces:
            if cur and len(cur) + len(p) + 2 > limit:
                parts.append(cur)
                cur = ""
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        parts.append(cur)
    return parts


def chunk_document(path: str, markdown: str, limit: int = 1200, include: Include | None = None) -> list[Chunk]:
    text = clean(markdown, include)
    secs = sections(text)
    title = next((p[0] for p, _ in secs if p), path)
    chunks = []
    for heading_path, body in secs:
        heading = " › ".join(heading_path) or title
        for part in pack(body, limit):
            chunks.append(Chunk(path, title, heading, len(chunks), part))
    return chunks
