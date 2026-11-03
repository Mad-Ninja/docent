"""Markdown → chunks. Each chunk keeps its heading path, so a hit can be cited as "Page › Section"."""
import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
ANCHOR_RE = re.compile(r"\s*\{\s*#[\w-]+\s*\}\s*$")  # "# First Steps { #first-steps }"
INCLUDE_RE = re.compile(r"^\s*\{\*.*\*\}\s*$")  # "{* ../../docs_src/x.py *}": code lives outside the docs
ADMONITION_RE = re.compile(r"^///\s*(\w+)?.*$")  # "/// tip" ... "///"
TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class Chunk:
    path: str
    title: str
    heading: str
    ord: int
    text: str


def clean(markdown: str) -> str:
    """Drop what carries no meaning for search: HTML, include directives, admonition fences, heading anchors."""
    out = []
    for line in markdown.splitlines():
        if INCLUDE_RE.match(line):
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
        pieces = [para] if len(para) <= limit else re.split(r"(?<=[.!?])\s+", para)
        for p in pieces:
            if cur and len(cur) + len(p) + 2 > limit:
                parts.append(cur)
                cur = ""
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        parts.append(cur)
    return parts


def chunk_document(path: str, markdown: str, limit: int = 1200) -> list[Chunk]:
    text = clean(markdown)
    secs = sections(text)
    title = next((p[0] for p, _ in secs if p), path)
    chunks = []
    for heading_path, body in secs:
        heading = " › ".join(heading_path) or title
        for part in pack(body, limit):
            chunks.append(Chunk(path, title, heading, len(chunks), part))
    return chunks
