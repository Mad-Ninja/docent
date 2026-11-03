from app.chunking import chunk_document, clean, pack, sections

DOC = """# First Steps { #first-steps }

The simplest FastAPI file could look like this:

{* ../../docs_src/first_steps/tutorial001_py310.py *}

/// tip

Use the <b>VS Code</b> extension.

///

## Run it { #run-it }

```console
# not a heading inside code
$ fastapi dev
```

### Check

Open the browser.
"""


def test_clean_drops_noise_and_keeps_meaning():
    text = clean(DOC)
    assert "{*" not in text and "#first-steps" not in text and "<b>" not in text
    assert "Tip:" in text and "VS Code" in text


def test_sections_keep_heading_path_and_ignore_code_comments():
    secs = sections(clean(DOC))
    paths = [p for p, _ in secs]
    assert paths == [["First Steps"], ["First Steps", "Run it"], ["First Steps", "Run it", "Check"]]
    assert "# not a heading" in secs[1][1]


def test_pack_respects_limit():
    body = "\n\n".join(["word " * 50] * 6)  # six ~250-char paragraphs
    parts = pack(body, 600)
    assert len(parts) == 3 and all(len(p) <= 600 for p in parts)


def test_chunk_document_titles_and_order():
    chunks = chunk_document("tutorial/first-steps.md", DOC)
    assert chunks[0].title == "First Steps"
    assert chunks[-1].heading == "First Steps › Run it › Check"
    assert [c.ord for c in chunks] == list(range(len(chunks)))
