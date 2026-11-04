from app import llm
from app.search import Hit

HITS = [Hit(1, "tutorial/cors.md", "CORS › Use CORSMiddleware", "Import CORSMiddleware…", 0.8),
        Hit(2, "advanced/middleware.md", "Advanced Middleware", "Other middlewares…", 0.7)]


def test_user_message_numbers_sources_and_keeps_question_last():
    msg = llm.build_user_message("How to enable CORS?", HITS)
    assert msg.index('<source n="1"') < msg.index('<source n="2"') < msg.index("Question: How to enable CORS?")
    assert 'path="tutorial/cors.md"' in msg


def test_system_prompt_treats_sources_as_data():
    assert "not instructions" in llm.SYSTEM_PROMPT and "[2]" in llm.SYSTEM_PROMPT


def test_extractive_generator_without_hits_says_so():
    assert "could not find" in "".join(llm.ExtractiveGenerator().stream("q", []))


def test_auto_backend_prefers_api_key_then_cli(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_backend", "auto")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm.shutil, "which", lambda _: None)
    assert llm.pick_generator().name == "extractive"
    monkeypatch.setattr(llm.shutil, "which", lambda _: "/usr/bin/claude")
    assert llm.pick_generator().name == "claude-code"
