from app.agent import Finish, Outcome, PromptedPlanner, Step, ToolCall, Tools, parse_decision, run_agent
from app.search import Hit

CORS = Hit(1, "tutorial/cors.md", "CORS", "Use CORSMiddleware…", 0.8)
MIDDLEWARE = Hit(2, "advanced/middleware.md", "Middleware", "Other middlewares…", 0.7)


class Scripted:
    """A planner that replays fixed decisions and records what it was shown."""
    name = "scripted"

    def __init__(self, *decisions):
        self.decisions, self.seen = list(decisions), []

    def start(self, question):
        pass

    def decide(self, observation, is_error=False):
        self.seen.append((observation, is_error))
        return self.decisions.pop(0) if self.decisions else Finish()


def tools(search=lambda q, k: [CORS, MIDDLEWARE], read=lambda p: [CORS] if p == "tutorial/cors.md" else []):
    return Tools(search, read)


def run(planner, **kw):
    items = list(run_agent("How to enable CORS?", planner, kw.pop("tools", tools()), **kw))
    return [i for i in items if isinstance(i, Step)], items[-1]


def test_search_then_finish_answers_with_deduplicated_evidence():
    steps, outcome = run(Scripted(ToolCall("search_docs", {"query": "cors"}), ToolCall("read_document", {"path": "tutorial/cors.md"})))
    assert isinstance(outcome, Outcome) and outcome.kind == "answer"
    assert [h.chunk_id for h in outcome.evidence] == [1, 2]
    assert [s.tool for s in steps] == ["search_docs", "read_document"]


def test_tool_errors_are_reported_back_to_the_model():
    planner = Scripted(ToolCall("read_document", {"path": "nope.md"}), ToolCall("drop_tables", {}), ToolCall("search_docs", {}))
    steps, outcome = run(planner)
    assert all(s.error for s in steps)
    assert "no document" in planner.seen[1][0] and planner.seen[1][1] is True
    assert "unknown tool" in planner.seen[2][0]
    assert outcome.kind == "answer" and outcome.evidence == []


def test_crashing_tool_does_not_break_the_loop():
    def boom(q, k):
        raise ConnectionError("db down")
    steps, outcome = run(Scripted(ToolCall("search_docs", {"query": "cors"})), tools=tools(search=boom))
    assert steps[0].error and "ConnectionError" in steps[0].result
    assert outcome.kind == "answer"


def test_step_budget_stops_a_looping_model():
    planner = Scripted(*[ToolCall("search_docs", {"query": "cors"})] * 10)
    steps, outcome = run(planner, max_steps=3)
    assert len(steps) == 3 and outcome.kind == "limit" and outcome.evidence


def test_budget_spent_without_evidence_hands_off():
    steps, outcome = run(Scripted(*[ToolCall("search_docs", {"query": "x y"})] * 5), tools=tools(search=lambda q, k: []), max_steps=2)
    assert outcome.kind == "handoff"


def test_handoff_ends_the_conversation_with_a_reason():
    steps, outcome = run(Scripted(ToolCall("handoff_to_human", {"reason": "billing question"}), ToolCall("search_docs", {"query": "x"})))
    assert outcome.kind == "handoff" and outcome.handoff_reason == "billing question" and len(steps) == 1


def test_parse_decision_tolerates_prose_and_rejects_garbage():
    assert parse_decision('Sure!\n{"tool": "search_docs", "input": {"query": "cors"}}') == ToolCall("search_docs", {"query": "cors"})
    assert isinstance(parse_decision('{"done": true}'), Finish)
    assert parse_decision("I think we are done").name == "invalid_reply"


def test_prompted_planner_feeds_results_into_the_transcript():
    prompts = []
    replies = iter(['{"tool": "search_docs", "input": {"query": "cors"}}', '{"done": true}'])
    planner = PromptedPlanner(complete=lambda system, prompt: prompts.append(prompt) or next(replies))
    steps, outcome = run(planner)
    assert outcome.kind == "answer" and len(steps) == 1
    assert "tutorial/cors.md" in prompts[1] and "Question: How to enable CORS?" in prompts[1]
