from app.agent_evals import summarize


def row(id, should_handoff, outcome, hit=None, steps=1, ms=100):
    return {"id": id, "kind": "natural", "should_handoff": should_handoff, "outcome": outcome,
            "evidence_hit": hit, "steps": [{}] * steps, "ms": ms}


def test_summary_counts_handoffs_and_misses():
    rows = [row("a", False, "answer", hit=True, steps=2),
            row("b", False, "handoff", hit=False),          # wrong hand-off: the docs cover it
            row("c", True, "handoff"),
            row("d", True, "answer"),                       # missed hand-off
            {"id": "e", "error": "boom", "ms": 5}]
    s = summarize(rows)
    assert s["errors"] == 1
    assert s["evidence_hit"] == 0.5
    assert s["handoff_precision"] == 0.5 and s["handoff_recall"] == 0.5
    assert s["missed"] == ["b"] and s["wrong_handoffs"] == ["b"] and s["missed_handoffs"] == ["d"]
    assert s["avg_steps"] == 1.25
