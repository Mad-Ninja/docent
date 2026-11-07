from app.judge import code_checks, parse_verdict, summarize
from app.search import Hit


def hit(path):
    return Hit(1, path, "h", "text", 0.9)


def test_code_checks_catch_missing_and_out_of_range_citations():
    ev = [hit("a.md"), hit("tutorial/cors.md")]
    c = code_checks("Use CORSMiddleware [2]. Also see [5].", ev, ["tutorial/cors.md"])
    assert c["cited"] == [2, 5] and c["bad_citation_numbers"] == [5] and c["cites_expected_page"]
    assert not code_checks("No sources here.", ev, ["tutorial/cors.md"])["has_citations"]
    assert code_checks("The documentation provided does not cover this.", ev, [])["refused"]


def test_parse_verdict_reads_first_object_and_flags_garbage():
    v = parse_verdict('Here: {"answers_question": true, "grounded": false, "citations_ok": true, '
                      '"unsupported": ["x"], "reason": "r"} done')
    assert v["grounded"] is False and v["unsupported"] == ["x"]
    assert "judge_error" in parse_verdict("no json")


def test_summary_requires_every_check_to_pass():
    good = {"id": "a", "answers_question": True, "grounded": True, "citations_ok": True,
            "has_citations": True, "bad_citation_numbers": [], "cites_expected_page": True, "refused": False}
    bad = {**good, "id": "b", "grounded": False, "reason": "invented a parameter"}
    s = summarize([good, bad, {"id": "c", "error": "x"}])
    assert s["pass"] == 0.5 and s["errors"] == 1 and s["failed"] == {"b": "invented a parameter"}
