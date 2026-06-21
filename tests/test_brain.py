"""Tool-call parsing and output cleaning in the brain."""
import json

from orchestrator.brain import Brain


def make_brain():
    return Brain()


def test_parse_tool_valid_json():
    b = make_brain()
    buf = 'text before <tool>{"name":"web_search","args":{"query":"weather"}}</tool> after'
    call = b._parse_tool(buf)
    assert call == {"name": "web_search", "args": {"query": "weather"}}


def test_parse_tool_missing_returns_none():
    b = make_brain()
    assert b._parse_tool("no tool call here") is None


def test_parse_tool_malformed_json_recovers_name_and_query():
    b = make_brain()
    buf = '<tool>{"name": "search_memory", "args": {"query": "interpretability"</tool>'
    call = b._parse_tool(buf)
    assert call["name"] == "search_memory"
    assert call["args"]["query"] == "interpretability"


def test_parse_tool_malformed_json_recovers_code():
    b = make_brain()
    buf = '<tool>{"name": "run_code", "args": {"code": "print(1)"}]</tool>'
    call = b._parse_tool(buf)
    assert call["name"] == "run_code"
    assert call["args"]["code"] == "print(1)"


def test_clean_strips_think_blocks():
    b = make_brain()
    assert b._clean("<think>internal</think>visible") == "visible"


def test_clean_strips_complete_and_partial_tool_tags():
    b = make_brain()
    assert b._clean('a<tool>{"name":"x"}</tool>b') == "ab"
    assert b._clean("answer<tool>{unfinished") == "answer"
    assert b._clean("<tool") == ""
