"""Tool registry dispatch and the code execution sandbox."""
import asyncio

from tools import registry, code_exec


def run(coro):
    return asyncio.run(coro)


def test_unknown_tool_is_reported():
    assert "Unknown tool" in run(registry.run_tool("nope", {}))


def test_run_code_captures_stdout():
    assert run(code_exec.run_tool("print(2 + 2)")) == "4"


def test_run_code_reports_errors():
    out = run(code_exec.run_tool("raise ValueError('boom')"))
    assert out.startswith("Error:")
    assert "boom" in out


def test_run_code_times_out():
    out = run(code_exec.run_tool("import time; time.sleep(5)", timeout=1))
    assert out == "Timeout after 1s"
