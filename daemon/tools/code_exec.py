"""Sandboxed Python execution via subprocess."""
import asyncio, subprocess, tempfile
from pathlib import Path

async def run_tool(code: str, timeout: int = 10) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _exec, code, timeout)

def _exec(code, timeout):
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(
            ["python3", path],
            capture_output=True, text=True, timeout=timeout
        )
        out = r.stdout.strip()
        err = r.stderr.strip()
        if err and not out:
            return f"Error: {err}"
        return out + (f"\n[stderr: {err}]" if err else "")
    except subprocess.TimeoutExpired:
        return f"Timeout after {timeout}s"
    finally:
        Path(path).unlink(missing_ok=True)
