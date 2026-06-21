"""Tool registry: dispatch by name."""
from . import vision, system, code_exec, web, memory_search

TOOLS = {
    "screenshot": vision.screenshot_tool,
    "search_memory": memory_search.search_tool,
    "run_code": code_exec.run_tool,
    "system": system.system_tool,
    "web_search": web.search_tool,
}

async def run_tool(name: str, args: dict) -> str:
    fn = TOOLS.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    return await fn(**args)
