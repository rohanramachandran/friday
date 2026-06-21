"""macOS system control via osascript."""
import asyncio, subprocess

async def system_tool(action: str, target: str = "") -> str:
    """Actions: open_app, play_music, pause_music, volume, say, notify, current_app."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run, action, target)

def _run(action, target):
    scripts = {
        "open_app": f'tell application "{target}" to activate',
        "play_music": 'tell application "Music" to play',
        "pause_music": 'tell application "Music" to pause',
        "volume": f'set volume output volume {target}',
        "say": f'say "{target.replace(chr(34), "")}"',
        "notify": f'display notification "{target}" with title "FRIDAY"',
        "current_app": 'tell application "System Events" to get name of first application process whose frontmost is true',
    }
    script = scripts.get(action)
    if not script:
        return f"Unknown action: {action}"
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "ok"
    except Exception as e:
        return f"Error: {e}"
