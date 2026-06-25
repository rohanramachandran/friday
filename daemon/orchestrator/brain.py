"""Orchestrator brain: tool loop, dedup, forced summary, think stripping."""
import asyncio, json, re, logging
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from .memory import Memory
from tools.registry import run_tool

log = logging.getLogger("brain")
MODEL_ID = "mlx-community/Qwen3-14B-4bit"

SYSTEM = """You are FRIDAY, a personal AI assistant running locally on the user's Mac.
You are concise and direct. Keep responses brief and conversational; they will be spoken aloud.

Available tools:
- screenshot: read the user's screen (text via OCR). Args: {"query": "..."}
- search_memory: search past conversations
- run_code: execute Python. Args: {"code": "..."}
- system: control macOS (open_app, play_music, pause_music, volume, notify, current_app). Args: {"action": "...", "target": "..."}
- web_search: search the web. Args: {"query": "..."}

To call a tool, emit EXACTLY: <tool>{"name":"...","args":{...}}</tool>
After a tool result, ALWAYS tell the user the answer in plain words. Do NOT call the same tool twice.
Do NOT use <think> tags. Respond directly.
/no_think
"""

TOOL_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class Brain:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.memory = Memory()
        self.sampler = None

    async def warmup(self):
        loop = asyncio.get_event_loop()
        self.model, self.tokenizer = await loop.run_in_executor(None, load, MODEL_ID)
        self.sampler = make_sampler(temp=0.7, top_p=0.9)

    def _build_prompt(self, user_text, screenshot_b64=None, tool_results=None):
        msgs = [{"role": "system", "content": SYSTEM}]
        msgs += self.memory.context_messages()
        content = user_text
        if screenshot_b64:
            content = f"[Screenshot attached]\n\n{user_text}"
        msgs.append({"role": "user", "content": content})
        if tool_results:
            joined = "\n\n".join(tool_results)
            msgs.append({"role": "user", "content": f"[Tool results]\n{joined}\n\n[Now respond to the user in 1-2 sentences. Do NOT call any more tools.]"})
        return self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def _clean(self, raw):
        """Strip <think> blocks and any tool tags from text."""
        out = THINK_RE.sub("", raw)
        out = re.sub(r"<tool>.*?</tool>", "", out, flags=re.DOTALL)
        out = re.sub(r"<tool>.*$", "", out, flags=re.DOTALL)
        out = re.sub(r"<tool?$", "", out)  # partial open
        out = re.sub(r"^<tool[^>]*>?", "", out)  # leading partial
        return out

    async def _stream(self, prompt, max_tokens=512):
        """Generator that yields cleaned token deltas from the model."""
        loop = asyncio.get_event_loop()
        q = asyncio.Queue()
        def producer():
            try:
                for r in stream_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=self.sampler):
                    loop.call_soon_threadsafe(q.put_nowait, ("tok", r.text))
                loop.call_soon_threadsafe(q.put_nowait, ("end", None))
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, ("err", str(e)))
        loop.run_in_executor(None, producer)

        buf = ""
        emitted = 0
        while True:
            kind, val = await q.get()
            if kind == "end":
                yield ("end", buf)
                return
            if kind == "err":
                yield ("err", val)
                return
            buf += val
            # if there's an unclosed <think>, hold
            if "<think>" in buf and "</think>" not in buf:
                continue
            # if a tool call is starting/in-progress, stop emitting and return
            if "<tool>" in buf:
                if "</tool>" in buf:
                    yield ("tool", buf)
                    return
                # tool tag opened but not closed: keep buffering, don't emit
                continue
            cleaned = self._clean(buf)
            new = cleaned[emitted:]
            if new:
                yield ("delta", new)
                emitted = len(cleaned)

    def _parse_tool(self, buf):
        m = TOOL_RE.search(buf)
        if not m:
            return None
        raw = m.group(1).strip()
        try:
            return json.loads(raw)
        except Exception:
            nm = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
            if not nm:
                return None
            call = {"name": nm.group(1), "args": {}}
            cm = re.search(r'"code"\s*:\s*"(.+?)"\s*[}\]]', raw, re.DOTALL)
            if cm:
                call["args"]["code"] = cm.group(1)
            qm = re.search(r'"query"\s*:\s*"([^"]*)"', raw)
            if qm:
                call["args"]["query"] = qm.group(1)
            return call

    async def run(self, user_text, screenshot_b64=None):
        self.memory.add_user(user_text)
        full_response = ""
        tool_results = []
        called_tools = set()

        # ---- Phase 1: tool loop ----
        for iteration in range(5):
            prompt = self._build_prompt(user_text, screenshot_b64, tool_results)
            tool_buf = None

            phase1_buf = ""
            async for kind, payload in self._stream(prompt):
                if kind == "delta":
                    phase1_buf += payload
                elif kind == "tool":
                    tool_buf = payload
                    break
                elif kind == "end":
                    # No tool called. Emit phase1 text now.
                    if phase1_buf:
                        cleaned = self._clean(phase1_buf)
                        if cleaned:
                            yield {"type": "token", "text": cleaned}
                            full_response += cleaned
                    if tool_results:
                        break
                    self.memory.add_assistant(full_response)
                    yield {"type": "done"}
                    return
                elif kind == "err":
                    yield {"type": "token", "text": f"[error: {payload}]"}
                    self.memory.add_assistant(full_response)
                    yield {"type": "done"}
                    return

            if tool_buf is None:
                break

            call = self._parse_tool(tool_buf)
            if not call:
                log.warning(f"unparseable tool block: {tool_buf[:200]}")
                break

            name = call["name"]
            args = call.get("args", {})
            if name == "screenshot" and screenshot_b64 and "image_b64" not in args:
                args["image_b64"] = screenshot_b64

            if name in called_tools:
                log.warning(f"dedup: refusing to call {name} twice")
                break
            called_tools.add(name)

            yield {"type": "tool", "name": name, "status": "running"}
            try:
                result = await run_tool(name, args)
            except Exception as e:
                result = f"Tool error: {e}"
            yield {"type": "tool", "name": name, "status": "done"}
            log.warning(f"TOOL {name} returned {len(str(result))} chars")
            tool_results.append(f"Tool {name} returned:\n{result}")

        # ---- Phase 2: forced summary, only if phase 1 never produced an answer ----
        # (phase 1 already streams the model's post-tool answer; regenerating here
        # would say everything twice)
        if tool_results and not full_response:
            log.warning(f"FORCING SUMMARY over {len(tool_results)} tool result(s)")
            prompt = self._build_prompt(user_text, screenshot_b64, tool_results)

            async for kind, payload in self._stream(prompt, max_tokens=300):
                if kind == "delta":
                    yield {"type": "token", "text": payload}
                    full_response += payload
                elif kind == "tool":
                    # ignore further tool attempts in summary phase
                    log.warning("summary phase tried to call tool, ignoring")
                    break
                elif kind in ("end", "err"):
                    break

        self.memory.add_assistant(full_response)
        yield {"type": "done"}
