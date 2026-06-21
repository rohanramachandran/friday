"""FRIDAY voice TUI: wake-word listening plus typed input."""
import asyncio, json, sys, time, base64, io, threading, queue, wave, os
import numpy as np
import sounddevice as sd
import websockets
from rich.console import Console, Group
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.prompt import Prompt

console = Console()

BANNER = """[bold cyan]
███████╗██████╗ ██╗██████╗  █████╗ ██╗   ██╗
██╔════╝██╔══██╗██║██╔══██╗██╔══██╗╚██╗ ██╔╝
█████╗  ██████╔╝██║██║  ██║███████║ ╚████╔╝ 
██╔══╝  ██╔══██╗██║██║  ██║██╔══██║  ╚██╔╝  
██║     ██║  ██║██║██████╔╝██║  ██║   ██║   
╚═╝     ╚═╝  ╚═╝╚═╝╚═════╝ ╚═╝  ╚═╝   ╚═╝   
[/bold cyan][dim]  Say [bold]"Friday"[/bold] anytime to activate · type to use text · [bold]/q[/bold] to quit[/dim]
"""

SAMPLE_RATE = 16000
WAKE_WORD = "friday"
WAKE_WINDOW_S = 2.0       # rolling window length for wake detection
WAKE_HOP_S = 0.5          # how often we check
SILENCE_END_S = 1.2       # stop recording command after 1.2s silence
VAD_THRESH = 0.4

# ---- Models (lazy) ----
_wake_model = None
def wake_model():
    global _wake_model
    if _wake_model is None:
        from pywhispercpp.model import Model
        _wake_model = Model("tiny.en", n_threads=4, print_realtime=False, print_progress=False)
    return _wake_model

_vad = None
def vad():
    global _vad
    if _vad is None:
        from silero_vad import load_silero_vad
        _vad = load_silero_vad()
    return _vad


# ---- Mic stream singleton ----
class MicStream:
    """Continuous 16kHz mono int16 mic, shared across modes."""
    def __init__(self):
        self.q = queue.Queue()
        self.stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=512, callback=self._cb)
        self.stream.start()
        self.muted = False

    def _cb(self, indata, frames, t, status):
        if not self.muted:
            self.q.put(bytes(indata))

    def drain(self):
        while not self.q.empty():
            try: self.q.get_nowait()
            except queue.Empty: break

    def mute(self, m):
        self.muted = m
        if m: self.drain()


# ---- Wake word detection ----
def wait_for_wake(mic, status_cb=None):
    """Block until wake word is heard. Returns recent audio buffer."""
    window_samples = int(WAKE_WINDOW_S * SAMPLE_RATE)
    hop_samples = int(WAKE_HOP_S * SAMPLE_RATE)
    buf = bytearray()
    last_check = time.time()

    while True:
        try:
            chunk = mic.q.get(timeout=0.1)
        except queue.Empty:
            continue
        buf.extend(chunk)
        # keep rolling window
        max_bytes = window_samples * 2
        if len(buf) > max_bytes:
            del buf[:len(buf) - max_bytes]

        now = time.time()
        if now - last_check < WAKE_HOP_S:
            continue
        last_check = now
        if len(buf) < SAMPLE_RATE * 2 * 0.8:
            continue

        # transcribe rolling window
        wav = _wav_bytes(bytes(buf))
        text = _quick_transcribe(wav)
        if status_cb:
            status_cb(text)
        if WAKE_WORD in text.lower():
            return text


def _quick_transcribe(wav_bytes):
    """Tiny whisper for wake detection."""
    import tempfile
    from pathlib import Path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes); path = f.name
    try:
        segs = wake_model().transcribe(path)
        return " ".join(s.text for s in segs).strip()
    except Exception:
        return ""
    finally:
        Path(path).unlink(missing_ok=True)


def record_until_silence(mic):
    """After wake fires, record full command via VAD."""
    import torch
    vmodel = vad()
    audio_buf = []
    silence_ms = 0
    speech_started = False
    max_silence_ms = int(SILENCE_END_S * 1000)
    pre_buf = []
    pre_keep = int(0.4 * SAMPLE_RATE / 512)
    start = time.time()
    mic.drain()

    while True:
        try:
            chunk = mic.q.get(timeout=0.1)
        except queue.Empty:
            if time.time() - start > 30: break
            continue
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if len(samples) < 512: continue
        prob = vmodel(torch.from_numpy(samples[:512]), SAMPLE_RATE).item()
        is_speech = prob > VAD_THRESH

        if not speech_started:
            pre_buf.append(chunk)
            if len(pre_buf) > pre_keep: pre_buf.pop(0)
            if is_speech:
                speech_started = True
                audio_buf.extend(pre_buf)
                audio_buf.append(chunk)
        else:
            audio_buf.append(chunk)
            if is_speech: silence_ms = 0
            else:
                silence_ms += 32
                if silence_ms > max_silence_ms: break
        if time.time() - start > 30: break

    return _wav_bytes(b"".join(audio_buf)) if audio_buf else None


def _wav_bytes(pcm):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


# ---- Audio playback ----
class AudioPlayer:
    def __init__(self):
        self.q = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def play(self, wav):
        self.q.put(wav)

    def _worker(self):
        while True:
            wav = self.q.get()
            try:
                with wave.open(io.BytesIO(wav), "rb") as w:
                    rate = w.getframerate()
                    frames = w.readframes(w.getnframes())
                pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                sd.play(pcm, rate); sd.wait()
            except Exception as e:
                console.print(f"[red]audio err: {e}[/red]")


def beep(freq=880, dur=0.12):
    """Acknowledgement chime when wake fires."""
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), False)
    wave_arr = 0.3 * np.sin(2 * np.pi * freq * t) * np.exp(-3 * t)
    sd.play(wave_arr.astype(np.float32), SAMPLE_RATE); sd.wait()


# ---- Main ----
async def main():
    console.clear()
    console.print(BANNER)

    try:
        ws = await websockets.connect(
            "ws://127.0.0.1:8765",
            max_size=50*1024*1024,
            ping_interval=None, ping_timeout=None)
    except Exception as e:
        console.print(f"[red]✗ daemon unreachable[/red]: {e}")
        return

    json.loads(await ws.recv())
    console.print(f"[green]● connected[/green]")

    # warm models
    console.print("[dim]warming wake-word model...[/dim]")
    wake_model()
    vad()
    console.print(f"[green]● listening for \"Friday\"[/green]\n")

    mic = MicStream()
    player = AudioPlayer()
    loop = asyncio.get_event_loop()

    # background wake listener writes to a queue
    wake_q: asyncio.Queue = asyncio.Queue()
    def wake_thread():
        while True:
            heard = wait_for_wake(mic)
            asyncio.run_coroutine_threadsafe(wake_q.put(heard), loop)

    threading.Thread(target=wake_thread, daemon=True).start()

    # also allow typed input
    text_q: asyncio.Queue = asyncio.Queue()
    def input_thread():
        while True:
            try:
                line = input()
                asyncio.run_coroutine_threadsafe(text_q.put(line), loop)
            except (EOFError, KeyboardInterrupt):
                asyncio.run_coroutine_threadsafe(text_q.put("__QUIT__"), loop)
                return
    threading.Thread(target=input_thread, daemon=True).start()

    while True:
        console.print("[dim cyan]◯ say \"friday ...\" or type a message[/dim cyan]")
        # wait for either wake or typed input
        get_wake = asyncio.create_task(wake_q.get())
        get_text = asyncio.create_task(text_q.get())
        done, pending = await asyncio.wait(
            [get_wake, get_text], return_when=asyncio.FIRST_COMPLETED)
        for p in pending: p.cancel()

        if get_text in done:
            text = get_text.result()
            if text in ("__QUIT__", "/q", "exit", "quit"):
                console.print("[dim]goodbye[/dim]"); return
            if not text.strip(): continue
            await ws.send(json.dumps({"type": "text", "text": text}))
            await render_response(ws, player, prompt_text=text)
            continue

        # wake fired
        heard = get_wake.result()
        beep()
        # extract the part after "friday"
        lower = heard.lower()
        idx = lower.find(WAKE_WORD)
        after_wake = heard[idx + len(WAKE_WORD):].strip(" ,.?!").strip()

        if after_wake and len(after_wake.split()) >= 2:
            # full command was in the wake window
            console.print(f"[yellow]● heard:[/yellow] [italic]\"{after_wake}\"[/italic]")
            await ws.send(json.dumps({"type": "text", "text": after_wake}))
            await render_response(ws, player, prompt_text=after_wake)
        else:
            # wake only: capture the follow-up command
            console.print("[yellow]● listening...[/yellow]")
            mic.drain()
            wav = await loop.run_in_executor(None, record_until_silence, mic)
            if not wav:
                console.print("[dim]nothing heard[/dim]"); continue
            await ws.send(json.dumps({
                "type": "audio",
                "data": base64.b64encode(wav).decode()}))
            await render_response(ws, player)


async def render_response(ws, player, prompt_text=None):
    console.print(Text("friday", style="bold magenta"), end="  ")
    response_buf = ""; tool_line = None; transcript = None
    t0 = time.time(); first_token = None

    with Live(console=console, refresh_per_second=20, transient=False) as live:
        while True:
            msg = json.loads(await ws.recv())
            t = msg["type"]
            if t == "transcript":
                transcript = msg["text"]
                live.update(_render(response_buf, tool_line, transcript))
            elif t == "token":
                if first_token is None: first_token = time.time() - t0
                response_buf += msg["text"]
                live.update(_render(response_buf, tool_line, transcript))
            elif t == "tool":
                name = msg["name"]
                if msg["status"] == "running":
                    tool_line = Spinner("dots", text=Text(f" using {name}...", style="yellow"))
                else:
                    tool_line = Text(f" ✓ {name}", style="dim green")
                live.update(_render(response_buf, tool_line, transcript))
            elif t == "audio":
                player.play(base64.b64decode(msg["data"]))
            elif t == "done":
                live.update(_render(response_buf, None, transcript))
                break

    if first_token:
        console.print(f"[dim]  {first_token:.1f}s first · {time.time()-t0:.1f}s total[/dim]\n")


def _render(text, tool_line, transcript):
    parts = []
    if transcript: parts.append(Text(f"  \"{transcript}\"", style="dim italic"))
    if tool_line is not None: parts.append(tool_line)
    if text: parts.append(Text(text, style="white"))
    if not parts: parts.append(Spinner("dots", text=Text(" thinking...", style="dim")))
    return Group(*parts)


if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
