"""FRIDAY daemon: WebSocket server bridging the Swift app to local models."""
import asyncio, json, base64, logging, signal, sys
from pathlib import Path
import websockets
from orchestrator.brain import Brain
from voice.stt import STT
from voice.tts import TTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("friday")

class Daemon:
    def __init__(self):
        self.brain = Brain()
        self.stt = STT()
        self.tts = TTS()
        self.clients = set()

    async def handle(self, ws):
        self.clients.add(ws)
        log.info("client connected")
        await ws.send(json.dumps({"type": "ready"}))
        try:
            async for raw in ws:
                msg = json.loads(raw)
                await self.route(ws, msg)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            log.info("client disconnected")

    async def route(self, ws, msg):
        t = msg.get("type")
        if t == "audio":
            # base64 wav bytes → transcribe → process
            audio = base64.b64decode(msg["data"])
            text = await self.stt.transcribe(audio)
            await ws.send(json.dumps({"type": "transcript", "text": text}))
            if text.strip():
                await self.process_query(ws, text, msg.get("screenshot"))
        elif t == "text":
            await self.process_query(ws, msg["text"], msg.get("screenshot"))
        elif t == "ping":
            await ws.send(json.dumps({"type": "pong"}))

    async def process_query(self, ws, text, screenshot_b64=None):
        """Run brain → stream tokens → buffer sentences → TTS → send audio."""
        sentence_buf = ""
        async for event in self.brain.run(text, screenshot_b64):
            if event["type"] == "token":
                await ws.send(json.dumps({"type": "token", "text": event["text"]}))
                sentence_buf += event["text"]
                # flush on sentence boundary
                while True:
                    idx = self._sentence_end(sentence_buf)
                    if idx < 0:
                        break
                    sentence, sentence_buf = sentence_buf[:idx+1], sentence_buf[idx+1:]
                    audio = await self.tts.synth(sentence.strip())
                    if audio:
                        await ws.send(json.dumps({
                            "type": "audio",
                            "data": base64.b64encode(audio).decode()
                        }))
            elif event["type"] == "tool":
                await ws.send(json.dumps({"type": "tool", "name": event["name"], "status": event["status"]}))
            elif event["type"] == "done":
                if sentence_buf.strip():
                    audio = await self.tts.synth(sentence_buf.strip())
                    if audio:
                        await ws.send(json.dumps({
                            "type": "audio",
                            "data": base64.b64encode(audio).decode()
                        }))
                await ws.send(json.dumps({"type": "done"}))

    @staticmethod
    def _sentence_end(s):
        for i, c in enumerate(s):
            if c in ".!?\n" and (i == len(s)-1 or s[i+1] in " \n"):
                return i
        # also flush on comma if chunk is long enough (latency optimization)
        if len(s) > 60:
            for i, c in enumerate(s):
                if c == "," and i > 30:
                    return i
        return -1

async def main():
    d = Daemon()
    log.info("warming models...")
    await d.brain.warmup()
    log.info("FRIDAY ready on ws://127.0.0.1:8765")
    async with websockets.serve(d.handle, "127.0.0.1", 8765, max_size=50*1024*1024, ping_interval=None, ping_timeout=None):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
