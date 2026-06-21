"""Speech-to-text via whisper.cpp (Metal-accelerated)."""
import asyncio, tempfile, wave
from pathlib import Path

_model = None

def _load():
    global _model
    if _model is None:
        from pywhispercpp.model import Model
        _model = Model("small.en", n_threads=6)
    return _model

class STT:
    async def transcribe(self, wav_bytes: bytes) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync, wav_bytes)

    def _sync(self, wav_bytes):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            path = f.name
        try:
            model = _load()
            segs = model.transcribe(path)
            return " ".join(s.text for s in segs).strip()
        finally:
            Path(path).unlink(missing_ok=True)
