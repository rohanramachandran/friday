"""Text-to-speech via Kokoro-82M (CoreML/PyTorch backend)."""
import asyncio, io, wave
import numpy as np

_pipeline = None

def _load():
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code="a")  # American English
    return _pipeline

VOICE = "bf_emma"  # warm female; alternatives: am_michael, af_bella

class TTS:
    async def synth(self, text: str) -> bytes:
        if not text.strip():
            return b""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync, text)

    def _sync(self, text):
        pipe = _load()
        chunks = []
        for _, _, audio in pipe(text, voice=VOICE):
            if audio is not None:
                arr = np.asarray(audio)
                chunks.append(arr)
        if not chunks:
            return b""
        full = np.concatenate(chunks)
        # to 16-bit PCM WAV at 24kHz (Kokoro's native rate)
        pcm = (np.clip(full, -1, 1) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm.tobytes())
        return buf.getvalue()
