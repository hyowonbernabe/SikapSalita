"""
Edge-TTS wrapper with lazy per-(text, lang) audio cache.

Uses Microsoft Edge neural voices — sounds human in both English and Filipino.
Requires internet on first call for each unique phrase; subsequent calls are
served from the in-process cache (instant, works offline after warm-up).

Voices:
  en  → en-US-JennyNeural  (clear, natural American English)
  fil → fil-PH-BlessicaNeural  (natural Filipino/Tagalog female)
"""

from __future__ import annotations

import edge_tts

VOICES: dict[str, str] = {
    "en":  "en-US-JennyNeural",
    "fil": "fil-PH-BlessicaNeural",
}

# (text, lang) → MP3 bytes
_cache: dict[tuple[str, str], bytes] = {}


async def get_audio(text: str, lang: str = "en") -> bytes:
    """Return MP3 bytes for *text* in *lang*, generating and caching on first call."""
    key = (text, lang)
    if key not in _cache:
        voice = VOICES.get(lang, VOICES["en"])
        communicate = edge_tts.Communicate(text, voice, rate="-5%")
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        _cache[key] = b"".join(chunks)
    return _cache[key]


async def warm_cache(labels_en: list[str], labels_fil: list[str]) -> None:
    """Pre-generate audio for all known labels in both languages.

    Called at server startup so the first commit during a demo is instant.
    Failures are silently skipped — the cache fills in lazily for any misses.
    """
    import asyncio

    async def _one(text: str, lang: str) -> None:
        try:
            await get_audio(text, lang)
        except Exception:
            pass  # network blip — will retry on first real request

    tasks = [_one(t, "en") for t in labels_en] + [_one(t, "fil") for t in labels_fil]
    await asyncio.gather(*tasks)
