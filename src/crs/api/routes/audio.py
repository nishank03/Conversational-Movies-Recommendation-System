"""Voice routes — ElevenLabs STT + TTS with word-level timestamps.

Endpoints:
  POST /voice/transcribe  — audio file → transcribed text  (ElevenLabs Scribe)
  POST /voice/speak        — text → audio + word timestamps (ElevenLabs TTS)
  POST /voice/converse     — audio → STT → CRS → TTS in a single round-trip
"""
import base64
import json

import httpx
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from crs.api.dependencies import get_app_settings, get_engine, get_loader
from crs.crs_engines.base import BaseCRS, EngineContext
from crs.schemas import Message
from crs.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/voice", tags=["voice"])

# ElevenLabs endpoints
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"

# Default voice — "Adam" (clear, warm male voice)
DEFAULT_VOICE_ID = "pNInz6obpgDQGcFmaJgB"


# ------------------------------------------------------------------ #
# STT — ElevenLabs Scribe v2
# ------------------------------------------------------------------ #

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """STT: Transcribe audio using ElevenLabs Scribe v2."""
    settings = get_app_settings()
    api_key = settings.elevenlabs_api_key

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="CRS_ELEVENLABS_API_KEY not set.",
        )

    audio_bytes = await file.read()
    logger.info("Sending audio to ElevenLabs STT (%d bytes)", len(audio_bytes))

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                ELEVENLABS_STT_URL,
                headers={"xi-api-key": api_key},
                files={
                    "file": (
                        file.filename or "recording.webm",
                        audio_bytes,
                        file.content_type or "audio/webm",
                    )
                },
                data={"model_id": "scribe_v2"},
            )
            resp.raise_for_status()
            result = resp.json()

        text = result.get("text", "")
        logger.info("ElevenLabs STT transcription: %r", text[:120])
        return {"text": text}

    except httpx.HTTPStatusError as e:
        logger.error("ElevenLabs STT HTTP error %s: %s", e.response.status_code, e.response.text)
        raise HTTPException(status_code=e.response.status_code, detail="ElevenLabs STT error.")
    except Exception as e:
        logger.exception("Failed to transcribe audio.")
        raise HTTPException(status_code=500, detail=f"STT error: {e}")


# ------------------------------------------------------------------ #
# TTS — ElevenLabs with word-level timestamps
# ------------------------------------------------------------------ #

class SpeakRequest(BaseModel):
    text: str
    voice_id: str | None = None


@router.post("/speak")
async def speak_text(req: SpeakRequest):
    """TTS: Convert text to speech with word-level alignment timestamps."""
    settings = get_app_settings()
    api_key = settings.elevenlabs_api_key

    if not api_key:
        raise HTTPException(status_code=500, detail="CRS_ELEVENLABS_API_KEY not set.")

    voice_id = req.voice_id or DEFAULT_VOICE_ID
    url = f"{ELEVENLABS_TTS_URL}/{voice_id}/with-timestamps"

    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "text": req.text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    logger.info("ElevenLabs TTS request (%d chars)", len(req.text))

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            result = resp.json()

        if "audio_base64" not in result:
            raise HTTPException(status_code=500, detail="ElevenLabs returned invalid payload.")

        logger.info("ElevenLabs TTS OK")
        return result

    except httpx.HTTPStatusError as e:
        logger.error("ElevenLabs TTS error %s: %s", e.response.status_code, e.response.text)
        raise HTTPException(status_code=e.response.status_code, detail="ElevenLabs TTS error.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("TTS failed.")
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")


# ------------------------------------------------------------------ #
# Combined voice conversation endpoint
# ------------------------------------------------------------------ #

@router.post("/converse")
async def voice_converse(
    request: Request,
    file: UploadFile = File(...),
):
    """Full voice round-trip: audio → STT → CRS → TTS → response."""
    try:
        return await _voice_converse_impl(request, file)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[converse] Unhandled error in voice pipeline")
        raise HTTPException(status_code=500, detail=f"Voice pipeline error: {e}")


async def _voice_converse_impl(request: Request, file: UploadFile):
    settings = get_app_settings()
    api_key = settings.elevenlabs_api_key

    if not api_key:
        raise HTTPException(status_code=500, detail="CRS_ELEVENLABS_API_KEY not set.")

    # -------- Step 1: STT --------
    audio_bytes = await file.read()
    logger.info("[converse] Step 1 — STT (%d bytes)", len(audio_bytes))

    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            stt_resp = await http.post(
                ELEVENLABS_STT_URL,
                headers={"xi-api-key": api_key},
                files={"file": (file.filename or "recording.webm", audio_bytes, file.content_type or "audio/webm")},
                data={"model_id": "scribe_v2"},
            )
            stt_resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error("ElevenLabs STT HTTP error %s: %s", e.response.status_code, e.response.text)
        raise HTTPException(status_code=e.response.status_code, detail=f"Could not transcribe audio (ElevenLabs {e.response.status_code}).")
    except Exception as e:
        logger.exception("Failed to transcribe audio in converse endpoint.")
        raise HTTPException(status_code=500, detail=f"STT error: {e}")

    try:
        user_text = stt_resp.json().get("text", "").strip()
    except Exception as e:
        logger.error("[converse] Failed to parse STT response: %s", stt_resp.text[:200])
        raise HTTPException(status_code=500, detail=f"STT returned invalid response: {e}")

    if not user_text:
        raise HTTPException(status_code=400, detail="Could not understand the audio. Please try again.")

    logger.info("[converse] Transcribed: %r", user_text[:120])

    # -------- Step 2: CRS recommendation --------
    logger.info("[converse] Step 2 — CRS engine")

    try:
        engine: BaseCRS = get_engine(request)
        loader = get_loader(request)

        profile = None
        try:
            profile = loader.get_user_profile("user_001")
        except Exception:
            pass

        ctx = EngineContext(
            message=user_text,
            history=[],
            profile=profile,
        )

        result = await engine.recommend(ctx)
        reply = result.reply
        recs = [r.model_dump() for r in result.recommendations]
        logger.info("[converse] CRS reply (%d chars, %d recs)", len(reply), len(recs))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("CRS engine failed in converse endpoint.")
        raise HTTPException(status_code=500, detail=f"Recommendation engine error: {e}")

    # -------- Step 3: TTS with timestamps --------
    logger.info("[converse] Step 3 — TTS")

    voice_id = DEFAULT_VOICE_ID
    tts_url = f"{ELEVENLABS_TTS_URL}/{voice_id}/with-timestamps"

    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            tts_resp = await http.post(
                tts_url,
                json={
                    "text": reply,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                },
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            )
            tts_resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error("ElevenLabs TTS error %s: %s", e.response.status_code, e.response.text)
        raise HTTPException(status_code=e.response.status_code, detail=f"ElevenLabs TTS error ({e.response.status_code}).")
    except Exception as e:
        logger.exception("TTS failed in converse endpoint.")
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")

    try:
        tts_data = tts_resp.json()
    except Exception as e:
        logger.error("[converse] Failed to parse TTS response: %s", tts_resp.text[:200])
        raise HTTPException(status_code=500, detail=f"TTS returned invalid response: {e}")

    logger.info("[converse] Full voice round-trip complete")

    return {
        "user_text": user_text,
        "reply": reply,
        "recommendations": recs,
        "audio_base64": tts_data.get("audio_base64", ""),
        "alignment": tts_data.get("alignment", {}),
    }
