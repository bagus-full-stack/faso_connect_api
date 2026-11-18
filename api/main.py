# api/main.py
import os
import json
import hashlib
import logging
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Literal

import redis
from fastapi import FastAPI, HTTPException, Request, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from huggingface_hub import login
from slowapi.errors import RateLimitExceeded

from api.deps import (
    ALLOWED_ORIGINS, check_config, limiter, verify_api_key,
)
from api.metrics import (
    CACHE_EVENTS, CONTENT_TYPE_LATEST, HTTP_LATENCY, HTTP_REQUESTS,
    INFER_IN_FLIGHT, INFER_WAITING, generate_latest,
)
from translation.nllb_engine import NLLBTranslator, BURKINA_LANG_CODES
from tts.mms_engine import MMSTTSEngine, MMS_TTS_MODELS
from history.database import create_db_tables
from history.router import router as history_router

# ── Logging JSON structuré ────────────────────────────────────────────

# Attributs standard d'un LogRecord : tout le reste vient de `extra={...}`.
_STD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "timestamp": self.formatTime(record),
            "level":     record.levelname,
            "message":   record.getMessage(),
        }
        log.update({k: v for k, v in record.__dict__.items() if k not in _STD_ATTRS})
        return json.dumps(log, ensure_ascii=False, default=str)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

HF_TOKEN          = os.getenv("HF_TOKEN")
CACHE_TTL         = int(os.getenv("CACHE_TTL", "3600"))
CACHE_MAX_BYTES   = int(os.getenv("CACHE_MAX_BYTES", "1500000"))
INFER_TIMEOUT     = int(os.getenv("INFER_TIMEOUT", "60"))
INFER_CONCURRENCY = int(os.getenv("INFER_CONCURRENCY", "1"))
QUEUE_TIMEOUT     = int(os.getenv("QUEUE_TIMEOUT", "15"))
REDIS_URL         = os.getenv("REDIS_URL", "redis://redis:6379/0")
TTS_PRELOAD       = os.getenv("TTS_PRELOAD", "all")  # "all", "none" ou "moore,dioula,..."

# ── Cache Redis (connexion paresseuse, nouvelle tentative toutes les 30 s) ──

_cache = None
_cache_retry_at = 0.0

def _get_cache():
    global _cache, _cache_retry_at
    if _cache is not None:
        return _cache
    if time.time() < _cache_retry_at:
        return None
    try:
        client = redis.Redis.from_url(
            REDIS_URL, decode_responses=True, socket_connect_timeout=1, socket_timeout=1,
        )
        client.ping()
        _cache = client
        logger.info("Redis connecté", extra={"event": "redis_connected"})
    except Exception:
        _cache_retry_at = time.time() + 30
        logger.warning("Redis non disponible — cache désactivé",
                       extra={"event": "redis_unavailable"})
    return _cache

def _cache_failed():
    global _cache, _cache_retry_at
    _cache, _cache_retry_at = None, time.time() + 30

def cache_get(key: str):
    client = _get_cache()
    if client is None:
        return None
    try:
        value = client.get(key)
    except Exception:
        _cache_failed()
        return None
    CACHE_EVENTS.labels("hit" if value else "miss").inc()
    return json.loads(value) if value else None

def cache_set(key: str, value: dict):
    client = _get_cache()
    if client is None:
        return
    payload = json.dumps(value)
    if len(payload) > CACHE_MAX_BYTES:  # évite de saturer Redis avec de gros audios
        return
    try:
        client.setex(key, CACHE_TTL, payload)
    except Exception:
        _cache_failed()

def make_cache_key(prefix: str, *args) -> str:
    raw = ":".join(str(a) for a in args)
    return f"{prefix}:{hashlib.sha256(raw.encode()).hexdigest()}"

# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    check_config()

    # 1. Auth HuggingFace
    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=False)
        logger.info("HuggingFace authentifié", extra={"event": "hf_auth_ok"})
    else:
        logger.warning("HF_TOKEN manquant", extra={"event": "hf_auth_missing"})

    # 2. Base de données (migrations Alembic)
    create_db_tables()
    logger.info("Base de données prête", extra={"event": "db_ready"})

    # 3. NLLB-200
    logger.info("Chargement NLLB-200...", extra={"event": "model_loading", "model": "nllb"})
    t0 = time.time()
    app.state.translator = NLLBTranslator()
    logger.info("NLLB-200 prêt", extra={
        "event": "model_ready", "model": "nllb",
        "device": str(app.state.translator.device),
        "duration_s": round(time.time() - t0, 1),
    })

    # 4. MMS-TTS
    logger.info("Chargement MMS-TTS...", extra={"event": "model_loading", "model": "mms"})
    t0 = time.time()
    app.state.tts = MMSTTSEngine()
    if TTS_PRELOAD.lower() != "none":
        langs = list(MMS_TTS_MODELS) if TTS_PRELOAD.lower() == "all" else \
            [l.strip() for l in TTS_PRELOAD.split(",") if l.strip()]
        await asyncio.to_thread(app.state.tts.preload, langs)
    logger.info("MMS-TTS prêt", extra={
        "event": "model_ready", "model": "mms",
        "duration_s": round(time.time() - t0, 1),
    })

    yield

    logger.info("Arrêt de l'API", extra={"event": "shutdown"})

# ── Application ───────────────────────────────────────────────────────

app = FastAPI(
    title="FasoConnect — API Linguistique Burkinabè",
    description=(
        "Traduction automatique et synthèse vocale pour les langues du Burkina Faso.\n\n"
        "**Langues supportées :** Mooré · Dioula · Fulfulde · Gourmantchéma · Dagaare · Français · Anglais\n\n"
        "**Modèles :** Meta NLLB-200 distilled 600M + Meta MMS-TTS\n\n"
        "**Auth :** header `X-API-Key` requis. L'historique exige aussi `X-Session-Id` (UUID)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Middlewares ───────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key", "X-Session-Id"],
)

app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Trop de requêtes. Réessaie dans quelques secondes."},
    )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    duration = time.time() - t0
    route = request.scope.get("route")
    path = route.path if route else "unmatched"  # évite l'explosion de cardinalité
    HTTP_REQUESTS.labels(request.method, path, response.status_code).inc()
    HTTP_LATENCY.labels(path).observe(duration)
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code}",
        extra={
            "event":       "http_request",
            "method":      request.method,
            "path":        request.url.path,
            "status":      response.status_code,
            "duration_ms": round(duration * 1000),
            "ip":          request.client.host if request.client else "unknown",
        }
    )
    return response

# ── Router historique ─────────────────────────────────────────────────

app.include_router(history_router)

# ── Schémas ───────────────────────────────────────────────────────────

Lang    = Literal[tuple(BURKINA_LANG_CODES)]
TTSLang = Literal[tuple(MMS_TTS_MODELS)]

class TranslationRequest(BaseModel):
    text:     str  = Field(..., min_length=1, max_length=2000, examples=["Bonjour tout le monde"])
    src_lang: Lang = Field(..., examples=["francais"])
    tgt_lang: Lang = Field(..., examples=["moore"])

class TranslationResponse(BaseModel):
    translated_text: str
    src_lang:        str
    tgt_lang:        str
    cached:          bool = False

class TTSRequest(BaseModel):
    text:  str     = Field(..., min_length=1, max_length=500, examples=["Ne y welame"])
    lang:  TTSLang = Field(..., examples=["moore"])
    speed: float   = Field(default=1.0, ge=0.5, le=2.0)

class TTSResponse(BaseModel):
    audio_b64:        str
    sample_rate:      int
    duration_seconds: float
    lang:             str

class TranslateAndSpeakRequest(BaseModel):
    text:     str     = Field(..., min_length=1, max_length=2000)
    src_lang: Lang
    tgt_lang: TTSLang  # la cible doit avoir une voix : validé avant toute inférence
    speed:    float   = Field(default=1.0, ge=0.5, le=2.0)

# ── Inférence async : file d'attente bornée + timeout ─────────────────

_infer_sem: asyncio.Semaphore | None = None

def _release(sem: asyncio.Semaphore, task: asyncio.Future):
    INFER_IN_FLIGHT.dec()
    sem.release()
    if not task.cancelled():
        task.exception()  # marque l'exception comme lue

async def run_with_timeout(fn, *args, timeout: int = None):
    """
    Exécute une inférence synchrone dans un thread, sans bloquer l'event loop.

    - Au plus INFER_CONCURRENCY inférences en même temps ; les autres attendent
      QUEUE_TIMEOUT s max, puis 503.
    - Le slot n'est libéré que quand le thread a vraiment fini : un timeout (504)
      ne laisse donc pas des inférences fantômes s'empiler sur le GPU.
    """
    global _infer_sem
    if _infer_sem is None:
        _infer_sem = asyncio.Semaphore(INFER_CONCURRENCY)
    sem, limit = _infer_sem, timeout or INFER_TIMEOUT

    INFER_WAITING.inc()
    try:
        await asyncio.wait_for(sem.acquire(), QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Serveur occupé, réessaie dans quelques secondes.",
            headers={"Retry-After": "10"},
        )
    finally:
        INFER_WAITING.dec()

    INFER_IN_FLIGHT.inc()
    task = asyncio.ensure_future(asyncio.to_thread(fn, *args))
    task.add_done_callback(lambda t: _release(sem, t))
    try:
        return await asyncio.wait_for(asyncio.shield(task), limit)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Inférence trop longue (>{limit}s). Réessaie avec un texte plus court.",
        )

# ── Endpoints système ─────────────────────────────────────────────────

@app.get(
    "/health",
    tags=["Système"],
    summary="Healthcheck",
)
async def health():
    """Vérifie que l'API est opérationnelle — utilisé par Docker et les load balancers."""
    return {
        "status":  "ok",
        "version": "1.0.0",
        "models":  [NLLBTranslator.BASE_MODEL_ID.split("/")[-1], "mms-tts"],
        "cache":   "redis" if _get_cache() is not None else "disabled",
    }

@app.get(
    "/metrics",
    tags=["Système"],
    summary="Métriques Prometheus",
    dependencies=[Depends(verify_api_key)],
    include_in_schema=False,
)
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get(
    "/languages",
    tags=["Système"],
    summary="Langues disponibles",
    dependencies=[Depends(verify_api_key)],
)
async def list_languages():
    """Retourne les langues supportées pour la traduction et la synthèse vocale."""
    return {
        "translation_supported": list(BURKINA_LANG_CODES.keys()),
        "tts_supported":         list(MMS_TTS_MODELS.keys()),
        "nllb_codes":            BURKINA_LANG_CODES,
    }

# ── Traduction ────────────────────────────────────────────────────────

@app.post(
    "/translate",
    response_model=TranslationResponse,
    status_code=200,
    tags=["Traduction"],
    summary="Traduire un texte",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("20/minute")
async def translate(request: Request, req: TranslationRequest):
    """
    Traduit un texte entre deux langues supportées.

    - **Rate limit** : 20 requêtes/minute par IP
    - **Cache** : résultat mis en cache Redis 1h
    - **Async** : ne bloque pas les autres requêtes pendant l'inférence
    """
    cache_key = make_cache_key("trans", req.text, req.src_lang, req.tgt_lang)
    cached = cache_get(cache_key)
    if cached:
        return TranslationResponse(**{**cached, "cached": True})

    try:
        result = await run_with_timeout(
            request.app.state.translator.translate,
            req.text, req.src_lang, req.tgt_lang,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur traduction : {e}", extra={"event": "translation_error"})
        raise HTTPException(status_code=500, detail="Erreur interne lors de la traduction")

    data = {
        "translated_text": result,
        "src_lang":        req.src_lang,
        "tgt_lang":        req.tgt_lang,
    }
    cache_set(cache_key, data)
    return TranslationResponse(**data, cached=False)

# ── Synthèse vocale ───────────────────────────────────────────────────

@app.post(
    "/tts",
    response_model=TTSResponse,
    status_code=200,
    tags=["Synthèse vocale"],
    summary="Synthétiser un texte en audio",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("10/minute")
async def text_to_speech(request: Request, req: TTSRequest):
    """
    Génère un audio WAV encodé en base64 à partir d'un texte.

    - **Rate limit** : 10 requêtes/minute par IP
    - **Speed** : 0.5 (lent) → 1.0 (normal) → 2.0 (rapide), hauteur conservée
    - **Format** : WAV mono, encodé base64
    """
    cache_key = make_cache_key("tts", req.text, req.lang, req.speed)
    cached = cache_get(cache_key)
    if cached:
        return TTSResponse(**cached)

    try:
        result = await run_with_timeout(
            request.app.state.tts.synthesize,
            req.text, req.lang, req.speed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur TTS : {e}", extra={"event": "tts_error"})
        raise HTTPException(status_code=500, detail="Erreur interne lors de la synthèse vocale")

    data = {
        "audio_b64":        result.audio_b64,
        "sample_rate":      result.sample_rate,
        "duration_seconds": result.duration_seconds,
        "lang":             req.lang,
    }
    cache_set(cache_key, data)
    return TTSResponse(**data)

# ── Pipeline combiné ──────────────────────────────────────────────────

@app.post(
    "/translate-and-speak",
    status_code=200,
    tags=["Pipeline"],
    summary="Traduire puis synthétiser en audio",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("10/minute")
async def translate_and_speak(request: Request, req: TranslateAndSpeakRequest):
    """
    Pipeline combiné : traduction + synthèse vocale en une seule requête.

    - **Rate limit** : 10 requêtes/minute par IP
    - Retourne le texte traduit ET l'audio base64
    - Résultat mis en cache Redis 1h
    """
    cache_key = make_cache_key("pipeline", req.text, req.src_lang, req.tgt_lang, req.speed)
    cached = cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}

    # Étape 1 — Traduction
    try:
        translated = await run_with_timeout(
            request.app.state.translator.translate,
            req.text, req.src_lang, req.tgt_lang,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur traduction : {e}", extra={"event": "translation_error"})
        raise HTTPException(status_code=500, detail="Erreur interne lors de la traduction")

    # Étape 2 — Synthèse vocale sur le texte traduit
    try:
        audio = await run_with_timeout(
            request.app.state.tts.synthesize,
            translated, req.tgt_lang, req.speed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur TTS : {e}", extra={"event": "tts_error"})
        raise HTTPException(status_code=500, detail="Erreur interne lors de la synthèse vocale")

    data = {
        "original_text":    req.text,
        "translated_text":  translated,
        "src_lang":         req.src_lang,
        "tgt_lang":         req.tgt_lang,
        "audio_b64":        audio.audio_b64,
        "sample_rate":      audio.sample_rate,
        "duration_seconds": audio.duration_seconds,
        "cached":           False,
    }
    cache_set(cache_key, data)
    return data
