# api/deps.py — configuration, auth, session et rate limiting partagés
import hashlib
import logging
import os
import re
import secrets

from fastapi import Header, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

APP_ENV = os.getenv("APP_ENV", "production").lower()
IS_DEV = APP_ENV in ("dev", "development", "test")

API_KEY = os.getenv("API_KEY")
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGIN", "http://localhost:3000").split(",") if o.strip()
]


def check_config() -> None:
    """Refuse de démarrer en production avec une config ouverte."""
    problems = []
    if not API_KEY:
        problems.append("API_KEY manquante")
    if "*" in ALLOWED_ORIGINS:
        problems.append("ALLOWED_ORIGIN='*' interdit")
    if not problems:
        return
    msg = "Configuration non sécurisée : " + ", ".join(problems)
    if IS_DEV:
        logger.warning(msg + " (toléré car APP_ENV=%s)", APP_ENV)
    else:
        raise RuntimeError(msg + ". Définis-les dans .env ou mets APP_ENV=dev.")


# ── Auth API Key ──────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str = Security(api_key_header)):
    if not API_KEY:  # uniquement possible en dev (voir check_config)
        return True
    if not secrets.compare_digest((key or "").encode(), API_KEY.encode()):
        raise HTTPException(
            status_code=403,
            detail="Clé API invalide ou manquante. Ajoute le header X-API-Key.",
        )
    return True


# ── Session utilisateur (historique) ─────────────────────────────────
# L'identifiant vient d'un UUID aléatoire généré par le navigateur : il sert de
# secret porteur. On ne stocke que son hash, jamais la valeur brute.

_SESSION_RE = re.compile(r"[0-9a-fA-F-]{32,64}")


def get_user_id(x_session_id: str = Header(..., alias="X-Session-Id")) -> str:
    if not _SESSION_RE.fullmatch(x_session_id):
        raise HTTPException(status_code=400, detail="X-Session-Id invalide (UUID attendu).")
    return hashlib.sha256(x_session_id.encode()).hexdigest()[:32]


# ── Rate limiting ─────────────────────────────────────────────────────
# Derrière un proxy, lance uvicorn avec --proxy-headers et FORWARDED_ALLOW_IPS.

limiter = Limiter(key_func=get_remote_address)
