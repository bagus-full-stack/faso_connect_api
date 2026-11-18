# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Cache HuggingFace : /models est un volume nommé (voir docker-compose.yml)
ENV HF_HOME=/models \
    PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 app \
    && mkdir -p /app /models \
    && chown app:app /app /models

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app . .

USER app

# Le token n'est jamais écrit dans l'image : monté en secret le temps du build.
RUN --mount=type=secret,id=hf_token,uid=1000,required=false \
    sh -c '[ -f /run/secrets/hf_token ] && export HF_TOKEN="$(cat /run/secrets/hf_token)"; python preload_models.py'

EXPOSE 8000
# --proxy-headers : IP réelle du client derrière un proxy (voir FORWARDED_ALLOW_IPS)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
