# tests/test_api.py — endpoints d'inférence avec moteurs factices
import asyncio
import json
import logging
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api import main
from api.deps import limiter

HEADERS = {"X-API-Key": "test-key"}


class FakeTranslator:
    def translate(self, text, src, tgt):
        return f"[{tgt}] {text}"


class FakeTTS:
    def synthesize(self, text, lang, speed=1.0):
        return SimpleNamespace(audio_b64="QUJD", sample_rate=16000, duration_seconds=1.0)


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    limiter.reset()
    monkeypatch.setattr(main, "_cache", None)
    monkeypatch.setattr(main, "_cache_retry_at", time.time() + 3600)  # pas de Redis en test
    monkeypatch.setattr(main, "_infer_sem", None)
    main.app.state.translator = FakeTranslator()
    main.app.state.tts = FakeTTS()


@pytest.fixture
def client():
    return TestClient(main.app, headers=HEADERS)


def test_health_is_public():
    assert TestClient(main.app).get("/health").status_code == 200


@pytest.mark.parametrize("path", ["/languages", "/metrics"])
def test_protected_routes_need_key(path):
    assert TestClient(main.app).get(path).status_code == 403
    assert TestClient(main.app, headers=HEADERS).get(path).status_code == 200


def test_translate_ok(client):
    r = client.post("/translate", json={"text": "Bonjour", "src_lang": "francais", "tgt_lang": "moore"})
    assert r.status_code == 200
    assert r.json()["translated_text"] == "[moore] Bonjour"


def test_translate_unknown_lang_422(client):
    r = client.post("/translate", json={"text": "x", "src_lang": "klingon", "tgt_lang": "moore"})
    assert r.status_code == 422


def test_translate_without_key_403():
    r = TestClient(main.app).post("/translate", json={"text": "x", "src_lang": "francais", "tgt_lang": "moore"})
    assert r.status_code == 403


def test_tts_ok_and_speed_bounds(client):
    ok = client.post("/tts", json={"text": "Ne y welame", "lang": "moore", "speed": 1.5})
    assert ok.status_code == 200 and ok.json()["audio_b64"] == "QUJD"
    bad = client.post("/tts", json={"text": "x", "lang": "moore", "speed": 3})
    assert bad.status_code == 422


def test_translate_and_speak_ok(client):
    r = client.post("/translate-and-speak", json={"text": "Bonjour", "src_lang": "francais", "tgt_lang": "dioula"})
    assert r.status_code == 200
    assert r.json()["translated_text"] == "[dioula] Bonjour"
    assert r.json()["audio_b64"] == "QUJD"


def test_translate_and_speak_rejects_unknown_target_before_inference(client):
    r = client.post("/translate-and-speak", json={"text": "x", "src_lang": "francais", "tgt_lang": "klingon"})
    assert r.status_code == 422


def test_rate_limit_429(client):
    codes = [client.post("/tts", json={"text": "x", "lang": "moore"}).status_code for _ in range(12)]
    assert codes[:10] == [200] * 10 and 429 in codes[10:]


def test_queue_timeout_returns_503(monkeypatch):
    monkeypatch.setattr(main, "QUEUE_TIMEOUT", 0.1)

    async def scenario():
        first = asyncio.create_task(main.run_with_timeout(lambda: time.sleep(0.5) or "ok"))
        await asyncio.sleep(0.05)  # le 1er tient le seul slot
        with pytest.raises(main.HTTPException) as exc:
            await main.run_with_timeout(lambda: "second")
        assert exc.value.status_code == 503
        assert await first == "ok"

    asyncio.run(scenario())


def test_timeout_504_keeps_slot_until_thread_ends(monkeypatch):
    async def scenario():
        with pytest.raises(main.HTTPException) as exc:
            await main.run_with_timeout(lambda: time.sleep(0.4), timeout=0.1)
        assert exc.value.status_code == 504
        monkeypatch.setattr(main, "QUEUE_TIMEOUT", 0.05)
        with pytest.raises(main.HTTPException) as busy:  # thread fantôme encore actif
            await main.run_with_timeout(lambda: "x")
        assert busy.value.status_code == 503
        await asyncio.sleep(0.5)
        assert await main.run_with_timeout(lambda: "x") == "x"

    asyncio.run(scenario())


def test_json_formatter_keeps_extra_fields():
    rec = logging.LogRecord("t", logging.INFO, "", 0, "msg", (), None)
    rec.event = "http_request"
    out = json.loads(main.JSONFormatter().format(rec))
    assert out["event"] == "http_request" and out["message"] == "msg"
