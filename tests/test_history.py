# tests/test_history.py
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine
from sqlmodel.pool import StaticPool

TEST_DATABASE_URL = "sqlite://"

@pytest.fixture(name="engine")
def engine_fixture():
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    yield engine
    SQLModel.metadata.drop_all(engine)

@pytest.fixture(name="client")
def client_fixture(engine):
    from history.database import get_session
    from api.main import app

    def get_test_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = get_test_session
    # sans `with` : le lifespan (chargement des modèles) n'est pas exécuté
    client = TestClient(app, headers={"X-API-Key": "test-key", "X-Session-Id": str(uuid.uuid4())})
    yield client
    app.dependency_overrides.clear()


def other_session(client):
    """Second client (autre session) sur la même base."""
    return TestClient(client.app, headers={"X-API-Key": "test-key", "X-Session-Id": str(uuid.uuid4())})

TRANSLATE_PAYLOAD = {
    "action_type": "translate",
    "src_lang":    "francais",
    "tgt_lang":    "moore",
    "source_text": "Bonjour tout le monde",
    "result_text": "Ne y welame",
}

TTS_PAYLOAD = {
    "action_type": "tts",
    "src_lang":    "moore",
    "source_text": "Ne y welame",
    "speed":       1.0,
}

PIPELINE_PAYLOAD = {
    "action_type": "translate_and_speak",
    "src_lang":    "francais",
    "tgt_lang":    "moore",
    "source_text": "Bonjour",
    "result_text": "Ne y welame",
    "speed":       1.0,
}


class TestCreateHistory:
    def test_create_translate_entry(self, client):
        response = client.post("/history", json=TRANSLATE_PAYLOAD)
        assert response.status_code == 201
        data = response.json()
        assert data["action_type"] == "translate"
        assert data["src_lang"]    == "francais"
        assert data["tgt_lang"]    == "moore"
        assert "id" in data
        assert "created_at" in data

    def test_create_tts_entry(self, client):
        response = client.post("/history", json=TTS_PAYLOAD)
        assert response.status_code == 201
        assert response.json()["speed"] == 1.0

    def test_create_pipeline_entry(self, client):
        response = client.post("/history", json=PIPELINE_PAYLOAD)
        assert response.status_code == 201
        assert response.json()["action_type"] == "translate_and_speak"

    def test_create_ignores_client_user_id(self, client):
        response = client.post("/history", json={**TRANSLATE_PAYLOAD, "user_id": "pirate"})
        assert response.status_code == 201
        assert "user_id" not in response.json()

    def test_create_returns_valid_uuid(self, client):
        response = client.post("/history", json=TRANSLATE_PAYLOAD)
        assert response.status_code == 201
        uuid.UUID(response.json()["id"])

    def test_unknown_src_lang_returns_422(self, client):
        response = client.post("/history", json={**TRANSLATE_PAYLOAD, "src_lang": "klingon"})
        assert response.status_code == 422

    def test_unknown_tgt_lang_returns_422(self, client):
        response = client.post("/history", json={**TRANSLATE_PAYLOAD, "tgt_lang": "martien"})
        assert response.status_code == 422

    def test_missing_source_text_returns_422(self, client):
        response = client.post("/history", json={**TRANSLATE_PAYLOAD, "source_text": None})
        assert response.status_code == 422

    def test_speed_too_high_returns_422(self, client):
        response = client.post("/history", json={**TTS_PAYLOAD, "speed": 5.0})
        assert response.status_code == 422

    def test_speed_too_low_returns_422(self, client):
        response = client.post("/history", json={**TTS_PAYLOAD, "speed": 0.1})
        assert response.status_code == 422

    def test_invalid_action_type_returns_422(self, client):
        response = client.post("/history", json={**TRANSLATE_PAYLOAD, "action_type": "teleportation"})
        assert response.status_code == 422


class TestListHistory:
    def _seed(self, client, n=3):
        return [client.post("/history", json=TRANSLATE_PAYLOAD).json()["id"] for _ in range(n)]

    def test_list_returns_all_entries(self, client):
        self._seed(client, 3)
        data = client.get("/history").json()
        assert data["total"] == 3
        assert len(data["items"]) == 3

    def test_list_pagination_limit(self, client):
        self._seed(client, 3)
        data = client.get("/history?limit=2").json()
        assert data["total"] == 3
        assert len(data["items"]) == 2

    def test_list_pagination_offset(self, client):
        self._seed(client, 3)
        data = client.get("/history?offset=2").json()
        assert len(data["items"]) == 1

    def test_list_isolated_per_session(self, client):
        client.post("/history", json=TRANSLATE_PAYLOAD)
        other = other_session(client)
        other.post("/history", json=TRANSLATE_PAYLOAD)
        other.post("/history", json=TRANSLATE_PAYLOAD)
        assert client.get("/history").json()["total"] == 1
        assert other.get("/history").json()["total"] == 2

    def test_list_filter_by_action_type(self, client):
        client.post("/history", json=TRANSLATE_PAYLOAD)
        client.post("/history", json=TTS_PAYLOAD)
        data = client.get("/history?action_type=tts").json()
        assert data["total"] == 1
        assert data["items"][0]["action_type"] == "tts"

    def test_list_filter_by_lang(self, client):
        client.post("/history", json=TRANSLATE_PAYLOAD)
        client.post("/history", json={**TRANSLATE_PAYLOAD, "src_lang": "anglais", "tgt_lang": "dioula"})
        data = client.get("/history?lang=moore").json()
        assert data["total"] == 1

    def test_list_sorted_desc(self, client):
        self._seed(client, 3)
        items = client.get("/history").json()["items"]
        dates = [i["created_at"] for i in items]
        assert dates == sorted(dates, reverse=True)

    def test_list_empty(self, client):
        data = client.get("/history").json()
        assert data["total"] == 0
        assert data["items"] == []


class TestDeleteHistoryEntry:
    def test_delete_returns_204(self, client):
        entry_id = client.post("/history", json=TRANSLATE_PAYLOAD).json()["id"]
        assert client.delete(f"/history/{entry_id}").status_code == 204

    def test_delete_removes_from_list(self, client):
        entry_id = client.post("/history", json=TRANSLATE_PAYLOAD).json()["id"]
        client.delete(f"/history/{entry_id}")
        assert client.get("/history").json()["total"] == 0

    def test_delete_nonexistent_returns_404(self, client):
        assert client.delete(f"/history/{uuid.uuid4()}").status_code == 404

    def test_delete_invalid_uuid_returns_422(self, client):
        assert client.delete("/history/pas-un-uuid").status_code == 422


class TestClearHistory:
    def test_clear_returns_204(self, client):
        client.post("/history", json=TRANSLATE_PAYLOAD)
        assert client.delete("/history").status_code == 204

    def test_clear_removes_all_session_entries(self, client):
        client.post("/history", json=TRANSLATE_PAYLOAD)
        client.post("/history", json=TRANSLATE_PAYLOAD)
        client.delete("/history")
        assert client.get("/history").json()["total"] == 0

    def test_clear_does_not_touch_other_sessions(self, client):
        other = other_session(client)
        other.post("/history", json=TRANSLATE_PAYLOAD)
        client.delete("/history")
        assert other.get("/history").json()["total"] == 1

    def test_clear_idempotent(self, client):
        assert client.delete("/history").status_code == 204


class TestAccess:
    def test_wrong_api_key_returns_403(self, client):
        assert client.get("/history", headers={"X-API-Key": "faux"}).status_code == 403

    def test_missing_session_returns_422(self, client):
        c = TestClient(client.app, headers={"X-API-Key": "test-key"})
        assert c.get("/history").status_code == 422

    def test_malformed_session_returns_400(self, client):
        assert client.get("/history", headers={"X-Session-Id": "pas un uuid!"}).status_code == 400

    def test_cannot_delete_other_session_entry(self, client):
        entry_id = other_session(client).post("/history", json=TRANSLATE_PAYLOAD).json()["id"]
        assert client.delete(f"/history/{entry_id}").status_code == 404
