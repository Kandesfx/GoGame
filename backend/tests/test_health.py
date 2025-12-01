from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app import dependencies


def override_get_db():
    class Dummy:
        pass

    yield Dummy()


async def override_get_mongo_db():
    yield None


def test_health_endpoint():
    app = create_app()
    app.dependency_overrides[dependencies.get_db] = override_get_db
    app.dependency_overrides[dependencies.get_mongo_db] = override_get_mongo_db

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

