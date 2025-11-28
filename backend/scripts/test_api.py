"""Script test API endpoints."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from httpx import AsyncClient

from app.main import create_app

BASE_URL = "http://localhost:8000"


async def test_health(client: AsyncClient):
    """Test health endpoint."""
    print("🔍 Testing GET /health...")
    response = await client.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("✅ Health check passed")


async def test_auth_register(client: AsyncClient):
    """Test registration."""
    print("\n🔍 Testing POST /auth/register...")
    payload = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123",
    }
    response = await client.post(f"{BASE_URL}/auth/register", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 201:
        data = response.json()
        print(f"✅ Registration successful: user_id={data.get('user_id')}")
        return data.get("token", {}).get("access_token")
    else:
        print(f"⚠️  Registration response: {response.text}")
        return None


async def test_auth_login(client: AsyncClient):
    """Test login."""
    print("\n🔍 Testing POST /auth/login...")
    payload = {
        "username_or_email": "testuser",
        "password": "testpass123",
    }
    response = await client.post(f"{BASE_URL}/auth/login", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Login successful: user_id={data.get('user_id')}")
        return data.get("token", {}).get("access_token")
    else:
        print(f"⚠️  Login response: {response.text}")
        return None


async def test_create_ai_match(client: AsyncClient, token: str):
    """Test creating AI match."""
    print("\n🔍 Testing POST /matches/ai...")
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"level": 1, "board_size": 9}
    response = await client.post(f"{BASE_URL}/matches/ai", json=payload, headers=headers)
    print(f"   Status: {response.status_code}")
    if response.status_code == 201:
        data = response.json()
        print(f"✅ AI match created: match_id={data.get('id')}")
        return data.get("id")
    else:
        print(f"⚠️  Create match response: {response.text}")
        return None


async def test_get_match(client: AsyncClient, token: str, match_id: str):
    """Test getting match."""
    print("\n🔍 Testing GET /matches/{match_id}...")
    headers = {"Authorization": f"Bearer {token}"}
    response = await client.get(f"{BASE_URL}/matches/{match_id}", headers=headers)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Match retrieved: board_size={data.get('board_size')}, ai_level={data.get('ai_level')}")
        return data
    else:
        print(f"⚠️  Get match response: {response.text}")
        return None


async def test_submit_move(client: AsyncClient, token: str, match_id: str):
    """Test submitting a move."""
    print("\n🔍 Testing POST /matches/{match_id}/move...")
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"x": 3, "y": 3, "move_number": 1, "color": "B"}
    response = await client.post(f"{BASE_URL}/matches/{match_id}/move", json=payload, headers=headers)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Move submitted: {data}")
        if "ai_move" in data:
            print(f"   🤖 AI responded with move: {data['ai_move']}")
        return data
    else:
        print(f"⚠️  Submit move response: {response.text}")
        return None


async def main():
    """Chạy tất cả tests."""
    print("=" * 60)
    print("API Endpoints Test")
    print("=" * 60)
    print("\n⚠️  Đảm bảo server đang chạy: uvicorn app.main:app --reload")
    print()

    app = create_app()
    async with AsyncClient(app=app, base_url=BASE_URL) as client:
        # Test health
        await test_health(client)

        # Test auth
        token = await test_auth_register(client)
        if not token:
            token = await test_auth_login(client)

        if not token:
            print("\n❌ Cannot proceed without authentication token")
            return 1

        # Test matches
        match_id = await test_create_ai_match(client, token)
        if match_id:
            await test_get_match(client, token, match_id)
            await test_submit_move(client, token, match_id)

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

