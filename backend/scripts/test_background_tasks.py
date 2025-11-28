"""Test script cho Background Tasks và SGF Export."""

import asyncio
import httpx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_URL = "http://localhost:8000"


async def test_background_tasks():
    """Test background tasks và SGF export."""
    print("=" * 60)
    print("Background Tasks & SGF Export Test")
    print("=" * 60)
    print("\n⚠️  Đảm bảo server đang chạy: uvicorn app.main:app --reload\n")

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        # 1. Login
        print("=" * 60)
        print("Step 1: Authentication")
        print("=" * 60)

        response = await client.post(
            "/auth/login", json={"username_or_email": "premium_test_user", "password": "testpass123"}
        )
        if response.status_code != 200:
            print(f"❌ Login failed: {response.status_code} - {response.text}")
            return

        access_token = response.json()["token"]["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}
        print("✅ User logged in")

        # 2. Create match và make moves
        print("\n" + "=" * 60)
        print("Step 2: Create Match & Make Moves")
        print("=" * 60)

        response = await client.post("/matches/ai", json={"level": 1, "board_size": 9}, headers=headers)
        assert response.status_code == 201
        match_id = response.json()["id"]
        print(f"✅ Match created: {match_id}")

        # Make a few moves
        moves = [
            {"x": 3, "y": 3, "move_number": 1, "color": "B"},
            {"x": 3, "y": 4, "move_number": 2, "color": "B"},
            {"x": 4, "y": 3, "move_number": 3, "color": "B"},
        ]

        for move in moves:
            response = await client.post(f"/matches/{match_id}/move", json=move, headers=headers)
            if response.status_code == 200:
                print(f"✅ Move {move['move_number']} submitted")

        # 3. Test SGF Export
        print("\n" + "=" * 60)
        print("Step 3: Test SGF Export")
        print("=" * 60)

        response = await client.get(f"/matches/{match_id}/sgf", headers=headers)
        if response.status_code == 200:
            sgf_data = response.json()
            print(f"✅ SGF export successful!")
            print(f"   Match ID: {sgf_data.get('match_id')}")
            print(f"   SGF ID: {sgf_data.get('sgf_id')}")
            sgf_content = sgf_data.get("sgf_content", "")
            print(f"   SGF Content (first 200 chars): {sgf_content[:200]}...")
            
            # Verify SGF format
            if sgf_content.startswith("(") and "FF[4]" in sgf_content and "SZ[9]" in sgf_content:
                print("   ✅ SGF format valid")
            else:
                print("   ⚠️  SGF format may be invalid")
        else:
            print(f"❌ SGF export failed: {response.status_code} - {response.text}")

        # 4. Test ML Training Job (if admin)
        print("\n" + "=" * 60)
        print("Step 4: Test ML Training Job")
        print("=" * 60)

        training_payload = {
            "model_type": "policy",
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
        }
        response = await client.post("/ml/train", json=training_payload, headers=headers)
        if response.status_code == 200:
            job_data = response.json()
            print(f"✅ Training job queued!")
            print(f"   Job ID: {job_data.get('job_id')}")
            print(f"   Status: {job_data.get('status')}")
            
            # Wait a bit và check status
            await asyncio.sleep(2)
            # Note: Cần endpoint để check job status, hiện tại chưa có
        else:
            print(f"⚠️  Training job request: {response.status_code} - {response.text}")

        # 5. Check Cache Stats
        print("\n" + "=" * 60)
        print("Step 5: Check Cache Stats")
        print("=" * 60)

        response = await client.get("/premium/cache/stats", headers=headers)
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Cache stats retrieved!")
            print(f"   Cache size: {stats.get('size', 0)}/{stats.get('max_size', 0)}")
            print(f"   Hits: {stats.get('hits', 0)}, Misses: {stats.get('misses', 0)}")
            print(f"   Hit rate: {stats.get('hit_rate', 0):.1%}")
        else:
            print(f"⚠️  Cache stats: {response.status_code} - {response.text}")

        print("\n" + "=" * 60)
        print("✅ Background Tasks Test Completed!")
        print("=" * 60)
        print("\n💡 Note: Background tasks (cache cleanup, statistics update)")
        print("   đang chạy tự động trong background khi server start.")


if __name__ == "__main__":
    asyncio.run(test_background_tasks())

