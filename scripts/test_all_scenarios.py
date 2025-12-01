"""Test tất cả scenarios: pass, resign, game over, và các edge cases."""

import asyncio
import httpx
import os
import sys
from pathlib import Path

# Fix encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

BASE_URL = "http://localhost:8000"


async def test_all_scenarios():
    """Test tất cả scenarios."""
    print("=" * 60)
    print("Testing All Scenarios")
    print("=" * 60)
    print("\n⚠️  Đảm bảo server đang chạy: uvicorn app.main:app --reload\n")

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        # 1. Authentication
        print("=" * 60)
        print("Step 1: Authentication")
        print("=" * 60)

        response = await client.post(
            "/auth/login", json={"username_or_email": "premium_test_user", "password": "testpass123"}
        )
        if response.status_code != 200:
            print(f"❌ Login failed: {response.status_code}")
            return

        access_token = response.json()["token"]["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}
        print("✅ User logged in")

        # 2. Test Pass Move
        print("\n" + "=" * 60)
        print("Step 2: Test Pass Move")
        print("=" * 60)

        # Create match
        response = await client.post("/matches/ai", json={"level": 1, "board_size": 9}, headers=headers)
        assert response.status_code == 201
        match_id_pass = response.json()["id"]
        print(f"✅ Match created: {match_id_pass}")

        # Make a move first
        move1 = {"x": 3, "y": 3, "move_number": 1, "color": "B"}
        await client.post(f"/matches/{match_id_pass}/move", json=move1, headers=headers)
        await asyncio.sleep(0.3)

        # Pass turn
        response = await client.post(
            f"/matches/{match_id_pass}/pass",
            json={"move_number": 3, "color": "B"},
            headers=headers,
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Pass move successful")
            print(f"   Response: {result.get('status')}")
            if "ai_move" in result:
                print(f"   AI responded after pass: {result['ai_move']}")
        else:
            print(f"❌ Pass failed: {response.status_code} - {response.text}")

        # 3. Test Resign
        print("\n" + "=" * 60)
        print("Step 3: Test Resign")
        print("=" * 60)

        # Create new match
        response = await client.post("/matches/ai", json={"level": 1, "board_size": 9}, headers=headers)
        assert response.status_code == 201
        match_id_resign = response.json()["id"]
        print(f"✅ Match created: {match_id_resign}")

        # Make a few moves
        for i, move_data in enumerate([(3, 3), (3, 4), (4, 3)], 1):
            move = {"x": move_data[0], "y": move_data[1], "move_number": i * 2 - 1, "color": "B"}
            await client.post(f"/matches/{match_id_resign}/move", json=move, headers=headers)
            await asyncio.sleep(0.3)

        # Resign
        response = await client.post(f"/matches/{match_id_resign}/resign", headers=headers)
        if response.status_code == 200:
            match_data = response.json()
            print(f"✅ Resign successful")
            print(f"   Result: {match_data.get('result')}")
            print(f"   Finished: {match_data.get('finished_at') is not None}")
        else:
            print(f"❌ Resign failed: {response.status_code} - {response.text}")

        # 4. Test Invalid Move
        print("\n" + "=" * 60)
        print("Step 4: Test Invalid Move")
        print("=" * 60)

        # Create new match
        response = await client.post("/matches/ai", json={"level": 1, "board_size": 9}, headers=headers)
        assert response.status_code == 201
        match_id_invalid = response.json()["id"]
        print(f"✅ Match created: {match_id_invalid}")

        # Try invalid move (out of bounds)
        invalid_move = {"x": 10, "y": 10, "move_number": 1, "color": "B"}
        response = await client.post(f"/matches/{match_id_invalid}/move", json=invalid_move, headers=headers)
        if response.status_code == 400 or response.status_code == 422:
            print(f"✅ Invalid move correctly rejected: {response.status_code}")
            print(f"   Error: {response.json().get('detail', 'N/A')}")
        else:
            print(f"⚠️  Unexpected response: {response.status_code} - {response.text}")

        # 5. Test Match History
        print("\n" + "=" * 60)
        print("Step 5: Test Match History")
        print("=" * 60)

        response = await client.get("/matches/history", headers=headers)
        if response.status_code == 200:
            matches = response.json()
            print(f"✅ Match history retrieved: {len(matches)} matches")
            if matches:
                print(f"   Latest match: {matches[0].get('id')}")
                print(f"   Result: {matches[0].get('result')}")
        else:
            print(f"❌ History failed: {response.status_code}")

        # 6. Test Get Match Details
        print("\n" + "=" * 60)
        print("Step 6: Test Get Match Details")
        print("=" * 60)

        response = await client.get(f"/matches/{match_id_pass}", headers=headers)
        if response.status_code == 200:
            match_data = response.json()
            print(f"✅ Match details retrieved")
            print(f"   ID: {match_data.get('id')}")
            print(f"   Board size: {match_data.get('board_size')}")
            print(f"   AI level: {match_data.get('ai_level')}")
            state = match_data.get("state")
            if state:
                print(f"   Current player: {state.get('to_move')}")
                print(f"   Total moves: {len(state.get('moves', []))}")
        else:
            print(f"❌ Get match failed: {response.status_code}")

        # 7. Test SGF Export
        print("\n" + "=" * 60)
        print("Step 7: Test SGF Export")
        print("=" * 60)

        response = await client.get(f"/matches/{match_id_pass}/sgf", headers=headers)
        if response.status_code == 200:
            sgf_data = response.json()
            print(f"✅ SGF Export successful")
            print(f"   Match ID: {sgf_data.get('match_id')}")
            sgf_content = sgf_data.get("sgf_content", "")
            print(f"   SGF length: {len(sgf_content)} chars")
            print(f"   SGF preview: {sgf_content[:100]}...")
        else:
            print(f"❌ SGF Export failed: {response.status_code} - {response.text}")

        # 8. Test Multiple AI Levels
        print("\n" + "=" * 60)
        print("Step 8: Test Multiple AI Levels")
        print("=" * 60)

        for level in [1, 2, 3, 4]:
            response = await client.post("/matches/ai", json={"level": level, "board_size": 9}, headers=headers)
            if response.status_code == 201:
                match_id = response.json()["id"]
                print(f"✅ AI Level {level} match created: {match_id}")

                # Make a move
                move = {"x": 3, "y": 3, "move_number": 1, "color": "B"}
                move_resp = await client.post(f"/matches/{match_id}/move", json=move, headers=headers)
                if move_resp.status_code == 200 and "ai_move" in move_resp.json():
                    ai_move = move_resp.json()["ai_move"]
                    print(f"   AI Level {level} responded: {ai_move}")
                await asyncio.sleep(0.2)

        print("\n" + "=" * 60)
        print("✅ All Scenarios Test Completed!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_all_scenarios())

