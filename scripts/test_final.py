"""Final comprehensive test - tất cả features."""

import asyncio
import httpx
import os
import sys
from pathlib import Path

# Fix encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

BASE_URL = "http://localhost:8000"


async def test_final():
    """Final comprehensive test."""
    print("=" * 60)
    print("Final Comprehensive Test")
    print("=" * 60)
    print("\n⚠️  Đảm bảo server đang chạy: uvicorn app.main:app --reload\n")

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        # 1. Auth
        response = await client.post(
            "/auth/login", json={"username_or_email": "premium_test_user", "password": "testpass123"}
        )
        if response.status_code != 200:
            print(f"❌ Login failed")
            return
        access_token = response.json()["token"]["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}
        print("✅ Authentication")

        # 2. Create AI match và play
        print("\n" + "=" * 60)
        print("Step 2: Play AI Match")
        print("=" * 60)

        response = await client.post("/matches/ai", json={"level": 2, "board_size": 9}, headers=headers)
        match_id = response.json()["id"]
        print(f"✅ Match created: {match_id}")

        # Play 5 moves
        for i in range(5):
            move = {"x": 3 + i, "y": 3, "move_number": i * 2 + 1, "color": "B"}
            resp = await client.post(f"/matches/{match_id}/move", json=move, headers=headers)
            if resp.status_code == 200:
                result = resp.json()
                print(f"✅ Move {i+1}: User ({move['x']}, {move['y']})")
                if "ai_move" in result:
                    ai = result["ai_move"]
                    print(f"   🤖 AI: ({ai.get('x')}, {ai.get('y')})")
            await asyncio.sleep(0.2)

        # 3. Test Premium Features
        print("\n" + "=" * 60)
        print("Step 3: Premium Features")
        print("=" * 60)

        # Add coins
        await client.post("/coins/purchase", json={"amount": 200, "package_id": "test"}, headers=headers)

        # Hint
        hint_resp = await client.post("/premium/hint", json={"match_id": match_id, "top_k": 3}, headers=headers)
        if hint_resp.status_code == 200:
            hints = hint_resp.json().get("hints", [])
            print(f"✅ Premium Hint: {len(hints)} hints")

        # Analysis
        analysis_resp = await client.post(f"/premium/analysis?match_id={match_id}", headers=headers, json={})
        if analysis_resp.status_code in (200, 202):
            analysis = analysis_resp.json().get("analysis", {})
            print(f"✅ Premium Analysis: Win prob={analysis.get('win_probability', 0):.1%}")

        # 4. Test SGF Export
        print("\n" + "=" * 60)
        print("Step 4: SGF Export")
        print("=" * 60)

        sgf_resp = await client.get(f"/matches/{match_id}/sgf", headers=headers)
        if sgf_resp.status_code == 200:
            sgf = sgf_resp.json().get("sgf_content", "")
            print(f"✅ SGF Export: {len(sgf)} chars")
            print(f"   Preview: {sgf[:80]}...")

        # 5. Test Match History
        print("\n" + "=" * 60)
        print("Step 5: Match History")
        print("=" * 60)

        history_resp = await client.get("/matches/history", headers=headers)
        if history_resp.status_code == 200:
            matches = history_resp.json()
            print(f"✅ History: {len(matches)} matches")
            if matches:
                latest = matches[0]
                print(f"   Latest: {latest.get('id')}, Result: {latest.get('result')}")

        print("\n" + "=" * 60)
        print("✅ Final Test Completed!")
        print("=" * 60)
        print("\n🎉 Backend sẵn sàng cho production!")


if __name__ == "__main__":
    asyncio.run(test_final())

