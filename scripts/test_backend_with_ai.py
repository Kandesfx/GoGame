"""Test backend với gogame_py module thực tế."""

import asyncio
import httpx
import os
import sys
from pathlib import Path

# Fix encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Backend sẽ tự động dùng wrapper nếu cần
# Không cần import gogame_py trực tiếp ở đây

BASE_URL = "http://localhost:8000"


async def test_backend_with_ai():
    """Test backend với AI thực tế."""
    print("=" * 60)
    print("Backend Test với gogame_py AI")
    print("=" * 60)
    print("Backend sẽ tự động dùng AI wrapper nếu cần")
    print("\n⚠️  Đảm bảo server đang chạy: uvicorn app.main:app --reload")
    print("⚠️  Đảm bảo MSYS2 Python có sẵn: C:/msys64/mingw64/bin/python3.exe\n")

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        # 1. Login
        print("=" * 60)
        print("Step 1: Authentication")
        print("=" * 60)

        response = await client.post(
            "/auth/login", json={"username_or_email": "premium_test_user", "password": "testpass123"}
        )
        if response.status_code != 200:
            print(f"❌ Login failed")
            return

        access_token = response.json()["token"]["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}
        print("✅ User logged in")

        # 2. Create AI match
        print("\n" + "=" * 60)
        print("Step 2: Create AI Match")
        print("=" * 60)

        response = await client.post("/matches/ai", json={"level": 1, "board_size": 9}, headers=headers)
        assert response.status_code == 201
        match_id = response.json()["id"]
        print(f"✅ Match created: {match_id}")

        # 3. Make moves và test AI response
        print("\n" + "=" * 60)
        print("Step 3: Test AI Moves")
        print("=" * 60)

        # User đi Black, AI đi White
        # Move 1: User (Black)
        move1 = {"x": 3, "y": 3, "move_number": 1, "color": "B"}
        response = await client.post(f"/matches/{match_id}/move", json=move1, headers=headers)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Move 1 (User Black) submitted: ({move1['x']}, {move1['y']})")
            
            if "ai_move" in result:
                ai_move = result["ai_move"]
                print(f"   🤖 AI responded: {ai_move}")
            else:
                print(f"   ⚠️  No AI move in response")
                print(f"   Response: {result}")
        
        # Đợi một chút để AI xử lý
        await asyncio.sleep(0.5)
        
        # Move 2: User (Black) - sau khi AI đã đi
        move2 = {"x": 3, "y": 4, "move_number": 3, "color": "B"}  # move_number = 3 vì AI đã đi move 2
        response = await client.post(f"/matches/{match_id}/move", json=move2, headers=headers)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Move 3 (User Black) submitted: ({move2['x']}, {move2['y']})")
            
            if "ai_move" in result:
                ai_move = result["ai_move"]
                print(f"   🤖 AI responded: {ai_move}")
            else:
                print(f"   ⚠️  No AI move in response")
                print(f"   Response: {result}")

        # 4. Get match state
        print("\n" + "=" * 60)
        print("Step 4: Get Match State")
        print("=" * 60)

        response = await client.get(f"/matches/{match_id}", headers=headers)
        if response.status_code == 200:
            match_data = response.json()
            state = match_data.get("state")
            if state:
                print(f"✅ Match state retrieved")
                print(f"   Board size: {state.get('size')}")
                print(f"   Current player: {state.get('to_move')}")
                print(f"   Total moves: {len(state.get('moves', []))}")

        # 5. Test Premium Features với AI thực tế
        print("\n" + "=" * 60)
        print("Step 5: Test Premium Features với AI")
        print("=" * 60)

        # Add coins if needed
        balance_resp = await client.get("/coins/balance", headers=headers)
        balance = balance_resp.json().get("coins", 0)
        if balance < 50:
            purchase_resp = await client.post("/coins/purchase", json={"amount": 100, "package_id": "test"}, headers=headers)
            if purchase_resp.status_code == 200:
                print(f"✅ Added coins: {purchase_resp.json().get('coins', 0)}")

        # Test Hint
        hint_resp = await client.post(
            "/premium/hint", json={"match_id": match_id, "top_k": 3}, headers=headers
        )
        if hint_resp.status_code == 200:
            hints = hint_resp.json().get("hints", [])
            print(f"✅ Premium Hint: {len(hints)} hints")
            for i, hint in enumerate(hints[:2], 1):
                print(f"   Hint {i}: {hint.get('move')}, confidence={hint.get('confidence', 0):.2f}")
        elif hint_resp.status_code == 402:
            print(f"⚠️  Insufficient coins for hint")
        else:
            print(f"⚠️  Hint request failed: {hint_resp.status_code}")

        # Test Analysis
        analysis_resp = await client.post(f"/premium/analysis?match_id={match_id}", headers=headers, json={})
        if analysis_resp.status_code in (200, 202):
            analysis_data = analysis_resp.json()
            if "analysis" in analysis_data:
                analysis = analysis_data["analysis"]
                print(f"✅ Premium Analysis:")
                print(f"   Win probability: {analysis.get('win_probability', 0):.1%}")
                print(f"   Game phase: {analysis.get('game_phase')}")
        elif analysis_resp.status_code == 402:
            print(f"⚠️  Insufficient coins for analysis")
        else:
            print(f"⚠️  Analysis request failed: {analysis_resp.status_code}")

        print("\n" + "=" * 60)
        print("✅ Backend Test với AI Completed!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_backend_with_ai())

