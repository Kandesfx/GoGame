"""Test script cho Premium Features."""

import asyncio
import httpx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings

settings = get_settings()
BASE_URL = "http://localhost:8000"


async def test_premium_features():
    """Test tất cả premium features."""
    print("=" * 60)
    print("Premium Features Test")
    print("=" * 60)
    print("\n⚠️  Đảm bảo server đang chạy: uvicorn app.main:app --reload")
    print("⚠️  Đảm bảo đã có user với coins đủ để test\n")

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        # 1. Register/Login user
        print("\n" + "=" * 60)
        print("Step 1: Authentication")
        print("=" * 60)

        # Try to register
        register_data = {
            "username": "premium_test_user",
            "email": "premium_test@example.com",
            "password": "testpass123",
        }
        response = await client.post("/auth/register", json=register_data)
        if response.status_code == 201:
            print("✅ User registered")
            user_data = response.json()
            access_token = user_data["token"]["access_token"]
        elif response.status_code in (400, 409):
            print("⚠️  User already exists, logging in...")
            response = await client.post(
                "/auth/login", json={"username_or_email": register_data["username"], "password": register_data["password"]}
            )
            if response.status_code == 200:
                user_data = response.json()
                access_token = user_data["token"]["access_token"]
                print("✅ User logged in")
            else:
                print(f"❌ Login failed: {response.status_code} - {response.text}")
                return
        else:
            print(f"❌ Registration failed: {response.status_code} - {response.text}")
            return

        headers = {"Authorization": f"Bearer {access_token}"}

        # 2. Add coins to user
        print("\n" + "=" * 60)
        print("Step 2: Add Coins")
        print("=" * 60)

        # Get current balance
        response = await client.get("/coins/balance", headers=headers)
        assert response.status_code == 200, f"Get balance failed: {response.text}"
        balance_data = response.json()
        balance = balance_data.get("coins", 0)
        print(f"Current balance: {balance} coins")

        # Add coins nếu cần (dùng purchase endpoint)
        if balance < 100:
            print("⚠️  Balance < 100, adding coins via purchase...")
            purchase_payload = {"amount": 200, "package_id": "test_package"}
            response = await client.post("/coins/purchase", json=purchase_payload, headers=headers)
            if response.status_code == 200:
                new_balance_data = (await client.get("/coins/balance", headers=headers)).json()
                new_balance = new_balance_data.get("coins", 0)
                print(f"✅ Coins added! New balance: {new_balance} coins")
            else:
                print(f"⚠️  Purchase failed: {response.status_code} - {response.text}")
                print("   Continuing with current balance...")

        # 3. Create an AI match
        print("\n" + "=" * 60)
        print("Step 3: Create AI Match")
        print("=" * 60)

        create_match_data = {"level": 1, "board_size": 9}
        response = await client.post("/matches/ai", json=create_match_data, headers=headers)
        assert response.status_code == 201, f"Create match failed: {response.text}"
        match_id = response.json()["id"]
        print(f"✅ AI Match created: {match_id}")

        # 4. Make a few moves để có game state
        print("\n" + "=" * 60)
        print("Step 4: Make Moves")
        print("=" * 60)

        moves = [
            {"x": 3, "y": 3, "move_number": 1, "color": "B"},
            {"x": 3, "y": 4, "move_number": 2, "color": "B"},
            {"x": 4, "y": 3, "move_number": 3, "color": "B"},
        ]

        for move in moves:
            response = await client.post(f"/matches/{match_id}/move", json=move, headers=headers)
            if response.status_code == 200:
                print(f"✅ Move {move['move_number']} submitted: ({move['x']}, {move['y']})")
                result = response.json()
                if "ai_move" in result:
                    ai_move = result["ai_move"]
                    print(f"   AI responded: {ai_move}")
            else:
                print(f"⚠️  Move {move['move_number']} failed: {response.status_code} - {response.text}")

        # 5. Test Premium Hint
        print("\n" + "=" * 60)
        print("Step 5: Test Premium Hint")
        print("=" * 60)

        hint_payload = {"match_id": match_id, "top_k": 3}
        response = await client.post("/premium/hint", json=hint_payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Hint request successful!")
            print(f"   Request ID: {result.get('request_id')}")
            hints = result.get("hints", [])
            print(f"   Number of hints: {len(hints)}")
            for i, hint in enumerate(hints, 1):
                move = hint.get("move")
                confidence = hint.get("confidence", 0)
                print(f"   Hint {i}: Move={move}, Confidence={confidence:.2f}")
        elif response.status_code == 402:
            print(f"❌ Insufficient coins: {response.json().get('detail')}")
        else:
            print(f"❌ Hint request failed: {response.status_code} - {response.text}")

        # 6. Test Premium Analysis
        print("\n" + "=" * 60)
        print("Step 6: Test Premium Analysis")
        print("=" * 60)

        response = await client.post(f"/premium/analysis?match_id={match_id}", headers=headers, json={})

        if response.status_code == 202:
            result = response.json()
            print(f"✅ Analysis request accepted!")
            print(f"   Request ID: {result.get('request_id')}")
            print(f"   Status: {result.get('status')}")
        elif response.status_code == 200:
            # Nếu completed ngay
            result = response.json()
            print(f"✅ Analysis completed!")
            analysis = result.get("analysis", {})
            print(f"   Win probability: {analysis.get('win_probability', 0):.1%}")
            print(f"   Evaluation score: {analysis.get('evaluation_score', 0)}")
            print(f"   Game phase: {analysis.get('game_phase', 'unknown')}")
            print(f"   Recommendation: {analysis.get('recommendation', 'unknown')}")
        elif response.status_code == 402:
            print(f"❌ Insufficient coins: {response.json().get('detail')}")
        else:
            print(f"❌ Analysis request failed: {response.status_code} - {response.text}")

        # 7. Test Premium Review
        print("\n" + "=" * 60)
        print("Step 7: Test Premium Review")
        print("=" * 60)

        response = await client.post(f"/premium/review?match_id={match_id}", headers=headers, json={})

        if response.status_code == 202:
            result = response.json()
            print(f"✅ Review request accepted!")
            print(f"   Request ID: {result.get('request_id')}")
            print(f"   Status: {result.get('status')}")
        elif response.status_code == 200:
            # Nếu completed ngay
            result = response.json()
            print(f"✅ Review completed!")
            review = result.get("review", {})
            stats = review.get("statistics", {})
            print(f"   Total moves: {stats.get('total_moves', 0)}")
            print(f"   Mistakes found: {stats.get('mistakes_count', 0)}")
            print(f"   Key moments: {stats.get('key_moments_count', 0)}")
            mistakes = review.get("mistakes", [])
            if mistakes:
                print(f"   Top mistakes:")
                for mistake in mistakes[:3]:
                    print(f"     - Move {mistake.get('move_number')}: {mistake.get('severity')} ({mistake.get('eval_delta', 0):.1f})")
        elif response.status_code == 402:
            print(f"❌ Insufficient coins: {response.json().get('detail')}")
        else:
            print(f"❌ Review request failed: {response.status_code} - {response.text}")

        # 8. Test Cache Stats
        print("\n" + "=" * 60)
        print("Step 8: Test Cache Stats")
        print("=" * 60)

        response = await client.get("/premium/cache/stats", headers=headers)
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Cache stats retrieved!")
            print(f"   Cache size: {stats.get('size', 0)}/{stats.get('max_size', 0)}")
            print(f"   Hits: {stats.get('hits', 0)}")
            print(f"   Misses: {stats.get('misses', 0)}")
            print(f"   Hit rate: {stats.get('hit_rate', 0):.1%}")
        else:
            print(f"⚠️  Cache stats failed: {response.status_code} - {response.text}")

        # 9. Test Get Premium Request
        print("\n" + "=" * 60)
        print("Step 9: Get Premium Request (if available)")
        print("=" * 60)

        # Lấy request_id từ hint result nếu có
        if "hints" in locals() and result.get("request_id"):
            request_id = result["request_id"]
            response = await client.get(f"/premium/requests/{request_id}", headers=headers)
            if response.status_code == 200:
                request_data = response.json()
                print(f"✅ Request retrieved!")
                print(f"   Feature: {request_data.get('feature')}")
                print(f"   Summary: {request_data.get('summary')}")
                print(f"   Coins spent: {request_data.get('coins_spent', 0)}")
            else:
                print(f"⚠️  Get request failed: {response.status_code} - {response.text}")
        else:
            print("⚠️  No request ID available to test")

        print("\n" + "=" * 60)
        print("✅ Premium Features Test Completed!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_premium_features())

