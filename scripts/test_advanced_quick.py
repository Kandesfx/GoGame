"""Quick test cho Advanced Features - từng endpoint một."""

import asyncio
import httpx
import os
import sys

# Fix encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

BASE_URL = "http://localhost:8000"


async def quick_test():
    """Quick test từng endpoint."""
    print("=" * 60)
    print("Quick Test - Advanced Features")
    print("=" * 60)
    print("\n⚠️  Đảm bảo server đang chạy!")
    print("   cd backend")
    print("   uvicorn app.main:app --reload\n")

    try:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
            # 1. Health check
            print("1. Health check...")
            try:
                resp = await client.get("/health", timeout=5.0)
                if resp.status_code == 200:
                    print("   ✅ Server is running")
                else:
                    print(f"   ⚠️  Server responded with {resp.status_code}")
            except Exception as e:
                print(f"   ❌ Server not running: {e}")
                print("\n💡 Start server:")
                print("   cd backend")
                print("   uvicorn app.main:app --reload")
                return

            # 2. Login
            print("\n2. Login...")
            resp = await client.post(
                "/auth/login",
                json={"username_or_email": "premium_test_user", "password": "testpass123"},
                timeout=10.0
            )
            if resp.status_code != 200:
                print(f"   ❌ Login failed: {resp.status_code}")
                print("   💡 User might not exist. Try registering first.")
                return
            token = resp.json()["token"]["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            print("   ✅ Logged in")

            # 3. Statistics
            print("\n3. Test Statistics...")
            resp = await client.get("/statistics/me", headers=headers, timeout=10.0)
            if resp.status_code == 200:
                stats = resp.json()
                print(f"   ✅ Statistics: Elo={stats.get('elo_rating')}, Matches={stats.get('total_matches')}")
            else:
                print(f"   ❌ Failed: {resp.status_code} - {resp.text[:100]}")

            # 4. Leaderboard
            print("\n4. Test Leaderboard...")
            resp = await client.get("/statistics/leaderboard?limit=5", headers=headers, timeout=10.0)
            if resp.status_code == 200:
                lb = resp.json()
                print(f"   ✅ Leaderboard: {len(lb)} entries")
            else:
                print(f"   ❌ Failed: {resp.status_code} - {resp.text[:100]}")

            # 5. SGF Import
            print("\n5. Test SGF Import...")
            sgf = "(;FF[4];SZ[9];EV[Test];B[dd];W[ee];B[ed];RE[B+2.5])"
            resp = await client.post(
                "/matches/import-sgf",
                headers=headers,
                json={"sgf_content": sgf},
                timeout=10.0
            )
            if resp.status_code == 201:
                match = resp.json()
                print(f"   ✅ SGF Imported: {match.get('id')}")
                imported_id = match.get("id")
            else:
                print(f"   ❌ Failed: {resp.status_code} - {resp.text[:100]}")
                imported_id = None

            # 6. Replay
            print("\n6. Test Replay...")
            # Create a match first
            resp = await client.post(
                "/matches/ai",
                headers=headers,
                json={"level": 1, "board_size": 9},
                timeout=10.0
            )
            if resp.status_code == 201:
                match_id = resp.json()["id"]
                # Make a move
                await client.post(
                    f"/matches/{match_id}/move",
                    headers=headers,
                    json={"x": 3, "y": 3, "move_number": 1, "color": "B"},
                    timeout=10.0
                )
                await asyncio.sleep(0.3)
                
                # Get replay
                resp = await client.get(f"/matches/{match_id}/replay", headers=headers, timeout=10.0)
                if resp.status_code == 200:
                    replay = resp.json()
                    print(f"   ✅ Replay: {replay.get('total_moves')} moves")
                else:
                    print(f"   ❌ Failed: {resp.status_code} - {resp.text[:100]}")

            print("\n" + "=" * 60)
            print("✅ Quick Test Completed!")
            print("=" * 60)

    except httpx.ConnectError:
        print("\n❌ Cannot connect to server")
        print("💡 Start server:")
        print("   cd backend")
        print("   uvicorn app.main:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(quick_test())

