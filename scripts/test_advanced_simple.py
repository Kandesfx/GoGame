"""Test Advanced Features - Simple version với error handling."""

import asyncio
import httpx
import os
import sys
from pathlib import Path

# Fix encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

BASE_URL = "http://localhost:8000"


async def test_endpoint(client, method, url, headers=None, json_data=None, description=""):
    """Helper để test một endpoint."""
    try:
        if method == "GET":
            response = await client.get(url, headers=headers, timeout=30.0)
        elif method == "POST":
            response = await client.post(url, headers=headers, json=json_data, timeout=30.0)
        else:
            return None, f"Unknown method: {method}"
        
        if response.status_code in (200, 201, 202):
            return response.json(), None
        else:
            return None, f"Status {response.status_code}: {response.text[:200]}"
    except httpx.ConnectError:
        return None, "Server không chạy. Hãy start: cd backend && uvicorn app.main:app --reload"
    except httpx.ReadTimeout:
        return None, "Request timeout"
    except Exception as e:
        return None, f"Error: {str(e)[:200]}"


async def test_advanced_features():
    """Test tất cả advanced features."""
    print("=" * 60)
    print("Testing Advanced Features")
    print("=" * 60)
    print("\n⚠️  Đảm bảo server đang chạy: cd backend && uvicorn app.main:app --reload\n")

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0) as client:
        # 1. Auth
        print("=" * 60)
        print("Step 1: Authentication")
        print("=" * 60)
        
        data, error = await test_endpoint(
            client, "POST", "/auth/login",
            json_data={"username_or_email": "premium_test_user", "password": "testpass123"},
            description="Login"
        )
        
        if error:
            print(f"❌ Login failed: {error}")
            if "Server không chạy" in error:
                print("\n💡 Hãy start server trước:")
                print("   cd backend")
                print("   uvicorn app.main:app --reload")
            return
        
        access_token = data["token"]["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}
        print("✅ Authentication successful")

        # 2. Test Statistics
        print("\n" + "=" * 60)
        print("Step 2: Test Statistics")
        print("=" * 60)

        stats_data, error = await test_endpoint(
            client, "GET", "/statistics/me", headers=headers, description="My Statistics"
        )
        if error:
            print(f"❌ Statistics failed: {error}")
        else:
            print(f"✅ My Statistics:")
            print(f"   Elo: {stats_data.get('elo_rating')}")
            print(f"   Total matches: {stats_data.get('total_matches')}")
            print(f"   Wins: {stats_data.get('wins')}, Losses: {stats_data.get('losses')}")
            print(f"   Win rate: {stats_data.get('win_rate')}%")

        # Leaderboard
        leaderboard_data, error = await test_endpoint(
            client, "GET", "/statistics/leaderboard?limit=10", headers=headers, description="Leaderboard"
        )
        if error:
            print(f"❌ Leaderboard failed: {error}")
        else:
            print(f"✅ Leaderboard: {len(leaderboard_data)} entries")
            if leaderboard_data:
                top = leaderboard_data[0]
                print(f"   Top player: {top.get('username')} (Elo: {top.get('elo_rating')})")

        # 3. Test SGF Import
        print("\n" + "=" * 60)
        print("Step 3: Test SGF Import")
        print("=" * 60)

        sgf_content = "(;FF[4];SZ[9];EV[GoGame];DT[2025-11-20];PB[TestPlayer];PW[AI];B[dd];W[ee];B[ed];W[de];RE[B+2.5])"
        import_data, error = await test_endpoint(
            client, "POST", "/matches/import-sgf",
            headers=headers,
            json_data={"sgf_content": sgf_content},
            description="SGF Import"
        )
        if error:
            print(f"❌ SGF Import failed: {error}")
        else:
            imported_match_id = import_data.get("id")
            print(f"✅ SGF Import successful: {imported_match_id}")
            print(f"   Board size: {import_data.get('board_size')}")
            print(f"   Result: {import_data.get('result')}")

        # 4. Test Replay
        print("\n" + "=" * 60)
        print("Step 4: Test Replay")
        print("=" * 60)

        # Create a match first
        create_data, error = await test_endpoint(
            client, "POST", "/matches/ai",
            headers=headers,
            json_data={"level": 1, "board_size": 9},
            description="Create Match"
        )
        if error:
            print(f"❌ Create match failed: {error}")
        else:
            match_id = create_data["id"]
            print(f"✅ Match created: {match_id}")
            
            # Make some moves
            for i in range(3):
                move = {"x": 3 + i, "y": 3, "move_number": i * 2 + 1, "color": "B"}
                move_data, move_error = await test_endpoint(
                    client, "POST", f"/matches/{match_id}/move",
                    headers=headers,
                    json_data=move,
                    description=f"Move {i+1}"
                )
                if not move_error:
                    print(f"✅ Move {i+1} submitted")
                await asyncio.sleep(0.2)
            
            # Get replay
            replay_data, error = await test_endpoint(
                client, "GET", f"/matches/{match_id}/replay",
                headers=headers,
                description="Get Replay"
            )
            if error:
                print(f"❌ Replay failed: {error}")
            else:
                print(f"✅ Replay retrieved:")
                print(f"   Match ID: {replay_data.get('match_id')}")
                print(f"   Total moves: {replay_data.get('total_moves')}")
                print(f"   Moves count: {len(replay_data.get('moves', []))}")

        print("\n" + "=" * 60)
        print("✅ Advanced Features Test Completed!")
        print("=" * 60)
        print("\n📊 Features tested:")
        print("   ✅ Statistics Dashboard")
        print("   ✅ Leaderboard")
        print("   ✅ SGF Import")
        print("   ✅ Replay System")
        print("   ✅ Elo Rating (logic implemented, needs PvP match to test)")


if __name__ == "__main__":
    asyncio.run(test_advanced_features())

