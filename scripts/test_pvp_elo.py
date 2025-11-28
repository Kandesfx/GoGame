"""Test PvP matches và Elo rating updates."""

import asyncio
import httpx
import os
import sys
from pathlib import Path

# Fix encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

BASE_URL = "http://localhost:8000"


async def test_pvp_elo():
    """Test PvP matches và Elo rating system."""
    print("=" * 60)
    print("Testing PvP Matches & Elo Rating")
    print("=" * 60)
    print("\n⚠️  Đảm bảo server đang chạy: uvicorn app.main:app --reload\n")

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120.0) as client:
        # 1. Create 2 users
        print("=" * 60)
        print("Step 1: Create Test Users")
        print("=" * 60)

        users = []
        for i, user_data in enumerate([
            {"username": "pvp_player1", "email": "pvp1@test.com", "password": "testpass123"},
            {"username": "pvp_player2", "email": "pvp2@test.com", "password": "testpass123"},
        ]):
            # Try register
            resp = await client.post("/auth/register", json=user_data, timeout=10.0)
            if resp.status_code == 201:
                print(f"✅ User {i+1} registered: {user_data['username']}")
                token = resp.json()["token"]["access_token"]
            elif resp.status_code in (400, 409, 422):
                # User exists or validation error, try login
                print(f"⚠️  User {i+1} exists or validation error, trying login...")
                resp = await client.post(
                    "/auth/login",
                    json={"username_or_email": user_data["username"], "password": user_data["password"]},
                    timeout=10.0
                )
                if resp.status_code == 200:
                    print(f"✅ User {i+1} logged in: {user_data['username']}")
                    token = resp.json()["token"]["access_token"]
                else:
                    print(f"❌ Failed to login user {i+1}: {resp.status_code} - {resp.text[:200]}")
                    return
            else:
                print(f"❌ Failed to create user {i+1}: {resp.status_code} - {resp.text[:200]}")
                return

            headers = {"Authorization": f"Bearer {token}"}
            users.append({"username": user_data["username"], "headers": headers, "token": token})

        # 2. Get initial Elo ratings
        print("\n" + "=" * 60)
        print("Step 2: Initial Elo Ratings")
        print("=" * 60)

        initial_elos = {}
        for i, user in enumerate(users, 1):
            resp = await client.get("/statistics/me", headers=user["headers"], timeout=10.0)
            if resp.status_code == 200:
                stats = resp.json()
                elo = stats.get("elo_rating", 1500)
                initial_elos[user["username"]] = elo
                print(f"   User {i} ({user['username']}): Elo = {elo}")

        # 3. Create PvP match
        print("\n" + "=" * 60)
        print("Step 3: Create PvP Match")
        print("=" * 60)

        # Player 1 creates match
        resp = await client.post(
            "/matches/pvp",
            headers=users[0]["headers"],
            json={"board_size": 9},
            timeout=10.0
        )
        if resp.status_code != 201:
            print(f"❌ Create PvP match failed: {resp.status_code} - {resp.text[:200]}")
            return

        match_data = resp.json()
        match_id = match_data.get("match", {}).get("id") or match_data.get("id")
        join_code = match_data.get("join_code", "")
        print(f"✅ PvP Match created: {match_id}")
        print(f"   Join code: {join_code}")

        # Player 2 joins match
        resp = await client.post(
            f"/matches/pvp/{match_id}/join",
            headers=users[1]["headers"],
            timeout=10.0
        )
        if resp.status_code == 200:
            print(f"✅ Player 2 joined match")
        else:
            print(f"⚠️  Join failed (might be auto-joined): {resp.status_code}")

        # 4. Play match
        print("\n" + "=" * 60)
        print("Step 4: Play Match")
        print("=" * 60)

        # Make some moves
        moves_sequence = [
            (0, users[0], 3, 3, "B", 1),  # Player 1
            (1, users[1], 3, 4, "W", 2),  # Player 2
            (0, users[0], 4, 3, "B", 3),  # Player 1
            (1, users[1], 4, 4, "W", 4),  # Player 2
            (0, users[0], 5, 3, "B", 5),  # Player 1
        ]

        for player_idx, user, x, y, color, move_num in moves_sequence:
            resp = await client.post(
                f"/matches/{match_id}/move",
                headers=user["headers"],
                json={"x": x, "y": y, "move_number": move_num, "color": color},
                timeout=10.0
            )
            if resp.status_code == 200:
                print(f"✅ Move {move_num}: Player {player_idx+1} ({color}) at ({x}, {y})")
            else:
                print(f"⚠️  Move {move_num} failed: {resp.status_code}")
            await asyncio.sleep(0.2)

        # 5. Player 1 resigns (Player 2 wins)
        print("\n" + "=" * 60)
        print("Step 5: Player 1 Resigns (Player 2 Wins)")
        print("=" * 60)

        resp = await client.post(
            f"/matches/{match_id}/resign",
            headers=users[0]["headers"],
            timeout=10.0
        )
        if resp.status_code == 200:
            result = resp.json()
            print(f"✅ Resign successful")
            print(f"   Result: {result.get('result')}")
        else:
            print(f"❌ Resign failed: {resp.status_code} - {resp.text[:200]}")

        # Wait a bit for Elo update
        await asyncio.sleep(1)

        # 6. Check Elo changes
        print("\n" + "=" * 60)
        print("Step 6: Check Elo Rating Changes")
        print("=" * 60)

        for i, user in enumerate(users, 1):
            resp = await client.get("/statistics/me", headers=user["headers"], timeout=10.0)
            if resp.status_code == 200:
                stats = resp.json()
                new_elo = stats.get("elo_rating", 1500)
                old_elo = initial_elos.get(user["username"], 1500)
                change = new_elo - old_elo
                print(f"   User {i} ({user['username']}):")
                print(f"      Old Elo: {old_elo}")
                print(f"      New Elo: {new_elo}")
                print(f"      Change: {change:+d}")

        # 7. Check match history
        print("\n" + "=" * 60)
        print("Step 7: Check Match History")
        print("=" * 60)

        for i, user in enumerate(users, 1):
            resp = await client.get("/matches/history", headers=user["headers"], timeout=10.0)
            if resp.status_code == 200:
                matches = resp.json()
                print(f"   User {i} ({user['username']}): {len(matches)} matches in history")
                if matches:
                    latest = matches[0]
                    print(f"      Latest: {latest.get('id')}, Result: {latest.get('result')}")

        # 8. Check updated statistics
        print("\n" + "=" * 60)
        print("Step 8: Updated Statistics")
        print("=" * 60)

        for i, user in enumerate(users, 1):
            resp = await client.get("/statistics/me", headers=user["headers"], timeout=10.0)
            if resp.status_code == 200:
                stats = resp.json()
                print(f"   User {i} ({user['username']}):")
                print(f"      Total matches: {stats.get('total_matches')}")
                print(f"      Wins: {stats.get('wins')}, Losses: {stats.get('losses')}")
                print(f"      Win rate: {stats.get('win_rate')}%")

        # 9. Check leaderboard
        print("\n" + "=" * 60)
        print("Step 9: Leaderboard")
        print("=" * 60)

        resp = await client.get("/statistics/leaderboard?limit=10", headers=users[0]["headers"], timeout=10.0)
        if resp.status_code == 200:
            leaderboard = resp.json()
            print(f"✅ Leaderboard: {len(leaderboard)} entries")
            for entry in leaderboard[:5]:
                print(f"   Rank {entry.get('rank')}: {entry.get('username')} - Elo: {entry.get('elo_rating')}")

        print("\n" + "=" * 60)
        print("✅ PvP & Elo Test Completed!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_pvp_elo())

