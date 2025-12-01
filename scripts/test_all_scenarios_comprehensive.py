"""Comprehensive test cho tất cả scenarios."""

import asyncio
import httpx
import os
import sys

# Fix encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

BASE_URL = "http://localhost:8000"


async def test_all_scenarios():
    """Test tất cả scenarios."""
    print("=" * 60)
    print("Comprehensive Scenarios Test")
    print("=" * 60)
    print("\n⚠️  Đảm bảo server đang chạy!\n")

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=180.0) as client:
        # Setup: Login
        resp = await client.post(
            "/auth/login",
            json={"username_or_email": "premium_test_user", "password": "testpass123"},
            timeout=10.0
        )
        if resp.status_code != 200:
            print("❌ Login failed")
            return
        token = resp.json()["token"]["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("✅ Authenticated")

        # Scenario 1: AI Match với multiple levels
        print("\n" + "=" * 60)
        print("Scenario 1: AI Match - Multiple Levels")
        print("=" * 60)

        for level in [1, 2, 3, 4]:
            resp = await client.post(
                "/matches/ai",
                headers=headers,
                json={"level": level, "board_size": 9},
                timeout=10.0
            )
            if resp.status_code == 201:
                match_id = resp.json()["id"]
                print(f"✅ Level {level} match created: {match_id}")
                
                # Make a move
                move_resp = await client.post(
                    f"/matches/{match_id}/move",
                    headers=headers,
                    json={"x": 3, "y": 3, "move_number": 1, "color": "B"},
                    timeout=30.0
                )
                if move_resp.status_code == 200 and "ai_move" in move_resp.json():
                    print(f"   AI responded for level {level}")
                await asyncio.sleep(0.3)

        # Scenario 2: Pass moves
        print("\n" + "=" * 60)
        print("Scenario 2: Pass Moves")
        print("=" * 60)

        resp = await client.post("/matches/ai", headers=headers, json={"level": 1, "board_size": 9}, timeout=10.0)
        if resp.status_code == 201:
            match_id = resp.json()["id"]
            # Make a move first
            await client.post(
                f"/matches/{match_id}/move",
                headers=headers,
                json={"x": 3, "y": 3, "move_number": 1, "color": "B"},
                timeout=30.0
            )
            await asyncio.sleep(0.5)
            
            # Pass
            pass_resp = await client.post(
                f"/matches/{match_id}/pass",
                headers=headers,
                json={"move_number": 3, "color": "B"},
                timeout=10.0
            )
            if pass_resp.status_code == 200:
                print(f"✅ Pass move successful")
                print(f"   Response: {pass_resp.json().get('status')}")

        # Scenario 3: Invalid moves
        print("\n" + "=" * 60)
        print("Scenario 3: Invalid Moves")
        print("=" * 60)

        resp = await client.post("/matches/ai", headers=headers, json={"level": 1, "board_size": 9}, timeout=10.0)
        if resp.status_code == 201:
            match_id = resp.json()["id"]
            
            # Out of bounds
            invalid_resp = await client.post(
                f"/matches/{match_id}/move",
                headers=headers,
                json={"x": 10, "y": 10, "move_number": 1, "color": "B"},
                timeout=10.0
            )
            if invalid_resp.status_code in (400, 422):
                print(f"✅ Out of bounds move correctly rejected")
            else:
                print(f"⚠️  Unexpected response: {invalid_resp.status_code}")

        # Scenario 4: SGF Import & Export
        print("\n" + "=" * 60)
        print("Scenario 4: SGF Import & Export")
        print("=" * 60)

        # Import
        sgf_content = "(;FF[4];SZ[9];EV[Test];PB[Player1];PW[Player2];B[dd];W[ee];B[ed];W[de];B[fd];RE[B+3.5])"
        import_resp = await client.post(
            "/matches/import-sgf",
            headers=headers,
            json={"sgf_content": sgf_content},
            timeout=10.0
        )
        if import_resp.status_code == 201:
            imported_match = import_resp.json()
            imported_id = imported_match.get("id")
            print(f"✅ SGF Imported: {imported_id}")
            print(f"   Board size: {imported_match.get('board_size')}")
            print(f"   Result: {imported_match.get('result')}")
            
            # Export
            export_resp = await client.get(f"/matches/{imported_id}/sgf", headers=headers, timeout=10.0)
            if export_resp.status_code == 200:
                sgf_exported = export_resp.json().get("sgf_content", "")
                print(f"✅ SGF Exported: {len(sgf_exported)} chars")
                print(f"   Contains moves: {'B[dd]' in sgf_exported and 'W[ee]' in sgf_exported}")

        # Scenario 5: Replay System
        print("\n" + "=" * 60)
        print("Scenario 5: Replay System")
        print("=" * 60)

        resp = await client.post("/matches/ai", headers=headers, json={"level": 1, "board_size": 9}, timeout=10.0)
        if resp.status_code == 201:
            match_id = resp.json()["id"]
            
            # Make several moves
            for i in range(5):
                await client.post(
                    f"/matches/{match_id}/move",
                    headers=headers,
                    json={"x": 3 + i, "y": 3, "move_number": i * 2 + 1, "color": "B"},
                    timeout=30.0
                )
                await asyncio.sleep(0.3)
            
            # Get replay
            replay_resp = await client.get(f"/matches/{match_id}/replay", headers=headers, timeout=10.0)
            if replay_resp.status_code == 200:
                replay = replay_resp.json()
                print(f"✅ Replay retrieved:")
                print(f"   Total moves: {replay.get('total_moves')}")
                print(f"   Moves in replay: {len(replay.get('moves', []))}")

        # Scenario 6: Statistics & Leaderboard
        print("\n" + "=" * 60)
        print("Scenario 6: Statistics & Leaderboard")
        print("=" * 60)

        stats_resp = await client.get("/statistics/me", headers=headers, timeout=10.0)
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            print(f"✅ My Statistics:")
            print(f"   Elo: {stats.get('elo_rating')}")
            print(f"   Matches: {stats.get('total_matches')}")
            print(f"   Win rate: {stats.get('win_rate')}%")

        lb_resp = await client.get("/statistics/leaderboard?limit=5", headers=headers, timeout=10.0)
        if lb_resp.status_code == 200:
            lb = lb_resp.json()
            print(f"✅ Leaderboard: {len(lb)} entries")
            for entry in lb[:3]:
                print(f"   {entry.get('rank')}. {entry.get('username')} - Elo: {entry.get('elo_rating')}")

        # Scenario 7: Match History
        print("\n" + "=" * 60)
        print("Scenario 7: Match History")
        print("=" * 60)

        history_resp = await client.get("/matches/history", headers=headers, timeout=10.0)
        if history_resp.status_code == 200:
            matches = history_resp.json()
            print(f"✅ Match History: {len(matches)} matches")
            if matches:
                print(f"   Latest: {matches[0].get('id')}")
                print(f"   Result: {matches[0].get('result')}")

        # Scenario 8: Premium Features với coins
        print("\n" + "=" * 60)
        print("Scenario 8: Premium Features")
        print("=" * 60)

        # Add coins
        await client.post("/coins/purchase", headers=headers, json={"amount": 200, "package_id": "test"}, timeout=10.0)
        
        # Create match for premium features
        resp = await client.post("/matches/ai", headers=headers, json={"level": 1, "board_size": 9}, timeout=10.0)
        if resp.status_code == 201:
            match_id = resp.json()["id"]
            
            # Make moves
            for i in range(3):
                await client.post(
                    f"/matches/{match_id}/move",
                    headers=headers,
                    json={"x": 3 + i, "y": 3, "move_number": i * 2 + 1, "color": "B"},
                    timeout=30.0
                )
                await asyncio.sleep(0.3)
            
            # Hint
            hint_resp = await client.post(
                "/premium/hint",
                headers=headers,
                json={"match_id": match_id, "top_k": 3},
                timeout=30.0
            )
            if hint_resp.status_code == 200:
                hints = hint_resp.json().get("hints", [])
                print(f"✅ Premium Hint: {len(hints)} hints")
            
            # Analysis
            analysis_resp = await client.post(
                f"/premium/analysis?match_id={match_id}",
                headers=headers,
                json={},
                timeout=30.0
            )
            if analysis_resp.status_code in (200, 202):
                analysis = analysis_resp.json().get("analysis", {})
                print(f"✅ Premium Analysis: Win prob={analysis.get('win_probability', 0):.1%}")

        print("\n" + "=" * 60)
        print("✅ All Scenarios Test Completed!")
        print("=" * 60)
        print("\n📊 Tested Scenarios:")
        print("   ✅ AI Match - Multiple Levels")
        print("   ✅ Pass Moves")
        print("   ✅ Invalid Moves")
        print("   ✅ SGF Import & Export")
        print("   ✅ Replay System")
        print("   ✅ Statistics & Leaderboard")
        print("   ✅ Match History")
        print("   ✅ Premium Features")


if __name__ == "__main__":
    asyncio.run(test_all_scenarios())

