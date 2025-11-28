"""Final comprehensive test - tất cả scenarios quan trọng."""

import asyncio
import httpx
import os
import sys

# Fix encoding
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

BASE_URL = "http://localhost:8000"


async def test_scenarios():
    """Test tất cả scenarios quan trọng."""
    print("=" * 60)
    print("Final Comprehensive Scenarios Test")
    print("=" * 60)
    print("\n⚠️  Đảm bảo server đang chạy!\n")

    results = {"passed": 0, "failed": 0}

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=180.0) as client:
        # Auth
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
        print("✅ Authenticated\n")

        # Test 1: AI Levels 1-2 (nhanh hơn)
        print("=" * 60)
        print("Test 1: AI Match Levels 1-2")
        print("=" * 60)
        for level in [1, 2]:
            try:
                resp = await client.post(
                    "/matches/ai",
                    headers=headers,
                    json={"level": level, "board_size": 9},
                    timeout=10.0
                )
                if resp.status_code == 201:
                    match_id = resp.json()["id"]
                    move_resp = await client.post(
                        f"/matches/{match_id}/move",
                        headers=headers,
                        json={"x": 3, "y": 3, "move_number": 1, "color": "B"},
                        timeout=60.0  # Longer timeout for AI
                    )
                    if move_resp.status_code == 200:
                        if "ai_move" in move_resp.json():
                            print(f"✅ Level {level}: AI responded")
                            results["passed"] += 1
                        else:
                            print(f"⚠️  Level {level}: No AI move in response")
                    else:
                        print(f"❌ Level {level}: Move failed")
                        results["failed"] += 1
            except Exception as e:
                print(f"❌ Level {level}: {str(e)[:100]}")
                results["failed"] += 1

        # Test 2: Pass Move
        print("\n" + "=" * 60)
        print("Test 2: Pass Move")
        print("=" * 60)
        try:
            resp = await client.post("/matches/ai", headers=headers, json={"level": 1, "board_size": 9}, timeout=10.0)
            if resp.status_code == 201:
                match_id = resp.json()["id"]
                await client.post(
                    f"/matches/{match_id}/move",
                    headers=headers,
                    json={"x": 3, "y": 3, "move_number": 1, "color": "B"},
                    timeout=60.0
                )
                await asyncio.sleep(0.5)
                
                pass_resp = await client.post(
                    f"/matches/{match_id}/pass",
                    headers=headers,
                    json={"move_number": 3, "color": "B"},
                    timeout=10.0
                )
                if pass_resp.status_code == 200:
                    print("✅ Pass move successful")
                    results["passed"] += 1
                else:
                    print(f"❌ Pass failed: {pass_resp.status_code}")
                    results["failed"] += 1
        except Exception as e:
            print(f"❌ Pass test failed: {str(e)[:100]}")
            results["failed"] += 1

        # Test 3: Invalid Move
        print("\n" + "=" * 60)
        print("Test 3: Invalid Move Validation")
        print("=" * 60)
        try:
            resp = await client.post("/matches/ai", headers=headers, json={"level": 1, "board_size": 9}, timeout=10.0)
            if resp.status_code == 201:
                match_id = resp.json()["id"]
                invalid_resp = await client.post(
                    f"/matches/{match_id}/move",
                    headers=headers,
                    json={"x": 10, "y": 10, "move_number": 1, "color": "B"},
                    timeout=10.0
                )
                if invalid_resp.status_code in (400, 422):
                    print("✅ Invalid move correctly rejected")
                    results["passed"] += 1
                else:
                    print(f"⚠️  Unexpected: {invalid_resp.status_code}")
                    results["failed"] += 1
        except Exception as e:
            print(f"❌ Invalid move test failed: {str(e)[:100]}")
            results["failed"] += 1

        # Test 4: SGF Import/Export
        print("\n" + "=" * 60)
        print("Test 4: SGF Import & Export")
        print("=" * 60)
        try:
            sgf = "(;FF[4];SZ[9];EV[Test];PB[P1];PW[P2];B[dd];W[ee];B[ed];RE[B+2.5])"
            import_resp = await client.post(
                "/matches/import-sgf",
                headers=headers,
                json={"sgf_content": sgf},
                timeout=10.0
            )
            if import_resp.status_code == 201:
                match_id = import_resp.json().get("id")
                print(f"✅ SGF Imported: {match_id}")
                
                export_resp = await client.get(f"/matches/{match_id}/sgf", headers=headers, timeout=10.0)
                if export_resp.status_code == 200:
                    exported = export_resp.json().get("sgf_content", "")
                    print(f"✅ SGF Exported: {len(exported)} chars")
                    results["passed"] += 2
                else:
                    print(f"❌ Export failed: {export_resp.status_code}")
                    results["failed"] += 1
            else:
                print(f"❌ Import failed: {import_resp.status_code}")
                results["failed"] += 1
        except Exception as e:
            print(f"❌ SGF test failed: {str(e)[:100]}")
            results["failed"] += 1

        # Test 5: Replay
        print("\n" + "=" * 60)
        print("Test 5: Replay System")
        print("=" * 60)
        try:
            resp = await client.post("/matches/ai", headers=headers, json={"level": 1, "board_size": 9}, timeout=10.0)
            if resp.status_code == 201:
                match_id = resp.json()["id"]
                for i in range(3):
                    await client.post(
                        f"/matches/{match_id}/move",
                        headers=headers,
                        json={"x": 3 + i, "y": 3, "move_number": i * 2 + 1, "color": "B"},
                        timeout=60.0
                    )
                    await asyncio.sleep(0.3)
                
                replay_resp = await client.get(f"/matches/{match_id}/replay", headers=headers, timeout=10.0)
                if replay_resp.status_code == 200:
                    replay = replay_resp.json()
                    print(f"✅ Replay: {replay.get('total_moves')} moves")
                    results["passed"] += 1
                else:
                    print(f"❌ Replay failed: {replay_resp.status_code}")
                    results["failed"] += 1
        except Exception as e:
            print(f"❌ Replay test failed: {str(e)[:100]}")
            results["failed"] += 1

        # Test 6: Statistics
        print("\n" + "=" * 60)
        print("Test 6: Statistics & Leaderboard")
        print("=" * 60)
        try:
            stats_resp = await client.get("/statistics/me", headers=headers, timeout=10.0)
            if stats_resp.status_code == 200:
                stats = stats_resp.json()
                print(f"✅ Statistics: Elo={stats.get('elo_rating')}, Matches={stats.get('total_matches')}")
                results["passed"] += 1
            
            lb_resp = await client.get("/statistics/leaderboard?limit=5", headers=headers, timeout=10.0)
            if lb_resp.status_code == 200:
                lb = lb_resp.json()
                print(f"✅ Leaderboard: {len(lb)} entries")
                results["passed"] += 1
        except Exception as e:
            print(f"❌ Statistics test failed: {str(e)[:100]}")
            results["failed"] += 1

        # Test 7: PvP & Elo (quick)
        print("\n" + "=" * 60)
        print("Test 7: PvP Match & Elo Update")
        print("=" * 60)
        try:
            # Create 2 users
            user1_data = {"username": "test_pvp1", "email": "tpvp1@test.com", "password": "testpass123"}
            user2_data = {"username": "test_pvp2", "email": "tpvp2@test.com", "password": "testpass123"}
            
            # Register/login user1
            resp1 = await client.post("/auth/register", json=user1_data, timeout=10.0)
            if resp1.status_code not in (201, 409):
                resp1 = await client.post("/auth/login", json={"username_or_email": user1_data["username"], "password": user1_data["password"]}, timeout=10.0)
            token1 = resp1.json()["token"]["access_token"] if resp1.status_code in (200, 201) else None
            
            # Register/login user2
            resp2 = await client.post("/auth/register", json=user2_data, timeout=10.0)
            if resp2.status_code not in (201, 409):
                resp2 = await client.post("/auth/login", json={"username_or_email": user2_data["username"], "password": user2_data["password"]}, timeout=10.0)
            token2 = resp2.json()["token"]["access_token"] if resp2.status_code in (200, 201) else None
            
            if token1 and token2:
                headers1 = {"Authorization": f"Bearer {token1}"}
                headers2 = {"Authorization": f"Bearer {token2}"}
                
                # Get initial Elo
                stats1 = (await client.get("/statistics/me", headers=headers1, timeout=10.0)).json()
                stats2 = (await client.get("/statistics/me", headers=headers2, timeout=10.0)).json()
                elo1_initial = stats1.get("elo_rating", 1500)
                elo2_initial = stats2.get("elo_rating", 1500)
                
                # Create PvP match
                pvp_resp = await client.post("/matches/pvp", headers=headers1, json={"board_size": 9}, timeout=10.0)
                if pvp_resp.status_code == 201:
                    match_id = pvp_resp.json().get("match", {}).get("id") or pvp_resp.json().get("id")
                    
                    # Join
                    await client.post(f"/matches/pvp/{match_id}/join", headers=headers2, timeout=10.0)
                    
                    # Make a few moves
                    await client.post(f"/matches/{match_id}/move", headers=headers1, json={"x": 3, "y": 3, "move_number": 1, "color": "B"}, timeout=10.0)
                    await asyncio.sleep(0.2)
                    await client.post(f"/matches/{match_id}/move", headers=headers2, json={"x": 3, "y": 4, "move_number": 2, "color": "W"}, timeout=10.0)
                    await asyncio.sleep(0.2)
                    
                    # Resign
                    await client.post(f"/matches/{match_id}/resign", headers=headers1, timeout=10.0)
                    await asyncio.sleep(1)
                    
                    # Check Elo
                    stats1_new = (await client.get("/statistics/me", headers=headers1, timeout=10.0)).json()
                    stats2_new = (await client.get("/statistics/me", headers=headers2, timeout=10.0)).json()
                    elo1_new = stats1_new.get("elo_rating", 1500)
                    elo2_new = stats2_new.get("elo_rating", 1500)
                    
                    if elo1_new != elo1_initial or elo2_new != elo2_initial:
                        print(f"✅ Elo updated:")
                        print(f"   Player 1: {elo1_initial} -> {elo1_new} ({elo1_new - elo1_initial:+d})")
                        print(f"   Player 2: {elo2_initial} -> {elo2_new} ({elo2_new - elo2_initial:+d})")
                        results["passed"] += 1
                    else:
                        print("⚠️  Elo not updated")
                        results["failed"] += 1
        except Exception as e:
            print(f"❌ PvP test failed: {str(e)[:100]}")
            results["failed"] += 1

        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"✅ Passed: {results['passed']}")
        print(f"❌ Failed: {results['failed']}")
        print(f"📊 Success rate: {results['passed'] / (results['passed'] + results['failed']) * 100:.1f}%")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_scenarios())

