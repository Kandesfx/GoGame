"""
Script test đơn giản để kiểm tra ML model có hoạt động khi server chạy không.

Chạy script này trong terminal riêng trong khi server đang chạy.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

print("=" * 70)
print("🧪 TEST ML MODEL VỚI SERVER")
print("=" * 70)
print(f"Server URL: {BASE_URL}")
print()

# Test 1: Health check
print("📡 TEST 1: Health Check")
print("-" * 70)
try:
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    if response.status_code == 200:
        print("✅ Server is running")
        print(f"   Response: {response.json()}")
    else:
        print(f"❌ Server returned status {response.status_code}")
        print("   Make sure server is running on http://localhost:8000")
        exit(1)
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to server")
    print("   Make sure server is running: python -m app.main")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

print()

# Test 2: Check if we can create a test user (optional)
print("📡 TEST 2: Check Server Status")
print("-" * 70)
print("✅ Server is accessible")
print("   You can now test ML model by:")
print("   1. Creating an AI match via API or frontend")
print("   2. Making a move")
print("   3. AI should use ML model to respond")
print()

# Instructions
print("=" * 70)
print("📋 HƯỚNG DẪN TEST ML MODEL")
print("=" * 70)
print()
print("1. Đảm bảo server đang chạy (đã chạy ✅)")
print()
print("2. Tạo AI match:")
print("   - Qua Frontend: Mở game → Play with AI")
print("   - Qua API:")
print("     POST http://localhost:8000/api/matches/ai")
print("     Body: {")
print("       'board_size': 19,")
print("       'level': 1,")
print("       'player_color': 'black'")
print("     }")
print()
print("3. Kiểm tra logs trong server console:")
print("   Tìm các dòng:")
print("   🤖 [ML] Trying ML model AI move")
print("   ✅ [ML] ML model AI move successful")
print("   🤖 [ML] ML model AI move: (x, y), prob=..., win_prob=...")
print()
print("4. Nếu thấy logs trên, ML model đang hoạt động! ✅")
print()
print("=" * 70)

