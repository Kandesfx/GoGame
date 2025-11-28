"""Backend API client."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx
from dotenv import load_dotenv

load_dotenv()


class APIClient:
    """Client để gọi backend API."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("BACKEND_URL", "http://localhost:8000")
        self.token: Optional[str] = None
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

    def set_token(self, token: str) -> None:
        """Set authentication token."""
        self.token = token
        self._client.headers.update({"Authorization": f"Bearer {token}"})

    def clear_token(self) -> None:
        """Clear authentication token."""
        self.token = None
        if "Authorization" in self._client.headers:
            del self._client.headers["Authorization"]

    @property
    def headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    # Authentication
    async def register(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """Register new user."""
        response = await self._client.post(
            "/auth/register",
            json={"username": username, "email": email, "password": password},
        )
        response.raise_for_status()
        data = response.json()
        if "token" in data:
            self.set_token(data["token"]["access_token"])
        return data

    async def login(self, username_or_email: str, password: str) -> Dict[str, Any]:
        """Login user."""
        response = await self._client.post(
            "/auth/login",
            json={"username_or_email": username_or_email, "password": password},
        )
        response.raise_for_status()
        data = response.json()
        if "token" in data:
            self.set_token(data["token"]["access_token"])
        return data

    # Matches
    async def create_ai_match(self, level: int, board_size: int = 9) -> Dict[str, Any]:
        """Create AI match."""
        response = await self._client.post(
            "/matches/ai",
            json={"level": level, "board_size": board_size},
        )
        response.raise_for_status()
        return response.json()

    async def create_pvp_match(self, board_size: int = 9) -> Dict[str, Any]:
        """Create PvP match."""
        response = await self._client.post(
            "/matches/pvp",
            json={"board_size": board_size},
        )
        response.raise_for_status()
        return response.json()

    async def join_pvp_match(self, match_id: UUID) -> Dict[str, Any]:
        """Join PvP match."""
        response = await self._client.post(f"/matches/pvp/{match_id}/join")
        response.raise_for_status()
        return response.json()

    async def get_match(self, match_id: UUID) -> Dict[str, Any]:
        """Get match details."""
        response = await self._client.get(f"/matches/{match_id}")
        response.raise_for_status()
        return response.json()

    async def submit_move(self, match_id: UUID, x: int, y: int, move_number: int, color: str) -> Dict[str, Any]:
        """Submit move."""
        response = await self._client.post(
            f"/matches/{match_id}/move",
            json={"x": x, "y": y, "move_number": move_number, "color": color},
        )
        response.raise_for_status()
        return response.json()

    async def pass_turn(self, match_id: UUID, move_number: int, color: str) -> Dict[str, Any]:
        """Pass turn."""
        response = await self._client.post(
            f"/matches/{match_id}/pass",
            json={"move_number": move_number, "color": color},
        )
        response.raise_for_status()
        return response.json()

    async def resign_match(self, match_id: UUID) -> Dict[str, Any]:
        """Resign match."""
        response = await self._client.post(f"/matches/{match_id}/resign")
        response.raise_for_status()
        return response.json()

    async def get_match_history(self) -> List[Dict[str, Any]]:
        """Get match history."""
        response = await self._client.get("/matches/history")
        response.raise_for_status()
        return response.json()

    async def get_replay(self, match_id: UUID) -> Dict[str, Any]:
        """Get replay data."""
        response = await self._client.get(f"/matches/{match_id}/replay")
        response.raise_for_status()
        return response.json()

    # Statistics
    async def get_my_statistics(self) -> Dict[str, Any]:
        """Get my statistics."""
        response = await self._client.get("/statistics/me")
        response.raise_for_status()
        return response.json()

    async def get_leaderboard(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get leaderboard."""
        response = await self._client.get(f"/statistics/leaderboard?limit={limit}")
        response.raise_for_status()
        return response.json()

    # Premium Features
    async def request_hint(self, match_id: UUID, top_k: int = 3) -> Dict[str, Any]:
        """Request hint."""
        response = await self._client.post(
            "/premium/hint",
            json={"match_id": match_id, "top_k": top_k},
        )
        response.raise_for_status()
        return response.json()

    async def request_analysis(self, match_id: UUID) -> Dict[str, Any]:
        """Request analysis."""
        response = await self._client.post(f"/premium/analysis?match_id={match_id}", json={})
        response.raise_for_status()
        return response.json()

    async def request_review(self, match_id: UUID) -> Dict[str, Any]:
        """Request review."""
        response = await self._client.post(f"/premium/review?match_id={match_id}", json={})
        response.raise_for_status()
        return response.json()

    # Coins
    async def get_coin_balance(self) -> Dict[str, Any]:
        """Get coin balance."""
        response = await self._client.get("/coins/balance")
        response.raise_for_status()
        return response.json()

    async def purchase_coins(self, amount: int, package_id: str = "default") -> Dict[str, Any]:
        """Purchase coins."""
        response = await self._client.post(
            "/coins/purchase",
            json={"amount": amount, "package_id": package_id},
        )
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()

