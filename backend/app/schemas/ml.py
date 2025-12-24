"""Schema cho endpoint ML (admin và analysis)."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class TrainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_type: str = "policy"
    iterations: int = 1


class ModelVersion(BaseModel):
    id: UUID
    policy_path: str | None = None
    value_path: str | None = None
    description: str | None = None
    created_at: datetime


# Analysis schemas
class ThreatRegion(BaseModel):
    type: str
    positions: List[List[int]]
    severity: float
    description: str
    recommendation: str


class ThreatAnalysis(BaseModel):
    heatmap: List[List[float]]
    regions: List[ThreatRegion]
    summary: Dict[str, int]


class AttackOpportunity(BaseModel):
    type: str
    position: List[int]
    confidence: float
    expected_gain: int
    description: str


class AttackAnalysis(BaseModel):
    heatmap: List[List[float]]
    opportunities: List[AttackOpportunity]
    summary: Dict[str, int]


class PositionEvaluation(BaseModel):
    win_probability: float
    territory_estimate: Dict[str, float]
    stone_count: Dict[str, int]
    game_phase: str


class IntentInfo(BaseModel):
    type: str
    probability: float
    description: str


class IntentAnalysis(BaseModel):
    primary_intent: str
    confidence: float
    all_intents: List[IntentInfo]
    heatmap: List[List[float]]
    strategic_advice: str


class BestMove(BaseModel):
    position: Optional[List[int]]
    confidence: float


class PositionAnalysisResponse(BaseModel):
    threats: ThreatAnalysis
    attacks: AttackAnalysis
    evaluation: PositionEvaluation
    intent: IntentAnalysis
    best_move: Optional[BestMove]
    fallback: bool = False


class AnalyzePositionRequest(BaseModel):
    match_id: UUID
    board_position: Dict[str, str]  # {'x,y': 'B' or 'W'}
    current_player: str  # 'B' or 'W'
    board_size: int = 19
    move_history: Optional[List[List[int]]] = None  # [[x, y], ...]


class AnalyzePositionFromMatchRequest(BaseModel):
    match_id: UUID

