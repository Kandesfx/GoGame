"""Schema cho endpoint ML (admin)."""

from datetime import datetime
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

