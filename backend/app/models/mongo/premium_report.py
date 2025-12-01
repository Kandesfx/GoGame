"""Model cho collection premium_reports."""

from datetime import datetime

from pydantic import BaseModel, Field


class PremiumReportDocument(BaseModel):
    id: str = Field(alias="_id")
    match_id: str
    feature: str
    summary: str | None = None
    details: dict | None = None
    coins_spent: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True

