"""Service ML dùng cho quản trị."""

from __future__ import annotations

import asyncio
import logging
from typing import List
from uuid import UUID, uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase
from sqlalchemy.orm import Session

from ..config import Settings
from ..schemas import ml as ml_schema
from ..tasks import background

logger = logging.getLogger(__name__)


class MLService:
    def __init__(self, db: Session, mongo_db: AsyncIOMotorDatabase, settings: Settings) -> None:
        self.db = db
        self.mongo_db = mongo_db
        self.settings = settings
        self._training_jobs: dict[str, dict] = {}  # In-memory job tracking

    async def trigger_training(self, request: ml_schema.TrainRequest) -> dict:
        """Trigger ML training job trong background."""
        job_id = str(uuid4())
        config = request.model_dump()
        
        # Lưu job info
        self._training_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "config": config,
        }
        
        # Start training job trong background
        asyncio.create_task(self._run_training_job(job_id, config))
        
        logger.info(f"ML training job {job_id} queued")
        return {"job_id": job_id, "status": "queued", "config": config}

    async def _run_training_job(self, job_id: str, config: dict) -> None:
        """Run training job và update status."""
        try:
            self._training_jobs[job_id]["status"] = "running"
            result = await background.process_ml_training_job(job_id, config)
            self._training_jobs[job_id].update(result)
        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}", exc_info=True)
            self._training_jobs[job_id] = {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
            }

    def get_training_job(self, job_id: str) -> dict | None:
        """Lấy status của training job."""
        return self._training_jobs.get(job_id)

    async def list_models(self) -> List[ml_schema.ModelVersion]:
        # TODO: truy vấn bảng model_versions. Hiện trả về rỗng.
        return []

    async def promote_model(self, model_id: UUID) -> dict:
        # TODO: cập nhật model active.
        return {"model_id": str(model_id), "status": "pending"}

