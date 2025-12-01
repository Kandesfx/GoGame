"""Schema chung dùng lại nhiều nơi."""

from uuid import UUID

from pydantic import BaseModel


class Message(BaseModel):
    message: str


class ObjectIdResponse(BaseModel):
    id: UUID

