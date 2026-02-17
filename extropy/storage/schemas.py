"""Pydantic schemas for canonical study DB payloads."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class AgentDBRecord(BaseModel):
    """Validated representation of an agent row in the study DB."""

    population_id: str
    agent_id: str
    attrs_json: dict[str, Any]
    sample_run_id: str
    scenario_id: str | None = None


class NetworkEdgeDBRecord(BaseModel):
    """Validated representation of a network edge row in the study DB."""

    network_id: str
    source_id: str
    target_id: str
    weight: float
    edge_type: str
    influence_st: float | None = None
    influence_ts: float | None = None
    scenario_id: str | None = None


class ChatMessagePayload(BaseModel):
    """Validated chat message payload persisted in chat_messages."""

    role: str = Field(min_length=1)
    content: str = Field(min_length=1)
    citations: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, Any] = Field(default_factory=dict)


class ReadOnlySQLRequest(BaseModel):
    """Read-only SQL request contract for query CLI."""

    model_config = ConfigDict(str_strip_whitespace=True)

    sql: str = Field(min_length=1)
    limit: int = Field(default=1000, ge=1)
