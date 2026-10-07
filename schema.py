"""Validated contracts shared by indexing, planning, execution, and UI."""
from __future__ import annotations

from enum import Enum
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EntityKind(str, Enum):
    object = "object"
    person = "person"
    zone = "zone"
    sound = "sound"
    action = "action"
    state = "state"


class Intent(str, Enum):
    find_event = "find_event"
    count = "count"
    order = "order"
    before_after = "before_after"
    window_before = "window_before"
    window_after = "window_after"
    duration_filter = "duration_filter"
    identify_track = "identify_track"
    describe = "describe"
    first = "first"
    last = "last"


class EntitySpec(BaseModel):
    phrase: str = Field(min_length=1)
    kind: EntityKind = EntityKind.action
    class_hint: str | None = None


class RelationSpec(BaseModel):
    a: str
    relation: Literal["after", "before", "during", "within_N_seconds"]
    b: str
    seconds: float | None = Field(default=None, ge=0)


class QueryFilters(BaseModel):
    min_duration: float | None = Field(default=None, ge=0)
    zone: str | None = None
    time_range: tuple[float, float] | None = None
    count_distinct: bool = True

    @model_validator(mode="after")
    def check_range(self) -> "QueryFilters":
        if self.time_range and self.time_range[1] < self.time_range[0]:
            raise ValueError("time_range end must not precede its start")
        return self


class QueryPlan(BaseModel):
    intent: Intent
    entities: list[EntitySpec] = Field(default_factory=list)
    relations: list[RelationSpec] = Field(default_factory=list)
    filters: QueryFilters = Field(default_factory=QueryFilters)
    raw_question: str
    asks_causation: bool = False
    preceding_seconds: float = Field(default=10.0, ge=0)


class TimedInterval(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    label: str
    source: Literal["rule", "audio", "clip", "vlm", "detector"] = "rule"
    confidence: float = Field(ge=0, le=1)
    track_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_interval(self) -> "TimedInterval":
        if self.end < self.start:
            raise ValueError("interval end must be at or after start")
        return self


class PrecedingEvent(BaseModel):
    description: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    event_ids: list[str] = Field(default_factory=list)
    relation_rating: Literal["plausibly-related", "unrelated", "not-assessed"] = "not-assessed"
    rating_confidence: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def valid_interval(self) -> "PrecedingEvent":
        if self.end < self.start:
            raise ValueError("preceding event end must be at or after start")
        return self


class Answer(BaseModel):
    """A timestamp is mandatory, including for low-confidence best candidates."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    t_start: float = Field(ge=0)
    t_end: float = Field(ge=0)
    track_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    low_confidence: bool = False
    id_merge_confidence: float | None = Field(default=None, ge=0, le=1)
    cross_shot_merges: list[str] = Field(default_factory=list)
    video_duration: float | None = Field(default=None, ge=0)
    timestamp_source: Literal["rule", "audio", "clip", "vlm", "detector"]
    uncertainty_seconds: float = Field(ge=0)
    preceding_events: list[PrecedingEvent] = Field(default_factory=list)
    caveat: str | None = None

    @model_validator(mode="after")
    def ensure_timestamp_and_low_flag(self) -> "Answer":
        if self.t_end <= self.t_start:
            raise ValueError("answer timestamp interval must have positive duration")
        if re.search(r"\b(?:because of|caused|triggered|resulted in|led to)\b", self.answer, re.I):
            raise ValueError("answers must describe temporal evidence, not causation")
        if not self.event_ids and not self.track_ids:
            raise ValueError("answer requires at least one supporting event or track ID")
        if self.confidence < 0.35 and not self.low_confidence:
            raise ValueError("confidence below 0.35 requires low_confidence=true")
        return self

    def render(self, duration: float | None = None) -> str:
        duration = duration if duration is not None else self.video_duration
        start = format_timestamp(self.t_start, duration)
        end = format_timestamp(self.t_end, duration)
        tracks = ",".join(self.track_ids) or "none"
        events = ",".join(self.event_ids) or "none"
        confidence = f"{self.confidence:.2f}" + (" LOW" if self.low_confidence else "")
        merge_confidence = (f" id_merge_confidence={self.id_merge_confidence:.2f}"
                    if self.id_merge_confidence is not None else "")
        cut_merges = f" cross_shot_merges={';'.join(self.cross_shot_merges)}" if self.cross_shot_merges else ""
        return (f"{self.answer} [{start}-{end}] tracks={tracks} events={events} "
            f"confidence={confidence}{merge_confidence}{cut_merges} "
            f"source={self.timestamp_source} uncertainty=+/-{self.uncertainty_seconds:.1f}s")


def format_timestamp(seconds: float, duration: float | None = None) -> str:
    total = max(0, int(seconds))
    if (duration or seconds) >= 3600:
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"
