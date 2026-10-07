"""User-defined zones in stabilized, normalized video coordinates."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: list[tuple[float, float]]
    shot_id: int | None = None

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError("a zone polygon needs at least three points")
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in self.polygon):
            raise ValueError("zone points must be normalized to [0, 1]")

    def contains(self, point: tuple[float, float]) -> bool:
        polygon = np.asarray(self.polygon, dtype=np.float32)
        return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

    @classmethod
    def from_pixels(cls, name: str, points: list[tuple[int, int]], width: int, height: int,
                    shot_id: int | None = None) -> "Zone":
        return cls(name, [(x / width, y / height) for x, y in points], shot_id)


@dataclass(frozen=True)
class Line:
    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    shot_id: int | None = None

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError("a crossing line needs two distinct points")
        if any(not (0 <= value <= 1) for point in (self.start, self.end) for value in point):
            raise ValueError("line points must be normalized to [0, 1]")


def load_zones(path: str | Path) -> list[Zone]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    values = raw.get("zones", raw) if isinstance(raw, dict) else raw
    return [Zone(name=item["name"], polygon=[tuple(point) for point in item["polygon"]],
                 shot_id=item.get("shot_id")) for item in values]


def load_lines(path: str | Path) -> list[Line]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    values = raw.get("lines", []) if isinstance(raw, dict) else []
    return [Line(name=item["name"], start=tuple(item["start"]), end=tuple(item["end"]),
                 shot_id=item.get("shot_id")) for item in values]


def propose_zone_from_question(question: str) -> str | None:
    """Return a phrase for user confirmation; geometry is never fabricated."""
    import re
    match = re.search(r"(?:in|into|inside|within|near) the ([\w -]+?)(?:\?| after| before|$)", question, re.I)
    return match.group(1).strip() if match else None
