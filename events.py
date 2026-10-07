"""Domain-agnostic temporal events derived from tracks, zones, and frame motion."""
from __future__ import annotations

import json
import math
import uuid
from bisect import bisect_left, bisect_right
from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from db import EvidenceDB
from detect_track import TrackObservation
from zones import Line, Zone


def _event(video_id: str, kind: str, start: float, end: float, track_id: str | None = None,
           class_name: str | None = None, confidence: float = 0.7, zone: str | None = None,
           meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"event_id": uuid.uuid4().hex, "video_id": video_id, "track_id": track_id,
            "class": class_name, "event_type": kind, "t_start": float(start), "t_end": float(end),
            "zone": zone, "confidence": float(confidence), "meta_json": json.dumps(meta or {})}


def build_track_events(video_id: str, observations: list[TrackObservation], db: EvidenceDB,
                       zones: list[Zone], stationary_epsilon: float = 0.025,
                       stationary_seconds: float = 8.0, interaction_distance: float = 0.12,
                       lines: list[Line] | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, list[TrackObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.track_id].append(observation)
    events: list[dict[str, Any]] = []
    for track_id, points in grouped.items():
        points.sort(key=lambda point: point.pts)
        first, last = points[0], points[-1]
        events.append(_event(video_id, "appear", first.pts, first.pts, track_id, first.class_name, first.confidence))
        events.append(_event(video_id, "disappear", last.pts, last.pts, track_id, last.class_name, last.confidence))
        previous_moving = False
        for previous, current in zip(points, points[1:]):
            elapsed = current.pts - previous.pts
            distance = math.dist(previous.stabilized_center, current.stabilized_center)
            moving = elapsed > 0 and distance / elapsed > stationary_epsilon
            if moving != previous_moving and elapsed <= 2.0:
                events.append(_event(video_id, "motion_start" if moving else "motion_stop",
                                     current.pts, current.pts, track_id, current.class_name,
                                     current.confidence, meta={"stabilized_speed": distance / max(elapsed, 1e-6)}))
            previous_moving = moving
            for line in lines or []:
                if line.shot_id is not None and line.shot_id != current.shot_id:
                    continue
                side_before = _cross(line.start, line.end, previous.stabilized_center)
                side_after = _cross(line.start, line.end, current.stabilized_center)
                if side_before * side_after < 0 and _segments_intersect(
                        previous.stabilized_center, current.stabilized_center, line.start, line.end):
                    direction = "forward" if side_before < side_after else "reverse"
                    events.append(_event(video_id, "line_crossing", previous.pts, current.pts,
                                         track_id, current.class_name, current.confidence,
                                         zone=line.name, meta={"direction": direction}))
        zone_states: dict[str, bool] = {}
        for point in points:
            for zone in zones:
                if zone.shot_id is not None and zone.shot_id != point.shot_id:
                    continue
                inside = zone.contains(point.stabilized_center)
                was_inside = zone_states.get(zone.name, False)
                if inside != was_inside:
                    events.append(_event(video_id, "zone_enter" if inside else "zone_exit", point.pts,
                                         point.pts, track_id, point.class_name, point.confidence, zone.name))
                    zone_states[zone.name] = inside

        run_start = 0
        for index in range(1, len(points) + 1):
            ended = index == len(points)
            if not ended:
                elapsed = points[index].pts - points[index - 1].pts
                dx = points[index].stabilized_center[0] - points[index - 1].stabilized_center[0]
                dy = points[index].stabilized_center[1] - points[index - 1].stabilized_center[1]
                moved = math.hypot(dx, dy) > stationary_epsilon or elapsed > 2.0
                ended = moved
            if ended:
                segment = points[run_start:index]
                if len(segment) > 1 and segment[-1].pts - segment[0].pts >= stationary_seconds:
                    midpoint = segment[len(segment) // 2]
                    person_nearby = any(
                        other.track_id != track_id
                        and any(token in other.class_name.lower() for token in ("person", "human"))
                        and abs(other.pts - midpoint.pts) <= 1.0
                        and math.dist(other.stabilized_center, midpoint.stabilized_center) <= interaction_distance
                        for other in observations
                    )
                    events.append(_event(video_id, "stationary", segment[0].pts, segment[-1].pts,
                                         track_id, segment[0].class_name, 0.65,
                                         meta={"motion_threshold": stationary_epsilon,
                                               "person_nearby_checked": True,
                                               "person_nearby": person_nearby}))
                if index < len(points):
                    run_start = index

    by_shot: dict[int, list[TrackObservation]] = defaultdict(list)
    for item in observations:
        by_shot[item.shot_id].append(item)
    for shot_points in by_shot.values():
        ordered = sorted(shot_points, key=lambda item: item.pts)
        active_zones: dict[str, tuple[float, str]] = {}
        for item in ordered:
            for zone in zones:
                if zone.shot_id is not None and zone.shot_id != item.shot_id:
                    continue
                inside = zone.contains(item.stabilized_center)
                state_key = f"{item.track_id}:{zone.name}"
                if inside and state_key not in active_zones:
                    active_zones[state_key] = (item.pts, item.track_id)
                elif not inside and state_key in active_zones:
                    start, _ = active_zones.pop(state_key)
                    if item.pts > start:
                        events.append(_event(video_id, "dwell", start, item.pts, item.track_id,
                                             item.class_name, 0.7, zone.name))
        for state_key, (start, track_id) in active_zones.items():
            if ordered and ordered[-1].pts > start:
                last = ordered[-1]
                events.append(_event(video_id, "dwell", start, last.pts, track_id,
                                     last.class_name, 0.7, state_key.split(":", 1)[1]))

    observations_at_pts: dict[float, list[TrackObservation]] = defaultdict(list)
    for item in observations:
        observations_at_pts[item.pts].append(item)
    for timestamp, same_time in observations_at_pts.items():
        for index, left in enumerate(same_time):
            for right in same_time[index + 1:]:
                distance = math.dist(left.stabilized_center, right.stabilized_center)
                if distance <= interaction_distance:
                    events.append(_event(video_id, "interaction", timestamp, timestamp, left.track_id,
                                         left.class_name, min(left.confidence, right.confidence),
                                         meta={"other_track_id": right.track_id, "distance": distance}))
    person_tracks = [items for items in grouped.values()
                     if any(token in items[0].class_name.lower() for token in ("person", "human"))]
    person_points = sorted((point for items in person_tracks for point in items), key=lambda item: item.pts)
    person_times = [point.pts for point in person_points]
    for track_id, points in grouped.items():
        if any(token in points[0].class_name.lower() for token in ("person", "human")):
            continue
        for edge, point, event_type in (("appear", points[0], "pickup_candidate"),
                                        ("disappear", points[-1], "putdown_candidate")):
            lo = bisect_left(person_times, point.pts - 1.5)
            hi = bisect_right(person_times, point.pts + 1.5)
            nearest = min((math.dist(point.stabilized_center, person.stabilized_center)
                           for person in person_points[lo:hi]), default=float("inf"))
            if nearest <= interaction_distance:
                events.append(_event(video_id, event_type, point.pts, point.pts, track_id,
                                     point.class_name, 0.45,
                                     meta={"heuristic": "object boundary near a person", "edge": edge,
                                           "distance": nearest, "not_verified": True}))
    for item in events:
        db.add_event(item)
    return events


def _cross(a: tuple[float, float], b: tuple[float, float], point: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def _segments_intersect(a: tuple[float, float], b: tuple[float, float],
                        c: tuple[float, float], d: tuple[float, float]) -> bool:
    movement = (b[0] - a[0], b[1] - a[1])
    line = (d[0] - c[0], d[1] - c[1])
    denominator = movement[0] * line[1] - movement[1] * line[0]
    if abs(denominator) < 1e-9:
        return False
    offset = (c[0] - a[0], c[1] - a[1])
    along_movement = (offset[0] * line[1] - offset[1] * line[0]) / denominator
    along_line = (offset[0] * movement[1] - offset[1] * movement[0]) / denominator
    return 0 <= along_movement <= 1 and 0 <= along_line <= 1


class MotionAnalyzer:
    def __init__(self, threshold: float = 0.015, min_seconds: float = 0.5):
        self.threshold = threshold
        self.min_seconds = min_seconds
        self.previous: np.ndarray | None = None
        self.state: bool | None = None
        self.state_start = 0.0

    def update(self, frame: np.ndarray, pts: float, video_id: str, db: EvidenceDB,
               roi: list[tuple[float, float]] | None = None,
               region_name: str | None = None) -> dict[str, Any] | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.previous is None:
            self.previous = gray
            self.state_start = pts
            self.state = False
            return None
        difference = cv2.absdiff(gray, self.previous)
        if roi:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            height, width = gray.shape
            polygon = np.asarray([(round(x * width), round(y * height)) for x, y in roi], dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 255)
            region = difference[mask > 0]
            energy = float(np.mean(region) / 255.0) if region.size else 0.0
        else:
            energy = float(np.mean(difference) / 255.0)
        moving = energy >= self.threshold
        self.previous = gray
        if moving == self.state:
            return None
        previous_state = bool(self.state)
        start = self.state_start
        self.state, self.state_start = moving, pts
        if pts - start < self.min_seconds:
            return None
        item = _event(video_id, "motion_start" if moving else "motion_stop", start, pts,
                      confidence=min(0.8, 0.4 + energy), meta={"motion_energy": energy,
                                                              "previous_state": previous_state,
                                                              "region": region_name or "frame"})
        db.add_event(item)
        return item


def detect_static_foreground(frame: np.ndarray, background: np.ndarray) -> list[tuple[int, int, int, int]]:
    """A lightweight generic background-difference proposal for missed small objects."""
    difference = cv2.absdiff(frame, background)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 24, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) >= 20]
