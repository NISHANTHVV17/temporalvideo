"""Generic object detection and shot-local tracking with optional learned backends."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import math

import cv2
import numpy as np
import logging
import os
from scipy.optimize import linear_sum_assignment

from ingest import FrameSample


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    class_name: str
    confidence: float
    embedding: list[float] | None = None
    external_track_id: str | None = None


@dataclass
class TrackObservation:
    track_id: str
    local_id: str
    shot_id: int
    pts: float
    class_name: str
    box: tuple[float, float, float, float]
    confidence: float
    embedding: list[float] | None
    stabilized_center: tuple[float, float]


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class ContourFallbackDetector:
    """Model-free generic motion/foreground proposal detector; labels remain unknown."""
    def __init__(self, min_area: float = 120.0):
        self.background = cv2.createBackgroundSubtractorMOG2(history=180, varThreshold=28,
                                                              detectShadows=False)
        self.min_area = min_area
        self.reference: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        mask = self.background.apply(frame)
        if self.reference is None:
            self.reference = frame.copy()
        else:
            static_difference = cv2.absdiff(frame, self.reference)
            static_gray = cv2.cvtColor(static_difference, cv2.COLOR_BGR2GRAY)
            _, static_mask = cv2.threshold(static_gray, 22, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_or(mask, static_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area >= self.min_area:
                detections.append(Detection((float(x), float(y), float(width), float(height)),
                                            "unknown", min(0.55, 0.25 + area / (frame.shape[0] * frame.shape[1]))))
        return detections


class UltralyticsDetector:
    def __init__(self, model_path: str = "yolo11n.pt", confidence: float = 0.25,
                 openvino: bool = False):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is optional; install it to enable YOLO detection") from exc
        self.confidence = confidence
        path = Path(model_path)
        model = YOLO(str(path))
        if openvino:
            path = Path(model.export(format="openvino"))
            model = YOLO(str(path))
        self.model = model
        self.names = self.model.names

    def reset(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) or []:
            if hasattr(tracker, "reset"):
                tracker.reset()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.track(frame, persist=True, tracker="botsort.yaml", device="cpu",
                                  verbose=False, conf=self.confidence)[0]
        output = []
        for item in result.boxes:
            x1, y1, x2, y2 = item.xyxy[0].cpu().numpy().tolist()
            class_id = int(item.cls[0])
            external_id = str(int(item.id[0])) if item.id is not None else None
            output.append(Detection((x1, y1, x2 - x1, y2 - y1), str(self.names[class_id]),
                                    float(item.conf[0]), external_track_id=external_id))
        return output


def make_detector(backend: str = "auto", model_path: str = "yolo11n.pt",
                  confidence: float = 0.25) -> Detector:
    if backend in {"auto", "ultralytics", "openvino"}:
        try:
            return UltralyticsDetector(model_path, confidence, openvino=(backend == "openvino"))
        except Exception as exc:
            logging.warning("Optional detector unavailable; using generic foreground proposals: %s", exc)
    return ContourFallbackDetector()


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    return intersection / max(1e-6, aw * ah + bw * bh - intersection)


def _stabilized_point(box: tuple[float, float, float, float], matrix: np.ndarray,
                      width: int, height: int) -> tuple[float, float]:
    x, y, w, h = box
    point = np.array([[[x + w / 2, y + h / 2]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, matrix)[0, 0]
    return float(transformed[0] / width), float(transformed[1] / height)


def _detection_center(detection: Detection | tuple[float, float, float, float]) -> tuple[float, float]:
    if isinstance(detection, Detection):
        x, y, w, h = detection.box
    else:
        x, y, w, h = detection
    return (x + w / 2.0, y + h / 2.0)


def _collapse_duplicate_detections(detections: list[Detection]) -> list[Detection]:
    if len(detections) < 2:
        return detections
    merged: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        kept = None
        for index, existing in enumerate(merged):
            same_box = _iou(existing.box, candidate.box) > 0.45 or (
                math.dist(_detection_center(existing), _detection_center(candidate)) <= 0.12
            )
            if not same_box:
                continue
            if existing.class_name == "unknown" and candidate.class_name != "unknown":
                merged[index] = Detection(candidate.box, candidate.class_name, max(existing.confidence, candidate.confidence),
                                          candidate.embedding or existing.embedding,
                                          candidate.external_track_id or existing.external_track_id)
            elif candidate.class_name == "unknown":
                merged[index] = Detection(existing.box, existing.class_name, max(existing.confidence, candidate.confidence),
                                          existing.embedding or candidate.embedding,
                                          existing.external_track_id or candidate.external_track_id)
            else:
                preferred = existing if existing.confidence >= candidate.confidence else candidate
                union_box = (
                    min(existing.box[0], candidate.box[0]),
                    min(existing.box[1], candidate.box[1]),
                    max(existing.box[0] + existing.box[2], candidate.box[0] + candidate.box[2]) - min(existing.box[0], candidate.box[0]),
                    max(existing.box[1] + existing.box[3], candidate.box[1] + candidate.box[3]) - min(existing.box[1], candidate.box[1]),
                )
                merged[index] = Detection(union_box, preferred.class_name, max(existing.confidence, candidate.confidence),
                                          preferred.embedding or existing.embedding or candidate.embedding,
                                          preferred.external_track_id or existing.external_track_id or candidate.external_track_id)
            kept = index
            break
        if kept is None:
            merged.append(candidate)
    return merged


class MultiObjectTracker:
    def __init__(self, max_gap_seconds: float = 2.0):
        self.max_gap_seconds = max_gap_seconds
        self.active: dict[str, TrackObservation] = {}
        self.previous: dict[str, TrackObservation] = {}
        self.external_to_track: dict[str, str] = {}
        self.next_id = 1
        self.shot_id: int | None = None

    def reset(self, shot_id: int) -> None:
        self.active.clear()
        self.previous.clear()
        self.external_to_track.clear()
        self.shot_id = shot_id

    def update(self, sample: FrameSample, detections: list[Detection]) -> list[TrackObservation]:
        if self.shot_id != sample.shot_id:
            self.reset(sample.shot_id)
        detections = _collapse_duplicate_detections(detections)
        height, width = sample.frame.shape[:2]
        track_ids = []
        track_states = []
        for track_id, observation in self.active.items():
            gap = sample.pts - observation.pts
            if gap < 0 or gap > self.max_gap_seconds:
                continue
            track_ids.append(track_id)
            track_states.append(observation)

        scores = np.full((len(track_ids), len(detections)), -1e6, dtype=np.float64)
        for track_index, (track_id, previous) in enumerate(zip(track_ids, track_states)):
            gap = sample.pts - previous.pts
            predicted = self._predict_center(track_id, previous, gap)
            for index, detection in enumerate(detections):
                if detection.class_name != previous.class_name and "unknown" not in {detection.class_name, previous.class_name}:
                    continue
                overlap = _iou(previous.box, detection.box)
                stabilized = _stabilized_point(detection.box, sample.to_reference, width, height)
                distance = math.dist(predicted, stabilized)
                if overlap > 0.01 or distance < 0.08:
                    appearance = self._appearance_similarity(previous.embedding, detection.embedding)
                    scores[track_index, index] = overlap * 0.05 + appearance * 0.2 - distance

        assignment: dict[int, str] = {}
        externally_matched_tracks: set[str] = set()
        for detection_index, detection in enumerate(detections):
            if detection.external_track_id is None:
                continue
            existing_id = self.external_to_track.get(detection.external_track_id)
            if existing_id in track_ids and existing_id not in externally_matched_tracks:
                assignment[detection_index] = existing_id
                externally_matched_tracks.add(existing_id)
        if scores.size:
            track_indices, detection_indices = linear_sum_assignment(-scores)
            for track_index, detection_index in zip(track_indices, detection_indices):
                if (scores[track_index, detection_index] > -1e5
                        and detection_index not in assignment
                        and track_ids[track_index] not in externally_matched_tracks):
                    assignment[detection_index] = track_ids[track_index]

        ambiguous_detections = set()
        if len(detections) < len(track_ids):
            for detection_index in range(len(detections)):
                if detection_index in assignment:
                    continue
                valid_scores = np.sort(scores[:, detection_index][scores[:, detection_index] > -1e5])
                if len(valid_scores) >= 2 and valid_scores[-1] - valid_scores[-2] < 0.04:
                    ambiguous_detections.add(detection_index)
                    assignment.pop(detection_index, None)

        observations = []
        for index, detection in enumerate(detections):
            if index in ambiguous_detections:
                continue
            track_id = assignment.get(index)
            if track_id is None:
                local_id = detection.external_track_id or str(self.next_id)
                if detection.external_track_id is None:
                    self.next_id += 1
                track_id = f"s{sample.shot_id}-t{local_id}"
            else:
                local_id = track_id.rsplit("t", 1)[-1]
            previous_observation = self.active.get(track_id)
            if previous_observation is not None:
                self.previous[track_id] = previous_observation
            observation = TrackObservation(track_id, local_id, sample.shot_id, sample.pts,
                                           detection.class_name, detection.box, detection.confidence,
                                           detection.embedding,
                                           _stabilized_point(detection.box, sample.to_reference, width, height))
            self.active[track_id] = observation
            observations.append(observation)
            if detection.external_track_id is not None:
                self.external_to_track[detection.external_track_id] = track_id
        retention_window = max(self.max_gap_seconds * 2.5, 1.5)
        self.active = {key: value for key, value in self.active.items()
                       if sample.pts - value.pts <= retention_window}
        self.previous = {key: value for key, value in self.previous.items() if key in self.active}
        return observations

    def _predict_center(self, track_id: str, latest: TrackObservation,
                        gap_seconds: float) -> tuple[float, float]:
        earlier = self.previous.get(track_id)
        if earlier is None:
            return latest.stabilized_center
        elapsed = latest.pts - earlier.pts
        if elapsed <= 0:
            return latest.stabilized_center
        velocity_x = (latest.stabilized_center[0] - earlier.stabilized_center[0]) / elapsed
        velocity_y = (latest.stabilized_center[1] - earlier.stabilized_center[1]) / elapsed
        speed = math.hypot(velocity_x, velocity_y)
        if speed > 1.5:
            velocity_x *= 1.5 / speed
            velocity_y *= 1.5 / speed
        return (latest.stabilized_center[0] + velocity_x * gap_seconds,
                latest.stabilized_center[1] + velocity_y * gap_seconds)

    @staticmethod
    def _appearance_similarity(first: list[float] | None, second: list[float] | None) -> float:
        if not first or not second or len(first) != len(second):
            return 0.0
        left, right = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        return float(np.dot(left, right) / denominator) if denominator else 0.0


def extract_question_classes(question: str) -> list[str]:
    """Extract lightweight noun-like phrases without imposing a fixed domain taxonomy."""
    import re
    text = re.sub(r"[^\w\s'-]", " ", question.lower())
    stop = {"which", "what", "when", "where", "who", "how", "many", "times", "did", "does", "do",
            "the", "a", "an", "after", "before", "during", "into", "from", "with", "without",
            "happen", "happened", "right", "first", "last", "same", "there", "that", "this", "is", "was"}
    words = [word for word in text.split() if word not in stop and len(word) > 1]
    return list(dict.fromkeys(words))[:8]


class OpenVocabularyDetector:
    """On-demand question-conditioned detector (YOLO-World when installed)."""
    def __init__(self, classes: list[str]):
        model_path = Path(os.getenv("TEMPORALVIDEO_YOLOWORLD_MODEL", "yolov8s-worldv2.pt"))
        if not model_path.exists():
            raise RuntimeError("YOLO-World weights are not available locally")
        from ultralytics import YOLO
        self.model = YOLO(str(model_path))
        self.model.set_classes(classes)
        self.classes = classes

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(frame, verbose=False)[0]
        return [Detection((float(box.xyxy[0][0]), float(box.xyxy[0][1]),
                           float(box.xyxy[0][2] - box.xyxy[0][0]),
                           float(box.xyxy[0][3] - box.xyxy[0][1])),
                          self.classes[int(box.cls[0])], float(box.conf[0])) for box in result.boxes]
