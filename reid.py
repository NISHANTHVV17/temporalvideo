"""Appearance embeddings and conservative local/global track association."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from detect_track import TrackObservation


@dataclass
class Merge:
    source_track_id: str
    target_track_id: str
    score: float
    crossed_cut: bool
    reason: str


def color_histogram(frame: np.ndarray, box: tuple[float, float, float, float]) -> list[float]:
    x, y, width, height = [int(round(value)) for value in box]
    crop = frame[max(0, y):max(1, y + height), max(0, x):max(1, x + width)]
    if crop.size == 0:
        return [0.0] * 48
    hist = cv2.calcHist([crop], [0, 1, 2], None, [4, 4, 3], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(float).tolist()


def appearance_embedding(frame: np.ndarray, box: tuple[float, float, float, float]) -> list[float]:
    """Convenience fallback; long-running indexers should reuse TrackAppearanceEmbedder."""
    return TrackAppearanceEmbedder().embed(frame, box)


class TrackAppearanceEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self._attempted = False
        self.processor = self.model = self.torch = None

    def _load_local_clip(self) -> None:
        if self._attempted:
            return
        self._attempted = True
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained(self.model_name, local_files_only=True)
            self.model = CLIPModel.from_pretrained(self.model_name, local_files_only=True).to("cpu").eval()
            self.torch = torch
        except (ImportError, OSError, RuntimeError):
            self.processor = self.model = self.torch = None

    def embed(self, frame: np.ndarray, box: tuple[float, float, float, float]) -> list[float]:
        self._load_local_clip()
        if self.model is not None:
            x, y, width, height = [int(round(value)) for value in box]
            crop = frame[max(0, y):max(1, y + height), max(0, x):max(1, x + width)]
            if crop.size:
                try:
                    inputs = self.processor(images=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), return_tensors="pt")
                    with self.torch.no_grad():
                        vector = self.model.get_image_features(**inputs)[0]
                        vector = vector / vector.norm()
                    return vector.cpu().numpy().astype(float).tolist()
                except (RuntimeError, ValueError):
                    pass
        return color_histogram(frame, box)


def cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


def merge_tracks(tracks: list[dict[str, Any]], similarity_threshold: float = 0.82,
                 max_gap_seconds: float = 600.0) -> list[Merge]:
    """Greedy conservative pass; crossed-shot decisions are separately auditable."""
    ordered = sorted(tracks, key=lambda track: (track["t_start"], track["track_id"]))
    merges: list[Merge] = []
    consumed: set[str] = set()
    for index, left in enumerate(ordered):
        if left["track_id"] in consumed:
            continue
        for right in ordered[index + 1:]:
            if right["track_id"] in consumed or left["class"] != right["class"]:
                continue
            gap = float(right["t_start"] - left["t_end"])
            if gap < 0 or gap > max_gap_seconds:
                continue
            score = cosine_similarity(left.get("embedding"), right.get("embedding"))
            if score >= similarity_threshold:
                crossed = left["shot_id"] != right["shot_id"]
                merges.append(Merge(right["track_id"], left["track_id"], score, crossed,
                                    "class, appearance similarity, and temporal gap agree"))
                consumed.add(right["track_id"])
    return merges


def merged_id(track_id: str, merges: list[Merge]) -> str:
    mapping = {merge.source_track_id: merge.target_track_id for merge in merges}
    while track_id in mapping:
        track_id = mapping[track_id]
    return track_id


def serialize_embedding(embedding: list[float] | None) -> str | None:
    return json.dumps(embedding) if embedding is not None else None
