"""Optional CLIP temporal grounding and lazy short-window caption hooks."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from db import EvidenceDB
from ingest import iter_video


class ClipSemanticIndex:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.model_name, self.device = model_name, device
        self._loaded = False
        self.processor = self.model = None

    def _load(self) -> bool:
        if self._loaded:
            return self.model is not None
        self._loaded = True
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            self.processor = CLIPProcessor.from_pretrained(self.model_name, local_files_only=True)
            self.model = CLIPModel.from_pretrained(self.model_name, local_files_only=True).to(self.device).eval()
            self.torch = torch
            return True
        except Exception:
            return False

    def index_video(self, path: str | Path, video_id: str, db: EvidenceDB,
                    cache_dir: str | Path, embedding_fps: float = 1.0) -> int:
        if not self._load():
            return 0
        import cv2
        output = Path(cache_dir) / video_id / "semantic"
        output.mkdir(parents=True, exist_ok=True)
        interval = 1.0 / embedding_fps
        next_time = 0.0
        count = 0
        for sample in iter_video(path, sample_fps=embedding_fps, max_dimension=640):
            if sample.pts + 1e-9 < next_time:
                continue
            next_time = sample.pts + interval
            rgb = cv2.cvtColor(sample.frame, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=rgb, return_tensors="pt")
            with self.torch.no_grad():
                vector = self.model.get_image_features(**inputs)[0]
                vector = vector / vector.norm()
            frame_path = output / f"{sample.pts:.3f}.jpg"
            cv2.imwrite(str(frame_path), sample.frame)
            db.execute("INSERT OR REPLACE INTO semantic_frames(video_id,pts,image_path,embedding_json) VALUES(?,?,?,?)",
                       (video_id, sample.pts, str(frame_path), json.dumps(vector.cpu().numpy().tolist())))
            count += 1
        return count

    def ground_text(self, phrase: str, db: EvidenceDB, video_id: str,
                    threshold: float = 0.22) -> list[dict[str, Any]]:
        if not self._load():
            return []
        rows = db.rows("SELECT pts,embedding_json FROM semantic_frames WHERE video_id=? ORDER BY pts", (video_id,))
        if not rows:
            return []
        inputs = self.processor(text=[phrase], return_tensors="pt", padding=True)
        with self.torch.no_grad():
            text = self.model.get_text_features(**inputs)[0]
            text = text / text.norm()
        scores = []
        for row in rows:
            image = self.torch.tensor(json.loads(row["embedding_json"]), dtype=text.dtype)
            scores.append((float(self.torch.dot(text.cpu(), image)), float(row["pts"])))
        candidates = [(score, pts) for score, pts in scores if score >= threshold]
        if not candidates:
            return []
        candidates.sort(key=lambda item: item[1])
        groups: list[list[tuple[float, float]]] = []
        for candidate in candidates:
            if not groups or candidate[1] - groups[-1][-1][1] > 2.0:
                groups.append([candidate])
            else:
                groups[-1].append(candidate)
        return [{"start": group[0][1], "end": group[-1][1] + 1.0,
                 "score": max(score for score, _ in group)} for group in groups]


def coarse_caption_event(video_id: str, start: float, end: float, caption: str,
                         confidence: float = 0.4) -> dict[str, Any]:
    import uuid
    return {"event_id": uuid.uuid4().hex, "video_id": video_id, "track_id": None,
            "class": "scene", "event_type": "vlm_caption", "t_start": start, "t_end": end,
            "zone": None, "confidence": confidence, "meta_json": json.dumps({"caption": caption})}
