"""Optional CLIP temporal grounding and lazy short-window caption hooks."""
from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from db import EvidenceDB
from ingest import iter_video

load_dotenv(Path(__file__).with_name(".env"), override=False)


def configured_embedding_provider(provider: str | None = None) -> str:
    selected = provider or os.getenv("TEMPORALVIDEO_EMBEDDING_BACKEND", "auto")
    if selected == "auto":
        return "nvidia" if os.getenv("NVIDIA_EMBEDDING_API_KEY") else "local"
    return selected


class ClipSemanticIndex:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu",
                 provider: str | None = None):
        self.model_name, self.device = model_name, device
        self.provider = configured_embedding_provider(provider)
        self._loaded = False
        self.processor = self.model = None
        self._nvidia_client = None

    def _embed_nvidia(self, inputs: list[str], modality: str) -> list[list[float]]:
        if self._nvidia_client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install optional package openai to use NVIDIA NV-CLIP") from exc
            api_key = os.getenv("NVIDIA_EMBEDDING_API_KEY")
            if not api_key:
                raise RuntimeError("NVIDIA_EMBEDDING_API_KEY is not configured")
            self._nvidia_client = OpenAI(api_key=api_key,
                                         base_url="https://integrate.api.nvidia.com/v1")
        try:
            response = self._nvidia_client.embeddings.create(
                input=inputs,
                model=os.getenv("TEMPORALVIDEO_EMBEDDING_MODEL",
                                "nvidia/llama-nemotron-embed-vl-1b-v2"),
                encoding_format="float",
                extra_body={"modality": [modality],
                            "input_type": "passage" if modality == "image" else "query",
                            "truncate": "NONE"})
        except Exception as exc:
            raise RuntimeError(
                "NVIDIA multimodal embedding failed; verify this API key has access to the configured model") from exc
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    @staticmethod
    def _normalized(vector: list[float]) -> np.ndarray:
        result = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(result))
        if norm == 0:
            raise ValueError("Embedding vector has zero norm")
        return result / norm

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
        if self.provider == "nvidia":
            return self._index_video_nvidia(path, video_id, db, cache_dir, embedding_fps)
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

    def _index_video_nvidia(self, path: str | Path, video_id: str, db: EvidenceDB,
                            cache_dir: str | Path, embedding_fps: float) -> int:
        import cv2

        output = Path(cache_dir) / video_id / "semantic"
        output.mkdir(parents=True, exist_ok=True)
        interval = 1.0 / embedding_fps
        next_time = 0.0
        records: list[tuple[float, str, str]] = []
        embedded_records: list[tuple[float, str, list[float]]] = []

        def flush_batch() -> None:
            if not records:
                return
            vectors = self._embed_nvidia([item[2] for item in records], modality="image")
            if len(vectors) != len(records):
                raise RuntimeError("NVIDIA NV-CLIP returned an unexpected embedding count")
            embedded_records.extend((pts, image_path, vector)
                                    for (pts, image_path, _), vector in zip(records, vectors))
            records.clear()

        for sample in iter_video(path, sample_fps=embedding_fps, max_dimension=640):
            if sample.pts + 1e-9 < next_time:
                continue
            next_time = sample.pts + interval
            image = sample.frame
            quality = 65
            while True:
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ok:
                    raise RuntimeError("Could not encode a frame for NVIDIA multimodal embeddings")
                image_bytes = encoded.tobytes()
                if len(image_bytes) <= 180_000:
                    break
                height, width = image.shape[:2]
                if min(height, width) <= 96:
                    raise RuntimeError("Frame exceeds NVIDIA's embedding image size limit")
                image = cv2.resize(image, (max(1, round(width * 0.75)), max(1, round(height * 0.75))),
                                   interpolation=cv2.INTER_AREA)
                quality = 55
            image_path = output / f"{sample.pts:.3f}.jpg"
            image_path.write_bytes(image_bytes)
            data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
            records.append((float(sample.pts), str(image_path), data_url))
            if len(records) == 32:
                flush_batch()
        flush_batch()

        with db.connection:
            db.connection.execute("DELETE FROM semantic_frames WHERE video_id=?", (video_id,))
            db.connection.executemany(
                "INSERT INTO semantic_frames(video_id,pts,image_path,embedding_json) VALUES(?,?,?,?)",
                [(video_id, pts, image_path, json.dumps(vector))
                 for pts, image_path, vector in embedded_records])
        return len(embedded_records)

    def ground_text(self, phrase: str, db: EvidenceDB, video_id: str,
                    threshold: float = 0.22) -> list[dict[str, Any]]:
        rows = db.rows("SELECT pts,embedding_json FROM semantic_frames WHERE video_id=? ORDER BY pts", (video_id,))
        if not rows:
            return []
        if self.provider == "nvidia":
            text_vector = self._normalized(self._embed_nvidia([phrase], modality="text")[0])
        else:
            if not self._load():
                return []
            inputs = self.processor(text=[phrase], return_tensors="pt", padding=True)
            with self.torch.no_grad():
                text = self.model.get_text_features(**inputs)[0]
                text = text / text.norm()
            text_vector = text.cpu().numpy()
        scores = []
        for row in rows:
            if not row["embedding_json"]:
                continue
            image_vector = np.asarray(json.loads(row["embedding_json"]), dtype=np.float32)
            if image_vector.shape != text_vector.shape:
                continue
            image_norm = float(np.linalg.norm(image_vector))
            if image_norm == 0:
                continue
            score = float(np.dot(text_vector, image_vector / image_norm))
            scores.append((score, float(row["pts"])))
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
