"""Pluggable local/hosted VLM clients with short-window-only inputs."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Any, Protocol


class VLMClient(Protocol):
    def answer(self, question: str, context: dict[str, Any], video_path: str,
               start: float, end: float) -> dict[str, Any]: ...

    def plan(self, question: str) -> dict[str, Any]: ...


class HostedOpenAIClient:
    def __init__(self, model: str | None = None, max_payload_bytes: int = 8_000_000):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("TEMPORALVIDEO_VLM_MODEL", "gpt-4o-mini")
        self.max_payload_bytes = max_payload_bytes

    def plan(self, question: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install optional package openai to enable hosted planning") from exc
        schema = {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": ["find_event", "count", "order", "before_after",
                    "window_before", "window_after", "duration_filter", "identify_track", "describe", "first", "last"]},
                "entities": {"type": "array", "items": {"type": "object", "properties": {
                    "phrase": {"type": "string"}, "kind": {"type": "string", "enum": ["object", "person", "zone", "sound", "action", "state"]},
                    "class_hint": {"type": ["string", "null"]}}, "required": ["phrase", "kind", "class_hint"], "additionalProperties": False}},
                "relations": {"type": "array", "items": {"type": "object", "properties": {
                    "a": {"type": "string"}, "relation": {"type": "string", "enum": ["after", "before", "during", "within_N_seconds"]},
                    "b": {"type": "string"}, "seconds": {"type": ["number", "null"]}},
                    "required": ["a", "relation", "b", "seconds"], "additionalProperties": False}},
                "filters": {"type": "object", "properties": {
                    "min_duration": {"type": ["number", "null"]}, "zone": {"type": ["string", "null"]},
                    "time_range": {"type": ["array", "null"], "items": {"type": "number"}},
                    "count_distinct": {"type": "boolean"}},
                    "required": ["min_duration", "zone", "time_range", "count_distinct"], "additionalProperties": False},
                "asks_causation": {"type": "boolean"}, "preceding_seconds": {"type": "number"}},
            "required": ["intent", "entities", "relations", "filters", "asks_causation", "preceding_seconds"],
            "additionalProperties": False}
        response = OpenAI(api_key=self.api_key).chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": "Convert the video question into a generic evidence query plan. "
                      "Do not answer the question. Separate entities on either side of temporal relations. "
                      "Do not encode causal claims; mark asks_causation and request preceding events."},
                      {"role": "user", "content": question}],
            response_format={"type": "json_schema", "json_schema": {"name": "video_query_plan",
                                                                        "strict": True, "schema": schema}},
            max_tokens=700)
        return json.loads(response.choices[0].message.content or "{}")

    def answer(self, question: str, context: dict[str, Any], video_path: str,
               start: float, end: float) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install optional package openai to enable hosted VLM") from exc
        frame_data = self._frames(video_path, start, end)
        client = OpenAI(api_key=self.api_key)
        frame_times = ", ".join(f"{pts:.3f}" for pts, _ in frame_data)
        prompt = ("Answer only from these short video frames and structured evidence. "
                  "Do not claim causation; report temporal sequence. Return JSON with keys answer, "
                  "timestamp_start, timestamp_end, confidence, and preceding_ratings. Timestamps must be "
                  "absolute source-video PTS seconds. The following PTS values correspond to the attached "
                  f"images in order: {frame_times}.\n"
                  f"Question: {question}\nEvidence: {json.dumps(context, ensure_ascii=True)[:6000]}")
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for _, data in frame_data:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
        response = client.chat.completions.create(model=self.model, response_format={"type": "json_object"},
                                                   messages=[{"role": "user", "content": content}], max_tokens=700)
        return json.loads(response.choices[0].message.content or "{}")

    def _frames(self, video_path: str, start: float, end: float) -> list[tuple[float, str]]:
        import cv2
        import av
        frames: list[tuple[float, str]] = []
        last_pts = float("-inf")
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            if start > 0:
                container.seek(int(start / float(stream.time_base)), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                pts = float(frame.pts * frame.time_base)
                if pts < start:
                    continue
                if pts > end:
                    break
                if pts - last_pts < 1.0:
                    continue
                image = frame.to_ndarray(format="bgr24")
                height, width = image.shape[:2]
                scale = min(1.0, 640 / max(height, width))
                if scale < 1.0:
                    image = cv2.resize(image, (round(width * scale), round(height * scale)),
                                       interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    frames.append((pts, base64.b64encode(encoded).decode("ascii")))
                    last_pts = pts
        while sum(len(data) for _, data in frames) > self.max_payload_bytes and len(frames) > 2:
            frames = frames[::2]
        return frames[:8]


class LocalQwenClient:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        self.model_name = model_name
        self._model = self._processor = self._torch = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Install transformers and torch for local Qwen VLM") from exc
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_name, local_files_only=True)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype="auto", device_map="cpu", local_files_only=True).eval()
            self._torch = torch
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(f"Local Qwen weights are not available: {self.model_name}") from exc

    def answer(self, question: str, context: dict[str, Any], video_path: str,
               start: float, end: float) -> dict[str, Any]:
        self._load()
        try:
            from qwen_vl_utils import process_vision_info
            import av
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install qwen-vl-utils for local Qwen frame preprocessing") from exc
        with tempfile.TemporaryDirectory(prefix="temporalvideo-qwen-") as directory:
            image_paths = []
            frame_pts = []
            last_pts = float("-inf")
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                if start > 0:
                    container.seek(int(start / float(stream.time_base)), stream=stream, backward=True)
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    pts = float(frame.pts * frame.time_base)
                    if pts < start:
                        continue
                    if pts > end:
                        break
                    if pts - last_pts < 1.0:
                        continue
                    path = Path(directory) / f"frame-{len(image_paths):02d}.jpg"
                    image = frame.to_ndarray(format="bgr24")
                    height, width = image.shape[:2]
                    scale = min(1.0, 640 / max(height, width))
                    if scale < 1.0:
                        image = cv2.resize(image, (round(width * scale), round(height * scale)),
                                           interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(path), image,
                                [cv2.IMWRITE_JPEG_QUALITY, 70])
                    image_paths.append(str(path))
                    frame_pts.append(pts)
                    last_pts = pts
                    if len(image_paths) >= 4:
                        break
            if not image_paths:
                raise RuntimeError("No PTS-valid frames in local VLM candidate window")
            prompt = ("Use only these short-window frames and the supplied structured context. "
                      "Do not claim causation. Return JSON with answer, timestamp_start, timestamp_end, "
                      "confidence, and relation. Timestamps must be absolute source-video PTS seconds. "
                      "Attached images correspond in order to these PTS seconds: " +
                      ", ".join(f"{pts:.3f}" for pts in frame_pts) +
                      ".\nQuestion: " + question +
                      "\nContext: " + json.dumps(context, ensure_ascii=True)[:6000])
            content = [{"type": "image", "image": path} for path in image_paths]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(text=[text], images=image_inputs, videos=video_inputs,
                                     padding=True, return_tensors="pt").to("cpu")
            generated = self._model.generate(**inputs, max_new_tokens=450)
            trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated)]
            response = self._processor.batch_decode(trimmed, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)[0]
            match = re.search(r"\{.*\}", response, re.S)
            if not match:
                return {"answer": response.strip(), "timestamp_start": start,
                        "timestamp_end": end, "confidence": 0.35}
            return json.loads(match.group(0))


def choose_vlm(prefer_local: bool = False, backend: str | None = None,
               model_name: str | None = None) -> VLMClient | None:
    selected = backend or os.getenv("TEMPORALVIDEO_VLM_BACKEND", "auto")
    if prefer_local:
        selected = "local"
    if selected == "local":
        return LocalQwenClient(model_name or os.getenv("TEMPORALVIDEO_QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"))
    if selected == "hosted" and os.getenv("OPENAI_API_KEY"):
        return HostedOpenAIClient()
    if selected == "auto" and os.getenv("OPENAI_API_KEY"):
        return HostedOpenAIClient()
    return None
