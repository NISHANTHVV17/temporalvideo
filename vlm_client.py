"""Pluggable local/hosted VLM clients with short-window-only inputs."""
from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import tempfile
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"), override=False)


class VLMClient(Protocol):
    def answer(self, question: str, context: dict[str, Any], video_path: str,
               start: float, end: float) -> dict[str, Any]: ...

    def plan(self, question: str) -> dict[str, Any]: ...

    def summarize(self, video_path: str, duration: float) -> list[dict[str, str]]: ...


def _summary_windows(duration: float, window_seconds: float,
                     overlap_seconds: float) -> list[tuple[float, float]]:
    if duration <= 0 or window_seconds <= 0 or not 0 <= overlap_seconds < window_seconds:
        raise ValueError("Video duration and summary window must be positive")
    windows = []
    start = 0.0
    step = window_seconds - overlap_seconds
    while start < duration:
        end = min(duration, start + window_seconds)
        windows.append((start, end))
        if end >= duration:
            break
        start += step
    return windows


def _summary_prompt(frame_pts: list[float], start: float, end: float) -> str:
    return (
        "Summarize only events visibly supported by these video frames. "
        "Do not infer causes, unseen actions, object identity, or what happens after the last frame. "
        "Describe observable changes in temporal order; qualify uncertainty when the frames are ambiguous. "
        "Return a JSON object with exactly one key, events, containing an array of objects with "
        "start_seconds, end_seconds, and description. Use absolute source-video PTS seconds; all event "
        f"times must be within this window: {start:.3f} to {end:.3f}. "
        "Use the visible transition interval, not the whole window, for each event. Return an empty "
        "events array if no event can be established. Frame PTS in order: "
        + ", ".join(f"{pts:.3f}" for pts in frame_pts)
    )


def _parse_summary_events(content: str, window_start: float, window_end: float,
                          duration: float) -> list[dict[str, str]]:
    result = _parse_json_response(content)
    events = result.get("events")
    if not isinstance(events, list):
        raise ValueError("VLM summary response must contain an events array")
    parsed = []
    for item in events:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start_seconds"])
            end = float(item["end_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        description = str(item.get("description") or "").strip()
        if not description or not math.isfinite(start) or not math.isfinite(end) or end < start:
            continue
        if start < window_start - 0.5 or end > window_end + 0.5:
            continue
        start = max(window_start, min(start, duration))
        end = max(start, min(end, window_end, duration))
        parsed.append({"time": f"{start:.2f}-{end:.2f} s", "event": description})
    return parsed


class GeminiVideoSummaryClient:
    def __init__(self, model: str = "gemini-3.8-flash"):
        self.api_key = os.getenv("GEMINI_KEY")
        self.model = model

    def summarize(self, video_path: str, duration: float) -> list[dict[str, str]]:
        if not self.api_key:
            raise RuntimeError("GEMINI_KEY is not configured")
        video_bytes = Path(video_path).read_bytes()
        prompt = (
            "Analyze the supplied video and list its notable visible events in chronological order. "
            "Return only JSON matching the requested schema. Use approximate start_seconds and "
            "end_seconds measured from the beginning of the video. Describe only what is visible; "
            "do not invent object details, causes, or events beyond the clip. If timing or interpretation "
            "is uncertain, say so in the event description. Video duration is approximately "
            f"{duration:.3f} seconds."
        )
        request_body = {
            "contents": [{"parts": [
                {"inlineData": {
                    "mimeType": "video/mp4",
                    "data": base64.b64encode(video_bytes).decode("ascii"),
                }, "videoMetadata": {"fps": 4.0}},
                {"text": prompt},
            ]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {"events": {
                        "type": "ARRAY",
                        "items": {"type": "OBJECT", "properties": {
                            "start_seconds": {"type": "NUMBER"},
                            "end_seconds": {"type": "NUMBER"},
                            "description": {"type": "STRING"},
                        }, "required": ["start_seconds", "end_seconds", "description"]},
                    }},
                    "required": ["events"],
                },
                "maxOutputTokens": 4096,
            },
        }
        models = [self.model]
        if self.model == "gemini-3.8-flash":
            models.append("gemini-3.7-flash")
        payload = None
        for model in models:
            url = ("https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{model}:generateContent")
            request = urllib.request.Request(
                url,
                data=json.dumps(request_body).encode("utf-8"),
                headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code == 503 and model != models[-1]:
                    continue
                raise RuntimeError(f"Gemini API returned HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Gemini API request failed: {exc.reason}") from exc
        if payload is None:
            raise RuntimeError("Gemini did not return a response")
        candidates = payload.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        response_text = "".join(part.get("text", "") for part in parts)
        if not response_text.strip():
            raise RuntimeError("Gemini returned no summary text")
        try:
            return _parse_summary_events(response_text, 0.0, duration, duration)
        except (json.JSONDecodeError, ValueError) as exc:
            excerpt = response_text[:300].replace("\n", " ")
            raise RuntimeError(f"Gemini returned an invalid event summary: {exc}; response={excerpt!r}") from exc


def _timestamp_seconds(value: str) -> float:
    parts = value.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Invalid event timestamp: {value}")


def _parse_timestamped_event_lines(content: str, duration: float) -> list[dict[str, str]]:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.I | re.S)
    content = re.sub(r"</?answer>", "", content, flags=re.I).strip()
    if content.casefold() in {"no_events", "no events", "no notable events"}:
        return []

    time_token = r"(?:\d+(?::\d{2}){1,2}(?:\.\d+)?|\d+(?:\.\d+)?)"
    event_line = re.compile(
        rf"^\s*(?:(?:[-*]\s*|\d+[.)]\s+))?(?P<start>{time_token})\s*"
        rf"(?:-|–|—|to)\s*(?P<end>{time_token})\s*(?:s|sec(?:onds)?)?\s*"
        rf"(?:\||\t|:)\s*(?P<description>.+?)\s*$", re.I)
    parsed = []
    for line in content.splitlines():
        match = event_line.match(line)
        if match is None:
            continue
        start = _timestamp_seconds(match.group("start"))
        end = _timestamp_seconds(match.group("end"))
        description = match.group("description").strip().strip("| ")
        if (not description or not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end < start or start > duration + 1 or end > duration + 1):
            continue
        start = min(start, duration)
        end = min(end, duration)
        parsed.append({"time": f"{start:.2f}-{end:.2f} s", "event": description})
    if not parsed and content:
        raise ValueError("Cosmos response contained no timestamped event lines")
    return sorted(parsed, key=lambda item: float(item["time"].split("-", 1)[0]))


class TwelveLabsVideoSummaryClient:
    def __init__(self, model: str = "pegasus1.6"):
        self.api_key = os.getenv("TWELVELABS_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
        self.model = model

    def summarize(self, video_path: str, duration: float) -> list[dict[str, str]]:
        if not self.api_key:
            raise RuntimeError("TWELVELABS_API_KEY is not configured")
        video_bytes = Path(video_path).read_bytes()
        if len(video_bytes) > 30 * 1024 * 1024:
            raise ValueError("TwelveLabs inline video input must be 30 MB or smaller")
        prompt = (
            "List the notable events with approximate timestamps. Return the events in chronological "
            "order, one per line as start_seconds-end_seconds | description. Use decimal seconds from "
            "the start of the video. Describe only events supported by this video; qualify uncertain "
            "details and do not infer causes. The video is approximately "
            f"{duration:.3f} seconds long."
        )
        request_body = {
            "model_name": self.model,
            "video": {
                "type": "base64_string",
                "base64_string": base64.b64encode(video_bytes).decode("ascii"),
            },
            "prompt": prompt,
            "stream": False,
            "temperature": 0.2,
            "max_tokens": 2048,
        }
        request = urllib.request.Request(
            "https://api.twelvelabs.io/v1.3/analyze",
            data=json.dumps(request_body).encode("utf-8"),
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"TwelveLabs API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"TwelveLabs API request failed: {exc.reason}") from exc
        text = str(result.get("data") or "").strip()
        if not text:
            raise RuntimeError("TwelveLabs returned an empty video summary")
        if result.get("finish_reason") == "length":
            raise RuntimeError("TwelveLabs summary was truncated; increase the output token limit")
        try:
            return _parse_timestamped_event_lines(text, duration)
        except ValueError as exc:
            excerpt = text[:300].replace("\n", " ")
            raise RuntimeError(f"TwelveLabs returned invalid timestamped events: {exc}; response={excerpt!r}") from exc

    def answer_question(self, video_path: str, question: str,
                        duration: float) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("TWELVELABS_API_KEY is not configured")
        video_bytes = Path(video_path).read_bytes()
        if len(video_bytes) > 30 * 1024 * 1024:
            raise ValueError("TwelveLabs inline video input must be 30 MB or smaller")
        prompt = (
            "Answer the user's question using only evidence from this video. Be direct and concise. "
            "When the question says 'normally' or implies a general rule, do not generalize from one clip; "
            "state what this clip shows and qualify the limitation. Give the absolute video interval that "
            "supports the answer, in seconds from the beginning of the clip. For duration questions, report "
            "the elapsed duration in the answer while the timestamp fields delimit the supporting interval. "
            "Describe temporal sequence and do not claim causation. Return only a JSON object with keys "
            "answer, timestamp_start, timestamp_end, confidence, and uncertainty_seconds. Confidence must "
            "be between 0 and 1; uncertainty_seconds must be nonnegative. The clip duration is "
            f"{duration:.3f} seconds. User question: {question}"
        )
        request_body = {
            "model_name": self.model,
            "video": {
                "type": "base64_string",
                "base64_string": base64.b64encode(video_bytes).decode("ascii"),
            },
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1,
            "max_tokens": 1024,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "timestamp_start": {"type": "number"},
                        "timestamp_end": {"type": "number"},
                        "confidence": {"type": "number"},
                        "uncertainty_seconds": {"type": "number"},
                    },
                    "required": ["answer", "timestamp_start", "timestamp_end",
                                 "confidence", "uncertainty_seconds"],
                },
            },
        }
        request = urllib.request.Request(
            "https://api.twelvelabs.io/v1.3/analyze",
            data=json.dumps(request_body).encode("utf-8"),
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"TwelveLabs API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"TwelveLabs API request failed: {exc.reason}") from exc
        if result.get("finish_reason") == "length":
            raise RuntimeError("TwelveLabs answer was truncated")
        try:
            answer = json.loads(str(result.get("data") or ""))
            start = float(answer["timestamp_start"])
            end = float(answer["timestamp_end"])
            confidence = float(answer["confidence"])
            uncertainty = float(answer["uncertainty_seconds"])
            text = str(answer["answer"]).strip()
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("TwelveLabs returned an invalid structured answer") from exc
        if (not text or not all(math.isfinite(value) for value in (start, end, confidence, uncertainty))
                or start < 0 or end <= start or end > duration or not 0 <= confidence <= 1
                or uncertainty < 0):
            raise RuntimeError("TwelveLabs answer failed timestamp or confidence validation")
        return {"answer": text, "timestamp_start": start, "timestamp_end": end,
                "confidence": confidence, "uncertainty_seconds": uncertainty}


def _merge_summary_events(events: list[dict[str, str]]) -> list[dict[str, str]]:
    ordered = sorted(events, key=lambda item: float(item["time"].split("-", 1)[0]))
    merged: list[dict[str, str]] = []
    for event in ordered:
        start_text, end_text = event["time"].removesuffix(" s").split("-", 1)
        start, end = float(start_text), float(end_text)
        duplicate = None
        for existing in merged:
            old_start_text, old_end_text = existing["time"].removesuffix(" s").split("-", 1)
            old_start, old_end = float(old_start_text), float(old_end_text)
            same_description = existing["event"].casefold() == event["event"].casefold()
            if same_description and start <= old_end + 0.75 and end >= old_start - 0.75:
                duplicate = existing
                break
        if duplicate is None:
            merged.append(event)
        else:
            old_start_text, old_end_text = duplicate["time"].removesuffix(" s").split("-", 1)
            combined_start = min(float(old_start_text), start)
            combined_end = max(float(old_end_text), end)
            duplicate["time"] = f"{combined_start:.2f}-{combined_end:.2f} s"
    return sorted(merged, key=lambda item: float(item["time"].split("-", 1)[0]))


class HostedOpenAIClient:
    def __init__(self, model: str | None = None, max_payload_bytes: int = 8_000_000,
                 provider: str | None = None):
        self.provider = provider or ("nvidia" if os.getenv("NVIDIA_VIDEO_API_KEY") else "openai")
        if self.provider == "nvidia":
            self.api_key = os.getenv("NVIDIA_VIDEO_API_KEY")
            default_model = "nvidia/cosmos-reason2-8b"
            self.base_url = "https://integrate.api.nvidia.com/v1"
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
            default_model = "gpt-4o-mini"
            self.base_url = None
        self.model = model or os.getenv("TEMPORALVIDEO_VLM_MODEL", default_model)
        self.max_payload_bytes = max_payload_bytes

    def _client(self):
        if not self.api_key:
            variable = "NVIDIA_VIDEO_API_KEY" if self.provider == "nvidia" else "OPENAI_API_KEY"
            raise RuntimeError(f"{variable} is not configured")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install optional package openai to enable hosted VLM") from exc
        options = {"api_key": self.api_key}
        if self.base_url:
            options["base_url"] = self.base_url
        return OpenAI(**options)

    def plan(self, question: str) -> dict[str, Any]:
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
        system_prompt = ("Convert the video question into a generic evidence query plan. "
                         "Do not answer the question. Separate entities on either side of temporal relations. "
                         "Do not encode causal claims; mark asks_causation and request preceding events.")
        if self.provider == "nvidia":
            example = {"intent": "find_event", "entities": [{"phrase": "person", "kind": "person",
                       "class_hint": "person"}], "relations": [],
                       "filters": {"min_duration": None, "zone": None, "time_range": None,
                                   "count_distinct": False},
                       "asks_causation": False, "preceding_seconds": 10.0}
            system_prompt += (" Return one populated JSON plan instance, not a JSON schema. Do not return "
                              "keys named type or properties. Use exactly the top-level keys shown in this "
                              "example and adapt the values to the question: "
                              + json.dumps(example, separators=(",", ":"))
                              + " Entity kind must be object, person, zone, sound, action, or state. "
                              "Intent must be find_event, count, order, before_after, window_before, "
                              "window_after, duration_filter, identify_track, describe, first, or last. "
                              "Use null for an absent time_range; otherwise it must contain two numbers.")
            response_format = {"type": "json_object"}
        else:
            response_format = {"type": "json_schema", "json_schema": {"name": "video_query_plan",
                                                                          "strict": True, "schema": schema}}
        response = self._client().chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": question}],
            response_format=response_format,
            max_tokens=700)
        result = _parse_json_response(response.choices[0].message.content or "{}")
        filters = result.get("filters")
        if isinstance(filters, dict):
            time_range = filters.get("time_range")
            if isinstance(time_range, list) and any(value is None for value in time_range):
                filters["time_range"] = None
        return result

    def answer(self, question: str, context: dict[str, Any], video_path: str,
               start: float, end: float) -> dict[str, Any]:
        frame_data = self._frames(video_path, start, end)
        client = self._client()
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
        try:
            return _parse_json_response(response.choices[0].message.content or "{}")
        except (json.JSONDecodeError, ValueError):
            return {"answer": response.choices[0].message.content or "",
                    "timestamp_start": start, "timestamp_end": end, "confidence": 0.35}

    def summarize(self, video_path: str, duration: float) -> list[dict[str, str]]:
        if self.provider == "nvidia":
            video_bytes = Path(video_path).read_bytes()
            video_url = "data:video/mp4;base64," + base64.b64encode(video_bytes).decode("ascii")
            prompt = (
                "List the notable events with approximate timestamps. For each event, write exactly one "
                "line in this format: start_seconds-end_seconds | event description. Use decimal seconds "
                "from the beginning of the video. Order events chronologically. Describe only what is "
                "visible; do not infer causes or events beyond the video. If no notable event is visible, "
                "return NO_EVENTS. Output no JSON, headings, or extra commentary."
            )
            response = self._client().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": prompt},
                ]}],
                max_tokens=4096,
                extra_body={"media_io_kwargs": {"video": {"fps": 4.0}}},
            )
            response_text = response.choices[0].message.content or ""
            if not response_text.strip():
                raise RuntimeError("Cosmos returned an empty video summary")
            try:
                return _parse_timestamped_event_lines(response_text, duration)
            except ValueError as exc:
                excerpt = response_text[:300].replace("\n", " ")
                raise RuntimeError(f"Cosmos returned an invalid summary: {exc}; response={excerpt!r}") from exc

        client = self._client()
        events = []
        for start, end in _summary_windows(duration, window_seconds=7.0, overlap_seconds=1.0):
            frame_data = self._frames(video_path, start, end)
            if not frame_data:
                continue
            frame_pts = [pts for pts, _ in frame_data]
            content: list[dict[str, Any]] = [{
                "type": "text",
                "text": _summary_prompt(frame_pts, start, end),
            }]
            content.extend({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}
                           for _, data in frame_data)
            response = client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": content}],
                max_tokens=1200,
            )
            response_text = response.choices[0].message.content or ""
            try:
                events.extend(_parse_summary_events(response_text, start, end, duration))
            except (json.JSONDecodeError, ValueError) as exc:
                excerpt = response_text[:300].replace("\n", " ")
                raise RuntimeError(
                    f"VLM returned an invalid summary for {start:.2f}-{end:.2f}s: "
                    f"{exc}; response={excerpt!r}") from exc
        return _merge_summary_events(events)

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


def _parse_json_response(content: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", content, re.S)
    if match:
        content = match.group(0)
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("Hosted model response must be a JSON object")
    return result


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

    def summarize(self, video_path: str, duration: float) -> list[dict[str, str]]:
        self._load()
        try:
            from qwen_vl_utils import process_vision_info
            import av
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install qwen-vl-utils for local Qwen frame preprocessing") from exc
        events = []
        for start, end in _summary_windows(duration, window_seconds=4.0, overlap_seconds=1.0):
            with tempfile.TemporaryDirectory(prefix="temporalvideo-qwen-summary-") as directory:
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
                        image = frame.to_ndarray(format="bgr24")
                        height, width = image.shape[:2]
                        scale = min(1.0, 640 / max(height, width))
                        if scale < 1.0:
                            image = cv2.resize(image, (round(width * scale), round(height * scale)),
                                               interpolation=cv2.INTER_AREA)
                        image_path = Path(directory) / f"frame-{len(image_paths):02d}.jpg"
                        cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        image_paths.append(str(image_path))
                        frame_pts.append(pts)
                        last_pts = pts
                        if len(image_paths) >= 4:
                            break
                if not image_paths:
                    continue
                content = [{"type": "image", "image": image_path} for image_path in image_paths]
                content.append({"type": "text", "text": _summary_prompt(frame_pts, start, end)})
                messages = [{"role": "user", "content": content}]
                prompt = self._processor.apply_chat_template(messages, tokenize=False,
                                                             add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self._processor(text=[prompt], images=image_inputs, videos=video_inputs,
                                         padding=True, return_tensors="pt").to("cpu")
                generated = self._model.generate(**inputs, max_new_tokens=700)
                trimmed = [output[len(input_ids):]
                           for input_ids, output in zip(inputs.input_ids, generated)]
                response = self._processor.batch_decode(trimmed, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)[0]
                events.extend(_parse_summary_events(response, start, end, duration))
        return _merge_summary_events(events)


def choose_vlm(prefer_local: bool = False, backend: str | None = None,
               model_name: str | None = None) -> VLMClient | None:
    selected = backend or os.getenv("TEMPORALVIDEO_VLM_BACKEND", "auto")
    if prefer_local:
        selected = "local"
    if selected == "local":
        return LocalQwenClient(model_name or os.getenv("TEMPORALVIDEO_QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"))
    if selected == "nvidia" and os.getenv("NVIDIA_VIDEO_API_KEY"):
        return HostedOpenAIClient(model=model_name, provider="nvidia")
    if selected == "hosted" and (os.getenv("NVIDIA_VIDEO_API_KEY") or os.getenv("OPENAI_API_KEY")):
        provider = "nvidia" if os.getenv("NVIDIA_VIDEO_API_KEY") else "openai"
        return HostedOpenAIClient(model=model_name, provider=provider)
    if selected == "auto" and (os.getenv("NVIDIA_VIDEO_API_KEY") or os.getenv("OPENAI_API_KEY")):
        provider = "nvidia" if os.getenv("NVIDIA_VIDEO_API_KEY") else "openai"
        return HostedOpenAIClient(model=model_name, provider=provider)
    return None


def choose_summary_client(backend: str, model_name: str | None = None):
    if backend == "twelvelabs":
        return TwelveLabsVideoSummaryClient(model_name or "pegasus1.6")
    if backend == "gemini":
        return GeminiVideoSummaryClient(model_name or "gemini-3.8-flash")
    return choose_vlm(backend=backend, model_name=model_name)
