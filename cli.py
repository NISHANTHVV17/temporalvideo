"""Command-line entry points for indexing, question answering, and evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import yaml
from dotenv import load_dotenv

from audio import detect_audio_events
from db import EvidenceDB
from detect_track import ContourFallbackDetector, MultiObjectTracker, make_detector
from events import MotionAnalyzer, build_track_events
from ingest import iter_video, probe_video
from reid import TrackAppearanceEmbedder, merge_tracks, serialize_embedding
from schema import Answer, format_timestamp
from zones import Line, Zone, load_lines, load_zones


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)


def read_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else ROOT / "config.yaml"
    with config_path.open(encoding="utf-8") as source:
        return yaml.safe_load(source) or {}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _db_path(config: dict[str, Any], video_id: str) -> Path:
    directory = Path(config["indexing"]["database_dir"])
    if not directory.is_absolute():
        directory = ROOT / directory
    return directory / f"{video_id}.sqlite3"


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def index_video(video_path: str | Path, config_path: str | Path | None = None,
                force: bool = False, zone_path: str | Path | None = None) -> tuple[str, Path]:
    config = read_config(config_path)
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    digest = file_sha256(video_path)
    video_id = digest[:20]
    db_path = _db_path(config, video_id)
    db = EvidenceDB(db_path)
    try:
        semantic_config = config.get("semantic", {})
        semantic_enabled = semantic_config.get("enabled", True)
        from semantic import configured_embedding_provider
        semantic_provider = (configured_embedding_provider(semantic_config.get("provider"))
                     if semantic_enabled else "disabled")
        cached = db.rows("SELECT sha256,metadata_json FROM videos WHERE video_id=?", (video_id,))
        cached_metadata = json.loads(cached[0]["metadata_json"] or "{}") if cached else {}
        complete = bool(cached_metadata.get("complete"))
        provider_changed = semantic_enabled and cached_metadata.get("semantic_provider") != semantic_provider
        if (cached and cached[0]["sha256"] == digest and complete and not force and not zone_path
            and not provider_changed):
            print(f"Index cache hit: {video_id}")
            return video_id, db_path

        info = probe_video(video_path)
        if zone_path:
            zones = load_zones(zone_path)
            lines = load_lines(zone_path)
        else:
            zones, lines = _load_geometry(db, video_id)
        if zone_path:
            db.execute("DELETE FROM zones WHERE video_id=?", (video_id,))
        for zone in zones:
            db.execute("INSERT OR REPLACE INTO zones(zone_id,video_id,name,polygon_json,shot_id) VALUES(?,?,?,?,?)",
                       (f"{video_id}:{zone.name}:{zone.shot_id}", video_id, zone.name,
                        json.dumps(zone.polygon), zone.shot_id))
        for line in lines:
            db.execute("INSERT OR REPLACE INTO zones(zone_id,video_id,name,polygon_json,shot_id) VALUES(?,?,?,?,?)",
                       (f"{video_id}:line:{line.name}:{line.shot_id}", video_id, line.name,
                        json.dumps([line.start, line.end]), line.shot_id))
        db.add_video(video_id, str(video_path), digest, info.duration, info.width, info.height,
                     {"time_base": info.time_base, "audio_streams": info.audio_streams,
                      "complete": False})
        if force:
            db.execute("DELETE FROM events WHERE video_id=?", (video_id,))
            db.execute("DELETE FROM track_points WHERE track_id IN (SELECT track_id FROM tracks WHERE video_id=?)", (video_id,))
            db.execute("DELETE FROM tracks WHERE video_id=?", (video_id,))
            db.execute("DELETE FROM checkpoints WHERE video_id=?", (video_id,))

        checkpoint = db.checkpoint(video_id) if config["indexing"].get("resume", True) and not force else None
        resume_after = float(checkpoint["last_pts"]) if checkpoint else -1.0
        detector_cfg = config["tracking"]
        detector = make_detector(detector_cfg.get("backend", "auto"),
                                 detector_cfg.get("detector_model", "yolo11n.pt"),
                                 float(detector_cfg.get("confidence", 0.25)))
        tracker = MultiObjectTracker(float(detector_cfg.get("max_track_gap_seconds", 2.0)))
        embedder = TrackAppearanceEmbedder()
        observations = []
        motion = MotionAnalyzer()
        roi_motions = {zone.name: MotionAnalyzer() for zone in zones}
        motion_events = []
        sample_count = 0
        max_pts = 0.0
        cache_root = _project_path(config["indexing"]["cache_dir"])
        video_cache = cache_root / video_id
        video_cache.mkdir(parents=True, exist_ok=True)
        embeddings_by_track: dict[str, list[float]] = {}
        for sample in iter_video(video_path, float(config["indexing"]["sample_fps"]),
                                 int(config["indexing"]["max_dimension"]),
                                 float(config["indexing"]["scene_cut_threshold"])):
            sample_count += 1
            max_pts = max(max_pts, sample.pts)
            if sample.shot_cut:
                motion = MotionAnalyzer()
                roi_motions = {zone.name: MotionAnalyzer() for zone in zones}
                reset_detector = getattr(detector, "reset", None)
                if reset_detector is not None:
                    reset_detector()
            try:
                detected = detector.detect(sample.frame)
            except Exception as exc:
                logging.warning("Detector failed during tracking; switching to generic proposals: %s", exc)
                detector = ContourFallbackDetector()
                detected = detector.detect(sample.frame)
            current = tracker.update(sample, detected)
            if sample.pts > resume_after:
                for item in current:
                    embedding = item.embedding or embeddings_by_track.get(item.track_id)
                    if embedding is None:
                        embedding = embedder.embed(sample.frame, item.box)
                        embeddings_by_track[item.track_id] = embedding
                    item.embedding = embedding
                    x, y, width, height = item.box
                    db.execute("""INSERT INTO tracks(track_id,video_id,local_id,shot_id,class,t_start,t_end,embedding_json)
                      VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(track_id) DO UPDATE SET
                      t_start=MIN(tracks.t_start,excluded.t_start),t_end=MAX(tracks.t_end,excluded.t_end),
                      embedding_json=COALESCE(tracks.embedding_json,excluded.embedding_json)""",
                      (item.track_id, video_id, item.local_id, item.shot_id, item.class_name,
                       item.pts, item.pts, serialize_embedding(embedding)))
                    db.execute("""INSERT OR REPLACE INTO track_points
                      (track_id,pts,x,y,w,h,confidence,stabilized_json) VALUES(?,?,?,?,?,?,?,?)""",
                      (item.track_id, item.pts, x, y, width, height, item.confidence,
                       json.dumps(item.stabilized_center)))
            stabilized = cv2.warpPerspective(sample.frame, sample.to_reference,
                                             (sample.frame.shape[1], sample.frame.shape[0]))
            motion_event = motion.update(stabilized, sample.pts, video_id, db)
            if motion_event is not None:
                motion_events.append(motion_event)
            for zone in zones:
                if zone.shot_id is not None and zone.shot_id != sample.shot_id:
                    continue
                roi_event = roi_motions[zone.name].update(stabilized, sample.pts, video_id, db,
                                                          zone.polygon, zone.name)
                if roi_event is not None:
                    motion_events.append(roi_event)
            if sample.pts > resume_after:
                db.set_checkpoint(video_id, sample.pts, sample_count, sample.shot_id)
            if sample_count % 300 == 0:
                print(f"Indexed {sample_count} sampled frames; PTS={sample.pts:.2f}s")

        observations = _load_observations(db, video_id)
        db.execute("DELETE FROM events WHERE video_id=?", (video_id,))
        build_track_events(video_id, observations, db, zones,
                           float(config["events"]["stationary_epsilon"]),
                           float(config["events"]["stationary_seconds"]),
                           float(config["events"]["interaction_distance"]), lines)
        for event in motion_events:
            db.add_event(event)
        if info.audio_streams and config["audio"].get("enabled", True):
            detect_audio_events(video_path, video_id, db, config["audio"].get("ffmpeg", "ffmpeg"))
        duration = info.duration or max_pts
        db.execute("UPDATE videos SET duration=? WHERE video_id=?", (duration, video_id))
        if semantic_enabled:
            try:
                from semantic import ClipSemanticIndex
                created = ClipSemanticIndex(semantic_config["model"], semantic_config.get("device", "cpu"),
                                            semantic_config.get("provider"))
                count = created.index_video(video_path, video_id, db, cache_root,
                                            float(semantic_config.get("embedding_fps", 1.0)))
                if count:
                    print(f"Stored {count} semantic frame embeddings")
            except Exception as exc:
                logging.warning("Optional semantic indexing skipped: %s", exc)
        _run_global_reid(db, video_id, config)
        db.execute("UPDATE videos SET metadata_json=? WHERE video_id=?",
               (json.dumps({"time_base": info.time_base, "audio_streams": info.audio_streams,
                    "complete": True, "semantic_provider": semantic_provider}), video_id))
        log_path = db.write_event_log(video_id)
        print(f"Indexed {sample_count} sampled frames to {db_path}")
        print(f"Saved event log to {log_path}")
        return video_id, db_path
    finally:
        db.close()


def _load_observations(db: EvidenceDB, video_id: str):
    from detect_track import TrackObservation
    import json as json_module
    rows = db.rows("""SELECT p.*,t.local_id,t.shot_id,t.class FROM track_points p
                     JOIN tracks t ON p.track_id=t.track_id WHERE t.video_id=? ORDER BY p.pts""", (video_id,))
    return [TrackObservation(track_id=row["track_id"], local_id=row["local_id"], shot_id=row["shot_id"],
                             pts=row["pts"], class_name=row["class"],
                             box=(row["x"], row["y"], row["w"], row["h"]),
                             confidence=row["confidence"], embedding=None,
                             stabilized_center=tuple(json_module.loads(row["stabilized_json"] or "[0,0]")))
            for row in rows]


def _load_geometry(db: EvidenceDB, video_id: str) -> tuple[list[Zone], list[Line]]:
    zones, lines = [], []
    for row in db.rows("SELECT name,polygon_json,shot_id FROM zones WHERE video_id=?", (video_id,)):
        points = json.loads(row["polygon_json"])
        if len(points) >= 3:
            zones.append(Zone(row["name"], [tuple(point) for point in points], row["shot_id"]))
        elif len(points) == 2:
            lines.append(Line(row["name"], tuple(points[0]), tuple(points[1]), row["shot_id"]))
    return zones, lines


def _run_global_reid(db: EvidenceDB, video_id: str, config: dict[str, Any]):
    rows = db.rows("SELECT * FROM tracks WHERE video_id=? ORDER BY t_start", (video_id,))
    tracks = []
    for row in rows:
        try:
            embedding = json.loads(row["embedding_json"]) if row["embedding_json"] else None
        except json.JSONDecodeError:
            embedding = None
        tracks.append({"track_id": row["track_id"], "class": row["class"], "shot_id": row["shot_id"],
                       "t_start": row["t_start"], "t_end": row["t_end"], "embedding": embedding})
    merges = merge_tracks(tracks, float(config["reid"]["merge_similarity"]),
                          float(config["reid"]["max_exit_reentry_seconds"]))
    db.execute("DELETE FROM track_merges WHERE video_id=?", (video_id,))
    db.execute("UPDATE tracks SET merged_into=NULL WHERE video_id=?", (video_id,))
    for merge in merges:
        db.execute("INSERT INTO track_merges(video_id,source_track_id,target_track_id,score,crossed_cut,reason) VALUES(?,?,?,?,?,?)",
                   (video_id, merge.source_track_id, merge.target_track_id, merge.score,
                    int(merge.crossed_cut), merge.reason))
        db.execute("UPDATE tracks SET merged_into=? WHERE track_id=?",
                   (merge.target_track_id, merge.source_track_id))


def ask_video(video_path: str | Path, question: str, config_path: str | Path | None = None,
              zone_path: str | Path | None = None) -> Answer:
    video_path = Path(video_path).resolve()
    video_id = file_sha256(video_path)[:20]
    config = read_config(config_path)
    timeline_path = summary_output_path(video_path, config_path)
    if not timeline_path.is_file():
        raise FileNotFoundError(
            f"No analyzed event summary exists for this video. Analyze it first; expected {timeline_path}")
    data_path = summary_data_path(video_path, config_path)
    entries, duration = _read_summary_evidence(timeline_path, data_path, video_id)
    if not entries:
        raise ValueError("The saved summary has no timestamped events to answer from")
    duration = max(duration, max(entry["end"] for entry in entries))
    llm_answer = _answer_from_log_llm(question, entries, duration)
    if llm_answer is not None:
        return llm_answer
    structured_answer = _answer_structured_log_query(question, entries, video_id, duration)
    if structured_answer is not None:
        return structured_answer
    question_lower = question.lower()
    if re.search(r"\bhow long\b|\bduration\b|\bfor how much time\b", question_lower):
        pre_fall_question = bool(re.search(r"\b(before|until|prior to|without)\b", question_lower)
                                 or re.search(r"\bon (?:the )?plank\b", question_lower))
        fall_index = next((index for index, entry in enumerate(entries)
                           if re.search(r"\b(fall|falls|falling|fell|drop|dropped|tip|tipping)\b",
                                        entry["description"].lower())), None)
        if pre_fall_question and fall_index is not None and fall_index > 0:
            prior = entries[fall_index - 1]
            fall_event = entries[fall_index]
            duration_seconds = max(0.0, fall_event["start"] - prior["start"])
            if duration_seconds > 0:
                answer_text = (
                    f"In this clip, {prior['description'].rstrip('.')} for approximately "
                    f"{duration_seconds:g} seconds before the fall begins. "
                    "One video cannot establish what normally happens."
                )
                return Answer(
                    answer=answer_text, t_start=prior["start"], t_end=fall_event["start"],
                    event_ids=[prior["event_id"], fall_event["event_id"]],
                    confidence=0.8, video_duration=duration, timestamp_source="rule",
                    uncertainty_seconds=max(0.5, (prior["end"] - prior["start"]) * 0.15),
                )
        query_words = _log_query_words(question_lower)
        target = _best_log_entry(query_words, entries)
        elapsed = target["end"] - target["start"]
        return Answer(
            answer=f"According to the analyzed log, {target['description']} This interval lasts approximately {elapsed:g} seconds.",
            t_start=target["start"], t_end=target["end"], event_ids=[target["event_id"]],
            confidence=0.65, video_duration=duration, timestamp_source="rule",
            uncertainty_seconds=max(0.5, elapsed * 0.15),
        )
    query_words = _log_query_words(question_lower)
    target = _best_log_entry(query_words, entries)
    return Answer(
        answer=f"According to the analyzed log, {target['description']}",
        t_start=target["start"], t_end=target["end"], event_ids=[target["event_id"]],
        confidence=0.65, video_duration=duration, timestamp_source="rule",
        uncertainty_seconds=max(0.5, (target["end"] - target["start"]) * 0.15),
    )


def _answer_from_log_llm(question: str, entries: list[dict[str, Any]],
                         duration: float) -> Answer | None:
    try:
        from vlm_client import choose_log_llm
        client = choose_log_llm()
        if client is None or not hasattr(client, "answer_log"):
            return None
        result = client.answer_log(question, entries, duration)
        return Answer(answer=str(result["answer"]),
                      t_start=float(result["timestamp_start"]),
                      t_end=float(result["timestamp_end"]),
                      event_ids=[str(event_id) for event_id in result["event_ids"]],
                      confidence=float(result["confidence"]), video_duration=duration,
                      timestamp_source="vlm",
                      uncertainty_seconds=float(result["uncertainty_seconds"]))
    except Exception:
        logging.warning("Log-grounded LLM answer unavailable; using deterministic fallback",
                        exc_info=True)
        return None


def _read_summary_evidence(timeline_path: Path, data_path: Path,
                           video_id: str) -> tuple[list[dict[str, Any]], float]:
    entries: list[dict[str, Any]] = []
    duration = 0.0
    if data_path.is_file():
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        duration = float(payload.get("video_duration") or 0.0)
        raw_events = payload.get("events", [])
        for index, event in enumerate(raw_events):
            if not isinstance(event, dict):
                continue
            try:
                start = float(event.get("start_seconds"))
                end = float(event.get("end_seconds"))
            except (TypeError, ValueError):
                time_match = re.fullmatch(
                    r"\s*(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s*s\s*", str(event.get("time", "")))
                if not time_match:
                    continue
                start, end = float(time_match.group(1)), float(time_match.group(2))
            if start < 0 or end <= start:
                continue
            entities = event.get("entities", [])
            if not isinstance(entities, list):
                entities = [str(entities)] if entities else []
            entries.append({
                "index": index,
                "start": start,
                "end": end,
                "description": str(event.get("event") or event.get("description") or "").strip(),
                "event_type": str(event.get("event_type") or "event").lower().replace(" ", "_"),
                "entities": [str(entity).strip() for entity in entities if str(entity).strip()],
                "location": str(event.get("location") or "").strip(),
                "event_id": str(event.get("event_id") or f"summary-{video_id}-{index}"),
            })
        return entries, duration

    for index, line in enumerate(timeline_path.read_text(encoding="utf-8").splitlines()):
        columns = line.split("\t", 1)
        if len(columns) != 2:
            continue
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s*s\s*", columns[0])
        if match is None:
            continue
        start, end = float(match.group(1)), float(match.group(2))
        if end <= start:
            continue
        entries.append({"index": index, "start": start, "end": end,
                        "description": columns[1].strip(), "event_type": "event",
                        "entities": [], "location": "",
                        "event_id": f"summary-{video_id}-{index}"})
    return entries, max((entry["end"] for entry in entries), default=0.0)


def _answer_structured_log_query(question: str, entries: list[dict[str, Any]],
                                 video_id: str, duration: float) -> Answer | None:
    lowered = question.lower()
    entities_text = lambda entry: " ".join(entry["entities"])
    combined = lambda entry: f"{entry['event_type']} {entry['description']} {entities_text(entry)} {entry['location']}".lower()

    if (re.search(r"\bwhich\b", lowered) and re.search(r"\b(person|who)\b", lowered)
            and re.search(r"\b(enter|entered|enters)\b", lowered)
            and re.search(r"\bafter\b", lowered)
            and re.search(r"\b(arriv|arrived|arrival)\b", lowered)):
        truck_terms = {word for word in _log_query_words(lowered) if word not in {"person", "enter", "area"}}
        arrivals = [entry for entry in entries
                    if re.search(r"arriv|arrives|arrived|arrival", combined(entry))
                    and (not truck_terms or truck_terms & _normalized_words(combined(entry)))]
        area_terms = {word for word in _log_query_words(lowered)
                      if word not in {"person", "enter", "arriv", "truck", "delivery"}}
        entries_into_area = [entry for entry in entries
                             if (entry["event_type"] in {"zone_entry", "area_entry", "entry"}
                                 or re.search(r"\b(enter|entered|enters)\b", combined(entry)))
                             and re.search(r"person|human", combined(entry))]
        if area_terms:
            entries_into_area = [entry for entry in entries_into_area
                                 if area_terms & _normalized_words(combined(entry))]
        if not arrivals or not entries_into_area:
            raise ValueError("The saved event log does not record both the truck arrival and a person's area entry")
        arrival = min(arrivals, key=lambda item: item["start"])
        later_entries = [entry for entry in entries_into_area if entry["start"] >= arrival["start"]]
        if not later_entries:
            raise ValueError("The saved event log does not show a person entering after the truck arrived")
        entry = min(later_entries, key=lambda item: item["start"])
        person = next((entity for entity in entry["entities"]
                       if re.search(r"person|human", entity, re.I)), "the person recorded in the log")
        area = entry["location"] or "the area named in the event log"
        return _make_log_answer(
            f"The saved log identifies {person} entering {area} after the delivery truck arrived.",
            [arrival, entry], video_id, duration, confidence=0.8)

    if re.search(r"\bhow many\b", lowered) and re.search(r"\bstop|stopped|stops\b", lowered):
        stops = [entry for entry in entries
                 if (entry["event_type"] in {"stop", "machine_stop", "stopped"}
                     or re.search(r"\b(stop|stops|stopped|stopping)\b", combined(entry)))]
        machine_terms = {word for word in _log_query_words(lowered)
                         if word not in {"machine", "unexpected", "unexpectedly", "times"}}
        if machine_terms:
            stops = [entry for entry in stops if machine_terms & _normalized_words(combined(entry))]
        if not stops:
            return _make_log_answer(
                "The saved event log records no matching machine-stop events; it cannot establish whether unlogged stops occurred.",
                entries, video_id, duration, confidence=0.45)
        unexpectedly = "unexpected" in lowered
        note = " The log does not establish whether they were unexpected." if unexpectedly else ""
        return _make_log_answer(
            f"The saved event log records {len(stops)} matching machine stop(s).{note}",
            stops, video_id, duration, confidence=0.8 if not unexpectedly else 0.65)

    if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", lowered):
        query_terms = _log_query_words(lowered)
        action_terms = query_terms & {
            "fall", "fell", "drop", "dropped", "tip", "tipping", "enter", "entered",
            "exit", "exited", "arrive", "arrived", "stop", "stopped", "start", "started",
            "move", "moved", "appear", "appeared", "alarm", "sound"
        }
        matches = [entry for entry in entries
                   if ((action_terms and action_terms & _normalized_words(combined(entry)))
                       or (not action_terms and query_terms & _normalized_words(combined(entry))))]
        if not matches:
            return _make_log_answer(
                "The saved event log records 0 matching events; it cannot establish whether unlogged events occurred.",
                entries, video_id, duration, confidence=0.45)
        return _make_log_answer(
            f"The saved event log records {len(matches)} matching event(s).",
            matches, video_id, duration, confidence=0.75)

    if re.search(r"\bwhat happened\b", lowered) and re.search(r"\bright before\b", lowered) \
            and re.search(r"\balarm\b", lowered):
        alarms = [entry for entry in entries if re.search(r"alarm", combined(entry))]
        if not alarms:
            raise ValueError("The saved event log does not contain a safety-alarm event")
        alarm = min(alarms, key=lambda item: item["start"])
        preceding = [entry for entry in entries if entry["end"] <= alarm["start"]
                     and entry["event_id"] != alarm["event_id"]]
        if not preceding:
            raise ValueError("The saved event log contains no recorded event before the alarm")
        event = max(preceding, key=lambda item: item["end"])
        return _make_log_answer(
            f"Right before the safety alarm, the log records: {event['description']}",
            [event, alarm], video_id, duration, confidence=0.8)

    if re.search(r"\b(find|list)\b", lowered) and re.search(r"\b(untouched|stationary|sat)\b", lowered):
        threshold = _stationary_threshold_seconds(lowered)
        stationary = [entry for entry in entries
                      if (entry["event_type"] in {"stationary", "untouched", "dwell"}
                          or re.search(r"\b(stationary|untouched|sat|sits|remained still)\b", combined(entry)))
                      and entry["end"] - entry["start"] > threshold]
        objects = []
        for entry in stationary:
            people = [entity for entity in entry["entities"]
                      if re.search(r"person|human|man|woman|child", entity, re.I)]
            candidates = [entity for entity in entry["entities"] if entity not in people]
            objects.extend(candidates)
        objects = list(dict.fromkeys(objects))
        if not objects:
            return _make_log_answer(
                f"The saved event log records no identified object stationary for more than {threshold:g} seconds. "
                "This means none was recorded, not that none existed.",
                stationary or entries, video_id, duration, confidence=0.5)
        return _make_log_answer(
            f"Objects recorded as stationary for more than {threshold:g} seconds: {', '.join(objects)}.",
            stationary, video_id, duration, confidence=0.75)
    return None


def _stationary_threshold_seconds(question: str) -> float:
    match = re.search(r"more than\s+(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\b", question)
    if not match:
        return 0.0
    amount = float(match.group(1))
    unit = match.group(2)
    if unit.startswith("h"):
        return amount * 3600
    if unit.startswith("m"):
        return amount * 60
    return amount


def _normalized_words(text: str) -> set[str]:
    return {_normalize_log_word(word) for word in re.findall(r"[a-z0-9]+", text.lower())}


def _make_log_answer(text: str, evidence: list[dict[str, Any]], video_id: str,
                     duration: float, confidence: float) -> Answer:
    if not evidence:
        raise ValueError("The saved event log does not contain evidence for this answer")
    start = min(entry["start"] for entry in evidence)
    end = max(entry["end"] for entry in evidence)
    if end <= start:
        end = min(duration, start + 0.1)
    if end <= start:
        raise ValueError("The saved event interval is too short to provide a timestamp")
    return Answer(
        answer=text, t_start=start, t_end=end,
        event_ids=list(dict.fromkeys(entry["event_id"] for entry in evidence)),
        confidence=confidence, video_duration=duration, timestamp_source="rule",
        uncertainty_seconds=max(0.5, (end - start) * 0.15),
    )


def _log_query_words(question: str) -> set[str]:
    ignored = {"how", "long", "was", "were", "is", "are", "the", "a", "an", "in", "on",
               "at", "to", "of", "for", "before", "after", "when", "what", "happened", "did",
               "does", "do", "this", "that", "it", "normally", "usual", "usually", "without",
               "during", "clip", "video", "show", "shown", "which", "who", "unexpected",
               "unexpectedly", "times", "every", "there"}
    return {_normalize_log_word(word) for word in re.findall(r"[a-z0-9]+", question)
            if word not in ignored}


def _normalize_log_word(word: str) -> str:
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("s"):
        return word[:-1]
    return word


def _best_log_entry(query_words: set[str], entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not query_words:
        raise ValueError("Ask a question that refers to an event described in the analyzed log")
    ranked = []
    for entry in entries:
        event_words = {_normalize_log_word(word)
                   for word in re.findall(r"[a-z0-9]+", entry["description"].lower())}
        overlap = query_words & event_words
        ranked.append((len(overlap) / len(query_words), len(overlap), entry))
    score, overlap, entry = max(ranked, key=lambda item: (item[0], item[1]))
    if overlap == 0 or score < 0.25:
        raise ValueError("The saved event log does not contain enough information to answer that question")
    return entry


def summary_output_path(video_path: str | Path, config_path: str | Path | None = None) -> Path:
    video_id = file_sha256(Path(video_path).resolve())[:20]
    return _db_path(read_config(config_path), video_id).with_name(f"{video_id}.summary.txt")


def summary_data_path(video_path: str | Path, config_path: str | Path | None = None) -> Path:
    return summary_output_path(video_path, config_path).with_suffix(".json")


def summarize_video(video_path: str | Path, config_path: str | Path | None = None,
                    zone_path: str | Path | None = None) -> list[dict[str, str]]:
    config = read_config(config_path)
    from vlm_client import choose_summary_client
    vlm_config = config.get("vlm", {})
    backend = vlm_config.get("summary_backend", "twelvelabs")
    model_name = (vlm_config.get("local_model") if backend == "local"
                  else vlm_config.get("twelvelabs_model", "pegasus1.6") if backend == "twelvelabs"
                  else vlm_config.get("gemini_model", "gemini-3.8-flash") if backend == "gemini"
                  else vlm_config.get("summary_model", "nvidia/cosmos-reason2-8b"))
    vlm = choose_summary_client(backend=backend, model_name=model_name)
    if vlm is None:
        raise RuntimeError("No VLM is configured. Set a hosted VLM API key or configure a local VLM model.")
    video_id, db_path = index_video(video_path, config_path, zone_path=zone_path)
    db = EvidenceDB(db_path)
    try:
        video = db.rows("SELECT duration FROM videos WHERE video_id=?", (video_id,))[0]
        summary = vlm.summarize(str(Path(video_path).resolve()), float(video["duration"]))
        summary_path = db.write_summary_log(video_id, summary, float(video["duration"]))
        print(f"Saved summary to {summary_path}")
        return summary
    finally:
        db.close()


def evaluate(ground_truth_path: str | Path, config_path: str | Path | None = None) -> dict[str, float]:
    records = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    errors, within_one, answer_hits, half_credit = [], 0, 0, 0.0
    for record in records:
        answer = ask_video(record["video"], record["question"], config_path)
        expected_start, expected_end = float(record["t_start"]), float(record["t_end"])
        error = (abs(answer.t_start - expected_start) + abs(answer.t_end - expected_end)) / 2
        errors.append(error)
        on_time = error <= 1.0
        within_one += int(on_time)
        correct = _answer_similarity(answer.answer, str(record.get("answer", ""))) >= 0.5
        answer_hits += int(correct)
        half_credit += 1.0 if correct and on_time else 0.5 if correct else 0.0
    total = max(1, len(records))
    return {"median_timestamp_error_seconds": float(np_median(errors)) if errors else 0.0,
            "within_plus_minus_1s_percent": 100 * within_one / total,
            "answer_accuracy_percent": 100 * answer_hits / total,
            "half_credit_score_percent": 100 * half_credit / total}


def _answer_similarity(actual: str, expected: str) -> float:
    import re
    tokens_a = set(re.findall(r"[a-z0-9]+", actual.lower()))
    tokens_b = set(re.findall(r"[a-z0-9]+", expected.lower()))
    return len(tokens_a & tokens_b) / max(1, len(tokens_b))


def np_median(values: list[float]) -> float:
    values = sorted(values)
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="TemporalVideoQA")
    parser.add_argument("--config", help="YAML config path")
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index", help="build or resume the temporal evidence index")
    index_parser.add_argument("video")
    index_parser.add_argument("--zones", help="JSON zone definitions")
    index_parser.add_argument("--force", action="store_true", help="rebuild a cached index")
    ask_parser = subparsers.add_parser("ask", help="ask a question about a video")
    ask_parser.add_argument("video")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--zones")
    summary_parser = subparsers.add_parser("summarize", help="generate a VLM-based Time/Event timeline")
    summary_parser.add_argument("video")
    eval_parser = subparsers.add_parser("eval", help="evaluate against ground-truth JSON")
    eval_parser.add_argument("gt_json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command == "index":
            video_id, db_path = index_video(args.video, args.config, args.force, args.zones)
            print(json.dumps({"video_id": video_id, "database": str(db_path)}))
        elif args.command == "ask":
            answer = ask_video(args.video, args.question, args.config, args.zones)
            print(answer.render())
            if answer.caveat:
                print(f"Caveat: {answer.caveat}")
            for item in answer.preceding_events:
                print(f"preceding event: {item.description} "
                    f"[{format_timestamp(item.start, answer.video_duration)}-"
                    f"{format_timestamp(item.end, answer.video_duration)}] "
                      f"rating={item.relation_rating}")
        elif args.command == "summarize":
            summary = summarize_video(args.video, args.config)
            print("Time\tEvent")
            for item in summary:
                print(f"{item['time']}\t{item['event']}")
        else:
            print(json.dumps(evaluate(args.gt_json, args.config), indent=2))
    except Exception as exc:
        logging.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
