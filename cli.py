"""Command-line entry points for indexing, question answering, and evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
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
        print(f"Indexed {sample_count} sampled frames to {db_path}")
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
    config = read_config(config_path)
    video_id, db_path = index_video(video_path, config_path, zone_path=zone_path)
    db = EvidenceDB(db_path)
    try:
        row = db.rows("SELECT duration FROM videos WHERE video_id=?", (video_id,))[0]
        tracks = db.rows("SELECT * FROM tracks WHERE video_id=?", (video_id,))
        merges = merge_tracks([{"track_id": item["track_id"], "class": item["class"],
                                "shot_id": item["shot_id"], "t_start": item["t_start"],
                                "t_end": item["t_end"],
                                "embedding": json.loads(item["embedding_json"] or "null")}
                               for item in tracks],
                              float(config["reid"]["merge_similarity"]),
                              float(config["reid"]["max_exit_reentry_seconds"]))
        from executor import QueryExecutor
        from vlm_client import choose_vlm
        vlm = choose_vlm(backend=config.get("vlm", {}).get("backend", "auto"),
                         model_name=config.get("vlm", {}).get("local_model"))
        answer = QueryExecutor(db, video_id, video_path, float(row["duration"]),
                               _project_path(config["indexing"]["cache_dir"]), merges, vlm=vlm,
                               before_after_seconds=float(config["query"]["before_after_seconds"]),
                               semantic_provider=config.get("semantic", {}).get("provider")).ask(question)
        return answer
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
        else:
            print(json.dumps(evaluate(args.gt_json, args.config), indent=2))
    except Exception as exc:
        logging.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
