from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import av
import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from db import EvidenceDB
from detect_track import Detection, MultiObjectTracker
from executor import QueryExecutor
from events import _segments_intersect, build_track_events
from ingest import FrameSample, iter_video
from planner import parse_question, rule_based_plan
from schema import Answer, EntityKind, EntitySpec, Intent, QueryPlan, TimedInterval, format_timestamp
from zones import Zone


class MockedDetector:
    def detect(self, frame):
        mask = cv2.inRange(frame, np.array([0, 0, 180], np.uint8), np.array([80, 80, 255], np.uint8))
        points = cv2.findNonZero(mask)
        if points is None:
            return []
        x, y, width, height = cv2.boundingRect(points)
        return [Detection((x, y, width, height), "test-shape", 0.99)]


def create_synthetic_video(directory: Path) -> tuple[Path, bool]:
    directory.mkdir(parents=True, exist_ok=True)
    silent = directory / "synthetic-silent.mp4"
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    assert writer.isOpened()
    for frame_number in range(60):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:] = (35, 45, 55)
        x = min(120, 12 + max(0, frame_number - 10) * 3) if frame_number < 30 else 72
        cv2.rectangle(frame, (x, 45), (x + 24, 69), (0, 0, 230), -1)
        writer.write(frame)
    writer.release()
    rate = 16000
    samples = np.zeros(rate * 6, dtype=np.int16)
    tone = (np.sin(2 * np.pi * 1000 * np.arange(int(rate * 0.35)) / rate) * 15000).astype(np.int16)
    samples[rate * 4:rate * 4 + len(tone)] = tone
    wav_path = directory / "tone.wav"
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(samples.tobytes())
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return silent, False
    with_audio = directory / "synthetic.mp4"
    result = subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(silent), "-i", str(wav_path),
                             "-c:v", "copy", "-c:a", "aac", "-shortest", str(with_audio)],
                            capture_output=True, check=False)
    return (with_audio, True) if result.returncode == 0 else (silent, False)


def test_answer_requires_timestamp_and_formats_long_video():
    with pytest.raises(ValidationError):
        Answer(answer="missing", confidence=0.1, low_confidence=True,
               timestamp_source="rule", uncertainty_seconds=1)
    with pytest.raises(ValidationError):
        Answer(answer="zero duration", t_start=1, t_end=1, confidence=0.8,
               timestamp_source="rule", uncertainty_seconds=0, event_ids=["event-1"])
    with pytest.raises(ValidationError):
        Answer(answer="The truck caused the alarm", t_start=1, t_end=2, confidence=0.8,
               timestamp_source="rule", uncertainty_seconds=1, event_ids=["event-1"])
    assert format_timestamp(65, 7200) == "00:01:05"
    assert format_timestamp(3661, 7200) == "01:01:01"
    answer = Answer(answer="A shape moved", t_start=3661, t_end=3662,
                    confidence=0.8, timestamp_source="rule", uncertainty_seconds=0.2,
                    video_duration=7200, event_ids=["event-1"])
    assert "[01:01:01-01:01:02]" in answer.render()


def test_planner_handles_counts_and_cause_language():
    plan = rule_based_plan("How many times did the object stop unexpectedly?")
    assert plan.intent.value == "count"
    assert not plan.filters.count_distinct
    assert any("object" in entity.phrase for entity in plan.entities)
    assert rule_based_plan("How many people entered?").filters.count_distinct
    gap = rule_based_plan("How long between the person entering and the sound starting?")
    assert gap.intent.value == "before_after"
    assert len(gap.entities) == 2
    cause_plan = rule_based_plan("What caused the loud sound?")
    assert cause_plan.asks_causation
    relation = rule_based_plan("Which person entered the area after the delivery truck arrived?")
    assert len(relation.entities) == 2
    assert relation.entities[0].kind.value == "person"
    assert relation.relations[0].relation == "after"
    assert relation.intent.value == "find_event"
    before = rule_based_plan("What happened right before the safety alarm?")
    assert before.intent.value == "window_before"
    assert len(before.entities) == 1
    assert "alarm" in before.entities[0].phrase
    first = rule_based_plan("Which person entered first?")
    assert first.intent.value == "first"
    assert first.entities[0].phrase == "person entered"
    ordering = rule_based_plan("What happened first, then what happened next?")
    assert ordering.intent.value == "order"


def test_stationary_duration_question_parses_and_enforces_threshold():
    question = "Find every object that sat there untouched for more than 2 minutes"
    plan = rule_based_plan(question)
    assert plan.filters.min_duration == 120.0
    database = EvidenceDB(":memory:")
    executor = QueryExecutor(database, "synthetic", "unused.mp4", 300.0, ".")
    candidates = [
        TimedInterval(start=10, end=129, label="stationary", source="rule", confidence=0.8,
                      event_ids=["short"], track_ids=["track-short"]),
        TimedInterval(start=140, end=270, label="stationary", source="rule", confidence=0.8,
                      event_ids=["long"], track_ids=["track-long"]),
    ]
    assert [item.event_ids for item in executor._apply_filters(candidates, plan)] == [["long"]]
    database.close()


def test_llm_planner_output_is_schema_validated():
    class MockPlanner:
        def plan(self, question):
            return {"intent": "before_after", "entities": [
                {"phrase": "entry", "kind": "action", "class_hint": None},
                {"phrase": "sound", "kind": "sound", "class_hint": "sound"}],
                "relations": [{"a": "entry", "relation": "before", "b": "sound", "seconds": None}],
                "filters": {"min_duration": None, "zone": None, "time_range": None,
                            "count_distinct": False},
                "asks_causation": False, "preceding_seconds": 10.0}

    plan = parse_question("How long between entry and sound?", llm_client=MockPlanner())
    assert plan.intent.value == "before_after"
    assert len(plan.entities) == 2


def test_pts_index_tracking_and_stationary_events(tmp_path):
    video, _ = create_synthetic_video(tmp_path)
    samples = list(iter_video(video, sample_fps=3, max_dimension=640))
    assert len(samples) >= 15
    assert all(a.pts < b.pts for a, b in zip(samples, samples[1:]))
    assert samples[0].pts == pytest.approx(0.0, abs=0.15)
    detector = MockedDetector()
    tracker = MultiObjectTracker(max_gap_seconds=1.0)
    observations = []
    for sample in samples:
        detections = detector.detect(sample.frame)
        observations.extend(tracker.update(sample, detections))
    assert observations
    database = EvidenceDB(tmp_path / "evidence.sqlite3")
    events = build_track_events("synthetic", observations, database, [],
                                stationary_epsilon=0.04, stationary_seconds=1.0)
    assert any(event["event_type"] == "appear" for event in events)
    assert any(event["event_type"] == "stationary" for event in events)
    database.close()


def test_tracker_maintains_identity_through_compensated_camera_pan():
    tracker = MultiObjectTracker(max_gap_seconds=1.0)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    first = FrameSample(pts=0.0, frame=frame, shot_id=0,
                        to_reference=np.eye(3), shot_cut=True)
    pan_to_reference = np.array([[1.0, 0.0, -40.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0]])
    second = FrameSample(pts=0.5, frame=frame, shot_id=0,
                         to_reference=pan_to_reference)
    first_track = tracker.update(first, [Detection((10, 10, 20, 20), "person", 0.9)])[0]
    second_track = tracker.update(second, [Detection((50, 10, 20, 20), "person", 0.9)])[0]
    assert second_track.track_id == first_track.track_id


def test_tracker_recovers_identity_after_occlusion_during_crossing():
    tracker = MultiObjectTracker(max_gap_seconds=2.0)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    def sample(pts):
        return FrameSample(pts=pts, frame=frame, shot_id=0, to_reference=np.eye(3))

    initial = tracker.update(sample(0.0), [
        Detection((10, 10, 10, 10), "person", 0.9),
        Detection((70, 10, 10, 10), "person", 0.9),
    ])
    first_id, second_id = initial[0].track_id, initial[1].track_id
    tracker.update(sample(0.5), [
        Detection((20, 10, 10, 10), "person", 0.9),
        Detection((60, 10, 10, 10), "person", 0.9),
    ])
    tracker.update(sample(1.0), [
        Detection((30, 10, 10, 10), "person", 0.9),
        Detection((55, 10, 10, 10), "person", 0.9),
    ])
    tracker.update(sample(1.5), [Detection((45, 10, 10, 10), "person", 0.9)])
    returned = tracker.update(sample(2.0), [
        Detection((35, 10, 10, 10), "person", 0.9),
        Detection((50, 10, 10, 10), "person", 0.9),
    ])
    by_x = {round(item.box[0]): item.track_id for item in returned}
    assert by_x[50] == first_id
    assert by_x[35] == second_id


def test_tracker_reconnects_after_multi_second_occlusion():
    tracker = MultiObjectTracker(max_gap_seconds=5.0)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    first = FrameSample(pts=0.0, frame=frame, shot_id=0, to_reference=np.eye(3))
    second = FrameSample(pts=0.5, frame=frame, shot_id=0, to_reference=np.eye(3))
    returned = FrameSample(pts=3.0, frame=frame, shot_id=0, to_reference=np.eye(3))
    first_track = tracker.update(first, [Detection((10, 10, 10, 10), "person", 0.9)])[0]
    tracker.update(second, [Detection((20, 10, 10, 10), "person", 0.9)])
    returned_track = tracker.update(returned, [Detection((70, 10, 10, 10), "person", 0.9)])[0]
    assert returned_track.track_id == first_track.track_id


def test_unmatched_question_still_returns_timestamp_and_evidence_id(tmp_path):
    video, _ = create_synthetic_video(tmp_path)
    database = EvidenceDB(tmp_path / "empty.sqlite3")
    answer = QueryExecutor(database, "synthetic", video, 6.0, tmp_path / "cache").ask(
        "Find a completely unsupported event description")
    assert answer.t_start >= 0
    assert answer.t_end >= answer.t_start
    assert answer.low_confidence
    assert answer.event_ids
    database.close()


def test_person_after_anchor_relation_uses_ordered_evidence(tmp_path):
    video, _ = create_synthetic_video(tmp_path)
    database = EvidenceDB(tmp_path / "relation.sqlite3")
    database.add_event({"event_id": "truck-arrival", "video_id": "synthetic", "track_id": "truck-1",
                        "class": "delivery truck", "event_type": "appear", "t_start": 1.0,
                        "t_end": 1.0, "zone": None, "confidence": 0.8, "meta_json": "{}"})
    database.add_event({"event_id": "person-enter", "video_id": "synthetic", "track_id": "person-1",
                        "class": "person", "event_type": "zone_enter", "t_start": 4.0,
                        "t_end": 4.0, "zone": "restricted", "confidence": 0.9, "meta_json": "{}"})
    database.execute("INSERT INTO zones(zone_id,video_id,name,polygon_json,shot_id) VALUES(?,?,?,?,?)",
                     ("restricted", "synthetic", "restricted", "[[0,0],[1,0],[1,1]]", 0))
    answer = QueryExecutor(database, "synthetic", video, 6.0, tmp_path / "cache").ask(
        "Which person entered the restricted area after the delivery truck arrived?")
    assert answer.track_ids == ["person-1"]
    assert "person" in answer.answer.lower()
    assert "after" in answer.answer.lower()
    assert answer.t_start <= 4.0 <= answer.t_end
    assert "person-enter" in answer.event_ids
    assert "truck-arrival" in answer.event_ids
    database.close()


def test_right_before_alarm_returns_ordered_preceding_event(tmp_path):
    video, _ = create_synthetic_video(tmp_path)
    database = EvidenceDB(tmp_path / "before.sqlite3")
    database.add_event({"event_id": "approach", "video_id": "synthetic", "track_id": "object-1",
                        "class": "object", "event_type": "motion_start", "t_start": 3.0,
                        "t_end": 3.5, "zone": None, "confidence": 0.7, "meta_json": "{}"})
    database.add_event({"event_id": "alarm-tone", "video_id": "synthetic", "track_id": None,
                        "class": "sound", "event_type": "sustained_tonal_sound", "t_start": 4.0,
                        "t_end": 4.5, "zone": None, "confidence": 0.8, "meta_json": "{}"})
    answer = QueryExecutor(database, "synthetic", video, 6.0, tmp_path / "cache").ask(
        "What happened right before the safety alarm?")
    assert answer.preceding_events
    assert answer.preceding_events[0].event_ids == ["approach"]
    assert answer.preceding_events[0].end < 4.0
    assert answer.t_start <= 4.0 <= answer.t_end
    assert "caused" not in answer.answer.lower()
    database.close()


def test_count_deduplicates_frame_and_track_motion_evidence():
    database = EvidenceDB(":memory:")
    executor = QueryExecutor(database, "synthetic", "unused.mp4", 10.0, ".")
    duplicate_global = TimedInterval(start=2.0, end=2.1, label="motion stop", source="rule",
                                     confidence=0.7, event_ids=["global"])
    tracked = TimedInterval(start=2.1, end=2.1, label="motion stop", source="rule",
                            confidence=0.8, track_ids=["track-1"], event_ids=["track-event"])
    repeated = TimedInterval(start=7.0, end=7.1, label="motion stop", source="rule",
                             confidence=0.8, track_ids=["track-1"], event_ids=["track-event-2"])
    assert len(executor._deduplicate_occurrences([duplicate_global, tracked, repeated])) == 2
    database.close()


def test_count_answer_reports_each_occurrence_timestamp():
    database = EvidenceDB(":memory:")
    executor = QueryExecutor(database, "synthetic", "unused.mp4", 10.0, ".")
    intervals = [
        TimedInterval(start=2.0, end=2.1, label="motion stop", source="rule",
                      confidence=0.7, event_ids=["global"]),
        TimedInterval(start=2.1, end=2.1, label="motion stop", source="rule",
                      confidence=0.8, track_ids=["track-1"], event_ids=["track-event"]),
        TimedInterval(start=7.0, end=7.1, label="motion stop", source="rule",
                      confidence=0.8, track_ids=["track-1"], event_ids=["track-event-2"]),
    ]
    plan = QueryPlan(intent=Intent.count,
                     entities=[EntitySpec(phrase="object stopped")],
                     raw_question="How many times did the object stop?")
    plan.filters.count_distinct = False
    answer = executor._compose(plan.raw_question, plan, intervals, intervals[0], [])
    assert "Observed 2 matching event(s)" in answer
    assert "00:02" in answer
    assert "00:07" in answer
    database.close()


def test_exhaustive_vlm_search_visits_late_video_buckets():
    class MockVLM:
        def __init__(self):
            self.windows = []

        def answer(self, question, context, video_path, start, end):
            self.windows.append((start, end))
            if start == 20.0:
                return {"answer": "NO_MATCH", "confidence": 0.0,
                        "timestamp_start": start, "timestamp_end": end}
            return {"answer": "matching event", "confidence": 0.8,
                    "timestamp_start": start, "timestamp_end": end}

        def plan(self, question):
            raise AssertionError("the test supplies its plan directly")

    vlm = MockVLM()
    database = EvidenceDB(":memory:")
    executor = QueryExecutor(database, "synthetic", "unused.mp4", 120.0, ".", vlm=vlm)
    rows = [{"event_id": f"event-{index}", "track_id": None,
             "t_start": float(index * 10), "t_end": float(index * 10 + 1),
             "confidence": 0.8} for index in range(12)]
    results = executor._vlm_resolve(
        "Find every matching event", [EntitySpec(phrase="matching event")], rows,
        exhaustive=True)
    assert len(results) == 11
    assert len(vlm.windows) == 12
    assert vlm.windows[-1] == (110.0, 120.0)
    assert all(item.start != 20.0 for item in results)
    database.close()


def test_exhaustive_open_vocabulary_scan_tracks_late_detections(monkeypatch):
    class MockOpenVocabularyDetector:
        def __init__(self, classes):
            self.classes = classes

        def detect(self, frame):
            return [Detection((10, 10, 20, 20), "delivery truck", 0.9)]

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    samples = [FrameSample(pts=timestamp, frame=frame, shot_id=0,
                           to_reference=np.eye(3))
               for timestamp in (0.0, 0.5, 110.0, 110.5)]
    calls = []

    def fake_iter_video(video_path, **kwargs):
        calls.append(kwargs)
        yield from samples

    monkeypatch.setattr("executor.OpenVocabularyDetector", MockOpenVocabularyDetector)
    monkeypatch.setattr("ingest.iter_video", fake_iter_video)
    database = EvidenceDB(":memory:")
    executor = QueryExecutor(database, "synthetic", "unused.mp4", 120.0, ".")
    intervals = executor._open_vocab_resolve(
        "How many delivery trucks appeared?",
        [EntitySpec(phrase="delivery truck", kind=EntityKind.object,
                    class_hint="delivery truck")], [])
    assert len(calls) == 1
    assert "start_time" not in calls[0]
    assert len(intervals) == 2
    assert intervals[-1].start == 110.0
    assert intervals[-1].end == 110.5
    assert intervals[0].track_ids
    database.close()


def test_hosted_vlm_frames_retain_source_pts(tmp_path):
    from vlm_client import HostedOpenAIClient

    video, _ = create_synthetic_video(tmp_path)
    frames = HostedOpenAIClient()._frames(str(video), 1.0, 4.0)
    timestamps = [pts for pts, _ in frames]
    assert timestamps
    assert timestamps == sorted(timestamps)
    assert timestamps[0] >= 1.0
    assert timestamps[-1] <= 4.0
    assert all(isinstance(image, str) and image for _, image in frames)


def test_nvidia_video_key_selects_nvidia_hosted_vlm(monkeypatch):
    from vlm_client import HostedOpenAIClient, choose_vlm

    monkeypatch.setenv("NVIDIA_VIDEO_API_KEY", "test-video-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TEMPORALVIDEO_VLM_MODEL", raising=False)

    client = choose_vlm()

    assert isinstance(client, HostedOpenAIClient)
    assert client.provider == "nvidia"
    assert client.api_key == "test-video-key"
    assert client.base_url == "https://integrate.api.nvidia.com/v1"
    assert client.model == "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"


def test_specific_sound_label_is_not_confident_from_onset_alone():
    database = EvidenceDB(":memory:")
    database.add_event({"event_id": "onset", "video_id": "synthetic", "track_id": None,
                        "class": "sound", "event_type": "audio_onset", "t_start": 3.0,
                        "t_end": 3.1, "zone": None, "confidence": 0.8, "meta_json": "{}"})
    executor = QueryExecutor(database, "synthetic", "unused.mp4", 10.0, ".")
    candidates = executor._audio_resolve([EntitySpec(phrase="safety alarm", kind=EntityKind.sound)],
                                         database.events("synthetic"))
    assert candidates
    assert all(item.confidence <= 0.34 for item in candidates)
    database.close()


def test_find_every_answer_lists_each_candidate_timestamp():
    database = EvidenceDB(":memory:")
    executor = QueryExecutor(database, "synthetic", "unused.mp4", 30.0, ".")
    plan = QueryPlan(intent=Intent.find_event, entities=[EntitySpec(phrase="object untouched")],
                     raw_question="Find every object untouched for more than 2 minutes")
    candidates = [TimedInterval(start=4, end=130, label="stationary", source="rule", confidence=0.8,
                                track_ids=["track-1"], event_ids=["event-1"]),
                  TimedInterval(start=150, end=290, label="stationary", source="rule", confidence=0.7,
                                track_ids=["track-2"], event_ids=["event-2"])]
    target = executor._select_target(candidates, plan, [])
    answer = executor._compose(plan.raw_question, plan, candidates, target, [])
    assert "track-1 [00:04-02:10]" in answer
    assert "track-2 [02:30-04:50]" in answer
    database.close()


def test_zone_membership_uses_normalized_coordinates():
    zone = Zone("roi", [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)])
    assert zone.contains((0.5, 0.5))
    assert not zone.contains((0.95, 0.5))
    assert _segments_intersect((0.5, 0.1), (0.5, 0.9), (0.2, 0.5), (0.8, 0.5))
    assert not _segments_intersect((0.1, 0.1), (0.2, 0.2), (0.5, 0.5), (0.8, 0.8))


def test_generated_tone_is_indexed_when_ffmpeg_available(tmp_path):
    video, has_audio = create_synthetic_video(tmp_path)
    if not has_audio:
        pytest.skip("ffmpeg unavailable for synthetic audio mux")
    from audio import detect_audio_events
    database = EvidenceDB(tmp_path / "audio.sqlite3")
    events = detect_audio_events(video, "synthetic", database)
    assert any(event["event_type"] in {"audio_onset", "sustained_tonal_sound", "loud_transient"}
               and 3.7 <= event["t_start"] <= 4.5 for event in events)
    database.close()
