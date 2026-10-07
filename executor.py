"""Evidence-first resolver and temporal query executor."""
from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from db import EvidenceDB
from detect_track import OpenVocabularyDetector, extract_question_classes
from planner import parse_question
from reid import Merge, merged_id
from schema import Answer, EntitySpec, Intent, PrecedingEvent, QueryPlan, TimedInterval, format_timestamp
from semantic import ClipSemanticIndex
from verify import refine_window
from vlm_client import VLMClient, choose_vlm


EVENT_SYNONYMS = {
    "stop": {"motion_stop"}, "stopped": {"motion_stop"},
    "start": {"motion_start"}, "started": {"motion_start"}, "appeared": {"appear"},
    "occurred": {"appear", "zone_enter", "motion_start", "motion_stop"},
    "enter": {"zone_enter"},
    "entered": {"zone_enter"}, "arrive": {"appear"}, "arrived": {"appear"},
    "arrival": {"appear"}, "leave": {"zone_exit", "disappear"},
    "left": {"zone_exit", "disappear"}, "alarm": {"audio_onset", "loud_transient", "sustained_tonal_sound"},
    "beep": {"audio_onset", "sustained_tonal_sound"}, "sound": {"audio_onset", "loud_transient"},
    "wave": {"motion_start", "interaction"}, "untouched": {"stationary"},
    "sit": {"stationary"}, "sat": {"stationary"}, "appear": {"appear"},
    "disappear": {"disappear"}, "speech": {"speech"},
    "back": {"appear"}, "return": {"appear"}, "came": {"appear"}, "come": {"appear"},
}


class QueryExecutor:
    def __init__(self, db: EvidenceDB, video_id: str, video_path: str | Path,
                 duration: float, cache_dir: str | Path, reid_merges: list[Merge] | None = None,
                 vlm: VLMClient | None = None, before_after_seconds: float = 10.0,
                 semantic_provider: str | None = None):
        self.db, self.video_id, self.video_path = db, video_id, str(video_path)
        self.duration, self.cache_dir = duration, Path(cache_dir)
        self.merges = reid_merges or []
        self.vlm = vlm if vlm is not None else choose_vlm()
        self.before_after_seconds = before_after_seconds
        self.semantic_provider = semantic_provider

    def ask(self, question: str, plan: QueryPlan | None = None) -> Answer:
        plan = plan or parse_question(question, llm_client=self.vlm,
                          default_window=self.before_after_seconds)
        all_events = self.db.events(self.video_id)
        exhaustive = self._needs_all_matches(plan)
        gap_query = bool(re.search(r"\bhow long\s+(?:is it\s+)?between\b", question.lower()))
        list_all = plan.intent == Intent.order or (
            plan.intent in {Intent.first, Intent.last}
            and (not plan.entities or all(entity.phrase in {"first", "last", "event", "events"}
                                          for entity in plan.entities)))
        if list_all:
            intervals = self._all_event_intervals(all_events)
            if plan.intent == Intent.order and self.vlm is not None:
                intervals = self._merge_resolver_results(
                    intervals, self._vlm_resolve(question, plan.entities, all_events, exhaustive=True))
        elif gap_query and len(plan.entities) >= 2:
            pair = self._resolve_gap(plan.entities[0], plan.entities[1], all_events, question)
            intervals = [pair] if pair is not None else []
        elif plan.relations and len(plan.entities) >= 2:
            intervals = self._resolve_relation(plan.entities[0], plan.entities[1],
                                               plan.relations[0].relation, all_events)
        else:
            intervals = []
            resolvers = [
                lambda: self._sql_resolve(plan.entities, all_events),
                lambda: self._audio_resolve(plan.entities, all_events),
                lambda: self._clip_resolve(plan.entities),
                lambda: self._open_vocab_resolve(question, plan.entities, all_events),
            ]
            for resolver in resolvers:
                intervals = self._merge_resolver_results(intervals, resolver())
                if not exhaustive and any(item.confidence >= 0.55 for item in intervals):
                    break
        if exhaustive or not any(item.confidence >= 0.55 for item in intervals):
            intervals = self._merge_resolver_results(
                intervals, self._lazy_caption_resolve(plan.entities, all_events, exhaustive))
        if exhaustive or not any(item.confidence >= 0.55 for item in intervals):
            intervals = self._merge_resolver_results(
                intervals, self._vlm_resolve(question, plan.entities, all_events, exhaustive))
        if intervals and self._needs_semantic_verification(question):
            intervals = self._verify_subjective(question, plan.entities, intervals, exhaustive)

        intervals = self._apply_filters(intervals, plan)
        target = self._select_target(intervals, plan, all_events)
        if target is None:
            target = self._best_candidate(all_events)
        target = self._persist_support(target)
        target = self._refine(target)
        preceding = self._preceding(target, plan, all_events) if plan.asks_causation or \
            ((plan.intent in {Intent.window_before, Intent.before_after} or "right before" in question.lower())
             and "before" in question.lower()) else []
        text = self._compose(question, plan, intervals, target, preceding)
        caveat = self._zone_caveat(question)
        identity_query = (plan.intent in {Intent.count, Intent.identify_track}
                  or any(token in question.lower() for token in ("same person", "come back", "left with")))
        used_tracks = {track_id for item in intervals for track_id in item.track_ids}
        used_merges = [merge for merge in self.merges
                   if merge.source_track_id in used_tracks or merge.target_track_id in used_tracks]
        merge_confidence = min((merge.score for merge in used_merges), default=None) if identity_query else None
        cross_shot_merges = [f"{merge.source_track_id}->{merge.target_track_id} ({merge.score:.2f})"
                     for merge in used_merges if merge.crossed_cut] if identity_query else []
        confidence = min(target.confidence, merge_confidence) if merge_confidence is not None else target.confidence
        confidence = min(confidence, 0.34) if caveat else confidence
        answer_start, answer_end = target.start, target.end
        if answer_end <= answer_start:
            margin = min(0.2, self.duration / 2) if self.duration > 0 else 0.001
            answer_start = max(0.0, answer_start - margin)
            answer_end = min(self.duration, target.end + margin) if self.duration > 0 else target.end + margin
            if answer_end <= answer_start:
                answer_end = answer_start + 0.001
        return Answer(answer=text, t_start=answer_start, t_end=answer_end,
                      track_ids=self._canonical_tracks(target.track_ids), event_ids=target.event_ids,
                      confidence=confidence, low_confidence=confidence < 0.35,
                  id_merge_confidence=merge_confidence, cross_shot_merges=cross_shot_merges,
                      video_duration=self.duration,
                      timestamp_source=target.source, uncertainty_seconds=float(target.metadata.get("uncertainty", 2.0)),
                      preceding_events=preceding, caveat=caveat)

    def _sql_resolve(self, entities: list[EntitySpec], rows) -> list[TimedInterval]:
        result = []
        for row in rows:
            if row["event_type"] in {"audio_onset", "loud_transient", "sustained_tonal_sound",
                                      "speech", "silence", "sudden_silence"} and any(
                    entity.kind.value == "sound" for entity in entities):
                continue
            if row["class"] and any(entity.kind.value == "person" for entity in entities):
                if not any(token in row["class"].lower() for token in ("person", "human")):
                    continue
            zone_entities = [entity for entity in entities
                             if entity.kind.value == "zone" or re.search(
                                 r"\b(?:zone|area|restricted|inside|sector|region|room|boundary)\b",
                                 entity.phrase, re.I)]
            if row["zone"] and zone_entities:
                row_zone = set(re.findall(r"[a-z0-9]+", row["zone"].lower()))
                stop_zone_words = {"person", "entered", "enter", "zone", "area", "inside",
                                   "sector", "region", "room", "boundary", "the", "into", "after"}
                specific_zone_terms = [set(re.findall(r"[a-z0-9]+", entity.phrase.lower())) - stop_zone_words
                                       for entity in zone_entities]
                specific_zone_terms = [terms for terms in specific_zone_terms if terms]
                if specific_zone_terms and not any(row_zone & terms for terms in specific_zone_terms):
                    continue
            searchable = " ".join([row["event_type"] or "", row["class"] or "", row["zone"] or "",
                                   row["meta_json"] or ""]).lower().replace("_", " ")
            if any(self._matches(entity.phrase, searchable, row["event_type"]) for entity in entities):
                result.append(self._from_row(row, "rule"))
        return result

    def _resolve_entity(self, entity: EntitySpec, rows) -> list[TimedInterval]:
        hits = self._sql_resolve([entity], rows)
        if not hits:
            hits = self._audio_resolve([entity], rows)
        if not hits:
            hits = self._clip_resolve([entity])
        return hits

    def _all_event_intervals(self, rows) -> list[TimedInterval]:
        selected = [row for row in rows if not str(row["event_type"]).startswith("evidence_")]
        selected.sort(key=lambda row: (float(row["t_start"]), -float(row["confidence"])))
        output: list[TimedInterval] = []
        for row in selected:
            candidate = self._from_row(row, "rule")
            duplicate = next((item for item in output[-8:]
                              if item.label == candidate.label and item.track_ids == candidate.track_ids
                              and abs(item.start - candidate.start) <= 0.75), None)
            if duplicate is None:
                output.append(candidate)
        return output

    def _resolve_relation(self, subject: EntitySpec, anchor: EntitySpec, relation: str,
                          rows) -> list[TimedInterval]:
        subjects, anchors = self._resolve_entity(subject, rows), self._resolve_entity(anchor, rows)
        if not subjects or not anchors:
            return []
        output = []
        for subject_hit in subjects:
            related = []
            for anchor_hit in anchors:
                if set(subject_hit.event_ids) & set(anchor_hit.event_ids):
                    continue
                if relation == "after" and subject_hit.start >= anchor_hit.end:
                    related.append(anchor_hit)
                elif relation == "before" and subject_hit.end <= anchor_hit.start:
                    related.append(anchor_hit)
                elif relation == "during" and subject_hit.start <= anchor_hit.end and anchor_hit.start <= subject_hit.end:
                    related.append(anchor_hit)
            if related:
                closest = min(related, key=lambda item: abs(subject_hit.start - item.end))
                subject_hit.event_ids = list(dict.fromkeys(subject_hit.event_ids + closest.event_ids))
                subject_hit.metadata["relation_anchor"] = closest.label
                subject_hit.metadata["relation_anchor_time"] = closest.start
                output.append(subject_hit)
        return output

    def _merge_resolver_results(self, current: list[TimedInterval],
                                incoming: list[TimedInterval]) -> list[TimedInterval]:
        priority = {"audio": 4, "rule": 3, "detector": 3, "clip": 2, "vlm": 1}
        merged = list(current)
        for candidate in incoming:
            overlap = next((index for index, existing in enumerate(merged)
                            if candidate.start <= existing.end + 1.0
                            and existing.start <= candidate.end + 1.0
                            and (candidate.source != existing.source
                                 or bool(set(candidate.event_ids) & set(existing.event_ids)))), None)
            if overlap is None:
                merged.append(candidate)
                continue
            existing = merged[overlap]
            winner, other = (candidate, existing) if priority[candidate.source] > priority[existing.source] else (existing, candidate)
            start = max(existing.start, candidate.start)
            end = min(existing.end, candidate.end)
            if end < start:
                start, end = min(existing.start, candidate.start), max(existing.end, candidate.end)
            winner.start, winner.end = start, end
            winner.confidence = min(0.95, max(existing.confidence, candidate.confidence) + 0.08)
            winner.event_ids = list(dict.fromkeys(existing.event_ids + candidate.event_ids))
            winner.track_ids = list(dict.fromkeys(existing.track_ids + candidate.track_ids))
            winner.metadata["corroborating_sources"] = list(dict.fromkeys(
                existing.metadata.get("corroborating_sources", [existing.source]) +
                candidate.metadata.get("corroborating_sources", [candidate.source])))
            merged[overlap] = winner
        return merged

    def _audio_resolve(self, entities: list[EntitySpec], rows) -> list[TimedInterval]:
        audio = [row for row in rows if row["event_type"] in {
            "audio_onset", "loud_transient", "sustained_tonal_sound", "speech", "silence", "sudden_silence"}]
        result = []
        for row in audio:
            for entity in entities:
                if entity.kind.value != "sound":
                    continue
                words = set(re.findall(r"[a-z0-9]+", entity.phrase.lower()))
                generic_sound = bool(words & {"sound", "noise", "audio", "onset"})
                match = self._matches(entity.phrase,
                                      (row["event_type"] + " " + row["meta_json"]).lower(),
                                      row["event_type"])
                if not match and not generic_sound:
                    continue
                candidate = self._from_row(row, "audio")
                if not generic_sound:
                    candidate.confidence = min(candidate.confidence, 0.34)
                    candidate.metadata["needs_semantic_sound_label"] = entity.phrase
                    candidate.metadata["uncertainty"] = max(5.0, candidate.end - candidate.start)
                result.append(candidate)
        result.sort(key=lambda item: (
            0 if "sustained tonal" in item.label else 1 if "loud transient" in item.label else 2,
            -float(item.metadata.get("meta", {}).get("spectral_flux", 0.0)),
            item.start))
        return result

    def _clip_resolve(self, entities: list[EntitySpec]) -> list[TimedInterval]:
        try:
            semantic = ClipSemanticIndex(provider=self.semantic_provider)
            result = []
            for entity in entities:
                for item in semantic.ground_text(entity.phrase, self.db, self.video_id):
                    result.append(TimedInterval(start=item["start"], end=item["end"], label=entity.phrase,
                                                source="clip", confidence=min(0.69, max(0.36, item["score"])),
                                                metadata={"similarity": item["score"]}))
            return result
        except (RuntimeError, OSError, ValueError) as exc:
            logging.warning("Semantic grounding unavailable: %s", exc)
            return []

    def _open_vocab_resolve(self, question: str, entities: list[EntitySpec], rows) -> list[TimedInterval]:
        classes = list(dict.fromkeys([entity.class_hint or entity.phrase for entity in entities]))
        if not classes:
            classes = extract_question_classes(question)
        if not classes:
            return []
        exhaustive = bool(re.search(
            r"\b(?:how many|count|find every|find all|list every|list all|in what order|what order|sequence)\b",
            question.lower()))
        ranked = sorted(rows, key=lambda row: float(row["confidence"]), reverse=True)
        windows = []
        seen_windows = set()
        for row in ranked:
            key = int(float(row["t_start"]) // 10)
            if key not in seen_windows:
                windows.append((float(row["t_start"]), float(row["t_end"])))
                seen_windows.add(key)
            if not exhaustive and len(windows) >= 8:
                break
        if not windows:
            windows = [(0.0, min(self.duration, 10.0))]
        try:
            detector = OpenVocabularyDetector(classes)
            from ingest import iter_video
            from detect_track import MultiObjectTracker
            found = []
            tracker = MultiObjectTracker(max_gap_seconds=5.0)
            sample_groups = ([iter_video(self.video_path, sample_fps=2.0, max_dimension=640)]
                             if exhaustive else
                             [iter_video(self.video_path, sample_fps=2.0, max_dimension=640,
                                         start_time=max(0.0, start - 1.0),
                                         end_time=min(self.duration, end + 1.0))
                              for start, end in windows])
            track_intervals: dict[str, TimedInterval] = {}
            for samples in sample_groups:
                for sample in samples:
                    detections = detector.detect(sample.frame)
                    for observation in tracker.update(sample, detections):
                        interval = track_intervals.get(observation.track_id)
                        if interval is None:
                            track_intervals[observation.track_id] = TimedInterval(
                                start=sample.pts, end=sample.pts, label=observation.class_name,
                                source="detector", confidence=observation.confidence,
                                track_ids=[observation.track_id],
                                metadata={"box": observation.box, "on_demand": True})
                        else:
                            interval.end = sample.pts
                            interval.confidence = max(interval.confidence, observation.confidence)
            found.extend(track_intervals.values())
            return found
        except Exception:
            return []

    def _vlm_resolve(self, question: str, entities: list[EntitySpec], rows,
                     exhaustive: bool = False) -> list[TimedInterval]:
        if self.vlm is None:
            return []
        ranked = sorted(rows, key=lambda row: float(row["confidence"]), reverse=True)
        candidates = []
        seen_buckets = set()
        for row in ranked:
            bucket = int(float(row["t_start"]) // 10)
            if bucket not in seen_buckets:
                candidates.append(row)
                seen_buckets.add(bucket)
            if not exhaustive and len(candidates) >= 8:
                break
        if exhaustive:
            by_bucket = {int(float(row["t_start"]) // 10): row for row in ranked}
            candidates = [(bucket, by_bucket.get(bucket))
                          for bucket in range(max(1, int(max(0.0, self.duration - 1e-9) // 10) + 1))]
        else:
            candidates = [(int(float(row["t_start"]) // 10), row) for row in candidates]
        if not candidates:
            candidates = [(0, None)]
        result = []
        for bucket, row in candidates:
            if exhaustive:
                window_start = bucket * 10.0
                window_end = min(self.duration, window_start + 10.0)
                start, end = window_start, window_end
            else:
                start = float(row["t_start"]) if row else 0.0
                end = float(row["t_end"]) if row else min(10.0, self.duration)
                end = min(max(end, start + 0.1), start + 20.0)
                window_start, window_end = max(0.0, start - 1), min(self.duration, end + 1)
            try:
                window_question = question
                if exhaustive:
                    window_question += (
                        " Inspect only this window. Return timestamps as absolute source-video PTS seconds. "
                        "If no matching event is visible in this window, set answer exactly to NO_MATCH "
                        "and confidence to 0.")
                data = self.vlm.answer(window_question, {"entities": [e.model_dump() for e in entities]},
                                       self.video_path, window_start, window_end)
                answer_label = str(data.get("answer", "candidate event")).strip()
                if exhaustive and answer_label.upper() == "NO_MATCH":
                    continue
                answer_start = max(window_start, min(window_end, float(data.get("timestamp_start", start))))
                answer_end = max(answer_start, min(window_end, float(data.get("timestamp_end", end))))
                result.append(TimedInterval(start=answer_start, end=answer_end,
                                            label=answer_label, source="vlm",
                                            confidence=max(0.0, min(0.7, float(data.get("confidence", 0.4)))),
                                            event_ids=[row["event_id"]] if row else [],
                                            track_ids=[row["track_id"]] if row and row["track_id"] else [],
                                            metadata={"vlm_result": data, "uncertainty": 7.5}))
                if not exhaustive and result[-1].confidence >= 0.55:
                    break
            except Exception:
                continue
        return result

    def _lazy_caption_resolve(self, entities: list[EntitySpec], rows,
                              exhaustive: bool = False) -> list[TimedInterval]:
        if self.vlm is None or self.duration <= 0:
            return []
        ranked = sorted(rows, key=lambda row: float(row["confidence"]), reverse=True)
        windows: dict[int, Any] = {}
        for row in ranked:
            windows.setdefault(int(float(row["t_start"]) // 10), row)
        if not windows:
            final_bucket = max(0, int((self.duration - 0.01) // 10))
            for bucket in sorted({0, final_bucket // 2, final_bucket}):
                windows[bucket] = None

        matches = []
        selected_windows = ([(bucket, windows.get(bucket)) for bucket in
                             range(max(1, int(max(0.0, self.duration - 1e-9) // 10) + 1))]
                    if exhaustive else sorted(windows.items())[:4])
        for bucket, row in selected_windows:
            start, end = bucket * 10.0, min(self.duration, (bucket + 1) * 10.0)
            context = {"indexed_events": ([{"event_type": row["event_type"], "class": row["class"],
                                            "t_start": row["t_start"], "t_end": row["t_end"]}]
                                         if row is not None else [])}
            try:
                caption = self.vlm.answer(
                    "Describe observable entities and actions in this short window. Do not infer causes. "
                    "Return JSON with answer and confidence.", context, self.video_path, start, end)
            except Exception:
                continue
            description = str(caption.get("answer", "")).strip()
            if not description:
                continue
            event_id = f"e-{uuid.uuid4().hex}"
            confidence = max(0.0, min(0.65, float(caption.get("confidence", 0.4))))
            self.db.add_event({"event_id": event_id, "video_id": self.video_id, "track_id": None,
                               "class": "scene", "event_type": "vlm_caption", "t_start": start,
                               "t_end": end, "zone": None, "confidence": confidence,
                               "meta_json": json.dumps({"caption": description, "lazy": True})})
            if any(self._matches(entity.phrase, description.lower(), "vlm_caption") for entity in entities):
                matches.append(TimedInterval(start=start, end=end, label=description, source="vlm",
                                             confidence=confidence, event_ids=[event_id],
                                             metadata={"caption": True, "uncertainty": 7.5}))
        return matches

    def _needs_semantic_verification(self, question: str) -> bool:
        return bool(re.search(r"\b(unexpectedly|suspiciously|delivery|safety|accidentally|intentionally|same person|who left with)\b",
                              question.lower()))

    def _needs_all_matches(self, plan: QueryPlan) -> bool:
        return (plan.intent in {Intent.count, Intent.order}
                or (plan.intent in {Intent.first, Intent.last}
                    and (not plan.entities or all(entity.phrase in {"first", "last", "event", "events"}
                                                  for entity in plan.entities)))
                or bool(re.search(r"\b(?:find every|find all|list every|list all)\b",
                                  plan.raw_question.lower())))

    def _verify_subjective(self, question: str, entities: list[EntitySpec],
                           candidates: list[TimedInterval], exhaustive: bool = False) -> list[TimedInterval]:
        if self.vlm is None:
            for candidate in candidates:
                candidate.confidence = min(candidate.confidence, 0.34)
                candidate.metadata["uncertainty"] = max(5.0, candidate.end - candidate.start)
                candidate.metadata["semantic_verification"] = "unavailable"
            return candidates
        verified = []
        ordered = sorted(candidates, key=lambda item: item.confidence, reverse=True)
        for candidate in ordered if exhaustive else ordered[:8]:
            try:
                result = self.vlm.answer(
                    question,
                    {"entities": [entity.model_dump() for entity in entities],
                     "rule_based_evidence": candidate.model_dump(),
                     "instruction": "Judge only the short candidate window; distinguish observation from inference."},
                    self.video_path, max(0.0, candidate.start - 1.0),
                    min(self.duration, max(candidate.end + 1.0, candidate.start + 0.1)))
                candidate.label = str(result.get("answer", candidate.label))
                candidate.source = "vlm"
                candidate.confidence = min(0.8, max(0.0, float(result.get("confidence", candidate.confidence))))
                candidate.metadata["uncertainty"] = max(5.0, candidate.end - candidate.start)
                candidate.metadata["vlm_result"] = result
                verified.append(candidate)
            except Exception:
                continue
        if verified:
            return verified
        for candidate in candidates:
            candidate.confidence = min(candidate.confidence, 0.34)
            candidate.metadata["uncertainty"] = max(5.0, candidate.end - candidate.start)
            candidate.metadata["semantic_verification"] = "unavailable"
        return candidates

    def _matches(self, phrase: str, searchable: str, event_type: str) -> bool:
        normalized = phrase.lower().replace("_", " ")
        tokens = [token for token in re.findall(r"[a-z0-9]+", normalized) if len(token) > 2]
        if normalized and normalized in searchable:
            return True
        mapped = set().union(*(EVENT_SYNONYMS.get(token, set()) for token in tokens))
        if event_type in mapped:
            return True
        informative = [token for token in tokens if token not in {"person", "object", "something", "unexpectedly"}]
        return bool(informative and any(token in searchable for token in informative))

    def _from_row(self, row, source: str) -> TimedInterval:
        if row["event_type"] in {"audio_onset", "loud_transient", "sustained_tonal_sound",
                                  "speech", "silence", "sudden_silence"}:
            source = "audio"
        if row["event_type"] == "vlm_caption":
            source = "vlm"
        event_label = row["event_type"].replace("_", " ")
        if row["class"]:
            event_label = f"{row['class']} {event_label}"
        if row["event_type"] == "vlm_caption":
            event_label = json.loads(row["meta_json"] or "{}").get("caption", event_label)
        return TimedInterval(start=float(row["t_start"]), end=float(row["t_end"]),
                             label=event_label, source=source,
                             confidence=float(row["confidence"]),
                             track_ids=[row["track_id"]] if row["track_id"] else [],
                             event_ids=[row["event_id"]], metadata={"class": row["class"],
                                                                     "zone": row["zone"],
                                                                     "meta": json.loads(row["meta_json"] or "{}")})

    def _apply_filters(self, intervals: list[TimedInterval], plan: QueryPlan) -> list[TimedInterval]:
        filters = plan.filters
        if filters.min_duration is not None:
            intervals = [item for item in intervals if item.end - item.start >= filters.min_duration]
        if filters.zone:
            intervals = [item for item in intervals if item.metadata.get("zone") == filters.zone]
        if filters.time_range:
            lo, hi = filters.time_range
            intervals = [item for item in intervals if item.end >= lo and item.start <= hi]
        return intervals

    def _select_target(self, intervals: list[TimedInterval], plan: QueryPlan, rows) -> TimedInterval | None:
        if not intervals:
            return None
        intervals.sort(key=lambda item: (item.start, -item.confidence))
        if plan.intent == Intent.last:
            return intervals[-1]
        if plan.intent == Intent.first:
            return intervals[0]
        if plan.intent == Intent.count or re.search(r"\b(find every|find all|list every|list all)\b",
                                                    plan.raw_question.lower()):
            priority = {"audio": 4, "rule": 3, "detector": 3, "clip": 2, "vlm": 1}
            return TimedInterval(start=min(item.start for item in intervals),
                                 end=max(item.end for item in intervals), label="all matching events",
                                 source=max((item.source for item in intervals), key=priority.get),
                                 confidence=min(item.confidence for item in intervals),
                                 track_ids=list(dict.fromkeys(track for item in intervals for track in item.track_ids)),
                                 event_ids=list(dict.fromkeys(event for item in intervals for event in item.event_ids)),
                                 metadata={"count": len(intervals), "aggregated": True,
                                           "uncertainty": max(float(item.metadata.get("uncertainty", 0))
                                                              for item in intervals)})
        if plan.intent in {Intent.window_before, Intent.before_after}:
            priority = {"audio": 4, "rule": 3, "detector": 3, "clip": 2, "vlm": 1}
            anchor = max(intervals, key=lambda item: (priority.get(item.source, 0), item.confidence))
            before = "before" in plan.raw_question.lower() or "right before" in plan.raw_question.lower()
            if before:
                start = max(0.0, anchor.start - plan.preceding_seconds)
                return TimedInterval(start=start, end=anchor.start, label=f"events preceding {anchor.label}",
                                     source=anchor.source, confidence=anchor.confidence,
                                     track_ids=anchor.track_ids, event_ids=anchor.event_ids,
                                     metadata={"anchor": anchor.label, "anchor_time": anchor.start})
            if "after" in plan.raw_question.lower():
                following = [row for row in rows if float(row["t_start"]) >= anchor.end
                             and row["event_id"] not in anchor.event_ids]
                if following:
                    result = self._from_row(following[0], "rule")
                    result.metadata["anchor"] = anchor.label
                    result.metadata["anchor_time"] = anchor.end
                    return result
                start = min(self.duration, anchor.end)
                return TimedInterval(start=start, end=min(self.duration, start + plan.preceding_seconds),
                                     label=f"candidate window after {anchor.label}", source=anchor.source,
                                     confidence=min(0.3, anchor.confidence), track_ids=anchor.track_ids,
                                     event_ids=anchor.event_ids,
                                     metadata={"uncertainty": plan.preceding_seconds})
        if plan.intent == Intent.count:
            return intervals[0]
        return max(intervals, key=lambda item: (item.confidence, item.source == "audio"))

    def _best_candidate(self, rows) -> TimedInterval:
        if rows:
            row = max(rows, key=lambda event: float(event["confidence"]))
            candidate = self._from_row(row, "rule")
            candidate.confidence = min(candidate.confidence, 0.25)
            candidate.metadata["uncertainty"] = max(5.0, candidate.end - candidate.start)
            candidate.label = "best available candidate; query not confidently resolved"
            return candidate
        end = min(max(self.duration, 0.05), 10.0)
        fallback = TimedInterval(start=0.0, end=end, label="best candidate window; no indexed evidence",
                                  source="rule", confidence=0.1, metadata={"uncertainty": end})
        return self._persist_support(fallback)

    def _persist_support(self, interval: TimedInterval) -> TimedInterval:
        if interval.event_ids or interval.track_ids:
            return interval
        event_id = f"e-{uuid.uuid4().hex}"
        self.db.add_event({"event_id": event_id, "video_id": self.video_id, "track_id": None,
                           "class": interval.label, "event_type": f"evidence_{interval.source}",
                           "t_start": interval.start, "t_end": interval.end, "zone": None,
                           "confidence": interval.confidence,
                           "meta_json": json.dumps({"label": interval.label,
                                                     "source": interval.source,
                                                     "uncertainty": interval.metadata.get("uncertainty")})})
        interval.event_ids = [event_id]
        return interval

    def _refine(self, interval: TimedInterval) -> TimedInterval:
        if interval.metadata.get("aggregated") and interval.end - interval.start > 20.0:
            interval.metadata.setdefault("uncertainty", 1.0)
            interval.metadata["refinement"] = "bounds aggregated from indexed event timestamps; no full-span decode"
            return interval
        try:
            refined = refine_window(self.video_path, interval)
            interval.start, interval.end = refined.start, refined.end
            interval.source = refined.source
            interval.metadata["uncertainty"] = refined.uncertainty_seconds
            interval.metadata["refinement"] = refined.method
        except (RuntimeError, OSError, ValueError):
            interval.metadata.setdefault("uncertainty", max(2.0, interval.end - interval.start))
        return interval

    def _preceding(self, target: TimedInterval, plan: QueryPlan, rows) -> list[PrecedingEvent]:
        anchor = float(target.metadata.get("anchor_time", target.start if target.start > 0 else target.end))
        start = max(0.0, anchor - plan.preceding_seconds)
        chosen = [row for row in rows if start <= float(row["t_start"]) < anchor]
        descriptions = []
        for row in chosen:
            label = row["event_type"].replace("_", " ")
            if row["class"]:
                label = f"{row['class']} {label}"
            if row["event_type"] == "vlm_caption":
                label = json.loads(row["meta_json"] or "{}").get("caption", label)
            descriptions.append(self._safe_label(label))
        ratings: dict[int, tuple[str, float]] = {}
        if self.vlm is not None and chosen:
            context = {"target_event": target.label,
                       "preceding_events": [{"event_id": row["event_id"], "description": label,
                                             "start": row["t_start"], "end": row["t_end"]}
                                            for row, label in zip(chosen, descriptions)],
                       "instruction": "Rate each event plausibly-related or unrelated. This is plausibility, never causation."}
            try:
                result = self.vlm.answer("Return JSON with preceding_ratings, each containing event_id, "
                                         "relation (plausibly-related or unrelated), and confidence.",
                                         context, self.video_path, start, min(self.duration, anchor))
                by_id = {item.get("event_id"): item for item in result.get("preceding_ratings", [])
                         if isinstance(item, dict)}
                for index, row in enumerate(chosen):
                    item = by_id.get(row["event_id"])
                    if item:
                        relation = str(item.get("relation", "")).lower()
                        rating = "plausibly-related" if "plausib" in relation else "unrelated"
                        ratings[index] = (rating, max(0.0, min(1.0, float(item.get("confidence", 0.0)))))
            except Exception:
                pass
        output = []
        for index, (row, label) in enumerate(zip(chosen, descriptions)):
            rating, rating_confidence = ratings.get(index, ("not-assessed", 0.0))
            output.append(PrecedingEvent(description=label, start=float(row["t_start"]),
                                         end=float(row["t_end"]), event_ids=[row["event_id"]],
                                         relation_rating=rating, rating_confidence=rating_confidence))
        return output

    def _compose(self, question: str, plan: QueryPlan, intervals: list[TimedInterval],
                 target: TimedInterval, preceding: list[PrecedingEvent]) -> str:
        if plan.intent == Intent.count and intervals:
            ids = {merged_id(track_id, self.merges) for item in intervals for track_id in item.track_ids}
            occurrences = self._deduplicate_occurrences(intervals)
            if plan.filters.count_distinct:
                count = len(ids) if ids else len({(item.label, item.start, item.end) for item in intervals})
                if ids:
                    first_by_track: dict[str, TimedInterval] = {}
                    for item in occurrences:
                        for track_id in self._canonical_tracks(item.track_ids):
                            first_by_track.setdefault(track_id, item)
                    timed_items = sorted(first_by_track.values(), key=lambda item: item.start)
                else:
                    timed_items = occurrences
            else:
                count = len(occurrences)
                timed_items = occurrences
            entries = []
            for item in timed_items:
                start = format_timestamp(item.start, self.duration)
                end = format_timestamp(item.end, self.duration)
                timestamp = f"{start}-{end}" if item.end > item.start else start
                entries.append(f"{self._safe_label(item.label)} [{timestamp}]")
            timestamps = "; ".join(entries)
            return (f"Observed {count} matching event(s) at {timestamps}; "
                    "this is an evidence-based count, not a causal judgment.")
        if re.search(r"\b(find every|find all|list every|list all)\b", plan.raw_question.lower()):
            entries = []
            seen = set()
            for item in sorted(intervals, key=lambda value: value.start):
                tracks = self._canonical_tracks(item.track_ids)
                label = tracks[0] if tracks else self._safe_label(item.label)
                key = (label, round(item.start, 1), round(item.end, 1))
                if key in seen:
                    continue
                seen.add(key)
                entries.append(f"{label} [{format_timestamp(item.start, self.duration)}-"
                               f"{format_timestamp(item.end, self.duration)}]")
            return "Matching candidates: " + ("; ".join(entries) if entries else "none resolved") + "."
        if plan.intent == Intent.order or (
                plan.intent in {Intent.first, Intent.last} and len(intervals) > 1):
            ordered = intervals if plan.intent == Intent.order else [target]
            if plan.intent == Intent.first:
                ordered = intervals[:1]
            elif plan.intent == Intent.last:
                ordered = intervals[-1:]
            entries = "; ".join(f"{item.label} [{format_timestamp(item.start, self.duration)}-"
                                 f"{format_timestamp(item.end, self.duration)}]"
                                 for item in ordered[:12])
            if entries:
                return f"Temporal order: {entries}. This describes sequence, not causation."
        if target.metadata.get("relation_anchor"):
            relation = next((item.relation for item in plan.relations), "after")
            subject_class = str(target.metadata.get("class") or "").strip()
            subject = self._safe_label(" ".join(part for part in
                                                   (subject_class, target.label.replace("_", " ")) if part))
            anchor = str(target.metadata.get("relation_anchor", "the anchor event"))
            return (f"{subject} happened {relation} {anchor}; "
                    "this describes observed temporal order, not causation.")
        if "gap_seconds" in target.metadata:
            return (f"{target.label}: {target.metadata['gap_seconds']:.2f} seconds between the two events. "
                    "This reports temporal spacing, not causation.")
        if "right before" in question.lower() or plan.intent == Intent.window_before:
            summary = "; ".join(f"{event.description} at {format_timestamp(event.start, self.duration)}"
                                 for event in preceding)
            if summary and preceding:
                nearest = max(preceding, key=lambda event: event.end)
                anchor = str(target.metadata.get("anchor", "the target event"))
                anchor_time = float(target.metadata.get("anchor_time", target.start))
                wording = "happened immediately before" if anchor_time - nearest.end <= 2.0 else "happened before"
                return (f"Preceding events: {summary}. {nearest.description} {wording} {anchor}; "
                        "temporal order alone does not establish causation.")
            return ("Preceding events: " + summary + ". These events occurred before the target; "
                    "temporal order alone does not establish causation." if summary else
                    "No preceding indexed event was resolved; this is the best candidate window, not a causal claim.")
        if plan.intent == Intent.window_after and target.metadata.get("anchor"):
            anchor_time = float(target.metadata.get("anchor_time", target.start))
            wording = "happened immediately after" if target.start - anchor_time <= 2.0 else "happened after"
            return f"{self._safe_label(target.label)} {wording} {target.metadata['anchor']}; no causation is implied."
        if plan.asks_causation:
            return (f"The indexed evidence shows {self._safe_label(target.label)} in this window. "
                    "Preceding events are listed separately with plausibility ratings; no causation is asserted.")
        if intervals:
            return f"{self._safe_label(target.label).replace('_', ' ')} is the strongest matching indexed evidence."
        return "No confident match was found; this timestamp is the best available candidate window."

    def _deduplicate_occurrences(self, intervals: list[TimedInterval]) -> list[TimedInterval]:
        ordered = sorted(intervals, key=lambda item: item.start)
        tracked = [item for item in ordered if item.track_ids]
        result = []
        for item in ordered:
            canonical = set(self._canonical_tracks(item.track_ids))
            if canonical:
                global_match = next((index for index, prior in enumerate(result)
                                     if not prior.track_ids
                                     and abs(item.start - prior.start) <= 0.75
                                     and item.label.split()[-1:] == prior.label.split()[-1:]), None)
                if global_match is not None:
                    result[global_match] = item
                    continue
            same_occurrence = any(
                abs(item.start - prior.start) <= 0.75
                and item.label.split()[-1:] == prior.label.split()[-1:]
                and (bool(canonical & set(self._canonical_tracks(prior.track_ids)))
                     or (not canonical and not prior.track_ids))
                for prior in result
            )
            if same_occurrence:
                continue
            if not item.track_ids and any(
                    abs(item.start - tracked_item.start) <= 0.75
                    for tracked_item in tracked
            ):
                continue
            result.append(item)
        return result

    @staticmethod
    def _safe_label(label: str) -> str:
        if re.search(r"\b(caus(?:e|ed|ing)?|because|triggered|resulted in|led to)\b", label, re.I):
            return "a candidate event"
        return label

    def _resolve_gap(self, first: EntitySpec, second: EntitySpec, rows,
                     question: str) -> TimedInterval | None:
        def resolve(entity: EntitySpec) -> list[TimedInterval]:
            hits = self._sql_resolve([entity], rows)
            if not hits:
                hits = self._audio_resolve([entity], rows)
            if not hits:
                hits = self._clip_resolve([entity])
            return hits

        first_hits, second_hits = resolve(first), resolve(second)
        candidates = [(a, b) for a in first_hits for b in second_hits if a.event_ids != b.event_ids]
        if not candidates:
            return None
        left, right = min(candidates, key=lambda pair: abs(pair[1].start - pair[0].end))
        earlier, later = sorted((left, right), key=lambda item: item.start)
        gap = max(0.0, later.start - earlier.end)
        return TimedInterval(start=earlier.start, end=later.end,
                             label=f"{earlier.label} followed by {later.label}",
                             source=max((left.source, right.source),
                                        key=lambda source: {"audio": 4, "rule": 3, "detector": 3,
                                                            "clip": 2, "vlm": 1}[source]),
                             confidence=min(left.confidence, right.confidence),
                             track_ids=list(dict.fromkeys(left.track_ids + right.track_ids)),
                             event_ids=list(dict.fromkeys(left.event_ids + right.event_ids)),
                             metadata={"gap_seconds": gap,
                                       "uncertainty": max(float(left.metadata.get("uncertainty", 0)),
                                                          float(right.metadata.get("uncertainty", 0)))})

    def _canonical_tracks(self, tracks: list[str]) -> list[str]:
        return list(dict.fromkeys(merged_id(item, self.merges) for item in tracks))

    def _zone_caveat(self, question: str) -> str | None:
        if not re.search(r"\b(zone|area|inside|within|restricted|near)\b", question.lower()):
            return None
        rows = self.db.rows("SELECT name FROM zones WHERE video_id=? AND json_array_length(polygon_json)>=3",
                            (self.video_id,))
        phrases = [phrase for phrase in re.findall(
            r"(?:in|into|inside|within|near|entered)\s+(?:the\s+)?([\w -]+?)(?:\s+after|\s+before|\?|$)",
            question.lower())]
        if rows and (not phrases or any(any(name.lower() in phrase or phrase in name.lower()
                                            for row in rows for name in [row["name"]]) for phrase in phrases)):
            return None
        return "No confirmed zone geometry is available; this uses the best candidate without asserting zone entry."
