"""Schema-validated question planning with a deterministic fallback parser."""
from __future__ import annotations

import re

from schema import EntityKind, EntitySpec, Intent, QueryFilters, QueryPlan, RelationSpec


CAUSE_WORDS = ("cause", "because", "led to", "resulted in", "made it", "triggered")


def parse_question(question: str, llm_client=None, default_window: float = 10.0) -> QueryPlan:
    if llm_client is not None:
        try:
            raw = llm_client.plan(question)
            return QueryPlan.model_validate(raw | {"raw_question": question})
        except Exception:
            pass
    return rule_based_plan(question, default_window)


def rule_based_plan(question: str, default_window: float = 10.0) -> QueryPlan:
    text = question.strip()
    lower = text.lower()
    asks_causation = any(word in lower for word in CAUSE_WORDS)
    if re.search(r"\bhow many\b|\bcount\b|\bnumber of times\b", lower):
        intent = Intent.count
    elif re.search(r"\bhow long\s+(?:is it\s+)?between\b", lower):
        intent = Intent.before_after
    elif re.search(r"\b(?:in what order|what order|sequence|then what)\b", lower):
        intent = Intent.order
    elif re.search(r"\bfirst\b", lower):
        intent = Intent.first
    elif re.search(r"\blast\b", lower):
        intent = Intent.last
    elif re.search(r"\bright before\b|\bbefore\b", lower):
        intent = Intent.window_before
    elif re.search(r"\bright after\b|\bafter\b", lower):
        intent = Intent.window_after
    elif re.search(r"\bhow long\b|\bduration\b|\bmore than\s+\d", lower):
        intent = Intent.duration_filter
    elif re.search(r"\border\b", lower):
        intent = Intent.order
    elif re.search(r"\bwho\b|\bwhich person\b|\bsame person\b", lower):
        intent = Intent.identify_track
    else:
        intent = Intent.find_event

    entities = _extract_entities(text)
    relation_match = re.search(r"(.+?)\s+(?<!right\s)(before|after|during)\s+(.+?)[?.]*$", text, re.I)
    relations = []
    if relation_match and len(entities) >= 2:
        relations.append(RelationSpec(a=entities[0].phrase, relation=relation_match.group(2).lower(),
                                      b=entities[1].phrase))
    if relations and intent in {Intent.window_before, Intent.window_after}:
        intent = Intent.find_event
    duration = None
    duration_match = re.search(r"(?:more than|over|at least)\s+(\d+(?:\.\d+)?)\s*(seconds?|minutes?|mins?|hours?)", lower)
    if duration_match:
        amount = float(duration_match.group(1))
        unit = duration_match.group(2)
        duration = amount * (60 if unit.startswith("min") else 3600 if unit.startswith("hour") else 1)
    count_distinct = intent == Intent.count and not re.search(r"\b(?:times|occasions|events)\b", lower)
    return QueryPlan(intent=intent, entities=entities, relations=relations,
                     filters=QueryFilters(min_duration=duration,
                                          count_distinct=count_distinct),
                     raw_question=text, asks_causation=asks_causation,
                     preceding_seconds=default_window)


def _extract_entities(question: str) -> list[EntitySpec]:
    lower = question.lower()
    phrases: list[str] = []
    right_before = re.search(r"\b(?:right\s+)?before\s+(?:the\s+)?(.+?)[?.]*$", lower)
    if lower.startswith("what happened") and right_before:
        phrases.append(right_before.group(1).strip())
    between = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)[?.]*$", lower)
    if between:
        phrases.extend([between.group(1).strip(), between.group(2).strip()])
    temporal = re.search(r"(.+?)\s+(?:right\s+)?(before|after|during)\s+(.+?)[?.]*$", lower)
    if temporal:
        left = re.sub(r"^(which|what|who)\s+", "", temporal.group(1)).strip()
        right = re.sub(r"^(the|a|an)\s+", "", temporal.group(3)).strip()
        if left and left not in {"happened", "right", "just", "immediately"}:
            phrases.append(left)
        if right and right not in phrases:
            phrases.append(right)
    if not temporal and not between:
        patterns = [r"(?:which|what)\s+(?:was\s+)?(?:the\s+)?(.+?\s+(?:happened|occurred|appeared|entered|stopped|started))\s+(?:first|last)(?:[?.]|$)",
                    r"(?:right before|before|after|during|when)\s+(.+?)(?:[?.]|$)",
                    r"(?:how many times did|find every|find all|which person|who)\s+(.+?)(?:[?.]|$)"]
        for pattern in patterns:
            match = re.search(pattern, lower)
            if match:
                phrase = match.group(1).strip()
                phrase = re.sub(r"^(the|a|an)\s+", "", phrase)
                if phrase and phrase not in phrases:
                    phrases.append(phrase)
    if not phrases:
        phrase = re.sub(r"^(what happened|what|describe|find)\s+", "", lower).strip(" ?.")
        if phrase:
            phrases.append(phrase)
    entities = []
    for phrase in phrases[:4]:
        kind = (EntityKind.sound if any(token in phrase for token in ("alarm", "sound", "beep", "speech", "noise"))
                else EntityKind.person if "person" in phrase or "who" in phrase or phrase.startswith("which person")
            else EntityKind.action)
        class_hint = None
        if kind == EntityKind.person:
            class_hint = "person"
        elif kind == EntityKind.sound:
            class_hint = phrase
        entities.append(EntitySpec(phrase=phrase, kind=kind,
                                  class_hint=class_hint or (phrase if len(phrase.split()) <= 3 else None)))
    return entities
