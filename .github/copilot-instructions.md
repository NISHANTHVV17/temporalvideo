# TemporalVideoQA

- Keep indexing CPU-first; all learned models and hosted APIs must be optional.
- Preserve source timestamps from frame PTS. Never derive video time from frame number or nominal FPS.
- Answers must pass the Pydantic schema and include a non-empty real timestamp interval, evidence IDs, confidence, source, and uncertainty.
- Describe temporal sequence, not causation. Any causal-sounding question must return preceding events and explicitly distinguish plausibility from fact.
- Keep entity/event logic domain-agnostic; question-derived classes may be used only for on-demand open-vocabulary detection.
- Run focused tests after edits. Synthetic tests must not download datasets or models.
- Treat heuristic events as evidence candidates, not domain-specific facts.
