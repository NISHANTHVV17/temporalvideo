"""Boundary refinement by PTS-based high-rate re-decode and evidence alignment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ingest import iter_video
from schema import TimedInterval


@dataclass
class RefinedWindow:
    start: float
    end: float
    source: str
    uncertainty_seconds: float
    method: str


def refine_window(video_path: str | Path, interval: TimedInterval, fps: float = 12.0,
                  padding: float = 1.0) -> RefinedWindow:
    """Re-decode around a candidate. Without a matching local rule, retain coarse bounds honestly."""
    start = max(0.0, interval.start - padding)
    end = max(interval.end + padding, start + 0.05)
    samples = iter_video(video_path, sample_fps=fps, max_dimension=960,
                         start_time=start, end_time=end)
    decoded = list(samples)
    if not decoded:
        return RefinedWindow(interval.start, max(interval.start, interval.end), interval.source,
                             max(5.0, interval.end - interval.start), "candidate retained; no refinable frames")
    # The candidate remains the source of truth until a semantic-specific verifier can tighten it.
    uncertainty = min(10.0, max(0.5, 1.0 / fps + interval.end - interval.start))
    return RefinedWindow(interval.start, interval.end, interval.source, uncertainty,
                         f"PTS re-decode at {fps:g} fps; no boundary-specific model available")
