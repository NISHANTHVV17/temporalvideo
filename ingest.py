"""PTS-preserving video ingestion, shot detection, and per-shot stabilization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import cv2
import numpy as np


@dataclass
class FrameSample:
    pts: float
    frame: np.ndarray
    shot_id: int
    to_reference: np.ndarray
    shot_cut: bool = False


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int
    time_base: float
    audio_streams: int


def probe_video(path: str | Path) -> VideoInfo:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        width, height = stream.width, stream.height
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        time_base = float(stream.time_base)
        audio_streams = len(container.streams.audio)
    return VideoInfo(duration, width, height, time_base, audio_streams)


def _histogram(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _resize(frame: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, max_dimension / max(width, height))
    if scale < 1:
        return cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                          interpolation=cv2.INTER_AREA)
    return frame


def iter_video(path: str | Path, sample_fps: float = 3.0, max_dimension: int = 640,
               scene_cut_threshold: float = 0.42, start_time: float = 0.0,
               end_time: float | None = None) -> Iterator[FrameSample]:
    """Yield sampled decoded frames using source PTS; frames without PTS are skipped."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required for true frame-PTS indexing; install requirements.txt") from exc
    container = av.open(str(path))
    stream = container.streams.video[0]
    interval = 1.0 / sample_fps
    next_sample = max(0.0, start_time)
    if start_time > 0:
        container.seek(int(start_time / float(stream.time_base)), stream=stream, backward=True)
    shot_id = -1
    previous_hist: np.ndarray | None = None
    previous_gray: np.ndarray | None = None
    previous_pts: float | None = None
    reference_from_previous = np.eye(3, dtype=np.float64)
    orb = cv2.ORB_create(nfeatures=700)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        for decoded in container.decode(stream):
            if decoded.pts is None:
                continue
            pts = float(decoded.pts * decoded.time_base)
            if pts < start_time:
                continue
            if end_time is not None and pts > end_time:
                break
            if pts + 1e-9 < next_sample:
                continue
            next_sample = pts + interval
            frame = _resize(decoded.to_ndarray(format="bgr24"), max_dimension)
            hist = _histogram(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cut = previous_hist is None
            if previous_hist is not None:
                hist_delta = cv2.compareHist(
                    previous_hist, hist, getattr(cv2, "HISTCMP_BHATTACHARYA", 3))
                keypoints_a, desc_a = orb.detectAndCompute(previous_gray, None)
                keypoints_b, desc_b = orb.detectAndCompute(gray, None)
                good_matches = []
                if desc_a is not None and desc_b is not None and len(desc_a) and len(desc_b):
                    pairs = matcher.knnMatch(desc_b, desc_a, k=2)
                    good_matches = [pair[0] for pair in pairs if len(pair) >= 2
                                    and pair[0].distance < 0.72 * pair[1].distance]
                feature_ratio = len(good_matches) / max(1, min(len(keypoints_a or []), len(keypoints_b or [])))
                cut = hist_delta >= scene_cut_threshold and feature_ratio < 0.12

            if cut:
                shot_id += 1
                reference_from_previous = np.eye(3, dtype=np.float64)
                transform = np.eye(3, dtype=np.float64)
            else:
                transform = np.eye(3, dtype=np.float64)
                if len(good_matches) >= 8:
                    source = np.float32([keypoints_b[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    target = np.float32([keypoints_a[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
                    if homography is not None and inliers is not None and int(inliers.sum()) >= 6:
                        transform = reference_from_previous @ homography
                reference_from_previous = transform

            yield FrameSample(pts=pts, frame=frame, shot_id=shot_id,
                              to_reference=reference_from_previous.copy(), shot_cut=cut)
            previous_hist, previous_gray, previous_pts = hist, gray, pts
    finally:
        container.close()
