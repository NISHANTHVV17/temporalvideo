"""Audio onset, transient, tonal, speech, and silence event indexing."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import uuid
import wave
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from db import EvidenceDB


def _wav_from_video(video_path: str | Path, ffmpeg: str = "ffmpeg") -> Path | None:
    if shutil.which(ffmpeg) is None:
        return None
    temporary = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temporary.close()
    result = subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(video_path), "-vn",
                             "-ac", "1", "-ar", "16000", temporary.name],
                            capture_output=True, check=False)
    if result.returncode != 0:
        Path(temporary.name).unlink(missing_ok=True)
        return None
    return Path(temporary.name)


def detect_audio_events(video_path: str | Path, video_id: str, db: EvidenceDB,
                        ffmpeg: str = "ffmpeg") -> list[dict[str, Any]]:
    wav_path = _wav_from_video(video_path, ffmpeg)
    if wav_path is None:
        return []
    events: list[dict[str, Any]] = []
    try:
        with wave.open(str(wav_path), "rb") as source:
            rate = source.getframerate()
            samples = np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        if samples.size < rate // 4:
            return []
        frame_size = int(rate * 0.05)
        count = samples.size // frame_size
        frames = samples[:count * frame_size].reshape(count, frame_size)
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        spectrum = np.abs(np.fft.rfft(frames * np.hanning(frame_size), axis=1))
        flux = np.maximum(0, np.diff(spectrum, axis=0)).sum(axis=1)
        flux = np.pad(flux, (1, 0))
        smooth = np.convolve(flux, np.ones(5) / 5, mode="same")
        peaks, _ = find_peaks(smooth, prominence=max(0.02, float(np.std(smooth) * 1.8)), distance=8)
        for peak in peaks:
            start = peak * frame_size / rate
            events.append(_audio_event(video_id, "audio_onset", start, start + 0.05,
                                       0.65, {"spectral_flux": float(smooth[peak])}))
        loud = np.where(rms > max(0.12, float(np.percentile(rms, 90))))[0]
        for frame_index in loud:
            start = frame_index * frame_size / rate
            events.append(_audio_event(video_id, "loud_transient", start, start + 0.05,
                                       0.55, {"rms": float(rms[frame_index])}))
        silent = rms < 0.008
        events.extend(_runs_as_events(video_id, "silence", silent, frame_size / rate, 0.4))
        if len(rms) > 3:
            drops = np.where((rms[1:] < 0.008) & (rms[:-1] > 0.04))[0] + 1
            for index in drops:
                start = index * frame_size / rate
                events.append(_audio_event(video_id, "sudden_silence", start, start + 0.05, 0.6, {}))
        events.extend(_tonal_segments(video_id, spectrum, frame_size, rate))
        events.extend(_vad_segments(video_id, samples, rate, frame_size))
        for item in events:
            db.add_event(item)
        return events
    finally:
        wav_path.unlink(missing_ok=True)


def _audio_event(video_id: str, kind: str, start: float, end: float, confidence: float,
                 meta: dict[str, Any]) -> dict[str, Any]:
    return {"event_id": uuid.uuid4().hex, "video_id": video_id, "track_id": None,
            "class": "sound", "event_type": kind, "t_start": start, "t_end": end,
            "zone": None, "confidence": confidence, "meta_json": json.dumps(meta)}


def _runs_as_events(video_id: str, kind: str, mask: np.ndarray, step: float,
                    minimum: float) -> list[dict[str, Any]]:
    output = []
    indexes = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    for start_index, end_index in zip(indexes[::2], indexes[1::2]):
        start, end = start_index * step, end_index * step
        if end - start >= minimum:
            output.append(_audio_event(video_id, kind, start, end, 0.7, {}))
    return output


def _tonal_segments(video_id: str, spectrum: np.ndarray, frame_size: int,
                    rate: int) -> list[dict[str, Any]]:
    if not spectrum.size:
        return []
    peak_bin = np.argmax(spectrum, axis=1)
    energy = np.max(spectrum, axis=1)
    flatness = np.exp(np.mean(np.log(spectrum + 1e-10), axis=1)) / (np.mean(spectrum, axis=1) + 1e-10)
    mask = (energy > np.percentile(energy, 65)) & (flatness < 0.22)
    return _runs_as_events(video_id, "sustained_tonal_sound", mask, frame_size / rate, 0.25)


def _vad_segments(video_id: str, samples: np.ndarray, rate: int,
                  frame_size: int) -> list[dict[str, Any]]:
    try:
        import webrtcvad
    except ImportError:
        return []
    vad = webrtcvad.Vad(2)
    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    frame_size = int(rate * 0.03)
    flags = []
    for start in range(0, len(pcm) - frame_size, frame_size):
        flags.append(vad.is_speech(pcm[start:start + frame_size].tobytes(), rate))
    return _runs_as_events(video_id, "speech", np.asarray(flags), 0.03, 0.3)
