"""Convert manually downloaded VIRAT, MOT17, or Charades-STA subsets to eval JSON.

Dataset licenses and access vary. This script intentionally does not download data:
obtain each dataset from its official provider and follow its terms before use.
"""
from __future__ import annotations

import argparse
import csv
import re
from configparser import ConfigParser
from pathlib import Path

import av
import cv2


def frame_pts(video_path: Path) -> list[float]:
    values = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is not None:
                values.append(float(frame.pts * frame.time_base))
    return values


def convert_charades(root: Path) -> list[dict]:
    records = []
    for annotation_file in root.rglob("Charades-STA_*.txt"):
        for line in annotation_file.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s*(\S+)\s+([\d.]+)\s+([\d.]+)\s+(.+?)\s*$", line)
            if not match:
                continue
            video_id, start, end, caption = match.groups()
            videos = list(root.rglob(f"{video_id}.mp4"))
            if not videos:
                continue
            records.append({"video": str(videos[0]), "question": "Describe the event: " + caption,
                            "answer": caption, "t_start": float(start), "t_end": float(end)})
    return records


def _make_mot_video(sequence: Path, video_output_dir: Path) -> Path | None:
    parser = ConfigParser()
    parser.read(sequence / "seqinfo.ini", encoding="utf-8")
    section = parser["Sequence"]
    image_dir = sequence / section.get("imDir", "img1")
    images = sorted(image_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    if not images:
        return None
    video_output_dir.mkdir(parents=True, exist_ok=True)
    output = video_output_dir / f"{sequence.name}.mp4"
    if output.exists():
        return output
    sample = cv2.imread(str(images[0]))
    if sample is None:
        return None
    fps = float(section.get("frameRate", "30"))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (sample.shape[1], sample.shape[0]))
    if not writer.isOpened():
        return None
    try:
        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is not None:
                writer.write(frame)
    finally:
        writer.release()
    return output


def convert_mot17(root: Path, video_output_dir: Path) -> list[dict]:
    records = []
    for sequence in root.rglob("seqinfo.ini"):
        parser = ConfigParser()
        parser.read(sequence, encoding="utf-8")
        section = parser["Sequence"]
        video_path = sequence.parent / section.get("name", sequence.parent.name) / "video.mp4"
        if not video_path.exists():
            video_path = sequence.parent / "video.mp4"
        if not video_path.exists():
            video_path = _make_mot_video(sequence.parent, video_output_dir)
        annotation = sequence.parent / "gt" / "gt.txt"
        if video_path is None or not video_path.exists() or not annotation.exists():
            continue
        pts = frame_pts(video_path)
        by_id: dict[int, list[int]] = {}
        with annotation.open(newline="", encoding="utf-8") as source:
            for row in csv.reader(source):
                if len(row) < 6:
                    continue
                frame_number, track_id = int(float(row[0])), int(float(row[1]))
                if 1 <= frame_number <= len(pts):
                    by_id.setdefault(track_id, []).append(frame_number - 1)
        for track_id, frames in by_id.items():
            records.append({"video": str(video_path),
                            "question": f"When does annotated track {track_id} appear?",
                            "answer": f"track {track_id}", "t_start": pts[min(frames)],
                            "t_end": pts[max(frames)]})
    return records


def convert_virat(root: Path) -> list[dict]:
    records = []
    for annotation in root.rglob("*.csv"):
        with annotation.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                continue
            fields = {name.lower(): name for name in reader.fieldnames}
            start_key = next((fields[key] for key in fields if "start" in key and "time" in key), None)
            end_key = next((fields[key] for key in fields if "end" in key and "time" in key), None)
            if not start_key or not end_key:
                continue
            video_path = next(iter(root.rglob(annotation.stem + ".mp4")), None)
            if video_path is None:
                video_path = next(iter(root.rglob(annotation.stem + ".avi")), None)
            if video_path is None:
                continue
            for row in reader:
                try:
                    start, end = float(row[start_key]), float(row[end_key])
                except (TypeError, ValueError):
                    continue
                label_key = next((fields[key] for key in fields if "event" in key or "type" in key), None)
                label = str(row.get(label_key, "annotated event")) if label_key else "annotated event"
                records.append({"video": str(video_path), "question": f"Find the {label} event.",
                                "answer": label, "t_start": start, "t_end": end})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["virat", "mot17", "charades-sta"], required=True)
    parser.add_argument("--root", type=Path, required=True,
                        help="Path to a manually downloaded and license-compliant dataset subset")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    converter = {"virat": convert_virat, "mot17": lambda root: convert_mot17(root, args.out.parent / "mot17_eval_videos"),
                 "charades-sta": convert_charades}[args.dataset]
    records = converter(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import json
    args.out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} evaluation records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
