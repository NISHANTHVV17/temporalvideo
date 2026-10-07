"""Render eval annotations as a PTS-preserving visual overlay video."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import av
import cv2


def render(video_path: Path, records: list[dict], output_path: Path) -> None:
    intervals = sorted(records, key=lambda item: float(item["t_start"]))
    source = av.open(str(video_path))
    output = av.open(str(output_path), mode="w")
    input_stream = source.streams.video[0]
    output_stream = output.add_stream("libx264", rate=input_stream.average_rate or 25)
    output_stream.width, output_stream.height = input_stream.width, input_stream.height
    output_stream.pix_fmt = "yuv420p"
    active = []
    next_interval = 0
    try:
        for frame in source.decode(input_stream):
            if frame.pts is None:
                continue
            pts = float(frame.pts * frame.time_base)
            while next_interval < len(intervals) and float(intervals[next_interval]["t_start"]) <= pts:
                active.append(intervals[next_interval])
                next_interval += 1
            active = [item for item in active if float(item["t_end"]) >= pts]
            image = frame.to_ndarray(format="bgr24")
            for index, item in enumerate(active):
                label = f"{item.get('answer', item.get('question', 'event'))} " \
                        f"[{float(item['t_start']):.2f}-{float(item['t_end']):.2f}s]"
                cv2.putText(image, label[:110], (12, 28 + index * 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (30, 245, 140), 2, cv2.LINE_AA)
            converted = av.VideoFrame.from_ndarray(image, format="bgr24")
            converted.pts, converted.time_base = frame.pts, frame.time_base
            for packet in output_stream.encode(converted):
                output.mux(packet)
        for packet in output_stream.encode():
            output.mux(packet)
    finally:
        source.close()
        output.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gt_json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("overlays"))
    args = parser.parse_args()
    records = json.loads(args.gt_json.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for record in records:
        grouped[str(Path(record["video"]).resolve())].append(record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source, items in grouped.items():
        path = Path(source)
        target = args.output_dir / f"{path.stem}.ground-truth.mp4"
        render(path, items, target)
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
