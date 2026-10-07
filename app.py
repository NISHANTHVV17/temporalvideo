"""Streamlit workflow for upload, zone drawing, indexing, and timestamped QA."""
from __future__ import annotations

import json
import html
import tempfile
from pathlib import Path

import streamlit as st

from cli import ask_video, index_video, read_config
from ingest import iter_video, probe_video


st.set_page_config(page_title="TemporalVideoQA", layout="wide")
st.title("TemporalVideoQA")
config = read_config()
upload = st.file_uploader("Video", type=["mp4", "mov", "mkv", "avi", "webm"],
                           help="Live uploads are limited to two minutes.")
if upload is not None:
    if upload.size > int(config["app"]["max_upload_bytes"]):
        st.error("This upload exceeds the configured size limit.")
        st.stop()
    suffix = Path(upload.name).suffix or ".mp4"
    temp_path = Path(tempfile.gettempdir()) / f"temporalvideo-upload{suffix}"
    temp_path.write_bytes(upload.getvalue())
    try:
        info = probe_video(temp_path)
    except Exception as exc:
        st.error(f"Could not inspect this video: {exc}")
        st.stop()
    if info.duration > float(config["app"]["max_upload_seconds"]):
        st.error("Uploads in this UI are capped at two minutes. Use the CLI for longer videos.")
        st.stop()

    left, right = st.columns([1.35, 1])
    with left:
        st.video(upload.getvalue())
    with right:
        st.caption(f"{info.width} x {info.height} | {info.duration:.1f}s | {info.audio_streams} audio stream(s)")
        first = next(iter_video(temp_path, sample_fps=0.2, max_dimension=960), None)
        zone_json = st.text_area("Zones and lines (JSON)",
                                 value='{"zones": [], "lines": []}', height=120,
                                 help='Zones use normalized polygon points [[x,y],...]; lines use start/end points.')
        try:
            geometry = json.loads(zone_json)
        except json.JSONDecodeError as exc:
            geometry = None
            st.error(f"Invalid zone JSON: {exc}")
        if first is not None:
            try:
                from PIL import Image
                from streamlit_drawable_canvas import st_canvas
                import cv2
                import numpy as np
                rgb = cv2.cvtColor(first.frame, cv2.COLOR_BGR2RGB)
                canvas = st_canvas(background_image=Image.fromarray(rgb), drawing_mode="polygon",
                                   height=rgb.shape[0], width=rgb.shape[1],
                                   stroke_width=3, stroke_color="#ef4444", fill_color="rgba(239,68,68,0.18)",
                                   key="zone-canvas")
                if canvas.json_data and canvas.json_data.get("objects"):
                    objects = canvas.json_data["objects"]
                    polygons = []
                    for index, item in enumerate(objects):
                        raw_points = item.get("points")
                        if not raw_points and item.get("path"):
                            raw_points = [{"x": command[1], "y": command[2]}
                                          for command in item["path"]
                                          if command and command[0] in {"M", "L"} and len(command) >= 3]
                        if not raw_points:
                            continue
                        scale_x, scale_y = float(item.get("scaleX", 1)), float(item.get("scaleY", 1))
                        left_px, top_px = float(item.get("left", 0)), float(item.get("top", 0))
                        points = [[max(0, min(1, (left_px + float(p["x"]) * scale_x) / rgb.shape[1])),
                                   max(0, min(1, (top_px + float(p["y"]) * scale_y) / rgb.shape[0]))]
                                  for p in raw_points]
                        if len(points) >= 3:
                            polygons.append({"name": f"zone-{index + 1}", "polygon": points,
                                             "shot_id": first.shot_id})
                    if polygons:
                        geometry = geometry or {"lines": []}
                        geometry["zones"] = polygons
            except ImportError:
                st.caption("Install streamlit-drawable-canvas to draw zones; JSON entry remains available.")

    if st.button("Index video", type="primary", disabled=geometry is None):
        zone_path = None
        if geometry and (geometry.get("zones") or geometry.get("lines")):
            zone_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
            json.dump(geometry, zone_file)
            zone_file.close()
            zone_path = zone_file.name
        with st.spinner("Indexing sampled frames and audio..."):
            video_id, database = index_video(temp_path, force=False, zone_path=zone_path)
        st.success(f"Index ready: {video_id}")
        st.session_state["indexed_video"] = str(temp_path)
        st.session_state["indexed_id"] = video_id
        st.session_state["database"] = str(database)

    if st.session_state.get("indexed_video") == str(temp_path):
        question = st.text_input("Question", placeholder="What happened right before the loud sound?")
        if st.button("Ask", disabled=not question.strip()):
            with st.spinner("Resolving indexed evidence..."):
                answer = ask_video(temp_path, question)
            st.markdown(f"**{answer.render(info.duration)}**")
            st.caption(f"Source: {answer.timestamp_source} | confidence {answer.confidence:.2f} | "
                       f"uncertainty +/-{answer.uncertainty_seconds:.1f}s")
            if answer.caveat:
                st.warning(answer.caveat)
            from schema import format_timestamp
            timestamp = (f"{format_timestamp(answer.t_start, info.duration)}-"
                         f"{format_timestamp(answer.t_end, info.duration)}")
            st.markdown(f'<a href="#answer-clip">{html.escape(timestamp)}</a>', unsafe_allow_html=True)
            st.markdown('<div id="answer-clip"></div>', unsafe_allow_html=True)
            st.video(upload.getvalue(), start_time=int(answer.t_start), end_time=max(int(answer.t_start) + 1, int(answer.t_end)))
            if answer.preceding_events:
                st.subheader("Preceding events")
                for event in answer.preceding_events:
                    st.write(f"{event.description} [{format_timestamp(event.start, info.duration)}-"
                             f"{format_timestamp(event.end, info.duration)}] · {event.relation_rating}")
