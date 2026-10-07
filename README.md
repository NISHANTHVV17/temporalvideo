# TemporalVideoQA

TemporalVideoQA builds a structured temporal index before answering natural-language questions. The CPU-first base uses PyAV PTS timestamps, OpenCV scene/motion analysis, SQLite evidence, and optional model backends. It does not send full videos to a VLM. Model-backed judgments operate only on short candidate windows.

## Quick Start

Requirements: Python 3.10+, FFmpeg available on `PATH` for audio extraction, and a CPU-capable OpenCV/PyAV install.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python cli.py index /path/to/video.mp4
python cli.py ask /path/to/video.mp4 "What happened right before the loud sound?"
python cli.py eval ground_truth.json
streamlit run app.py
```

The Streamlit UI accepts uploads up to 120 seconds, allows normalized zone polygons to be drawn or supplied as JSON, indexes the video, and plays the selected time window. Longer videos are supported through the CLI. The first index run is linear; later runs reuse an index keyed by SHA-256. A checkpoint is written during indexing. Resume replays decoded frames to restore tracker state, then persists from the last checkpoint; it favors deterministic continuity over fast seek and may therefore repeat decode work after interruption.

## What Runs By Default

- Ingestion samples at 3 FPS and scales the longest dimension to 640 pixels. Each sampled time comes from decoded frame PTS and time base, not frame number divided by FPS. Frames without PTS are skipped.
- Histogram difference plus ORB feature matching detects probable shot cuts. Shot-local tracking and stabilization reset at a cut. ORB homographies map sampled coordinates to the shot reference frame. Track association uses stabilized positions, constant-velocity prediction, global assignment, and available appearance embeddings; the default local re-association window is 5 seconds.
- In `tracking.backend: auto`, Ultralytics is attempted with the configured model (default `yolo11n.pt`) on CPU using BoT-SORT; Ultralytics may download that model the first time if it is not cached. Set `tracking.backend: openvino` to request an OpenVINO export. If the learned detector cannot be loaded, a generic foreground-contour proposal detector runs and labels proposals `unknown`, so named-class questions become less reliable.
- Track appearance defaults to color histograms; optional CLIP embeddings can be used when Transformers and model weights are present. Global ReID merges are logged with score and whether the merge crossed a cut.
- Rule-derived track, zone, stationary, motion, interaction, line-crossing, audio, and candidate pickup/putdown events are stored in SQLite. User geometry is JSON in normalized coordinates, for example:

```json
{
  "zones": [{"name": "entry", "polygon": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.9], [0.1, 0.9]]}],
  "lines": [{"name": "threshold", "start": [0.2, 0.5], "end": [0.8, 0.5]}]
}
```

- Audio extraction uses FFmpeg and spectral flux, RMS, tonalness, and silence checks. WebRTC VAD is optional. Videos without audio are skipped without error.
- Frame embeddings are optional and indexed at 1 FPS. Set `NVIDIA_EMBEDDING_API_KEY` to use NVIDIA's multimodal Nemotron Embed VL model (`nvidia/llama-nemotron-embed-vl-1b-v2`) for frame and text-query embeddings in the same retrieval space; set `semantic.provider: local` to use locally cached CLIP instead. Summarization defaults to TwelveLabs Pegasus 1.6 using `TWELVELABS_API_KEY`: it sends the MP4 inline once and saves the model's timestamped event lines beside the evidence database as `<video-id>.summary.txt`. Follow-up `ask` questions are answered locally from that saved timeline; the video is not uploaded again, and questions unsupported by the log are rejected rather than guessed. A local Qwen, Gemini, or hosted NVIDIA backend can still be selected for summary generation.
- The rule-based planner is always available. With a hosted client configured, an LLM can produce a strict JSON query plan that is validated by Pydantic; otherwise deterministic parsing is used. Single-answer queries use candidate-first SQL/audio/CLIP/open-vocabulary/VLM resolution. Count, order, and "find every/all" queries exhaustively inspect indexed evidence and semantic windows across the full video; on-demand open-vocabulary detection samples the full video at 2 FPS and consolidates detections into tracks. VLM prompts include each sampled frame's source PTS and require absolute PTS timestamps and explicit no-match responses. Count and list answers include per-event timestamps, and duration predicates filter intervals by their measured PTS span. A low-confidence best-candidate window is returned when nothing matches. Every answer has a positive-duration PTS interval, track/event IDs, confidence, source, and uncertainty, and renders as `answer [mm:ss-mm:ss] tracks=... events=... confidence=... source=... uncertainty=...`. Videos over one hour use `hh:mm:ss`.
- Causal wording is not emitted. Cause/effect-style questions receive an ordered preceding-events list with timestamps and, when a VLM is configured, plausibility ratings that are explicitly not causal proof.

## Requirement Demonstration

| Requirement | How to demonstrate it | Scope and caveat |
|---|---|---|
| Timestamp every answer | Analyze the video, then ask with `python cli.py ask video.mp4 "..."`; output includes a supporting interval, source, confidence, and summary-line evidence IDs. | Answers only use details recorded in the saved summary. Unrecorded details are not inferred. |
| Order, before/after, and elapsed time | Ask `What happened right before ...?`, `... after ...?`, or `How long between ... and ...?`. | Reports observed sequence; it does not claim one event caused another. |
| Restricted-area entry | Supply a named polygon in `--zones zones.json` or draw it in the UI, then ask who entered it. | Use a real detector label; generic fallback proposals are `unknown`. |
| Repeated stops and counts | Ask `How many times did the machine stop?`; each occurrence is returned with its timestamp. | The stop is a motion/event heuristic. Define what counts as unexpected and validate against labeled examples. |
| Stationary objects over two minutes | Ask for objects untouched for more than 2 minutes; stationary intervals are duration-filtered using PTS. | Occlusion or missed detections can split an interval. |
| People/object tracking | Auto mode attempts CPU YOLO11n with BoT-SORT; persistent tracker IDs and appearance-based cross-shot re-identification are retained. | Identity can still be wrong after long occlusion, similar appearances, or leaving and re-entering. Review reported merge scores. |
| Camera motion and shot changes | Ingestion estimates shot cuts and ORB-based stabilization; tracker association uses stabilized coordinates. | Strong blur, low texture, or abrupt viewpoint changes reduce reliability. |
| Long videos | Use CLI `summarize` and `ask`; each question reads the generated local timeline. | Streamlit uploads are capped at 120 seconds; TwelveLabs analysis is subject to provider limits. |
| Measured evaluation | Convert licensed Charades-STA, MOT17, or VIRAT annotations, then run `python cli.py eval ground_truth.json`. | Evaluation reflects only the supplied dataset/subset; accuracy claims need a measured result. |

Offline synthetic regression coverage can be run with `python -m pytest -vv tests`. It exercises PTS requirements, ordering/counts, restricted-zone entry, 140-second stationary detection, occlusion, camera motion, and long-window search without downloading datasets or model weights.

## Optional Model Backends

The base install does not download model weights. Optional integrations are intentionally lazy or guarded:

- YOLO11/OpenVINO: install Ultralytics (and `lap` for BoT-SORT); auto mode attempts CPU YOLO11n and downloads `yolo11n.pt` on first use if needed. OpenVINO export is requested with `tracking.backend: openvino`.
- YOLO-World: loaded only for on-demand, question-conditioned detection in candidate windows; provide local weights with `TEMPORALVIDEO_YOLOWORLD_MODEL`.
- Local CLIP: install `transformers` and `torch`, and ensure the configured model weights are already available locally. Automatic indexing does not download them.
- Hosted VLM/planner: install `openai` and set `NVIDIA_VIDEO_API_KEY` to use NVIDIA NIM (`https://integrate.api.nvidia.com/v1`) for video-window answering and planning. Set `GROQ_API_KEY` to use Groq's OpenAI-compatible API for text-only follow-up answers grounded in the saved TwelveLabs event log; the default model is `openai/gpt-oss-20b` and can be changed with `GROQ_MODEL`. Keys may be placed in a project-local `.env` file, which is loaded automatically and ignored by Git. The default vision model is `meta/llama-3.2-11b-vision-instruct`; set `TEMPORALVIDEO_VLM_MODEL` to override it.
- NVIDIA semantic embeddings: set `NVIDIA_EMBEDDING_API_KEY` for `nvidia/llama-nemotron-embed-vl-1b-v2`. Set `TEMPORALVIDEO_EMBEDDING_BACKEND=local` or `semantic.provider: local` to use local CLIP. Changing providers automatically invalidates the semantic index; use `python cli.py index video.mp4 --force` to rebuild it.
- WebRTC VAD: install `webrtcvad`.

If these are missing, temporal answers fall back to rule, motion, audio, and appearance evidence with confidence reduced where semantic judgment is required.

## Commands

```bash
python cli.py index video.mp4 [--zones zones.json] [--force]
python cli.py ask video.mp4 "question" [--zones zones.json]
python cli.py summarize video.mp4
python cli.py eval gt.json
python scripts/prepare_data.py --dataset charades-sta --root /data/Charades-STA --out eval.json
python scripts/render_ground_truth.py eval.json --output-dir overlays
python -m pytest -q
```

Evaluation JSON is an array of `{ "video", "question", "answer", "t_start", "t_end" }`. Reports median mean-boundary timestamp error, the percentage within +/-1 second, token-coverage answer accuracy, and a half-credit score (correct answer but wrong time earns 0.5).

`prepare_data.py` converts subsets already downloaded by the user. VIRAT, MOT17, and Charades-STA have dataset-specific terms and access requirements; download manually from the official dataset source and confirm your permitted use. MOT17 frame annotations are mapped to decoded PTS before writing evaluation timestamps.

## File Tree

```text
TemporalVideoQA/
|-- .github/copilot-instructions.md
|-- .gitignore
|-- app.py
|-- audio.py
|-- cli.py
|-- config.yaml
|-- db.py
|-- detect_track.py
|-- events.py
|-- executor.py
|-- ingest.py
|-- planner.py
|-- reid.py
|-- schema.py
|-- semantic.py
|-- verify.py
|-- vlm_client.py
|-- zones.py
|-- requirements.txt
|-- README.md
|-- scripts/
|   |-- prepare_data.py
|   `-- render_ground_truth.py
`-- tests/
    `-- test_pipeline.py
```

## Module Smoke Commands

Run these from the repository root after installing requirements. They import each module without loading optional model weights.

```bash
python -c "import schema"
python -c "import zones"
python -c "import db"
python -c "import ingest"
python -c "import detect_track"
python -c "import reid"
python -c "import events"
python -c "import audio"
python -c "import semantic"
python -c "import planner"
python -c "import executor"
python -c "import verify"
python -c "import vlm_client"
python -c "import cli"
python -m py_compile app.py
python -m pytest tests/test_pipeline.py -q
```

## Known Limits

- The 0.6-1.0x video-length indexing target is an estimate for the configured Intel i5-1335U CPU when the detector is exported to OpenVINO and optional heavyweight semantic models are disabled. Actual speed depends on codec, resolution, thermal limits, and installed inference runtimes; the Python/OpenCV contour fallback can be slower and is not a semantic object detector.
- Generic foreground proposals do not identify arbitrary object categories. Open-vocabulary class quality depends on YOLO-World/OWLv2 availability; neither can guarantee recognition of every phrase or fine-grained action.
- Long-absence ReID is conservative and appearance-only fallback embeddings can merge similar-looking objects or miss the same identity under major appearance/view changes. Every merge has a score, but this is not identity certainty.
- VLM-reported timestamps are generally +/-5-10 seconds until an applicable boundary-specific verifier is available. High-rate re-decode currently narrows sampling uncertainty but does not itself infer semantic start/end boundaries.
- ORB camera stabilization is approximate. Strong parallax, blur, zoom, low texture, and moving foreground can corrupt stabilized coordinates. Zones are shot-specific; moving-camera zones should be reviewed and confirmed by the user.
- Stationary, pick-up/put-down, interaction, and line-crossing records are heuristic evidence. Small static objects that were never detected cannot be tracked reliably; background subtraction only proposes foreground candidates. Speech detection needs optional VAD; audio tone labels do not prove a siren, beep, or alarm class.
- The local Qwen/VLM and hosted VLM are optional. With no working client, subjective judgments such as "unexpectedly" or "delivery" are returned as low-confidence candidates rather than invented facts.
- The current planner is a deterministic fallback, not a general-purpose LLM. Complex count semantics, ambiguous references, and arbitrary compound relations may need VLM/API configuration and user review.
- This is an evidence-indexing baseline, not a guarantee of correct answers to arbitrary questions. The no-model fallback detects motion/foreground, not object meaning or action semantics. For the judging examples involving arbitrary classes (“delivery truck”), subjective states (“unexpectedly”), identity through long occlusions, or named acoustic classes (“safety alarm”), install/configure appropriate detector, appearance, and hosted/local VLM backends and validate them on held-out videos. Without those backends, the answer is intentionally low confidence.
