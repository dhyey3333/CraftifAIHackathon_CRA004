# Argonyx on PipeGen: privacy-first threat detection at the edge

Submission for the HackwithBangalore 4.0 wildcard (Embedded SF: CraftifAI
Physical AI Hackathon problem statement), **Track A: PipeGen**, under
**#12 Open Innovation**. It's closest to suggested track #1 (Workplace Safety
Sentinel).

## The problem

Security cameras record everything and understand nothing. Someone has to watch
the feed to notice a bag left behind, a person in a no-go zone, a crowd building
up, or someone who has fallen and not got up. Meanwhile the raw video, faces
included, gets stored and shipped around.

Argonyx watches the feed, raises ranked alerts an operator can act on, and keeps
faces out of everything it shows or stores.

## What PipeGen does in it

PipeGen is the eyes of the system. All detection and tracking run in the
pipeline PipeGen built, as a TensorRT engine on the GPU. Argonyx no longer runs
YOLO for this; its rules engine, severity scoring and dashboard work from
PipeGen's output.

```
video / camera ──► PipeGen pipeline (DeepStream 7.1 container, TensorRT FP16, RTX 3050 laptop GPU)
                     YOLO11s (COCO): person, backpack, handbag, suitcase
                     multi-object tracking: a persistent ID per object
                     head redaction: solid box + blur on the top of every person box
                     ├─► MQTT  localhost:1883  argonyx/detections   one JSON message per frame
                     └─► RTSP  rtsp://<host>:8554/argonyx          redacted, annotated video
Argonyx (backend/detection/pipegen_detector.py)
     MQTT objects ─► rules: unattended object, restricted-zone intrusion, loitering,
                     crowd surge, wrong direction, person down
                  ─► severity ranking ─► alerts with evidence frames ─► operator dashboard
     RTSP video   ─► dashboard view and evidence images (already redacted by PipeGen)
```

| PipeGen stage | What we chose |
|---|---|
| Requirement | People and bag detection with tracking for a security camera; per-frame JSON over MQTT; RTSP video; heads covered before any video is displayed, saved or streamed |
| Scene | Fixed camera, small/far targets (no frame skipping), privacy: faces |
| Pipeline IR | input → preprocess (letterbox) → inference (YOLO11s) → postprocess → tracking → solid-box head redaction → head blur → RTSP, with MQTT off the tracker (`docs/evidence/10_pipeline_ir_graph.png`) |
| Model | Catalog: AbandonedObjectDetection (YOLO11s, COCO, 640×640) |
| Compiler | TensorRT **FP16**: 242 layers, 28.6 MB, built in 24.5 s; INT8 (MinMax) candidate also built: 14.8 MB |
| Run | `vtest_h264.mp4` (768×576, 10 fps); the generated app is `pipegen/deepstream/run.sh` (nvurisrcbin → nvinfer TensorRT → nvtracker → privacy redaction → MQTT → nvdsosd → H.264 → RTSP) |

## What Argonyx does with PipeGen's detections

PipeGen answers "what is where, and which one is it" (boxes and tracker IDs
for people, bags and vehicles). Argonyx turns that into threats an operator
can act on:

| Capability | How | In the PipeGen demo |
|---|---|---|
| Restricted-zone intrusion | tracked person enters a no-go polygon | **fired: 14 HIGH alerts, 12 people** |
| Crowd threshold / crowd surge | zone head-count over a limit / rising fast vs its rolling average | **fired: 4 threshold, 1 HIGH surge** |
| Loitering | a track stays still in a zone too long | **fired: 3** |
| Unattended object | a bag stays put after its owner walks away | runs; the clip has no abandoned bag |
| Wrong-direction movement | track heading vs the zone's allowed direction | runs once a zone has a direction set |
| Person down (fall) | lying box + pose, after being upright (UR Fall: 10/15 falls, 0/10 daily activities, measured earlier with YOLO) | runs; the clip has no fall |
| Fight, hands raised, crowd panic | pose + a learned CLIP fight classifier | runs; not in this clip. Pose-based checks see redacted heads, so hands-raised (wrists vs nose) is weaker here |
| Severity ranking | weighted by type, bonus when alerts co-occur | every alert is scored (e.g. intrusion 75, crowd 80) |
| Evidence + audit | each alert keeps its (redacted) frame; hash-chained audit log | yes |
| Multi-camera, re-ID, search, map | up to 4 cameras, cross-camera re-identification, "Ask Vigil" search, camera map / GPS | available; demo uses one camera |
| Weapons | separate YOLO model | **off in PipeGen mode** (too many false alerts on small models) |

MQTT message, as PipeGen's compiled DeepStream publisher writes it (boxes in
DeepStream's frame size, stated in the message; `"id": null` until the tracker
confirms an object):

```json
{"frame": 412, "source_id": 0, "pts_ns": 41200000000, "width": 1920, "height": 1080,
 "objects": [{"id": 7, "class_id": 0, "label": "person", "confidence": 0.8712,
              "bbox": {"left": 960.0, "top": 540.0, "width": 480.0, "height": 540.0}}]}
```

## Measured

| What | Result | Source |
|---|---|---|
| Engine latency, FP16 | **6.373 ms per frame** (mean), about **157 fps**, at **48.4 W**; 1×3×640×640 input, RTX 3050 6 GB laptop | PipeGen TensorRT report (`docs/evidence/11_tensorrt_speed_report.png`) |
| Accuracy after compiling, FP16 | mAP@0.5 **0.4778** vs ONNX 0.4782 (**−0.0004**) on a 30-image COCO 2017 val slice | PipeGen accuracy check |
| Engine size | FP16 28.6 MB · INT8 14.8 MB | PipeGen compiler |
| INT8 speed / accuracy | built, not measured before the deadline | PipeGen |
| Live run, `vtest_h264.mp4` | **795/795 frames, 6,431 objects, every frame published over MQTT at 10 msg/s** (real time for the 10 fps clip), 79.4 s | PipeGen Run output (`pipegen/deepstream/.run_output.txt`) |
| Privacy at the source | up to **5 heads redacted per frame**, reported in each MQTT message (`fields.heads_redacted`) | PipeGen MQTT feed |
| Argonyx on PipeGen's feed | camera at **9.6 fps** (source is 10 fps); **19–22 ms per frame** in Argonyx. Argonyx's own YOLO detector and weapon model are not loaded; only its behaviour models (pose, OpenCLIP) run, on PipeGen's redacted frames | Argonyx `/api/cameras`, audit log |
| Alerts from PipeGen's detections | **22 alerts in ~40 s** with two zones drawn: 14 HIGH restricted-zone intrusions (12 different people, by PipeGen tracker ID), 4 crowd threshold, 1 HIGH crowd surge, 3 loitering, each with an evidence frame | Argonyx `/api/alerts` |

The accuracy slice is small (30 images), so the meaningful number is the change
(−0.0004): compiling to TensorRT did not change the model's answers.

## Privacy (carried over from our PrivAgent project)

- **Redaction at the source.** PipeGen covers every head with a solid box plus
  blur inside the pipeline, on the exact frames it streams, before the video
  leaves it. Argonyx never receives an unredacted picture.
- **Solid boxes, not just blur.** Blur and pixelation can be partly reversed.
  Argonyx's own masking (`backend/privacy.py`, on by default when it runs its
  own YOLO) uses solid boxes and covers a lying person whole, since the head
  could be at either end.
- **Only coordinates go over the network.** MQTT carries IDs, classes and boxes;
  never pixels.
- **A failed frame is never shown raw.** If processing a frame fails with
  privacy on, the dashboard gets a blank frame instead.

Known gap: the optional "Ask Vigil" search keeps person crops for its search
results. Switch that module off in the dashboard for a fully redacted setup.

## What existed before vs. what was built for this hackathon

Argonyx existed before this hackathon (repo `Argonyx-26/T1-TechPoint`): the rules
engine, severity scoring, alert manager, dashboard, multi-camera handling and
their tests.

Built for this submission (branch `pipegen`):
- the PipeGen pipeline itself (requirement, PIR, model, TensorRT engines, run)
- `backend/detection/pipegen_detector.py`: the MQTT bridge, a drop-in for the
  YOLO detector (`ARGONYX_DETECTOR=pipegen`)
- `backend/privacy.py` and its hooks in `pipeline.py` / `cameras.py`
- `tools/run_pipegen.ps1`: starts Argonyx on PipeGen's feed
- the WSL2 setup: Ubuntu 22.04 + Docker + NVIDIA Container Toolkit, Mosquitto,
  MediaMTX (RTSP)
- 16 new tests (`tests/test_pipegen_detector.py`, `tests/test_privacy.py`)

Weapon detection is switched off in PipeGen mode. Small weapon models raise
too many false alerts to put in front of an operator (see README).

## Run it

1. WSL2 Ubuntu 22.04 with Docker, NVIDIA Container Toolkit, `mosquitto` and
   MediaMTX (systemd service, RTSP on :8554). PipeGen 0.5.1 started with
   `env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 pipegen`. WSLg advertises
   Wayland, which PipeGen refuses.
2. In PipeGen, run the pipeline (`run.sh` in the PipeGen export folder).
3. On Windows: `powershell -ExecutionPolicy Bypass -File tools\run_pipegen.ps1`,
   then open `http://127.0.0.1:8000/dashboard`.

Why the RTSP URL uses the WSL VM's IP: MQTT reaches Windows on `localhost`, but
RTSP video only plays from the VM's own address (measured: `localhost` and
`127.0.0.1` open the stream but never deliver a frame). The launcher looks the
address up each time, because it changes when WSL restarts.

## Evidence

- `docs/evidence/`: **Pipeline IR node graph** (`10_…`), **TensorRT speed
  report** (`11_…`), PipeGen flow draft, catalog match, requirement checks,
  TensorRT FP16 build card, accuracy check, INT8 calibration, the live
  dashboard and alert feed, and an Argonyx alert evidence frame (`07_alert_evidence_intrusion.jpg`: PipeGen's boxes and
  tracker IDs, heads covered by PipeGen, Argonyx's zones on top)
- `pipegen/`: the PipeGen export: `PipelineIR.json`, the generated DeepStream
  app (`deepstream/run.sh`, configs, parsers incl. `mqtt_publish.cpp` and
  `privacy_render_process.cpp`), model-stage log and benchmark profiles, and
  the Run output. TensorRT engines, ONNX and built `.so` files are left out:
  they are GPU-specific and PipeGen rebuilds them.
