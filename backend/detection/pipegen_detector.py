"""Detections from the CraftifAI PipeGen pipeline instead of in-process YOLO.

PipeGen compiles the detector to a TensorRT engine and runs it, with
DeepStream's tracker, on the GPU. The generated pipeline publishes each
frame's objects as JSON over MQTT. PipeGenDetector subscribes to that feed
and hands the newest objects to FramePipeline through the same
infer(frame) -> List[Detection] call the YOLO detector answers, so the rules,
alerts and dashboard run unchanged on top of PipeGen.

One topic feeds one camera: run a single camera in this mode.

The DeepStream app PipeGen compiled for us publishes one message per frame:
  {"frame": 412, "source_id": 0, "pts_ns": ..., "width": 1920, "height": 1080,
   "objects": [{"id": 7, "class_id": 0, "label": "person", "confidence": 0.87,
                "bbox": {"left": .., "top": .., "width": .., "height": ..}}]}
with boxes in nvstreammux pixels (the message's width/height) and "id": null
for objects the tracker hasn't confirmed. parse_message() also accepts the
per-frame sketch in PipeGen's Pipeline IR ("tracker_id" / "bounding_box") and
the shapes DeepStream's nvmsgconv produces -- the "minimal" schema
("id|left|top|right|bottom|label" strings) and the full schema (one message
per object, grouped back into frames by timestamp) -- so a regenerated
pipeline doesn't break the bridge.
"""
import json
import threading
import time
from typing import Any, List, Optional, Tuple

from backend import config
from backend.audit import audit
from backend.detection.object_detector import Detection

Box = Tuple[float, float, float, float]
Size = Tuple[int, int]

# DeepStream gives objects its tracker has not assigned an id UINT64_MAX;
# anything this large is "untracked" (Argonyx's TrackManager wants -1 then).
UNTRACKED_FROM = 2 ** 63
# Keys the full schema uses for an object's type-specific block.
FULL_SCHEMA_TYPES = ("person", "face", "bag", "vehicle", "bicycle")


def _class_name(label: Any) -> str:
    name = str(label).strip().lower()
    return config.PIPEGEN_CLASS_MAP.get(name, name)


def _track_id(raw: Any) -> int:
    try:
        tid = int(str(raw).strip())   # string first: float() would round a 64-bit id
    except (TypeError, ValueError):
        try:
            tid = int(float(raw))
        except (TypeError, ValueError):
            return -1
    return -1 if tid < 0 or tid >= UNTRACKED_FROM else tid


def _valid(box: Box) -> Optional[Box]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _box_from(obj: dict) -> Optional[Box]:
    """xyxy box from the shapes we have seen DeepStream / PipeGen emit."""
    b = next((obj[k] for k in ("bounding_box", "bbox", "box", "rect") if k in obj), None)
    if b is None and {"left", "top", "width", "height"} <= obj.keys():
        b = obj
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return _valid(b)   # a bare 4-list is read as x1, y1, x2, y2
    if not isinstance(b, dict):
        return None
    if "topleftx" in b:
        return _valid((b["topleftx"], b["toplefty"], b["bottomrightx"], b["bottomrighty"]))
    if "left" in b and "width" in b:
        x, y = float(b["left"]), float(b["top"])
        return _valid((x, y, x + float(b["width"]), y + float(b["height"])))
    if "x1" in b:
        return _valid((b["x1"], b["y1"], b["x2"], b["y2"]))
    if "x" in b and "w" in b:
        x, y = float(b["x"]), float(b["y"])
        return _valid((x, y, x + float(b["w"]), y + float(b["h"])))
    return None


def _from_minimal(entry: str) -> Optional[Detection]:
    """'id|left|top|right|bottom|label[|confidence][|#|attributes...]'"""
    parts = entry.split("|")
    if len(parts) < 6:
        return None
    box = _valid(parts[1:5])
    if box is None:
        return None
    conf = 1.0
    if len(parts) > 6 and parts[6] != "#":
        try:
            conf = float(parts[6])
        except ValueError:
            pass
    return Detection(track_id=_track_id(parts[0]), cls_name=_class_name(parts[5]), conf=conf, bbox=box)


def _from_dict(obj: dict) -> Optional[Detection]:
    box = _box_from(obj)
    if box is None:
        return None
    label = next((obj[k] for k in ("label", "class", "class_name", "cls", "obj_label", "type") if k in obj), None)
    if label is None:   # full schema: the type is the key of a nested block
        label = next((k for k in FULL_SCHEMA_TYPES if isinstance(obj.get(k), dict)), "object")
    tid = next((obj[k] for k in ("tracker_id", "track_id", "trackingId", "object_id", "id") if k in obj), -1)
    conf = next((obj[k] for k in ("confidence", "conf", "score") if k in obj), 1.0)
    return Detection(track_id=_track_id(tid), cls_name=_class_name(label), conf=float(conf), bbox=box)


def _frame_size(msg: dict) -> Optional[Size]:
    frame = msg.get("frame") if isinstance(msg.get("frame"), dict) else msg
    for wk, hk in (("frame_width", "frame_height"), ("frameWidth", "frameHeight"), ("width", "height")):
        if wk in frame and hk in frame:
            return int(frame[wk]), int(frame[hk])
    return None


def parse_message(payload: Any) -> Tuple[List[Detection], Optional[Size], Optional[str]]:
    """One MQTT payload -> (objects, the pixel size their boxes are in if the
    message says, a frame key). The key is set only for the per-object full
    schema, where several messages make up one frame."""
    msg = json.loads(payload) if isinstance(payload, (bytes, bytearray, str)) else payload
    if isinstance(msg, list):
        msg = {"objects": msg}
    if not isinstance(msg, dict):
        raise ValueError("payload is not a JSON object")
    if isinstance(msg.get("object"), dict):   # full schema: one object per message
        det = _from_dict(msg["object"])
        key = str(msg.get("@timestamp") or msg.get("timestamp") or "") or None
        return ([det] if det else []), _frame_size(msg), key
    detections = []
    for entry in msg.get("objects", msg.get("detections", [])):
        det = _from_minimal(entry) if isinstance(entry, str) else _from_dict(entry) if isinstance(entry, dict) else None
        if det is not None:
            detections.append(det)
    return detections, _frame_size(msg), None


class PipeGenDetector:
    """Newest PipeGen objects, rescaled to the frame Argonyx is analysing."""

    def __init__(self, host: str = None, port: int = None, topic: str = None,
                 frame_size: Size = None, max_age_s: float = None, connect: bool = True):
        self.topic = topic or config.PIPEGEN_MQTT_TOPIC
        self.frame_size = tuple(frame_size or config.PIPEGEN_FRAME_SIZE)
        self.max_age_s = config.PIPEGEN_MAX_AGE_S if max_age_s is None else max_age_s
        # Pixel size of the video Argonyx reads (set by the camera each frame).
        # Argonyx reads PipeGen's own RTSP output, whose pixels are exactly the
        # ones the boxes are in, so this beats the configured fallback.
        self.source_size: Optional[Size] = None
        self._lock = threading.Lock()
        self._objects: List[Detection] = []
        self._size: Optional[Size] = None   # from the message itself, when it says
        self._frame_key: Optional[str] = None
        self._received_at = 0.0
        self.messages = 0
        self.bad_messages = 0
        self._client = None
        if connect:
            self._connect(host or config.PIPEGEN_MQTT_HOST, port or config.PIPEGEN_MQTT_PORT)

    def _connect(self, host: str, port: int):
        import paho.mqtt.client as mqtt   # local import: only this mode needs it

        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:   # paho-mqtt 1.x
            client = mqtt.Client()
        client.on_connect = self._on_connect
        client.on_message = lambda _client, _userdata, msg: self.on_payload(msg.payload)
        client.reconnect_delay_set(min_delay=1, max_delay=10)
        # Async + background loop: Argonyx starts even if the pipeline isn't
        # up yet, and reconnects by itself when the broker comes back.
        client.connect_async(host, port, keepalive=30)
        client.loop_start()
        self._client = client
        audit("MODEL_LOADED", f"PipeGen detector: mqtt://{host}:{port}/{self.topic}")

    def _on_connect(self, client, _userdata, _flags, reason_code, _properties=None):
        client.subscribe(self.topic)
        audit("PIPEGEN_CONNECTED", f"subscribed to {self.topic} (reason {reason_code})")

    def on_payload(self, payload) -> None:
        try:
            detections, size, key = parse_message(payload)
        except (ValueError, TypeError, KeyError):
            with self._lock:
                self.bad_messages += 1
            return
        now = time.time()
        with self._lock:
            same_frame = key is not None and key == self._frame_key and now - self._received_at < self.max_age_s
            self._objects = self._objects + detections if same_frame else detections
            self._frame_key = key
            self._size = size
            self._received_at = now
            self.messages += 1

    def infer(self, frame) -> List[Detection]:
        with self._lock:
            objects, size = list(self._objects), self._size
            age = time.time() - self._received_at
        if not objects or age > self.max_age_s:
            return []
        src_w, src_h = size or self.source_size or self.frame_size
        h, w = frame.shape[:2]
        sx, sy = w / src_w, h / src_h
        return [
            Detection(d.track_id, d.cls_name, d.conf,
                      (d.bbox[0] * sx, d.bbox[1] * sy, d.bbox[2] * sx, d.bbox[3] * sy))
            for d in objects
        ]

    def close(self):
        if self._client is not None:
            self._client.disconnect()
            self._client.loop_stop()
            self._client = None
