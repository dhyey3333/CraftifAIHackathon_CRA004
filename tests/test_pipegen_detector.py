import json
import time

import numpy as np
import pytest

from backend.detection.pipegen_detector import PipeGenDetector, parse_message


def detector(**kw):
    kw.setdefault("frame_size", (1920, 1080))
    kw.setdefault("max_age_s", 1.0)
    return PipeGenDetector(connect=False, **kw)


def test_deepstream_frame_json_payload():
    """The shape PipeGen's compiled DeepStream publisher (frame_json.h) writes:
    boxes in nvstreammux pixels, which the message states as width/height."""
    msg = {"frame": 412, "source_id": 0, "pts_ns": 41200000000, "width": 1920, "height": 1080,
           "objects": [
               {"id": 7, "class_id": 0, "label": "person", "confidence": 0.8712,
                "bbox": {"left": 960.0, "top": 540.0, "width": 480.0, "height": 540.0},
                "tracker_confidence": 0.91},
               {"id": None, "class_id": 28, "label": "suitcase", "confidence": 0.4,
                "bbox": {"left": 0.0, "top": 0.0, "width": 96.0, "height": 54.0}},
           ]}
    dets, size, key = parse_message(json.dumps(msg))
    assert size == (1920, 1080) and key is None
    assert [(d.track_id, d.cls_name) for d in dets] == [(7, "person"), (-1, "suitcase")]
    d = detector(frame_size=(768, 576))
    d.source_size = (768, 576)              # the message's own size must win over both
    d.on_payload(json.dumps(msg))
    person = d.infer(np.zeros((480, 640, 3), np.uint8))[0]
    assert person.bbox == pytest.approx((320.0, 240.0, 480.0, 480.0))


def test_pipegen_generated_payload():
    """The exact shape our PipeGen pipeline publishes (from its PipelineIR mqtt node)."""
    msg = {"frame_index": 412, "timestamp_s": 41.2, "objects": [
        {"tracker_id": 7, "class": "person", "confidence": 0.87, "bounding_box": [100.0, 50.0, 160.0, 230.0]},
        {"tracker_id": 9, "class": "suitcase", "confidence": 0.61, "bounding_box": [300.0, 400.0, 350.0, 470.0]},
    ]}
    dets, size, key = parse_message(json.dumps(msg))
    assert size is None and key is None
    assert [(d.track_id, d.cls_name, d.conf, d.bbox) for d in dets] == [
        (7, "person", 0.87, (100.0, 50.0, 160.0, 230.0)),
        (9, "suitcase", 0.61, (300.0, 400.0, 350.0, 470.0)),
    ]


def test_source_size_from_the_camera_sets_the_scale():
    d = detector(frame_size=(1920, 1080))          # configured fallback is wrong for this clip...
    d.source_size = (768, 576)                    # ...the camera reports the real stream size
    d.on_payload(json.dumps({"objects": [{"tracker_id": 1, "class": "person", "confidence": 0.9,
                                          "bounding_box": [384, 288, 768, 576]}]}))
    (det,) = d.infer(np.zeros((480, 640, 3), np.uint8))   # Argonyx analyses at 640 wide
    assert det.bbox == pytest.approx((320.0, 240.0, 640.0, 480.0))


def test_minimal_schema_strings():
    msg = {"version": "4.0", "id": 120, "@timestamp": "2026-10-17T13:00:00.000Z",
           "objects": ["7|100|200|300|600|Person", "18446744073709551615|50|60|90|120|Bag|#|x"]}
    dets, size, key = parse_message(json.dumps(msg))
    assert key is None and size is None
    assert [(d.track_id, d.cls_name, d.bbox) for d in dets] == [
        (7, "person", (100.0, 200.0, 300.0, 600.0)),
        (-1, "backpack", (50.0, 60.0, 90.0, 120.0)),   # untracked id, mapped class
    ]


def test_minimal_schema_confidence_field():
    dets, _, _ = parse_message('{"objects": ["3|0|0|10|20|person|0.82"]}')
    assert dets[0].conf == pytest.approx(0.82)


def test_full_schema_one_object_per_message():
    msg = {"messageid": "a1", "@timestamp": "2026-10-17T13:00:00.040Z",
           "object": {"id": "42", "confidence": 0.9, "person": {"age": 0},
                      "bbox": {"topleftx": 10, "toplefty": 20, "bottomrightx": 110, "bottomrighty": 320}}}
    dets, _, key = parse_message(json.dumps(msg).encode())
    assert key == "2026-10-17T13:00:00.040Z"
    assert (dets[0].track_id, dets[0].cls_name, dets[0].conf) == (42, "person", 0.9)
    assert dets[0].bbox == (10.0, 20.0, 110.0, 320.0)


def test_generic_object_list_with_frame_size():
    msg = {"frame_width": 1280, "frame_height": 720,
           "objects": [{"track_id": 5, "label": "person", "confidence": 0.7,
                        "bbox": {"left": 10, "top": 20, "width": 30, "height": 90}}]}
    dets, size, _ = parse_message(msg)
    assert size == (1280, 720)
    assert dets[0].bbox == (10.0, 20.0, 40.0, 110.0)


def test_degenerate_boxes_are_dropped():
    dets, _, _ = parse_message({"objects": ["1|50|50|50|90|person", "2|0|0|10|10|person"]})
    assert [d.track_id for d in dets] == [2]


def test_infer_rescales_to_the_analysed_frame():
    d = detector()
    d.on_payload('{"objects": ["7|960|540|1920|1080|person"]}')
    frame = np.zeros((360, 640, 3), np.uint8)   # Argonyx analyses at 640 wide
    (det,) = d.infer(frame)
    assert det.bbox == pytest.approx((320.0, 180.0, 640.0, 360.0))


def test_stale_feed_reports_nothing():
    d = detector(max_age_s=0.05)
    d.on_payload('{"objects": ["7|0|0|100|100|person"]}')
    time.sleep(0.1)
    assert d.infer(np.zeros((1080, 1920, 3), np.uint8)) == []


def test_full_schema_messages_group_into_one_frame():
    d = detector()
    for tid in (1, 2):
        d.on_payload(json.dumps({"@timestamp": "t1", "object": {
            "id": str(tid), "person": {}, "bbox": {"topleftx": 0, "toplefty": 0, "bottomrightx": 10, "bottomrighty": 30}}}))
    assert sorted(x.track_id for x in d.infer(np.zeros((1080, 1920, 3), np.uint8))) == [1, 2]
    d.on_payload(json.dumps({"@timestamp": "t2", "object": {
        "id": "3", "person": {}, "bbox": {"topleftx": 0, "toplefty": 0, "bottomrightx": 10, "bottomrighty": 30}}}))
    assert [x.track_id for x in d.infer(np.zeros((1080, 1920, 3), np.uint8))] == [3]


def test_bad_payload_is_counted_not_raised():
    d = detector()
    d.on_payload(b"not json")
    assert d.bad_messages == 1 and d.infer(np.zeros((10, 10, 3), np.uint8)) == []
