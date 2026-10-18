"""Privacy layer: heads are covered before a frame is shown or stored.

Rules carried over from PrivAgent (our on-device redaction project):
- Redaction happens on the edge device, before a frame reaches the dashboard
  stream or an evidence file. The analysis itself still sees the raw frame.
- A solid box, never blur or pixelation: those can be partly reversed, and a
  box cannot.
- When we can't tell where the head is (a person lying down), the whole body
  is covered: hiding too much is a cosmetic cost, hiding too little is a leak.
"""
from typing import Iterable, Optional, Tuple

import numpy as np

from backend import config

Box = Tuple[float, float, float, float]
MASK_COLOR = (40, 40, 40)   # BGR, dark grey: reads as "redacted", not as a shadow


def head_region(bbox: Box, frame_w: int, frame_h: int) -> Optional[Tuple[int, int, int, int]]:
    """Pixel rectangle to cover for one person box, clamped to the frame;
    None when nothing of it is inside the frame."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    if h >= w:
        # Upright: the head is in the top band of the box. Full box width, since
        # a turned or tilted head can sit anywhere across it.
        y2 = y1 + h * config.PRIVACY_HEAD_FRACTION
    # Wider than tall (lying down, or a crouch seen side-on): the head can be at
    # either end, so the whole box is covered.
    left, top = max(0, int(x1)), max(0, int(y1))
    right, bottom = min(frame_w, int(round(x2))), min(frame_h, int(round(y2)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def mask_heads(frame: np.ndarray, boxes: Iterable[Box], faces: Iterable[Box] = ()) -> int:
    """Cover each person's head region (and any detected face box) in place.
    Returns how many regions were covered."""
    frame_h, frame_w = frame.shape[:2]
    covered = 0
    for bbox in boxes:
        region = head_region(bbox, frame_w, frame_h)
        if region is not None:
            left, top, right, bottom = region
            frame[top:bottom, left:right] = MASK_COLOR
            covered += 1
    for x1, y1, x2, y2 in faces:
        left, top = max(0, int(x1)), max(0, int(y1))
        right, bottom = min(frame_w, int(round(x2))), min(frame_h, int(round(y2)))
        if right > left and bottom > top:
            frame[top:bottom, left:right] = MASK_COLOR
            covered += 1
    return covered
