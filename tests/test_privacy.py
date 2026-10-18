import numpy as np

from backend import config
from backend.privacy import MASK_COLOR, head_region, mask_heads


def test_upright_person_covers_top_band_only():
    region = head_region((100, 100, 200, 400), 640, 480)
    assert region == (100, 100, 200, int(100 + 300 * config.PRIVACY_HEAD_FRACTION))


def test_lying_person_is_covered_whole():
    assert head_region((100, 300, 400, 380), 640, 480) == (100, 300, 400, 380)


def test_region_is_clamped_to_the_frame():
    assert head_region((-50, -20, 60, 300), 640, 480) == (0, 0, 60, int(-20 + 320 * config.PRIVACY_HEAD_FRACTION))
    assert head_region((700, 10, 800, 200), 640, 480) is None


def test_mask_heads_paints_a_solid_box_in_place():
    frame = np.full((480, 640, 3), 255, np.uint8)
    n = mask_heads(frame, [(100, 100, 200, 400)], faces=[(300, 50, 340, 90)])
    assert n == 2
    assert (frame[110, 150] == MASK_COLOR).all()        # inside the head band
    assert (frame[350, 150] == 255).all()               # body below it untouched
    assert (frame[70, 320] == MASK_COLOR).all()         # face box
