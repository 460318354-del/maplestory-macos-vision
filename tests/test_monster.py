import numpy as np

from maplevision.vision.monster import (
    MonsterTemplate,
    MonsterTracker,
    _match_image,
    detect_monsters_with_templates,
)


def test_detect_monsters_with_template_finds_two_same_shape_targets():
    template_image = np.zeros((20, 20, 3), dtype=np.uint8)
    template_image[4:16, 8:12] = [255, 255, 255]
    template_image[8:12, 4:16] = [255, 255, 255]
    template = MonsterTemplate(
        name="stump_shape",
        image_bgr=template_image,
        match_image=_match_image(template_image),
    )

    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[30:50, 20:40] = template_image
    frame[30:50, 60:80] = template_image

    detections = detect_monsters_with_templates(
        frame,
        region=(0, 0, 120, 80),
        templates=(template,),
        config={
            "threshold": 0.9,
            "overlap_threshold": 0.3,
            "max_results": 5,
        },
    )

    assert len(detections) == 2
    assert detections[0].box == (20, 30, 20, 20)
    assert detections[1].box == (60, 30, 20, 20)


def test_tracker_requires_two_frames_before_confirmation():
    tracker = MonsterTracker(confirm_frames=2, max_distance=50)
    first = _detection("stump", 20, 30)
    second = _detection("stump", 28, 30)

    assert tracker.update((first,)) == ()
    confirmed = tracker.update((second,))

    assert len(confirmed) == 1
    assert confirmed[0].box == second.box


def _detection(label, x, y):
    from maplevision.vision.monster import MonsterDetection

    return MonsterDetection(label=label, score=0.9, box=(x, y, 20, 20))
