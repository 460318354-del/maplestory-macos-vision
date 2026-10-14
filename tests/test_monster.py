import numpy as np
import cv2

from maplevision.vision.monster import (
    MonsterTemplate,
    MonsterDetector,
    MonsterTracker,
    _match_image,
    detect_monsters_with_templates,
    load_template_groups,
    render_monster_debug_frame,
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


def test_tracker_stops_returning_confirmed_target_when_current_frame_loses_it():
    tracker = MonsterTracker(confirm_frames=2, max_distance=50, max_misses=2)
    first = _detection("stump", 20, 30)
    second = _detection("stump", 24, 30)

    assert tracker.update((first,)) == ()
    assert tracker.update((second,))
    assert tracker.update(()) == ()


def test_template_group_assigns_category_label(tmp_path):
    template_dir = tmp_path / "yellow_mushroom"
    template_dir.mkdir()
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[3:9, 3:9] = [20, 180, 240]
    template_path = template_dir / "template_01.png"
    assert cv2.imwrite(str(template_path), image)

    templates = load_template_groups({"yellow_mushroom": [template_dir]})

    assert len(templates) == 1
    assert templates[0].label == "yellow_mushroom"


def test_monster_detector_prepares_templates_once_without_scaling(tmp_path):
    template_dir = tmp_path / "yellow_mushroom"
    template_dir.mkdir()
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[3:9, 3:9] = [20, 180, 240]
    template_path = template_dir / "template_01.png"
    assert cv2.imwrite(str(template_path), image)

    detector = MonsterDetector(
        {
            "enabled": True,
            "template_groups": {"yellow_mushroom": [template_dir]},
        }
    )

    assert len(detector.templates) == 1
    assert len(detector.prepared_templates) == 1
    first_images = tuple(item.match_image for item in detector.prepared_templates)
    detector.detect(
        np.zeros((80, 80, 3), dtype=np.uint8),
        (0, 0, 80, 80),
    )
    assert all(
        before is after
        for before, after in zip(
            first_images,
            (item.match_image for item in detector.prepared_templates),
        )
    )
    assert detector.last_stats.match_calls == 1


def test_monster_detector_matches_only_enabled_categories(tmp_path):
    groups = {}
    for label, color in (
        ("yellow_mushroom", [20, 180, 240]),
        ("red_snail", [20, 20, 220]),
    ):
        template_dir = tmp_path / label
        template_dir.mkdir()
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        image[3:9, 3:9] = color
        assert cv2.imwrite(str(template_dir / "template_01.png"), image)
        groups[label] = [template_dir]

    detector = MonsterDetector(
        {
            "enabled": True,
            "template_groups": groups,
            "allowed_labels": ["red_snail"],
        }
    )
    detector.detect(np.zeros((80, 80, 3), dtype=np.uint8), (0, 0, 80, 80))

    assert len(detector.prepared_templates) == 2
    assert detector.last_stats.template_count == 1
    assert detector.last_stats.match_calls == 1


def test_render_monster_debug_frame_marks_category_and_box():
    from maplevision.vision.monster import MonsterDetection

    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    detection = MonsterDetection(
        label="red_snail",
        score=0.91,
        box=(20, 15, 18, 16),
    )

    output = render_monster_debug_frame(frame, (detection,))

    assert output.shape == frame.shape
    assert np.any(output[15:32, 20:39] != 0)


def _detection(label, x, y):
    from maplevision.vision.monster import MonsterDetection

    return MonsterDetection(label=label, score=0.9, box=(x, y, 20, 20))
