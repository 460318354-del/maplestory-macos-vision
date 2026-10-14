import numpy as np
from types import SimpleNamespace

from maplevision.gui_common import (
    clip_region_to_frame,
    crop_region,
    minimap_frame_point,
    minimap_local_point,
    player_search_region,
    runtime_event_messages,
    safe_template_category,
)


def test_gui_common_crop_region_clips_to_frame():
    frame = np.zeros((10, 12, 3), dtype=np.uint8)
    frame[2:7, 4:9] = [1, 2, 3]

    cropped = crop_region(frame, (2, 1, 8, 8))

    assert cropped.shape == (8, 8, 3)
    assert tuple(cropped[1, 2]) == (1, 2, 3)


def test_clip_region_reports_actual_visible_region():
    frame = np.zeros((10, 12, 3), dtype=np.uint8)

    assert clip_region_to_frame(frame, (-2, 8, 20, 8)) == (
        0,
        8,
        12,
        2,
    )


def test_gui_common_preserves_minimap_local_coordinates():
    region = (5, 72, 157, 85)
    point = minimap_local_point((20, 84), region)

    assert point == (15, 12)
    assert minimap_frame_point(point, region) == (20, 84)


def test_gui_common_player_search_region_and_safe_category():
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    assert player_search_region(frame, None) == [0, 0, 800, 600]
    assert safe_template_category("new/boss") == "new_boss"


def test_runtime_event_messages_only_reports_state_changes():
    state = {}
    controller = SimpleNamespace(
        route_path_name="寻路1",
        last_coin_detections=(),
        config={"combat": {"attack_action": "tap_attack(ctrl)"}},
    )
    no_target = SimpleNamespace(
        confirmed_targets=(),
        targets=(),
        action="move_right(right)",
        reason="route-move_right",
        player=None,
    )

    assert runtime_event_messages(no_target, controller, state) == [
        "寻路 寻路1 启动"
    ]
    assert runtime_event_messages(no_target, controller, state) == []

    target = object()
    controller.last_coin_detections = (target,)
    found = SimpleNamespace(
        confirmed_targets=(target,),
        targets=(target,),
        action="jump_right(right,alt)",
        reason="route-move_right;obstacle-no-progress:0px",
        player=object(),
    )
    messages = runtime_event_messages(found, controller, state)
    assert messages == ["发现怪物", "发现金币", "有障碍，启动跳跃"]
    assert runtime_event_messages(found, controller, state) == []

    controller.route_path_name = ""
    controller.last_coin_detections = ()
    gone = SimpleNamespace(
        confirmed_targets=(),
        targets=(),
        action="observe",
        reason="route-arrived",
        player=None,
    )
    assert runtime_event_messages(gone, controller, state) == [
        "寻路完成",
        "怪物消失",
        "金币区域已清空",
    ]


def test_runtime_event_messages_reports_fixed_point_return_transitions():
    state = {}
    controller = SimpleNamespace(
        route_path_name="",
        last_coin_detections=(),
        config={"combat": {"attack_action": "tap_attack(ctrl)"}},
    )
    returning = SimpleNamespace(
        confirmed_targets=(),
        targets=(),
        action="move_left(left)",
        reason="fixed-point-return:route-combat-approach",
    )
    holding = SimpleNamespace(
        confirmed_targets=(),
        targets=(),
        action="observe",
        reason="fixed-point-hold",
    )

    assert runtime_event_messages(returning, controller, state) == [
        "偏离固定点，开始回标点"
    ]
    assert runtime_event_messages(returning, controller, state) == []
    assert runtime_event_messages(holding, controller, state) == [
        "已回到固定点"
    ]
