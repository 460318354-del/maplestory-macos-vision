from __future__ import annotations

import logging
from pathlib import Path
import re
from datetime import datetime
from typing import Any

import AppKit
import cv2
import objc
import Quartz
import numpy as np
import yaml

from .config.loader import load_config
from .gui_common import (
    clip_region_to_frame,
    crop_region,
    ensure_editor_data_layout,
    minimap_frame_point,
    minimap_local_point,
    player_search_region,
    runtime_event_messages,
    safe_template_category,
)
from .platform.macos.capture import MacOSWindowSource
from .platform.macos.input import MacOSInputController
from .decision.controller import VisionController
from .gui_canvas import FrameSelectionCanvas
from .pathing import (
    ROUTE_STEP_TYPES,
    active_route_group,
    minimap_local_rect,
    minimap_scaled_rect_to_local,
    path_name,
    route_group_names,
    route_ladder_group,
    route_path_group,
    route_point_group,
    step_label,
)
from .route_test import RouteTestController
from .vision.height_band import screen_region_from_band
from .vision.minimap import observe_minimap
from .vision.monster import detect_monsters, render_monster_debug_frame
from .vision.player_screen import detect_player_screen, resolve_player_config

logger = logging.getLogger(__name__)

PLAYER_ANCHOR_BELOW_NAMEPLATE = 24
PLAYER_TEMPLATE_CATEGORY = "角色"
NON_MONSTER_TEMPLATE_CATEGORIES = {
    PLAYER_TEMPLATE_CATEGORY,
    "血条",
    "金币",
}
DEFAULT_TEMPLATE_DISPLAY_NAMES = {
    "yellow_mushroom": "大型黄蘑菇",
    "red_snail": "红色蜗牛",
    "small_mushroom": "小型蘑菇",
    PLAYER_TEMPLATE_CATEGORY: PLAYER_TEMPLATE_CATEGORY,
}

def _nsimage_from_bgr(image_bgr: np.ndarray):
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        raise RuntimeError("无法编码游戏画面")
    data = AppKit.NSData.dataWithBytes_length_(
        encoded.tobytes(),
        len(encoded),
    )
    source = Quartz.CGImageSourceCreateWithData(data, None)
    if source is None:
        raise RuntimeError("Quartz 无法读取游戏画面")
    cg_image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
    if cg_image is None:
        raise RuntimeError("Quartz 无法解码游戏画面")
    image = AppKit.NSImage.alloc().initWithCGImage_size_(
        cg_image,
        AppKit.NSZeroSize,
    )
    if image is None:
        raise RuntimeError("AppKit 无法解码游戏画面")
    return image


class MapleVisionActions(AppKit.NSObject):
    def initWithController_(self, controller):
        self = objc.super(MapleVisionActions, self).init()
        if self is None:
            return None
        self.controller = controller
        return self

    def bindWindow_(self, sender):
        self.controller.bind_window()

    def captureGameFrame_(self, sender):
        self.controller.capture_game_frame()

    def detectPlayer_(self, sender):
        self.controller.detect_player()

    def markScreenBand_(self, sender):
        self.controller.toggle_screen_band_marking()

    def markPatrolPoint_(self, sender):
        self.controller.toggle_patrol_point_marking()

    def markFixedPoint_(self, sender):
        self.controller.toggle_fixed_point_marking()

    def saveConfig_(self, sender):
        self.controller.save_config()

    def selectMonsterRegion_(self, sender):
        self.controller.toggle_monster_region_selection()

    def toggleAttackMonster_(self, sender):
        self.controller.toggle_attack_monster(sender)

    def showAttackMonsterMenu_(self, sender):
        self.controller.show_attack_monster_menu(sender)

    def togglePatrolMode_(self, sender):
        self.controller.toggle_patrol_mode(sender)

    def runModeChanged_(self, sender):
        self.controller.run_mode_changed()

    def routeCombatPointChanged_(self, sender):
        self.controller.route_combat_point_changed()

    def routeGroupChanged_(self, sender):
        self.controller.route_group_changed()

    def selectPlayer_(self, sender):
        self.controller.toggle_player_selection()

    def captureTrainingFrame_(self, sender):
        self.controller.capture_training_frame()

    def selectMonster_(self, sender):
        self.controller.toggle_monster_selection()

    def testMonsters_(self, sender):
        self.controller.test_monsters()

    def undoMonsterTemplate_(self, sender):
        self.controller.undo_monster_template()

    def deleteSelectedMonsterTemplate_(self, sender):
        self.controller.delete_selected_monster_template()

    def addTemplateCategory_(self, sender):
        self.controller.add_template_category()

    def openTestCategories_(self, sender):
        self.controller.open_test_categories(sender)

    def toggleTestCategory_(self, sender):
        self.controller.toggle_test_category(sender)

    def selectAllTestCategories_(self, sender):
        self.controller.select_all_test_categories()

    def clearTestCategories_(self, sender):
        self.controller.clear_test_categories()

    def startRuntime_(self, sender):
        self.controller.start_runtime()

    def stopRuntime_(self, sender):
        self.controller.stop_runtime()

    def runtimeTick_(self, timer):
        self.controller.runtime_tick()

    def startRouteTest_(self, sender):
        self.controller.start_route_test()

    def stopRouteTest_(self, sender):
        self.controller.stop_route_test()

    def routeTestTick_(self, timer):
        self.controller.route_test_tick()

    def captureRouteFrame_(self, sender):
        self.controller.capture_route_frame()

    def toggleRouteAnnotations_(self, sender):
        self.controller.toggle_route_annotations()

    def markRoutePoint_(self, sender):
        self.controller.toggle_route_point_marking()

    def calibrateRouteMinimap_(self, sender):
        self.controller.calibrate_route_minimap()

    def recordLadder_(self, sender):
        self.controller.record_ladder()

    def addRouteStep_(self, sender):
        self.controller.add_route_step()

    def removeRouteStep_(self, sender):
        self.controller.remove_route_step()

    def newRoutePath_(self, sender):
        self.controller.new_route_path()

    def saveRoutePath_(self, sender):
        self.controller.save_route_path()

    def deleteRoutePath_(self, sender):
        self.controller.delete_route_path()

    def deleteRoutePoint_(self, sender):
        self.controller.delete_route_point()

    def routePathChanged_(self, sender):
        self.controller.route_path_changed()

    def routeStepTypeChanged_(self, sender):
        self.controller.route_step_type_changed()
class MapleVisionApp:
    def __init__(
        self,
        config: dict[str, Any],
        config_path: Path | None = None,
    ):
        self.config = config
        self.config_path = config_path or Path("config/default.yaml")
        ensure_editor_data_layout(self.config, self._project_root())
        self.source: MacOSWindowSource | None = None
        self.current_frame = None
        self.current_screen_player = None
        self.player_template_path: Path | None = None
        self.screen_region: tuple[int, int, int, int] | None = None
        self.patrol_target_point: tuple[int, int] | None = None
        self.fixed_point_target: tuple[int, int] | None = None
        self.patrol_point_marking = False
        self.point_marking_kind: str | None = None
        self.run_rectangle_mode: str | None = None
        route_config = config.get("route_combat", {})
        self.route_points = [
            dict(point)
            for point in (route_config.get("points") or [])
            if isinstance(point, dict)
        ]
        self.route_combat_profiles = {
            str(name): dict(profile)
            for name, profile in (route_config.get("point_profiles") or {}).items()
            if isinstance(profile, dict)
        }
        self.route_combat_point_name = (
            str(self.route_points[0].get("name", "")).strip()
            if self.route_points
            else None
        )
        self.route_ladders = [
            dict(ladder)
            for ladder in (route_config.get("ladders") or [])
            if isinstance(ladder, dict)
        ]
        self.route_paths = [
            dict(route)
            for route in (route_config.get("paths") or [])
            if isinstance(route, dict)
        ]
        self.route_active_group = active_route_group(route_config)
        self.route_frame = None
        self.route_minimap_frame = None
        self.route_minimap_scale = 3
        self.route_show_annotations = False
        self.route_rectangle_mode = False
        self.route_current_steps: list[dict[str, Any]] = []
        self.route_current_path_key: str | None = None
        self.training_frame = None
        self.training_detections = ()
        self.selected_training_detection = None
        self.training_test_streak = 0
        self.training_test_count = 0
        self.last_monster_template_path: Path | None = None
        self.template_categories = self._load_template_categories()
        self.test_template_labels = {
            label for _, label in self.template_categories
        }
        self.runtime_controller: VisionController | None = None
        self.runtime_timer = None
        self.runtime_running = False
        self.runtime_log_lines: list[str] = []
        self.runtime_log_state: dict[str, Any] = {}
        self.route_test_controller: RouteTestController | None = None
        self.route_test_input: MacOSInputController | None = None
        self.route_test_timer = None
        self.route_test_running = False
        self.app = AppKit.NSApplication.sharedApplication()
        self.app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)

        rect = AppKit.NSMakeRect(0, 0, 1020, 780)
        style = (
            AppKit.NSWindowStyleMaskTitled
            | AppKit.NSWindowStyleMaskClosable
            | AppKit.NSWindowStyleMaskMiniaturizable
            | AppKit.NSWindowStyleMaskResizable
        )
        self.window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect,
            style,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        self.window.setTitle_("MapleVision")
        self.window.setMinSize_(AppKit.NSMakeSize(860, 640))
        self.window.center()
        self.window.setReleasedWhenClosed_(False)

        root = self.window.contentView()
        self.tab_view = AppKit.NSTabView.alloc().initWithFrame_(
            AppKit.NSMakeRect(10, 10, 1000, 750)
        )
        run_item = AppKit.NSTabViewItem.alloc().initWithIdentifier_("run")
        run_item.setLabel_("运行识别")
        run_view = AppKit.NSView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, 1000, 720)
        )
        run_item.setView_(run_view)
        training_item = AppKit.NSTabViewItem.alloc().initWithIdentifier_("training")
        training_item.setLabel_("怪物模板训练")
        training_view = AppKit.NSView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, 1000, 720)
        )
        training_item.setView_(training_view)
        self.tab_view.addTabViewItem_(run_item)
        self.tab_view.addTabViewItem_(training_item)
        route_item = AppKit.NSTabViewItem.alloc().initWithIdentifier_("route")
        route_item.setLabel_("路径")
        route_view = AppKit.NSView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, 1000, 720)
        )
        route_item.setView_(route_view)
        self.tab_view.addTabViewItem_(route_item)
        root.addSubview_(self.tab_view)
        content = run_view
        self.title_label = self._label("窗口标题包含", AppKit.NSMakeRect(20, 675, 100, 24))
        content.addSubview_(self.title_label)

        self.title_field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(124, 673, 300, 28)
        )
        self.title_field.setStringValue_(
            str(config["game_window"].get("title_contains", "MapleStory"))
        )
        content.addSubview_(self.title_field)

        self.actions = MapleVisionActions.alloc().initWithController_(self)
        self.bind_button = self._button(
            "绑定窗口",
            AppKit.NSMakeRect(432, 671, 100, 32),
            self.actions,
            "bindWindow:",
        )
        content.addSubview_(self.bind_button)
        self.save_button = self._button(
            "保存配置",
            AppKit.NSMakeRect(540, 671, 100, 32),
            self.actions,
            "saveConfig:",
        )
        content.addSubview_(self.save_button)
        self.run_mode_label = self._label(
            "运行模式：路径打怪",
            AppKit.NSMakeRect(650, 675, 125, 24),
        )
        content.addSubview_(self.run_mode_label)

        self.player_id_label = self._label(
            "角色 ID",
            AppKit.NSMakeRect(20, 635, 100, 24),
        )
        content.addSubview_(self.player_id_label)
        self.player_id_field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(124, 633, 300, 28)
        )
        player_config = config.get("combat", {}).get("player", {})
        self.player_id_field.setStringValue_(
            str(player_config.get("id", player_config.get("name", "")))
        )
        self.player_id_field.setPlaceholderString_("例如 Conquerork")
        content.addSubview_(self.player_id_field)
        self.detect_button = self._button(
            "识别角色",
            AppKit.NSMakeRect(868, 631, 90, 32),
            self.actions,
            "detectPlayer:",
        )
        content.addSubview_(self.detect_button)
        self.game_frame_button = self._button(
            "获取游戏画面",
            AppKit.NSMakeRect(432, 631, 120, 32),
            self.actions,
            "captureGameFrame:",
        )
        self.monster_region_button = self._button(
            "框选寻怪区域",
            AppKit.NSMakeRect(562, 631, 120, 32),
            self.actions,
            "selectMonsterRegion:",
        )
        self.monster_region_button.setEnabled_(False)
        content.addSubview_(self.game_frame_button)
        content.addSubview_(self.monster_region_button)

        self.heal_threshold_label = self._label(
            "回血 %",
            AppKit.NSMakeRect(20, 590, 45, 24),
        )
        content.addSubview_(self.heal_threshold_label)
        self.heal_threshold_field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(65, 588, 42, 28)
        )
        self.heal_threshold_field.setStringValue_(
            self._percent_string(
                config.get("combat", {}).get("health", {}).get("low_threshold", 0.40)
            )
        )
        content.addSubview_(self.heal_threshold_field)
        self.heal_key_label = self._label("按键1", AppKit.NSMakeRect(110, 590, 42, 24))
        content.addSubview_(self.heal_key_label)

        self.mana_threshold_label = self._label(
            "回蓝 %",
            AppKit.NSMakeRect(155, 590, 45, 24),
        )
        content.addSubview_(self.mana_threshold_label)
        self.mana_threshold_field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(200, 588, 42, 28)
        )
        self.mana_threshold_field.setStringValue_(
            self._percent_string(
                config.get("combat", {}).get("mana", {}).get("low_threshold", 0.30)
            )
        )
        content.addSubview_(self.mana_threshold_field)
        self.mana_key_label = self._label("按键2", AppKit.NSMakeRect(245, 590, 42, 24))
        content.addSubview_(self.mana_key_label)

        self.attack_distance_label = self._label(
            "攻击距离",
            AppKit.NSMakeRect(295, 590, 65, 24),
        )
        content.addSubview_(self.attack_distance_label)
        self.attack_distance_field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(360, 588, 50, 28)
        )
        attack_range = config.get("combat", {}).get("attack_range", {})
        self.attack_distance_field.setStringValue_(
            str(attack_range.get("distance", attack_range.get("x", 50)))
        )
        self.attack_distance_field.setPlaceholderString_("像素")
        content.addSubview_(self.attack_distance_field)
        self.attack_monster_label = self._label(
            "攻击怪物",
            AppKit.NSMakeRect(420, 590, 70, 24),
        )
        content.addSubview_(self.attack_monster_label)
        self._build_attack_monster_menu(
            content,
            config.get("combat", {}).get("monsters", {}).get("allowed_labels"),
        )
        pickup_config = config.get("combat", {}).get("pickup", {})
        self.pickup_interval_label = self._label(
            "捡金币每",
            AppKit.NSMakeRect(20, 460, 62, 24),
        )
        content.addSubview_(self.pickup_interval_label)
        self.pickup_interval_field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(84, 458, 54, 28)
        )
        self.pickup_interval_field.setStringValue_(
            str(pickup_config.get("interval_seconds", 30))
        )
        self.pickup_interval_field.setPlaceholderString_("秒")
        content.addSubview_(self.pickup_interval_field)
        self.pickup_interval_suffix_label = self._label(
            "秒启动",
            AppKit.NSMakeRect(142, 460, 58, 24),
        )
        content.addSubview_(self.pickup_interval_suffix_label)
        fixed_point_config = config.get("combat", {}).get("fixed_point", {})
        fixed_target = fixed_point_config.get("point")
        if isinstance(fixed_target, (list, tuple)) and len(fixed_target) == 2:
            self.fixed_point_target = (
                int(fixed_target[0]),
                int(fixed_target[1]),
            )
        self.fixed_point_button = self._button(
            "记录当前点位",
            AppKit.NSMakeRect(545, 492, 100, 32),
            self.actions,
            "markFixedPoint:",
        )
        self.fixed_point_button.setEnabled_(False)
        content.addSubview_(self.fixed_point_button)
        self.route_combat_point_label = self._label(
            "路径挂机点",
            AppKit.NSMakeRect(668, 424, 72, 24),
        )
        content.addSubview_(self.route_combat_point_label)
        self.route_combat_point_popup = self._popup(
            AppKit.NSMakeRect(744, 420, 160, 28),
            self.route_point_names(),
        )
        self.route_combat_point_popup.setTarget_(self.actions)
        self.route_combat_point_popup.setAction_("routeCombatPointChanged:")
        if self.route_combat_point_name:
            self.route_combat_point_popup.selectItemWithTitle_(
                self.route_combat_point_name
            )
        content.addSubview_(self.route_combat_point_popup)
        self.fixed_point_checkbox = self._checkbox(
            "固定点战斗",
            AppKit.NSMakeRect(420, 494, 120, 28),
        )
        self.fixed_point_checkbox.setState_(
            1 if fixed_point_config.get("enabled", False) else 0
        )
        content.addSubview_(self.fixed_point_checkbox)
        self.fixed_point_interval_label = self._label(
            "无怪回定点距离px",
            AppKit.NSMakeRect(650, 496, 95, 24),
        )
        content.addSubview_(self.fixed_point_interval_label)
        self.fixed_point_return_distance_field = (
            AppKit.NSTextField.alloc().initWithFrame_(
                AppKit.NSMakeRect(750, 492, 58, 28)
            )
        )
        self.fixed_point_return_distance_field.setStringValue_(
            str(fixed_point_config.get("return_distance_px", 10))
        )
        self.fixed_point_return_distance_field.setPlaceholderString_("10")
        content.addSubview_(self.fixed_point_return_distance_field)
        self.minimap_region_label = self._label(
            "固定区域",
            AppKit.NSMakeRect(815, 496, 160, 24),
        )
        content.addSubview_(self.minimap_region_label)
        minimap_region = tuple(config.get("minimap", {}).get("region") or ())
        if len(minimap_region) == 4:
            self.minimap_region_label.setStringValue_(
                f"{minimap_region[0]},{minimap_region[1]},"
                f"{minimap_region[2]}x{minimap_region[3]}"
            )
        periodic_config = config.get("combat", {}).get("periodic_key", {})
        self.periodic_key_label = self._label(
            "每",
            AppKit.NSMakeRect(662, 526, 20, 24),
        )
        content.addSubview_(self.periodic_key_label)
        self.periodic_key_interval_field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(682, 524, 52, 28)
        )
        self.periodic_key_interval_field.setStringValue_(
            str(periodic_config.get("interval_seconds", 0) or "")
        )
        self.periodic_key_interval_field.setPlaceholderString_("秒")
        content.addSubview_(self.periodic_key_interval_field)
        self.periodic_key_suffix_label = self._label(
            "秒按键3一次",
            AppKit.NSMakeRect(738, 526, 105, 24),
        )
        content.addSubview_(self.periodic_key_suffix_label)
        self.start_button = self._button(
            "开始",
            AppKit.NSMakeRect(845, 588, 65, 32),
            self.actions,
            "startRuntime:",
        )
        content.addSubview_(self.start_button)
        self.stop_button = self._button(
            "停止",
            AppKit.NSMakeRect(915, 588, 65, 32),
            self.actions,
            "stopRuntime:",
        )
        self.stop_button.setEnabled_(False)
        content.addSubview_(self.stop_button)

        self.window_label = self._label(
            "窗口：未绑定",
            AppKit.NSMakeRect(20, 552, 390, 24),
        )
        content.addSubview_(self.window_label)
        self.image_info_label = self._label(
            "画面：尚未获取",
            AppKit.NSMakeRect(20, 526, 390, 24),
        )
        content.addSubview_(self.image_info_label)
        self.player_info_label = self._label(
            "角色：尚未识别",
            AppKit.NSMakeRect(20, 500, 390, 24),
        )
        content.addSubview_(self.player_info_label)
        self.height_info_label = self._label(
            "识别区域：尚未标注",
            AppKit.NSMakeRect(20, 474, 390, 24),
        )
        content.addSubview_(self.height_info_label)

        self.preview = FrameSelectionCanvas.alloc().initWithFrame_(
            AppKit.NSMakeRect(20, 54, 533, 400)
        )
        self.preview.on_selection = self._height_selected
        self.preview.on_rectangle = self._run_rectangle_selected
        self.preview.on_click = self._patrol_point_clicked
        content.addSubview_(self.preview)

        self.runtime_log_label = self._label(
            "识别日志",
            AppKit.NSMakeRect(570, 430, 410, 24),
        )
        content.addSubview_(self.runtime_log_label)
        self.runtime_log_scroll = AppKit.NSScrollView.alloc().initWithFrame_(
            AppKit.NSMakeRect(570, 54, 410, 370)
        )
        self.runtime_log_scroll.setHasVerticalScroller_(True)
        self.runtime_log_scroll.setHasHorizontalScroller_(False)
        self.runtime_log_scroll.setBorderType_(AppKit.NSBezelBorder)
        self.runtime_log_view = AppKit.NSTextView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, 390, 350)
        )
        self.runtime_log_view.setEditable_(False)
        self.runtime_log_view.setSelectable_(True)
        self.runtime_log_view.setFont_(AppKit.NSFont.monospacedSystemFontOfSize_weight_(11, 0.0))
        self.runtime_log_view.setString_("尚未启动识别\n")
        self.runtime_log_scroll.setDocumentView_(self.runtime_log_view)
        content.addSubview_(self.runtime_log_scroll)

        self.status_label = self._label(
            "未绑定窗口",
            AppKit.NSMakeRect(20, 20, 960, 22),
        )
        self.status_label.setTextColor_(AppKit.NSColor.secondaryLabelColor())
        content.addSubview_(self.status_label)

        self._build_training_page(training_view)
        self._build_route_page(route_view)
        self.run_mode_changed()

    def _build_training_page(self, content) -> None:
        title = self._label(
            "怪物模板训练与测试",
            AppKit.NSMakeRect(20, 675, 260, 24),
        )
        content.addSubview_(title)
        self.training_type_label = self._label(
            "模板类别",
            AppKit.NSMakeRect(20, 635, 90, 24),
        )
        content.addSubview_(self.training_type_label)
        self.training_type_popup = AppKit.NSPopUpButton.alloc().initWithFrame_pullsDown_(
            AppKit.NSMakeRect(120, 631, 150, 28),
            False,
        )
        self.training_type_popup.addItemsWithTitles_(
            [display for display, _ in self.template_categories]
        )
        content.addSubview_(self.training_type_popup)
        self.training_category_field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(280, 631, 115, 28)
        )
        self.training_category_field.setPlaceholderString_("新增类别")
        content.addSubview_(self.training_category_field)
        self.training_add_category_button = self._button(
            "添加",
            AppKit.NSMakeRect(402, 629, 70, 32),
            self.actions,
            "addTemplateCategory:",
        )
        content.addSubview_(self.training_add_category_button)
        self.training_capture_button = self._button(
            "获取训练画面",
            AppKit.NSMakeRect(480, 629, 120, 32),
            self.actions,
            "captureTrainingFrame:",
        )
        self.training_select_button = self._button(
            "框选怪物",
            AppKit.NSMakeRect(608, 629, 110, 32),
            self.actions,
            "selectMonster:",
        )
        self.training_select_button.setEnabled_(False)
        content.addSubview_(self.training_capture_button)
        content.addSubview_(self.training_select_button)
        self.training_test_button = self._button(
            "测试识别",
            AppKit.NSMakeRect(728, 629, 110, 32),
            self.actions,
            "testMonsters:",
        )
        self.training_test_button.setEnabled_(False)
        content.addSubview_(self.training_test_button)
        self.training_undo_button = self._button(
            "撤销最近模板",
            AppKit.NSMakeRect(846, 629, 130, 32),
            self.actions,
            "undoMonsterTemplate:",
        )
        self.training_undo_button.setEnabled_(False)
        content.addSubview_(self.training_undo_button)
        self.training_delete_button = self._button(
            "删除选中模板",
            AppKit.NSMakeRect(790, 590, 130, 32),
            self.actions,
            "deleteSelectedMonsterTemplate:",
        )
        self.training_delete_button.setEnabled_(False)
        content.addSubview_(self.training_delete_button)
        self.training_test_categories_button = self._button(
            "测试类别：全部",
            AppKit.NSMakeRect(20, 585, 180, 32),
            self.actions,
            "openTestCategories:",
        )
        content.addSubview_(self.training_test_categories_button)
        self.training_info_label = self._label(
            "流程：选择类别 -> 获取画面 -> 框选保存 -> 测试识别 -> 必要时补充模板",
            AppKit.NSMakeRect(210, 585, 560, 24),
        )
        content.addSubview_(self.training_info_label)
        self.training_result_label = self._label(
            "测试结果：尚未测试",
            AppKit.NSMakeRect(20, 555, 960, 24),
        )
        content.addSubview_(self.training_result_label)
        self.training_preview = FrameSelectionCanvas.alloc().initWithFrame_(
            AppKit.NSMakeRect(20, 54, 533, 400)
        )
        self.training_preview.on_rectangle = self._monster_rectangle_selected
        self.training_preview.on_click = self._training_detection_clicked
        content.addSubview_(self.training_preview)

    def _build_route_page(self, content) -> None:
        title = self._label(
            "路径配置",
            AppKit.NSMakeRect(20, 675, 70, 24),
        )
        content.addSubview_(title)
        self.route_group_label = self._label(
            "路线组",
            AppKit.NSMakeRect(95, 675, 55, 24),
        )
        content.addSubview_(self.route_group_label)
        self.route_group_popup = self._popup(
            AppKit.NSMakeRect(150, 671, 90, 28),
            self.route_group_names(),
        )
        self.route_group_popup.setTarget_(self.actions)
        self.route_group_popup.setAction_("routeGroupChanged:")
        self.route_group_popup.selectItemWithTitle_(self.route_active_group)
        content.addSubview_(self.route_group_popup)
        self.route_climb_rise_label = self._label(
            "黄点上移 px",
            AppKit.NSMakeRect(250, 675, 90, 24),
        )
        content.addSubview_(self.route_climb_rise_label)
        self.route_climb_rise_field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(345, 673, 55, 28)
        )
        self.route_climb_rise_field.setStringValue_(
            str(
                self.config.get("route_combat", {}).get(
                    "climb_minimap_rise_px",
                    1,
                )
            )
        )
        self.route_climb_rise_field.setPlaceholderString_("1")
        content.addSubview_(self.route_climb_rise_field)
        self.route_alt_delay_label = self._label(
            "Alt后延迟 ms",
            AppKit.NSMakeRect(475, 675, 105, 24),
        )
        content.addSubview_(self.route_alt_delay_label)
        self.route_alt_delay_field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(585, 673, 70, 28)
        )
        self.route_alt_delay_field.setStringValue_(
            str(
                self.config.get("route_combat", {}).get(
                    "alt_delay_ms",
                    600,
                )
            )
        )
        self.route_alt_delay_field.setPlaceholderString_("600")
        content.addSubview_(self.route_alt_delay_field)
        self.route_no_monster_label = self._label(
            "无怪切换秒数",
            AppKit.NSMakeRect(735, 675, 100, 24),
        )
        content.addSubview_(self.route_no_monster_label)
        self.route_no_monster_field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(840, 673, 70, 28)
        )
        self.route_no_monster_field.setStringValue_(
            str(
                self.config.get("route_combat", {}).get(
                    "no_monster_seconds",
                    20,
                )
            )
        )
        content.addSubview_(self.route_no_monster_field)
        self.route_capture_button = self._button(
            "获取当前画面",
            AppKit.NSMakeRect(20, 631, 120, 32),
            self.actions,
            "captureRouteFrame:",
        )
        self.route_point_button = self._button(
            "框选挂机区域",
            AppKit.NSMakeRect(275, 631, 110, 32),
            self.actions,
            "markRoutePoint:",
        )
        self.route_point_button.setEnabled_(False)
        content.addSubview_(self.route_capture_button)
        content.addSubview_(self.route_point_button)
        self.route_annotations_button = self._button(
            "显示已保存标注",
            AppKit.NSMakeRect(148, 631, 120, 32),
            self.actions,
            "toggleRouteAnnotations:",
        )
        self.route_annotations_button.setEnabled_(False)
        content.addSubview_(self.route_annotations_button)
        self.route_point_name_label = self._label(
            "挂机点名称",
            AppKit.NSMakeRect(395, 635, 78, 24),
        )
        content.addSubview_(self.route_point_name_label)
        self.route_point_name_field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(475, 633, 110, 28)
        )
        self.route_point_name_field.setPlaceholderString_("点1")
        content.addSubview_(self.route_point_name_field)
        self.route_delete_point_label = self._label(
            "删除挂机点",
            AppKit.NSMakeRect(20, 492, 78, 24),
        )
        content.addSubview_(self.route_delete_point_label)
        self.route_delete_point_popup = self._popup(
            AppKit.NSMakeRect(100, 488, 145, 28),
            self.route_point_names(),
        )
        content.addSubview_(self.route_delete_point_popup)
        self.route_delete_point_button = self._button(
            "删除",
            AppKit.NSMakeRect(253, 486, 65, 32),
            self.actions,
            "deleteRoutePoint:",
        )
        content.addSubview_(self.route_delete_point_button)
        self.route_ladder_name_label = self._label(
            "梯子名称",
            AppKit.NSMakeRect(600, 635, 62, 24),
        )
        content.addSubview_(self.route_ladder_name_label)
        self.route_ladder_name_field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(665, 633, 130, 28)
        )
        self.route_ladder_name_field.setPlaceholderString_("蘑菇梯子1")
        content.addSubview_(self.route_ladder_name_field)
        self.route_ladder_duration_label = self._label(
            "爬梯 ms",
            AppKit.NSMakeRect(800, 635, 58, 24),
        )
        content.addSubview_(self.route_ladder_duration_label)
        self.route_ladder_duration_field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(860, 633, 62, 28)
        )
        self.route_ladder_duration_field.setStringValue_("1800")
        content.addSubview_(self.route_ladder_duration_field)
        self.route_record_ladder_button = self._button(
            "记录梯子",
            AppKit.NSMakeRect(925, 631, 70, 32),
            self.actions,
            "recordLadder:",
        )
        self.route_record_ladder_button.setEnabled_(False)
        content.addSubview_(self.route_record_ladder_button)

        self.route_path_label = self._label(
            "路径",
            AppKit.NSMakeRect(20, 585, 40, 24),
        )
        content.addSubview_(self.route_path_label)
        self.route_source_popup = self._popup(
            AppKit.NSMakeRect(62, 581, 100, 28),
            self.route_point_names(),
        )
        content.addSubview_(self.route_source_popup)
        self.route_arrow_label = self._label(
            "->",
            AppKit.NSMakeRect(166, 585, 24, 24),
        )
        content.addSubview_(self.route_arrow_label)
        self.route_target_popup = self._popup(
            AppKit.NSMakeRect(194, 581, 100, 28),
            self.route_point_names(),
        )
        content.addSubview_(self.route_target_popup)
        self.route_new_path_button = self._button(
            "新建路径",
            AppKit.NSMakeRect(302, 579, 90, 32),
            self.actions,
            "newRoutePath:",
        )
        content.addSubview_(self.route_new_path_button)
        self.route_path_popup = self._popup(
            AppKit.NSMakeRect(400, 581, 190, 28),
            self.route_path_names(),
        )
        self.route_path_popup.setTarget_(self.actions)
        self.route_path_popup.setAction_("routePathChanged:")
        content.addSubview_(self.route_path_popup)
        self.route_save_path_button = self._button(
            "保存路径",
            AppKit.NSMakeRect(600, 579, 90, 32),
            self.actions,
            "saveRoutePath:",
        )
        content.addSubview_(self.route_save_path_button)
        self.route_delete_path_button = self._button(
            "删除路径",
            AppKit.NSMakeRect(698, 579, 90, 32),
            self.actions,
            "deleteRoutePath:",
        )
        content.addSubview_(self.route_delete_path_button)
        self.route_remove_step_button = self._button(
            "删除末步",
            AppKit.NSMakeRect(796, 579, 90, 32),
            self.actions,
            "removeRouteStep:",
        )
        content.addSubview_(self.route_remove_step_button)
        self.route_test_start_button = self._button(
            "测试当前路径",
            AppKit.NSMakeRect(894, 579, 96, 32),
            self.actions,
            "startRouteTest:",
        )
        content.addSubview_(self.route_test_start_button)
        self.route_test_stop_button = self._button(
            "停止测试",
            AppKit.NSMakeRect(894, 532, 96, 32),
            self.actions,
            "stopRouteTest:",
        )
        self.route_test_stop_button.setEnabled_(False)
        content.addSubview_(self.route_test_stop_button)

        self.route_step_type_label = self._label(
            "步骤",
            AppKit.NSMakeRect(20, 540, 40, 24),
        )
        content.addSubview_(self.route_step_type_label)
        self.route_step_type_popup = self._popup(
            AppKit.NSMakeRect(62, 536, 155, 28),
            [
                "移动到挂机点",
                "移动到梯子",
                "爬梯",
                "向左移动",
                "向右移动",
                "向左跳跃",
                "向右跳跃",
                "等待",
            ],
        )
        self.route_step_type_popup.setTarget_(self.actions)
        self.route_step_type_popup.setAction_("routeStepTypeChanged:")
        content.addSubview_(self.route_step_type_popup)
        self.route_step_object_popup = self._popup(
            AppKit.NSMakeRect(227, 536, 150, 28),
            self.route_ladder_names() or self.route_point_names(),
        )
        content.addSubview_(self.route_step_object_popup)
        self.route_step_duration_field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(387, 536, 70, 28)
        )
        self.route_step_duration_field.setStringValue_("1000")
        self.route_step_duration_field.setPlaceholderString_("等待/跳跃 ms")
        content.addSubview_(self.route_step_duration_field)
        self.route_step_key_popup = self._popup(
            AppKit.NSMakeRect(467, 536, 90, 28),
            ["向上", "向下"],
        )
        content.addSubview_(self.route_step_key_popup)
        self.route_jump_key_label = self._label(
            "跳跃键",
            AppKit.NSMakeRect(680, 540, 55, 24),
        )
        content.addSubview_(self.route_jump_key_label)
        self.route_jump_key_field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(735, 536, 65, 28)
        )
        self.route_jump_key_field.setStringValue_(
            str(
                self.config.get("route_combat", {}).get(
                    "jump_key",
                    "alt",
                )
            )
        )
        self.route_jump_key_field.setPlaceholderString_("alt")
        content.addSubview_(self.route_jump_key_field)
        self.route_add_step_button = self._button(
            "添加步骤",
            AppKit.NSMakeRect(810, 532, 80, 32),
            self.actions,
            "addRouteStep:",
        )
        content.addSubview_(self.route_add_step_button)
        self.route_steps_label = self._label(
            "当前路径步骤",
            AppKit.NSMakeRect(630, 420, 350, 24),
        )
        content.addSubview_(self.route_steps_label)
        self.route_status_scroll = AppKit.NSScrollView.alloc().initWithFrame_(
            AppKit.NSMakeRect(630, 270, 350, 140)
        )
        self.route_status_scroll.setHasVerticalScroller_(True)
        self.route_status_scroll.setHasHorizontalScroller_(False)
        self.route_status_scroll.setBorderType_(AppKit.NSBezelBorder)
        self.route_status_label = AppKit.NSTextView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, 330, 120)
        )
        self.route_status_label.setEditable_(False)
        self.route_status_label.setSelectable_(True)
        self.route_status_label.setFont_(AppKit.NSFont.systemFontOfSize_(11))
        self.route_status_label.setString_("路径步骤：尚未新建")
        self.route_status_scroll.setDocumentView_(self.route_status_label)
        content.addSubview_(self.route_status_scroll)
        self.route_info_label = self._label(
            "先获取画面；在右侧放大小地图框选挂机区域；同名挂机点可重新框选覆盖；角色在梯子下方时记录梯子。",
            AppKit.NSMakeRect(20, 438, 590, 42),
        )
        self.route_info_label.setUsesSingleLineMode_(False)
        self.route_info_label.setMaximumNumberOfLines_(2)
        self.route_info_label.setLineBreakMode_(
            AppKit.NSLineBreakByWordWrapping
        )
        content.addSubview_(self.route_info_label)
        self.route_preview = FrameSelectionCanvas.alloc().initWithFrame_(
            AppKit.NSMakeRect(20, 54, 590, 355)
        )
        self.route_preview.on_rectangle = self._route_rectangle_selected
        self.route_preview.on_mouse_move = self._route_magnifier_moved
        content.addSubview_(self.route_preview)
        self.route_minimap_label = self._label(
            "放大小地图：框选挂机区域，紫点为梯子/跳跃点",
            AppKit.NSMakeRect(760, 235, 220, 24),
        )
        content.addSubview_(self.route_minimap_label)
        self.route_minimap_calibrate_button = self._button(
            "校准小地图",
            AppKit.NSMakeRect(630, 232, 120, 30),
            self.actions,
            "calibrateRouteMinimap:",
        )
        content.addSubview_(self.route_minimap_calibrate_button)
        self.route_minimap_preview = FrameSelectionCanvas.alloc().initWithFrame_(
            AppKit.NSMakeRect(630, 20, 350, 210)
        )
        self.route_minimap_preview.on_rectangle = (
            self._route_minimap_rectangle_selected
        )
        content.addSubview_(self.route_minimap_preview)
        self._refresh_route_annotations()

    def route_point_names(self) -> list[str]:
        group = self._selected_route_group()
        names = [
            str(point.get("name", "")).strip()
            for point in self.route_points
            if str(point.get("name", "")).strip()
            and route_point_group(point) == group
        ]
        return names or ["未设置"]

    def route_group_names(self) -> list[str]:
        return route_group_names(
            {
                "active_group": self.route_active_group,
                "points": self.route_points,
                "paths": self.route_paths,
            }
        )

    def _selected_route_group(self) -> str:
        if hasattr(self, "route_group_popup"):
            selected = str(
                self.route_group_popup.titleOfSelectedItem() or ""
            ).strip()
            if selected and selected != "未设置":
                self.route_active_group = selected
        return self.route_active_group or "默认"

    def _route_group_for_point(self, point_name: str | None) -> str | None:
        name = str(point_name or "").strip()
        if not name:
            return None
        for point in self.route_points:
            if str(point.get("name", "")).strip() == name:
                return route_point_group(point)
        return None

    def route_ladder_names(self) -> list[str]:
        names = [
            str(ladder.get("name", "")).strip()
            for ladder in self.route_ladders
            if str(ladder.get("name", "")).strip()
        ]
        return names or ["未设置"]

    def route_path_names(self) -> list[str]:
        group = self._selected_route_group()
        names = [
            str(route.get("name", "")).strip()
            for route in self.route_paths
            if str(route.get("name", "")).strip()
            and route_path_group(route) == group
        ]
        return names or ["未设置"]

    @staticmethod
    def _replace_popup_items(popup, titles: list[str]) -> None:
        current = popup.titleOfSelectedItem()
        popup.removeAllItems()
        popup.addItemsWithTitles_(list(titles) or ["未设置"])
        if current and current in titles:
            popup.selectItemWithTitle_(current)

    def _selected_route_annotation_context(
        self,
    ) -> tuple[set[str], set[str]]:
        """Return point and ladder names belonging to the selected path."""
        selected_name = str(
            self.route_current_path_key
            or self.route_path_popup.titleOfSelectedItem()
            or ""
        ).strip()
        route = next(
            (
                item
                for item in self.route_paths
                if str(item.get("name", "")).strip() == selected_name
            ),
            None,
        )
        if route is None:
            source = str(
                self.route_source_popup.titleOfSelectedItem() or ""
            ).strip()
            target = str(
                self.route_target_popup.titleOfSelectedItem() or ""
            ).strip()
            steps = self.route_current_steps
        else:
            source = str(route.get("source", "")).strip()
            target = str(route.get("target", "")).strip()
            steps = [
                step
                for step in (route.get("steps") or [])
                if isinstance(step, dict)
            ]
        point_names = {name for name in (source, target) if name}
        ladder_names = {
            str(step.get("ladder", "")).strip()
            for step in steps
            if str(step.get("type", "")).strip()
            in {"move_to_ladder", "climb"}
            and str(step.get("ladder", "")).strip()
        }
        return point_names, ladder_names

    def _refresh_route_annotations(self) -> None:
        if not hasattr(self, "route_preview"):
            return
        minimap = tuple(self.config.get("minimap", {}).get("region") or ())
        minimap_rect = minimap if len(minimap) == 4 else None
        regions = []
        point_names, ladder_names = self._selected_route_annotation_context()
        if self.route_show_annotations and minimap_rect is not None:
            map_x, map_y, _, _ = minimap_rect
            for point in self.route_points:
                point_name = str(point.get("name", "")).strip()
                if point_name not in point_names:
                    continue
                region = point.get("region")
                if not isinstance(region, (list, tuple)) or len(region) != 4:
                    continue
                x, y, width, height = (int(value) for value in region)
                regions.append(
                    (
                        point_name,
                        (map_x + x, map_y + y, width, height),
                    )
                )
            ladder_markers = [
                (
                    str(ladder.get("name", "")),
                    (
                        map_x + int(ladder["point"][0]),
                        map_y + int(ladder["point"][1]),
                    ),
                )
                for ladder in self.route_ladders
                if str(ladder.get("name", "")).strip() in ladder_names
                if isinstance(ladder.get("point"), (list, tuple))
                and len(ladder["point"]) == 2
            ]
            for name in point_names:
                profile = self.route_combat_profiles.get(name, {})
                fixed = profile.get("fixed_point")
                if isinstance(fixed, (list, tuple)) and len(fixed) == 2:
                    ladder_markers.append(
                        (
                            f"{name}固定点",
                            (
                                map_x + int(fixed[0]),
                                map_y + int(fixed[1]),
                            ),
                        )
                    )
                region = profile.get("monster_region")
                if isinstance(region, (list, tuple)) and len(region) == 4:
                    regions.append(
                        (
                            f"{name}寻怪",
                            tuple(int(value) for value in region),
                        )
                    )
        else:
            ladder_markers = ()
        self.route_preview.setMinimapRect_(minimap_rect)
        self.route_preview.setRouteRegions_(regions)
        self.route_preview.setLadderMarkers_(ladder_markers)
        self.route_preview.setRectangleEnabled_(
            False
        )
        self._refresh_route_minimap_annotations()
        if hasattr(self, "route_group_popup"):
            self._replace_popup_items(
                self.route_group_popup,
                self.route_group_names(),
            )
            self.route_group_popup.selectItemWithTitle_(
                self._selected_route_group()
            )
        self._replace_popup_items(self.route_source_popup, self.route_point_names())
        self._replace_popup_items(self.route_target_popup, self.route_point_names())
        self._replace_popup_items(self.route_path_popup, self.route_path_names())
        if hasattr(self, "route_delete_point_popup"):
            self._replace_popup_items(
                self.route_delete_point_popup,
                self.route_point_names(),
            )
        if hasattr(self, "route_combat_point_popup"):
            self._replace_popup_items(
                self.route_combat_point_popup,
                self.route_point_names(),
            )
            point_names = self.route_point_names()
            if self.route_combat_point_name not in point_names:
                self.route_combat_point_name = (
                    point_names[0] if point_names != ["未设置"] else None
                )
            if self.route_combat_point_name:
                self.route_combat_point_popup.selectItemWithTitle_(
                    self.route_combat_point_name
                )
        self._update_route_step_object_popup()
        self._update_route_status()

    def route_group_changed(self) -> None:
        self._save_current_route_combat_profile()
        self.route_active_group = self._selected_route_group()
        self.route_current_path_key = None
        self.route_current_steps = []
        self.route_show_annotations = False
        if hasattr(self, "route_annotations_button"):
            self.route_annotations_button.setTitle_("显示已保存标注")
        self._refresh_route_annotations()
        self.status_label.setStringValue_(
            f"已切换路线组：{self.route_active_group}"
        )

    def _refresh_route_minimap_annotations(self) -> None:
        if not hasattr(self, "route_minimap_preview"):
            return
        minimap = tuple(self.config.get("minimap", {}).get("region") or ())
        if self.route_minimap_frame is None or len(minimap) != 4:
            return
        scale = self.route_minimap_scale
        regions = []
        point_names, ladder_names = self._selected_route_annotation_context()
        for point in self.route_points if self.route_show_annotations else []:
            point_name = str(point.get("name", "")).strip()
            if point_name not in point_names:
                continue
            region = point.get("region")
            if not isinstance(region, (list, tuple)) or len(region) != 4:
                continue
            x, y, width, height = (int(value) for value in region)
            regions.append(
                (
                    point_name,
                    (x * scale, y * scale, width * scale, height * scale),
                )
            )
        ladder_markers = [
            (
                str(ladder.get("name", "")),
                (
                    int(ladder["point"][0]) * scale,
                    int(ladder["point"][1]) * scale,
                ),
            )
            for ladder in (
                self.route_ladders if self.route_show_annotations else []
            )
            if str(ladder.get("name", "")).strip() in ladder_names
            if isinstance(ladder.get("point"), (list, tuple))
            and len(ladder["point"]) == 2
        ]
        self.route_minimap_preview.setRouteRegions_(regions)
        self.route_minimap_preview.setLadderMarkers_(ladder_markers)
        self.route_minimap_preview.setRectangleEnabled_(
            self.route_rectangle_mode == "point"
        )

    def _update_route_step_object_popup(self) -> None:
        if not hasattr(self, "route_step_type_popup"):
            return
        step_type = ROUTE_STEP_TYPES[
            max(
                0,
                min(
                    len(ROUTE_STEP_TYPES) - 1,
                    self.route_step_type_popup.indexOfSelectedItem(),
                ),
            )
        ]
        if step_type in {"move_to_point"}:
            titles = self.route_point_names()
            enabled = bool(self.route_points)
        elif step_type in {"move_to_ladder", "climb"}:
            titles = self.route_ladder_names()
            enabled = bool(self.route_ladders)
        elif step_type in {"move_left", "move_right"}:
            titles = ["目标挂机点固定标记"]
            enabled = False
        else:
            titles = ["无"]
            enabled = False
        self._replace_popup_items(self.route_step_object_popup, titles)
        self.route_step_object_popup.setEnabled_(enabled)
        self.route_step_key_popup.setEnabled_(step_type == "climb")
        self.route_step_duration_field.setEnabled_(
            step_type not in {
                "move_to_point",
                "move_to_ladder",
                "move_left",
                "move_right",
            }
        )
        if step_type == "climb" and self.route_ladder_names() != ["未设置"]:
            ladder_name = self.route_step_object_popup.titleOfSelectedItem()
            ladder = next(
                (
                    item
                    for item in self.route_ladders
                    if str(item.get("name", "")) == ladder_name
                ),
                None,
            )
            if ladder is not None:
                self.route_step_duration_field.setStringValue_(
                    str(int(ladder.get("climb_duration_ms", 1800)))
                )

    def _update_route_status(self) -> None:
        if not hasattr(self, "route_status_label"):
            return
        if self.route_current_path_key is None:
            self.route_status_label.setString_("路径步骤：尚未新建")
            return
        if self.route_current_steps:
            steps = "\n".join(
                f"{index + 1}.{step_label(step)}"
                for index, step in enumerate(self.route_current_steps)
            )
            summary = f"路径步骤：{self.route_current_path_key}\n{steps}"
        else:
            summary = f"路径步骤：{self.route_current_path_key}\n暂无步骤"
        self.route_status_label.setString_(
            summary
        )
        end = self.route_status_label.string().length()
        self.route_status_label.scrollRangeToVisible_(
            AppKit.NSMakeRange(end, 0)
        )

    def capture_route_frame(self) -> None:
        if self.source is None:
            self.status_label.setStringValue_("请先绑定窗口")
            return
        try:
            frame = self.source.next_frame()
            if frame is None:
                raise RuntimeError("窗口截图为空")
            self.route_frame = frame
            self.route_rectangle_mode = False
            self.route_preview.setImage_pixelSize_(
                _nsimage_from_bgr(frame),
                (frame.shape[1], frame.shape[0]),
            )
            self.route_preview.setSelectionMode_("rect")
            self.route_preview.setSelectionEnabled_(False)
            self.route_preview.setClickEnabled_(False)
            self.route_preview.setRectangleEnabled_(False)
            self.route_preview.setMagnifierEnabled_(False)
            self._set_route_minimap_frame(frame)
            self.route_show_annotations = False
            self.route_annotations_button.setTitle_("显示已保存标注")
            self.route_minimap_calibrate_button.setTitle_("校准小地图")
            self.route_point_button.setEnabled_(True)
            self.route_record_ladder_button.setEnabled_(True)
            self.route_annotations_button.setEnabled_(True)
            self._refresh_route_annotations()
            self.status_label.setStringValue_(
                f"路径页已获取游戏画面，尺寸={frame.shape[1]}x{frame.shape[0]}；"
                "当前为干净标注视图，可在右侧放大小地图标注"
            )
        except Exception as exc:
            logger.exception("unable to capture route frame")
            self.status_label.setStringValue_(f"路径画面获取失败：{exc}")

    def toggle_route_annotations(self) -> None:
        if self.route_frame is None:
            self.status_label.setStringValue_("请先获取路径画面")
            return
        self.route_show_annotations = not self.route_show_annotations
        self.route_annotations_button.setTitle_(
            "隐藏已保存标注"
            if self.route_show_annotations
            else "显示已保存标注"
        )
        self._refresh_route_annotations()
        self.status_label.setStringValue_(
            "已显示已保存的挂机点、梯子和固定配置"
            if self.route_show_annotations
            else "已清除当前画面上的旧标注，配置未删除"
        )

    def toggle_route_point_marking(self) -> None:
        if self.route_frame is None:
            self.status_label.setStringValue_("请先获取路径页游戏画面")
            return
        enabled = self.route_rectangle_mode != "point"
        self.route_rectangle_mode = "point" if enabled else None
        self.route_preview.setRectangleEnabled_(False)
        self.route_preview.setMagnifierEnabled_(False)
        self.route_minimap_preview.setSelectionMode_("rect")
        self.route_minimap_preview.setSelectionEnabled_(False)
        self.route_minimap_preview.setClickEnabled_(False)
        self.route_minimap_preview.setRectangleEnabled_(enabled)
        self.route_minimap_preview.setNeedsDisplay_(True)
        self.route_point_button.setTitle_("框选中" if enabled else "框选挂机区域")
        self.status_label.setStringValue_(
            "请在右侧放大小地图内框选挂机区域"
            if enabled
            else "已停止框选挂机区域"
        )

    def _set_route_minimap_frame(self, frame) -> None:
        minimap = tuple(self.config.get("minimap", {}).get("region") or ())
        if len(minimap) != 4:
            self.route_minimap_frame = None
            self.route_minimap_label.setStringValue_("小地图未配置")
            return
        actual = clip_region_to_frame(frame, minimap)
        cropped = crop_region(frame, minimap)
        if cropped.size == 0:
            self.route_minimap_frame = None
            self.route_minimap_label.setStringValue_("小地图截取为空，请重新校准")
            return
        self.route_minimap_frame = cropped
        self.route_minimap_label.setStringValue_(
            f"小地图 {actual[2]}x{actual[3]}  "
            f"(配置 {minimap[0]},{minimap[1]},{minimap[2]}x{minimap[3]})"
        )
        scale = self.route_minimap_scale
        enlarged = cv2.resize(
            cropped,
            (cropped.shape[1] * scale, cropped.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        self.route_minimap_preview.setImage_pixelSize_(
            _nsimage_from_bgr(enlarged),
            (enlarged.shape[1], enlarged.shape[0]),
        )
        self.route_minimap_preview.setSelectionMode_("rect")
        self.route_minimap_preview.setSelectionEnabled_(False)
        self.route_minimap_preview.setClickEnabled_(False)
        self._refresh_route_minimap_annotations()

    def _route_rectangle_selected(
        self,
        rectangle: tuple[int, int, int, int],
    ) -> None:
        if self.route_rectangle_mode == "minimap":
            if rectangle[2] < 8 or rectangle[3] < 8:
                self.status_label.setStringValue_("小地图区域太小，请重新框选")
                return
            x, y, width, height = clip_region_to_frame(
                self.route_frame,
                rectangle,
            )
            if width < 8 or height < 8:
                self.status_label.setStringValue_("小地图区域无效，请重新框选")
                return
            self.config.setdefault("minimap", {})["region"] = [
                x,
                y,
                width,
                height,
            ]
            self.route_rectangle_mode = False
            self.route_preview.setRectangleEnabled_(False)
            self.route_preview.setMagnifierEnabled_(False)
            self.route_minimap_calibrate_button.setTitle_("校准小地图")
            self._set_route_minimap_frame(self.route_frame)
            self._refresh_route_annotations()
            self.status_label.setStringValue_(
                f"小地图区域已更新：x={x} y={y} w={width} h={height}；"
                "请点击保存配置"
            )
            return
        if self.route_rectangle_mode != "point":
            return
        minimap = tuple(self.config.get("minimap", {}).get("region") or ())
        if len(minimap) == 4:
            map_x, map_y, map_width, map_height = (
                int(value) for value in minimap
            )
            x, y, width, height = (
                int(value) for value in rectangle
            )
            if (
                x < map_x
                or y < map_y
                or x + width > map_x + map_width
                or y + height > map_y + map_height
            ):
                self.status_label.setStringValue_("挂机区域必须完全落在小地图内")
                return
        local = minimap_local_rect(rectangle, minimap)
        if local is None or local[2] < 3 or local[3] < 3:
            self.status_label.setStringValue_("挂机区域必须完全落在小地图内")
            return
        name = str(self.route_point_name_field.stringValue()).strip()
        if not name:
            name = f"点{len(self.route_points) + 1}"
        item = {
            "name": name,
            "region": list(local),
            "group": self._selected_route_group(),
        }
        self.route_points = [
            point
            for point in self.route_points
            if str(point.get("name", "")).strip() != name
        ]
        self.route_points.append(item)
        self.route_combat_profiles.setdefault(name, {})
        if not self.route_combat_point_name:
            self.route_combat_point_name = name
        self.route_rectangle_mode = None
        self.route_point_button.setTitle_("框选挂机区域")
        self.route_preview.setRectangleEnabled_(False)
        if hasattr(self, "route_combat_point_popup"):
            self._replace_popup_items(
                self.route_combat_point_popup,
                self.route_point_names(),
            )
        self._refresh_route_annotations()
        self.status_label.setStringValue_(
            f"已保存挂机点 {name}：x={local[0]} y={local[1]} "
            f"w={local[2]} h={local[3]}"
        )

    def _route_minimap_rectangle_selected(
        self,
        rectangle: tuple[int, int, int, int],
    ) -> None:
        if self.route_rectangle_mode != "point":
            return
        scale = max(1, int(self.route_minimap_scale))
        local = minimap_scaled_rect_to_local(rectangle, scale)
        if local is None:
            self.status_label.setStringValue_("挂机区域坐标无效")
            return
        if local[2] < 3 or local[3] < 3:
            self.status_label.setStringValue_("挂机区域太小，请重新框选")
            return
        name = str(self.route_point_name_field.stringValue()).strip()
        if not name:
            name = f"点{len(self.route_points) + 1}"
        item = {
            "name": name,
            "region": list(local),
            "group": self._selected_route_group(),
        }
        self.route_points = [
            point
            for point in self.route_points
            if str(point.get("name", "")).strip() != name
        ]
        self.route_points.append(item)
        self.route_combat_profiles.setdefault(name, {})
        if not self.route_combat_point_name:
            self.route_combat_point_name = name
        self.route_rectangle_mode = False
        self.route_point_button.setTitle_("框选挂机区域")
        self.route_minimap_preview.setRectangleEnabled_(False)
        self.route_minimap_preview.setNeedsDisplay_(True)
        self._refresh_route_annotations()
        self.status_label.setStringValue_(
            f"已覆盖挂机点 {name}：小地图 x={local[0]} y={local[1]} "
            f"w={local[2]} h={local[3]}，请点击保存配置"
        )

    def calibrate_route_minimap(self) -> None:
        if self.route_frame is None:
            self.status_label.setStringValue_("请先获取路径页游戏画面")
            return
        enabled = self.route_rectangle_mode != "minimap"
        self.route_rectangle_mode = "minimap" if enabled else False
        self.route_preview.setRectangleEnabled_(enabled)
        self.route_preview.setMagnifierEnabled_(enabled)
        self.route_minimap_preview.setRectangleEnabled_(False)
        self.route_minimap_preview.setNeedsDisplay_(True)
        self.route_minimap_calibrate_button.setTitle_(
            "停止校准" if enabled else "校准小地图"
        )
        self.status_label.setStringValue_(
            "请在左侧完整游戏画面中框选完整小地图"
            if enabled
            else "已停止校准小地图"
        )

    def _route_magnifier_moved(self, point: tuple[int, int]) -> None:
        if (
            self.route_rectangle_mode != "minimap"
            or self.route_frame is None
        ):
            return
        frame_height, frame_width = self.route_frame.shape[:2]
        center_x, center_y = (int(value) for value in point)
        radius_x, radius_y = 18, 14
        left = max(0, center_x - radius_x)
        top = max(0, center_y - radius_y)
        right = min(frame_width, center_x + radius_x + 1)
        bottom = min(frame_height, center_y + radius_y + 1)
        crop = self.route_frame[top:bottom, left:right]
        if crop.size == 0:
            return
        enlarged = cv2.resize(
            crop,
            (max(1, crop.shape[1] * 5), max(1, crop.shape[0] * 5)),
            interpolation=cv2.INTER_NEAREST,
        )
        self.route_preview.setMagnifierImage_pixelSize_(
            _nsimage_from_bgr(enlarged),
            (enlarged.shape[1], enlarged.shape[0]),
        )
        self.route_preview.setMagnifierCursor_(
            (
                (center_x - left) * 5,
                (center_y - top) * 5,
            )
        )

    def record_ladder(self) -> None:
        if self.route_frame is None:
            self.status_label.setStringValue_("请先获取路径页游戏画面")
            return
        minimap = tuple(self.config.get("minimap", {}).get("region") or ())
        if len(minimap) != 4:
            self.status_label.setStringValue_("小地图区域未配置")
            return
        try:
            minimap_config = self.config.get("minimap", {})
            hsv = minimap_config.get("player_color_hsv", {})
            lower = tuple(int(value) for value in hsv["lower"])
            upper = tuple(int(value) for value in hsv["upper"])
            observation = observe_minimap(
                self.route_frame,
                minimap,
                lower,
                upper,
            )
            if observation.player_local is None:
                raise ValueError("当前画面没有识别到小地图黄点")
            name = str(self.route_ladder_name_field.stringValue()).strip()
            if not name:
                name = f"梯子{len(self.route_ladders) + 1}"
            duration = int(float(self.route_ladder_duration_field.stringValue()))
            if duration <= 0:
                raise ValueError("爬梯时长必须大于 0 毫秒")
            point = [
                int(observation.player_local.x),
                int(observation.player_local.y),
            ]
            item = {
                "name": name,
                "point": point,
                "climb_duration_ms": duration,
                "group": self._selected_route_group(),
            }
            self.route_ladders = [
                ladder
                for ladder in self.route_ladders
                if str(ladder.get("name", "")).strip() != name
            ]
            self.route_ladders.append(item)
            self._refresh_route_annotations()
            self.status_label.setStringValue_(
                f"已记录梯子 {name}：小地图=({point[0]},{point[1]})，"
                f"向上 {duration}ms"
            )
        except (KeyError, TypeError, ValueError) as exc:
            self.status_label.setStringValue_(f"记录梯子失败：{exc}")

    def route_step_type_changed(self) -> None:
        self._update_route_step_object_popup()

    def new_route_path(self) -> None:
        source = str(self.route_source_popup.titleOfSelectedItem() or "").strip()
        target = str(self.route_target_popup.titleOfSelectedItem() or "").strip()
        names = [
            name
            for name in self.route_point_names()
            if name != "未设置"
        ]
        if len(names) < 2:
            message = "请先创建至少两个挂机点"
            self.status_label.setStringValue_(message)
            self.route_status_label.setString_(f"路径步骤：{message}")
            return
        if source == target:
            source, target = names[0], names[1]
            self.route_source_popup.selectItemWithTitle_(source)
            self.route_target_popup.selectItemWithTitle_(target)
        self.route_current_path_key = path_name(source, target)
        self.route_current_steps = []
        self.route_show_annotations = False
        if hasattr(self, "route_annotations_button"):
            self.route_annotations_button.setTitle_("显示已保存标注")
        self._refresh_route_annotations()
        self.route_path_popup.selectItemWithTitle_(self.route_current_path_key)
        self._update_route_status()
        self.status_label.setStringValue_(
            f"已新建路径 {self.route_current_path_key}，请添加步骤后保存"
        )

    def route_path_changed(self) -> None:
        selected = str(self.route_path_popup.titleOfSelectedItem() or "").strip()
        route = next(
            (
                item
                for item in self.route_paths
                if str(item.get("name", "")).strip() == selected
            ),
            None,
        )
        if route is None:
            self.route_current_path_key = None
            self.route_current_steps = []
        else:
            self.route_current_path_key = selected
            source = str(route.get("source", ""))
            target = str(route.get("target", ""))
            self.route_source_popup.selectItemWithTitle_(source)
            self.route_target_popup.selectItemWithTitle_(target)
            self.route_combat_point_name = source
            if hasattr(self, "route_combat_point_popup"):
                self.route_combat_point_popup.selectItemWithTitle_(source)
            self.route_current_steps = [
                dict(step)
                for step in (route.get("steps") or [])
                if isinstance(step, dict)
            ]
            self.route_show_annotations = True
            self.route_annotations_button.setTitle_("隐藏已保存标注")
            self._refresh_route_annotations()
        self._update_route_status()

    def delete_route_path(self) -> None:
        selected = str(self.route_path_popup.titleOfSelectedItem() or "").strip()
        if not selected or selected == "未设置":
            self.status_label.setStringValue_("请先选择要删除的路径")
            return
        remaining = [
            route
            for route in self.route_paths
            if str(route.get("name", "")).strip() != selected
        ]
        if len(remaining) == len(self.route_paths):
            self.status_label.setStringValue_(f"未找到路径：{selected}")
            return
        previous_paths = self.route_paths
        self.route_paths = remaining
        self.route_current_path_key = None
        self.route_current_steps = []
        self.route_show_annotations = False
        try:
            route_config = self._route_config_from_ui()
            route_config["paths"] = [dict(route) for route in self.route_paths]
            saved_config = dict(self.config)
            saved_config["route_combat"] = route_config
            saved_config = self._make_config_paths_relative(saved_config)
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    saved_config,
                    handle,
                    allow_unicode=True,
                    sort_keys=False,
                )
            self.config = saved_config
            self._refresh_route_annotations()
            self.route_annotations_button.setTitle_("显示已保存标注")
            self.status_label.setStringValue_(f"路径已删除：{selected}")
        except Exception as exc:
            self.route_paths = previous_paths
            logger.exception("unable to delete route path")
            self._refresh_route_annotations()
            self.status_label.setStringValue_(f"路径删除失败：{exc}")

    def delete_route_point(self) -> None:
        selected = str(
            self.route_delete_point_popup.titleOfSelectedItem() or ""
        ).strip()
        if not selected or selected == "未设置":
            self.status_label.setStringValue_("请先选择要删除的挂机点")
            return
        referenced = [
            str(route.get("name", "")).strip()
            for route in self.route_paths
            if str(route.get("source", "")).strip() == selected
            or str(route.get("target", "")).strip() == selected
        ]
        if referenced:
            self.status_label.setStringValue_(
                f"挂机点仍被路径引用，请先删除路径：{', '.join(referenced)}"
            )
            return
        previous_points = self.route_points
        previous_profiles = self.route_combat_profiles
        self.route_points = [
            point
            for point in self.route_points
            if str(point.get("name", "")).strip() != selected
        ]
        self.route_combat_profiles = {
            name: profile
            for name, profile in self.route_combat_profiles.items()
            if name != selected
        }
        if self.route_combat_point_name == selected:
            names = self.route_point_names()
            self.route_combat_point_name = (
                names[0] if names and names[0] != "未设置" else None
            )
        try:
            route_config = self._route_config_from_ui()
            saved_config = dict(self.config)
            saved_config["route_combat"] = route_config
            saved_config = self._make_config_paths_relative(saved_config)
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    saved_config,
                    handle,
                    allow_unicode=True,
                    sort_keys=False,
                )
            self.config = saved_config
            self._refresh_route_annotations()
            self.status_label.setStringValue_(f"挂机点已删除：{selected}")
        except Exception as exc:
            self.route_points = previous_points
            self.route_combat_profiles = previous_profiles
            logger.exception("unable to delete route point")
            self._refresh_route_annotations()
            self.status_label.setStringValue_(f"挂机点删除失败：{exc}")

    def add_route_step(self) -> None:
        if self.route_current_path_key is None:
            self.status_label.setStringValue_("请先点击“新建路径”")
            return
        index = self.route_step_type_popup.indexOfSelectedItem()
        if not 0 <= index < len(ROUTE_STEP_TYPES):
            self.status_label.setStringValue_("步骤类型无效")
            return
        step_type = ROUTE_STEP_TYPES[index]
        step: dict[str, Any] = {"type": step_type}
        selected_object = str(
            self.route_step_object_popup.titleOfSelectedItem() or ""
        ).strip()
        if step_type == "move_to_point":
            if selected_object == "未设置":
                self.status_label.setStringValue_("请先创建挂机点")
                return
            step["point"] = selected_object
        elif step_type in {"move_to_ladder", "climb"}:
            if selected_object == "未设置":
                self.status_label.setStringValue_("请先记录梯子")
                return
            step["ladder"] = selected_object
        if step_type == "climb":
            step["key"] = (
                "down"
                if self.route_step_key_popup.indexOfSelectedItem() == 1
                else "up"
            )
        if step_type not in {
            "move_to_point",
            "move_to_ladder",
            "move_left",
            "move_right",
        }:
            try:
                duration = int(float(self.route_step_duration_field.stringValue()))
            except ValueError:
                self.status_label.setStringValue_("步骤时长必须是整数毫秒")
                return
            if duration <= 0:
                self.status_label.setStringValue_("步骤时长必须大于 0 毫秒")
                return
            step["duration_ms"] = duration
        self.route_current_steps.append(step)
        self._update_route_status()

    def remove_route_step(self) -> None:
        if self.route_current_steps:
            self.route_current_steps.pop()
            self._update_route_status()
            self.status_label.setStringValue_("已删除最后一个路径步骤")
        else:
            self.status_label.setStringValue_("当前路径没有可删除的步骤")

    def _route_config_from_ui(self) -> dict[str, Any]:
        self._save_current_route_combat_profile()
        try:
            no_monster_seconds = float(
                self.route_no_monster_field.stringValue()
            )
        except ValueError as exc:
            raise ValueError("无怪切换秒数必须是数字") from exc
        if no_monster_seconds <= 0:
            raise ValueError("无怪切换秒数必须大于 0")
        try:
            alt_delay_ms = int(float(self.route_alt_delay_field.stringValue()))
        except ValueError as exc:
            raise ValueError("Alt后延迟必须是整数毫秒") from exc
        if alt_delay_ms <= 0:
            raise ValueError("Alt后延迟必须大于 0 毫秒")
        jump_key = str(self.route_jump_key_field.stringValue()).strip().lower()
        if not jump_key:
            raise ValueError("跳跃键不能为空")
        try:
            climb_minimap_rise_px = int(
                float(self.route_climb_rise_field.stringValue())
            )
        except ValueError as exc:
            raise ValueError("黄点上移必须是整数像素") from exc
        if climb_minimap_rise_px <= 0:
            raise ValueError("黄点上移必须大于 0 像素")
        existing = self.config.get("route_combat", {})
        active_group = self._selected_route_group()
        return {
            "enabled": bool(
                existing.get("enabled", True)
            ),
            "mode": "path_combat",
            "active_group": active_group,
            "no_monster_seconds": no_monster_seconds,
            "alt_delay_ms": alt_delay_ms,
            "jump_key": jump_key,
            "obstacle_check_interval_ms": int(
                existing.get("obstacle_check_interval_ms", 300)
            ),
            "obstacle_jump_delay_min_ms": int(
                existing.get("obstacle_jump_delay_min_ms", 500)
            ),
            "obstacle_jump_delay_max_ms": int(
                existing.get("obstacle_jump_delay_max_ms", 700)
            ),
            "climb_minimap_rise_px": climb_minimap_rise_px,
            "point_tolerance": int(existing.get("point_tolerance", 3)),
            "ladder_tolerance": int(existing.get("ladder_tolerance", 3)),
            "jump_point_tolerance": int(
                existing.get("jump_point_tolerance", 8)
            ),
            "points": [
                {
                    **dict(point),
                    "group": route_point_group(point),
                }
                for point in self.route_points
            ],
            "point_profiles": {
                str(name): dict(profile)
                for name, profile in self.route_combat_profiles.items()
            },
            "ladders": [
                {
                    **dict(ladder),
                    "group": route_ladder_group(ladder),
                }
                for ladder in self.route_ladders
            ],
            "paths": [
                {
                    **dict(route),
                    "group": route_path_group(route),
                }
                for route in self.route_paths
            ],
        }

    def save_route_path(self) -> None:
        if self.route_current_path_key is None:
            self.status_label.setStringValue_("请先新建或选择一条路径")
            return
        source, separator, target = self.route_current_path_key.partition(" -> ")
        if not separator:
            self.status_label.setStringValue_("当前路径名称无效")
            return
        record = {
            "name": self.route_current_path_key,
            "source": source,
            "target": target,
            "steps": [dict(step) for step in self.route_current_steps],
            "group": self._selected_route_group(),
        }
        self.route_paths = [
            route
            for route in self.route_paths
            if str(route.get("name", "")).strip()
            != self.route_current_path_key
        ]
        self.route_paths.append(record)
        try:
            route_config = self._route_config_from_ui()
            route_config["paths"] = [dict(route) for route in self.route_paths]
            saved_config = dict(self.config)
            saved_config["route_combat"] = route_config
            saved_config = self._make_config_paths_relative(saved_config)
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    saved_config,
                    handle,
                    allow_unicode=True,
                    sort_keys=False,
                )
            self.config = saved_config
            self._refresh_route_annotations()
            self.status_label.setStringValue_(
                f"路径已保存：{self.route_current_path_key}"
            )
        except Exception as exc:
            logger.exception("unable to save route configuration")
            self.status_label.setStringValue_(f"路径保存失败：{exc}")

    def _selected_route_for_test(self) -> dict[str, Any] | None:
        if self.route_current_path_key and self.route_current_steps:
            source, separator, target = self.route_current_path_key.partition(
                " -> "
            )
            if separator:
                return {
                    "name": self.route_current_path_key,
                    "source": source,
                    "target": target,
                    "steps": [dict(step) for step in self.route_current_steps],
                }
        selected = str(
            self.route_path_popup.titleOfSelectedItem() or ""
        ).strip()
        return next(
            (
                dict(route)
                for route in self.route_paths
                if str(route.get("name", "")).strip() == selected
            ),
            None,
        )

    def start_route_test(self) -> None:
        if self.source is None:
            self.status_label.setStringValue_("请先绑定窗口")
            return
        if self.runtime_running:
            self.status_label.setStringValue_("请先停止主程序运行，再测试路径")
            return
        path = self._selected_route_for_test()
        if path is None:
            self.status_label.setStringValue_("请先选择或保存一条路径")
            return
        if not path.get("steps"):
            self.status_label.setStringValue_("当前路径没有步骤")
            return
        try:
            route_config = self._route_config_from_ui()
            route_config["paths"] = [path]
            test_config = dict(self.config)
            test_config["route_combat"] = route_config
            self.route_test_input = MacOSInputController(
                dry_run=bool(
                    test_config.get("system", {}).get("dry_run", False)
                ),
                target_pid=getattr(
                    self.source.window_info,
                    "owner_pid",
                    None,
                ),
            )
            self.route_test_controller = RouteTestController(
                config=test_config,
                path=path,
            )
            self.route_test_running = True
            self.route_test_start_button.setEnabled_(False)
            self.route_test_stop_button.setEnabled_(True)
            dry_run = bool(
                test_config.get("system", {}).get("dry_run", False)
            )
            mode = "仅日志" if dry_run else "真实按键"
            self.status_label.setStringValue_(
                f"路径测试已开始（{mode}）：{path.get('name', '')}"
            )
            interval_ms = int(
                self.config.get("system", {}).get(
                    "gui_runtime_interval_ms",
                    150,
                )
            )
            interval = max(0.05, interval_ms / 1000.0)
            self.route_test_timer = (
                AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                    interval,
                    self.actions,
                    "routeTestTick:",
                    None,
                    True,
                )
            )
        except Exception as exc:
            logger.exception("unable to start route test")
            self.status_label.setStringValue_(f"路径测试启动失败：{exc}")
            self.stop_route_test()

    def stop_route_test(self) -> None:
        if self.route_test_timer is not None:
            self.route_test_timer.invalidate()
            self.route_test_timer = None
        if self.route_test_controller is not None:
            self.route_test_controller.stop()
        if self.route_test_input is not None:
            self.route_test_input.release_all()
            self.route_test_input = None
        self.route_test_running = False
        self.route_test_controller = None
        if hasattr(self, "route_test_start_button"):
            self.route_test_start_button.setEnabled_(True)
        if hasattr(self, "route_test_stop_button"):
            self.route_test_stop_button.setEnabled_(False)
        if hasattr(self, "status_label"):
            self.status_label.setStringValue_("路径测试已停止")

    def route_test_tick(self) -> None:
        if not self.route_test_running or self.route_test_controller is None:
            return
        try:
            frame = self.source.next_frame()
            if frame is None:
                self.stop_route_test()
                self.status_label.setStringValue_("路径测试停止：窗口画面为空")
                return
            result = self.route_test_controller.process_frame(frame)
            self.route_test_input.emit(result.action)
            player_text = (
                f"名牌=({result.player.name_box[0]},"
                f"{result.player.name_box[1]})"
                if result.player is not None
                else "名牌=未识别"
            )
            minimap_text = (
                f"黄点=({result.minimap.player_local.x},"
                f"{result.minimap.player_local.y})"
                if result.minimap is not None
                and result.minimap.player_local is not None
                else "黄点=未识别"
            )
            self.route_status_label.setString_(
                f"路径测试：步骤={self.route_test_controller.step_index + 1}/"
                f"{len(self.route_test_controller.steps)}；"
                f"状态={result.state}/{self.route_test_controller.phase}；"
                f"动作={result.action}；"
                f"{result.reason}；{player_text}；{minimap_text}"
            )
            end = self.route_status_label.string().length()
            self.route_status_label.scrollRangeToVisible_(
                AppKit.NSMakeRange(end, 0)
            )
            if result.state in {"complete", "failed"}:
                self.route_test_input.release_all()
                self.route_test_input = None
                self.route_test_controller = None
                self.route_test_running = False
                self.route_test_timer.invalidate()
                self.route_test_timer = None
                self.route_test_start_button.setEnabled_(True)
                self.route_test_stop_button.setEnabled_(False)
                self.status_label.setStringValue_(
                    "路径测试完成" if result.state == "complete" else "路径测试失败"
                )
        except Exception as exc:
            logger.exception("route test tick failed")
            self.stop_route_test()
            self.status_label.setStringValue_(f"路径测试失败：{exc}")

    @staticmethod
    def _popup(frame, titles):
        popup = AppKit.NSPopUpButton.alloc().initWithFrame_pullsDown_(
            frame,
            False,
        )
        popup.addItemsWithTitles_(list(titles) or ["未设置"])
        return popup

    @staticmethod
    def _label(text: str, frame):
        label = AppKit.NSTextField.alloc().initWithFrame_(frame)
        label.setStringValue_(text)
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(True)
        return label

    @staticmethod
    def _button(title: str, frame, target, action: str):
        button = AppKit.NSButton.alloc().initWithFrame_(frame)
        button.setTitle_(title)
        button.setBezelStyle_(AppKit.NSBezelStyleRounded)
        button.setTarget_(target)
        button.setAction_(action)
        return button

    @staticmethod
    def _checkbox(title: str, frame):
        checkbox = AppKit.NSButton.alloc().initWithFrame_(frame)
        checkbox.setTitle_(title)
        checkbox.setButtonType_(AppKit.NSButtonTypeSwitch)
        checkbox.setState_(1)
        return checkbox

    @staticmethod
    def _radio(title: str, frame, target, action: str):
        radio = AppKit.NSButton.alloc().initWithFrame_(frame)
        radio.setTitle_(title)
        radio.setButtonType_(AppKit.NSButtonTypeRadio)
        radio.setTarget_(target)
        radio.setAction_(action)
        return radio

    def _attackable_template_categories(self) -> list[tuple[str, str]]:
        excluded_labels = NON_MONSTER_TEMPLATE_CATEGORIES
        return [
            (display, label)
            for display, label in self.template_categories
            if label not in excluded_labels
            and list(self._template_category_dir(label).glob("template_*.png"))
        ]

    def _build_attack_monster_menu(self, content, configured_labels=None):
        """Build a multi-select menu from the current monster template groups."""
        self.attack_monster_content = content
        configured = (
            {str(label) for label in configured_labels}
            if configured_labels
            else {label for _, label in self._attackable_template_categories()}
        )
        self.attack_monster_menu = AppKit.NSMenu.alloc().initWithTitle_("攻击怪物")
        self.attack_monster_menu.setAutoenablesItems_(False)
        self.attack_monster_items = {}
        for display, label in self._attackable_template_categories():
            item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                display,
                "toggleAttackMonster:",
                "",
            )
            item.setTarget_(self.actions)
            item.setRepresentedObject_(label)
            item.setState_(1 if label in configured else 0)
            self.attack_monster_menu.addItem_(item)
            self.attack_monster_items[label] = item
        if not hasattr(self, "attack_monster_button"):
            self.attack_monster_button = self._button(
                "攻击怪物",
                AppKit.NSMakeRect(490, 588, 210, 28),
                self.actions,
                "showAttackMonsterMenu:",
            )
            content.addSubview_(self.attack_monster_button)
        self._update_attack_monster_button_title()

    def _refresh_attack_monster_menu(self) -> None:
        if not hasattr(self, "attack_monster_content"):
            return
        self._build_attack_monster_menu(
            self.attack_monster_content,
            self._selected_attack_monster_labels(),
        )

    def _selected_attack_monster_labels(self) -> list[str]:
        return [
            label
            for label, item in getattr(self, "attack_monster_items", {}).items()
            if item.state() == AppKit.NSControlStateValueOn
        ]

    def _update_attack_monster_button_title(self) -> None:
        selected = self._selected_attack_monster_labels()
        display_by_label = {
            label: display for display, label in self._attackable_template_categories()
        }
        names = [display_by_label.get(label, label) for label in selected]
        if not names:
            summary = "未选择"
        elif len(names) <= 2:
            summary = "、".join(names)
        else:
            summary = f"已选{len(names)}类"
        self.attack_monster_button.setTitle_(f"攻击怪物：{summary}")

    def toggle_attack_monster(self, sender) -> None:
        item = sender
        item.setState_(
            AppKit.NSControlStateValueOff
            if item.state() == AppKit.NSControlStateValueOn
            else AppKit.NSControlStateValueOn
        )
        self._update_attack_monster_button_title()

    def show_attack_monster_menu(self, sender) -> None:
        if not self.attack_monster_menu.numberOfItems():
            return
        bounds = sender.bounds()
        self.attack_monster_menu.popUpMenuPositioningItem_atLocation_inView_(
            None,
            AppKit.NSMakePoint(0, bounds.size.height),
            sender,
        )

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def _load_template_categories(self) -> list[tuple[str, str]]:
        labels: list[str] = []
        monster_config = self.config.get("combat", {}).get("monsters", {})
        labels.extend(str(label) for label in (monster_config.get("template_groups") or {}))
        template_root = self._project_root() / "assets" / "templates" / "monsters"
        if template_root.is_dir():
            labels.extend(path.name for path in sorted(template_root.iterdir()) if path.is_dir())
        labels.append(PLAYER_TEMPLATE_CATEGORY)

        categories: list[tuple[str, str]] = []
        seen: set[str] = set()
        for label in labels:
            label = safe_template_category(label)
            if label in seen:
                continue
            seen.add(label)
            categories.append((DEFAULT_TEMPLATE_DISPLAY_NAMES.get(label, label), label))
        return categories

    def _selected_template_category(self) -> tuple[str, str]:
        selected = str(self.training_type_popup.titleOfSelectedItem())
        for display, label in self.template_categories:
            if selected == display:
                return display, label
        label = safe_template_category(selected)
        return selected, label

    def _template_category_dir(self, label: str) -> Path:
        return (
            self._project_root()
            / "assets"
            / "templates"
            / "monsters"
            / safe_template_category(label)
        )

    def _use_trained_role_templates(self, player_config: dict[str, Any]) -> bool:
        role_template_dir = self._template_category_dir(PLAYER_TEMPLATE_CATEGORY)
        if not list(role_template_dir.glob("template_*.png")):
            return False
        player_config["templates"] = [str(role_template_dir)]
        player_config.pop("template", None)
        player_config.pop("anchor_offset", None)
        player_config["anchor_below_nameplate"] = PLAYER_ANCHOR_BELOW_NAMEPLATE
        monster_config = self.config.get("combat", {}).get("monsters", {})
        player_config["threshold"] = float(monster_config.get("threshold", 0.72))
        player_config["scales"] = [1.0]
        return True

    def add_template_category(self) -> None:
        raw_value = str(self.training_category_field.stringValue()).strip()
        if not raw_value:
            self.training_info_label.setStringValue_("请输入要新增的模板类别")
            return
        label = safe_template_category(raw_value)
        display = DEFAULT_TEMPLATE_DISPLAY_NAMES.get(label, label)
        if label not in {item[1] for item in self.template_categories}:
            self.template_categories.append((display, label))
            self.training_type_popup.addItemWithTitle_(display)
        self.test_template_labels.add(label)
        self._template_category_dir(label).mkdir(parents=True, exist_ok=True)
        self.training_type_popup.selectItemWithTitle_(display)
        self.training_category_field.setStringValue_("")
        self._refresh_attack_monster_menu()
        self._update_test_category_button()
        self.training_info_label.setStringValue_(
            f"已添加模板类别：{display}，可开始框选训练"
        )

    def _test_category_summary(self) -> str:
        selected = [
            display
            for display, label in self.template_categories
            if label in self.test_template_labels
        ]
        if not selected:
            return "未选择"
        if len(selected) == len(self.template_categories):
            return "全部"
        if len(selected) <= 2:
            return "、".join(selected)
        return f"{len(selected)}类"

    def _update_test_category_button(self) -> None:
        self.training_test_categories_button.setTitle_(
            f"测试类别：{self._test_category_summary()}"
        )

    def open_test_categories(self, sender) -> None:
        menu = AppKit.NSMenu.alloc().initWithTitle_("测试类别")
        for display, label in self.template_categories:
            item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                display,
                "toggleTestCategory:",
                "",
            )
            item.setTarget_(self.actions)
            item.setState_(
                AppKit.NSControlStateValueOn
                if label in self.test_template_labels
                else AppKit.NSControlStateValueOff
            )
            menu.addItem_(item)

        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        select_all = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "全选",
            "selectAllTestCategories:",
            "",
        )
        select_all.setTarget_(self.actions)
        menu.addItem_(select_all)
        clear_all = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "清空",
            "clearTestCategories:",
            "",
        )
        clear_all.setTarget_(self.actions)
        menu.addItem_(clear_all)
        menu.popUpMenuPositioningItem_atLocation_inView_(
            None,
            AppKit.NSMakePoint(0, 0),
            sender,
        )

    def toggle_test_category(self, sender) -> None:
        display = str(sender.title())
        label = next(
            (
                label
                for category_display, label in self.template_categories
                if category_display == display
            ),
            None,
        )
        if label is None:
            return
        if label in self.test_template_labels:
            self.test_template_labels.remove(label)
            sender.setState_(AppKit.NSControlStateValueOff)
        else:
            self.test_template_labels.add(label)
            sender.setState_(AppKit.NSControlStateValueOn)
        self._update_test_category_button()

    def select_all_test_categories(self) -> None:
        self.test_template_labels = {
            label for _, label in self.template_categories
        }
        self._update_test_category_button()

    def clear_test_categories(self) -> None:
        self.test_template_labels.clear()
        self._update_test_category_button()

    @staticmethod
    def _percent_string(value: Any) -> str:
        try:
            return str(round(float(value) * 100))
        except (TypeError, ValueError):
            return "0"

    @staticmethod
    def _percent_threshold_from_field(field, label: str) -> float:
        raw_value = str(field.stringValue()).strip().rstrip("%")
        try:
            percent = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{label}必须是 0 到 100 的百分比") from exc
        if percent < 0 or percent > 100:
            raise ValueError(f"{label}必须在 0 到 100 之间")
        return percent / 100.0

    @staticmethod
    def _positive_float_from_field(field, label: str) -> float:
        raw_value = str(field.stringValue()).strip()
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{label}必须是正数") from exc
        if value <= 0:
            raise ValueError(f"{label}必须大于 0")
        return value

    @classmethod
    def _optional_positive_float_from_field(
        cls,
        field,
        label: str,
    ) -> float | None:
        raw_value = str(field.stringValue()).strip()
        if not raw_value:
            return None
        return cls._positive_float_from_field(field, label)

    @staticmethod
    def _milliseconds_field(content, x: int, y: int, values, index: int):
        field = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(x, y, 42, 28)
        )
        try:
            value = values[index]
        except (TypeError, IndexError):
            value = 0
        field.setStringValue_(str(int(float(value))))
        field.setPlaceholderString_("ms")
        content.addSubview_(field)
        return field

    @classmethod
    def _milliseconds_range_from_fields(cls, minimum_field, maximum_field, label):
        minimum = cls._positive_float_from_field(minimum_field, f"{label}最小值")
        maximum = cls._positive_float_from_field(maximum_field, f"{label}最大值")
        if maximum < minimum:
            raise ValueError(f"{label}最大值必须大于等于最小值")
        return [int(minimum), int(maximum)]

    def bind_window(self) -> None:
        self.close_source()
        title = str(self.title_field.stringValue()).strip()
        if not title:
            self.status_label.setStringValue_("窗口标题不能为空")
            return
        try:
            self.source = MacOSWindowSource(
                title,
                output_size=self.config.get("capture", {}).get("output_size"),
            )
        except Exception as exc:
            logger.exception("unable to bind game window")
            self.window_label.setStringValue_("窗口：绑定失败")
            self.status_label.setStringValue_(f"绑定失败：{exc}")
            return

        info = self.source.window_info
        self.window_label.setStringValue_(
            f"窗口：id={info.window_id} {info.owner} / {info.title} "
            f"bounds={info.bounds}"
        )
        self.status_label.setStringValue_("窗口已绑定，可以获取游戏画面")

    def _player_config(self) -> dict[str, Any] | None:
        player_config = dict(self.config.get("combat", {}).get("player", {}))
        player_id = str(self.player_id_field.stringValue()).strip()
        if not player_id:
            player_id = str(player_config.get("id", player_config.get("name", "player")))

        project_root = self._project_root()
        player_config["id"] = player_id
        player_config["name"] = player_id
        if self._use_trained_role_templates(player_config):
            return player_config

        resolved = resolve_player_config(player_id, player_config, project_root)
        if resolved is None:
            self.status_label.setStringValue_(
                "没有可用角色模板：请在模板训练中新增/选择“角色”类别并框选训练"
            )
            return None
        return resolved

    def detect_player(self) -> None:
        if self.source is None:
            self.status_label.setStringValue_("请先绑定窗口")
            return
        try:
            frame = self.source.next_frame()
            if frame is None:
                raise RuntimeError("窗口截图为空")
            self.current_frame = frame
            self.current_screen_player = self._detect_player_in_frame(frame)
            self._update_player_info()
        except Exception as exc:
            logger.exception("unable to detect player")
            self.status_label.setStringValue_(f"识别失败：{exc}")

    def _detect_player_in_frame(self, frame: np.ndarray):
        player_config = self._player_config()
        if player_config is None:
            return None
        player_config["search_region"] = player_search_region(
            frame,
            self.screen_region,
        )
        if self.player_template_path is not None:
            player_config["template"] = str(self.player_template_path)
            player_config.pop("templates", None)
            template = cv2.imread(str(self.player_template_path), cv2.IMREAD_COLOR)
            if template is not None:
                player_config["anchor_offset"] = [
                    template.shape[1] // 2,
                    template.shape[0] + PLAYER_ANCHOR_BELOW_NAMEPLATE,
                ]
        return detect_player_screen(frame, player_config)

    def _update_player_info(self) -> None:
        player_id = str(self.player_id_field.stringValue()).strip()
        player = self.current_screen_player
        if player is None:
            self.preview.setDetectionRect_(None)
            self.player_info_label.setStringValue_(
                f"角色：ID={player_id or '未设置'}，未识别到"
            )
            if player_id:
                self.status_label.setStringValue_(
                    f"未识别到角色名牌 ID={player_id}，请检查名牌模板和画面"
                )
            return
        self.preview.setDetectionRect_(player.name_box)
        self.player_info_label.setStringValue_(
            f"角色：ID={player.name} anchor=({player.anchor[0]},{player.anchor[1]}) "
            f"score={player.score:.3f}"
        )
        self.status_label.setStringValue_(
            f"已识别角色 ID={player.name}，位置=({player.anchor[0]},{player.anchor[1]})"
        )

    def capture_game_frame(self) -> None:
        if self.source is None:
            self.status_label.setStringValue_("请先绑定窗口")
            return
        try:
            frame = self.source.next_frame()
            if frame is None:
                raise RuntimeError("窗口截图为空")
            self.current_frame = frame
            self._set_training_frame(frame)
            self.current_screen_player = None
            self.player_template_path = None
            self.preview.setImage_pixelSize_(
                _nsimage_from_bgr(frame),
                (frame.shape[1], frame.shape[0]),
            )
            self.preview.setPatrolMarkerPoint_(
                self._minimap_point_in_frame(self.patrol_target_point)
            )
            self.preview.setFixedMarkerPoint_(
                self._minimap_point_in_frame(self.fixed_point_target)
            )
            self.preview.setSelectionMode_("band")
            self.preview.setSelectionEnabled_(False)
            self.preview.setRectangleEnabled_(False)
            self.preview.setClickEnabled_(False)
            self.monster_region_button.setEnabled_(True)
            self.fixed_point_button.setEnabled_(True)
            minimap_region = tuple(
                self.config.get("minimap", {}).get("region") or ()
            )
            self.preview.setMinimapRect_(
                minimap_region if len(minimap_region) == 4 else None
            )
            self._load_route_combat_profile(
                self._selected_route_combat_point()
            )
            self.image_info_label.setStringValue_(
                f"画面：已获取，尺寸={frame.shape[1]}x{frame.shape[0]}"
            )
            self.player_info_label.setStringValue_("角色：尚未识别")
            self.status_label.setStringValue_(
                "已获取完整游戏画面；路径挂机点请在路径页分别配置"
            )
        except Exception as exc:
            logger.exception("unable to capture game frame")
            self.status_label.setStringValue_(f"获取失败：{exc}")

    def toggle_screen_band_marking(self) -> None:
        self.status_label.setStringValue_(
            "画面高度标注已移除，请在路径页配置当前挂机点寻怪区域"
        )

    def toggle_patrol_mode(self, sender) -> None:
        self.status_label.setStringValue_("当前仅支持路径打怪")

    def _path_combat_mode_selected(self) -> bool:
        return True

    def _selected_route_combat_point(self) -> str | None:
        if not hasattr(self, "route_combat_point_popup"):
            return self.route_combat_point_name
        selected = str(
            self.route_combat_point_popup.titleOfSelectedItem() or ""
        ).strip()
        if not selected or selected == "未设置":
            return None
        return selected

    def _save_current_route_combat_profile(self) -> None:
        if not self._path_combat_mode_selected():
            return
        name = self.route_combat_point_name or self._selected_route_combat_point()
        if not name or name == "未设置":
            return
        profile = dict(self.route_combat_profiles.get(name, {}))
        profile["fixed_point"] = (
            list(self.fixed_point_target)
            if self.fixed_point_target is not None
            else profile.get("fixed_point")
        )
        profile["fixed_point_enabled"] = (
            self.fixed_point_checkbox.state()
            == AppKit.NSControlStateValueOn
        )
        profile["monster_region"] = (
            list(self.screen_region)
            if self.screen_region is not None
            else profile.get("monster_region")
        )
        self.route_combat_profiles[name] = profile

    def _load_route_combat_profile(self, name: str | None) -> None:
        if not name:
            self.route_combat_point_name = None
            self.fixed_point_target = None
            self.screen_region = None
            if hasattr(self, "fixed_point_checkbox"):
                self.fixed_point_checkbox.setState_(0)
            return

        profile = self.route_combat_profiles.get(name, {})
        fixed_point = profile.get("fixed_point")
        if isinstance(fixed_point, dict):
            fixed_point = fixed_point.get("point")
        if isinstance(fixed_point, (list, tuple)) and len(fixed_point) == 2:
            self.fixed_point_target = (
                int(fixed_point[0]),
                int(fixed_point[1]),
            )
        else:
            self.fixed_point_target = None

        monster_region = profile.get("monster_region")
        if isinstance(monster_region, (list, tuple)) and len(monster_region) == 4:
            self.screen_region = tuple(int(value) for value in monster_region)
        else:
            self.screen_region = None

        if hasattr(self, "fixed_point_checkbox"):
            fixed_enabled = bool(
                profile.get(
                    "fixed_point_enabled",
                    self.fixed_point_target is not None,
                )
            )
            self.fixed_point_checkbox.setState_(1 if fixed_enabled else 0)
        if hasattr(self, "preview"):
            self.preview.setFixedMarkerPoint_(
                self._minimap_point_in_frame(self.fixed_point_target)
            )
            self.preview.setRectangleEnabled_(False)
            self.preview.setDetectionRect_(None)
            self.preview.setRegionRect_(self.screen_region)
        if hasattr(self, "height_info_label"):
            if self.screen_region is None:
                self.height_info_label.setStringValue_(
                    "寻怪区域：当前挂机点尚未标注"
                )
            else:
                x, y, width, height = self.screen_region
                self.height_info_label.setStringValue_(
                    f"寻怪区域：x={x}..{x + width} y={y}..{y + height}"
                )
        self.route_combat_point_name = name

    def run_mode_changed(self) -> None:
        self._load_route_combat_profile(self._selected_route_combat_point())
        self.route_combat_point_popup.setEnabled_(True)
        self.fixed_point_button.setTitle_("记录当前点位")
        self.monster_region_button.setTitle_("框选当前点区域")
        self.status_label.setStringValue_("路径打怪：请选择当前挂机点并配置独立区域")

    def route_combat_point_changed(self) -> None:
        self._save_current_route_combat_profile()
        selected = self._selected_route_combat_point()
        self._load_route_combat_profile(selected)
        if selected:
            self.status_label.setStringValue_(
                f"已切换挂机点 {selected}，请设置并保存该点的固定校准点和寻怪区域"
            )

    def _load_global_combat_markers(self) -> None:
        combat = self.config.get("combat", {})
        fixed_config = combat.get("fixed_point", {})
        point = fixed_config.get("point")
        self.fixed_point_target = (
            (int(point[0]), int(point[1]))
            if isinstance(point, (list, tuple)) and len(point) == 2
            else None
        )
        region = combat.get("monsters", {}).get("region")
        self.screen_region = (
            tuple(int(value) for value in region)
            if isinstance(region, (list, tuple)) and len(region) == 4
            else None
        )
        if hasattr(self, "fixed_point_checkbox"):
            self.fixed_point_checkbox.setState_(
                1 if fixed_config.get("enabled", False) else 0
            )
        if hasattr(self, "preview"):
            self.preview.setFixedMarkerPoint_(
                self._minimap_point_in_frame(self.fixed_point_target)
            )
            self.preview.setRegionRect_(self.screen_region)
        if hasattr(self, "height_info_label"):
            if self.screen_region is None:
                self.height_info_label.setStringValue_("寻怪区域：全局尚未标注")
            else:
                x, y, width, height = self.screen_region
                self.height_info_label.setStringValue_(
                    f"寻怪区域：x={x}..{x + width} y={y}..{y + height}"
                )

    def toggle_patrol_point_marking(self) -> None:
        self.status_label.setStringValue_("普通挂机巡逻已移除，当前仅支持路径打怪")

    def toggle_fixed_point_marking(self) -> None:
        if self.current_frame is None:
            self.status_label.setStringValue_("请先获取游戏画面")
            return
        minimap_region = tuple(
            self.config.get("minimap", {}).get("region") or ()
        )
        if len(minimap_region) != 4:
            self.status_label.setStringValue_("小地图区域未配置")
            return
        try:
            minimap_config = self.config.get("minimap", {})
            hsv = minimap_config.get("player_color_hsv", {})
            lower = tuple(int(value) for value in hsv["lower"])
            upper = tuple(int(value) for value in hsv["upper"])
            observation = observe_minimap(
                self.current_frame,
                minimap_region,
                lower,
                upper,
            )
        except (KeyError, TypeError, ValueError) as exc:
            self.status_label.setStringValue_(f"读取小地图颜色配置失败：{exc}")
            return

        if observation.player_local is None:
            self.status_label.setStringValue_("当前截图没有识别到角色黄色点")
            return

        local_point = (
            int(observation.player_local.x),
            int(observation.player_local.y),
        )
        self.fixed_point_target = local_point
        self.point_marking_kind = None
        self.patrol_point_marking = False
        self.preview.setRectangleEnabled_(False)
        self.preview.setClickEnabled_(False)

        if self._path_combat_mode_selected():
            name = self.route_combat_point_name or self._selected_route_combat_point()
            if name:
                profile = self.route_combat_profiles.setdefault(name, {})
                profile["fixed_point"] = list(local_point)
                profile["fixed_point_enabled"] = True
            self.fixed_point_checkbox.setState_(1)
            display_name = self.route_combat_point_name or "当前"
            message = (
                f"已记录挂机点 {display_name} 固定点="
                f"({local_point[0]},{local_point[1]})"
            )
        else:
            fixed_config = self.config.setdefault("combat", {}).setdefault(
                "fixed_point",
                {},
            )
            fixed_config["point"] = list(local_point)
            fixed_config["enabled"] = True
            message = f"已记录固定点=({local_point[0]},{local_point[1]})"

        self.preview.setFixedMarkerPoint_(
            minimap_frame_point(local_point, minimap_region)
        )
        self.fixed_point_button.setTitle_("记录当前点位")
        self.status_label.setStringValue_(message + "，保存配置后生效")

    def toggle_monster_region_selection(self) -> None:
        if self.current_frame is None:
            self.status_label.setStringValue_("请先获取游戏画面")
            return
        enabled = self.run_rectangle_mode != "monsters"
        self.run_rectangle_mode = "monsters" if enabled else None
        self.preview.setSelectionMode_("rect")
        self.preview.setRectangleEnabled_(enabled)
        self.preview.setSelectionEnabled_(False)
        self.preview.setClickEnabled_(False)
        self.patrol_point_marking = False
        self.point_marking_kind = None
        self.fixed_point_button.setTitle_("记录当前点位")
        self.monster_region_button.setTitle_(
            "拖动框选中" if enabled else "框选寻怪区域"
        )
        self.status_label.setStringValue_(
            "请在游戏画面中框选怪物识别区域"
            if enabled
            else "已退出寻怪区域框选"
        )

    def _run_rectangle_selected(
        self,
        rectangle: tuple[int, int, int, int],
    ) -> None:
        if self.run_rectangle_mode != "monsters":
            self._player_rectangle_selected(rectangle)
            return
        if self.current_frame is None:
            return
        x, y, width, height = rectangle
        frame_height, frame_width = self.current_frame.shape[:2]
        x = max(0, min(frame_width, int(x)))
        y = max(0, min(frame_height, int(y)))
        width = max(0, min(frame_width - x, int(width)))
        height = max(0, min(frame_height - y, int(height)))
        if width < 4 or height < 4:
            self.status_label.setStringValue_("寻怪区域太小")
            return
        region = (x, y, width, height)
        self.screen_region = region
        self.preview.setRegionRect_(region)
        if self._path_combat_mode_selected():
            name = self.route_combat_point_name or self._selected_route_combat_point()
            if name:
                profile = self.route_combat_profiles.setdefault(name, {})
                profile["monster_region"] = list(region)
        else:
            self.config.setdefault("combat", {}).setdefault("monsters", {})[
                "region"
            ] = list(region)
        self.preview.setRectangleEnabled_(False)
        self.preview.setSelectionMode_("band")
        self.run_rectangle_mode = None
        self.monster_region_button.setTitle_(
            "框选当前点区域"
            if self._path_combat_mode_selected()
            else "框选寻怪区域"
        )
        self.height_info_label.setStringValue_(
            f"寻怪区域：x={x}..{x + width} y={y}..{y + height}"
        )
        self.status_label.setStringValue_(
            (
                f"已设置挂机点 {self.route_combat_point_name} 的寻怪区域"
                if self._path_combat_mode_selected()
                else "已设置寻怪区域，只在该区域识别怪物"
            )
        )

    def _patrol_point_clicked(self, point: tuple[int, int]) -> None:
        minimap_region = tuple(
            self.config.get("minimap", {}).get("region") or ()
        )
        if len(minimap_region) != 4:
            self.status_label.setStringValue_("小地图区域未配置")
            return
        local_point = minimap_local_point(point, minimap_region)
        if local_point is None:
            self.status_label.setStringValue_("请点击小地图区域内的目标点")
            return
        if self.point_marking_kind == "fixed":
            self.fixed_point_target = local_point
            if self._path_combat_mode_selected():
                name = self.route_combat_point_name or self._selected_route_combat_point()
                if name:
                    profile = self.route_combat_profiles.setdefault(name, {})
                    profile["fixed_point"] = list(local_point)
                    profile["fixed_point_enabled"] = True
                self.fixed_point_checkbox.setState_(1)
            else:
                self.config.setdefault("combat", {}).setdefault("fixed_point", {})[
                    "point"
                ] = list(local_point)
            self.preview.setFixedMarkerPoint_(point)
            message = (
                f"{self.route_combat_point_name or '当前'}固定战斗点="
                f"({local_point[0]},{local_point[1]})"
                if self._path_combat_mode_selected()
                else f"固定战斗点=({local_point[0]},{local_point[1]})"
            )
        else:
            self.status_label.setStringValue_("当前只支持标记路径挂机点固定位置")
            return
        self.preview.setClickEnabled_(False)
        self.patrol_point_marking = False
        marked_kind = self.point_marking_kind
        self.point_marking_kind = None
        self.fixed_point_button.setTitle_("记录当前点位")
        self.status_label.setStringValue_(message)

    def _minimap_point_in_frame(self, point):
        region = tuple(self.config.get("minimap", {}).get("region") or ())
        return minimap_frame_point(point, region)

    def toggle_player_selection(self) -> None:
        if self.current_frame is None:
            self.status_label.setStringValue_("请先获取游戏画面")
            return
        enabled = not self.preview.rectangle_enabled
        self.preview.setSelectionMode_("rect")
        self.preview.setRectangleEnabled_(enabled)
        self.preview.setSelectionEnabled_(False)
        self.preview.setClickEnabled_(False)
        self.patrol_point_marking = False
        if enabled:
            if hasattr(self, "select_player_button"):
                self.select_player_button.setTitle_("框选名牌中")
            if hasattr(self, "screen_band_button"):
                self.screen_band_button.setEnabled_(False)
            self.status_label.setStringValue_(
                "请只拖动矩形框框住角色名牌，不要包含角色身体"
            )
        else:
            if hasattr(self, "select_player_button"):
                self.select_player_button.setTitle_("框选名牌")
            if hasattr(self, "screen_band_button"):
                self.screen_band_button.setEnabled_(True)
            self.status_label.setStringValue_("已退出角色名牌框选")

    def _player_rectangle_selected(
        self,
        rectangle: tuple[int, int, int, int],
    ) -> None:
        if self.current_frame is None:
            return
        x, y, width, height = rectangle
        frame_height, frame_width = self.current_frame.shape[:2]
        x = max(0, min(frame_width, int(x)))
        y = max(0, min(frame_height, int(y)))
        width = max(0, min(frame_width - x, int(width)))
        height = max(0, min(frame_height - y, int(height)))
        if width < 30 or height < 8:
            self.status_label.setStringValue_("角色名牌框选区域太小，请完整框住文字")
            return

        player_id = str(self.player_id_field.stringValue()).strip() or "player"
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", player_id).strip("._")
        template_dir = Path(__file__).resolve().parents[2] / "work" / "templates"
        template_dir.mkdir(parents=True, exist_ok=True)
        template_path = template_dir / f"player_{safe_id or 'current'}.png"
        template = self.current_frame[y : y + height, x : x + width].copy()
        if not cv2.imwrite(str(template_path), template):
            self.status_label.setStringValue_("角色名牌模板保存失败")
            return

        self.player_template_path = template_path
        self.preview.setRectangleEnabled_(False)
        if hasattr(self, "select_player_button"):
            self.select_player_button.setTitle_("重新框选名牌")
        if hasattr(self, "screen_band_button"):
            self.screen_band_button.setEnabled_(True)
        self.player_info_label.setStringValue_(
            f"角色：已保存名牌模板 {template_path.name}，"
            f"区域=({x},{y},{width},{height})"
        )
        self.status_label.setStringValue_(
            "名牌模板已保存，请点击“识别角色”验证匹配结果"
        )

    def _height_selected(self, band_y: tuple[int, int]) -> None:
        self.preview.setSelectionEnabled_(False)
        if hasattr(self, "screen_band_button"):
            self.screen_band_button.setTitle_("重新标注")
        if self.current_frame is None:
            return
        frame_height, frame_width = self.current_frame.shape[:2]
        self.screen_region = screen_region_from_band(
            band_y,
            frame_width,
            frame_height,
        )
        self.config.setdefault("combat", {}).setdefault("monsters", {})[
            "region"
        ] = list(self.screen_region)
        _, screen_min, _, screen_height = self.screen_region
        screen_max = screen_min + screen_height
        self.height_info_label.setStringValue_(
            f"识别区域：x=0..{frame_width} y={screen_min}..{screen_max} "
            "(直接使用游戏画面坐标)"
        )
        self.status_label.setStringValue_("已标注游戏画面寻怪高度，识别直接使用画面坐标")

    def capture_training_frame(self) -> None:
        if self.source is None:
            self.training_info_label.setStringValue_("请先绑定窗口")
            return
        try:
            frame = self.source.next_frame()
            if frame is None:
                raise RuntimeError("窗口截图为空")
            self._set_training_frame(frame)
        except Exception as exc:
            logger.exception("unable to capture training frame")
            self.training_info_label.setStringValue_(f"获取失败：{exc}")

    def _set_training_frame(self, frame: np.ndarray) -> None:
        self.training_frame = frame
        self.training_preview.setImage_pixelSize_(
            _nsimage_from_bgr(frame),
            (frame.shape[1], frame.shape[0]),
        )
        self.training_preview.setSelectionMode_("rect")
        self.training_preview.setSelectionEnabled_(False)
        self.training_preview.setClickEnabled_(False)
        self.training_preview.setRectangleEnabled_(False)
        self.training_select_button.setEnabled_(True)
        self.training_test_button.setEnabled_(True)
        self.training_undo_button.setEnabled_(
            self.last_monster_template_path is not None
            and self.last_monster_template_path.exists()
        )
        self.training_detections = ()
        self.selected_training_detection = None
        self.training_delete_button.setEnabled_(False)
        self.training_test_streak = 0
        self.training_test_count = 0
        self.training_result_label.setStringValue_("测试结果：尚未测试")
        self.training_info_label.setStringValue_(
            f"训练画面={frame.shape[1]}x{frame.shape[0]}，"
            f"{self._monster_template_status()}，"
            "选择类别后点击“框选怪物”"
        )

    def toggle_monster_selection(self) -> None:
        if self.training_frame is None:
            self.training_info_label.setStringValue_("请先获取训练画面")
            return
        enabled = not self.training_preview.rectangle_enabled
        self.training_preview.setSelectionMode_("rect")
        self.training_preview.setRectangleEnabled_(enabled)
        self.training_preview.setSelectionEnabled_(False)
        self.training_preview.setClickEnabled_(False)
        self.training_select_button.setTitle_("停止框选" if enabled else "框选怪物")
        if enabled:
            self.training_info_label.setStringValue_(
                "连续框选模式：可连续框选多个完整怪物，再次点击“停止框选”关闭"
            )
        else:
            self.training_info_label.setStringValue_("已停止框选")

    def _monster_rectangle_selected(
        self,
        rectangle: tuple[int, int, int, int],
    ) -> None:
        if self.training_frame is None:
            return
        x, y, width, height = rectangle
        frame_height, frame_width = self.training_frame.shape[:2]
        x = max(0, min(frame_width, int(x)))
        y = max(0, min(frame_height, int(y)))
        width = max(0, min(frame_width - x, int(width)))
        height = max(0, min(frame_height - y, int(height)))
        if width < 12 or height < 12:
            self.training_info_label.setStringValue_("框选区域太小，请重新选择")
            return

        display_label, label = self._selected_template_category()
        template_dir = self._template_category_dir(label)
        template_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(template_dir.glob("template_*.png"))
        used_indexes = {
            int(match.group(1))
            for path in existing
            if (match := re.fullmatch(r"template_(\d+)", path.stem))
        }
        next_index = 1
        while next_index in used_indexes:
            next_index += 1
        template_path = template_dir / f"template_{next_index:02d}.png"
        template = self.training_frame[y : y + height, x : x + width].copy()
        if not cv2.imwrite(str(template_path), template):
            self.training_info_label.setStringValue_("模板保存失败")
            return
        self.last_monster_template_path = template_path
        self._refresh_attack_monster_menu()
        self.training_preview.setRectangleEnabled_(True)
        self.training_select_button.setTitle_("停止框选")
        self.training_undo_button.setEnabled_(True)
        self.training_info_label.setStringValue_(
            f"已保存{display_label}模板：{template_path.name}，"
            f"区域=({x},{y},{width},{height})，"
            f"当前类别样本数={len(list(template_dir.glob('template_*.png')))}"
        )

    def undo_monster_template(self) -> None:
        path = self.last_monster_template_path
        if path is None or not path.exists():
            self.training_info_label.setStringValue_("没有可撤销的最近模板")
            self.training_undo_button.setEnabled_(False)
            return

        _, selected_label = self._selected_template_category()
        if selected_label != path.parent.name:
            self.training_info_label.setStringValue_(
                "最近模板属于其他类别，请切换到对应类别后再撤销"
            )
            return

        try:
            path.unlink()
        except OSError as exc:
            self.training_info_label.setStringValue_(f"撤销失败：{exc}")
            return

        self.last_monster_template_path = None
        self._refresh_attack_monster_menu()
        self.training_undo_button.setEnabled_(False)
        self.training_preview.setImage_pixelSize_(
            _nsimage_from_bgr(self.training_frame),
            (self.training_frame.shape[1], self.training_frame.shape[0]),
        )
        self.training_result_label.setStringValue_("测试结果：模板已撤销，请重新测试")
        self.training_info_label.setStringValue_(
            f"已撤销最近模板：{path.name}，当前类别样本数="
            f"{len(list(path.parent.glob('template_*.png')))}"
        )
        self.training_test_streak = 0

    def _training_detection_clicked(self, point: tuple[int, int]) -> None:
        selected = next(
            (
                detection
                for detection in reversed(self.training_detections)
                if _point_in_box(point, detection.box)
            ),
            None,
        )
        self.selected_training_detection = selected
        self.training_delete_button.setEnabled_(
            selected is not None and bool(selected.template_path)
        )
        if selected is None:
            self.training_result_label.setStringValue_(
                f"测试结果：点击位置=({point[0]},{point[1]})，未选中识别框"
            )
            return
        template_name = Path(selected.template_path).name if selected.template_path else "未知"
        self.training_result_label.setStringValue_(
            f"已选中：{selected.label}，模板={template_name}，"
            f"位置=({selected.box[0]},{selected.box[1]})，"
            "如确认误识别，请点击“删除选中模板”"
        )

    def delete_selected_monster_template(self) -> None:
        selected = self.selected_training_detection
        if selected is None or not selected.template_path:
            self.training_result_label.setStringValue_("请先点击一个识别框")
            return
        path = Path(selected.template_path)
        try:
            path.unlink()
        except OSError as exc:
            self.training_result_label.setStringValue_(f"删除失败：{exc}")
            return
        self.selected_training_detection = None
        self._refresh_attack_monster_menu()
        self.training_delete_button.setEnabled_(False)
        self.training_result_label.setStringValue_(
            f"已删除错误模板：{path.name}，正在重新测试"
        )
        self.test_monsters(use_current_frame=True)

    def _monster_template_status(self) -> str:
        display_label, label = self._selected_template_category()
        template_dir = self._template_category_dir(label)
        count = len(list(template_dir.glob("template_*.png")))
        return f"当前类别={display_label}，已有样本={count}"

    def _monster_runtime_config(self) -> dict[str, Any]:
        monster_config = dict(self.config.get("combat", {}).get("monsters", {}))
        groups = {}
        project_root = self._project_root()
        for label, paths in (monster_config.get("template_groups") or {}).items():
            resolved_paths = []
            values = [paths] if isinstance(paths, (str, Path)) else paths
            for value in values:
                path = Path(value)
                if not path.is_absolute():
                    path = project_root / path
                resolved_paths.append(str(path))
            groups[str(label)] = resolved_paths
        for _, label in self.template_categories:
            template_dir = self._template_category_dir(label)
            if template_dir.is_dir() and list(template_dir.glob("template_*.png")):
                groups[label] = [str(template_dir)]
        monster_config["template_groups"] = groups
        allowed_labels = sorted(
            label for label in self.test_template_labels
        )
        monster_config["allowed_labels"] = (
            allowed_labels
            if allowed_labels
            else ["__no_selected_monster_category__"]
        )
        return monster_config

    def _training_detector_config(self, frame: np.ndarray) -> dict[str, Any]:
        frame_height, frame_width = frame.shape[:2]
        detector_config = self._monster_runtime_config()
        detector_config["region"] = [0, 0, frame_width, frame_height]
        return detector_config

    def test_monsters(self, use_current_frame: bool = False) -> None:
        if self.training_frame is None:
            self.training_info_label.setStringValue_("请先获取训练画面")
            return
        try:
            frame = self.training_frame
            if not use_current_frame and self.source is not None:
                captured = self.source.next_frame()
                if captured is not None:
                    frame = captured
            self.training_frame = frame
            # The run page and each route point can have their own ROI. Those
            # coordinates are not necessarily valid for the training frame
            # currently shown here, so template testing must scan this frame
            # directly instead of inheriting a stale runtime ROI.
            frame_height, frame_width = frame.shape[:2]
            detector_config = self._training_detector_config(frame)
            logger.info(
                "monster template test: frame=%sx%s region=%s labels=%s "
                "threshold=%s template_groups=%s",
                frame_width,
                frame_height,
                detector_config["region"],
                detector_config.get("allowed_labels", []),
                detector_config.get("threshold"),
                sorted(detector_config.get("template_groups", {})),
            )
            detections = detect_monsters(frame, detector_config)
            self.training_detections = detections
            self.training_test_count += 1
            if detections:
                self.training_test_streak += 1
            else:
                self.training_test_streak = 0
            annotated = render_monster_debug_frame(frame, detections)
            self.training_preview.setImage_pixelSize_(
                _nsimage_from_bgr(annotated),
                (annotated.shape[1], annotated.shape[0]),
            )
            self.training_preview.setSelectionMode_("rect")
            self.training_preview.setSelectionEnabled_(False)
            self.training_preview.setRectangleEnabled_(False)
            self.training_preview.setClickEnabled_(True)
            if detections:
                labels = ", ".join(
                    f"{label}={count}"
                    for label, count in self._count_detection_labels(detections).items()
                )
                completion = (
                    "模板状态：连续 3 次测试通过"
                    if self.training_test_streak >= 3
                    else f"连续成功 {self.training_test_streak}/3 次"
                )
                self.training_result_label.setStringValue_(
                    f"测试结果：识别到 {labels}；{completion}"
                )
            else:
                self.training_result_label.setStringValue_(
                    "测试结果：未识别到怪物；请点击“框选怪物”补充当前类别模板"
                )
            self.training_info_label.setStringValue_(
                f"测试次数={self.training_test_count}，"
                f"当前类别={self.training_type_popup.titleOfSelectedItem()}，"
                f"测试范围=全画面 {frame_width}x{frame_height}，彩色框为识别结果"
            )
            self.training_preview.setRectangleEnabled_(False)
        except Exception as exc:
            logger.exception("unable to test monster templates")
            self.training_result_label.setStringValue_(f"测试失败：{exc}")

    @staticmethod
    def _count_detection_labels(detections) -> dict[str, int]:
        counts: dict[str, int] = {}
        for detection in detections:
            counts[detection.label] = counts.get(detection.label, 0) + 1
        return counts

    def _runtime_config(self) -> dict[str, Any]:
        runtime_config = dict(self.config)
        runtime_config["game_window"] = dict(
            self.config.get("game_window", {})
        )
        runtime_config["game_window"]["title_contains"] = str(
            self.title_field.stringValue()
        ).strip()
        runtime_config["minimap"] = dict(self.config.get("minimap", {}))
        combat = dict(self.config.get("combat", {}))
        project_root = Path(__file__).resolve().parents[2]
        player_config = dict(combat.get("player", {}))
        if self.current_frame is not None:
            player_config["search_region"] = player_search_region(
                self.current_frame,
                self.screen_region,
            )
        if self.player_template_path is not None:
            player_config["template"] = str(self.player_template_path)
            player_config.pop("templates", None)
            template = cv2.imread(str(self.player_template_path), cv2.IMREAD_COLOR)
            if template is not None:
                player_config["anchor_offset"] = [
                    template.shape[1] // 2,
                    template.shape[0] + PLAYER_ANCHOR_BELOW_NAMEPLATE,
                ]
        else:
            self._use_trained_role_templates(player_config)
        if not player_config.get("templates") and player_config.get("template"):
            template_path = Path(str(player_config["template"]))
            if not template_path.is_absolute():
                player_config["template"] = str(project_root / template_path)
        combat["player"] = player_config

        attack_range = dict(combat.get("attack_range", {}))
        raw_distance = str(self.attack_distance_field.stringValue()).strip()
        try:
            distance = int(raw_distance)
        except ValueError as exc:
            raise ValueError("攻击距离必须是正整数") from exc
        if distance <= 0:
            raise ValueError("攻击距离必须大于 0")
        attack_range["distance"] = distance
        combat["attack_range"] = attack_range

        health_config = dict(combat.get("health", {}))
        health_config["low_threshold"] = self._percent_threshold_from_field(
            self.heal_threshold_field,
            "回血阈值",
        )
        combat["health"] = health_config

        mana_config = dict(combat.get("mana", {}))
        mana_config["low_threshold"] = self._percent_threshold_from_field(
            self.mana_threshold_field,
            "回蓝阈值",
        )
        combat["mana"] = mana_config

        # 路径打怪只使用挂机点之间的已保存路径，不启用普通挂机巡逻。
        patrol_config = dict(combat.get("patrol", {}))
        patrol_config["enabled"] = False
        combat["patrol"] = patrol_config

        fixed_point_config = dict(combat.get("fixed_point", {}))
        fixed_enabled = (
            self.fixed_point_checkbox.state()
            == AppKit.NSControlStateValueOn
        )
        return_distance_px = self._positive_float_from_field(
            self.fixed_point_return_distance_field,
            "无怪回定点距离",
        )
        fixed_point_config["enabled"] = fixed_enabled
        fixed_point_config["point"] = (
            list(self.fixed_point_target)
            if self.fixed_point_target is not None
            else None
        )
        fixed_point_config["return_distance_px"] = return_distance_px
        fixed_point_config.pop("no_target_return_seconds", None)
        fixed_point_config.pop("check_interval_ms", None)
        fixed_point_config.pop("check_interval_seconds", None)
        fixed_point_config["target_tolerance"] = max(
            0,
            int(fixed_point_config.get("target_tolerance", 2)),
        )
        fixed_point_config["roam_enabled"] = False
        minimap = runtime_config["minimap"]
        minimap_region = tuple(minimap.get("region") or ())
        if fixed_enabled and self.fixed_point_target is None:
            raise ValueError("启用固定点战斗前，请先标记定点")
        if fixed_enabled and len(minimap_region) != 4:
            raise ValueError("固定小地图区域未配置，请检查 config/default.yaml")
        combat["fixed_point"] = fixed_point_config

        periodic_key_config = dict(combat.get("periodic_key", {}))
        periodic_interval = self._optional_positive_float_from_field(
            self.periodic_key_interval_field,
            "按键3间隔秒数",
        )
        periodic_key_config["enabled"] = periodic_interval is not None
        periodic_key_config["interval_seconds"] = periodic_interval or 0
        periodic_key_config["key"] = "3"
        combat["periodic_key"] = periodic_key_config

        selected_labels = self._selected_attack_monster_labels()
        if not selected_labels:
            raise ValueError("至少选择一种攻击怪物")
        monster_config = dict(combat.get("monsters", {}))
        monster_config["allowed_labels"] = selected_labels
        template_groups = {}
        for label, paths in (monster_config.get("template_groups") or {}).items():
            values = [paths] if isinstance(paths, (str, Path)) else paths
            template_groups[str(label)] = [
                str(
                    Path(value)
                    if Path(value).is_absolute()
                    else project_root / Path(value)
                )
                for value in values
            ]
        for _, label in self.template_categories:
            if label in NON_MONSTER_TEMPLATE_CATEGORIES:
                continue
            template_dir = self._template_category_dir(label)
            if template_dir.is_dir() and list(template_dir.glob("template_*.png")):
                template_groups[label] = [str(template_dir)]
        monster_config["template_groups"] = template_groups
        if self.screen_region is not None:
            monster_config["region"] = list(self.screen_region)
        else:
            # The minimap rectangle is only for yellow-dot position tracking.
            # Never reuse it as the monster ROI when no gameplay band was marked.
            frame = self.current_frame
            if frame is not None:
                monster_config["region"] = [
                    0,
                    0,
                    int(frame.shape[1]),
                    int(frame.shape[0]),
                ]
        combat["monsters"] = monster_config
        pickup_config = dict(combat.get("pickup", {}))
        pickup_interval = self._positive_float_from_field(
            self.pickup_interval_field,
            "捡金币周期",
        )
        pickup_config["enabled"] = True
        pickup_config["interval_seconds"] = pickup_interval
        pickup_config["scan_seconds"] = 1.0
        coin_config = dict(pickup_config.get("coins", {}))
        coin_config["enabled"] = True
        coin_template_dir = self._template_category_dir("金币")
        if coin_template_dir.is_dir() and list(
            coin_template_dir.glob("template_*.png")
        ):
            coin_config["template_groups"] = {
                "金币": [str(coin_template_dir)]
            }
        if self.screen_region is not None:
            coin_config["region"] = list(self.screen_region)
        pickup_config["coins"] = coin_config
        combat["pickup"] = pickup_config
        runtime_config["combat"] = combat
        route_config = self._route_config_from_ui()
        selected_route_point = self._selected_route_combat_point()
        selected_group = self._route_group_for_point(selected_route_point)
        if selected_group:
            route_config["active_group"] = selected_group
        runtime_config["route_combat"] = route_config
        runtime_config["system"] = dict(self.config.get("system", {}))
        runtime_config["system"]["dry_run"] = False
        return runtime_config

    def save_config(self) -> None:
        try:
            saved_config = self._runtime_config()
            saved_config["system"]["dry_run"] = bool(
                self.config.get("system", {}).get("dry_run", False)
            )
            self._make_config_paths_relative(saved_config)
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    saved_config,
                    handle,
                    allow_unicode=True,
                    sort_keys=False,
                )
            self.config = saved_config
            self.status_label.setStringValue_(
                f"配置已保存：{self.config_path}"
            )
        except Exception as exc:
            logger.exception("unable to save configuration")
            self.status_label.setStringValue_(f"保存失败：{exc}")

    def _make_config_paths_relative(self, value: Any) -> Any:
        project_root = self._project_root()
        if isinstance(value, dict):
            for key, item in value.items():
                value[key] = self._make_config_paths_relative(item)
            return value
        if isinstance(value, list):
            return [self._make_config_paths_relative(item) for item in value]
        if not isinstance(value, str):
            return value
        path = Path(value)
        if not path.is_absolute():
            return value
        try:
            return str(path.relative_to(project_root))
        except ValueError:
            return value

    def start_runtime(self) -> None:
        if self.source is None:
            self.status_label.setStringValue_("请先绑定窗口")
            return
        if self.runtime_running:
            return
        try:
            self.runtime_controller = VisionController(
                source=self.source,
                input_controller=MacOSInputController(
                    dry_run=False,
                    target_pid=getattr(
                        self.source.window_info,
                        "owner_pid",
                        None,
                    ),
                ),
                config=self._runtime_config(),
            )
        except Exception as exc:
            self.status_label.setStringValue_(f"启动失败：{exc}")
            return
        self.runtime_running = True
        self.runtime_log_lines = []
        self.runtime_log_state = {}
        self._append_runtime_log("识别已启动")
        self.start_button.setEnabled_(False)
        self.stop_button.setEnabled_(True)
        self.status_label.setStringValue_(
            "识别已开始，正在输出真实按键"
        )
        interval_ms = int(
            self.config.get("system", {}).get("gui_runtime_interval_ms", 500)
        )
        if interval_ms <= 0:
            raise ValueError("识别间隔必须大于 0 毫秒")
        interval = interval_ms / 1000.0
        self.runtime_timer = AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            interval,
            self.actions,
            "runtimeTick:",
            None,
            True,
        )

    def stop_runtime(self) -> None:
        if self.runtime_timer is not None:
            self.runtime_timer.invalidate()
            self.runtime_timer = None
        if self.runtime_controller is not None:
            self.runtime_controller.input.release_all()
        self.runtime_running = False
        self.runtime_controller = None
        self._runtime_tick_busy = False
        self._append_runtime_log("识别已停止")
        self.start_button.setEnabled_(True)
        self.stop_button.setEnabled_(False)
        self.status_label.setStringValue_("识别已停止")

    def _append_runtime_log(self, message: str) -> None:
        if not hasattr(self, "runtime_log_view"):
            return
        text = str(message).strip()
        if not text:
            return
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.runtime_log_lines.append(f"[{stamp}] {text}")
        self.runtime_log_lines = self.runtime_log_lines[-300:]
        self.runtime_log_view.setString_("\n".join(self.runtime_log_lines) + "\n")
        end = self.runtime_log_view.string().length()
        self.runtime_log_view.scrollRangeToVisible_(
            AppKit.NSMakeRange(end, 0)
        )

    def runtime_tick(self) -> None:
        if not self.runtime_running or self.runtime_controller is None:
            return
        if self._runtime_tick_busy:
            logger.debug("runtime tick skipped: previous frame is still processing")
            return
        self._runtime_tick_busy = True
        try:
            frame = self.source.next_frame()
            if frame is None:
                self.stop_runtime()
                self.status_label.setStringValue_("识别已停止：窗口画面为空")
                return
            combat = self.runtime_controller.process_frame(frame)
            self.runtime_controller.input.emit(combat.action or "none")
            self.status_label.setStringValue_("识别运行中")
            for message in runtime_event_messages(
                combat,
                self.runtime_controller,
                self.runtime_log_state,
            ):
                self._append_runtime_log(message)
        except Exception as exc:
            logger.exception("runtime tick failed")
            self.stop_runtime()
            self.status_label.setStringValue_("识别异常，已停止")
            self._append_runtime_log("识别异常，已停止")
        finally:
            self._runtime_tick_busy = False

    def close_source(self) -> None:
        self.stop_route_test()
        self.stop_runtime()
        if self.source is not None:
            self.source.close()
            self.source = None

    def run(self) -> None:
        self.window.makeKeyAndOrderFront_(None)
        self.app.activateIgnoringOtherApps_(True)
        AppKit.NSApp.run()


def _point_in_box(
    point: tuple[int, int],
    box: tuple[int, int, int, int],
) -> bool:
    px, py = point
    x, y, width, height = box
    return x <= px <= x + width and y <= py <= y + height


def run_gui(config_path: Path) -> None:
    app = MapleVisionApp(load_config(config_path), config_path)
    app.run()
