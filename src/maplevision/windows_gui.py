from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime
import logging
import os
from pathlib import Path
import re
import subprocess
import tkinter as tk
from tkinter import ttk
from typing import Any

import cv2
import numpy as np
import yaml

from .config.loader import load_config
from .decision.controller import VisionController
from .gui_common import (
    application_root,
    clip_region_to_frame,
    crop_region,
    ensure_editor_data_layout,
    minimap_local_point,
    percentage_threshold,
    positive_int,
    resolve_app_path,
    runtime_event_messages,
    safe_template_category,
)
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
from .platform.windows.capture import WindowsWindowSource
from .platform.windows.input import WindowsInputController
from .route_test import RouteTestController
from .vision.minimap import observe_minimap
from .vision.monster import detect_monsters, render_monster_debug_frame
from .vision.image_io import write_image
from .vision.player_screen import detect_player_screen, resolve_player_config

logger = logging.getLogger(__name__)


class WindowsMapleVisionApp:
    """Windows editor for runtime, template training, and route authoring."""

    def __init__(self, config: dict[str, Any], config_path: Path):
        self.config = config
        self.config_path = config_path
        self._ensure_data_layout()
        self.source: WindowsWindowSource | None = None
        self.runtime_controller: VisionController | None = None
        self.runtime_running = False
        self.runtime_after_id: str | None = None
        self.current_frame: np.ndarray | None = None
        self.screen_region: tuple[int, int, int, int] | None = None
        self.fixed_point_target: tuple[int, int] | None = None
        self.selection_mode: str | None = None
        self.selection_start: tuple[float, float] | None = None
        self.selection_rect_id: int | None = None
        self.display_scale = 1.0
        self._photo = None
        self._runtime_tick_busy = False
        self.runtime_log_lines: list[str] = []
        self.runtime_log_state: dict[str, Any] = {}
        self.training_frame: np.ndarray | None = None
        self.training_detections = ()
        self.training_selected_detection = None
        self.training_selection_mode = False
        self.training_selection_start: tuple[float, float] | None = None
        self.training_rect_id: int | None = None
        self.training_display_scale = 1.0
        self._training_photo = None
        self.last_template_path: Path | None = None
        self.training_test_count = 0
        self.training_test_streak = 0
        self.route_frame: np.ndarray | None = None
        self.route_selection_mode = False
        self.route_selection_start: tuple[float, float] | None = None
        self.route_rect_id: int | None = None
        self.route_display_scale = 1.0
        self._route_photo = None
        self._route_magnifier_photo = None
        self.route_magnifier_id: int | None = None
        self.route_minimap_scale = 3
        self.route_minimap_photo = None
        self.route_minimap_display_scale = 1.0
        self.route_show_annotations = False
        route_config = self.config.get("route_combat", {})
        self.route_active_group = active_route_group(route_config)
        self.route_points = [
            dict(item)
            for item in (route_config.get("points") or [])
            if isinstance(item, dict)
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
        fixed_config = self.config.get("combat", {}).get("fixed_point", {})
        fixed_point = fixed_config.get("point")
        if isinstance(fixed_point, (list, tuple)) and len(fixed_point) == 2:
            self.fixed_point_target = (
                int(fixed_point[0]),
                int(fixed_point[1]),
            )
        self.route_ladders = [
            dict(item)
            for item in (route_config.get("ladders") or [])
            if isinstance(item, dict)
        ]
        self.route_paths = [
            dict(item)
            for item in (route_config.get("paths") or [])
            if isinstance(item, dict)
        ]
        self.route_current_path_key: str | None = None
        self.route_current_steps: list[dict[str, Any]] = []
        self.route_test_controller: RouteTestController | None = None
        self.route_test_input: WindowsInputController | None = None
        self.route_test_running = False
        self.route_test_after_id: str | None = None

        self.root = tk.Tk()
        self.root.title("MapleVision Windows")
        self.root.minsize(980, 760)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_widgets()

    def _build_widgets(self) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(root, padding=(10, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(1, weight=1)

        ttk.Label(toolbar, text="窗口标题").grid(
            row=0,
            column=0,
            padx=(0, 6),
            sticky="w",
        )
        self.title_var = tk.StringVar(
            value=str(
                self.config.get("game_window", {}).get(
                    "title_contains",
                    "MapleStory",
                )
            )
        )
        ttk.Entry(toolbar, textvariable=self.title_var, width=28).grid(
            row=0,
            column=1,
            padx=(0, 8),
            sticky="ew",
        )
        ttk.Button(
            toolbar,
            text="绑定窗口",
            command=self.bind_window,
        ).grid(row=0, column=2, padx=4)
        ttk.Button(
            toolbar,
            text="保存配置",
            command=self.save_config,
        ).grid(row=0, column=3, padx=4)

        options = ttk.Frame(root, padding=(10, 0, 10, 8))
        options.grid(row=1, column=0, sticky="new")
        self.dry_run_var = tk.BooleanVar(
            value=bool(self.config.get("system", {}).get("dry_run", True))
        )
        ttk.Checkbutton(
            options,
            text="dry-run（不发送按键）",
            variable=self.dry_run_var,
        ).grid(row=0, column=0, padx=(0, 12), sticky="w")

        ttk.Label(options, text="Windows按键模式").grid(
            row=0,
            column=1,
            padx=(0, 6),
            sticky="w",
        )
        input_mode = str(
            self.config.get("system", {}).get(
                "windows_input_mode",
                "post_message",
            )
        )
        self.input_mode_var = tk.StringVar(value=input_mode)
        ttk.Combobox(
            options,
            textvariable=self.input_mode_var,
            values=("post_message", "foreground"),
            state="readonly",
            width=16,
        ).grid(row=0, column=2, padx=(0, 12), sticky="w")
        ttk.Label(
            options,
            text="post_message不抢焦点；部分游戏只接受foreground",
        ).grid(row=0, column=3, sticky="w")

        self.window_label = ttk.Label(
            root,
            text="窗口：未绑定",
            padding=(10, 0),
        )
        self.window_label.grid(row=2, column=0, sticky="ew")

        self.notebook = ttk.Notebook(root)
        self.notebook.grid(row=3, column=0, sticky="nsew", padx=10, pady=6)
        root.rowconfigure(3, weight=1)
        self.runtime_page = ttk.Frame(self.notebook, padding=8)
        self.training_page = ttk.Frame(self.notebook, padding=8)
        self.route_page = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.runtime_page, text="运行识别")
        self.notebook.add(self.training_page, text="怪物模板训练")
        self.notebook.add(self.route_page, text="路径")
        self._build_runtime_page()
        self._build_training_page()
        self._build_route_page()

        bottom = ttk.Frame(root, padding=(10, 4, 10, 8))
        bottom.grid(row=4, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="请先绑定游戏窗口")
        ttk.Label(
            bottom,
            textvariable=self.status_var,
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        self.region_var = tk.StringVar(value="寻怪区域：未设置")
        ttk.Label(
            bottom,
            textvariable=self.region_var,
            anchor="w",
        ).grid(row=1, column=0, sticky="ew")

    def _make_canvas(self, parent):
        canvas = tk.Canvas(
            parent,
            width=800,
            height=520,
            background="#202124",
            highlightthickness=1,
            highlightbackground="#777777",
        )
        scroll_y = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_x = ttk.Scrollbar(parent, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
        canvas.grid(row=0, column=0, rowspan=2, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        return canvas

    def _build_runtime_page(self) -> None:
        page = self.runtime_page
        controls = ttk.Frame(page)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.start_button = ttk.Button(
            controls, text="开始", command=self.start_runtime, state="disabled"
        )
        self.start_button.grid(row=0, column=0, padx=(0, 4))
        self.stop_button = ttk.Button(
            controls, text="停止", command=self.stop_runtime, state="disabled"
        )
        self.stop_button.grid(row=0, column=1, padx=4)
        ttk.Button(
            controls, text="获取游戏画面", command=self.capture_game_frame
        ).grid(row=0, column=2, padx=4)
        ttk.Button(
            controls, text="标注寻怪区域", command=self.toggle_region_marking
        ).grid(row=0, column=3, padx=4)
        ttk.Label(
            controls,
            text="运行页使用当前窗口画面和已保存模板",
        ).grid(row=0, column=4, padx=12, sticky="w")

        settings = ttk.LabelFrame(page, text="运行参数", padding=(6, 4))
        settings.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        for column in range(8):
            settings.columnconfigure(column, weight=0)
        settings.columnconfigure(7, weight=1)
        ttk.Label(settings, text="捡金币每").grid(row=0, column=0, padx=3)
        pickup_config = self.config.get("combat", {}).get("pickup", {})
        self.pickup_interval_var = tk.StringVar(
            value=str(pickup_config.get("interval_seconds", 30))
        )
        ttk.Entry(
            settings,
            textvariable=self.pickup_interval_var,
            width=7,
        ).grid(row=0, column=1, padx=2)
        ttk.Label(settings, text="秒启动（默认开启，扫描1秒）").grid(
            row=0,
            column=2,
            padx=3,
            sticky="w",
        )
        ttk.Label(settings, text="攻击距离").grid(
            row=0, column=3, padx=(16, 3), sticky="w"
        )
        attack_range = self.config.get("combat", {}).get("attack_range", {})
        self.attack_distance_var = tk.StringVar(
            value=str(attack_range.get("distance", attack_range.get("x", 50)))
        )
        ttk.Entry(
            settings,
            textvariable=self.attack_distance_var,
            width=7,
        ).grid(row=0, column=4, padx=2)
        ttk.Label(settings, text="像素").grid(row=0, column=5, padx=(0, 3))
        health_config = self.config.get("combat", {}).get("health", {})
        mana_config = self.config.get("combat", {}).get("mana", {})
        self.heal_threshold_var = tk.StringVar(
            value=str(float(health_config.get("low_threshold", 0.4)) * 100).rstrip("0").rstrip(".")
        )
        ttk.Label(settings, text="回血").grid(
            row=0, column=6, padx=(16, 3), sticky="w"
        )
        ttk.Entry(
            settings,
            textvariable=self.heal_threshold_var,
            width=6,
        ).grid(row=0, column=7, padx=2, sticky="w")
        ttk.Label(settings, text="% 按键1").grid(row=0, column=8, padx=(0, 3))
        ttk.Label(settings, text="回蓝").grid(
            row=0, column=9, padx=(12, 3), sticky="w"
        )
        self.mana_threshold_var = tk.StringVar(
            value=str(float(mana_config.get("low_threshold", 0.3)) * 100).rstrip("0").rstrip(".")
        )
        ttk.Entry(
            settings,
            textvariable=self.mana_threshold_var,
            width=6,
        ).grid(row=0, column=10, padx=2)
        ttk.Label(settings, text="% 按键2").grid(row=0, column=11, padx=(0, 3))

        ttk.Label(settings, text="角色ID").grid(
            row=1, column=0, padx=3, pady=(5, 0), sticky="w"
        )
        player_config = self.config.get("combat", {}).get("player", {})
        self.player_id_var = tk.StringVar(
            value=str(player_config.get("id", player_config.get("name", "")))
        )
        ttk.Entry(
            settings,
            textvariable=self.player_id_var,
            width=18,
        ).grid(row=1, column=1, columnspan=2, padx=2, pady=(5, 0), sticky="w")
        self.detect_player_button = ttk.Button(
            settings,
            text="识别角色",
            command=self.detect_player,
            state="disabled",
        )
        self.detect_player_button.grid(
            row=1, column=3, padx=(16, 3), pady=(5, 0), sticky="w"
        )
        fixed_config = self.config.get("combat", {}).get("fixed_point", {})
        self.fixed_point_enabled_var = tk.BooleanVar(
            value=bool(fixed_config.get("enabled", False))
        )
        ttk.Checkbutton(
            settings,
            text="固定点战斗",
            variable=self.fixed_point_enabled_var,
        ).grid(row=1, column=4, columnspan=2, padx=3, pady=(5, 0), sticky="w")
        self.fixed_point_button = ttk.Button(
            settings,
            text="记录当前黄点",
            command=self.mark_current_fixed_point,
            state="disabled",
        )
        self.fixed_point_button.grid(
            row=1, column=6, columnspan=2, padx=3, pady=(5, 0), sticky="w"
        )
        ttk.Label(settings, text="回位距离px").grid(
            row=1, column=8, padx=(12, 3), pady=(5, 0), sticky="w"
        )
        self.fixed_point_return_distance_var = tk.StringVar(
            value=str(fixed_config.get("return_distance_px", 10))
        )
        ttk.Entry(
            settings,
            textvariable=self.fixed_point_return_distance_var,
            width=7,
        ).grid(row=1, column=9, padx=2, pady=(5, 0))

        ttk.Label(settings, text="攻击怪物").grid(
            row=2, column=0, padx=3, pady=(5, 0), sticky="w"
        )
        self._build_attack_monster_menu(settings, row=2, column=1)
        frame = ttk.Frame(page)
        frame.grid(row=2, column=0, sticky="nsew")
        page.rowconfigure(2, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        canvas_frame = ttk.Frame(frame)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.canvas = self._make_canvas(canvas_frame)
        self.canvas.bind("<ButtonPress-1>", self._selection_start)
        self.canvas.bind("<B1-Motion>", self._selection_drag)
        self.canvas.bind("<ButtonRelease-1>", self._selection_end)
        log_frame = ttk.LabelFrame(frame, text="识别日志", padding=6)
        log_frame.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.runtime_log = tk.Text(
            log_frame,
            width=42,
            height=20,
            state="disabled",
            wrap="none",
        )
        self.runtime_log.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.runtime_log.yview,
        )
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.runtime_log.configure(yscrollcommand=log_scroll.set)
        self.runtime_log.insert("1.0", "尚未启动识别\n")

    def _build_training_page(self) -> None:
        page = self.training_page
        controls = ttk.Frame(page)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        labels = self._template_categories()
        self.training_label_var = tk.StringVar(
            value=labels[0][0] if labels else "自定义"
        )
        ttk.Label(controls, text="模板类别").grid(row=0, column=0, padx=4)
        self.training_category_combo = ttk.Combobox(
            controls,
            textvariable=self.training_label_var,
            values=[display for display, _ in labels],
            state="readonly",
            width=18,
        )
        self.training_category_combo.grid(row=0, column=1, padx=4)
        self.training_category_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._training_status(),
        )
        self.training_new_category_var = tk.StringVar()
        ttk.Entry(
            controls,
            textvariable=self.training_new_category_var,
            width=14,
        ).grid(row=0, column=2, padx=4)
        ttk.Button(
            controls, text="添加类别", command=self.add_template_category
        ).grid(row=0, column=3, padx=4)
        ttk.Button(
            controls, text="获取训练画面", command=self.capture_training_frame
        ).grid(row=0, column=4, padx=4)
        self.training_select_button = ttk.Button(
            controls, text="框选怪物", command=self.toggle_training_selection,
            state="disabled",
        )
        self.training_select_button.grid(row=0, column=5, padx=4)
        self.training_test_button = ttk.Button(
            controls, text="测试识别", command=self.test_monsters,
            state="disabled",
        )
        self.training_test_button.grid(row=0, column=6, padx=4)
        ttk.Button(
            controls, text="撤销最近模板", command=self.undo_template
        ).grid(row=0, column=7, padx=4)
        ttk.Button(
            controls, text="删除选中模板", command=self.delete_selected_template
        ).grid(row=0, column=8, padx=4)
        ttk.Button(
            controls, text="打开模板目录", command=self.open_template_directory
        ).grid(row=0, column=9, padx=4)

        body = ttk.Frame(page)
        body.grid(row=1, column=0, sticky="nsew")
        page.rowconfigure(1, weight=1)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.training_canvas = self._make_canvas(body)
        self.training_canvas.bind(
            "<ButtonPress-1>", self._training_selection_start
        )
        self.training_canvas.bind("<B1-Motion>", self._training_selection_drag)
        self.training_canvas.bind(
            "<ButtonRelease-1>", self._training_selection_end
        )
        self.training_canvas.bind(
            "<Button-3>", self._training_detection_click
        )
        side = ttk.Frame(body, padding=(8, 0, 0, 0))
        side.grid(row=0, column=2, sticky="ns")
        ttk.Label(side, text="测试类别（多选）").pack(anchor="w")
        self.training_category_list = tk.Listbox(
            side,
            selectmode=tk.MULTIPLE,
            height=12,
            exportselection=False,
        )
        for _, label in labels:
            self.training_category_list.insert(tk.END, label)
        self.training_category_list.pack(fill="y", pady=4)
        ttk.Button(
            side, text="全选", command=lambda: self._select_training_categories(True)
        ).pack(fill="x", pady=2)
        ttk.Button(
            side, text="清空", command=lambda: self._select_training_categories(False)
        ).pack(fill="x", pady=2)
        ttk.Separator(side, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(side, text="当前类别样本").pack(anchor="w")
        self.training_samples_list = tk.Listbox(
            side,
            height=10,
            width=24,
            exportselection=False,
        )
        self.training_samples_list.pack(fill="both", expand=True, pady=4)
        ttk.Button(
            side, text="刷新样本", command=self._refresh_training_samples
        ).pack(fill="x", pady=2)
        ttk.Button(
            side, text="删除选中样本", command=self.delete_selected_sample
        ).pack(fill="x", pady=2)
        self.training_info_var = tk.StringVar(value="请先获取训练画面")
        self.training_result_var = tk.StringVar(value="测试结果：尚未测试")
        ttk.Label(
            page, textvariable=self.training_info_var, anchor="w"
        ).grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(
            page, textvariable=self.training_result_var, anchor="w"
        ).grid(row=3, column=0, sticky="ew")

    def _build_route_page(self) -> None:
        page = self.route_page
        route_config = self.config.get("route_combat", {})
        top = ttk.Frame(page)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(
            top, text="获取路径画面", command=self.capture_route_frame
        ).grid(row=0, column=0, padx=3)
        self.route_annotations_button = ttk.Button(
            top,
            text="显示已保存标注",
            command=self.toggle_route_annotations,
            state="disabled",
        )
        self.route_annotations_button.grid(row=0, column=1, padx=3)
        self.route_minimap_calibrate_button = ttk.Button(
            top,
            text="校准小地图",
            command=self.toggle_route_minimap_calibration,
            state="disabled",
        )
        self.route_minimap_calibrate_button.grid(
            row=1, column=0, padx=3, pady=(3, 0), sticky="w"
        )
        ttk.Label(top, text="路线组").grid(
            row=1, column=2, padx=(12, 3), pady=(3, 0), sticky="w"
        )
        self.route_group_var = tk.StringVar(value=self.route_active_group)
        self.route_group_combo = ttk.Combobox(
            top,
            textvariable=self.route_group_var,
            state="readonly",
            width=14,
        )
        self.route_group_combo.grid(
            row=1, column=3, padx=3, pady=(3, 0), sticky="w"
        )
        self.route_group_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.route_group_changed(),
        )
        self.route_point_name_var = tk.StringVar()
        ttk.Label(top, text="挂机点").grid(row=0, column=2, padx=3)
        ttk.Entry(top, textvariable=self.route_point_name_var, width=12).grid(
            row=0, column=3, padx=3
        )
        self.route_mark_button = ttk.Button(
            top, text="框选挂机区域", command=self.toggle_route_selection,
            state="disabled",
        )
        self.route_mark_button.grid(row=0, column=4, padx=3)
        self.route_ladder_name_var = tk.StringVar()
        ttk.Label(top, text="梯子/跳跃点").grid(row=0, column=5, padx=3)
        ttk.Entry(top, textvariable=self.route_ladder_name_var, width=14).grid(
            row=0, column=6, padx=3
        )
        self.route_ladder_duration_var = tk.StringVar(value="1800")
        ttk.Label(top, text="时长ms").grid(row=0, column=7, padx=3)
        ttk.Entry(
            top, textvariable=self.route_ladder_duration_var, width=8
        ).grid(row=0, column=8, padx=3)
        self.route_ladder_button = ttk.Button(
            top, text="记录黄点", command=self.record_ladder, state="disabled"
        )
        self.route_ladder_button.grid(row=0, column=9, padx=3)

        profile = ttk.LabelFrame(page, text="当前挂机点独立配置", padding=5)
        profile.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(profile, text="挂机点").grid(row=0, column=0, padx=3)
        self.route_profile_point_var = tk.StringVar()
        self.route_profile_point_combo = ttk.Combobox(
            profile,
            textvariable=self.route_profile_point_var,
            state="readonly",
            width=16,
        )
        self.route_profile_point_combo.grid(row=0, column=1, padx=3)
        self.route_profile_point_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.route_profile_changed(),
        )
        self.route_profile_fixed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            profile,
            text="固定点战斗",
            variable=self.route_profile_fixed_var,
            command=self.save_current_route_profile,
        ).grid(row=0, column=2, padx=6)
        self.route_profile_fixed_button = ttk.Button(
            profile,
            text="标记固定点",
            command=self.toggle_route_profile_fixed,
            state="disabled",
        )
        self.route_profile_fixed_button.grid(row=0, column=3, padx=3)
        self.route_profile_region_button = ttk.Button(
            profile,
            text="框选寻怪区域",
            command=self.toggle_route_profile_region,
            state="disabled",
        )
        self.route_profile_region_button.grid(row=0, column=4, padx=3)
        ttk.Button(
            profile,
            text="保存当前点",
            command=self.save_current_route_profile,
        ).grid(row=0, column=5, padx=3)
        ttk.Button(
            profile,
            text="删除当前点",
            command=self.delete_route_point,
        ).grid(row=0, column=6, padx=3)
        self.route_profile_status_var = tk.StringVar(
            value="先创建挂机点并选择它"
        )
        ttk.Label(
            profile,
            textvariable=self.route_profile_status_var,
            anchor="w",
        ).grid(row=1, column=0, columnspan=7, sticky="ew", padx=3, pady=(4, 0))

        route_options = ttk.Frame(page)
        route_options.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(route_options, text="Alt后延迟ms").grid(row=0, column=0, padx=3)
        self.route_alt_delay_var = tk.StringVar(
            value=str(route_config.get("alt_delay_ms", 600))
        )
        ttk.Entry(
            route_options, textvariable=self.route_alt_delay_var, width=8
        ).grid(row=0, column=1, padx=3)
        ttk.Label(route_options, text="黄点上移px").grid(row=0, column=2, padx=3)
        self.route_rise_var = tk.StringVar(
            value=str(route_config.get("climb_minimap_rise_px", 1))
        )
        ttk.Entry(
            route_options, textvariable=self.route_rise_var, width=8
        ).grid(row=0, column=3, padx=3)
        ttk.Label(route_options, text="跳跃键").grid(row=0, column=4, padx=3)
        self.route_jump_key_var = tk.StringVar(
            value=str(route_config.get("jump_key", "alt"))
        )
        ttk.Entry(
            route_options, textvariable=self.route_jump_key_var, width=8
        ).grid(row=0, column=5, padx=3)
        ttk.Label(route_options, text="无怪切换秒").grid(row=0, column=6, padx=3)
        self.route_no_monster_var = tk.StringVar(
            value=str(route_config.get("no_monster_seconds", 20))
        )
        ttk.Entry(
            route_options, textvariable=self.route_no_monster_var, width=8
        ).grid(row=0, column=7, padx=3)
        editor = ttk.Frame(page)
        editor.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(editor, text="路径").grid(row=0, column=0, padx=3)
        self.route_source_var = tk.StringVar()
        self.route_target_var = tk.StringVar()
        self.route_source_combo = ttk.Combobox(
            editor, textvariable=self.route_source_var, width=12
        )
        self.route_source_combo.grid(row=0, column=1, padx=3)
        ttk.Label(editor, text="->").grid(row=0, column=2)
        self.route_target_combo = ttk.Combobox(
            editor, textvariable=self.route_target_var, width=12
        )
        self.route_target_combo.grid(row=0, column=3, padx=3)
        ttk.Button(
            editor, text="新建路径", command=self.new_route_path
        ).grid(row=0, column=4, padx=3)
        self.route_path_var = tk.StringVar()
        self.route_path_combo = ttk.Combobox(
            editor, textvariable=self.route_path_var, width=24
        )
        self.route_path_combo.grid(row=0, column=5, padx=3)
        self.route_path_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self.route_path_changed()
        )
        ttk.Button(
            editor, text="保存路径", command=self.save_route_path
        ).grid(row=0, column=6, padx=3)
        ttk.Button(
            editor, text="删除路径", command=self.delete_route_path
        ).grid(row=0, column=7, padx=3)
        ttk.Button(
            editor, text="删除末步", command=self.remove_route_step
        ).grid(row=1, column=0, padx=3, pady=(4, 0), sticky="w")
        self.route_test_start_button = ttk.Button(
            editor, text="测试当前路径", command=self.start_route_test,
            state="disabled",
        )
        self.route_test_start_button.grid(row=1, column=1, padx=3, pady=(4, 0), sticky="w")
        self.route_test_stop_button = ttk.Button(
            editor, text="停止测试", command=self.stop_route_test,
            state="disabled",
        )
        self.route_test_stop_button.grid(row=1, column=2, padx=3, pady=(4, 0), sticky="w")

        steps = ttk.Frame(page)
        steps.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(steps, text="步骤类型").grid(row=0, column=0, padx=3)
        self.route_step_type_var = tk.StringVar(value=ROUTE_STEP_TYPES[0])
        self.route_step_type_combo = ttk.Combobox(
            steps,
            textvariable=self.route_step_type_var,
            values=list(ROUTE_STEP_TYPES),
            state="readonly",
            width=18,
        )
        self.route_step_type_combo.grid(row=0, column=1, padx=3)
        self.route_step_type_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._update_route_objects()
        )
        ttk.Label(steps, text="对象").grid(row=0, column=2, padx=3)
        self.route_step_object_var = tk.StringVar()
        self.route_step_object_combo = ttk.Combobox(
            steps, textvariable=self.route_step_object_var, width=18
        )
        self.route_step_object_combo.grid(row=0, column=3, padx=3)
        ttk.Label(steps, text="时长ms（等待/跳跃）").grid(
            row=0, column=4, padx=3
        )
        self.route_step_duration_var = tk.StringVar(value="1000")
        self.route_step_duration_entry = ttk.Entry(
            steps, textvariable=self.route_step_duration_var, width=9
        )
        self.route_step_duration_entry.grid(row=0, column=5, padx=3)
        ttk.Button(
            steps, text="添加步骤", command=self.add_route_step
        ).grid(row=0, column=6, padx=3)
        self.route_status_var = tk.StringVar(value="路径步骤：尚未新建")
        ttk.Label(
            page, textvariable=self.route_status_var, anchor="w", justify="left"
        ).grid(row=5, column=0, sticky="ew")
        route_body = ttk.Frame(page)
        route_body.grid(row=6, column=0, sticky="nsew")
        page.rowconfigure(6, weight=1)
        route_body.columnconfigure(0, weight=1)
        route_body.rowconfigure(0, weight=1)
        self.route_canvas = self._make_canvas(route_body)
        self.route_canvas.bind("<ButtonPress-1>", self._route_selection_start)
        self.route_canvas.bind("<B1-Motion>", self._route_selection_drag)
        self.route_canvas.bind("<ButtonRelease-1>", self._route_selection_end)
        self.route_canvas.bind("<Motion>", self._route_magnifier_motion)
        minimap_panel = ttk.LabelFrame(
            route_body,
            text="放大小地图标注",
            padding=6,
        )
        minimap_panel.grid(row=0, column=2, sticky="nsew", padx=(8, 0))
        self.route_minimap_canvas = tk.Canvas(
            minimap_panel,
            width=310,
            height=230,
            background="#202124",
            highlightthickness=1,
            highlightbackground="#777777",
        )
        self.route_minimap_info_var = tk.StringVar(value="小地图未获取")
        ttk.Label(
            minimap_panel,
            textvariable=self.route_minimap_info_var,
            anchor="w",
        ).pack(fill="x", pady=(0, 3))
        self.route_minimap_canvas.pack(fill="both", expand=True)
        self.route_minimap_canvas.bind(
            "<ButtonPress-1>", self._route_minimap_selection_start
        )
        self.route_minimap_canvas.bind(
            "<B1-Motion>", self._route_minimap_selection_drag
        )
        self.route_minimap_canvas.bind(
            "<ButtonRelease-1>", self._route_minimap_selection_end
        )
        route_steps_side = ttk.Frame(route_body, padding=(8, 0, 0, 0))
        route_steps_side.grid(row=0, column=3, sticky="ns")
        ttk.Label(route_steps_side, text="当前路径步骤").pack(anchor="w")
        self.route_steps_list = tk.Listbox(
            route_steps_side,
            width=34,
            height=18,
            exportselection=False,
        )
        self.route_steps_list.pack(
            side="left",
            fill="both",
            expand=True,
            pady=4,
        )
        route_steps_scroll = ttk.Scrollbar(
            route_steps_side,
            orient="vertical",
            command=self.route_steps_list.yview,
        )
        route_steps_scroll.pack(side="right", fill="y", pady=4)
        self.route_steps_list.configure(yscrollcommand=route_steps_scroll.set)
        ttk.Button(
            route_steps_side,
            text="删除选中步骤",
            command=self.remove_selected_route_step,
        ).pack(fill="x", pady=2)
        self._refresh_training_samples()
        self._refresh_route_controls()

    @staticmethod
    def _project_root() -> Path:
        return application_root()

    def _ensure_data_layout(self) -> None:
        """Create the editable data folders used by the GUI on first launch."""
        ensure_editor_data_layout(self.config, self._project_root())

    def _template_categories(self) -> list[tuple[str, str]]:
        display_names = {
            "yellow_mushroom": "大型黄蘑菇",
            "red_snail": "红色蜗牛",
            "small_mushroom": "小型蘑菇",
            "角色": "角色",
            "血条": "血条",
            "金币": "金币",
        }
        labels: set[str] = set()
        monster_config = self.config.get("combat", {}).get("monsters", {})
        labels.update(
            str(label)
            for label in (monster_config.get("template_groups") or {})
        )
        template_root = self._project_root() / "assets" / "templates" / "monsters"
        if template_root.is_dir():
            labels.update(
                path.name for path in template_root.iterdir() if path.is_dir()
            )
        ordered = [
            "yellow_mushroom",
            "red_snail",
            "small_mushroom",
            "角色",
            "金币",
            "血条",
        ]
        ordered.extend(sorted(labels - set(ordered)))
        return [
            (display_names.get(label, label), label)
            for label in ordered
            if label in labels
        ]

    def _template_label_from_display(self, display: str) -> str:
        for label, category in self._template_categories():
            if label == display:
                return category
        return safe_template_category(display)

    def _attackable_template_categories(self) -> list[tuple[str, str]]:
        excluded = {"角色", "金币", "血条"}
        return [
            (display, label)
            for display, label in self._template_categories()
            if label not in excluded
            and list(self._template_dir(label).glob("template_*.png"))
        ]

    def _build_attack_monster_menu(
        self,
        parent,
        row: int = 1,
        column: int = 1,
    ) -> None:
        configured = {
            str(label)
            for label in self.config.get("combat", {})
            .get("monsters", {})
            .get("allowed_labels", [])
        }
        categories = self._attackable_template_categories()
        if not configured:
            configured = {label for _, label in categories}
        self.attack_monster_vars: dict[str, tk.BooleanVar] = {}
        self.attack_monster_menu = tk.Menu(parent, tearoff=False)
        for display, label in categories:
            variable = tk.BooleanVar(value=label in configured)
            self.attack_monster_vars[label] = variable
            self.attack_monster_menu.add_checkbutton(
                label=display,
                variable=variable,
                command=self._update_attack_monster_button,
            )
        self.attack_monster_button = ttk.Menubutton(
            parent,
            text="攻击怪物",
            menu=self.attack_monster_menu,
            width=22,
        )
        self.attack_monster_button.grid(
            row=row,
            column=column,
            columnspan=2,
            padx=3,
            pady=(4, 0),
            sticky="w",
        )
        self._update_attack_monster_button()

    def _selected_attack_monster_labels(self) -> list[str]:
        return [
            label
            for label, variable in getattr(self, "attack_monster_vars", {}).items()
            if variable.get()
        ]

    def _update_attack_monster_button(self) -> None:
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
        self.attack_monster_button.configure(text=f"攻击怪物：{summary}")

    def _refresh_attack_monster_menu(self) -> None:
        if not hasattr(self, "attack_monster_menu"):
            return
        selected = set(self._selected_attack_monster_labels())
        categories = self._attackable_template_categories()
        self.attack_monster_menu.delete(0, tk.END)
        self.attack_monster_vars = {}
        if not selected:
            selected = {label for _, label in categories}
        for display, label in categories:
            variable = tk.BooleanVar(value=label in selected)
            self.attack_monster_vars[label] = variable
            self.attack_monster_menu.add_checkbutton(
                label=display,
                variable=variable,
                command=self._update_attack_monster_button,
            )
        self._update_attack_monster_button()

    def _template_dir(self, label: str) -> Path:
        return self._project_root() / "assets" / "templates" / "monsters" / label

    def _selected_training_label(self) -> str:
        return self._template_label_from_display(
            self.training_label_var.get().strip()
        )

    def _training_status(self) -> None:
        label = self._selected_training_label()
        count = len(list(self._template_dir(label).glob("template_*.png")))
        self.training_info_var.set(f"当前类别={label}，已有样本={count}")
        self._refresh_training_samples()

    def _refresh_training_samples(self) -> None:
        if not hasattr(self, "training_samples_list"):
            return
        self.training_samples_list.delete(0, tk.END)
        label = self._selected_training_label()
        for path in sorted(self._template_dir(label).glob("template_*.png")):
            self.training_samples_list.insert(tk.END, path.name)

    def open_template_directory(self) -> None:
        directory = self._template_dir(self._selected_training_label())
        directory.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(directory))
        except AttributeError:
            subprocess.Popen(["explorer", str(directory)])
        self.training_info_var.set(f"已打开模板目录：{directory}")

    def delete_selected_sample(self) -> None:
        if not hasattr(self, "training_samples_list"):
            return
        selection = self.training_samples_list.curselection()
        if not selection:
            self.training_info_var.set("请先选择要删除的样本")
            return
        name = str(self.training_samples_list.get(selection[0]))
        path = self._template_dir(self._selected_training_label()) / name
        if path.is_file():
            path.unlink()
            self.training_selected_detection = None
            self._refresh_training_samples()
            self.training_info_var.set(f"已删除样本：{name}")

    def add_template_category(self) -> None:
        raw = self.training_new_category_var.get().strip()
        label = safe_template_category(raw)
        if not raw:
            self.training_info_var.set("请输入新模板类别")
            return
        path = self._template_dir(label)
        path.mkdir(parents=True, exist_ok=True)
        self.training_new_category_var.set("")
        categories = self._template_categories()
        self.training_category_combo.configure(
            values=[display for display, _ in categories]
        )
        for display, category in categories:
            if category == label:
                self.training_label_var.set(display)
                break
        self.training_category_list.delete(0, tk.END)
        for _, category in categories:
            self.training_category_list.insert(tk.END, category)
        self._refresh_attack_monster_menu()
        self._training_status()

    def capture_training_frame(self) -> None:
        if self.source is None:
            self.training_info_var.set("请先绑定窗口")
            return
        try:
            frame = self.source.next_frame()
            if frame is None:
                raise RuntimeError("窗口截图为空")
            self.training_frame = frame
            self.training_detections = ()
            self.training_selected_detection = None
            self.training_test_count = 0
            self.training_test_streak = 0
            self._show_training_frame(frame)
            self.training_select_button.configure(state="normal")
            self.training_test_button.configure(state="normal")
            self._training_status()
            self.training_info_var.set(
                f"训练画面={frame.shape[1]}x{frame.shape[0]}，"
                "点击“框选怪物”后可连续框选"
            )
        except Exception as exc:
            logger.exception("unable to capture Windows training frame")
            self.training_info_var.set(f"获取训练画面失败：{exc}")

    def toggle_training_selection(self) -> None:
        if self.training_frame is None:
            self.training_info_var.set("请先获取训练画面")
            return
        self.training_selection_mode = not self.training_selection_mode
        self.training_select_button.configure(
            text="停止框选" if self.training_selection_mode else "框选怪物"
        )
        self.training_info_var.set(
            "连续框选模式已开启，请框选完整怪物"
            if self.training_selection_mode
            else "已停止框选"
        )

    def _frame_point(
        self,
        canvas: tk.Canvas,
        event,
        frame: np.ndarray | None,
        scale: float,
    ) -> tuple[float, float] | None:
        if frame is None or scale <= 0:
            return None
        x = canvas.canvasx(event.x) / scale
        y = canvas.canvasy(event.y) / scale
        height, width = frame.shape[:2]
        if not (0 <= x <= width and 0 <= y <= height):
            return None
        return x, y

    def _training_selection_start(self, event) -> None:
        if not self.training_selection_mode:
            return
        point = self._frame_point(
            self.training_canvas,
            event,
            self.training_frame,
            self.training_display_scale,
        )
        if point is None:
            return
        self.training_selection_start = point
        if self.training_rect_id is not None:
            self.training_canvas.delete(self.training_rect_id)
        x, y = point
        self.training_rect_id = self.training_canvas.create_rectangle(
            x * self.training_display_scale,
            y * self.training_display_scale,
            x * self.training_display_scale,
            y * self.training_display_scale,
            outline="#30d158",
            width=2,
        )

    def _training_selection_drag(self, event) -> None:
        if self.training_selection_start is None or self.training_rect_id is None:
            return
        point = self._frame_point(
            self.training_canvas,
            event,
            self.training_frame,
            self.training_display_scale,
        )
        if point is None:
            return
        x0, y0 = self.training_selection_start
        x1, y1 = point
        self.training_canvas.coords(
            self.training_rect_id,
            min(x0, x1) * self.training_display_scale,
            min(y0, y1) * self.training_display_scale,
            max(x0, x1) * self.training_display_scale,
            max(y0, y1) * self.training_display_scale,
        )

    def _training_selection_end(self, event) -> None:
        if self.training_selection_start is None or self.training_frame is None:
            return
        point = self._frame_point(
            self.training_canvas,
            event,
            self.training_frame,
            self.training_display_scale,
        )
        if point is None:
            return
        x0, y0 = self.training_selection_start
        x1, y1 = point
        left, right = sorted((int(x0), int(x1)))
        top, bottom = sorted((int(y0), int(y1)))
        if right - left < 12 or bottom - top < 12:
            self.training_info_var.set("框选区域太小，请重新选择")
            self.training_selection_start = None
            return
        label = self._selected_training_label()
        directory = self._template_dir(label)
        directory.mkdir(parents=True, exist_ok=True)
        used = {
            int(match.group(1))
            for path in directory.glob("template_*.png")
            if (match := re.fullmatch(r"template_(\d+)", path.stem))
        }
        index = 1
        while index in used:
            index += 1
        path = directory / f"template_{index:02d}.png"
        crop = self.training_frame[top:bottom, left:right].copy()
        if not write_image(path, crop):
            self.training_info_var.set("模板保存失败")
            return
        self.last_template_path = path
        self.training_selection_start = None
        self._refresh_training_samples()
        self._refresh_attack_monster_menu()
        self.training_info_var.set(
            f"已保存{label}模板：{path.name}，"
            f"区域=({left},{top},{right-left},{bottom-top})"
        )

    def _show_training_frame(self, frame: np.ndarray) -> None:
        self._show_canvas_image(
            self.training_canvas,
            frame,
            "_training_photo",
            "training_display_scale",
        )

    def _show_canvas_image(
        self,
        canvas: tk.Canvas,
        frame: np.ndarray,
        photo_attribute: str,
        scale_attribute: str,
    ) -> None:
        height, width = frame.shape[:2]
        scale = min(1.0, 900 / max(1, width), 560 / max(1, height))
        display = frame
        if scale != 1.0:
            display = cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        ok, encoded = cv2.imencode(".ppm", rgb)
        if not ok:
            raise RuntimeError("无法编码预览画面")
        photo = tk.PhotoImage(
            data=base64.b64encode(encoded.tobytes()).decode("ascii"),
            format="PPM",
        )
        setattr(self, photo_attribute, photo)
        setattr(self, scale_attribute, scale)
        canvas.delete("all")
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.configure(scrollregion=(0, 0, display.shape[1], display.shape[0]))

    def _select_training_categories(self, select_all: bool) -> None:
        self.training_category_list.selection_clear(0, tk.END)
        if select_all:
            self.training_category_list.selection_set(
                0, tk.END
            )

    def _training_allowed_labels(self) -> list[str]:
        selected = [
            str(self.training_category_list.get(index))
            for index in self.training_category_list.curselection()
        ]
        if selected:
            return selected
        return [
            str(self.training_category_list.get(index))
            for index in range(self.training_category_list.size())
        ]

    def _training_detector_config(self) -> dict[str, Any]:
        monster_config = deepcopy(
            self.config.get("combat", {}).get("monsters", {})
        )
        groups: dict[str, list[str]] = {}
        for _, label in self._template_categories():
            directory = self._template_dir(label)
            if directory.is_dir() and list(directory.glob("template_*.png")):
                groups[label] = [str(directory)]
        monster_config["template_groups"] = groups
        monster_config["allowed_labels"] = self._training_allowed_labels()
        if self.screen_region is not None:
            monster_config["region"] = list(self.screen_region)
        return monster_config

    def test_monsters(self) -> None:
        if self.training_frame is None:
            self.training_info_var.set("请先获取训练画面")
            return
        try:
            frame = self.training_frame
            if self.source is not None:
                captured = self.source.next_frame()
                if captured is not None:
                    frame = captured
                    self.training_frame = frame
            detections = detect_monsters(frame, self._training_detector_config())
            self.training_detections = detections
            self.training_test_count += 1
            self.training_test_streak = (
                self.training_test_streak + 1 if detections else 0
            )
            annotated = render_monster_debug_frame(frame, detections)
            self._show_training_frame(annotated)
            counts: dict[str, int] = {}
            for detection in detections:
                counts[detection.label] = counts.get(detection.label, 0) + 1
            result = ", ".join(
                f"{label}={count}" for label, count in counts.items()
            ) or "未识别到怪物"
            self.training_result_var.set(
                f"测试结果：{result}；连续成功 "
                f"{self.training_test_streak}/3 次"
            )
            self.training_info_var.set(
                "右键点击识别框可选中模板，再点击“删除选中模板”"
            )
        except Exception as exc:
            logger.exception("unable to test Windows monster templates")
            self.training_result_var.set(f"测试失败：{exc}")

    def _training_detection_click(self, event) -> None:
        point = self._frame_point(
            self.training_canvas,
            event,
            self.training_frame,
            self.training_display_scale,
        )
        if point is None:
            return
        px, py = point
        selected = next(
            (
                detection
                for detection in reversed(self.training_detections)
                if detection.box[0] <= px <= detection.box[0] + detection.box[2]
                and detection.box[1] <= py <= detection.box[1] + detection.box[3]
            ),
            None,
        )
        self.training_selected_detection = selected
        if selected is None:
            self.training_result_var.set("未选中识别框")
            return
        self.training_result_var.set(
            f"已选中：{selected.label}，模板={Path(selected.template_path).name}"
        )

    def undo_template(self) -> None:
        if self.last_template_path is None or not self.last_template_path.exists():
            self.training_info_var.set("没有可撤销的最近模板")
            return
        path = self.last_template_path
        path.unlink()
        self.last_template_path = None
        self.training_info_var.set(f"已撤销模板：{path.name}")

    def delete_selected_template(self) -> None:
        detection = self.training_selected_detection
        if detection is None or not detection.template_path:
            self.training_result_var.set("请先右键点击一个识别框")
            return
        path = Path(detection.template_path)
        if path.exists():
            path.unlink()
        self.training_selected_detection = None
        self.training_result_var.set(f"已删除模板：{path.name}")
        self.test_monsters()

    def _route_point_names(self) -> list[str]:
        group = self._selected_route_group()
        names = [
            str(item.get("name", "")).strip()
            for item in self.route_points
            if str(item.get("name", "")).strip()
            and route_point_group(item) == group
        ]
        return names

    def _selected_route_group(self) -> str:
        value = self.route_group_var.get().strip()
        if value:
            self.route_active_group = value
        return self.route_active_group or "默认"

    def _route_group_names(self) -> list[str]:
        return route_group_names(
            {
                "active_group": self.route_active_group,
                "points": self.route_points,
                "paths": self.route_paths,
            }
        )

    def route_group_changed(self) -> None:
        self.route_active_group = self.route_group_var.get().strip() or "默认"
        self.route_current_path_key = None
        self.route_current_steps = []
        self.route_path_var.set("")
        self.route_show_annotations = False
        self.route_annotations_button.configure(text="显示已保存标注")
        self._refresh_route_controls()
        if self.route_frame is not None:
            self._show_route_frame(self.route_frame)
            self._show_route_minimap(self.route_frame)
        self.status_var.set(f"已切换路线组：{self.route_active_group}")

    def _selected_route_profile_name(self) -> str | None:
        value = self.route_profile_point_var.get().strip()
        return value or None

    def _route_profile(self, name: str | None) -> dict[str, Any]:
        if not name:
            return {}
        return self.route_combat_profiles.setdefault(name, {})

    def _route_ladder_names(self) -> list[str]:
        return [
            str(item.get("name", "")).strip()
            for item in self.route_ladders
            if str(item.get("name", "")).strip()
        ]

    def _route_path_names(self) -> list[str]:
        group = self._selected_route_group()
        return [
            str(item.get("name", "")).strip()
            for item in self.route_paths
            if str(item.get("name", "")).strip()
            and route_path_group(item) == group
        ]

    def _selected_route_annotation_context(
        self,
    ) -> tuple[set[str], set[str]]:
        """Return the point and ladder names used by the selected path."""
        selected_name = str(
            self.route_current_path_key
            or self.route_path_var.get()
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
            source = self.route_source_var.get().strip()
            target = self.route_target_var.get().strip()
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

    def _refresh_route_controls(self) -> None:
        point_names = self._route_point_names()
        path_names = self._route_path_names()
        groups = self._route_group_names()
        if self.route_active_group not in groups:
            self.route_active_group = groups[0]
        self.route_group_combo.configure(values=groups)
        self.route_group_var.set(self.route_active_group)
        self.route_profile_point_combo.configure(values=point_names)
        if self.route_combat_point_name in point_names:
            self.route_profile_point_var.set(self.route_combat_point_name)
        elif point_names:
            self.route_profile_point_var.set(point_names[0])
            self.route_combat_point_name = point_names[0]
        else:
            self.route_profile_point_var.set("")
            self.route_combat_point_name = None
        self.route_source_combo.configure(values=point_names)
        self.route_target_combo.configure(values=point_names)
        self.route_path_combo.configure(values=path_names)
        self._update_route_objects()
        self._update_route_status()
        self._load_route_profile(self.route_combat_point_name)

    def _update_route_objects(self) -> None:
        step_type = self.route_step_type_var.get()
        if step_type == "move_to_point":
            values = self._route_point_names()
        elif step_type in {"move_to_ladder", "climb"}:
            values = self._route_ladder_names()
        elif step_type in {"move_left", "move_right"}:
            values = ["目标挂机点固定标记"]
        else:
            values = []
        self.route_step_object_combo.configure(values=values)
        if values:
            self.route_step_object_var.set(values[0])
        else:
            self.route_step_object_var.set("")
        self.route_step_object_combo.configure(
            state=(
                "disabled"
                if step_type in {"move_left", "move_right"}
                else "normal"
            )
        )
        self.route_step_duration_var.set(
            self.route_step_duration_var.get()
            if step_type not in {"move_left", "move_right"}
            else ""
        )
        self.route_step_duration_entry.configure(
            state=(
                "disabled"
                if step_type in {"move_left", "move_right"}
                else "normal"
            )
        )

    def capture_route_frame(self) -> None:
        if self.source is None:
            self.status_var.set("请先绑定窗口")
            return
        try:
            frame = self.source.next_frame()
            if frame is None:
                raise RuntimeError("窗口截图为空")
            self.route_frame = frame
            self.route_selection_mode = None
            self.route_selection_start = None
            self._clear_route_magnifier()
            # 新截图默认使用干净视图，旧配置只在用户主动点击时显示。
            self.route_show_annotations = False
            self._show_route_frame(frame)
            self._show_route_minimap(frame)
            self.route_annotations_button.configure(
                state="normal",
                text="显示已保存标注",
            )
            self.route_minimap_calibrate_button.configure(state="normal")
            self.route_minimap_calibrate_button.configure(text="校准小地图")
            self.route_mark_button.configure(state="normal")
            self.route_ladder_button.configure(state="normal")
            self.route_test_start_button.configure(
                state="normal" if self.route_paths else "disabled"
            )
            self.route_profile_fixed_button.configure(state="normal")
            self.route_profile_region_button.configure(state="normal")
            self.status_var.set("路径页已获取画面，可在放大小地图内框选挂机区域")
        except Exception as exc:
            logger.exception("unable to capture Windows route frame")
            self.status_var.set(f"路径画面获取失败：{exc}")

    def toggle_route_annotations(self) -> None:
        if self.route_frame is None:
            self.status_var.set("请先获取路径画面")
            return
        self.route_show_annotations = not self.route_show_annotations
        self.route_annotations_button.configure(
            text=(
                "隐藏已保存标注"
                if self.route_show_annotations
                else "显示已保存标注"
            )
        )
        self._show_route_frame(self.route_frame)
        self._show_route_minimap(self.route_frame)
        self.status_var.set(
            "已显示已保存的挂机点、梯子和固定配置"
            if self.route_show_annotations
            else "已清除当前画面上的旧标注，配置未删除"
        )

    def _show_route_frame(self, frame: np.ndarray) -> None:
        self._show_canvas_image(
            self.route_canvas,
            frame,
            "_route_photo",
            "route_display_scale",
        )
        point_names, _ = self._selected_route_annotation_context()
        minimap = tuple(self.config.get("minimap", {}).get("region") or ())
        if len(minimap) == 4:
            x, y, width, height = minimap
            self.route_canvas.create_rectangle(
                x * self.route_display_scale,
                y * self.route_display_scale,
                (x + width) * self.route_display_scale,
                (y + height) * self.route_display_scale,
                outline="#ffd60a",
                width=2,
            )
            if not self.route_show_annotations:
                return
            for point in self.route_points:
                point_name = str(point.get("name", "")).strip()
                if point_name not in point_names:
                    continue
                region = point.get("region")
                if not isinstance(region, (list, tuple)) or len(region) != 4:
                    continue
                px, py, pw, ph = (int(value) for value in region)
                self.route_canvas.create_rectangle(
                    (x + px) * self.route_display_scale,
                    (y + py) * self.route_display_scale,
                    (x + px + pw) * self.route_display_scale,
                    (y + py + ph) * self.route_display_scale,
                    outline="#bf5af2",
                    width=2,
                )
            for profile_name in point_names:
                profile = self._route_profile(profile_name)
                fixed = profile.get("fixed_point")
                if isinstance(fixed, (list, tuple)) and len(fixed) == 2:
                    self.route_canvas.create_oval(
                        (x + int(fixed[0]) - 3) * self.route_display_scale,
                        (y + int(fixed[1]) - 3) * self.route_display_scale,
                        (x + int(fixed[0]) + 3) * self.route_display_scale,
                        (y + int(fixed[1]) + 3) * self.route_display_scale,
                        outline="#bf5af2",
                        fill="#bf5af2",
                    )
        if not self.route_show_annotations:
            return
        for profile_name in point_names:
            profile = self._route_profile(profile_name)
            monster_region = profile.get("monster_region")
            if isinstance(monster_region, (list, tuple)) and len(monster_region) == 4:
                rx, ry, rw, rh = (int(value) for value in monster_region)
                self.route_canvas.create_rectangle(
                    rx * self.route_display_scale,
                    ry * self.route_display_scale,
                    (rx + rw) * self.route_display_scale,
                    (ry + rh) * self.route_display_scale,
                    outline="#ff3b30",
                    width=2,
                )

    def _show_route_minimap(self, frame: np.ndarray) -> None:
        minimap = tuple(self.config.get("minimap", {}).get("region") or ())
        if len(minimap) != 4:
            self.route_minimap_canvas.delete("all")
            self.route_minimap_info_var.set("小地图未配置")
            return
        x, y, width, height = (int(value) for value in minimap)
        actual = clip_region_to_frame(frame, minimap)
        cropped = crop_region(frame, minimap)
        if cropped.size == 0:
            self.route_minimap_canvas.delete("all")
            self.route_minimap_info_var.set("小地图截取为空，请重新校准")
            return
        self.route_minimap_info_var.set(
            f"实际 {actual[2]}x{actual[3]}；配置 "
            f"x={x} y={y} w={width} h={height}"
        )
        scale = self.route_minimap_scale
        enlarged = cv2.resize(
            cropped,
            (cropped.shape[1] * scale, cropped.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
        self._show_canvas_image(
            self.route_minimap_canvas,
            enlarged,
            "route_minimap_photo",
            "route_minimap_display_scale",
        )
        if not self.route_show_annotations:
            return
        point_names, ladder_names = self._selected_route_annotation_context()
        for point in self.route_points:
            point_name = str(point.get("name", "")).strip()
            if point_name not in point_names:
                continue
            region = point.get("region")
            if not isinstance(region, (list, tuple)) or len(region) != 4:
                continue
            px, py, pw, ph = (int(value) for value in region)
            self.route_minimap_canvas.create_rectangle(
                px * scale * self.route_minimap_display_scale,
                py * scale * self.route_minimap_display_scale,
                (px + pw) * scale * self.route_minimap_display_scale,
                (py + ph) * scale * self.route_minimap_display_scale,
                outline="#bf5af2",
                width=2,
            )
        for ladder in self.route_ladders:
            ladder_name = str(ladder.get("name", "")).strip()
            if ladder_name not in ladder_names:
                continue
            point = ladder.get("point")
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            px, py = (int(value) for value in point)
            cx = px * scale * self.route_minimap_display_scale
            cy = py * scale * self.route_minimap_display_scale
            radius = 4
            self.route_minimap_canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                fill="#bf5af2",
                outline="#bf5af2",
            )
        for profile_name in point_names:
            fixed = self._route_profile(profile_name).get("fixed_point")
            if not isinstance(fixed, (list, tuple)) or len(fixed) != 2:
                continue
            cx = int(fixed[0]) * scale * self.route_minimap_display_scale
            cy = int(fixed[1]) * scale * self.route_minimap_display_scale
            radius = 3
            self.route_minimap_canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                fill="#30d158",
                outline="#30d158",
            )

    def _route_minimap_point(self, event) -> tuple[float, float] | None:
        if self.route_frame is None or self.route_minimap_display_scale <= 0:
            return None
        x = self.route_minimap_canvas.canvasx(event.x)
        y = self.route_minimap_canvas.canvasy(event.y)
        return (
            x / self.route_minimap_display_scale,
            y / self.route_minimap_display_scale,
        )

    def _route_minimap_selection_start(self, event) -> None:
        if self.route_selection_mode != "point_region":
            return
        point = self._route_minimap_point(event)
        if point is None:
            return
        self.route_selection_start = point
        if self.route_rect_id is not None:
            self.route_minimap_canvas.delete(self.route_rect_id)
        self.route_rect_id = self.route_minimap_canvas.create_rectangle(
            point[0] * self.route_minimap_display_scale,
            point[1] * self.route_minimap_display_scale,
            point[0] * self.route_minimap_display_scale,
            point[1] * self.route_minimap_display_scale,
            outline="#ff3b30",
            width=2,
        )

    def _route_minimap_selection_drag(self, event) -> None:
        if self.route_selection_start is None or self.route_rect_id is None:
            return
        point = self._route_minimap_point(event)
        if point is None:
            return
        x0, y0 = self.route_selection_start
        self.route_minimap_canvas.coords(
            self.route_rect_id,
            min(x0, point[0]) * self.route_minimap_display_scale,
            min(y0, point[1]) * self.route_minimap_display_scale,
            max(x0, point[0]) * self.route_minimap_display_scale,
            max(y0, point[1]) * self.route_minimap_display_scale,
        )

    def _route_minimap_selection_end(self, event) -> None:
        if self.route_selection_start is None:
            return
        point = self._route_minimap_point(event)
        if point is None:
            return
        x0, y0 = self.route_selection_start
        scaled_rect = (
            min(x0, point[0]),
            min(y0, point[1]),
            abs(point[0] - x0),
            abs(point[1] - y0),
        )
        local = minimap_scaled_rect_to_local(
            tuple(int(value) for value in scaled_rect),
            self.route_minimap_scale,
        )
        self.route_selection_start = None
        if local is None or local[2] < 3 or local[3] < 3:
            if self.route_rect_id is not None:
                self.route_minimap_canvas.delete(self.route_rect_id)
                self.route_rect_id = None
            self.status_var.set("挂机区域太小，请重新框选")
            return
        name = self.route_point_name_var.get().strip() or (
            f"点{len(self.route_points) + 1}"
        )
        self.route_points = [
            item
            for item in self.route_points
            if str(item.get("name", "")).strip() != name
        ]
        self.route_points.append(
            {
                "name": name,
                "region": list(local),
                "group": self._selected_route_group(),
            }
        )
        self.route_combat_profiles.setdefault(name, {})
        if not self.route_combat_point_name:
            self.route_combat_point_name = name
        self.route_selection_mode = None
        self.route_mark_button.configure(text="框选挂机区域")
        self.route_rect_id = None
        self._refresh_route_controls()
        self._show_route_minimap(self.route_frame)
        self.status_var.set(
            f"已覆盖挂机点 {name}：小地图 x={local[0]} y={local[1]} "
            f"w={local[2]} h={local[3]}，请点击保存当前点或保存配置"
        )

    def toggle_route_selection(self) -> None:
        if self.route_frame is None:
            self.status_var.set("请先获取路径画面")
            return
        self.route_selection_mode = (
            None
            if self.route_selection_mode == "point_region"
            else "point_region"
        )
        self.route_mark_button.configure(
            text="停止框选" if self.route_selection_mode else "框选挂机区域"
        )
        self.status_var.set(
            "请在小地图内框选挂机区域"
            if self.route_selection_mode
            else "已停止框选"
        )

    def route_profile_changed(self) -> None:
        self._load_route_profile(self._selected_route_profile_name())
        self.status_var.set(
            f"已切换挂机点：{self.route_combat_point_name or '未设置'}"
        )

    def _load_route_profile(self, name: str | None) -> None:
        self.route_combat_point_name = name
        profile = self._route_profile(name)
        fixed = profile.get("fixed_point")
        region = profile.get("monster_region")
        self.route_profile_fixed_var.set(
            bool(profile.get("fixed_point_enabled", fixed is not None))
        )
        fixed_text = (
            f"固定点=({int(fixed[0])},{int(fixed[1])})"
            if isinstance(fixed, (list, tuple)) and len(fixed) == 2
            else "固定点=未设置"
        )
        region_text = (
            "寻怪区域=已设置"
            if isinstance(region, (list, tuple)) and len(region) == 4
            else "寻怪区域=未设置"
        )
        self.route_profile_status_var.set(f"{fixed_text}；{region_text}")
        if self.route_frame is not None:
            self._show_route_frame(self.route_frame)

    def save_current_route_profile(self) -> None:
        name = self._selected_route_profile_name()
        if not name:
            self.route_profile_status_var.set("请先创建并选择挂机点")
            return
        profile = self._route_profile(name)
        profile["fixed_point_enabled"] = bool(self.route_profile_fixed_var.get())
        self._load_route_profile(name)
        try:
            self._write_config()
            self.status_var.set(f"已保存挂机点配置：{name}")
        except Exception as exc:
            logger.exception("unable to save route combat profile")
            self.status_var.set(f"保存挂机点配置失败：{exc}")

    def toggle_route_profile_fixed(self) -> None:
        if self.route_frame is None:
            self.status_var.set("请先获取路径画面")
            return
        enabled = self.route_selection_mode != "profile_fixed"
        self.route_selection_mode = "profile_fixed" if enabled else None
        self.route_profile_fixed_button.configure(
            text="点击小地图中" if enabled else "标记固定点"
        )
        self.status_var.set(
            "请在小地图上点击当前挂机点的固定校准位置"
            if enabled
            else "已停止标记固定点"
        )

    def toggle_route_profile_region(self) -> None:
        if self.route_frame is None:
            self.status_var.set("请先获取路径画面")
            return
        enabled = self.route_selection_mode != "profile_region"
        self.route_selection_mode = "profile_region" if enabled else None
        self.route_profile_region_button.configure(
            text="停止框选" if enabled else "框选寻怪区域"
        )
        self.status_var.set(
            "请在游戏画面中框选当前挂机点的寻怪区域"
            if enabled
            else "已停止框选寻怪区域"
        )

    def _route_selection_start(self, event) -> None:
        if not self.route_selection_mode:
            return
        point = self._frame_point(
            self.route_canvas,
            event,
            self.route_frame,
            self.route_display_scale,
        )
        if point is None:
            return
        if self.route_selection_mode == "profile_fixed":
            self._set_route_profile_fixed_point(point)
            return
        self.route_selection_start = point
        self.route_rect_id = self.route_canvas.create_rectangle(
            point[0] * self.route_display_scale,
            point[1] * self.route_display_scale,
            point[0] * self.route_display_scale,
            point[1] * self.route_display_scale,
            outline="#bf5af2",
            width=2,
        )

    def _route_selection_drag(self, event) -> None:
        if self.route_selection_start is None or self.route_rect_id is None:
            return
        point = self._frame_point(
            self.route_canvas,
            event,
            self.route_frame,
            self.route_display_scale,
        )
        if point is None:
            return
        x0, y0 = self.route_selection_start
        x1, y1 = point
        self.route_canvas.coords(
            self.route_rect_id,
            min(x0, x1) * self.route_display_scale,
            min(y0, y1) * self.route_display_scale,
            max(x0, x1) * self.route_display_scale,
            max(y0, y1) * self.route_display_scale,
        )

    def _route_selection_end(self, event) -> None:
        if self.route_selection_start is None or self.route_frame is None:
            return
        point = self._frame_point(
            self.route_canvas,
            event,
            self.route_frame,
            self.route_display_scale,
        )
        if point is None:
            return
        x0, y0 = self.route_selection_start
        x1, y1 = point
        left, right = sorted((int(x0), int(x1)))
        top, bottom = sorted((int(y0), int(y1)))
        if self.route_selection_mode == "minimap_region":
            frame_height, frame_width = self.route_frame.shape[:2]
            left = max(0, min(frame_width, left))
            top = max(0, min(frame_height, top))
            right = max(left, min(frame_width, right))
            bottom = max(top, min(frame_height, bottom))
            if right - left < 8 or bottom - top < 8:
                self.status_var.set("小地图区域太小，请重新框选")
                self.route_selection_start = None
                return
            self.config.setdefault("minimap", {})["region"] = [
                left,
                top,
                right - left,
                bottom - top,
            ]
            self.route_selection_mode = None
            self.route_selection_start = None
            self.route_minimap_calibrate_button.configure(text="校准小地图")
            self._clear_route_magnifier()
            self._show_route_frame(self.route_frame)
            self._show_route_minimap(self.route_frame)
            self.status_var.set(
                f"小地图区域已更新：x={left} y={top} "
                f"w={right-left} h={bottom-top}；请点击保存配置"
            )
            return
        if self.route_selection_mode == "profile_region":
            frame_height, frame_width = self.route_frame.shape[:2]
            left = max(0, min(frame_width, left))
            top = max(0, min(frame_height, top))
            right = max(left, min(frame_width, right))
            bottom = max(top, min(frame_height, bottom))
            if right - left < 4 or bottom - top < 4:
                self.status_var.set("寻怪区域太小，请重新框选")
                self.route_selection_start = None
                return
            name = self._selected_route_profile_name()
            if not name:
                self.status_var.set("请先选择挂机点")
                self.route_selection_start = None
                return
            self._route_profile(name)["monster_region"] = [
                left,
                top,
                right - left,
                bottom - top,
            ]
            self.route_selection_start = None
            self.route_selection_mode = None
            self.route_profile_region_button.configure(text="框选寻怪区域")
            self._load_route_profile(name)
            self.status_var.set(f"已设置挂机点 {name} 的寻怪区域")
            return
        local = minimap_local_rect(
            (left, top, right - left, bottom - top),
            tuple(self.config.get("minimap", {}).get("region") or ()),
        )
        if local is None or local[2] < 3 or local[3] < 3:
            self.status_var.set("挂机区域必须完全落在小地图内")
            return
        name = self.route_point_name_var.get().strip() or (
            f"点{len(self.route_points) + 1}"
        )
        self.route_points = [
            item
            for item in self.route_points
            if str(item.get("name", "")).strip() != name
        ]
        self.route_points.append(
            {
                "name": name,
                "region": list(local),
                "group": self._selected_route_group(),
            }
        )
        self.route_selection_mode = False
        self.route_selection_start = None
        self.route_mark_button.configure(text="框选挂机区域")
        self._refresh_route_controls()
        self._show_route_frame(self.route_frame)
        self.status_var.set(
            f"已保存挂机点 {name}：x={local[0]} y={local[1]} "
            f"w={local[2]} h={local[3]}"
        )

    def toggle_route_minimap_calibration(self) -> None:
        if self.route_frame is None:
            self.status_var.set("请先获取路径画面")
            return
        enabled = self.route_selection_mode != "minimap_region"
        self.route_selection_mode = "minimap_region" if enabled else None
        self.route_selection_start = None
        if not enabled:
            self._clear_route_magnifier()
        self.route_minimap_calibrate_button.configure(
            text="停止校准" if enabled else "校准小地图"
        )
        self.status_var.set(
            "请在左侧完整游戏画面中框选完整小地图"
            if enabled
            else "已停止校准小地图"
        )

    def _clear_route_magnifier(self) -> None:
        self.route_canvas.delete("route_magnifier")
        self.route_magnifier_id = None

    def _route_magnifier_motion(self, event) -> None:
        if (
            self.route_selection_mode != "minimap_region"
            or self.route_frame is None
        ):
            self._clear_route_magnifier()
            return
        point = self._frame_point(
            self.route_canvas,
            event,
            self.route_frame,
            self.route_display_scale,
        )
        if point is None:
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
        rgb = cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB)
        ok, encoded = cv2.imencode(".ppm", rgb)
        if not ok:
            return
        photo = tk.PhotoImage(
            data=base64.b64encode(encoded.tobytes()).decode("ascii"),
            format="PPM",
        )
        self._route_magnifier_photo = photo
        self._clear_route_magnifier()
        canvas_x = self.route_canvas.canvasx(event.x)
        canvas_y = self.route_canvas.canvasy(event.y)
        image_width = enlarged.shape[1]
        image_height = enlarged.shape[0]
        canvas_width = max(image_width + 4, self.route_canvas.winfo_width())
        canvas_height = max(image_height + 4, self.route_canvas.winfo_height())
        draw_x = min(canvas_width - image_width - 4, canvas_x + 16)
        draw_y = min(canvas_height - image_height - 4, canvas_y + 16)
        draw_x = max(2, draw_x)
        draw_y = max(2, draw_y)
        self.route_magnifier_id = self.route_canvas.create_image(
            draw_x,
            draw_y,
            image=photo,
            anchor="nw",
            tags=("route_magnifier",),
        )
        self.route_canvas.create_rectangle(
            draw_x - 1,
            draw_y - 1,
            draw_x + image_width + 1,
            draw_y + image_height + 1,
            outline="#bf5af2",
            width=2,
            tags=("route_magnifier",),
        )
        cursor_x = draw_x + (center_x - left) * 5
        cursor_y = draw_y + (center_y - top) * 5
        self.route_canvas.create_line(
            cursor_x - 10,
            cursor_y,
            cursor_x + 10,
            cursor_y,
            fill="#bf5af2",
            width=2,
            tags=("route_magnifier",),
        )
        self.route_canvas.create_line(
            cursor_x,
            cursor_y - 10,
            cursor_x,
            cursor_y + 10,
            fill="#bf5af2",
            width=2,
            tags=("route_magnifier",),
        )

    def _set_route_profile_fixed_point(
        self,
        point: tuple[float, float],
    ) -> None:
        minimap = tuple(self.config.get("minimap", {}).get("region") or ())
        local = minimap_local_point(
            (int(point[0]), int(point[1])),
            minimap,
        )
        name = self._selected_route_profile_name()
        if local is None:
            self.status_var.set("固定点必须点击在小地图区域内")
            return
        if not name:
            self.status_var.set("请先选择挂机点")
            return
        profile = self._route_profile(name)
        profile["fixed_point"] = [int(local[0]), int(local[1])]
        profile["fixed_point_enabled"] = True
        self.route_selection_mode = None
        self.route_profile_fixed_button.configure(text="标记固定点")
        self.route_profile_fixed_var.set(True)
        self._load_route_profile(name)
        self.status_var.set(
            f"已设置挂机点 {name} 固定点：({local[0]},{local[1]})"
        )

    def record_ladder(self) -> None:
        if self.route_frame is None:
            self.status_var.set("请先获取路径画面")
            return
        minimap = tuple(self.config.get("minimap", {}).get("region") or ())
        hsv = self.config.get("minimap", {}).get("player_color_hsv", {})
        try:
            observation = observe_minimap(
                self.route_frame,
                minimap,
                tuple(hsv["lower"]),
                tuple(hsv["upper"]),
            )
            if observation.player_local is None:
                raise ValueError("当前画面没有识别到小地图黄点")
            name = self.route_ladder_name_var.get().strip() or (
                f"梯子{len(self.route_ladders) + 1}"
            )
            duration = int(float(self.route_ladder_duration_var.get()))
            if duration <= 0:
                raise ValueError("时长必须大于0")
            item = {
                "name": name,
                "point": [
                    int(observation.player_local.x),
                    int(observation.player_local.y),
                ],
                "climb_duration_ms": duration,
                "group": self._selected_route_group(),
            }
            self.route_ladders = [
                old
                for old in self.route_ladders
                if str(old.get("name", "")).strip() != name
            ]
            self.route_ladders.append(item)
            self._refresh_route_controls()
            self.status_var.set(
                f"已记录点位 {name}：小地图=({item['point'][0]},"
                f"{item['point'][1]})"
            )
        except (KeyError, TypeError, ValueError) as exc:
            self.status_var.set(f"记录点位失败：{exc}")

    def new_route_path(self) -> None:
        names = self._route_point_names()
        source = self.route_source_var.get().strip()
        target = self.route_target_var.get().strip()
        if len(names) < 2:
            self.route_status_var.set("路径步骤：请先创建至少两个挂机点")
            return
        if source not in names:
            source = names[0]
        if target not in names or target == source:
            target = next(name for name in names if name != source)
        self.route_source_var.set(source)
        self.route_target_var.set(target)
        self.route_current_path_key = path_name(source, target)
        self.route_current_steps = []
        self.route_show_annotations = False
        if hasattr(self, "route_annotations_button"):
            self.route_annotations_button.configure(text="显示已保存标注")
        if self.route_frame is not None:
            self._show_route_frame(self.route_frame)
            self._show_route_minimap(self.route_frame)
        self.route_path_var.set(self.route_current_path_key)
        self._update_route_status()
        self.status_var.set(f"已新建路径 {self.route_current_path_key}")

    def route_path_changed(self) -> None:
        selected = self.route_path_var.get().strip()
        route = next(
            (
                item for item in self.route_paths
                if str(item.get("name", "")).strip() == selected
            ),
            None,
        )
        if route is None:
            return
        self.route_current_path_key = selected
        self.route_source_var.set(str(route.get("source", "")))
        self.route_target_var.set(str(route.get("target", "")))
        self.route_combat_point_name = self.route_source_var.get().strip() or None
        self.route_show_annotations = True
        self.route_annotations_button.configure(
            text="隐藏已保存标注",
            state="normal",
        )
        self.route_current_steps = [
            dict(step)
            for step in (route.get("steps") or [])
            if isinstance(step, dict)
        ]
        if self.route_frame is not None:
            self._show_route_frame(self.route_frame)
            self._show_route_minimap(self.route_frame)
        self._update_route_status()

    def delete_route_path(self) -> None:
        selected = self.route_path_var.get().strip()
        if not selected:
            self.status_var.set("请先选择要删除的路径")
            return
        remaining = [
            item
            for item in self.route_paths
            if str(item.get("name", "")).strip() != selected
        ]
        if len(remaining) == len(self.route_paths):
            self.status_var.set(f"未找到路径：{selected}")
            return
        self.route_paths = remaining
        self.route_current_path_key = None
        self.route_current_steps = []
        self.route_show_annotations = False
        try:
            self._write_config()
        except (OSError, ValueError) as exc:
            self.status_var.set(f"路径删除失败：{exc}")
            return
        self.route_path_var.set("")
        self._refresh_route_controls()
        self.route_annotations_button.configure(text="显示已保存标注")
        if self.route_frame is not None:
            self._show_route_frame(self.route_frame)
            self._show_route_minimap(self.route_frame)
        self.status_var.set(f"路径已删除：{selected}")

    def delete_route_point(self) -> None:
        selected = self._selected_route_profile_name()
        if not selected:
            self.status_var.set("请先选择要删除的挂机点")
            return
        referenced = [
            str(item.get("name", "")).strip()
            for item in self.route_paths
            if str(item.get("source", "")).strip() == selected
            or str(item.get("target", "")).strip() == selected
        ]
        if referenced:
            self.status_var.set(
                f"挂机点仍被路径引用，请先删除路径：{', '.join(referenced)}"
            )
            return
        self.route_points = [
            item
            for item in self.route_points
            if str(item.get("name", "")).strip() != selected
        ]
        self.route_combat_profiles.pop(selected, None)
        if self.route_combat_point_name == selected:
            names = self._route_point_names()
            self.route_combat_point_name = names[0] if names else None
        try:
            self._write_config()
        except (OSError, ValueError) as exc:
            self.status_var.set(f"挂机点删除失败：{exc}")
            return
        self._refresh_route_controls()
        if self.route_frame is not None:
            self._show_route_frame(self.route_frame)
            self._show_route_minimap(self.route_frame)
        self.status_var.set(f"挂机点已删除：{selected}")

    def add_route_step(self) -> None:
        if self.route_current_path_key is None:
            self.status_var.set("请先新建路径")
            return
        step_type = self.route_step_type_var.get().strip()
        step: dict[str, Any] = {"type": step_type}
        if step_type == "move_to_point":
            value = self.route_step_object_var.get().strip()
            if not value:
                self.status_var.set("请先创建挂机点")
                return
            step["point"] = value
        elif step_type in {"move_to_ladder", "climb"}:
            value = self.route_step_object_var.get().strip()
            if not value:
                self.status_var.set("请先记录梯子或跳跃点")
                return
            step["ladder"] = value
        if step_type not in {
            "move_to_point",
            "move_to_ladder",
            "move_left",
            "move_right",
        }:
            try:
                duration = int(float(self.route_step_duration_var.get()))
            except ValueError:
                self.status_var.set("步骤时长必须是整数毫秒")
                return
            if duration <= 0:
                self.status_var.set("步骤时长必须大于0")
                return
            step["duration_ms"] = duration
        self.route_current_steps.append(step)
        self._update_route_status()

    def remove_route_step(self) -> None:
        if self.route_current_steps:
            self.route_current_steps.pop()
            self._update_route_status()
            self.status_var.set("已删除最后一个步骤")

    def remove_selected_route_step(self) -> None:
        selection = self.route_steps_list.curselection()
        if not selection:
            self.status_var.set("请先选择要删除的路径步骤")
            return
        index = int(selection[0])
        if 0 <= index < len(self.route_current_steps):
            self.route_current_steps.pop(index)
            self._update_route_status()
            self.status_var.set(f"已删除第 {index + 1} 个路径步骤")

    def _update_route_status(self) -> None:
        if self.route_current_path_key is None:
            self.route_status_var.set("路径步骤：尚未新建")
            if hasattr(self, "route_steps_list"):
                self.route_steps_list.delete(0, tk.END)
            return
        labels = [
            f"{index + 1}. {step_label(step)}"
            for index, step in enumerate(self.route_current_steps)
        ]
        if hasattr(self, "route_steps_list"):
            self.route_steps_list.delete(0, tk.END)
            for label in labels:
                self.route_steps_list.insert(tk.END, label)
        steps = "\n".join(labels)
        self.route_status_var.set(
            f"路径步骤：{self.route_current_path_key}\n"
            f"{steps or '暂无步骤'}"
        )

    def _route_config_for_save(self) -> dict[str, Any]:
        route_config = deepcopy(self.config.get("route_combat", {}))
        try:
            alt_delay = max(1, int(float(self.route_alt_delay_var.get())))
            rise_px = max(1, int(float(self.route_rise_var.get())))
            no_monster = max(0.1, float(self.route_no_monster_var.get()))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "路径参数必须是有效数字：Alt延迟、黄点上移、无怪切换秒"
            ) from exc
        jump_key = self.route_jump_key_var.get().strip().lower()
        if not jump_key:
            raise ValueError("跳跃键不能为空")
        # The runtime editor now supports path combat only. Keep this explicit
        # so an older config cannot silently switch the runtime back to normal.
        route_config["mode"] = "path_combat"
        route_config["active_group"] = self._selected_route_group()
        route_config["alt_delay_ms"] = alt_delay
        route_config["climb_minimap_rise_px"] = rise_px
        route_config["no_monster_seconds"] = no_monster
        route_config["jump_key"] = jump_key
        route_config.setdefault("obstacle_check_interval_ms", 300)
        route_config.setdefault("obstacle_jump_delay_min_ms", 500)
        route_config.setdefault("obstacle_jump_delay_max_ms", 700)
        route_config["points"] = [
            {
                **dict(item),
                "group": route_point_group(item),
            }
            for item in self.route_points
        ]
        route_config["point_profiles"] = {
            str(name): dict(profile)
            for name, profile in self.route_combat_profiles.items()
        }
        route_config["ladders"] = [
            {
                **dict(item),
                "group": route_ladder_group(item),
            }
            for item in self.route_ladders
        ]
        route_config["paths"] = [
            {
                **dict(item),
                "group": route_path_group(item),
            }
            for item in self.route_paths
        ]
        return route_config

    def save_route_path(self) -> None:
        if self.route_current_path_key is None:
            self.status_var.set("请先新建路径")
            return
        source, separator, target = self.route_current_path_key.partition(" -> ")
        if not separator:
            self.status_var.set("路径名称无效")
            return
        record = {
            "name": self.route_current_path_key,
            "source": source,
            "target": target,
            "steps": [dict(step) for step in self.route_current_steps],
            "group": self._selected_route_group(),
        }
        self.route_paths = [
            item
            for item in self.route_paths
            if str(item.get("name", "")).strip() != self.route_current_path_key
        ]
        self.route_paths.append(record)
        try:
            self._write_config()
        except (OSError, ValueError) as exc:
            self.status_var.set(f"路径保存失败：{exc}")
            return
        self._refresh_route_controls()
        self.route_test_start_button.configure(state="normal")
        self.status_var.set(f"路径已保存：{self.route_current_path_key}")

    def _write_config(self) -> None:
        saved = deepcopy(self._runtime_config())
        saved["route_combat"] = self._route_config_for_save()
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(saved, handle, allow_unicode=True, sort_keys=False)
        self.config = saved

    def _selected_route(self) -> dict[str, Any] | None:
        if self.route_current_path_key and self.route_current_steps:
            source, separator, target = self.route_current_path_key.partition(" -> ")
            if separator:
                return {
                    "name": self.route_current_path_key,
                    "source": source,
                    "target": target,
                    "steps": [dict(step) for step in self.route_current_steps],
                }
        selected = self.route_path_var.get().strip()
        return next(
            (
                dict(item)
                for item in self.route_paths
                if str(item.get("name", "")).strip() == selected
            ),
            None,
        )

    def start_route_test(self) -> None:
        if self.source is None:
            self.status_var.set("请先绑定窗口")
            return
        path = self._selected_route()
        if path is None or not path.get("steps"):
            self.status_var.set("请先选择一条包含步骤的路径")
            return
        route_config = self._route_config_for_save()
        route_config["mode"] = "path_combat"
        test_config = deepcopy(self._runtime_config())
        test_config["route_combat"] = route_config
        try:
            self.route_test_input = WindowsInputController(
                dry_run=bool(self.dry_run_var.get()),
                target_hwnd=self.source.window_info.window_id,
                mode=self.input_mode_var.get(),
            )
            self.route_test_controller = RouteTestController(
                config=test_config,
                path=path,
            )
            self.route_test_running = True
            self.route_test_start_button.configure(state="disabled")
            self.route_test_stop_button.configure(state="normal")
            self.status_var.set(f"路径测试已开始：{path['name']}")
            self._schedule_route_test()
        except Exception as exc:
            logger.exception("unable to start Windows route test")
            self.status_var.set(f"路径测试启动失败：{exc}")
            self.stop_route_test()

    def _schedule_route_test(self) -> None:
        if self.route_test_running:
            interval = int(
                self.config.get("system", {}).get(
                    "gui_runtime_interval_ms", 150
                )
            )
            self.route_test_after_id = self.root.after(
                max(25, interval), self.route_test_tick
            )

    def route_test_tick(self) -> None:
        self.route_test_after_id = None
        if not self.route_test_running or self.route_test_controller is None:
            return
        try:
            frame = self.source.next_frame()
            if frame is None:
                raise RuntimeError("窗口截图为空")
            result = self.route_test_controller.process_frame(frame)
            self.route_test_input.emit(result.action)
            self.status_var.set(
                f"路径测试：步骤={self.route_test_controller.step_index + 1} "
                f"action={result.action} reason={result.reason}"
            )
            if result.state in {"complete", "failed"}:
                self.stop_route_test()
                return
        except Exception as exc:
            logger.exception("Windows route test tick failed")
            self.status_var.set(f"路径测试失败：{exc}")
            self.stop_route_test()
            return
        self._schedule_route_test()

    def stop_route_test(self) -> None:
        if self.route_test_after_id is not None:
            self.root.after_cancel(self.route_test_after_id)
            self.route_test_after_id = None
        if self.route_test_controller is not None:
            self.route_test_controller.stop()
        if self.route_test_input is not None:
            self.route_test_input.release_all()
        self.route_test_controller = None
        self.route_test_input = None
        self.route_test_running = False
        self.route_test_start_button.configure(
            state="normal" if self.route_paths else "disabled"
        )
        self.route_test_stop_button.configure(state="disabled")

    def bind_window(self) -> None:
        self.stop_runtime()
        self._close_source()
        title = self.title_var.get().strip()
        try:
            self.source = WindowsWindowSource(
                title,
                output_size=self.config.get("capture", {}).get("output_size"),
            )
        except Exception as exc:
            logger.exception("unable to bind Windows window")
            self.window_label.configure(text="窗口：绑定失败")
            self.status_var.set(f"绑定失败：{exc}")
            return
        info = self.source.window_info
        self.window_label.configure(
            text=(
                f"窗口：hwnd={info.window_id} class={info.owner} "
                f"title={info.title} bounds={info.bounds}"
            )
        )
        self.start_button.configure(state="normal")
        self.detect_player_button.configure(state="normal")
        self.fixed_point_button.configure(state="normal")
        self.status_var.set("窗口已绑定，可以获取游戏画面")

    def capture_game_frame(self) -> None:
        if self.source is None:
            self.status_var.set("请先绑定窗口")
            return
        try:
            frame = self.source.next_frame()
            if frame is None:
                raise RuntimeError("窗口截图为空")
            self.current_frame = frame
            self.screen_region = None
            self.selection_mode = None
            self.region_var.set(
                f"寻怪区域：未设置，当前画面={frame.shape[1]}x{frame.shape[0]}"
            )
            self._show_frame(frame)
            self.detect_player_button.configure(state="normal")
            self.fixed_point_button.configure(state="normal")
            self.status_var.set(
                f"已获取游戏画面：{frame.shape[1]}x{frame.shape[0]}，"
                "可点击“标注寻怪区域”后在画面上拖动"
            )
        except Exception as exc:
            logger.exception("unable to capture Windows game frame")
            self.status_var.set(f"获取失败：{exc}")

    def toggle_region_marking(self) -> None:
        if self.current_frame is None:
            self.status_var.set("请先获取游戏画面")
            return
        self.selection_mode = (
            None if self.selection_mode == "band" else "band"
        )
        if self.selection_mode == "band":
            self.status_var.set(
                "请在游戏画面上从上到下拖动，程序会按横向全屏区域识别"
            )
        else:
            self.status_var.set("已取消标注")

    def _selection_start(self, event) -> None:
        if self.selection_mode != "band":
            return
        point = self._canvas_to_frame(event.x, event.y)
        if point is None:
            return
        self.selection_start = point
        if self.selection_rect_id is not None:
            self.canvas.delete(self.selection_rect_id)
        self.selection_rect_id = self.canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="#ff3b30",
            width=2,
        )

    def _selection_drag(self, event) -> None:
        if self.selection_mode != "band" or self.selection_start is None:
            return
        if self.selection_rect_id is None:
            return
        x0, y0 = self.selection_start
        current = self._canvas_to_frame(event.x, event.y)
        if current is None:
            return
        _, y1 = current
        display_y0 = y0 * self.display_scale
        display_y1 = y1 * self.display_scale
        self.canvas.coords(
            self.selection_rect_id,
            0,
            display_y0,
            self.current_frame.shape[1] * self.display_scale,
            display_y1,
        )

    def _selection_end(self, event) -> None:
        if self.selection_mode != "band" or self.selection_start is None:
            return
        current = self._canvas_to_frame(event.x, event.y)
        if current is None or self.current_frame is None:
            return
        _, start_y = self.selection_start
        _, end_y = current
        y0, y1 = sorted((int(start_y), int(end_y)))
        frame_height, frame_width = self.current_frame.shape[:2]
        y0 = max(0, min(frame_height - 1, y0))
        y1 = max(y0 + 1, min(frame_height, y1))
        self.screen_region = (0, y0, frame_width, y1 - y0)
        self._apply_screen_region(self.screen_region)
        self.selection_mode = None
        self.selection_start = None
        self.region_var.set(
            f"寻怪区域：x=0 y={y0} width={frame_width} height={y1 - y0}"
        )
        self.status_var.set("寻怪区域已设置，点击开始进入识别")

    def _canvas_to_frame(
        self,
        canvas_x: float,
        canvas_y: float,
    ) -> tuple[float, float] | None:
        if self.current_frame is None or self.display_scale <= 0:
            return None
        x = canvas_x / self.display_scale
        y = canvas_y / self.display_scale
        height, width = self.current_frame.shape[:2]
        if x < 0 or y < 0 or x > width or y > height:
            return None
        return x, y

    def _apply_screen_region(
        self,
        region: tuple[int, int, int, int],
    ) -> None:
        combat = self.config.setdefault("combat", {})
        monsters = combat.setdefault("monsters", {})
        monsters["region"] = list(region)
        player = combat.setdefault("player", {})
        player["search_region"] = list(region)

    def _player_runtime_config(
        self,
        runtime_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve the selected role ID and role templates for this run."""
        combat = runtime_config.setdefault("combat", {})
        player = dict(combat.get("player", {}))
        player_id = self.player_id_var.get().strip()
        if not player_id:
            player_id = str(
                player.get("id", player.get("name", ""))
            ).strip()
        player["id"] = player_id
        player["name"] = player_id
        root = self._project_root()
        role_dir = self._template_dir("角色")
        role_templates = sorted(role_dir.glob("template_*.png"))
        if role_templates:
            player["templates"] = [str(role_dir.resolve())]
            player.pop("template", None)
            player.pop("anchor_offset", None)
            player["anchor_below_nameplate"] = int(
                player.get("anchor_below_nameplate", 24)
            )
        else:
            resolved = resolve_player_config(player_id, player, root)
            if resolved is not None:
                player = resolved
            else:
                raw_templates = player.get("templates")
                if raw_templates:
                    values = (
                        [raw_templates]
                        if isinstance(raw_templates, (str, Path))
                        else list(raw_templates)
                    )
                    player["templates"] = [
                        resolve_app_path(value, root) for value in values
                    ]
                if player.get("template"):
                    player["template"] = resolve_app_path(
                        player["template"],
                        root,
                    )
        if self.screen_region is not None:
            player["search_region"] = list(self.screen_region)
        combat["player"] = player
        return player

    def detect_player(self) -> None:
        if self.source is None:
            self.status_var.set("请先绑定窗口")
            return
        try:
            frame = self.source.next_frame()
            if frame is None:
                raise RuntimeError("窗口截图为空")
            detection_config = deepcopy(self.config)
            player_config = self._player_runtime_config(detection_config)
            detection = detect_player_screen(frame, player_config)
            if detection is None:
                self.current_frame = frame
                self._show_frame(frame)
                self.status_var.set(
                    f"未识别到角色：ID={player_config.get('id') or '未设置'}"
                )
                return
            annotated = frame.copy()
            x, y, width, height = detection.name_box
            cv2.rectangle(
                annotated,
                (x, y),
                (x + width, y + height),
                (0, 215, 255),
                2,
            )
            cv2.circle(
                annotated,
                tuple(int(value) for value in detection.anchor),
                4,
                (255, 0, 255),
                -1,
            )
            self.current_frame = frame
            self._show_frame(annotated)
            self.status_var.set(
                f"已识别角色 {detection.name}："
                f"位置=({detection.anchor[0]},{detection.anchor[1]}) "
                f"相似度={detection.score:.3f}"
            )
        except Exception as exc:
            logger.exception("unable to detect Windows player")
            self.status_var.set(f"识别角色失败：{exc}")

    def mark_current_fixed_point(self) -> None:
        if self.current_frame is None:
            self.status_var.set("请先获取游戏画面")
            return
        minimap = tuple(self.config.get("minimap", {}).get("region") or ())
        hsv = self.config.get("minimap", {}).get("player_color_hsv", {})
        if len(minimap) != 4:
            self.status_var.set("小地图区域未配置")
            return
        try:
            observation = observe_minimap(
                self.current_frame,
                minimap,
                tuple(hsv["lower"]),
                tuple(hsv["upper"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            self.status_var.set(f"读取小地图颜色配置失败：{exc}")
            return
        if observation.player_local is None:
            self.status_var.set("当前截图没有识别到角色黄色点")
            return

        local_point = (
            int(observation.player_local.x),
            int(observation.player_local.y),
        )
        self.fixed_point_target = local_point
        self.fixed_point_enabled_var.set(True)
        profile_name = self.route_combat_point_name
        if profile_name and profile_name in self.route_combat_profiles:
            profile = self.route_combat_profiles.setdefault(profile_name, {})
            profile["fixed_point"] = list(local_point)
            profile["fixed_point_enabled"] = True
            target = f"挂机点 {profile_name}"
        else:
            fixed = self.config.setdefault("combat", {}).setdefault(
                "fixed_point",
                {},
            )
            fixed["point"] = list(local_point)
            fixed["enabled"] = True
            target = "全局"
        self.status_var.set(
            f"已记录{target}固定点："
            f"({local_point[0]},{local_point[1]})，请保存配置"
        )

    def _show_frame(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        max_width, max_height = 900, 620
        self.display_scale = min(
            1.0,
            max_width / max(1, width),
            max_height / max(1, height),
        )
        display = frame
        if self.display_scale != 1.0:
            display = cv2.resize(
                frame,
                (
                    max(1, int(round(width * self.display_scale))),
                    max(1, int(round(height * self.display_scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        ok, encoded = cv2.imencode(".ppm", rgb)
        if not ok:
            raise RuntimeError("无法编码预览画面")
        data = base64.b64encode(encoded.tobytes()).decode("ascii")
        self._photo = tk.PhotoImage(data=data, format="PPM")
        self.canvas.delete("all")
        self.selection_rect_id = None
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        self.canvas.configure(
            scrollregion=(0, 0, display.shape[1], display.shape[0])
        )
        if self.screen_region is not None:
            x, y, region_width, region_height = self.screen_region
            self.selection_rect_id = self.canvas.create_rectangle(
                x * self.display_scale,
                y * self.display_scale,
                (x + region_width) * self.display_scale,
                (y + region_height) * self.display_scale,
                outline="#ff3b30",
                width=2,
            )

    def _runtime_config(self) -> dict[str, Any]:
        runtime_config = deepcopy(self.config)
        runtime_config.setdefault("game_window", {})[
            "title_contains"
        ] = self.title_var.get().strip()
        runtime_config.setdefault("system", {})[
            "windows_input_mode"
        ] = self.input_mode_var.get().strip() or "post_message"
        runtime_config["system"]["dry_run"] = bool(self.dry_run_var.get())
        root = self._project_root()
        combat = runtime_config.setdefault("combat", {})

        player = self._player_runtime_config(runtime_config)
        player["id"] = self.player_id_var.get().strip() or str(
            player.get("id", player.get("name", ""))
        )
        player["name"] = player["id"]
        combat["player"] = player

        monster_config = runtime_config.setdefault("combat", {}).setdefault(
            "monsters", {}
        )
        selected_labels = self._selected_attack_monster_labels()
        if selected_labels:
            monster_config["allowed_labels"] = selected_labels
        else:
            monster_config["allowed_labels"] = [
                label for _, label in self._attackable_template_categories()
            ]
        configured_groups = monster_config.get("template_groups") or {}
        template_groups: dict[str, list[str]] = {}
        for label, values in configured_groups.items():
            raw_values = (
                [values]
                if isinstance(values, (str, Path))
                else list(values or [])
            )
            template_groups[str(label)] = [
                resolve_app_path(value, root) for value in raw_values
            ]
        for _, label in self._template_categories():
            directory = self._template_dir(label)
            if directory.is_dir() and list(directory.glob("template_*.png")):
                template_groups[label] = [str(directory.resolve())]
        monster_config["template_groups"] = template_groups

        try:
            attack_distance = positive_int(
                self.attack_distance_var.get(),
                "攻击距离",
            )
        except ValueError:
            raise
        attack_range = runtime_config["combat"].setdefault("attack_range", {})
        attack_range["distance"] = attack_distance

        health = runtime_config["combat"].setdefault("health", {})
        health["low_threshold"] = percentage_threshold(
            self.heal_threshold_var.get(),
            "回血阈值",
        )
        mana = runtime_config["combat"].setdefault("mana", {})
        mana["low_threshold"] = percentage_threshold(
            self.mana_threshold_var.get(),
            "回蓝阈值",
        )

        fixed = runtime_config["combat"].setdefault("fixed_point", {})
        fixed_enabled = bool(self.fixed_point_enabled_var.get())
        fixed["enabled"] = fixed_enabled
        fixed["point"] = (
            list(self.fixed_point_target)
            if self.fixed_point_target is not None
            else fixed.get("point")
        )
        try:
            fixed["return_distance_px"] = float(
                self.fixed_point_return_distance_var.get()
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("固定点回位距离必须是数字") from exc
        if fixed["return_distance_px"] < 0:
            raise ValueError("固定点回位距离不能小于 0")
        if fixed_enabled and (
            not isinstance(fixed["point"], (list, tuple))
            or len(fixed["point"]) != 2
        ):
            raise ValueError("启用固定点战斗前，请先记录当前黄点")

        pickup = runtime_config.setdefault("combat", {}).setdefault(
            "pickup", {}
        )
        try:
            pickup_interval = float(self.pickup_interval_var.get())
        except (TypeError, ValueError) as exc:
            raise ValueError("捡金币周期必须是正数秒") from exc
        if pickup_interval <= 0:
            raise ValueError("捡金币周期必须大于 0 秒")
        pickup["enabled"] = True
        pickup["interval_seconds"] = pickup_interval
        pickup["scan_seconds"] = 1.0
        coins = pickup.setdefault("coins", {})
        coins["enabled"] = True
        coins["allowed_labels"] = ["金币"]
        coin_dir = self._template_dir("金币")
        if coin_dir.is_dir() and list(coin_dir.glob("template_*.png")):
            coins["template_groups"] = {
                "金币": [str(coin_dir.resolve())]
            }
        if self.screen_region is not None:
            coins["region"] = list(self.screen_region)
        runtime_config["route_combat"] = self._route_config_for_save()
        if self.screen_region is not None:
            combat = runtime_config.setdefault("combat", {})
            combat.setdefault("monsters", {})["region"] = list(
                self.screen_region
            )
            combat.setdefault("player", {})["search_region"] = list(
                self.screen_region
            )
        return runtime_config

    def start_runtime(self) -> None:
        if self.source is None:
            self.status_var.set("请先绑定窗口")
            return
        if self.runtime_running:
            return
        try:
            runtime_config = self._runtime_config()
            self.runtime_controller = VisionController(
                source=self.source,
                input_controller=WindowsInputController(
                    dry_run=bool(self.dry_run_var.get()),
                    target_hwnd=self.source.window_info.window_id,
                    mode=self.input_mode_var.get(),
                ),
                config=runtime_config,
            )
        except Exception as exc:
            logger.exception("unable to start Windows runtime")
            self.status_var.set(f"启动失败：{exc}")
            return
        self.runtime_running = True
        self.runtime_log_lines = []
        self.runtime_log_state = {}
        self._append_runtime_log("识别已启动")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set(
            "识别已开始" + ("（dry-run）" if self.dry_run_var.get() else "（真实按键）")
        )
        self._schedule_runtime_tick()

    def _schedule_runtime_tick(self) -> None:
        if not self.runtime_running:
            return
        interval = int(
            self.config.get("system", {}).get(
                "gui_runtime_interval_ms",
                250,
            )
        )
        self.runtime_after_id = self.root.after(
            max(25, interval),
            self.runtime_tick,
        )

    def runtime_tick(self) -> None:
        self.runtime_after_id = None
        if not self.runtime_running or self.runtime_controller is None:
            return
        if self._runtime_tick_busy:
            self._schedule_runtime_tick()
            return
        self._runtime_tick_busy = True
        try:
            frame = self.source.next_frame()
            if frame is None:
                raise RuntimeError("窗口截图为空")
            combat = self.runtime_controller.process_frame(frame)
            self.runtime_controller.input.emit(combat.action or "none")
            for message in runtime_event_messages(
                combat,
                self.runtime_controller,
                self.runtime_log_state,
            ):
                self._append_runtime_log(message)
            self.status_var.set("识别运行中")
        except Exception as exc:
            logger.exception("Windows runtime tick failed")
            self.status_var.set(f"运行失败：{exc}")
            self._append_runtime_log("识别异常，已停止")
            self.stop_runtime()
        finally:
            self._runtime_tick_busy = False
        self._schedule_runtime_tick()

    def stop_runtime(self) -> None:
        if self.runtime_after_id is not None:
            self.root.after_cancel(self.runtime_after_id)
            self.runtime_after_id = None
        if self.runtime_controller is not None:
            self.runtime_controller.input.release_all()
        self.runtime_controller = None
        self.runtime_running = False
        self.start_button.configure(
            state="normal" if self.source is not None else "disabled"
        )
        self.stop_button.configure(state="disabled")
        self._append_runtime_log("识别已停止")
        self.status_var.set("识别已停止")

    def _append_runtime_log(self, message: str) -> None:
        if not hasattr(self, "runtime_log"):
            return
        text = str(message).strip()
        if not text:
            return
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.runtime_log_lines.append(f"[{stamp}] {text}")
        self.runtime_log_lines = self.runtime_log_lines[-300:]
        self.runtime_log.configure(state="normal")
        self.runtime_log.delete("1.0", tk.END)
        self.runtime_log.insert("1.0", "\n".join(self.runtime_log_lines) + "\n")
        self.runtime_log.configure(state="disabled")
        self.runtime_log.see(tk.END)

    def save_config(self) -> None:
        try:
            config = self._runtime_config()
            config.setdefault("system", {})["dry_run"] = bool(
                self.dry_run_var.get()
            )
            config.setdefault("system", {})["windows_input_mode"] = (
                self.input_mode_var.get().strip() or "post_message"
            )
            config["route_combat"] = self._route_config_for_save()
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    config,
                    handle,
                    allow_unicode=True,
                    sort_keys=False,
                )
            self.config = config
            self.status_var.set(f"配置已保存：{self.config_path}")
        except Exception as exc:
            logger.exception("unable to save Windows config")
            self.status_var.set(f"保存失败：{exc}")

    def _close_source(self) -> None:
        if self.source is not None:
            self.source.close()
            self.source = None

    def close(self) -> None:
        self.stop_route_test()
        self.stop_runtime()
        self._close_source()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_gui(config_path: Path) -> None:
    app = WindowsMapleVisionApp(load_config(config_path), config_path)
    app.run()

