from maplevision.platform.macos.input import MacOSInputController
import maplevision.platform.macos.input as input_module


class FakeQuartz:
    kCGHIDEventTap = 0

    def __init__(self):
        self.events = []
        self.pid_events = []

    def CGEventCreateKeyboardEvent(self, _source, key_code, down):
        return key_code, down

    def CGEventPost(self, _tap, event):
        self.events.append(event)

    def CGEventPostToPid(self, pid, event):
        self.pid_events.append((pid, event))


class FakeInputController(MacOSInputController):
    def __init__(self):
        super().__init__(dry_run=False)
        self.fake_quartz = FakeQuartz()

    def _load_quartz(self):
        return self.fake_quartz


def test_live_input_holds_movement_then_switches_to_attack_then_releases():
    controller = FakeInputController()

    controller.emit("move_right(right)")
    controller.emit("move_right(right)")
    controller.emit("hold_attack(ctrl)")
    controller.emit("observe")

    assert controller.fake_quartz.events == [
        (124, True),
        (124, False),
        (59, True),
        (59, False),
    ]


def test_live_input_releases_held_key_before_tapping_heal():
    controller = FakeInputController()

    controller.emit("move_left(left)")
    controller.emit("heal(1)")

    assert controller.fake_quartz.events == [
        (123, True),
        (123, False),
        (18, True),
        (18, False),
    ]


def test_live_input_taps_z_while_direction_is_held(monkeypatch):
    controller = FakeInputController()
    now = [0.0]
    monkeypatch.setattr(input_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(input_module.random, "uniform", lambda _low, _high: 0.2)
    monkeypatch.setattr(input_module.time, "sleep", lambda _seconds: None)

    controller.emit("move_left(left)")
    now[0] = 0.19
    controller.emit("move_left(left)")
    now[0] = 0.20
    controller.emit("move_left(left)")

    assert controller.fake_quartz.events == [
        (123, True),
        (6, True),
        (6, False),
    ]
    assert controller._held_keys == {"left"}

    controller.emit("observe")
    assert controller.fake_quartz.events[-1] == (123, False)


def test_live_input_taps_z_while_attack_is_held(monkeypatch):
    controller = FakeInputController()
    now = [0.0]
    monkeypatch.setattr(input_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(input_module.random, "uniform", lambda _low, _high: 0.2)
    monkeypatch.setattr(input_module.time, "sleep", lambda _seconds: None)

    controller.emit("hold_attack(ctrl)")
    now[0] = 0.20
    controller.emit("hold_attack(ctrl)")

    assert controller.fake_quartz.events == [
        (59, True),
        (6, True),
        (6, False),
    ]
    assert controller._held_keys == {"ctrl"}


def test_live_input_taps_attack_with_a_300_to_450ms_interval(monkeypatch):
    controller = FakeInputController()
    now = [0.0]
    monkeypatch.setattr(input_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(input_module.random, "uniform", lambda _low, _high: 0.3)
    monkeypatch.setattr(input_module.time, "sleep", lambda _seconds: None)

    controller.emit("tap_attack(ctrl)")
    controller._next_z_at = 999.0
    now[0] = 0.29
    controller.emit("tap_attack(ctrl)")
    assert controller.fake_quartz.events == [
        (59, True),
        (59, False),
    ]

    now[0] = 0.30
    controller.emit("tap_attack(ctrl)")
    assert controller.fake_quartz.events[-2:] == [
        (59, True),
        (59, False),
    ]
    assert controller._held_keys == set()


def test_live_input_faces_target_before_oriented_attack(monkeypatch):
    controller = FakeInputController()
    monkeypatch.setattr(input_module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(input_module.random, "uniform", lambda _low, _high: 0.3)
    monkeypatch.setattr(input_module.time, "sleep", lambda _seconds: None)

    controller.emit("tap_attack(ctrl,left)")
    controller.emit("tap_attack(ctrl,left)")
    controller.emit("tap_attack(ctrl,right)")

    assert controller.fake_quartz.events == [
        (123, True),
        (123, False),
        (59, True),
        (59, False),
        (124, True),
        (124, False),
        (59, True),
        (59, False),
    ]


def test_live_input_picks_up_with_three_z_taps(monkeypatch):
    controller = FakeInputController()
    now = [0.0]
    monkeypatch.setattr(input_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(input_module.random, "uniform", lambda _low, _high: 0.2)
    monkeypatch.setattr(input_module.time, "sleep", lambda _seconds: None)

    controller.emit("move_right(right)")
    controller.emit("pickup(z,3)")
    now[0] = 0.20
    controller.emit("pickup(z,3)")
    now[0] = 0.40
    controller.emit("pickup(z,3)")
    now[0] = 0.60
    controller.emit("pickup(z,3)")

    assert controller.fake_quartz.events == [
        (124, True),
        (124, False),
        (6, True),
        (6, False),
        (6, True),
        (6, False),
        (6, True),
        (6, False),
    ]


def test_live_input_can_post_events_to_bound_window_pid():
    controller = FakeInputController()
    controller.target_pid = 4321

    controller.emit("hold_attack(ctrl)")
    controller.emit("observe")

    assert controller.fake_quartz.events == []
    assert controller.fake_quartz.pid_events == [
        (4321, (59, True)),
        (4321, (59, False)),
    ]


def test_live_input_supports_ladder_alt_and_up_actions(monkeypatch):
    controller = FakeInputController()
    monkeypatch.setattr(input_module.time, "sleep", lambda _seconds: None)

    controller.emit("tap_alt")
    controller.emit("hold_up")
    controller.emit("observe")

    assert controller.fake_quartz.events == [
        (58, True),
        (58, False),
        (126, True),
        (126, False),
    ]


def test_live_input_jumps_right_after_holding_direction(monkeypatch):
    controller = FakeInputController()
    monkeypatch.setattr(input_module.time, "sleep", lambda _seconds: None)

    controller.emit("jump_right(right,alt)")
    controller.emit("jump_right(right,alt)")
    controller.emit("observe")

    assert controller.fake_quartz.events == [
        (124, True),
        (58, True),
        (58, False),
        (124, False),
    ]
    assert controller._held_keys == set()


def test_live_input_rearms_direction_before_jump(monkeypatch):
    controller = FakeInputController()
    monkeypatch.setattr(input_module.time, "sleep", lambda _seconds: None)

    controller.emit("move_right(right)")
    controller.emit("jump_right(right,alt)")

    assert controller.fake_quartz.events == [
        (124, True),
        (124, False),
        (124, True),
        (58, True),
        (58, False),
    ]


def test_live_input_supports_space_as_jump_key(monkeypatch):
    controller = FakeInputController()
    monkeypatch.setattr(input_module.time, "sleep", lambda _seconds: None)

    controller.emit("jump_right(right,space)")
    controller.emit("observe")

    assert controller.fake_quartz.events == [
        (124, True),
        (49, True),
        (49, False),
        (124, False),
    ]
