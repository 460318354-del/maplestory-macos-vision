from pathlib import Path

from maplevision.config.loader import load_config


def test_default_config_loads():
    config = load_config(Path("config/default.yaml"))
    assert config["system"]["dry_run"] is True
    assert config["game_window"]["title_contains"]
