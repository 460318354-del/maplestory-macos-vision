from pathlib import Path

from maplevision.config.loader import load_config


def test_default_config_loads():
    config = load_config(Path("config/default.yaml"))
    assert config["system"]["dry_run"] is False
    assert config["game_window"]["title_contains"]
    assert {
        "yellow_mushroom",
        "red_snail",
        "small_mushroom",
        "710猪",
    }.issubset(set(config["combat"]["monsters"]["template_groups"]))
    assert config["combat"]["player"]["threshold"] == 0.72
    assert config["combat"]["pickup"]["min_coins"] == 1
    assert config["route_combat"]["mode"] in {"normal", "path_combat"}
    assert isinstance(config["route_combat"]["point_profiles"], dict)
