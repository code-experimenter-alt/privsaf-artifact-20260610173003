from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "private_telemetry_benchmark_config.json"
CONFIG_ENV = "PRIVATE_TELEMETRY_BENCHMARK_CONFIG"


def load_benchmark_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or os.environ.get(CONFIG_ENV, DEFAULT_CONFIG))
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    return json.loads(config_path.read_text(encoding="utf-8"))


def config_path_from_env(path: str | Path | None = None) -> Path:
    config_path = Path(path or os.environ.get(CONFIG_ENV, DEFAULT_CONFIG))
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    return config_path


def result_path(relative_path: str) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else ROOT / path
