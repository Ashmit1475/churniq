"""Configuration loading and path resolution.

Every script imports `load_config()` from here so that no path or parameter is
ever hardcoded in the pipeline itself.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Config:
    """Thin wrapper over the YAML config with path resolution and dot access."""

    def __init__(self, data: dict[str, Any], root: Path):
        self._data = data
        self.root = root

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def path(self, key: str) -> Path:
        """Resolve a key from the `paths` block into an absolute path."""
        rel = self._data["paths"][key]
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_url(self) -> str:
        """Database URL. Env var wins, then config, then local SQLite."""
        env_url = os.environ.get("DATABASE_URL")
        if env_url:
            return env_url
        cfg_url = self._data["database"].get("url")
        if cfg_url:
            return cfg_url
        db_path = self.root / self._data["paths"]["database"]
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path}"

    @property
    def random_state(self) -> int:
        return int(self._data["project"]["random_state"])


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data, PROJECT_ROOT)
