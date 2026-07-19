import os
import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
EXAMPLE_CONFIG_PATH = PROJECT_ROOT / "config.example.yaml"


def _resolve_env(value: str) -> str:
    pattern = re.compile(r"\$\{(\w+)\}")
    def replacer(match):
        return os.environ.get(match.group(1), "")
    return pattern.sub(replacer, value)


def _walk_resolve(obj):
    if isinstance(obj, str):
        return _resolve_env(obj)
    elif isinstance(obj, dict):
        return {k: _walk_resolve(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_walk_resolve(v) for v in obj]
    return obj


def load_config(path: Path | None = None) -> dict:
    if path:
        path = Path(path)
    else:
        configured = os.environ.get("SCHOLARHOUND_CONFIG")
        if configured:
            path = Path(configured)
        elif DEFAULT_CONFIG_PATH.exists():
            path = DEFAULT_CONFIG_PATH
        else:
            path = EXAMPLE_CONFIG_PATH
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _walk_resolve(raw)
