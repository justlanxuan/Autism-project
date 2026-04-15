"""Config assembly for extract stage (mode A fragments without extractors folder).

This module is part of video_pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[3]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_fragment(category: str, name: str) -> Dict[str, Any]:
    """Load a fragment config from configs/{category}/{name}.yaml."""
    path = REPO_ROOT / "configs" / category / f"{name}.yaml"
    if path.exists():
        return load_config(path)
    return {}


def assemble_extract_config(extract_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble extract configuration from workflow + per-component fragments.

    Fragment loading rules:
      - detector:    configs/detectors/{name}.yaml
      - tracker:     configs/trackers/{name}.yaml
      - pose_estimator: configs/pose_estimators/{name}.yaml

    Workflow-level extract_cfg takes precedence over fragments.
    """
    merged: Dict[str, Any] = {}

    for component in ("detector", "tracker", "pose_estimator"):
        name = extract_cfg.get(component)
        if name:
            fragment = load_fragment(f"{component}s", name)
            # Put fragment keys under the same flat namespace for backward compat
            merged = _deep_merge(merged, fragment)

    # Workflow extract section overrides everything
    merged = _deep_merge(merged, extract_cfg)
    return merged
