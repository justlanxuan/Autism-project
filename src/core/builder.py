"""Factory helpers for building configured modules."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.registry import (
	MATCHER_REGISTRY,
	POSE_ESTIMATOR_REGISTRY,
	TRACKER_REGISTRY,
)

# Import module registries for side-effect registration.
from src.modules.trackers import registry as _tracker_registry  # noqa: F401
from src.modules.pose_estimators import registry as _pose_registry  # noqa: F401
from src.modules.matchers import registry as _matcher_registry  # noqa: F401


def _normalize_config(config: Optional[Dict[str, Any]], key: str) -> Dict[str, Any]:
	if config is None:
		return {}
	section = config.get(key, {})
	if section is None:
		return {}
	if not isinstance(section, dict):
		raise TypeError(f"Config section '{key}' must be a mapping")
	return section


def build_tracker(config: Optional[Dict[str, Any]] = None):
	section = _normalize_config(config, "tracker")
	name = section.get("name", section.get("backend", "bytetrack"))
	params: Dict[str, Any] = {}
	if "params" in section and isinstance(section["params"], dict):
		params.update(section["params"])

	nested = section.get(name, {})
	if isinstance(nested, dict):
		params.update(nested)

	for key, value in section.items():
		if key not in {"name", "backend", "enabled", "valid_backends", "params", name}:
			params.setdefault(key, value)
	return TRACKER_REGISTRY.build(name, params)


def build_pose_estimator(config: Optional[Dict[str, Any]] = None):
	section = _normalize_config(config, "pose_estimator")
	name = section.get("name", "alphapose")
	params: Dict[str, Any] = {}
	if isinstance(section.get("params", {}), dict):
		params.update(section.get("params", {}))

	nested = section.get(name, {})
	if isinstance(nested, dict):
		params.update(nested)

	for key, value in section.items():
		if key not in {"name", "params", name}:
			params.setdefault(key, value)
	return POSE_ESTIMATOR_REGISTRY.build(name, params)


def _build_from_section(section: Dict[str, Any], default_name: str) -> tuple[str, Dict[str, Any]]:
	name = section.get("name", default_name)
	params: Dict[str, Any] = {}
	if isinstance(section.get("params", {}), dict):
		params.update(section.get("params", {}))
		
	nested = section.get(name, {})
	if isinstance(nested, dict):
		params.update(nested)
		
	for key, value in section.items():
		if key not in {"name", "params", name}:
			params.setdefault(key, value)
	return name, params


def build_matcher(config: Optional[Dict[str, Any]] = None):
	section = _normalize_config(config, "matcher")
	name, params = _build_from_section(section, default_name="hungarian")
	return MATCHER_REGISTRY.build(name, params)

