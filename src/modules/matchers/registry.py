"""Matcher registry entries."""

from __future__ import annotations

from src.core.registry import MATCHER_REGISTRY

from src.modules.matchers.despite import DeSPITEMatcher
from src.modules.matchers.hungarian import HungarianMatcher


@MATCHER_REGISTRY.register("despite")
def _build_despite_matcher(config_dict):
    return DeSPITEMatcher(config_dict)


@MATCHER_REGISTRY.register("hungarian")
def _build_hungarian_matcher(config_dict):
    return HungarianMatcher(config_dict)
