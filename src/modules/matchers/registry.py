"""Matcher registry."""

from src.utils.factory import Registry

MATCHER_REGISTRY = Registry()

# Register built-in matchers
from src.modules.matchers.hungarian import HungarianMatcher
from src.modules.matchers.dl_matchers.despite_matcher import DeSPITEMatcher

MATCHER_REGISTRY.register("hungarian")(HungarianMatcher)
MATCHER_REGISTRY.register("despite")(DeSPITEMatcher)
