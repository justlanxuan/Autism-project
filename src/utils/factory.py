"""Lightweight factory registry."""

from __future__ import annotations

from typing import Any, Callable, Dict, TypeVar

T = TypeVar("T")


class Registry:
    """Simple name -> factory registry."""

    def __init__(self) -> None:
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
            key = name.lower()
            if key in self._items:
                raise KeyError(f"Registry item already exists: {name}")
            self._items[key] = factory
            return factory

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        key = name.lower()
        if key not in self._items:
            raise KeyError(f"Unknown registry item: {name}")
        return self._items[key]

    def build(self, name: str, *args: Any, **kwargs: Any):
        return self.get(name)(*args, **kwargs)
