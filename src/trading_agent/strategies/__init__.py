from __future__ import annotations

from dataclasses import fields

from trading_agent.strategies.base import Strategy
from trading_agent.strategies.library import DonchianBreakout, EmaCross, RsiDip

REGISTRY: dict[str, type[Strategy]] = {
    cls.name: cls for cls in (EmaCross, DonchianBreakout, RsiDip)
}


def build_strategy(name: str, params: dict[str, str] | None = None) -> Strategy:
    """Instantiate a strategy, casting string params to the field types."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {', '.join(REGISTRY)}")
    cls = REGISTRY[name]
    types = {f.name: type(f.default) for f in fields(cls)}
    kwargs = {}
    for key, value in (params or {}).items():
        if key not in types:
            raise ValueError(f"Strategy '{name}' has no parameter '{key}'. Available: {', '.join(types)}")
        kwargs[key] = types[key](value)
    return cls(**kwargs)


__all__ = ["REGISTRY", "Strategy", "build_strategy"]
