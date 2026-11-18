from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import pandas as pd


class Strategy(ABC):
    """Turns candles into long-only entry/exit signals.

    Subclasses are dataclasses whose fields are the tunable parameters.

    `generate` returns a DataFrame aligned with the candles:
      entry          bool   open a position at the next bar's open
      exit           bool   close the position at the next bar's open
      stop_distance  float  price distance from entry to the stop-loss (NaN = no stop)

    Signals at bar i may only use data up to bar i's close.
    """

    name: ClassVar[str]
    description: ClassVar[str] = ""

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame: ...


def signals_frame(
    index: pd.Index,
    entry: pd.Series,
    exit: pd.Series,
    stop_distance: pd.Series | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["entry"] = entry.fillna(False).astype(bool)
    out["exit"] = exit.fillna(False).astype(bool)
    out["stop_distance"] = np.nan if stop_distance is None else stop_distance
    return out


def crossed_above(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on the bar where `a` moves from below `b` to above it."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossed_below(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))
