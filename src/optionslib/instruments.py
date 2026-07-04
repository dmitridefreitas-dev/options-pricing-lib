"""Contract definitions shared by every pricing engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True)
class Option:
    """A vanilla equity option.

    Attributes:
        spot: Current underlying price S.
        strike: Strike price K.
        maturity: Time to expiry T, in years. Zero means "at expiry":
            every engine returns intrinsic value.
        rate: Continuously compounded risk-free rate r.
        volatility: Annualised volatility sigma. Zero means a deterministic
            underlying: engines return the discounted forward intrinsic.
        dividend_yield: Continuous dividend yield q.
        option_type: CALL or PUT.
        style: EUROPEAN or AMERICAN.
    """

    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    dividend_yield: float = 0.0
    option_type: OptionType = OptionType.CALL
    style: ExerciseStyle = ExerciseStyle.EUROPEAN

    def __post_init__(self) -> None:
        if self.spot <= 0.0:
            raise ValueError(f"spot must be positive, got {self.spot}")
        if self.strike <= 0.0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.maturity < 0.0:
            raise ValueError(f"maturity must be non-negative, got {self.maturity}")
        if self.volatility < 0.0:
            raise ValueError(f"volatility must be non-negative, got {self.volatility}")

    @property
    def is_call(self) -> bool:
        return self.option_type is OptionType.CALL

    def intrinsic(self, spot: float | None = None) -> float:
        """Exercise value at the given spot (defaults to current spot)."""
        s = self.spot if spot is None else spot
        payoff = s - self.strike if self.is_call else self.strike - s
        return max(payoff, 0.0)
