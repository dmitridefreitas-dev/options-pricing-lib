"""Implied-volatility surface construction from a grid of market quotes."""

from __future__ import annotations

import numpy as np

from optionslib.implied_vol import implied_volatility
from optionslib.instruments import Option, OptionType


def iv_surface(
    spot: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    prices: np.ndarray,
    rate: float,
    dividend_yield: float = 0.0,
    option_type: OptionType = OptionType.CALL,
) -> np.ndarray:
    """Invert a grid of market quotes into an implied-vol surface.

    Args:
        prices: quote grid of shape (len(maturities), len(strikes)).

    Returns:
        Implied vols, same shape as `prices`; NaN where a quote violates
        no-arbitrage bounds (rather than failing the whole grid).
    """
    prices = np.asarray(prices, dtype=float)
    expected = (len(maturities), len(strikes))
    if prices.shape != expected:
        raise ValueError(f"prices shape {prices.shape} != (maturities, strikes) {expected}")

    surface = np.full(expected, np.nan)
    for i, maturity in enumerate(maturities):
        for j, strike in enumerate(strikes):
            opt = Option(
                spot=spot, strike=float(strike), maturity=float(maturity),
                rate=rate, volatility=0.2, dividend_yield=dividend_yield,
                option_type=option_type,
            )
            try:
                surface[i, j] = implied_volatility(float(prices[i, j]), opt)
            except ValueError:
                pass
    return surface
