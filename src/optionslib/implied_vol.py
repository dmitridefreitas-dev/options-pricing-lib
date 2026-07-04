"""Implied volatility: invert a market price to a Black-Scholes vol.

Black-Scholes price is strictly increasing in vol, so Brent's method on
[1e-6, 5.0] is guaranteed a unique root whenever the quote sits strictly
between the zero-vol floor and the infinite-vol cap. Quotes outside that
band are arbitrage violations and are rejected with a clear message before
the solver runs — a bracket error from brentq is useless to a caller.

    floor (sigma -> 0):    call max(S e^{-qT} - K e^{-rT}, 0), put mirrored
    cap   (sigma -> inf):  call S e^{-qT},  put K e^{-rT}

Conditioning caveat: accuracy in vol is limited by vega (d_sigma ~=
d_price / vega), so recovered vols for deep ITM/OTM short-dated options
are only good to their conditioning limit. In the extreme the time value
underflows double precision, the quote sits exactly on the floor, and the
solver refuses rather than returning a junk vol.
"""

from __future__ import annotations

import math
from dataclasses import replace

from scipy.optimize import brentq

from optionslib import black_scholes
from optionslib.instruments import ExerciseStyle, Option

_VOL_LO = 1e-6
_VOL_HI = 5.0


def implied_volatility(market_price: float, option: Option) -> float:
    """Back out the Black-Scholes vol from a market price.

    The volatility field on `option` is ignored (it is the unknown).
    """
    o = option
    if o.style is not ExerciseStyle.EUROPEAN:
        raise ValueError("implied vol is defined against the European closed form")

    disc_spot = o.spot * math.exp(-o.dividend_yield * o.maturity)
    disc_strike = o.strike * math.exp(-o.rate * o.maturity)
    if o.is_call:
        floor, cap = max(disc_spot - disc_strike, 0.0), disc_spot
    else:
        floor, cap = max(disc_strike - disc_spot, 0.0), disc_strike

    if market_price <= floor:
        raise ValueError(
            f"price {market_price:.6f} is at or below the no-arbitrage floor "
            f"{floor:.6f} (the zero-vol limit) — no implied vol exists"
        )
    if market_price >= cap:
        raise ValueError(
            f"price {market_price:.6f} is at or above the no-arbitrage cap "
            f"{cap:.6f} (the infinite-vol limit) — no implied vol exists"
        )

    def objective(sigma: float) -> float:
        return black_scholes.price(replace(o, volatility=sigma)) - market_price

    return float(brentq(objective, _VOL_LO, _VOL_HI, xtol=1e-12))
