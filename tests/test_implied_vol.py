"""Implied vol must round-trip: price at sigma, invert, recover sigma.

Inversion is ill-conditioned where vega is tiny (deep ITM/OTM, short
expiry, low vol): d_sigma ~= d_price / vega, and in the extreme the time
value underflows double precision entirely — the quote sits exactly on
the no-arbitrage floor and no vol is recoverable. The tests encode that
honestly: price-space round-trip is asserted everywhere (the well-posed
invariant), vol-space accuracy only to its conditioning limit.
"""

import math
from dataclasses import replace

import pytest

from optionslib import Option, OptionType
from optionslib import black_scholes
from optionslib.implied_vol import implied_volatility


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("moneyness", [0.7, 0.9, 1.0, 1.1, 1.3])
@pytest.mark.parametrize("maturity", [0.05, 0.5, 2.0])
@pytest.mark.parametrize("sigma", [0.08, 0.25, 0.90])
def test_round_trip(option_type, moneyness, maturity, sigma):
    opt = Option(
        spot=100.0, strike=100.0 * moneyness, maturity=maturity,
        rate=0.05, volatility=sigma, dividend_yield=0.01, option_type=option_type,
    )
    market_price = black_scholes.price(opt)

    # Deep ITM + short expiry + low vol: the time value can vanish
    # entirely in double precision (quote bit-for-bit at a positive
    # floor). No vol is recoverable there and the solver must refuse.
    # OTM quotes never hit this: their floor is 0 and floats keep full
    # relative precision near zero, so even a 1e-89 price inverts fine.
    disc_spot = opt.spot * math.exp(-opt.dividend_yield * maturity)
    disc_strike = opt.strike * math.exp(-opt.rate * maturity)
    forward_intrinsic = disc_spot - disc_strike if opt.is_call else disc_strike - disc_spot
    if market_price <= max(forward_intrinsic, 0.0):
        with pytest.raises(ValueError, match="floor"):
            implied_volatility(market_price, opt)
        return

    recovered = implied_volatility(market_price, opt)

    # Well-posed invariant: repricing at the recovered vol reproduces the quote.
    assert black_scholes.price(replace(opt, volatility=recovered)) == pytest.approx(
        market_price, abs=1e-9
    )

    # Vol-space accuracy is limited by conditioning: d_sigma ~= d_price / vega.
    vol_tol = max(1e-8, 1e-12 / black_scholes.vega(opt))
    assert recovered == pytest.approx(sigma, abs=vol_tol)


def test_price_below_arbitrage_floor_rejected():
    opt = Option(
        spot=100.0, strike=80.0, maturity=1.0, rate=0.05,
        volatility=0.2, option_type=OptionType.CALL,
    )
    # A call can never be worth less than its discounted forward intrinsic.
    with pytest.raises(ValueError, match="floor"):
        implied_volatility(1.0, opt)


def test_price_above_arbitrage_cap_rejected():
    opt = Option(
        spot=100.0, strike=80.0, maturity=1.0, rate=0.05,
        volatility=0.2, option_type=OptionType.CALL,
    )
    # A call can never be worth more than the (dividend-discounted) spot.
    with pytest.raises(ValueError, match="cap"):
        implied_volatility(101.0, opt)
