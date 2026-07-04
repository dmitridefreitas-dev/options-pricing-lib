"""Implied vol must round-trip: price at sigma, invert, recover sigma."""

import pytest

from optionslib import Option, OptionType
from optionslib import black_scholes
from optionslib.implied_vol import implied_volatility

not_implemented = pytest.mark.xfail(
    raises=NotImplementedError, reason="solver not implemented yet", strict=True
)


@not_implemented
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
    assert implied_volatility(market_price, opt) == pytest.approx(sigma, abs=1e-8)


@not_implemented
def test_price_below_arbitrage_floor_rejected():
    opt = Option(
        spot=100.0, strike=80.0, maturity=1.0, rate=0.05,
        volatility=0.2, option_type=OptionType.CALL,
    )
    # A call can never be worth less than its discounted forward intrinsic.
    with pytest.raises(ValueError):
        implied_volatility(1.0, opt)
