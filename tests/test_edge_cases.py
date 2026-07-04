"""Degenerate limits and extreme moneyness, across all three engines."""

import math

import pytest

from optionslib import ExerciseStyle, Option, OptionType
from optionslib import binomial, black_scholes, monte_carlo


def make(option_type=OptionType.CALL, **overrides) -> Option:
    params = dict(
        spot=100.0, strike=95.0, maturity=1.0, rate=0.05, volatility=0.25,
        option_type=option_type,
    )
    params.update(overrides)
    return Option(**params)


class TestZeroVolatility:
    """sigma = 0: deterministic underlying, price = discounted forward intrinsic."""

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    @pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
    def test_all_engines_agree_on_the_deterministic_limit(self, option_type, strike):
        opt = make(option_type, strike=strike, volatility=0.0)
        forward = opt.spot - strike * math.exp(-opt.rate * opt.maturity)
        expected = max(forward if opt.is_call else -forward, 0.0)

        assert black_scholes.price(opt) == pytest.approx(expected, abs=1e-12)
        assert binomial.price(opt, steps=100) == pytest.approx(expected, abs=1e-12)
        mc = monte_carlo.price(opt, n_paths=1_000, seed=1)
        assert mc.price == pytest.approx(expected, abs=1e-12)
        # Every path is identical; the ~1e-16 residual is summation
        # rounding in numpy's mean, not sampling noise.
        assert mc.std_error == pytest.approx(0.0, abs=1e-12)

    def test_zero_vol_american_put_exercises_immediately(self):
        # Deterministic drift pushes the spot up while the put payoff
        # decays — exercising a deep ITM put now beats waiting.
        opt = make(
            OptionType.PUT, strike=150.0, volatility=0.0,
            style=ExerciseStyle.AMERICAN,
        )
        assert binomial.price(opt, steps=100) == pytest.approx(opt.intrinsic(), abs=1e-12)


class TestZeroMaturity:
    """T = 0: the option is at expiry and worth exactly intrinsic."""

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    @pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
    def test_all_engines_return_intrinsic(self, option_type, strike):
        opt = make(option_type, strike=strike, maturity=0.0)
        expected = opt.intrinsic()

        assert black_scholes.price(opt) == expected
        assert binomial.price(opt, steps=100) == expected
        assert monte_carlo.price(opt, n_paths=1_000, seed=1).price == pytest.approx(
            expected, abs=1e-12
        )


class TestGreeksAtDegenerateLimits:
    def test_greeks_rejected_at_zero_vol(self):
        with pytest.raises(ValueError, match="zero-vol"):
            black_scholes.delta(make(volatility=0.0))

    def test_greeks_rejected_at_zero_maturity(self):
        with pytest.raises(ValueError, match="zero-vol|zero-maturity"):
            black_scholes.gamma(make(maturity=0.0))


class TestExtremeMoneyness:
    def test_deep_itm_call_is_discounted_forward(self):
        opt = make(strike=1.0)
        expected = opt.spot - opt.strike * math.exp(-opt.rate * opt.maturity)
        assert black_scholes.price(opt) == pytest.approx(expected, abs=1e-9)
        assert binomial.price(opt, steps=500) == pytest.approx(expected, abs=1e-6)

    def test_deep_otm_call_is_worthless(self):
        opt = make(strike=100_000.0)
        assert black_scholes.price(opt) == pytest.approx(0.0, abs=1e-12)
        assert binomial.price(opt, steps=500) == pytest.approx(0.0, abs=1e-12)

    def test_deep_itm_delta_approaches_one(self):
        assert black_scholes.delta(make(strike=1.0)) == pytest.approx(1.0, abs=1e-9)

    def test_deep_otm_delta_approaches_zero(self):
        assert black_scholes.delta(make(strike=100_000.0)) == pytest.approx(0.0, abs=1e-9)

    def test_vega_dies_away_from_the_money(self):
        atm = black_scholes.vega(make(strike=100.0))
        deep_itm = black_scholes.vega(make(strike=1.0))
        deep_otm = black_scholes.vega(make(strike=100_000.0))
        assert deep_itm < 1e-6 < atm
        assert deep_otm < 1e-6 < atm


class TestInputValidation:
    def test_negative_inputs_rejected(self):
        with pytest.raises(ValueError, match="spot"):
            make(spot=-1.0)
        with pytest.raises(ValueError, match="strike"):
            make(strike=-1.0)
        with pytest.raises(ValueError, match="maturity"):
            make(maturity=-0.1)
        with pytest.raises(ValueError, match="volatility"):
            make(volatility=-0.2)
