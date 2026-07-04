"""Cross-validation: the three engines must agree where their domains overlap.

These tests are the point of the whole library — three independently built
pricers agreeing to tolerance is how a pricing library earns trust.
"""

import pytest

from optionslib import ExerciseStyle, Option, OptionType
from optionslib import binomial, black_scholes, monte_carlo


def make(option_type=OptionType.CALL, **overrides) -> Option:
    params = dict(
        spot=100.0, strike=105.0, maturity=1.0, rate=0.05, volatility=0.25,
        dividend_yield=0.02, option_type=option_type,
    )
    params.update(overrides)
    return Option(**params)


class TestBinomialVsBlackScholes:
    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    @pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
    def test_european_converges(self, option_type, strike):
        # CRR error is O(1/N) with an ATM sawtooth; for these parameters
        # N * |error| ~= 2.42, so 2e-3 is the honest bound at N=2000.
        opt = make(option_type, strike=strike)
        assert binomial.price(opt, steps=2000) == pytest.approx(
            black_scholes.price(opt), abs=2e-3
        )

    def test_american_call_no_dividends_equals_european(self):
        # Without dividends, early exercise of a call is never optimal.
        american = make(OptionType.CALL, dividend_yield=0.0, style=ExerciseStyle.AMERICAN)
        european = make(OptionType.CALL, dividend_yield=0.0)
        assert binomial.price(american, steps=2000) == pytest.approx(
            black_scholes.price(european), abs=1e-3
        )

    def test_american_put_carries_early_exercise_premium(self):
        american = make(OptionType.PUT, style=ExerciseStyle.AMERICAN)
        european = make(OptionType.PUT)
        premium = binomial.price(american, steps=2000) - black_scholes.price(european)
        assert premium > 0.0

    def test_deep_itm_american_put_equals_intrinsic(self):
        # Deep ITM with high carry: immediate exercise dominates, so the
        # tree price should pin to intrinsic value.
        opt = make(
            OptionType.PUT, strike=200.0, rate=0.10, volatility=0.10,
            dividend_yield=0.0, style=ExerciseStyle.AMERICAN,
        )
        assert binomial.price(opt, steps=1000) == pytest.approx(opt.intrinsic(), abs=1e-6)


class TestMonteCarloVsBlackScholes:
    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_price_within_confidence_band(self, option_type):
        opt = make(option_type)
        result = monte_carlo.price(opt, n_paths=200_000, seed=42)
        assert abs(result.price - black_scholes.price(opt)) < 3.0 * result.std_error

    def test_antithetic_reduces_variance(self):
        opt = make()
        plain = monte_carlo.price(opt, n_paths=100_000, antithetic=False, seed=7)
        anti = monte_carlo.price(opt, n_paths=100_000, antithetic=True, seed=7)
        assert anti.std_error < plain.std_error

    def test_seed_reproducibility(self):
        opt = make()
        a = monte_carlo.price(opt, n_paths=50_000, seed=123)
        b = monte_carlo.price(opt, n_paths=50_000, seed=123)
        assert a.price == b.price
        assert a.std_error == b.std_error
