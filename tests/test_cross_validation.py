"""Cross-validation: the three engines must agree where their domains overlap.

These tests are the point of the whole library. They are marked xfail
while binomial and monte_carlo are unimplemented — remove the markers
module-by-module as you build.
"""

import pytest

from optionslib import ExerciseStyle, Option, OptionType
from optionslib import binomial, black_scholes, monte_carlo

not_implemented = pytest.mark.xfail(
    raises=NotImplementedError, reason="engine not implemented yet", strict=True
)


def make(option_type=OptionType.CALL, **overrides) -> Option:
    params = dict(
        spot=100.0, strike=105.0, maturity=1.0, rate=0.05, volatility=0.25,
        dividend_yield=0.02, option_type=option_type,
    )
    params.update(overrides)
    return Option(**params)


class TestBinomialVsBlackScholes:
    @not_implemented
    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    @pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
    def test_european_converges(self, option_type, strike):
        opt = make(option_type, strike=strike)
        assert binomial.price(opt, steps=2000) == pytest.approx(
            black_scholes.price(opt), abs=1e-3
        )

    @not_implemented
    def test_american_call_no_dividends_equals_european(self):
        # Without dividends, early exercise of a call is never optimal.
        american = make(OptionType.CALL, dividend_yield=0.0, style=ExerciseStyle.AMERICAN)
        european = make(OptionType.CALL, dividend_yield=0.0)
        assert binomial.price(american, steps=2000) == pytest.approx(
            black_scholes.price(european), abs=1e-3
        )

    @not_implemented
    def test_american_put_carries_early_exercise_premium(self):
        american = make(OptionType.PUT, style=ExerciseStyle.AMERICAN)
        european = make(OptionType.PUT)
        premium = binomial.price(american, steps=2000) - black_scholes.price(european)
        assert premium > 0.0


class TestMonteCarloVsBlackScholes:
    @not_implemented
    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_price_within_confidence_band(self, option_type):
        opt = make(option_type)
        result = monte_carlo.price(opt, n_paths=200_000, seed=42)
        assert abs(result.price - black_scholes.price(opt)) < 3.0 * result.std_error

    @not_implemented
    def test_antithetic_reduces_variance(self):
        opt = make()
        plain = monte_carlo.price(opt, n_paths=100_000, antithetic=False, seed=7)
        anti = monte_carlo.price(opt, n_paths=100_000, antithetic=True, seed=7)
        assert anti.std_error < plain.std_error
