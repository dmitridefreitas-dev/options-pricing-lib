"""greeks.numerical against greeks.analytic — and against the tree."""

import functools

import pytest

from optionslib import ExerciseStyle, Option, OptionType
from optionslib import binomial, black_scholes, greeks


def make(option_type=OptionType.CALL, **overrides) -> Option:
    params = dict(
        spot=100.0, strike=105.0, maturity=1.0, rate=0.05, volatility=0.25,
        dividend_yield=0.02, option_type=option_type,
    )
    params.update(overrides)
    return Option(**params)


class TestNumericalMatchesAnalytic:
    """Central differences on the closed form must recover its own Greeks."""

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_all_greeks(self, option_type):
        opt = make(option_type)
        numeric = greeks.numerical(opt, black_scholes.price)
        for name, exact in greeks.analytic(opt).items():
            assert numeric[name] == pytest.approx(exact, rel=1e-4, abs=1e-6), name


class TestTreeGreeks:
    """Bump-and-reprice on the binomial tree, cross-checked against BSM.

    The tree's O(1/N) pricing error is amplified by 1/bump, so the bump is
    large (1% of spot) and the tolerance honest rather than flattering.
    """

    def test_european_delta_matches_analytic(self):
        opt = make()
        pricer = functools.partial(binomial.price, steps=2000)
        tree_delta = greeks.delta(opt, pricer, bump=1.0)
        assert tree_delta == pytest.approx(black_scholes.delta(opt), abs=5e-3)

    def test_american_put_delta_is_steeper(self):
        # Early exercise makes the American put more sensitive to spot
        # (more negative delta) than its European twin.
        american = make(OptionType.PUT, style=ExerciseStyle.AMERICAN)
        european = make(OptionType.PUT)
        pricer = functools.partial(binomial.price, steps=1000)
        assert greeks.delta(american, pricer, bump=1.0) < greeks.delta(
            european, pricer, bump=1.0
        )
