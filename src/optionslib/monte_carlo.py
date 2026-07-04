"""Monte Carlo pricing under geometric Brownian motion.

STATUS: not implemented yet. The spec below is the contract; tests in
tests/test_cross_validation.py are already written against it and are
marked xfail until this module is done. Remove the xfail markers as you
implement.

Spec:
    European payoffs depend only on S_T, so simulate the terminal price
    directly (no path loop needed):

        S_T = S * exp((r - q - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)

    with Z ~ N(0, 1) drawn via numpy Generator (seed parameter -> reproducible).

    1. Draw n_paths standard normals; with antithetic=True use (Z, -Z) pairs
       so the total sample count stays n_paths.
    2. Discount the mean payoff at exp(-r * T).
    3. Report the standard error of the discounted payoff mean — a Monte
       Carlo price without a standard error is meaningless.

Acceptance criteria (encoded in tests):
    |mc_price - black_scholes| < 3 * std_error  (99.7% of seeds)
    antithetic std_error < plain std_error at equal sample count
"""

from __future__ import annotations

from dataclasses import dataclass

from optionslib.instruments import Option


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    n_paths: int


def price(
    option: Option,
    n_paths: int = 100_000,
    antithetic: bool = True,
    seed: int | None = None,
) -> MCResult:
    """Price a European option by GBM simulation. See module docstring for spec."""
    raise NotImplementedError("TODO: implement terminal-price GBM sampling (spec in module docstring)")
