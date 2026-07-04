"""Monte Carlo pricing under geometric Brownian motion.

European payoffs depend only on S_T, so the terminal price is simulated
directly (no path loop):

    S_T = S * exp((r - q - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)

with Z ~ N(0, 1) from a seeded numpy Generator, so runs are reproducible.

Every price comes with its standard error — a Monte Carlo estimate without
one is meaningless. With antithetic variates the (Z, -Z) pair average is
treated as a single i.i.d. sample; computing the error over the pooled 2n
draws would be wrong, because paired draws are negatively correlated (which
is exactly why the trick reduces variance for monotone payoffs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from optionslib.instruments import ExerciseStyle, Option


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
    """Price a European option by GBM terminal-price simulation."""
    if option.style is not ExerciseStyle.EUROPEAN:
        raise ValueError(
            "plain GBM Monte Carlo only prices European options; "
            "American exercise needs the binomial tree (or Longstaff-Schwartz)."
        )
    if n_paths < 2:
        raise ValueError(f"n_paths must be >= 2, got {n_paths}")

    o = option
    drift = (o.rate - o.dividend_yield - 0.5 * o.volatility**2) * o.maturity
    vol_sqrt_t = o.volatility * math.sqrt(o.maturity)
    discount = math.exp(-o.rate * o.maturity)
    payoff_sign = 1.0 if o.is_call else -1.0

    def discounted_payoff(z: np.ndarray) -> np.ndarray:
        terminal = o.spot * np.exp(drift + vol_sqrt_t * z)
        return discount * np.maximum(payoff_sign * (terminal - o.strike), 0.0)

    rng = np.random.default_rng(seed)
    if antithetic:
        n_pairs = n_paths // 2
        z = rng.standard_normal(n_pairs)
        samples = 0.5 * (discounted_payoff(z) + discounted_payoff(-z))
        paths_used = 2 * n_pairs
    else:
        samples = discounted_payoff(rng.standard_normal(n_paths))
        paths_used = n_paths

    std_error = samples.std(ddof=1) / math.sqrt(len(samples))
    return MCResult(price=float(samples.mean()), std_error=float(std_error), n_paths=paths_used)
