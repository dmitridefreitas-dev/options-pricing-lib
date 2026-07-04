"""Cox-Ross-Rubinstein binomial tree — European and American exercise.

CRR parameterisation:
    dt = T / steps
    u  = exp(sigma * sqrt(dt)),  d = 1 / u
    p  = (exp((r - q) * dt) - d) / (u - d)      # risk-neutral up-probability

Terminal payoffs are rolled back one step at a time, discounting at
exp(-r * dt); American exercise takes max(continuation, intrinsic) at every
node. The rollback is a single numpy op per step, so 2000-step trees price
in milliseconds.

European prices converge to Black-Scholes at O(1/steps), with the classic
odd/even sawtooth (see the convergence plot in the validation notebook).
"""

from __future__ import annotations

import math

import numpy as np

from optionslib.instruments import ExerciseStyle, Option


def price(option: Option, steps: int = 500) -> float:
    """Price a vanilla option on a CRR tree."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    o = option
    payoff_sign = 1.0 if o.is_call else -1.0
    if o.maturity == 0.0:
        return o.intrinsic()
    if o.volatility == 0.0:
        # Deterministic underlying: S_t = S e^{(r-q)t}. European value is
        # the discounted certain payoff; American is the best discounted
        # exercise value along the deterministic path.
        times = (
            np.linspace(0.0, o.maturity, steps + 1)
            if o.style is ExerciseStyle.AMERICAN
            else np.array([o.maturity])
        )
        spots = o.spot * np.exp((o.rate - o.dividend_yield) * times)
        exercise = np.exp(-o.rate * times) * np.maximum(
            payoff_sign * (spots - o.strike), 0.0
        )
        return float(exercise.max())

    dt = o.maturity / steps
    u = math.exp(o.volatility * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp((o.rate - o.dividend_yield) * dt)
    p = (growth - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"risk-neutral up-probability {p:.4f} outside (0, 1); "
            "the drift per step exceeds the tree spacing — increase steps"
        )
    discount = math.exp(-o.rate * dt)

    # Node j at step n has j up-moves: S * u^j * d^(n-j) = S * u^(2j - n).
    exponents = 2.0 * np.arange(steps + 1) - steps
    spots = o.spot * u**exponents
    values = np.maximum(payoff_sign * (spots - o.strike), 0.0)

    american = o.style is ExerciseStyle.AMERICAN
    for step in range(steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            exponents = 2.0 * np.arange(step + 1) - step
            spots = o.spot * u**exponents
            values = np.maximum(values, payoff_sign * (spots - o.strike))

    return float(values[0])
