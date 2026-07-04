"""Black-Scholes-Merton closed-form prices and Greeks for European options.

This module is the analytical reference the numerical engines (binomial,
Monte Carlo) are cross-validated against. All formulas include a continuous
dividend yield q (Merton's extension).

Conventions:
    theta is per year — divide by 365 for a per-calendar-day theta.
    vega and rho are per unit change (1.00 = 100 points of vol / rate) —
    divide by 100 for the per-1%-move numbers quoted on desks.
"""

from __future__ import annotations

import math

from scipy.stats import norm

from optionslib.instruments import ExerciseStyle, Option


def _require_european(option: Option) -> None:
    if option.style is not ExerciseStyle.EUROPEAN:
        raise ValueError(
            "Black-Scholes closed form only prices European options; "
            "use the binomial tree for American exercise."
        )


def _deterministic_limit(option: Option) -> float | None:
    """Price in the degenerate limits; None if the option is non-degenerate.

    T = 0: the option is at expiry, worth intrinsic value.
    sigma = 0: the underlying is deterministic, S_T = S e^{(r-q)T}, so the
    option is worth the discounted certain payoff (forward intrinsic).
    """
    o = option
    if o.maturity == 0.0:
        return o.intrinsic()
    if o.volatility == 0.0:
        forward = o.spot * math.exp(-o.dividend_yield * o.maturity) - o.strike * math.exp(
            -o.rate * o.maturity
        )
        payoff = forward if o.is_call else -forward
        return max(payoff, 0.0)
    return None


def _d1_d2(option: Option) -> tuple[float, float]:
    o = option
    if o.maturity == 0.0 or o.volatility == 0.0:
        raise ValueError(
            "Greeks are not defined in the zero-vol / zero-maturity limit "
            "(delta degenerates to a step function)"
        )
    sig_sqrt_t = o.volatility * math.sqrt(o.maturity)
    d1 = (
        math.log(o.spot / o.strike)
        + (o.rate - o.dividend_yield + 0.5 * o.volatility**2) * o.maturity
    ) / sig_sqrt_t
    return d1, d1 - sig_sqrt_t


def price(option: Option) -> float:
    """Black-Scholes-Merton price of a European call or put."""
    _require_european(option)
    o = option
    limit = _deterministic_limit(o)
    if limit is not None:
        return limit
    d1, d2 = _d1_d2(o)
    disc_spot = o.spot * math.exp(-o.dividend_yield * o.maturity)
    disc_strike = o.strike * math.exp(-o.rate * o.maturity)
    if o.is_call:
        return disc_spot * norm.cdf(d1) - disc_strike * norm.cdf(d2)
    return disc_strike * norm.cdf(-d2) - disc_spot * norm.cdf(-d1)


def delta(option: Option) -> float:
    """dV/dS."""
    _require_european(option)
    o = option
    d1, _ = _d1_d2(o)
    dividend_disc = math.exp(-o.dividend_yield * o.maturity)
    if o.is_call:
        return dividend_disc * norm.cdf(d1)
    return dividend_disc * (norm.cdf(d1) - 1.0)


def gamma(option: Option) -> float:
    """d2V/dS2 — identical for calls and puts."""
    _require_european(option)
    o = option
    d1, _ = _d1_d2(o)
    dividend_disc = math.exp(-o.dividend_yield * o.maturity)
    return dividend_disc * norm.pdf(d1) / (o.spot * o.volatility * math.sqrt(o.maturity))


def vega(option: Option) -> float:
    """dV/dsigma, per unit of vol — identical for calls and puts."""
    _require_european(option)
    o = option
    d1, _ = _d1_d2(o)
    dividend_disc = math.exp(-o.dividend_yield * o.maturity)
    return o.spot * dividend_disc * norm.pdf(d1) * math.sqrt(o.maturity)


def theta(option: Option) -> float:
    """dV/dt, per year (negative for long options away from deep-ITM puts)."""
    _require_european(option)
    o = option
    d1, d2 = _d1_d2(o)
    dividend_disc = math.exp(-o.dividend_yield * o.maturity)
    rate_disc = math.exp(-o.rate * o.maturity)
    decay = -o.spot * dividend_disc * norm.pdf(d1) * o.volatility / (
        2.0 * math.sqrt(o.maturity)
    )
    if o.is_call:
        return (
            decay
            - o.rate * o.strike * rate_disc * norm.cdf(d2)
            + o.dividend_yield * o.spot * dividend_disc * norm.cdf(d1)
        )
    return (
        decay
        + o.rate * o.strike * rate_disc * norm.cdf(-d2)
        - o.dividend_yield * o.spot * dividend_disc * norm.cdf(-d1)
    )


def rho(option: Option) -> float:
    """dV/dr, per unit of rate."""
    _require_european(option)
    o = option
    _, d2 = _d1_d2(o)
    rate_disc = math.exp(-o.rate * o.maturity)
    if o.is_call:
        return o.strike * o.maturity * rate_disc * norm.cdf(d2)
    return -o.strike * o.maturity * rate_disc * norm.cdf(-d2)


def greeks(option: Option) -> dict[str, float]:
    """All five Greeks in one call."""
    return {
        "delta": delta(option),
        "gamma": gamma(option),
        "vega": vega(option),
        "theta": theta(option),
        "rho": rho(option),
    }
