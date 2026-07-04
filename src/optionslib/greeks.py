"""Greeks for any pricing engine.

Two routes to the same numbers:

    analytic(option)            closed-form BSM Greeks (European only)
    numerical(option, pricer)   central-difference bump-and-reprice against
                                any pricer, e.g. a binomial tree — the only
                                route to American Greeks here

Conventions match black_scholes: theta per year, vega/rho per unit.

Numerical caveat (worth understanding, it shows up in the tests): a
first-difference Greek amplifies the pricer's own error by 1/bump and a
second difference by 1/bump^2, so bumps for lattice pricers must be much
larger than for smooth closed forms, and tolerances looser.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from optionslib import black_scholes
from optionslib.instruments import Option

Pricer = Callable[[Option], float]


def analytic(option: Option) -> dict[str, float]:
    """Closed-form BSM Greeks (delta, gamma, vega, theta, rho)."""
    return black_scholes.greeks(option)


def _bump(option: Option, field: str, h: float) -> tuple[Option, Option]:
    value = getattr(option, field)
    return replace(option, **{field: value + h}), replace(option, **{field: value - h})


def delta(option: Option, pricer: Pricer, bump: float = 1e-4) -> float:
    up, dn = _bump(option, "spot", bump)
    return (pricer(up) - pricer(dn)) / (2.0 * bump)


def gamma(option: Option, pricer: Pricer, bump: float = 1e-2) -> float:
    up, dn = _bump(option, "spot", bump)
    return (pricer(up) - 2.0 * pricer(option) + pricer(dn)) / bump**2


def vega(option: Option, pricer: Pricer, bump: float = 1e-4) -> float:
    up, dn = _bump(option, "volatility", bump)
    return (pricer(up) - pricer(dn)) / (2.0 * bump)


def theta(option: Option, pricer: Pricer, bump: float = 1e-4) -> float:
    # theta is dV/dt with t = calendar time; maturity shrinks as t grows,
    # hence the sign flip on the maturity bump.
    up, dn = _bump(option, "maturity", bump)
    return -(pricer(up) - pricer(dn)) / (2.0 * bump)


def rho(option: Option, pricer: Pricer, bump: float = 1e-4) -> float:
    up, dn = _bump(option, "rate", bump)
    return (pricer(up) - pricer(dn)) / (2.0 * bump)


def numerical(option: Option, pricer: Pricer) -> dict[str, float]:
    """All five Greeks by central differences against `pricer`."""
    return {
        "delta": delta(option, pricer),
        "gamma": gamma(option, pricer),
        "vega": vega(option, pricer),
        "theta": theta(option, pricer),
        "rho": rho(option, pricer),
    }
