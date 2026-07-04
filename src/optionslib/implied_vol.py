"""Implied volatility: invert a market price to a Black-Scholes vol.

STATUS: not implemented yet. Tests in tests/test_implied_vol.py are
written against this spec and marked xfail until implemented.

Spec:
    Root-find sigma such that black_scholes.price(option with sigma) equals
    market_price. Use scipy.optimize.brentq on [1e-6, 5.0] — price is
    monotone increasing in vol, so the bracket is guaranteed when a
    solution exists.

    Reject inputs below the no-arbitrage floor before root-finding:
        call: market_price >= max(S*exp(-qT) - K*exp(-rT), 0)
        put:  market_price >= max(K*exp(-rT) - S*exp(-qT), 0)
    Raise ValueError with a clear message otherwise — the solver's own
    bracket error is useless to a caller.

Next step after this works: build the IV surface. Take a strikes x
maturities grid of market quotes, invert each, and plot smile slices and
the 3D surface in the validation notebook.

Acceptance criteria (encoded in tests):
    round-trip: implied_volatility(price(opt), opt-without-vol) == opt.volatility
    to 1e-8 across moneyness 0.7-1.3 and maturities 0.05-2.0
"""

from __future__ import annotations

from optionslib.instruments import Option


def implied_volatility(market_price: float, option: Option) -> float:
    """Back out the Black-Scholes vol from a market price.

    The volatility field on `option` is ignored (it is the unknown).
    See module docstring for spec.
    """
    raise NotImplementedError("TODO: brentq root-find on vol (spec in module docstring)")
