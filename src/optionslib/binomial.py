"""Cox-Ross-Rubinstein binomial tree — European and American exercise.

STATUS: not implemented yet. The spec below is the contract; tests in
tests/test_cross_validation.py are already written against it and are
marked xfail until this module is done. Remove the xfail markers as you
implement.

Spec (CRR parameterisation):
    dt = T / steps
    u  = exp(sigma * sqrt(dt)),  d = 1 / u
    p  = (exp((r - q) * dt) - d) / (u - d)      # risk-neutral up-probability

    1. Build terminal payoffs at step N: S * u^j * d^(N-j), j = 0..N.
    2. Roll back one step at a time, discounting at exp(-r * dt).
    3. For AMERICAN style, at every node take
       max(continuation, intrinsic) before stepping back.

    Vectorise the rollback with numpy (one array op per step) — a Python
    double loop will be ~100x slower and makes the convergence study painful.

Acceptance criteria (encoded in tests):
    European: |tree(steps=2000) - black_scholes| < 1e-3
    American call, q=0: equals the European price (no early exercise)
    American put: price >= European put (early-exercise premium >= 0)
"""

from __future__ import annotations

from optionslib.instruments import Option


def price(option: Option, steps: int = 500) -> float:
    """Price a vanilla option on a CRR tree. See module docstring for spec."""
    raise NotImplementedError("TODO: implement the CRR rollback (spec in module docstring)")
