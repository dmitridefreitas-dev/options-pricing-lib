# options-pricing-lib

Price European and American vanilla options three independent ways — Black-Scholes
closed form, Cox-Ross-Rubinstein binomial tree, and Monte Carlo simulation — and
cross-validate the engines against each other. Full analytic Greeks, an implied-vol
solver, and a validation notebook that shows the numerical methods converging to the
closed form. The point is not any single pricer (each is textbook material); it is
demonstrating that three independently built engines agree to tolerance, which is how
pricing libraries are actually trusted in practice.

## Status

- [x] Contract types (`Option`, dataclass-based)
- [x] Black-Scholes-Merton closed form: price + delta, gamma, vega, theta, rho (with dividend yield)
- [x] Test suite: known values (Hull), put-call parity, Greeks vs finite differences
- [ ] CRR binomial tree — European + American (spec in `binomial.py`, tests written)
- [ ] Monte Carlo with standard errors + antithetic variates (spec in `monte_carlo.py`, tests written)
- [ ] Implied-vol solver (spec in `implied_vol.py`, tests written)
- [ ] Convergence plots in the validation notebook
- [ ] IV smile / surface construction
- [ ] Stretch: barrier options (analytic vs Monte Carlo)

## Quick start

```bash
pip install -e ".[dev]"
pytest
```

```python
from optionslib import Option, OptionType
from optionslib import black_scholes as bs

opt = Option(spot=42, strike=40, maturity=0.5, rate=0.10, volatility=0.20,
             option_type=OptionType.CALL)
bs.price(opt)   # 4.7594
bs.greeks(opt)  # {'delta': 0.7791, 'gamma': 0.0500, ...}
```

## Layout

```
src/optionslib/
  instruments.py     option contract definition shared by all engines
  black_scholes.py   closed-form reference (done)
  binomial.py        CRR tree (spec'd, in progress)
  monte_carlo.py     GBM simulation (spec'd, in progress)
  implied_vol.py     price -> vol inversion (spec'd, in progress)
tests/               each engine validated against the others
notebooks/           validation & convergence plots (cell-format .py)
```

## Methodology

Unimplemented engines already have their acceptance tests written and marked
`xfail` — the spec lives in the tests, and the markers come off as each engine
lands. Conventions: continuous compounding throughout, continuous dividend
yield q, theta per year, vega/rho per unit (divide by 100 for per-point).

## Results

*(convergence and smile plots land here as the numerical engines are completed)*

## Limitations / what I'd do next

- Constant-vol GBM world: no term structure of rates or vol, no discrete dividends.
- American exercise only via the tree; would add Longstaff-Schwartz least-squares
  Monte Carlo to cross-check the early-exercise premium.
- Scalar float API; would vectorise over strikes/maturities for surface work.
