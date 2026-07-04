# How this project works — study notes

These notes are the plain-English walkthrough: what each piece does, why it is built
that way, and the stories worth telling about it. Read alongside `notebooks/demo.ipynb`.

## What this project is

A library that prices vanilla options three independent ways and proves the three ways
agree. The pricing math is textbook; the *value* is the validation discipline — the
same reason a real desk trusts its pricing library is the reason to trust this one:
independent implementations cross-checked against each other, convergence measured
against theory, and known identities holding to machine precision.

## The model, in one paragraph

Everything lives in the Black-Scholes-Merton world: the stock follows geometric
Brownian motion with constant volatility σ, constant risk-free rate r, and a continuous
dividend yield q. Under the risk-neutral measure the stock drifts at (r − q), and an
option's value is the discounted expected payoff. The three engines are three ways of
computing that same expectation: in closed form (integrating the lognormal density
analytically), on a lattice (discrete approximation of the process), and by simulation
(sampling terminal prices directly).

## Module by module

- **`instruments.py`** — one frozen dataclass, `Option`, shared by every engine so a
  contract can't mean different things to different pricers. Validates inputs; zero
  maturity means "at expiry" (worth intrinsic), zero vol means "deterministic world"
  (worth discounted forward intrinsic) — handled explicitly, not by divide-by-zero.

- **`black_scholes.py`** — the analytic reference. Price plus all five Greeks (delta,
  gamma, vega, theta, rho), all extended for dividend yield. Conventions: theta per
  year, vega/rho per whole unit (divide by 100 for the per-point numbers quoted on
  desks). Greeks refuse the degenerate limits where delta becomes a step function.

- **`binomial.py`** — Cox-Ross-Rubinstein tree. Up/down factors u = e^(σ√Δt), d = 1/u;
  risk-neutral up-probability p = (e^((r−q)Δt) − d)/(u − d). Terminal payoffs rolled
  back one step at a time with one numpy operation per step (a Python double loop
  would be ~100x slower). American exercise = take max(continuation, intrinsic) at
  every node during rollback — that comparison is the entire difference between
  European and American pricing.

- **`monte_carlo.py`** — European payoffs depend only on the terminal price, so it
  samples S_T = S·exp((r − q − σ²/2)T + σ√T·Z) directly, no path loop. Seeded numpy
  Generator → bit-for-bit reproducible. Antithetic variates: pair each draw Z with −Z;
  because the payoff is monotone in Z, the pair's errors partially cancel.

- **`greeks.py`** — bump-and-reprice (central finite differences) against *any*
  pricer. This is the only route to American Greeks here, and it cross-checks the
  analytic formulas.

- **`implied_vol.py` / `vol_surface.py`** — the inverse problem: given a market
  price, find the σ that reproduces it. Price is strictly increasing in vol, so
  Brent's method on [1e-6, 5] always finds the unique root — *provided the quote is
  inside the no-arbitrage band*, which is checked first. `iv_surface` maps a grid of
  quotes through the solver; bad quotes become NaN instead of killing the grid.

## What each result shows

- **Convergence plot (left panel)**: |tree − closed form| falls as O(1/N) with the
  classic odd/even sawtooth. The measured constant is N·error ≈ 2.42 for the test
  option — perfectly flat across N, which is the fingerprint of a correct tree.
- **Convergence plot (right panel)**: Monte Carlo standard error falls as N^(−1/2)
  (halving error costs 4x paths); antithetic sits below plain at equal path count;
  realised errors scatter within a few standard errors of zero as they must.
- **Greeks panel**: gamma and vega peak at the money (where hedging is hardest and
  vol matters most); call and put share identical gamma/vega; rho flips sign.
- **Parity check**: C − P = S·e^(−qT) − K·e^(−rT) to ~1e-14. Model-free, so any
  miss beyond float noise is a bug by definition.
- **American premium figure**: put premium grows deep ITM (interest on the strike
  beats remaining optionality → exercise now); call premium with no dividends is
  exactly zero (Merton 1973). Computed as tree-minus-tree so discretisation cancels.
- **IV surface**: synthetic quotes from a known smile invert back onto the true
  curves to ~1e-9 — the well-posed round trip.

## The three stories worth knowing cold

1. **The CRR error constant.** The European convergence test tolerance is 2e-3 at
   N=2000 — not a guess, but measured: N·|error| ≈ 2.42 for the test case, constant
   across N. Setting tolerances from measured error constants instead of wishful
   thinking is the difference between a test suite and a rubber stamp.

2. **Implied vol is conditioning-limited.** The error in recovered vol ≈ (price
   noise)/vega. Deep ITM, short-dated, low vol → vega ~1e-17 → the time value
   underflows double precision and the quote sits *bit-for-bit on the arbitrage
   floor*: no vol is recoverable and the solver refuses rather than returning junk.
   Subtle twist: OTM options never hit this, because their floor is zero and floating
   point keeps full relative precision near zero — a 1e-89 price still inverts fine.

3. **Antithetic standard errors.** Each (Z, −Z) pair must be treated as ONE sample
   when computing the standard error. Pooling all 2n draws would be wrong: paired
   draws are negatively correlated — which is exactly why the trick reduces variance.
   The test suite verifies antithetic SE < plain SE at equal sample count.

## Design decisions, briefly

- Three engines rather than one: cross-validation is the product.
- Frozen dataclass contract: immutability prevents an engine mutating an option
  mid-computation; `dataclasses.replace` gives clean bumped copies for Greeks.
- Tests assert the *well-posed* invariant where the problem is ill-conditioned
  (price-space round-trip for IV) and the conditioning-limited one only to its limit.

## Likely interview questions

- *Why does the binomial tree converge as O(1/N)?* The discrete lattice misplaces the
  strike relative to the terminal nodes; the leading error term scales with Δt = T/N.
  The sawtooth comes from the strike falling alternately between/on node levels as N
  flips odd/even.
- *Why is early exercise never optimal for a call on a non-dividend stock?* Exercising
  forfeits the time value and pays K early (losing interest on K). Selling the option
  always dominates exercising; with q = 0 there is no dividend to capture that would
  offset it.
- *Why Brent and not Newton for implied vol?* Newton needs vega and can diverge where
  vega ≈ 0 (exactly the hard cases); Brent is derivative-free and guaranteed on a
  bracketed monotone function.
- *Why does your MC not simulate paths?* European payoffs are functions of S_T only;
  the terminal distribution is known in closed form (lognormal), so simulating
  intermediate steps adds cost and discretisation error for nothing. Path simulation
  becomes necessary for path-dependent payoffs (barriers, Asians) — a stated next step.
