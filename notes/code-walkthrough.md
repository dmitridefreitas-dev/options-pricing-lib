# Code walkthrough — how every module actually works

Companion to `how-it-works.md` (concepts). This one is the code-level defense:
what each function does mechanically, and why it is written the way it is.

## Map

| File | Role | Key entry points |
|---|---|---|
| `instruments.py` | the shared contract type | `Option`, `Option.intrinsic` |
| `black_scholes.py` | analytic reference | `price`, `delta..rho`, `greeks` |
| `binomial.py` | CRR lattice, European + American | `price(option, steps)` |
| `monte_carlo.py` | GBM terminal simulation | `price(...) -> MCResult` |
| `greeks.py` | bump-and-reprice for any engine | `numerical(option, pricer)` |
| `implied_vol.py` | price → vol inversion | `implied_volatility` |
| `vol_surface.py` | quote grid → IV grid | `iv_surface` |

## `instruments.py` — the contract

`Option` is a **frozen** dataclass. Frozen buys three things: an engine can never
mutate a contract mid-computation; instances are hashable; and bumping for Greeks is
forced through `dataclasses.replace`, which creates a clean copy and re-runs
`__post_init__` validation — so an invalid bump (negative vol) fails loudly instead
of pricing garbage. Validation allows `maturity == 0` and `volatility == 0` on
purpose: those are meaningful limits (at-expiry, deterministic world), while negative
values are nonsense and raise.

## `black_scholes.py` — the closed form

- `_d1_d2` computes d1 = [ln(S/K) + (r − q + σ²/2)T] / (σ√T) and d2 = d1 − σ√T once,
  shared by price and all Greeks. It **raises** on σ=0 or T=0 — the Greeks degenerate
  to step functions there — while `price` handles those limits *before* ever calling
  it, via `_deterministic_limit` (T=0 → intrinsic; σ=0 → discounted forward
  intrinsic `max(±(S·e^{−qT} − K·e^{−rT}), 0)`).
- `price`: call = S·e^{−qT}·N(d1) − K·e^{−rT}·N(d2); put mirrored with N(−d2), N(−d1).
- Conventions to state before anyone asks: **theta is per year** (÷365 for per-day),
  **vega and rho are per whole unit** (÷100 for per-point), theta is dV/dt so the
  test bumps *maturity down* to move time forward.
- Greeks worth being able to derive on a whiteboard: put delta = e^{−qT}(N(d1) − 1);
  gamma = e^{−qT}·φ(d1)/(S·σ·√T), identical for call and put.

## `binomial.py` — the lattice

Parameters: `dt = T/steps`, `u = e^{σ√dt}`, `d = 1/u`,
`p = (e^{(r−q)dt} − d)/(u − d)`. Guard: if p leaves (0,1) — huge drift on a coarse
tree — raise with the fix in the message ("increase steps").

The two lines that make it fast and correct:

1. **Node pricing without a 2-D tree.** Node j at step n has price S·u^j·d^{n−j} =
   S·u^{2j−n} (because d = 1/u), so `exponents = 2*arange(n+1) − n` builds a whole
   level in one vectorised power.
2. **Rollback as one array op per step:**
   `values = disc * (p * values[1:] + (1−p) * values[:-1])`. Index arithmetic: after
   the slice, position j holds p·V(up child at j+1) + (1−p)·V(down child at j) —
   exactly the risk-neutral expectation. A Python double loop is ~100x slower.

American exercise inserts `values = max(values, payoff_sign*(spots − K))` at every
step — recomputing that level's spot array — which is the entire difference between
European and American pricing. The σ=0 branch prices the deterministic path
directly: European discounts the certain terminal payoff; American takes the max of
`e^{−rt}·payoff(S·e^{(r−q)t})` over the discretised exercise dates.

## `monte_carlo.py` — the simulation

European payoffs depend only on S_T, whose distribution is known:
`S_T = S·exp((r − q − σ²/2)T + σ√T·Z)`. So there is **no path loop** — one vector of
normals, one exp, one payoff. `default_rng(seed)` makes runs bit-reproducible.

The line interviewers probe: **antithetic standard error**. With antithetic on, the
samples are `0.5*(payoff(Z) + payoff(−Z))` — each *pair average* is one i.i.d.
sample, and the standard error is `samples.std(ddof=1)/√n_pairs`. Pooling all 2n
legs and dividing by √(2n) would be wrong because the legs are negatively
correlated — which is precisely why the trick reduces variance for monotone payoffs.
`ddof=1` = sample (not population) std. American exercise raises: plain terminal
GBM cannot see early exercise (Longstaff-Schwartz would be needed).

## `greeks.py` — bump-and-reprice

`_bump` uses `replace(option, field=value ± h)` and every Greek is a central
difference: `(V(+h) − V(−h)) / 2h`; gamma is the second difference
`(V(+h) − 2V(0) + V(−h)) / h²`. Theta flips sign because bumping *maturity up* moves
time backward. The error story to know cold: a first difference amplifies the
pricer's own noise by 1/h and a second difference by 1/h², so bumps for lattice
pricers must be large (the tree-delta test uses h = 1.0 on spot 100, i.e. 1%) and
tolerances honest (5e-3), while the closed form takes h = 1e-4 and matches at 1e-4
relative. Gamma defaults to a larger bump (1e-2) for exactly the 1/h² reason.

## `implied_vol.py` — the inversion

Order of operations is the design: compute the no-arbitrage floor
(zero-vol limit) and cap (infinite-vol limit) **first**, reject quotes outside
`(floor, cap)` with a message that names the violated bound — then run
`brentq(objective, 1e-6, 5.0, xtol=1e-12)` where `objective(σ) = BS(replace(option,
volatility=σ)) − market`. Brent needs only a sign change, which monotonicity of
price in vol guarantees inside the band. Why not Newton: it needs vega, and diverges
exactly where the problem is hard (vega → 0). The conditioning fact to volunteer
before being asked: recovered-vol error ≈ price-noise / vega, so deep-ITM
short-dated low-vol quotes can sit bit-for-bit *on* the floor (time value under
machine epsilon of a ~30-unit price) — no vol exists to recover, and the solver
refuses. OTM quotes never hit this: their floor is 0 and floats keep full relative
precision near zero, so even a 1e-89 price inverts.

## `vol_surface.py` — the grid

A plain double loop over (maturity, strike): build the `Option` (dummy vol 0.2 —
the solver ignores it), try `implied_volatility`, write NaN on `ValueError`. NaN-not-
crash is a policy: one arbitrage-violating quote must not poison a 100-cell grid.
Shape is validated up front (`prices.shape == (len(maturities), len(strikes))`)
because a silently transposed grid is the classic surface bug.

## The tests, as a defense layer

- `test_black_scholes.py`: Hull's worked example (S=42, K=40 → call 4.76/put 0.81),
  put-call parity to 1e-10 across a grid, every Greek vs finite differences.
- `test_cross_validation.py`: tree→BS at 2e-3 for N=2000 — tolerance set from the
  *measured* constant N·|err| ≈ 2.42, not wishful thinking; American call (q=0) =
  European; American put premium > 0 and pins to intrinsic deep ITM; MC within
  3 standard errors; antithetic < plain variance.
- `test_implied_vol.py`: price-space round-trip everywhere (the well-posed
  invariant); vol-space only to `max(1e-8, 1e-12/vega)` — tolerance scaled by
  conditioning; floor/cap rejection.

## Grilling Q&A (implementation level)

- *Why does the sawtooth exist?* The strike's position relative to the terminal node
  grid alternates as N flips parity; the leading error term oscillates while decaying
  as 1/N. Averaging N and N+1 trees, or Leisen-Reimer trees, smooth it — named as
  next steps, not silently applied.
- *Why scipy's `norm.cdf` and not `math.erf`?* Equivalent accuracy; scipy is already
  a dependency for `brentq`, and `norm.pdf/cdf` reads as the math.
- *Why is vega the same for call and put?* Parity: C − P is vol-independent, so
  ∂C/∂σ = ∂P/∂σ. Same argument gives equal gamma.
- *Why `xtol=1e-12` on brentq?* The round-trip test demands 1e-8 in vol where vega
  is healthy; 1e-12 leaves headroom without chasing noise below machine precision of
  the objective.
- *What breaks first if I extend to barriers?* Terminal-only MC — barrier payoffs are
  path-dependent, so simulation gains a time loop and a discretisation bias
  (continuity correction), and the tree needs barrier-aligned levels.
