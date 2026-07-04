# %% [markdown]
# # Validation notebook — the README plots come from here
#
# Runs as interactive cells in VS Code / PyCharm (`# %%` markers) or convert
# with jupytext. Sections 2-4 unlock as the corresponding engine is built.

# %%
import numpy as np
import matplotlib.pyplot as plt

from optionslib import Option, OptionType
from optionslib import black_scholes as bs

# %% [markdown]
# ## 1. Black-Scholes sanity: price and Greeks across moneyness

# %%
strikes = np.linspace(60, 140, 81)
base = dict(spot=100.0, maturity=0.5, rate=0.05, volatility=0.25)

calls = [Option(strike=float(k), option_type=OptionType.CALL, **base) for k in strikes]
prices = [bs.price(o) for o in calls]
deltas = [bs.delta(o) for o in calls]
gammas = [bs.gamma(o) for o in calls]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, values, title in zip(axes, [prices, deltas, gammas], ["Call price", "Delta", "Gamma"]):
    ax.plot(strikes, values)
    ax.axvline(base["spot"], color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Strike")
    ax.set_title(f"{title} vs strike (S=100, T=0.5, vol=25%)")
fig.tight_layout()

# %% [markdown]
# ## 2. Convergence study — binomial tree vs closed form
#
# TODO once `binomial.price` is implemented:
# plot |tree(N) - BS| against N on log-log axes for an ATM European call.
# You should see O(1/N) convergence with the classic odd/even sawtooth.

# %% [markdown]
# ## 3. Monte Carlo — error vs path count
#
# TODO once `monte_carlo.price` is implemented:
# plot standard error against n_paths on log-log axes (expect slope -1/2),
# with and without antithetic variates, plus the BS line the estimates
# should straddle.

# %% [markdown]
# ## 4. Implied-vol surface
#
# TODO once `implied_volatility` is implemented:
# invert a grid of synthetic quotes (or real chain data from the
# options-data-pipeline project) and plot smile slices per maturity plus
# the 3D surface.

# %%
plt.show()
