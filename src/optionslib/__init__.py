"""optionslib — option pricing three ways, cross-validated.

Engines:
    black_scholes  closed-form European prices and Greeks (reference)
    binomial       Cox-Ross-Rubinstein tree, European and American
    monte_carlo    GBM simulation with standard errors

Utilities:
    greeks         analytic Greeks + bump-and-reprice for any engine
    implied_vol    invert a market price to a Black-Scholes vol
    vol_surface    build an IV surface from a grid of quotes
"""

from optionslib.instruments import ExerciseStyle, Option, OptionType

__all__ = ["ExerciseStyle", "Option", "OptionType"]
