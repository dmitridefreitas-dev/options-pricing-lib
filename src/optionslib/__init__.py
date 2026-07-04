"""optionslib — option pricing three ways, cross-validated.

Engines:
    black_scholes  closed-form European prices and Greeks (reference)
    binomial       Cox-Ross-Rubinstein tree, European and American
    monte_carlo    GBM simulation with standard errors

Utilities:
    implied_vol    invert a market price to a Black-Scholes vol
"""

from optionslib.instruments import ExerciseStyle, Option, OptionType

__all__ = ["ExerciseStyle", "Option", "OptionType"]
