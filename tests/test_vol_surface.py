"""Surface construction: a grid of quotes round-trips through the solver."""

import numpy as np

from optionslib import Option
from optionslib import black_scholes
from optionslib.vol_surface import iv_surface


def test_grid_round_trip_with_smile():
    spot, rate, q = 100.0, 0.05, 0.01
    strikes = np.linspace(80, 120, 9)
    maturities = np.array([0.25, 0.5, 1.0, 2.0])

    # Synthetic smile: quadratic in log-moneyness, lifted at the short end.
    log_m = np.log(strikes / spot)
    true_vols = 0.18 + 0.3 * log_m[None, :] ** 2 + 0.02 * np.exp(-2.0 * maturities[:, None])

    quotes = np.array([
        [
            black_scholes.price(Option(
                spot=spot, strike=float(k), maturity=float(t), rate=rate,
                volatility=float(v), dividend_yield=q,
            ))
            for k, v in zip(strikes, true_vols[i])
        ]
        for i, t in enumerate(maturities)
    ])

    recovered = iv_surface(spot, strikes, maturities, quotes, rate, dividend_yield=q)
    np.testing.assert_allclose(recovered, true_vols, atol=1e-8)


def test_bad_quote_yields_nan_not_failure():
    spot, rate = 100.0, 0.05
    strikes = np.array([90.0, 110.0])
    maturities = np.array([1.0])
    quotes = np.array([[0.001, 5.0]])  # first quote below the arbitrage floor

    surface = iv_surface(spot, strikes, maturities, quotes, rate)
    assert np.isnan(surface[0, 0])
    assert np.isfinite(surface[0, 1])


def test_shape_mismatch_rejected():
    import pytest

    with pytest.raises(ValueError, match="shape"):
        iv_surface(100.0, np.array([90.0, 110.0]), np.array([1.0]),
                   np.zeros((2, 2)), rate=0.05)
