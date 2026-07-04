"""Black-Scholes closed form: known values, parity, and Greek consistency."""

import math

import pytest

from optionslib import Option, OptionType
from optionslib import black_scholes as bs


def make(option_type=OptionType.CALL, **overrides) -> Option:
    params = dict(
        spot=42.0, strike=40.0, maturity=0.5, rate=0.10, volatility=0.20,
        option_type=option_type,
    )
    params.update(overrides)
    return Option(**params)


class TestKnownValues:
    """Hull, 'Options, Futures, and Other Derivatives', ch. 15 example."""

    def test_hull_call(self):
        assert bs.price(make(OptionType.CALL)) == pytest.approx(4.76, abs=0.01)

    def test_hull_put(self):
        assert bs.price(make(OptionType.PUT)) == pytest.approx(0.81, abs=0.01)


class TestPutCallParity:
    @pytest.mark.parametrize("strike", [30.0, 40.0, 42.0, 55.0])
    @pytest.mark.parametrize("maturity", [0.05, 0.5, 2.0])
    @pytest.mark.parametrize("q", [0.0, 0.03])
    def test_parity(self, strike, maturity, q):
        call = make(OptionType.CALL, strike=strike, maturity=maturity, dividend_yield=q)
        put = make(OptionType.PUT, strike=strike, maturity=maturity, dividend_yield=q)
        lhs = bs.price(call) - bs.price(put)
        rhs = call.spot * math.exp(-q * maturity) - strike * math.exp(-call.rate * maturity)
        assert lhs == pytest.approx(rhs, abs=1e-10)


class TestGreeksAgainstFiniteDifferences:
    """Analytic Greeks must match central-difference bumps of the price."""

    H = 1e-4

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_delta(self, option_type):
        opt = make(option_type)
        up = bs.price(make(option_type, spot=opt.spot + self.H))
        dn = bs.price(make(option_type, spot=opt.spot - self.H))
        assert bs.delta(opt) == pytest.approx((up - dn) / (2 * self.H), abs=1e-6)

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_gamma(self, option_type):
        opt = make(option_type)
        up = bs.price(make(option_type, spot=opt.spot + self.H))
        mid = bs.price(opt)
        dn = bs.price(make(option_type, spot=opt.spot - self.H))
        fd = (up - 2 * mid + dn) / self.H**2
        assert bs.gamma(opt) == pytest.approx(fd, rel=1e-4)

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_vega(self, option_type):
        opt = make(option_type)
        up = bs.price(make(option_type, volatility=opt.volatility + self.H))
        dn = bs.price(make(option_type, volatility=opt.volatility - self.H))
        assert bs.vega(opt) == pytest.approx((up - dn) / (2 * self.H), abs=1e-6)

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_theta(self, option_type):
        opt = make(option_type)
        # theta is dV/dt with t = calendar time, so bumping maturity down
        # by h moves time forward: dV/dt = (V(T-h) - V(T)) / h.
        shorter = bs.price(make(option_type, maturity=opt.maturity - self.H))
        fd = (shorter - bs.price(opt)) / self.H
        assert bs.theta(opt) == pytest.approx(fd, abs=1e-3)

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_rho(self, option_type):
        opt = make(option_type)
        up = bs.price(make(option_type, rate=opt.rate + self.H))
        dn = bs.price(make(option_type, rate=opt.rate - self.H))
        assert bs.rho(opt) == pytest.approx((up - dn) / (2 * self.H), abs=1e-6)


class TestSanity:
    def test_call_delta_bounds(self):
        assert 0.0 < bs.delta(make(OptionType.CALL)) < 1.0

    def test_put_delta_bounds(self):
        assert -1.0 < bs.delta(make(OptionType.PUT)) < 0.0

    def test_gamma_and_vega_positive(self):
        opt = make()
        assert bs.gamma(opt) > 0.0
        assert bs.vega(opt) > 0.0

    def test_deep_itm_call_approaches_forward_intrinsic(self):
        opt = make(OptionType.CALL, strike=1.0)
        forward_intrinsic = opt.spot - opt.strike * math.exp(-opt.rate * opt.maturity)
        assert bs.price(opt) == pytest.approx(forward_intrinsic, abs=1e-6)

    def test_american_style_rejected(self):
        from optionslib import ExerciseStyle

        opt = make(style=ExerciseStyle.AMERICAN)
        with pytest.raises(ValueError, match="European"):
            bs.price(opt)
