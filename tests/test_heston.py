import numpy as np
import pytest

import bsplus as bs
from bsplus import heston
from bsplus.heston import HestonParams

BASE = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
STRIKES = np.array([60.0, 80.0, 95.0, 100.0, 105.0, 120.0, 150.0])


@pytest.mark.parametrize("T", [0.05, 0.5, 2.0, 10.0])
@pytest.mark.parametrize("params", [
    BASE,
    HestonParams(0.01, 0.5, 0.09, 1.5, -0.9),   # high vol-of-vol, Feller violated
    HestonParams(0.09, 5.0, 0.02, 0.2, 0.3),    # positive correlation, fast reversion
])
def test_fast_quadrature_matches_adaptive_reference(T, params):
    gl = heston.call_price(100, STRIKES, T, 0.03, params, q=0.01, method="gl")
    ref = heston.call_price(100, STRIKES, T, 0.03, params, q=0.01, method="quad")
    np.testing.assert_allclose(gl, ref, atol=1e-6)


def test_reduces_to_black_scholes_without_vol_of_vol():
    # xi -> 0 and v0 = theta: variance is constant, so Heston == BS.
    p = HestonParams(v0=0.0625, kappa=1.0, theta=0.0625, xi=1e-4, rho=0.0)
    hs = heston.call_price(100, STRIKES, 1.0, 0.05, p)
    np.testing.assert_allclose(hs, bs.price(100, STRIKES, 1.0, 0.05, 0.25, "call"), atol=1e-4)


def test_matches_monte_carlo():
    T, r = 1.0, 0.03
    s_t = heston.simulate_paths(100, T, r, BASE, n_paths=200_000, n_steps=250, seed=7)
    disc = np.exp(-r * T)
    for K in (80.0, 100.0, 120.0):
        payoff = disc * np.maximum(s_t - K, 0.0)
        mc, se = payoff.mean(), payoff.std(ddof=1) / np.sqrt(payoff.size)
        exact = heston.call_price(100, K, T, r, BASE)
        # Allow 4 standard errors plus a small Euler discretization bias.
        assert abs(mc - exact) < 4 * se + 0.03, (K, mc, exact, se)


def test_put_call_parity_and_skew():
    c = heston.price(100, STRIKES, 1.0, 0.03, BASE, "call")
    p = heston.price(100, STRIKES, 1.0, 0.03, BASE, "put")
    np.testing.assert_allclose(c - p, 100 - STRIKES * np.exp(-0.03), atol=1e-8)
    vols = heston.implied_vol_smile(100, STRIKES, 1.0, 0.03, BASE)
    assert np.all(np.diff(vols) < 0)  # rho < 0 produces a downward equity skew


def test_feller_condition():
    assert HestonParams(0.04, 3.0, 0.05, 0.4, -0.5).feller_satisfied()  # 0.30 > 0.16
    assert not BASE.feller_satisfied()  # 2*2*0.04 = 0.16 < 0.25


def test_calibration_recovers_synthetic_surface():
    true = HestonParams(v0=0.05, kappa=1.5, theta=0.06, xi=0.6, rho=-0.65)
    strikes = np.tile(np.array([70, 85, 95, 100, 105, 115, 130.0]), 3)
    mats = np.repeat([0.25, 1.0, 2.0], 7)
    vols = np.concatenate([
        heston.implied_vol_smile(100, strikes[mats == t], t, 0.02, true)
        for t in (0.25, 1.0, 2.0)
    ])
    res = heston.calibrate(100, strikes, mats, vols, r=0.02)
    assert res.success
    assert res.rmse_vol < 1e-4  # fits the surface to within 0.01 vol points
    assert res.params.rho == pytest.approx(true.rho, abs=0.05)
    assert res.params.v0 == pytest.approx(true.v0, abs=0.005)
