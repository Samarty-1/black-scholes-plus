import numpy as np
import pytest

import bsplus as bs
from bsplus import heston, monte_carlo, svi


def test_svi_fits_heston_smile_without_arbitrage():
    p = heston.HestonParams(0.04, 2.0, 0.04, 0.5, -0.7)
    strikes = np.linspace(70, 140, 15)
    T, r = 0.5, 0.02
    vols = heston.implied_vol_smile(100, strikes, T, r, p)
    k = np.log(strikes / (100 * np.exp(r * T)))
    fit = svi.fit(k, vols, T)
    assert fit.rmse_vol < 2e-3
    assert fit.arbitrage_free


def test_svi_recovers_known_parameters():
    true = svi.SVIParams(a=0.02, b=0.12, rho=-0.4, m=0.05, s=0.15)
    k = np.linspace(-0.6, 0.5, 25)
    vols = svi.implied_vol(k, 1.0, true)
    fit = svi.fit(k, vols, 1.0)
    assert fit.rmse_vol < 1e-6
    np.testing.assert_allclose(svi.total_variance(k, fit.params), svi.total_variance(k, true),
                               atol=1e-8)


def test_butterfly_check_flags_arbitrage():
    # Very large b with tiny s produces a negative density near the vertex.
    bad = svi.SVIParams(a=-0.05, b=2.0, rho=-0.95, m=0.0, s=0.01)
    assert svi.butterfly_g(np.linspace(-0.5, 0.5, 501), bad).min() < 0


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_european_mc_within_confidence_interval(option_type):
    exact = bs.price(100, 105, 1.0, 0.04, 0.3, option_type, q=0.01)
    res = monte_carlo.european_mc(100, 105, 1.0, 0.04, 0.3, option_type, q=0.01, seed=1)
    assert abs(res.price - exact) < 4 * res.std_error
    # Plain MC with 200k paths has SE ~0.045 here (measured); the control
    # variate removes the linear-in-S_T part of the payoff.
    naive_se = 0.045
    assert res.std_error < 0.6 * naive_se


def test_asian_mc_bounded_and_variance_reduced():
    geo = monte_carlo.geometric_asian_price(100, 100, 1.0, 0.05, 0.25, n_fixings=52)
    res = monte_carlo.asian_mc(100, 100, 1.0, 0.05, 0.25, n_fixings=52, n_paths=60_000, seed=3)
    # AM-GM: the arithmetic average dominates the geometric one.
    assert res.price > geo
    assert res.price < bs.price(100, 100, 1.0, 0.05, 0.25)  # averaging reduces vol
    assert res.std_error < 0.005  # geometric control variate is very effective


def test_geometric_asian_matches_its_own_mc():
    exact = monte_carlo.geometric_asian_price(100, 95, 0.5, 0.03, 0.2, n_fixings=20)
    rng = np.random.default_rng(11)
    dt = 0.5 / 20
    z = rng.standard_normal((400_000, 20))
    logp = np.log(100) + np.cumsum((0.03 - 0.02) * dt + 0.2 * np.sqrt(dt) * z, axis=1)
    pay = np.exp(-0.03 * 0.5) * np.maximum(np.exp(logp.mean(axis=1)) - 95, 0)
    assert abs(pay.mean() - exact) < 4 * pay.std() / np.sqrt(pay.size)


def test_svi_does_not_bend_past_the_last_quote():
    # Regression: the dashboard's default 90-day slice (seed 42, 0.3 vol-pt
    # noise) used to fit b=1.5, rho=0.97 with the vertex beyond the last
    # quote, so the smile shot up 2 vol points just 5% past the data.
    p = heston.HestonParams(0.2**2, 1.5, 0.24**2, 0.6, -0.65)
    rng = np.random.default_rng(42)
    for days in (30, 90):
        t = days / 365
        strikes = 100 * np.exp(np.linspace(-0.35, 0.25, 13) * np.sqrt(max(t, 0.25)))
        vols = heston.implied_vol_smile(100, strikes, t, 0.04, p) + rng.normal(0, 0.003, 13)
    fwd = 100 * np.exp(0.04 * t)
    fit = svi.fit(np.log(strikes / fwd), vols, t)
    k = np.log(strikes / fwd)
    assert k.min() <= fit.params.m <= k.max()
    beyond = strikes.max() * 1.05
    truth = heston.implied_vol_smile(100, beyond, t, 0.04, p)[0]
    assert abs(svi.implied_vol(np.log(beyond / fwd), t, fit.params) - truth) < 0.005


def test_calendar_crossings_detects_crossed_slices():
    short = svi.SVIParams(a=0.01, b=0.1, rho=-0.5, m=0.0, s=0.1)
    longer = svi.SVIParams(a=0.03, b=0.1, rho=-0.5, m=0.0, s=0.1)
    k = np.linspace(-1, 1, 201)
    assert svi.calendar_crossings([(0.25, short), (1.0, longer)], k)[0][1] == 0.0
    assert svi.calendar_crossings([(0.25, longer), (1.0, short)], k)[0][1] == 1.0
