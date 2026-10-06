from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bsplus as bs
from bsplus import market
from bsplus.american import american_price
from bsplus.surface import SSVISurface, fit_ssvi, local_vol_mc

# --- market: synthetic chains with known answers ----------------------------


def _chain(spot, r, q, days, strikes, vol_fn, american=False, half_spread=0.02):
    rows = []
    T = days / 365
    for K in strikes:
        for kind in ("call", "put"):
            vol = vol_fn(np.log(K / (spot * np.exp((r - q) * T))))
            if american:
                mid = american_price(spot, K, T, r, vol, kind, q, steps=151).price
            else:
                mid = bs.price(spot, K, T, r, vol, kind, q)
            rows.append({"expiry": "x", "days": days, "type": kind, "strike": K,
                         "bid": mid - half_spread, "ask": mid + half_spread,
                         "volume": 10.0, "open_interest": 100.0})
    return pd.DataFrame(rows)


def smile(k):
    return 0.2 - 0.25 * k + 0.4 * k * k


def test_parity_regression_recovers_forward_and_discount_on_european_chain():
    spot, r, q, days = 100.0, 0.05, 0.02, 90
    chain = market.clean_chain(_chain(spot, r, q, days, np.arange(80, 121, 2.5), smile))
    T = days / 365
    F, D = market.implied_forward(chain, T)
    assert F == pytest.approx(spot * np.exp((r - q) * T), abs=1e-6)
    assert D == pytest.approx(np.exp(-r * T), abs=1e-9)


def test_american_chain_breaks_regression_but_rate_mode_recovers_forward():
    # The SPY finding, reproduced from first principles: American puts carry
    # an early-exercise premium that grows with strike, so the regression
    # slope implies a discount factor above 1 (a negative rate).
    spot, r, q, days = 100.0, 0.05, 0.015, 180
    T = days / 365
    chain = market.clean_chain(_chain(spot, r, q, days, np.arange(84, 117, 2.0), smile,
                                      american=True, half_spread=0.0005))
    with pytest.raises(ValueError, match="discount factor"):
        market.implied_forward(chain, T)
    true_F = spot * np.exp((r - q) * T)
    F_plain, D = market.implied_forward(chain, T, rate=r)
    assert D == pytest.approx(np.exp(-r * T))
    # The ATM put's early-exercise premium still drags the plain estimate low...
    assert F_plain < true_F * (1 - 1e-3)
    # ...and de-Americanizing removes it.
    F_de, _ = market.implied_forward(chain, T, rate=r, spot=spot, american=True)
    assert F_de == pytest.approx(true_F, rel=2e-4)


def test_american_surface_recovers_the_smile():
    spot, r, q, days = 100.0, 0.05, 0.015, 180
    chain = market.clean_chain(_chain(spot, r, q, days, np.arange(80, 121, 4.0), smile,
                                      american=True, half_spread=0.0005))
    surf = market.build_surface(chain, spot, rate=r, american=True)
    k_true = np.log(surf["strike"] / (spot * np.exp((r - q) * days / 365)))
    np.testing.assert_allclose(surf["iv"], smile(k_true), atol=2e-3)


def test_build_surface_recovers_smile_from_otm_quotes():
    spot, r, q = 100.0, 0.04, 0.01
    chain = pd.concat([_chain(spot, r, q, d, np.arange(70, 131, 2.5), smile, half_spread=0.0)
                       for d in (30, 120)])
    chain["bid"] = chain["bid"].clip(lower=0.06)  # keep far wings two-sided
    chain["ask"] = chain["bid"] + 0.001
    surf = market.build_surface(market.clean_chain(chain, max_rel_spread=1.0), spot)
    assert set(surf["days"]) == {30, 120}
    good = surf[surf["mid"] > 0.2]  # where a 0.0005 mid shift is negligible
    np.testing.assert_allclose(good["iv"], smile(good["k"]), atol=2e-3)
    # OTM rule: puts below the forward, calls above.
    assert (surf.loc[surf["type"] == "put", "k"] < 0).all()
    assert (surf.loc[surf["type"] == "call", "k"] >= 0).all()


def test_clean_chain_drops_bad_quotes():
    df = pd.DataFrame({
        "expiry": "x", "days": 30, "type": "call", "strike": [100, 101, 102, 103, 104],
        "bid": [1.0, 0.0, 1.0, 1.2, 1.0], "ask": [1.1, 0.5, 0.9, 3.0, 1.05],
        "volume": 1.0, "open_interest": [10, 10, 10, 10, 0],
    })
    kept = market.clean_chain(df)
    # no bid / crossed / 86% wide / zero open interest are all removed
    assert kept["strike"].tolist() == [100]
    assert kept["mid"].iloc[0] == pytest.approx(1.05)


def test_bundled_spy_snapshot_builds_a_sane_surface():
    data = Path(__file__).resolve().parents[1] / "data"
    chain = pd.read_csv(data / "spy_chain_2026-10-06.csv")
    surf = market.build_surface(market.clean_chain(chain), 781.3300170898438, rate=0.04035)
    assert surf["days"].nunique() == 8
    assert surf["iv"].between(0.03, 1.0).all()
    carry = surf.groupby("days").apply(
        lambda s: np.log(s["forward"].iloc[0] / 781.33) / s["T"].iloc[0], include_groups=False)
    # Beyond a month, carry = rate - dividend yield should be a few percent.
    assert carry[carry.index >= 30].between(0.0, 0.06).all()


# --- SSVI and local vol ------------------------------------------------------

TRUE = SSVISurface(np.array([0.1, 0.25, 0.5, 1.0, 2.0]),
                   np.array([0.004, 0.011, 0.022, 0.045, 0.09]), -0.6, 1.1, 0.4)


def test_ssvi_fit_recovers_known_surface():
    T = np.repeat(TRUE.expiries, 15)
    k = np.tile(np.linspace(-0.4, 0.3, 15), 5) * np.sqrt(np.maximum(T, 0.25))
    fit = fit_ssvi(k, T, TRUE.implied_vol(k, T))
    assert fit.rmse_vol < 1e-8
    s = fit.surface
    assert (s.rho, s.eta, s.gamma) == pytest.approx((-0.6, 1.1, 0.4), abs=1e-6)
    np.testing.assert_allclose(s.thetas, TRUE.thetas, rtol=1e-6)


def test_ssvi_fit_is_arbitrage_free_even_on_noisy_crossed_data():
    rng = np.random.default_rng(0)
    T = np.repeat([0.25, 0.5, 1.0], 11)
    k = np.tile(np.linspace(-0.3, 0.3, 11), 3)
    vols = 0.25 - 0.2 * k + rng.normal(0, 0.01, k.size)
    vols[T == 1.0] -= 0.08  # long expiry quoted far too low: calendar arbitrage in the data
    fit = fit_ssvi(k, T, vols)
    assert fit.surface.is_arbitrage_free()
    grid = np.linspace(-1, 1, 201)
    for t in fit.surface.expiries:
        assert fit.surface.density_g(grid, t).min() > 0
    w = [fit.surface.total_variance(grid, t) for t in fit.surface.expiries]
    assert np.all(np.diff(np.array(w), axis=0) >= -1e-12)  # no calendar arbitrage


def test_local_vol_of_flat_surface_is_the_flat_vol():
    flat = SSVISurface(np.array([0.5, 1.0]), np.array([0.02, 0.04]), 0.0, 1e-9, 0.5)
    lv = flat.local_vol(np.linspace(-0.5, 0.5, 11), 0.75)
    np.testing.assert_allclose(lv, 0.2, rtol=1e-4)


def test_local_vol_mc_reproduces_surface_prices():
    strikes = np.array([75.0, 90.0, 100.0, 110.0, 125.0])
    T = 0.5
    prices, se = local_vol_mc(TRUE, 100.0, strikes, T, n_paths=120_000, n_steps=120, seed=3)
    exact = bs.price(100.0, strikes, T, 0.0, TRUE.implied_vol(np.log(strikes / 100.0), T))
    z = (prices - exact) / se
    assert np.all(np.abs(z) < 4), z


# --- discrete dividends ----------------------------------------------------------


def test_no_dividends_changes_nothing():
    a = american_price(100, 100, 1.0, 0.05, 0.2, "put")
    b = american_price(100, 100, 1.0, 0.05, 0.2, "put", dividends=[(1.5, 3.0)])  # after expiry
    assert a.price == pytest.approx(b.price)


def test_european_tree_with_dividends_matches_escrowed_closed_form():
    from bsplus.american import european_tree_price

    divs = [(0.25, 1.5), (0.75, 1.5)]
    for kind in ("call", "put"):
        tree = european_tree_price(100, 95, 1.0, 0.04, 0.25, kind, steps=501, dividends=divs)
        closed = bs.black_scholes.price_cash_dividends(100, 95, 1.0, 0.04, 0.25, divs, kind)
        assert tree == pytest.approx(closed, abs=2e-3)


def test_large_dividend_triggers_early_exercise_of_a_call():
    # A $5 dividend just before expiry on a deep ITM call: exercising
    # cum-dividend beats holding, so the American call is worth more.
    res = american_price(100, 80, 0.5, 0.03, 0.2, "call", dividends=[(0.45, 5.0)])
    assert res.early_exercise_premium > 0.5
    assert res.price >= 100 - 80  # at least immediate exercise


def test_dividends_exceeding_spot_raise():
    with pytest.raises(ValueError):
        american_price(10, 10, 1.0, 0.03, 0.2, "call", dividends=[(0.5, 11.0)])
