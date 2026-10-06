import numpy as np
import pytest

import bsplus as bs
from bsplus.black_scholes import put_call_parity_gap

# --- Published reference values -------------------------------------------

def test_hull_textbook_example():
    # Hull, Options Futures & Other Derivatives, Example 15.6
    assert bs.price(42, 40, 0.5, 0.10, 0.20, "call") == pytest.approx(4.76, abs=5e-3)
    assert bs.price(42, 40, 0.5, 0.10, 0.20, "put") == pytest.approx(0.81, abs=5e-3)


def test_black76_haug_example():
    # Haug, Complete Guide to Option Pricing Formulas, Black-76 put example
    assert bs.black76_price(19, 19, 0.75, 0.10, 0.28, "put") == pytest.approx(1.7011, abs=1e-4)


def test_garman_kohlhagen_haug_example():
    # Haug, Garman-Kohlhagen currency call example
    value = bs.garman_kohlhagen_price(1.56, 1.60, 0.5, 0.06, 0.08, 0.12, "call")
    assert value == pytest.approx(0.0291, abs=1e-4)


# --- Structural properties --------------------------------------------------

@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(q):
    S, T, r, sigma = 100.0, 0.75, 0.04, 0.3
    K = np.linspace(50, 150, 21)
    c = bs.price(S, K, T, r, sigma, "call", q)
    p = bs.price(S, K, T, r, sigma, "put", q)
    assert np.max(np.abs(put_call_parity_gap(c, p, S, K, T, r, q))) < 1e-10


def test_vectorized_matches_scalar():
    K = np.array([80.0, 100.0, 120.0])
    types = np.array(["call", "put", "call"])
    vec = bs.price(100, K, 1.0, 0.02, 0.25, types)
    for i in range(3):
        assert vec[i] == pytest.approx(bs.price(100, K[i], 1.0, 0.02, 0.25, types[i]))


def test_degenerate_limits_are_intrinsic():
    # Zero time / zero vol: discounted forward intrinsic value.
    assert bs.price(110, 100, 0.0, 0.05, 0.2, "call") == pytest.approx(10.0)
    assert bs.price(90, 100, 0.0, 0.05, 0.2, "put") == pytest.approx(10.0)
    expected = 100 - 90 * np.exp(-0.05)
    assert bs.price(100, 90, 1.0, 0.05, 0.0, "call") == pytest.approx(expected)


def test_price_bounds_and_monotonicity():
    K = np.linspace(60, 140, 41)
    c = bs.price(100, K, 1.0, 0.03, 0.2, "call")
    assert np.all(np.diff(c) < 0)  # decreasing in strike
    assert np.all(np.diff(c, 2) > 0)  # convex in strike (no butterfly arbitrage)
    assert np.all(c <= 100) and np.all(c >= np.maximum(100 - K * np.exp(-0.03), 0))


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        bs.price(-1, 100, 1, 0.01, 0.2)
    with pytest.raises(ValueError):
        bs.price(100, 100, 1, 0.01, 0.2, "straddle")


# --- Greeks vs. finite differences -----------------------------------------

def _fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("S,K,T,r,sigma,q", [
    (100, 100, 1.0, 0.05, 0.2, 0.0),
    (100, 120, 0.25, 0.01, 0.35, 0.02),
    (50, 40, 2.0, 0.03, 0.15, 0.04),
])
def test_greeks_match_finite_differences(option_type, S, K, T, r, sigma, q):
    g = bs.greeks(S, K, T, r, sigma, option_type, q)
    base = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "option_type": option_type, "q": q}

    def P(**kw):
        return bs.price(**{**base, **kw})

    def D(**kw):
        return bs.greeks(**{**base, **kw}).delta

    def vega(x):
        return bs.greeks(**{**base, "sigma": x}).vega

    assert g.delta == pytest.approx(_fd(lambda x: P(S=x), S, 1e-3), rel=1e-6, abs=1e-8)
    assert g.gamma == pytest.approx(_fd(lambda x: D(S=x), S, 1e-3), rel=1e-5, abs=1e-8)
    assert g.vega == pytest.approx(_fd(lambda x: P(sigma=x), sigma, 1e-5), rel=1e-6)
    assert g.rho == pytest.approx(_fd(lambda x: P(r=x), r, 1e-5), rel=1e-6, abs=1e-8)
    assert g.theta == pytest.approx(-_fd(lambda x: P(T=x), T, 1e-5), rel=1e-5, abs=1e-7)
    assert g.vanna == pytest.approx(_fd(lambda x: D(sigma=x), sigma, 1e-5), rel=1e-5, abs=1e-7)
    assert g.volga == pytest.approx(_fd(vega, sigma, 1e-5), rel=1e-5, abs=1e-6)
    assert g.charm == pytest.approx(-_fd(lambda x: D(T=x), T, 1e-5), rel=1e-5, abs=1e-7)


def test_greeks_degenerate_are_nan_but_price_defined():
    g = bs.greeks(100, 100, 0.0, 0.05, 0.2, "call")
    assert g.price == 0.0
    assert np.isnan(g.delta)


def test_accepts_pandas_string_column_for_option_type():
    import pandas as pd

    types = pd.Series(["call", "put", "CALL"])
    vec = bs.price(100, 100, 1.0, 0.02, 0.2, types.to_numpy())
    assert vec[0] == pytest.approx(vec[2])
    assert vec[1] == pytest.approx(bs.price(100, 100, 1.0, 0.02, 0.2, "put"))
    with pytest.raises(TypeError):
        bs.price(100, 100, 1.0, 0.02, 0.2, [1, 2])
