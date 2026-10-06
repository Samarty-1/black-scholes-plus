import numpy as np
import pytest

import bsplus as bs


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_round_trip_over_wide_grid(option_type):
    # Includes very deep ITM/OTM strikes, tiny/large vols, short/long expiries
    # - exactly where plain Newton-Raphson fails.
    S, r, q = 100.0, 0.03, 0.01
    K, T, sig = np.meshgrid(
        np.array([20, 50, 80, 95, 100, 105, 125, 200, 400.0]),
        np.array([1 / 365, 0.05, 0.5, 2.0, 10.0]),
        np.array([0.03, 0.1, 0.3, 0.8, 2.0]),
        indexing="ij",
    )
    prices = bs.price(S, K, T, r, sig, option_type, q)
    # Only prices that carry measurable time value can be inverted at all;
    # where the option is worth its intrinsic to machine precision the
    # volatility is genuinely unidentifiable.
    intrinsic = bs.price(S, K, T, r, 0.0, option_type, q)
    identifiable = (prices - intrinsic) > 1e-10 * np.maximum(S, K)
    iv = bs.implied_vol(prices, S, K, T, r, option_type, q)
    assert identifiable.sum() >= 150  # guard against a vacuous test (152 of 225)
    err = np.abs(iv[identifiable] - sig[identifiable])
    assert np.nanmax(err) < 1e-6
    assert not np.any(np.isnan(iv[identifiable]))


def test_arbitrage_violations_return_nan():
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    below_intrinsic = 0.5 * max(S - K * np.exp(-r * T), 0)
    assert np.isnan(bs.implied_vol(below_intrinsic, S, K, T, r, "call"))
    assert np.isnan(bs.implied_vol(S + 1, S, K, T, r, "call"))  # above upper bound
    assert np.isnan(bs.implied_vol(5.0, S, K, 0.0, r, "call"))  # expired


def test_scalar_returns_float():
    iv = bs.implied_vol(10.45, 100, 100, 1.0, 0.05, "call")
    assert isinstance(iv, float)
    assert iv == pytest.approx(0.2, abs=1e-3)


def test_mixed_option_types_vectorized():
    K = np.array([90.0, 100.0, 110.0])
    types = np.array(["put", "call", "put"])
    prices = bs.price(100, K, 0.5, 0.02, 0.27, types)
    np.testing.assert_allclose(bs.implied_vol(prices, 100, K, 0.5, 0.02, types), 0.27, atol=1e-8)
