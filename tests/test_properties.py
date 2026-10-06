"""Property-based tests: invariants that must hold for *any* valid input.

Hypothesis generates random contracts (and shrinks any failure to a minimal
example), which catches the edge cases hand-picked tests miss.
"""

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import bsplus as bs
from bsplus.american import american_price
from bsplus.barrier import barrier_price

spot = st.floats(1.0, 1_000.0)
moneyness = st.floats(0.5, 2.0)  # K / S
expiry = st.floats(1 / 365, 5.0)
rate = st.floats(-0.02, 0.10)
carry = st.floats(0.0, 0.08)
vol = st.floats(0.05, 1.5)
kind = st.sampled_from(["call", "put"])

SETTINGS = settings(max_examples=150, deadline=None)


@SETTINGS
@given(spot, moneyness, expiry, rate, vol, carry)
def test_put_call_parity(S, m, T, r, sigma, q):
    K = S * m
    c = bs.price(S, K, T, r, sigma, "call", q)
    p = bs.price(S, K, T, r, sigma, "put", q)
    assert abs(c - p - (S * np.exp(-q * T) - K * np.exp(-r * T))) < 1e-9 * max(S, K)


@SETTINGS
@given(spot, moneyness, expiry, rate, vol, carry, kind)
def test_price_within_no_arbitrage_bounds(S, m, T, r, sigma, q, typ):
    K = S * m
    v = bs.price(S, K, T, r, sigma, typ, q)
    fwd_intrinsic = bs.price(S, K, T, r, 0.0, typ, q)
    upper = S * np.exp(-q * T) if typ == "call" else K * np.exp(-r * T)
    tol = 1e-10 * max(S, K)
    assert fwd_intrinsic - tol <= v <= upper + tol


@SETTINGS
@given(spot, moneyness, expiry, rate, vol, carry, kind)
def test_price_increases_with_volatility(S, m, T, r, sigma, q, typ):
    K = S * m
    lo = bs.price(S, K, T, r, sigma, typ, q)
    hi = bs.price(S, K, T, r, sigma * 1.1, typ, q)
    assert hi >= lo - 1e-12 * max(S, K)
    g = bs.greeks(S, K, T, r, sigma, typ, q)
    assert g.vega >= 0 and g.gamma >= 0


@SETTINGS
@given(spot, moneyness, expiry, rate, vol, carry, kind)
def test_implied_vol_round_trip(S, m, T, r, sigma, q, typ):
    K = S * m
    p = bs.price(S, K, T, r, sigma, typ, q)
    intrinsic = bs.price(S, K, T, r, 0.0, typ, q)
    # Only prices with measurable time value identify the volatility.
    assume(p - intrinsic > 1e-8 * max(S, K))
    iv = bs.implied_vol(p, S, K, T, r, typ, q)
    assert abs(iv - sigma) < 1e-5 * max(1.0, sigma)


@settings(max_examples=40, deadline=None)
@given(spot, st.floats(0.7, 1.4), st.floats(0.05, 2.0), st.floats(0.0, 0.08),
       st.floats(0.1, 0.8), carry, kind)
def test_american_at_least_european_and_intrinsic(S, m, T, r, sigma, q, typ):
    K = S * m
    res = american_price(S, K, T, r, sigma, typ, q, steps=101)
    euro = bs.price(S, K, T, r, sigma, typ, q)
    intrinsic = max(S - K, 0.0) if typ == "call" else max(K - S, 0.0)
    tol = 2e-3 * S  # tree discretization error
    assert res.price >= euro - tol
    assert res.price >= intrinsic - 1e-9 * S


@SETTINGS
@given(st.floats(50, 150), st.floats(0.6, 1.4), st.floats(0.6, 1.4), st.floats(0.05, 2.0),
       rate, st.floats(0.05, 0.8), carry, kind, st.sampled_from(["down", "up"]))
def test_barrier_in_out_parity(S, m, h, T, r, sigma, q, typ, direction):
    K, H = S * m, S * h
    assume((direction == "down" and H < S * 0.999) or (direction == "up" and H > S * 1.001))
    knock_in = barrier_price(S, K, H, T, r, sigma, typ, f"{direction}-and-in", q)
    knock_out = barrier_price(S, K, H, T, r, sigma, typ, f"{direction}-and-out", q)
    vanilla = bs.price(S, K, T, r, sigma, typ, q)
    assert knock_in >= -1e-12 and knock_out >= -1e-12
    assert abs(knock_in + knock_out - vanilla) < 1e-8 * max(S, 1.0)
