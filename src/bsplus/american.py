"""American option pricing on binomial trees.

Black-Scholes only prices European options. An American put (or a call on a
dividend-paying asset) can be worth more because it may be exercised early.

``leisen_reimer`` is the default engine: its probabilities come from the
Peizer-Pratt inversion of the Black-Scholes ``d1``/``d2``, so the tree is
centred on the strike and converges with error roughly ``O(1/n^2)`` instead
of the oscillating ``O(1/n)`` of the classic Cox-Ross-Rubinstein (CRR) tree.
A CRR engine is kept as an independent cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import _is_call
from .black_scholes import price as bs_price


@dataclass(frozen=True)
class TreeResult:
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    early_exercise_premium: float


def _peizer_pratt(z, n):
    """Peizer-Pratt method 2 inversion of the normal CDF for an n-step tree."""
    denom = n + 1.0 / 3.0 + 0.1 / (n + 1.0)
    return 0.5 + np.sign(z) * 0.5 * np.sqrt(
        1.0 - np.exp(-((z / denom) ** 2) * (n + 1.0 / 6.0))
    )


def _tree_params(S, K, T, r, sigma, q, steps, method):
    dt = T / steps
    growth = np.exp((r - q) * dt)
    if method == "lr":
        if steps % 2 == 0:
            steps += 1  # LR requires an odd number of steps
            dt = T / steps
            growth = np.exp((r - q) * dt)
        vst = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vst
        d2 = d1 - vst
        p = _peizer_pratt(d2, steps)
        p_bar = _peizer_pratt(d1, steps)
        if not (0.0 < p < 1.0):
            raise ValueError("tree probability outside (0, 1); increase steps")
        u = growth * p_bar / p
        d = (growth - p * u) / (1.0 - p)
    elif method == "crr":
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (growth - d) / (u - d)
    else:
        raise ValueError("method must be 'lr' or 'crr'")
    if not (0.0 < p < 1.0):
        raise ValueError("tree probability outside (0, 1); increase steps")
    return steps, dt, u, d, p


def _pv_dividends(dividends, t, T, r):
    """Value at time ``t`` of the cash dividends paid in ``(t, T]``."""
    return sum(amt * np.exp(-r * (td - t)) for td, amt in dividends if t < td <= T)


def _price_tree(S, K, T, r, sigma, option_type, q, steps, method, american, dividends=()):
    if T <= 0 or sigma <= 0:
        raise ValueError("tree pricing requires T > 0 and sigma > 0")
    is_call = bool(_is_call(option_type))
    # Escrowed-dividend model: the tree is built on the "risky part" of the
    # stock, S* = S - PV(dividends before expiry), so it stays recombining.
    # At each node the actual share price is S* plus the PV of dividends
    # still to be paid, and that is what the exercise decision uses.
    s_star = S - _pv_dividends(dividends, 0.0, T, r)
    if s_star <= 0:
        raise ValueError("dividends exceed the share price")
    steps, dt, u, d, p = _tree_params(s_star, K, T, r, sigma, q, steps, method)
    disc = np.exp(-r * dt)
    sign = 1.0 if is_call else -1.0

    j = np.arange(steps + 1)
    spots = s_star * u**j * d ** (steps - j)
    values = np.maximum(sign * (spots - K), 0.0)

    snapshot = {}
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        j = np.arange(i + 1)
        spots = s_star * u**j * d ** (i - j) + _pv_dividends(dividends, i * dt, T, r)
        if american:
            values = np.maximum(values, sign * (spots - K))
        if i <= 2:
            snapshot[i] = (values.copy(), spots)

    v2, s2 = snapshot[2]
    v1, s1 = snapshot[1]
    v0 = float(snapshot[0][0][0])
    # At vanishing vol the nodes coincide and delta/gamma are undefined (nan);
    # the price is still valid, and is all the implied-vol solver needs.
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = (v1[1] - v1[0]) / (s1[1] - s1[0])
        delta_up = (v2[2] - v2[1]) / (s2[2] - s2[1])
        delta_dn = (v2[1] - v2[0]) / (s2[1] - s2[0])
        gamma = (delta_up - delta_dn) / (0.5 * (s2[2] - s2[0]))
    return v0, float(delta), float(gamma)


def american_price(
    S, K, T, r, sigma, option_type="put", q=0.0, steps=201, method="lr", dividends=()
) -> TreeResult:
    """Price an American option and report tree Greeks.

    Delta and gamma come from the tree's first two time steps. Theta, vega
    and rho are found by re-pricing on the same tree with a bumped input;
    this works well with Leisen-Reimer because its price is smooth in the
    inputs (a CRR price jumps as nodes cross the strike). Units match
    :func:`bsplus.black_scholes.greeks`: per year and per unit of vol/rate.

    ``early_exercise_premium`` is the American price minus the closed-form
    European Black-Scholes price.
    """

    def reprice(**bump):
        args = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "q": q, **bump}
        return _price_tree(
            args["S"], args["K"], args["T"], args["r"], args["sigma"], option_type,
            args["q"], steps, method, american=True, dividends=dividends,
        )[0]

    value, delta, gamma = _price_tree(
        S, K, T, r, sigma, option_type, q, steps, method, american=True, dividends=dividends
    )
    # Theta by re-pricing one day closer to expiry (the LR tree's middle node
    # at step 2 is not exactly at S, so the classic tree theta is biased).
    h_t = min(1.0 / 365.0, 0.5 * T)
    theta = (reprice(T=T - h_t) - value) / h_t
    h_s = min(1e-3, 0.5 * sigma)
    vega = (reprice(sigma=sigma + h_s) - reprice(sigma=sigma - h_s)) / (2 * h_s)
    h_r = 1e-4
    rho = (reprice(r=r + h_r) - reprice(r=r - h_r)) / (2 * h_r)
    s_star = S - _pv_dividends(dividends, 0.0, T, r)
    european = bs_price(s_star, K, T, r, sigma, option_type, q)
    return TreeResult(
        price=value,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
        early_exercise_premium=max(value - european, 0.0),
    )


def european_tree_price(
    S, K, T, r, sigma, option_type="call", q=0.0, steps=201, method="lr", dividends=()
):
    """European price on the same tree (used to measure discretization error)."""
    return _price_tree(
        S, K, T, r, sigma, option_type, q, steps, method, american=False, dividends=dividends
    )[0]


def american_implied_vol(price, S, K, T, r, option_type="put", q=0.0, steps=101,
                         lo=1e-3, hi=5.0):
    """Volatility at which the Leisen-Reimer American price equals ``price``.

    Returns ``nan`` when the price is outside what the model can produce
    (below immediate exercise / intrinsic or above the ``hi`` vol price).
    The American price is increasing in volatility, so Brent's method on the
    bracket ``[lo, hi]`` always converges when a root exists.
    """
    from scipy.optimize import brentq

    sign = 1.0 if bool(_is_call(option_type)) else -1.0

    def f(sig):
        try:
            value = _price_tree(S, K, T, r, sig, option_type, q, steps, "lr", american=True)[0]
        except ValueError:
            # At very low vol the LR probabilities degenerate; use the exact
            # low-vol limit instead: the better of immediate exercise and the
            # European value.
            value = max(sign * (S - K), bs_price(S, K, T, r, sig, option_type, q))
        return value - price

    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0 or f_hi < 0:
        return float("nan")
    return float(brentq(f, lo, hi, xtol=1e-7))
