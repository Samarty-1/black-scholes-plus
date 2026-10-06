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


def _price_tree(S, K, T, r, sigma, option_type, q, steps, method, american):
    if T <= 0 or sigma <= 0:
        raise ValueError("tree pricing requires T > 0 and sigma > 0")
    is_call = bool(_is_call(option_type))
    steps, dt, u, d, p = _tree_params(S, K, T, r, sigma, q, steps, method)
    disc = np.exp(-r * dt)
    sign = 1.0 if is_call else -1.0

    j = np.arange(steps + 1)
    spots = S * u**j * d ** (steps - j)
    values = np.maximum(sign * (spots - K), 0.0)

    snapshot = {}
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        j = np.arange(i + 1)
        spots = S * u**j * d ** (i - j)
        if american:
            values = np.maximum(values, sign * (spots - K))
        if i <= 2:
            snapshot[i] = (values.copy(), spots)

    v2, s2 = snapshot[2]
    v1, s1 = snapshot[1]
    v0 = float(snapshot[0][0][0])
    delta = (v1[1] - v1[0]) / (s1[1] - s1[0])
    delta_up = (v2[2] - v2[1]) / (s2[2] - s2[1])
    delta_dn = (v2[1] - v2[0]) / (s2[1] - s2[0])
    gamma = (delta_up - delta_dn) / (0.5 * (s2[2] - s2[0]))
    return v0, float(delta), float(gamma)


def american_price(
    S, K, T, r, sigma, option_type="put", q=0.0, steps=201, method="lr"
) -> TreeResult:
    """Price an American option and report tree Greeks.

    ``early_exercise_premium`` is the American price minus the closed-form
    European Black-Scholes price.
    """
    value, delta, gamma = _price_tree(
        S, K, T, r, sigma, option_type, q, steps, method, american=True
    )
    # Theta by re-pricing one day closer to expiry (the LR tree's middle node
    # at step 2 is not exactly at S, so the classic tree theta is biased).
    h = min(1.0 / 365.0, 0.5 * T)
    shorter = _price_tree(S, K, T - h, r, sigma, option_type, q, steps, method, american=True)[0]
    theta = (shorter - value) / h
    european = bs_price(S, K, T, r, sigma, option_type, q)
    return TreeResult(
        price=value,
        delta=delta,
        gamma=gamma,
        theta=theta,
        early_exercise_premium=max(value - european, 0.0),
    )


def european_tree_price(S, K, T, r, sigma, option_type="call", q=0.0, steps=201, method="lr"):
    """European price on the same tree (used to measure discretization error)."""
    return _price_tree(S, K, T, r, sigma, option_type, q, steps, method, american=False)[0]
