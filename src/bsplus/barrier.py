"""Single-barrier options: closed form and Monte Carlo.

Closed form: Merton (1973) / Reiner & Rubinstein (1991), in the notation of
Haug, *The Complete Guide to Option Pricing Formulas* (2007), section 4.17.1,
for continuously monitored barriers under generalized Black-Scholes with a
cash rebate (paid at expiry for knock-ins that never knocked in, and on hit
for knock-outs).

Monte Carlo: discretely simulated paths with a **Brownian-bridge** correction:
between two simulated points the probability that the continuous path
crossed the barrier is known exactly, so continuous monitoring is priced
without the large bias of only checking the barrier at the grid points.
The MC engine is used to validate the closed forms independently.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr as N

from .black_scholes import _is_call
from .black_scholes import price as bs_price

KINDS = ("down-and-in", "up-and-in", "down-and-out", "up-and-out")


def barrier_price(S, K, H, T, r, sigma, option_type="call", kind="down-and-out",
                  q=0.0, rebate=0.0):
    """Closed-form price of a continuously monitored single-barrier option.

    ``kind`` is one of :data:`KINDS`. If the barrier is already breached at
    inception, an "in" option is the vanilla and an "out" option is worth the
    rebate.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if min(S, K, H, T, sigma) <= 0:
        raise ValueError("S, K, H, T and sigma must be positive")
    is_call = bool(_is_call(option_type))
    down = kind.startswith("down")
    knock_in = kind.endswith("-in")

    if (down and S <= H) or (not down and S >= H):
        if knock_in:
            return float(bs_price(S, K, T, r, sigma, option_type, q))
        return float(rebate)

    b = r - q
    vst = sigma * np.sqrt(T)
    mu = (b - 0.5 * sigma**2) / sigma**2
    lam = np.sqrt(mu**2 + 2.0 * r / sigma**2)
    x1 = np.log(S / K) / vst + (1 + mu) * vst
    x2 = np.log(S / H) / vst + (1 + mu) * vst
    y1 = np.log(H**2 / (S * K)) / vst + (1 + mu) * vst
    y2 = np.log(H / S) / vst + (1 + mu) * vst
    z = np.log(H / S) / vst + lam * vst

    eta = 1.0 if down else -1.0
    phi = 1.0 if is_call else -1.0
    carry = S * np.exp((b - r) * T)
    df = np.exp(-r * T)
    hs = H / S

    A = phi * carry * N(phi * x1) - phi * K * df * N(phi * x1 - phi * vst)
    B = phi * carry * N(phi * x2) - phi * K * df * N(phi * x2 - phi * vst)
    C = (phi * carry * hs ** (2 * (mu + 1)) * N(eta * y1)
         - phi * K * df * hs ** (2 * mu) * N(eta * y1 - eta * vst))
    D = (phi * carry * hs ** (2 * (mu + 1)) * N(eta * y2)
         - phi * K * df * hs ** (2 * mu) * N(eta * y2 - eta * vst))
    E = rebate * df * (N(eta * x2 - eta * vst) - hs ** (2 * mu) * N(eta * y2 - eta * vst))
    F = rebate * (hs ** (mu + lam) * N(eta * z)
                  + hs ** (mu - lam) * N(eta * z - 2 * eta * lam * vst))

    high_strike = K > H
    table = {
        # (kind, is_call, K > H): formula
        ("down-and-in", True, True): C + E,
        ("down-and-in", True, False): A - B + D + E,
        ("up-and-in", True, True): A + E,
        ("up-and-in", True, False): B - C + D + E,
        ("down-and-in", False, True): B - C + D + E,
        ("down-and-in", False, False): A + E,
        ("up-and-in", False, True): A - B + D + E,
        ("up-and-in", False, False): C + E,
        ("down-and-out", True, True): A - C + F,
        ("down-and-out", True, False): B - D + F,
        ("up-and-out", True, True): F,
        ("up-and-out", True, False): A - B + C - D + F,
        ("down-and-out", False, True): A - B + C - D + F,
        ("down-and-out", False, False): F,
        ("up-and-out", False, True): B - D + F,
        ("up-and-out", False, False): A - C + F,
    }
    return float(max(table[(kind, is_call, high_strike)], 0.0))


def barrier_mc(S, K, H, T, r, sigma, option_type="call", kind="down-and-out", q=0.0,
               n_paths=200_000, n_steps=250, seed=None):
    """Monte Carlo barrier price (no rebate) with a Brownian-bridge hit test.

    Returns ``(price, std_error)``.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    rng = np.random.default_rng(seed)
    is_call = bool(_is_call(option_type))
    down = kind.startswith("down")
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    log_h = np.log(H)

    x = np.full(n_paths, np.log(S))
    hit = (x <= log_h) if down else (x >= log_h)
    for _ in range(n_steps):
        x_new = x + drift + vol * rng.standard_normal(n_paths)
        # P(continuous path crossed H between two points that are both on
        # the safe side) = exp(-2 (x - h)(x_new - h) / (sigma^2 dt)).
        dist_old = (x - log_h) if down else (log_h - x)
        dist_new = (x_new - log_h) if down else (log_h - x_new)
        crossed = (dist_new <= 0) | (
            rng.random(n_paths) < np.exp(-2.0 * np.maximum(dist_old, 0) * np.maximum(dist_new, 0)
                                         / (sigma**2 * dt))
        )
        hit |= crossed
        x = x_new
    s_t = np.exp(x)
    payoff = np.maximum((s_t - K) if is_call else (K - s_t), 0.0)
    alive = hit if kind.endswith("-in") else ~hit
    disc_pay = np.exp(-r * T) * payoff * alive
    return float(disc_pay.mean()), float(disc_pay.std(ddof=1) / np.sqrt(n_paths))
