"""Generalized Black-Scholes-Merton pricing and analytic Greeks.

All functions are vectorized: every numeric argument may be a scalar or a
NumPy array, and arrays broadcast together.

The model uses a continuous yield ``q`` so a single code path covers:

* stocks paying a continuous dividend yield  (``q`` = dividend yield)
* options on futures, Black (1976)           (``q = r``, ``S`` = futures price)
* FX options, Garman-Kohlhagen (1983)        (``q`` = foreign risk-free rate)

Conventions
-----------
* ``T`` is in years, rates and volatility are annualized and continuously
  compounded.
* ``theta`` is the derivative with respect to calendar time (``-dV/dT``) and is
  quoted **per year**; divide by 365 for a per-day figure.
* ``vega`` and ``rho`` are per unit change (1.00 = 100 vol points / 100% rate);
  divide by 100 for the per-1% figure traders usually quote.
* ``charm`` is ``-dDelta/dT`` per year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _pdf(x):
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _is_call(option_type):
    """Map 'call'/'put' (or arrays of them) to a boolean array."""
    arr = np.asarray(option_type)
    if arr.dtype.kind in "US":
        lowered = np.char.lower(arr.astype(str))
        valid = np.isin(lowered, ["call", "put", "c", "p"])
        if not np.all(valid):
            raise ValueError("option_type must be 'call' or 'put'")
        return np.isin(lowered, ["call", "c"])
    raise TypeError("option_type must be a string or array of strings")


def _d1_d2(S, K, T, r, sigma, q):
    vol_sqrt_t = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2, vol_sqrt_t


def _prepare(S, K, T, r, sigma, q):
    S, K, T, r, sigma, q = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    )
    if np.any(S <= 0) or np.any(K <= 0):
        raise ValueError("S and K must be strictly positive")
    if np.any(T < 0) or np.any(sigma < 0):
        raise ValueError("T and sigma must be non-negative")
    return S, K, T, r, sigma, q


def _ret(x):
    """Return a Python float for 0-d results, otherwise the array."""
    return float(x) if np.ndim(x) == 0 else x


def price(S, K, T, r, sigma, option_type="call", q=0.0):
    """European option price under generalized Black-Scholes-Merton.

    When ``sigma * sqrt(T) == 0`` the price collapses to the discounted
    intrinsic value of the forward, which is the exact limit of the formula.
    """
    S, K, T, r, sigma, q = _prepare(S, K, T, r, sigma, q)
    call = np.broadcast_to(_is_call(option_type), S.shape)
    d1, d2, vst = _d1_d2(S, K, T, r, sigma, q)

    df_q = np.exp(-q * T)
    df_r = np.exp(-r * T)
    sign = np.where(call, 1.0, -1.0)

    degenerate = vst <= 0
    d1 = np.where(degenerate, 0.0, d1)
    d2 = np.where(degenerate, 0.0, d2)
    value = sign * (S * df_q * ndtr(sign * d1) - K * df_r * ndtr(sign * d2))
    intrinsic = np.maximum(sign * (S * df_q - K * df_r), 0.0)
    return _ret(np.where(degenerate, intrinsic, value))


@dataclass(frozen=True)
class Greeks:
    """Container for a full set of Black-Scholes sensitivities."""

    price: float | np.ndarray
    delta: float | np.ndarray
    gamma: float | np.ndarray
    vega: float | np.ndarray
    theta: float | np.ndarray
    rho: float | np.ndarray
    vanna: float | np.ndarray
    volga: float | np.ndarray
    charm: float | np.ndarray

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def greeks(S, K, T, r, sigma, option_type="call", q=0.0) -> Greeks:
    """Price plus first- and second-order analytic Greeks.

    Greeks are undefined when ``sigma * sqrt(T) == 0``; those entries are NaN
    (the price is still returned).
    """
    S, K, T, r, sigma, q = _prepare(S, K, T, r, sigma, q)
    call = np.broadcast_to(_is_call(option_type), S.shape)
    sign = np.where(call, 1.0, -1.0)
    d1, d2, vst = _d1_d2(S, K, T, r, sigma, q)
    degenerate = vst <= 0
    vst = np.where(degenerate, np.nan, vst)

    sqrt_t = np.sqrt(T)
    df_q = np.exp(-q * T)
    df_r = np.exp(-r * T)
    n_d1 = _pdf(d1)
    N_sd1 = ndtr(sign * d1)
    N_sd2 = ndtr(sign * d2)

    p = price(S, K, T, r, sigma, option_type, q)
    delta = sign * df_q * N_sd1
    gamma = df_q * n_d1 / (S * vst)
    vega = S * df_q * n_d1 * sqrt_t
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = (
            -S * df_q * n_d1 * sigma / (2.0 * sqrt_t)
            - sign * r * K * df_r * N_sd2
            + sign * q * S * df_q * N_sd1
        )
        vanna = -df_q * n_d1 * d2 / sigma
        volga = vega * d1 * d2 / sigma
        charm = sign * q * df_q * N_sd1 - df_q * n_d1 * (
            2.0 * (r - q) * T - d2 * vst
        ) / (2.0 * T * vst)
    rho = sign * K * T * df_r * N_sd2

    out = {
        "price": p,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
        "vanna": vanna,
        "volga": volga,
        "charm": charm,
    }
    out = {
        key: value if key == "price" else _ret(np.where(degenerate, np.nan, value))
        for key, value in out.items()
    }
    return Greeks(**out)


def black76_price(F, K, T, r, sigma, option_type="call"):
    """Black (1976) price of a European option on a futures/forward ``F``."""
    return price(F, K, T, r, sigma, option_type, q=r)


def garman_kohlhagen_price(S, K, T, r_dom, r_for, sigma, option_type="call"):
    """Garman-Kohlhagen price of a European FX option (spot quoted dom/for)."""
    return price(S, K, T, r_dom, sigma, option_type, q=r_for)


def put_call_parity_gap(call_price, put_price, S, K, T, r, q=0.0):
    """``C - P - (S e^{-qT} - K e^{-rT})``; zero for arbitrage-free quotes."""
    return call_price - put_price - (S * np.exp(-q * T) - K * np.exp(-r * T))
