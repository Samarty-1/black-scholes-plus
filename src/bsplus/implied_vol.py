"""Robust, vectorized Black-Scholes implied volatility.

Plain Newton-Raphson (the textbook approach) diverges for deep in/out-of-the-
money options because vega collapses towards zero away from the money. This
solver keeps a bracket ``[lo, hi]`` that always contains the root and falls
back to bisection whenever a Newton step would leave it or vega is too small,
so it converges for every price that is inside the no-arbitrage bounds.

Three further details make it robust:

* **Out-of-the-money reduction** - an in-the-money quote is converted to the
  equivalent OTM quote via put-call parity. The OTM price carries all of the
  time value, so the root-finding problem is better conditioned.
* **Manaster-Koehler start** - Newton starts at
  ``sigma0 = sqrt(2 |ln(F/K)| / T)``, the inflection point of the price in
  volatility, from which Newton iterates move monotonically to the root.
* **Bounds check** - quotes outside ``(intrinsic, upper bound)`` have no
  implied volatility and return ``NaN`` instead of a misleading number.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .black_scholes import _is_call, _pdf, _ret

_SIGMA_MIN = 1e-8
_SIGMA_MAX = 20.0


def _otm_call_price_and_vega(F, K, T, sigma, disc, call):
    """Undiscounted-forward Black price/vega for vectors (no validation)."""
    vst = sigma * np.sqrt(T)
    d1 = np.log(F / K) / vst + 0.5 * vst
    d2 = d1 - vst
    sign = np.where(call, 1.0, -1.0)
    value = disc * sign * (F * ndtr(sign * d1) - K * ndtr(sign * d2))
    vega = disc * F * _pdf(d1) * np.sqrt(T)
    return value, vega


def implied_vol(
    price,
    S,
    K,
    T,
    r,
    option_type="call",
    q=0.0,
    tol=1e-12,
    max_iter=100,
):
    """Implied volatility of European option prices.

    Parameters
    ----------
    price : float or array
        Observed option premium(s).
    S, K, T, r, q : float or array
        Spot, strike, maturity (years), risk-free rate, carry yield.
    option_type : 'call' / 'put' or array of them.
    tol : float
        Relative tolerance on the out-of-the-money option price.

    Returns
    -------
    float or ndarray
        Implied volatility; ``NaN`` where the price violates no-arbitrage
        bounds or ``T <= 0``.
    """
    price, S, K, T, r, q, call = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (price, S, K, T, r, q)), _is_call(option_type)
    )

    disc = np.exp(-r * T)
    F = S * np.exp((r - q) * T)
    intrinsic_fwd = disc * np.where(call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    upper = np.where(call, disc * F, disc * K)

    # Convert to the OTM quote at the same strike: put-call parity says the
    # call and put share the same time value, so subtracting the forward
    # intrinsic gives the OTM option's price whichever type was quoted.
    otm_call = F <= K  # the call is the OTM option when strike >= forward
    otm_price = price - intrinsic_fwd

    valid = (T > 0) & (price < upper) & (otm_price > 0) & np.isfinite(price)
    result = np.full(price.shape, np.nan)
    if not np.any(valid):
        return _ret(result)

    Fv, Kv, Tv = F[valid], K[valid], T[valid]
    dv, target, cv = disc[valid], otm_price[valid], otm_call[valid]

    # Errors are measured relative to the OTM price itself, so an option with
    # 1e-7 of time value is solved as accurately as an at-the-money one.
    scale = target
    target_s = np.ones_like(target)

    lo = np.full(Fv.shape, _SIGMA_MIN)
    hi = np.full(Fv.shape, _SIGMA_MAX)
    sigma = np.sqrt(2.0 * np.abs(np.log(Fv / Kv)) / Tv)
    sigma = np.clip(np.where(sigma < 0.05, 0.3, sigma), 0.01, 5.0)

    active = np.ones(Fv.shape, dtype=bool)
    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        s = sigma[idx]
        val, vega = _otm_call_price_and_vega(Fv[idx], Kv[idx], Tv[idx], s, dv[idx], cv[idx])
        err = val / scale[idx] - target_s[idx]
        vega_s = vega / scale[idx]

        done = np.abs(err) < tol
        # Price is increasing in sigma: shrink the bracket around the root.
        hi[idx] = np.where(err > 0, s, hi[idx])
        lo[idx] = np.where(err <= 0, s, lo[idx])

        with np.errstate(divide="ignore", invalid="ignore"):
            newton = s - err / vega_s
        inside = (newton > lo[idx]) & (newton < hi[idx]) & np.isfinite(newton)
        bisect = 0.5 * (lo[idx] + hi[idx])
        new_s = np.where(inside, newton, bisect)
        new_s = np.where(done, s, new_s)

        narrow = (hi[idx] - lo[idx]) < 1e-14 * np.maximum(1.0, hi[idx])
        sigma[idx] = new_s
        active[idx] = ~(done | narrow)

    out = sigma
    # Roots pinned at the search limits are not genuine solutions.
    out = np.where((out <= _SIGMA_MIN * 1.0001) | (out >= _SIGMA_MAX * 0.9999), np.nan, out)
    result[valid] = out
    return _ret(result)
