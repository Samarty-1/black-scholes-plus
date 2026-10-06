"""SVI volatility-smile parameterization (Gatheral 2004).

Raw SVI models the total implied variance ``w = sigma^2 T`` of one expiry as a
function of log-moneyness ``k = ln(K / F)``:

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + s^2))

Five parameters are enough to fit most listed equity smiles closely, and the
form extrapolates linearly in the wings (consistent with Lee's moment
formula). A fitted smile is only usable if it is free of butterfly
arbitrage, i.e. the implied risk-neutral density is non-negative;
``butterfly_g`` evaluates Gatheral's density condition ``g(k) >= 0``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import optimize


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    s: float

    def as_dict(self) -> dict:
        return asdict(self)


def total_variance(k, p: SVIParams):
    k = np.asarray(k, dtype=float)
    x = k - p.m
    return p.a + p.b * (p.rho * x + np.sqrt(x * x + p.s * p.s))


def implied_vol(k, T, p: SVIParams):
    """Implied volatility at log-moneyness ``k`` for an expiry ``T``."""
    w = total_variance(k, p)
    return np.sqrt(np.maximum(w, 0.0) / T)


def butterfly_g(k, p: SVIParams):
    """Gatheral's density function ``g(k)``; negative values mean arbitrage."""
    k = np.asarray(k, dtype=float)
    x = k - p.m
    root = np.sqrt(x * x + p.s * p.s)
    w = total_variance(k, p)
    w1 = p.b * (p.rho + x / root)
    w2 = p.b * p.s * p.s / root**3
    return (1.0 - k * w1 / (2.0 * w)) ** 2 - 0.25 * w1 * w1 * (1.0 / w + 0.25) + 0.5 * w2


@dataclass(frozen=True)
class SVIFit:
    params: SVIParams
    rmse_vol: float
    arbitrage_free: bool
    min_g: float


def fit(k, market_vols, T, weights=None) -> SVIFit:
    """Fit raw SVI to one expiry's implied vols by bounded least squares.

    Runs a small multi-start (several ``m``/``rho`` seeds) because the SVI
    objective has local minima, then reports whether the best fit is free of
    butterfly arbitrage on a wide grid of log-moneyness.
    """
    k = np.asarray(k, dtype=float)
    vols = np.asarray(market_vols, dtype=float)
    w_mkt = vols**2 * T
    wts = np.ones_like(k) if weights is None else np.asarray(weights, dtype=float)

    w_min, w_max = float(w_mkt.min()), float(w_mkt.max())
    k_span = float(max(np.ptp(k), 0.1))
    lb = np.array([-w_max, 1e-6, -0.999, k.min() - k_span, 1e-4])
    ub = np.array([w_max, 10.0, 0.999, k.max() + k_span, 5.0 * k_span])

    def residuals(x):
        p = SVIParams(*x)
        model = implied_vol(k, T, p)
        # Penalise parameter sets where w dips below zero (min of w is
        # a + b s sqrt(1 - rho^2)).
        floor = x[0] + x[1] * x[4] * np.sqrt(1.0 - x[2] ** 2)
        penalty = 10.0 * min(floor, 0.0)
        return np.append(wts * (model - vols), penalty)

    best = None
    for m0 in (0.0, float(k[np.argmin(vols)])):
        for rho0 in (-0.5, 0.0, 0.3):
            x0 = np.clip([0.5 * w_min, 0.1, rho0, m0, 0.1], lb + 1e-9, ub - 1e-9)
            sol = optimize.least_squares(residuals, x0, bounds=(lb, ub))
            if best is None or sol.cost < best.cost:
                best = sol

    p = SVIParams(*map(float, best.x))
    rmse = float(np.sqrt(np.mean((implied_vol(k, T, p) - vols) ** 2)))
    grid = np.linspace(k.min() - 1.0, k.max() + 1.0, 801)
    g = butterfly_g(grid, p)
    return SVIFit(
        params=p, rmse_vol=rmse, arbitrage_free=bool(g.min() >= -1e-10), min_g=float(g.min())
    )
