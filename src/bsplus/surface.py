"""Arbitrage-free volatility surface (SSVI) and Dupire local volatility.

SSVI (Gatheral & Jacquier 2014) writes total implied variance as

    w(k, t) = theta_t / 2 * (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2)),
    phi = phi(theta_t) = eta / (theta_t^gamma (1 + theta_t)^(1 - gamma))

with ``theta_t`` the ATM total variance of each expiry and three parameters
``rho, eta, gamma`` shared by the whole surface. With this power-law
``phi`` the surface is free of static arbitrage when

* ``theta_t`` is non-decreasing in ``t``          (no calendar arbitrage),
* ``eta (1 + |rho|) <= 2`` and ``0 < gamma <= 1/2`` (no butterfly arbitrage).

``fit_ssvi`` builds those conditions into its parameterization, so every
surface it returns satisfies them by construction - unlike per-expiry SVI,
where each slice is fitted on its own and nothing ties the slices together.

From an arbitrage-free surface, Dupire's local volatility in total-variance
form is ``sigma_loc^2(k, t) = dw/dt / g(k, t)``, where ``g`` is the same
density function used for the butterfly check. ``local_vol_mc`` prices
vanillas by simulating that local-vol diffusion, which reproduces the
surface's own prices (a standard consistency check, used in the tests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass(frozen=True)
class SSVISurface:
    expiries: np.ndarray  # year fractions, increasing
    thetas: np.ndarray  # ATM total variance per expiry, non-decreasing
    rho: float
    eta: float
    gamma: float

    # --- core formulas -----------------------------------------------------
    def phi(self, theta):
        theta = np.asarray(theta, dtype=float)
        return self.eta / (theta**self.gamma * (1.0 + theta) ** (1.0 - self.gamma))

    def theta_at(self, t):
        """ATM total variance at any ``t``, linear in ``t`` between expiries.

        Before the first expiry the ATM *volatility* is held flat
        (``theta = theta_1 t / t_1``); after the last, the last slope is
        extended. Both keep ``theta`` increasing.
        """
        t = np.asarray(t, dtype=float)
        T, th = self.expiries, self.thetas
        inside = np.interp(t, T, th)
        before = th[0] * t / T[0]
        if len(T) > 1:
            slope = (th[-1] - th[-2]) / (T[-1] - T[-2])
        else:
            slope = th[0] / T[0]
        after = th[-1] + slope * (t - T[-1])
        return np.where(t < T[0], before, np.where(t > T[-1], after, inside))

    def total_variance(self, k, t):
        k = np.asarray(k, dtype=float)
        theta = self.theta_at(t)
        p = self.phi(theta)
        r = self.rho
        return 0.5 * theta * (1 + r * p * k + np.sqrt((p * k + r) ** 2 + 1 - r * r))

    def implied_vol(self, k, t):
        t = np.asarray(t, dtype=float)
        return np.sqrt(self.total_variance(k, t) / t)

    # --- no-arbitrage diagnostics -----------------------------------------
    def no_arbitrage_conditions(self) -> dict:
        return {
            "theta_non_decreasing": bool(np.all(np.diff(self.thetas) >= -1e-14)),
            "eta(1+|rho|) <= 2": bool(self.eta * (1 + abs(self.rho)) <= 2 + 1e-12),
            "0 < gamma <= 1/2": bool(0 < self.gamma <= 0.5 + 1e-12),
        }

    def is_arbitrage_free(self) -> bool:
        return all(self.no_arbitrage_conditions().values())

    def density_g(self, k, t, h=1e-4):
        """Gatheral's ``g(k)`` (risk-neutral density up to a positive factor)."""
        k = np.asarray(k, dtype=float)
        w = self.total_variance(k, t)
        w_p = self.total_variance(k + h, t)
        w_m = self.total_variance(k - h, t)
        w1 = (w_p - w_m) / (2 * h)
        w2 = (w_p - 2 * w + w_m) / (h * h)
        return (1 - k * w1 / (2 * w)) ** 2 - 0.25 * w1**2 * (1 / w + 0.25) + 0.5 * w2

    # --- local volatility ----------------------------------------------------
    def local_vol(self, k, t, dt=1e-4):
        """Dupire local volatility at log-moneyness ``k = ln(S / F_t)``."""
        t = np.asarray(t, dtype=float)
        dt_ = np.minimum(dt, 0.5 * t)
        dw_dt = (self.total_variance(k, t + dt_) - self.total_variance(k, t - dt_)) / (2 * dt_)
        g = self.density_g(k, t)
        return np.sqrt(np.maximum(dw_dt, 0.0) / np.maximum(g, 1e-12))


def _unpack(x, n):
    """Map unconstrained optimizer variables to an arbitrage-free surface.

    ``theta_1 = exp(x0)``, ``theta_i = theta_{i-1} + exp(x_i)`` (increasing),
    ``rho = tanh(.)``, ``gamma = 0.5 * sigmoid(.)`` and
    ``eta = 2 / (1 + |rho|) * sigmoid(.)`` so the constraints hold exactly.
    """
    thetas = np.cumsum(np.exp(x[:n]))
    rho = float(np.tanh(x[n]))
    gamma = float(0.5 / (1 + np.exp(-x[n + 1])))
    eta = float(2.0 / (1 + abs(rho)) / (1 + np.exp(-x[n + 2])))
    return thetas, rho, eta, gamma


@dataclass(frozen=True)
class SSVIFit:
    surface: SSVISurface
    rmse_vol: float
    rmse_by_expiry: dict


def fit_ssvi(k, t, market_vols, weights=None) -> SSVIFit:
    """Fit SSVI to implied vols given as flat arrays (one entry per quote).

    ``k`` is log-moneyness against each expiry's forward, ``t`` the expiry
    in years. Returns a surface that is arbitrage-free by construction.
    """
    k = np.asarray(k, dtype=float)
    t = np.asarray(t, dtype=float)
    vols = np.asarray(market_vols, dtype=float)
    wts = np.ones_like(vols) if weights is None else np.asarray(weights, dtype=float)
    expiries = np.unique(t)
    n = len(expiries)
    idx = np.searchsorted(expiries, t)

    # Start: ATM total variance per expiry from the quote nearest k = 0,
    # made increasing, then converted to log-increments.
    atm = np.array([vols[idx == i][np.argmin(np.abs(k[idx == i]))] ** 2 * expiries[i]
                    for i in range(n)])
    atm = np.maximum.accumulate(np.maximum(atm, 1e-6))
    incr = np.diff(np.concatenate([[0.0], atm]))
    incr = np.maximum(incr, 1e-3 * atm[-1] / n)

    def residuals(x):
        thetas, rho, eta, gamma = _unpack(x, n)
        surf = SSVISurface(expiries, thetas, rho, eta, gamma)
        theta_q = thetas[idx]
        with np.errstate(all="ignore"):  # extreme trial steps can overflow exp()
            p = surf.phi(theta_q)
            w = 0.5 * theta_q * (1 + rho * p * k + np.sqrt((p * k + rho) ** 2 + 1 - rho * rho))
            res = wts * (np.sqrt(w / t) - vols)
        return np.where(np.isfinite(res), res, 1.0)

    best = None
    for rho0 in (-0.7, -0.3, 0.0):
        for eta_frac in (0.3, 0.7):
            x0 = np.concatenate([np.log(incr), [np.arctanh(rho0), 0.0,
                                                np.log(eta_frac / (1 - eta_frac))]])
            sol = optimize.least_squares(residuals, x0)
            if best is None or sol.cost < best.cost:
                best = sol

    assert best is not None  # the multi-start loop always runs
    thetas, rho, eta, gamma = _unpack(best.x, n)
    surf = SSVISurface(expiries, thetas, rho, eta, gamma)
    err = surf.implied_vol(k, t) - vols
    by_exp = {float(T): float(np.sqrt(np.mean(err[idx == i] ** 2)))
              for i, T in enumerate(expiries)}
    return SSVIFit(surface=surf, rmse_vol=float(np.sqrt(np.mean(err**2))), rmse_by_expiry=by_exp)


def local_vol_mc(surface: SSVISurface, S, strikes, T, r=0.0, q=0.0, n_paths=100_000,
                 n_steps=200, seed=None):
    """European call prices under the surface's Dupire local-vol diffusion.

    Simulates ``d ln S = (r - q - sigma_loc^2 / 2) dt + sigma_loc dW`` with
    ``sigma_loc`` evaluated at ``k = ln(S_t / F_t)``. Returns
    ``(prices, std_errors)`` for each strike (antithetic pairs averaged).
    """
    rng = np.random.default_rng(seed)
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    half = n_paths // 2
    dt = T / n_steps
    x = np.zeros(2 * half)  # ln(S_t / F_t); the forward drift is removed exactly
    for i in range(n_steps):
        t_mid = (i + 0.5) * dt
        sig = surface.local_vol(x, t_mid)
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
        x += -0.5 * sig * sig * dt + sig * np.sqrt(dt) * z
    s_t = S * np.exp((r - q) * T + x)
    disc = np.exp(-r * T)
    pay = disc * np.maximum(s_t[None, :] - strikes[:, None], 0.0)
    pair = 0.5 * (pay[:, :half] + pay[:, half:])
    return pair.mean(axis=1), pair.std(axis=1, ddof=1) / np.sqrt(half)
