"""Heston (1993) stochastic-volatility model.

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW1
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW2,   d<W1, W2> = rho dt

Black-Scholes assumes a single constant volatility, so it cannot reproduce
the volatility smile/skew seen in real option markets. Heston lets variance
mean-revert randomly and correlate with spot, which generates a skew
(``rho < 0`` gives the familiar equity skew) and fat tails (``xi > 0``).

Pricing uses the characteristic function in the "little trap" form of
Albrecher et al. (2007), which avoids the complex-log branch-cut
discontinuity of Heston's original formula at long maturities, and a single
Gil-Pelaez style integral evaluated by Gauss-Legendre quadrature (vectorized
over strikes), with ``scipy.integrate.quad`` available as a reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import integrate, optimize

from .implied_vol import implied_vol


@dataclass(frozen=True)
class HestonParams:
    v0: float  # initial variance
    kappa: float  # mean-reversion speed
    theta: float  # long-run variance
    xi: float  # volatility of variance ("vol of vol")
    rho: float  # spot/variance correlation

    def feller_satisfied(self) -> bool:
        """``2 kappa theta > xi^2`` keeps variance strictly positive."""
        return 2.0 * self.kappa * self.theta > self.xi**2

    def as_dict(self) -> dict:
        return asdict(self)


def char_func(u, S, T, r, q, p: HestonParams):
    """Characteristic function of ``ln S_T``: ``E[exp(i u ln S_T)]``."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    xi2 = p.xi * p.xi
    beta = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(beta * beta + xi2 * (u * u + iu))
    g = (beta - d) / (beta + d)
    exp_dt = np.exp(-d * T)
    C = (p.kappa * p.theta / xi2) * (
        (beta - d) * T - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    D = ((beta - d) / xi2) * (1.0 - exp_dt) / (1.0 - g * exp_dt)
    return np.exp(iu * (np.log(S) + (r - q) * T) + C + D * p.v0)


def _integrand(u, logK, K, S, T, r, q, p):
    """Re[e^{-iu lnK} (phi(u - i) - K phi(u)) / (iu)] for the call integral."""
    phi_shift = char_func(u - 1j, S, T, r, q, p)
    phi = char_func(u, S, T, r, q, p)
    return np.real(np.exp(-1j * u * logK) * (phi_shift - K * phi) / (1j * u))


# Gauss-Legendre nodes on [0, 1], reused for every panel.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)
_GL_X = 0.5 * (_GL_X + 1.0)
_GL_W = 0.5 * _GL_W
_PANEL_WIDTH = 25.0


def _upper_limit(T, p):
    """Truncation point for the Fourier integral.

    For small vol-of-vol the integrand decays like a Gaussian,
    ``exp(-var * u^2 / 2)``. For large ``u`` it decays only exponentially,
    ``exp(-c u)`` with ``c = sqrt(1 - rho^2) (v0 + kappa theta T) / xi``, which
    is slow when ``xi`` is large or ``|rho|`` is near one. The log-magnitude
    grows quadratically and then linearly, so the cutoff must satisfy both
    bounds (``exp(-36)`` ~ 2e-16 relative truncation error).
    """
    var = max(min(p.v0, p.theta), 1e-4) * T
    gaussian = np.sqrt(72.0 / var)
    c = np.sqrt(max(1.0 - p.rho**2, 1e-6)) * (p.v0 + p.kappa * p.theta * T) / p.xi
    exponential = 36.0 / max(c, 1e-6)
    return float(np.clip(max(gaussian, exponential), 50.0, 50_000.0))


def _quadrature_grid(U):
    """Composite Gauss-Legendre nodes/weights on [0, U].

    Geometric panels near zero (where the integrand varies fastest relative
    to ``u``) then fixed-width panels, so the oscillating tail is resolved.
    """
    head = np.concatenate([[0.0], np.geomspace(0.25, min(U, _PANEL_WIDTH), 6)])
    tail = np.arange(_PANEL_WIDTH, U, _PANEL_WIDTH)[1:]
    edges = np.unique(np.concatenate([head, tail, [U]]))
    a, b = edges[:-1, None], edges[1:, None]
    return (a + (b - a) * _GL_X).ravel(), ((b - a) * _GL_W).ravel()


def call_price(S, K, T, r, p: HestonParams, q=0.0, method="gl"):
    """Heston European call price for one or many strikes.

    ``method='gl'`` (default) is vectorized Gauss-Legendre quadrature;
    ``method='quad'`` uses adaptive quadrature per strike (slower, reference).
    """
    if T <= 0:
        raise ValueError("T must be positive")
    K = np.atleast_1d(np.asarray(K, dtype=float))
    logK = np.log(K)
    disc = np.exp(-r * T)
    base = 0.5 * (S * np.exp(-q * T) - K * disc)

    if method == "quad":
        integral = np.array(
            [
                integrate.quad(
                    _integrand, 0.0, np.inf, args=(lk, k, S, T, r, q, p), limit=500
                )[0]
                for lk, k in zip(logK, K, strict=True)
            ]
        )
    elif method == "gl":
        nodes, weights = _quadrature_grid(_upper_limit(T, p))
        vals = _integrand(nodes[None, :], logK[:, None], K[:, None], S, T, r, q, p)
        integral = vals @ weights
    else:
        raise ValueError("method must be 'gl' or 'quad'")

    prices = base + disc * integral / np.pi
    # Clip tiny negative values from numerical integration noise.
    lower = np.maximum(S * np.exp(-q * T) - K * disc, 0.0)
    prices = np.maximum(prices, lower)
    return prices if prices.size > 1 else float(prices[0])


def price(S, K, T, r, p: HestonParams, option_type="call", q=0.0, method="gl"):
    """Heston European price; puts come from put-call parity."""
    c = np.asarray(call_price(S, K, T, r, p, q, method))
    if option_type.lower() in ("call", "c"):
        out = c
    elif option_type.lower() in ("put", "p"):
        out = c - S * np.exp(-q * T) + np.asarray(K, dtype=float) * np.exp(-r * T)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return float(out) if out.ndim == 0 else out


def implied_vol_smile(S, strikes, T, r, p: HestonParams, q=0.0):
    """Black-Scholes implied vols of Heston prices (OTM side for accuracy)."""
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    calls = np.atleast_1d(call_price(S, strikes, T, r, p, q))
    return implied_vol(calls, S, strikes, T, r, "call", q)


def simulate_paths(S, T, r, p: HestonParams, q=0.0, n_paths=50_000, n_steps=200, seed=None):
    """Monte Carlo paths with full-truncation Euler (Lord et al. 2010).

    Returns terminal spot values (antithetic pairs included). Full truncation
    (using ``max(v, 0)`` in drift and diffusion) is the least biased simple
    Euler scheme for the CIR variance process.
    """
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    log_s = np.full(2 * half, np.log(S))
    v = np.full(2 * half, p.v0)
    rho_c = np.sqrt(1.0 - p.rho**2)
    for _ in range(n_steps):
        z1 = rng.standard_normal(half)
        z2 = rng.standard_normal(half)
        z1 = np.concatenate([z1, -z1])
        z2 = np.concatenate([z2, -z2])
        w_s = z1
        w_v = p.rho * z1 + rho_c * z2
        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)
        log_s += (r - q - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * w_s
        v += p.kappa * (p.theta - v_pos) * dt + p.xi * sqrt_v * sqrt_dt * w_v
    return np.exp(log_s)


@dataclass(frozen=True)
class CalibrationResult:
    params: HestonParams
    rmse_vol: float  # root-mean-square implied-vol error
    success: bool
    message: str
    model_vols: np.ndarray


def calibrate(
    S,
    strikes,
    maturities,
    market_vols,
    r,
    q=0.0,
    x0: HestonParams | None = None,
):
    """Fit Heston parameters to a set of market implied volatilities.

    Minimizes implied-volatility errors (rather than price errors) so that
    cheap OTM wings carry as much weight as expensive ITM options. Quotes are
    grouped by maturity so each slice is priced with one vectorized call.
    """
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    market_vols = np.asarray(market_vols, dtype=float)
    if not (strikes.shape == maturities.shape == market_vols.shape):
        raise ValueError("strikes, maturities and market_vols must share a shape")

    atm = float(np.median(market_vols))
    if x0 is None:
        x0 = HestonParams(v0=atm**2, kappa=2.0, theta=atm**2, xi=0.5, rho=-0.5)
    lb = np.array([1e-4, 0.05, 1e-4, 0.01, -0.99])
    ub = np.array([2.0, 15.0, 2.0, 3.0, 0.99])
    unique_t = np.unique(maturities)

    def model_vols(x):
        p = HestonParams(*x)
        out = np.empty_like(market_vols)
        for t in unique_t:
            m = maturities == t
            out[m] = implied_vol_smile(S, strikes[m], t, r, p, q)
        return out

    def residuals(x):
        mv = model_vols(x)
        res = mv - market_vols
        # A failed inversion means the price left the arbitrage bounds - penalise.
        return np.where(np.isfinite(res), res, 1.0)

    sol = optimize.least_squares(
        residuals,
        x0=np.clip(np.array(list(x0.as_dict().values())), lb, ub),
        bounds=(lb, ub),
        x_scale="jac",
        max_nfev=400,
    )
    fitted = model_vols(sol.x)
    rmse = float(np.sqrt(np.nanmean((fitted - market_vols) ** 2)))
    return CalibrationResult(
        params=HestonParams(*map(float, sol.x)),
        rmse_vol=rmse,
        success=bool(sol.success),
        message=str(sol.message),
        model_vols=fitted,
    )
