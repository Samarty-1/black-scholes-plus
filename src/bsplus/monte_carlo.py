"""Monte Carlo pricing under geometric Brownian motion.

Used as an independent check on the closed-form prices and as the engine for
an arithmetic-average Asian option, which has no closed form under
Black-Scholes. Two textbook variance-reduction techniques are applied:

* **antithetic variates** - each normal draw ``z`` is paired with ``-z``;
* **control variates** - the European payoff is regressed on the discounted
  terminal spot (whose expectation ``S e^{-qT}`` is known exactly); the Asian
  payoff is regressed on the *geometric* Asian payoff, which has a closed form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from .black_scholes import _is_call


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float

    def ci95(self) -> tuple[float, float]:
        return self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error


def _control_variate(y, x, x_mean):
    """Optimal control-variate estimator for E[y] given E[x] = x_mean.

    Inputs are antithetic: the second half mirrors the first. The two halves
    are averaged pairwise first, because a path and its mirror are not
    independent and treating them as such misstates the standard error.
    """
    half = y.size // 2
    y = 0.5 * (y[:half] + y[half:])
    x = 0.5 * (x[:half] + x[half:])
    cov = np.cov(y, x)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
    adj = y - beta * (x - x_mean)
    return MCResult(float(adj.mean()), float(adj.std(ddof=1) / np.sqrt(adj.size)))


def european_mc(S, K, T, r, sigma, option_type="call", q=0.0, n_paths=200_000, seed=None):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths // 2)
    z = np.concatenate([z, -z])
    s_t = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    sign = 1.0 if bool(_is_call(option_type)) else -1.0
    disc = np.exp(-r * T)
    payoff = disc * np.maximum(sign * (s_t - K), 0.0)
    return _control_variate(payoff, disc * s_t, S * np.exp(-q * T))


def geometric_asian_price(S, K, T, r, sigma, option_type="call", q=0.0, n_fixings=252):
    """Closed form for a discretely monitored geometric-average Asian option."""
    n = n_fixings
    dt = T / n
    t = dt * np.arange(1, n + 1)
    mu = np.log(S) + (r - q - 0.5 * sigma**2) * t.mean()
    # Var of mean of log prices: sigma^2 * sum_ij min(t_i, t_j) / n^2
    var = sigma**2 * dt * (n + 1) * (2 * n + 1) / (6.0 * n)
    vol = np.sqrt(var)
    d1 = (mu - np.log(K) + var) / vol
    d2 = d1 - vol
    fwd = np.exp(mu + 0.5 * var)
    disc = np.exp(-r * T)
    if bool(_is_call(option_type)):
        return float(disc * (fwd * ndtr(d1) - K * ndtr(d2)))
    return float(disc * (K * ndtr(-d2) - fwd * ndtr(-d1)))


def asian_mc(
    S, K, T, r, sigma, option_type="call", q=0.0, n_fixings=252, n_paths=100_000, seed=None
):
    """Arithmetic-average Asian option with a geometric-Asian control variate."""
    rng = np.random.default_rng(seed)
    half = n_paths // 2
    dt = T / n_fixings
    z = rng.standard_normal((half, n_fixings))
    z = np.concatenate([z, -z])
    log_paths = np.log(S) + np.cumsum(
        (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1
    )
    arith = np.exp(log_paths).mean(axis=1)
    geo = np.exp(log_paths.mean(axis=1))
    sign = 1.0 if bool(_is_call(option_type)) else -1.0
    disc = np.exp(-r * T)
    y = disc * np.maximum(sign * (arith - K), 0.0)
    x = disc * np.maximum(sign * (geo - K), 0.0)
    exact = geometric_asian_price(S, K, T, r, sigma, option_type, q, n_fixings)
    return _control_variate(y, x, exact)
