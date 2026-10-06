# Volatility surfaces

Black-Scholes assumes one volatility for every strike and expiry. Real markets quote a
**smile** (a different implied volatility per strike) that changes with maturity. BS+
offers four ways to describe it, trading fit against guarantees.

## SVI: one smile at a time

Raw SVI models the total implied variance $w = \sigma^2 T$ of one expiry as

$$
w(k) = a + b\left(\rho (k - m) + \sqrt{(k - m)^2 + s^2}\right), \qquad k = \ln(K/F).
$$

```python
from bsplus import svi

fit = svi.fit(k, market_vols, T)
fit.params, fit.rmse_vol, fit.arbitrage_free     # butterfly check via Gatheral's g(k)
svi.calendar_crossings([(T1, p1), (T2, p2)], k_grid)  # do slices cross?
```

The vertex $m$ is kept inside the quoted strikes. Unconstrained, the fit can put a sharp
bend just past the last quote, which is invisible in the data and disastrous when
extrapolating.

## SSVI: an arbitrage-free surface

SSVI (Gatheral & Jacquier 2014) ties all expiries together with one ATM variance
$\theta_t$ per expiry and three global parameters:

$$
w(k, t) = \frac{\theta_t}{2}\left(1 + \rho\varphi k + \sqrt{(\varphi k + \rho)^2 + 1 - \rho^2}\right),
\qquad \varphi = \frac{\eta}{\theta_t^{\gamma}(1 + \theta_t)^{1-\gamma}}.
$$

`fit_ssvi` builds the sufficient no-arbitrage conditions (increasing $\theta_t$,
$\eta(1+|\rho|) \le 2$, $0 < \gamma \le \tfrac12$) into its parameterization, so every
surface it returns is free of static arbitrage by construction.

```python
import bsplus as bs

surf = bs.fit_ssvi(k, t, market_vols).surface
surf.implied_vol(k, t), surf.is_arbitrage_free(), surf.no_arbitrage_conditions()
```

## Dupire local volatility

From an arbitrage-free surface, local volatility in total-variance form is

$$
\sigma_{\text{loc}}^2(k, t) = \frac{\partial_t w}{g(k, t)},
$$

where $g$ is the same density function used in the butterfly check.

```python
surf.local_vol(k, t)
from bsplus.surface import local_vol_mc
prices, std_err = local_vol_mc(surf, S, strikes, T)   # reproduces the surface's prices
```

## Heston: a stochastic-volatility model

```python
from bsplus import heston

p = heston.HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
heston.call_price(100, strikes, 1.0, 0.03, p)
cal = heston.calibrate(spot, strikes, maturities, market_vols, r, q=q_per_quote, n_starts=8)
```

Calibration minimizes implied-vol errors. With `n_starts > 1` it runs short fits from
scrambled-Sobol starting points and refines the best one, which guards against local
minima.

## Which one to use

| Need | Use |
|---|---|
| Interpolate quotes of a single expiry | SVI |
| A surface you can extrapolate, or feed into local vol or exotic pricing | SSVI |
| Model dynamics (forward smiles, hedging) | Heston |

The [SPY study](../spy-study.md) compares all three on real data.
