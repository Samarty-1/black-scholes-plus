# BS+ — Black-Scholes and beyond

[![tests](https://github.com/Samarty-1/black-scholes-plus/actions/workflows/ci.yml/badge.svg)](https://github.com/Samarty-1/black-scholes-plus/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A compact, **tested** option-pricing library in Python (NumPy + SciPy only) with an
interactive Streamlit dashboard. It starts from Black-Scholes and then fixes the
things a plain Black-Scholes implementation gets wrong or leaves out.

| Plain Black-Scholes | BS+ |
|---|---|
| Non-dividend stock only | Generalized BSM with carry yield `q`: dividends, **Black-76** (futures), **Garman-Kohlhagen** (FX) |
| Delta, gamma, vega, theta, rho | Plus second-order **vanna, volga, charm**, every Greek checked against finite differences |
| Newton-Raphson implied vol that diverges in the wings | **Bracketed Newton + bisection** solver that converges for every arbitrage-free price and returns `NaN` for the rest |
| European exercise only | **American** options on a Leisen-Reimer tree (error ~1/n² vs CRR's oscillating ~1/n) |
| One flat volatility | **Heston** stochastic volatility (pricing, Monte Carlo, calibration) and **SVI** smile fitting with a butterfly-arbitrage check |
| Closed form only | Variance-reduced **Monte Carlo** (antithetic + control variates), incl. arithmetic **Asian** options |

## Quick start

```bash
git clone https://github.com/Samarty-1/black-scholes-plus.git
cd black-scholes-plus
pip install -e ".[app,dev]"
pytest                          # 55 tests, ~5 s
streamlit run app/dashboard.py  # dashboard at http://localhost:8501
```

## Library tour

```python
import numpy as np
import bsplus as bs
from bsplus import heston, svi

# Black-Scholes price and a full set of Greeks (everything is vectorized)
bs.price(42, 40, 0.5, 0.10, 0.20, "call")            # 4.7594  (Hull's textbook example)
g = bs.greeks(100, 100, 0.25, 0.04, 0.25, "put", q=0.01)
g.delta, g.gamma, g.vega / 100, g.theta / 365, g.vanna

# Same engine, other markets
bs.black76_price(19, 19, 0.75, 0.10, 0.28, "put")    # 1.7011  (Haug, futures option)
bs.garman_kohlhagen_price(1.56, 1.60, 0.5, 0.06, 0.08, 0.12, "call")  # 0.0291 (FX)

# Implied vol that does not blow up in the wings
strikes = np.array([20, 80, 100, 125, 400.0])
prices = bs.price(100, strikes, 0.5, 0.03, 0.35, "call")
bs.implied_vol(prices, 100, strikes, 0.5, 0.03, "call")
# -> [0.3500011, 0.35, 0.35, 0.35, 0.35]; the K=20 call is ~$80 of intrinsic and
#    ~1e-12 of time value, so float64 rounding of the price itself limits precision
bs.implied_vol(150.0, 100, 100, 1.0, 0.03, "call")        # nan: above the upper bound

# American put with early-exercise premium
res = bs.american_price(100, 110, 1.0, 0.08, 0.2, "put")
res.price, res.early_exercise_premium, res.delta

# Heston: a smile that Black-Scholes cannot produce
p = heston.HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
heston.implied_vol_smile(100, [80, 100, 120], 1.0, 0.03, p)  # downward equity skew

# Calibrate Heston to a vol surface (swap in your own market quotes)
Ks = np.tile(np.linspace(75, 125, 9), 2)
Ts = np.repeat([0.5, 1.0], 9)
vols = np.concatenate([heston.implied_vol_smile(100, Ks[:9], t, 0.03, p) for t in (0.5, 1.0)])
cal = heston.calibrate(100, Ks, Ts, vols, r=0.03)
cal.params, cal.rmse_vol

# SVI smile for one expiry, with a butterfly-arbitrage check
k = np.log(Ks[9:] / (100 * np.exp(0.03 * 1.0)))  # log-moneyness vs the forward
fit = svi.fit(k, vols[9:], T=1.0)
fit.rmse_vol, fit.arbitrage_free
```

## Dashboard

`streamlit run app/dashboard.py` opens six tabs:

| Tab | What it shows |
|---|---|
| **Pricer** | European vs American price, early-exercise premium, every Greek (analytic and tree), value and Greek profiles vs spot across three expiries |
| **Scenarios** | P&L heatmap over spot × volatility shocks and holding period (diverging scale centred on break-even), with a table fallback |
| **Strategy builder** | Editable multi-leg positions (straddle, condor, butterfly, …): P&L today and at expiry, break-evens, aggregate Greeks |
| **Implied vol** | Solve IV from a quoted price, with the no-arbitrage range shown |
| **Smile lab** | Synthetic Heston market (or your own CSV of `strike, days, iv`): SVI fit per expiry, one-click Heston calibration, recovered vs true parameters |
| **Convergence** | Log-log error of CRR vs Leisen-Reimer trees against the exact price |

The visual design (dark slate + amber, Fira Sans/Fira Code, dense layout, colour-blind-safe
line styles) was produced with the *ui-ux-pro-max* design system; tokens live in
[`design-system/bs-plus/pages/dashboard.md`](design-system/bs-plus/pages/dashboard.md).

## How it is validated

Correctness checks are in `tests/` and run on every push (Python 3.10–3.13):

- **Published reference values**: Hull (BS call/put), Haug (Black-76, Garman-Kohlhagen,
  American calls on a high-carry asset).
- **Greeks vs finite differences**: all eight Greeks, calls and puts, three parameter sets.
- **Implied vol round trip** on a 225-point grid (strikes 20–400, 1 day–10 years, vol 3%–200%):
  every price with measurable time value recovers its vol to within 1e-6.
- **Trees**: Leisen-Reimer vs Black-Scholes and vs a 4,001-step CRR tree, Merton's
  no-early-exercise result for calls without dividends.
- **Heston**: fast Gauss-Legendre pricer vs adaptive quadrature to 1e-6 across 12 stress
  cases (including vol-of-vol 1.5, ρ = -0.9, Feller violated, 10-year expiry); reduces to
  Black-Scholes as vol-of-vol → 0; agrees with Monte Carlo; calibration recovers a
  synthetic surface to < 0.01 vol points.
- **SVI**: recovers known parameters, fits a Heston smile without arbitrage, and the
  density check flags a deliberately arbitrageable smile.
- **Monte Carlo**: within 4 standard errors of the closed form; the control variate cuts the
  standard error from 0.045 to about 0.008 on the same 200k paths.

### Measured performance (one laptop CPU, NumPy 2.3)

| Task | Time |
|---|---|
| 1,000,000 Black-Scholes prices | 0.16 s |
| 1,000,000 implied vols | 1.4 s |
| Heston prices, 50 strikes | 1.6 ms |
| American option, 201-step LR tree | 4.6 ms |

| Steps | CRR abs. error | Leisen-Reimer abs. error |
|---|---|---|
| 51 | 8.8e-3 | 1.9e-4 |
| 101 | 2.2e-2 | 4.9e-5 |
| 201 | 3.2e-3 | 1.2e-5 |

(European put, S=100, K=105, T=1, r=5%, σ=25%. Note CRR's error *rises* from 51 to 101
steps — the odd/even oscillation Leisen-Reimer avoids.)

## Implementation notes

- **Implied vol**: the quote is first reduced to the out-of-the-money option at the same
  strike (put-call parity keeps the time value), Newton starts at the Manaster-Koehler point
  `sqrt(2|ln(F/K)|/T)`, and a bracket `[lo, hi]` that always contains the root is shrunk each
  step; any Newton step leaving the bracket becomes a bisection. Tolerance is relative to the
  OTM price so that options with tiny time value are solved as accurately as ATM ones.
- **Heston**: characteristic function in the Albrecher et al. (2007) "little trap" form (no
  complex-log branch-cut jumps at long maturities). The truncation point of the Fourier
  integral uses both the Gaussian small-`u` decay and the exponential large-`u` decay rate
  `sqrt(1-ρ²)(v0 + κθT)/ξ`; the first version used only the Gaussian bound and was off by
  1.7e-4 for high vol-of-vol, which the stress test caught.
- **Calibration** minimizes implied-vol (not price) errors so cheap wing options carry weight.

## Limitations

- European/American vanillas only (plus the Asian MC); no barriers or discrete dividends.
- Heston calibration is a local optimizer from a heuristic start; on real data, try a few
  starting points and check `feller_satisfied()`.
- SVI is fitted slice by slice; calendar-spread arbitrage between expiries is not enforced.
- This is an educational/research tool, not trading advice.

## Project layout

```
src/bsplus/        black_scholes.py  implied_vol.py  american.py  heston.py  svi.py  monte_carlo.py
tests/             55 tests (pytest)
app/dashboard.py   Streamlit + Plotly dashboard
design-system/     ui-ux-pro-max design tokens
```

## License

MIT
