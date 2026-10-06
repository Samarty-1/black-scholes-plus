# Getting started

## Conventions

Every pricing function takes the same core arguments, in the same order:

| Argument | Meaning | Units |
|---|---|---|
| `S` | spot (or futures price for Black-76) | currency |
| `K` | strike | currency |
| `T` | time to expiry | years |
| `r` | risk-free rate | continuously compounded, decimal |
| `sigma` | volatility | annualized, decimal (0.2 = 20%) |
| `option_type` | `"call"` or `"put"` (case-insensitive, arrays allowed) | |
| `q` | continuous carry yield (dividend yield, foreign rate, or `r` for futures) | decimal |

All closed-form functions are **vectorized**: any argument may be a NumPy array and
arrays broadcast together, including `option_type`.

```python
import numpy as np
import bsplus as bs

strikes = np.linspace(80, 120, 5)
bs.price(100, strikes, 0.5, 0.04, 0.25, "call")
bs.price(100, 100, 0.5, 0.04, 0.25, np.array(["call", "put"]))
```

Greek units follow the usual quant convention and are documented on
[`greeks`](../api/black_scholes.md): `theta` is per year (divide by 365 for per day),
`vega` and `rho` are per unit (divide by 100 for per point).

## The dashboard

```bash
pip install -e ".[app]"
streamlit run app/dashboard.py
```

Seven tabs: Pricer, Scenarios, Strategy builder, Implied vol, Smile lab, Exotics and
Convergence. The Smile lab can load the bundled SPY snapshot or a live chain from Yahoo
Finance.

## Running the checks

```bash
pip install -e ".[dev]"
pytest          # unit, property-based and headless dashboard tests
ruff check src tests app examples
mypy            # type check (configured in pyproject.toml)
```
