"""bsplus - Black-Scholes and beyond.

A compact, tested option-pricing toolkit:

* :mod:`bsplus.black_scholes` - generalized BSM (dividends, Black-76, FX),
  first/second-order analytic Greeks, discrete cash dividends
* :mod:`bsplus.implied_vol` - safeguarded Newton/bisection implied volatility
* :mod:`bsplus.american` - Leisen-Reimer / CRR trees with early exercise, cash
  dividends and an American implied-vol solver
* :mod:`bsplus.barrier` - single-barrier options (closed form + Monte Carlo)
* :mod:`bsplus.heston` - Heston stochastic volatility: pricing, MC, multi-start
  calibration
* :mod:`bsplus.svi` - SVI smile fitting with butterfly and calendar checks
* :mod:`bsplus.surface` - arbitrage-free SSVI surface and Dupire local vol
* :mod:`bsplus.monte_carlo` - variance-reduced MC (European, Asian)
* :mod:`bsplus.market` - option chain -> implied-vol surface (needs pandas;
  imported explicitly: ``from bsplus import market``)
"""

from . import american, barrier, black_scholes, heston, monte_carlo, surface, svi
from .american import american_implied_vol, american_price
from .barrier import barrier_price
from .black_scholes import (
    Greeks,
    black76_price,
    garman_kohlhagen_price,
    greeks,
    price,
    price_cash_dividends,
)
from .heston import HestonParams
from .implied_vol import implied_vol
from .surface import SSVISurface, fit_ssvi

__all__ = [
    "Greeks",
    "HestonParams",
    "SSVISurface",
    "american",
    "american_implied_vol",
    "american_price",
    "barrier",
    "barrier_price",
    "black76_price",
    "black_scholes",
    "fit_ssvi",
    "garman_kohlhagen_price",
    "greeks",
    "heston",
    "implied_vol",
    "monte_carlo",
    "price",
    "price_cash_dividends",
    "surface",
    "svi",
]

__version__ = "0.2.0"
