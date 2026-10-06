"""bsplus - Black-Scholes and beyond.

A compact, tested option-pricing toolkit:

* :mod:`bsplus.black_scholes` - generalized BSM (dividends, Black-76, FX) and
  first/second-order analytic Greeks
* :mod:`bsplus.implied_vol` - safeguarded Newton/bisection implied volatility
* :mod:`bsplus.american` - Leisen-Reimer / CRR binomial trees with early exercise
* :mod:`bsplus.heston` - Heston stochastic volatility: pricing, MC, calibration
* :mod:`bsplus.svi` - SVI smile fitting with a butterfly-arbitrage check
* :mod:`bsplus.monte_carlo` - variance-reduced MC (European, Asian)
"""

from . import american, black_scholes, heston, monte_carlo, svi
from .american import american_price
from .black_scholes import Greeks, black76_price, garman_kohlhagen_price, greeks, price
from .heston import HestonParams
from .implied_vol import implied_vol

__all__ = [
    "Greeks",
    "HestonParams",
    "american",
    "american_price",
    "black76_price",
    "black_scholes",
    "garman_kohlhagen_price",
    "greeks",
    "heston",
    "implied_vol",
    "monte_carlo",
    "price",
    "svi",
]

__version__ = "0.1.0"
