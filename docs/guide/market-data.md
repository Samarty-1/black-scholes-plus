# Real market data

`bsplus.market` turns a listed option chain into a clean implied-volatility surface.
It needs `pandas` (and `yfinance` for downloading): `pip install "bsplus[data]"`.

```python
from bsplus import market

chain, spot, as_of = market.fetch_chain("SPY", min_days=7, max_days=400, max_expiries=8)
rate = market.fetch_rate()                     # 13-week T-bill (^IRX)
clean = market.clean_chain(chain)              # drop no-bid, crossed, wide, zero-OI quotes
surface = market.build_surface(clean, spot, rate=rate, american=True)
```

`build_surface` returns one row per (expiry, strike) with `days, T, strike, k, iv,
forward, discount, type, mid, rel_spread`. Any data vendor works: pass a DataFrame with
the columns in `market.CHAIN_COLUMNS`.

## Forwards from put-call parity

For European options, put-call parity $C - P = D(F - K)$ is linear in the strike.
Regressing $C - P$ on $K$ over near-the-money pairs gives the discount factor $D$ and
the forward $F$, with no need to guess the dividend yield or the funding rate. That is
what `implied_forward(slice, T)` does when no rate is given.

!!! warning "American options break the regression"
    The early-exercise premium of in-the-money American puts grows with the strike,
    which steepens the regression slope. On SPY it implied negative rates of −2% to
    −49%. `implied_forward` raises an error in that case. Pass `rate=` (and
    `american=True` with `spot=`) for American chains.

## De-Americanization

With `american=True`, each quote's volatility is solved on the American (Leisen-Reimer)
tree, the quote is re-priced as a European option at that volatility, and parity is
re-run on the European equivalents. The tree needs the carry, which depends on the
forward, so this is iterated a few times. For SPY the correction is negligible under a
month but reaches 1.2 vol points at one year. See the [SPY study](../spy-study.md).

Out-of-the-money quotes are used for the final vols (puts below the forward, calls
above): they carry the time value and almost no early-exercise premium.
