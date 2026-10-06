# Pricing and Greeks

## European options

The generalized Black-Scholes-Merton price with carry yield $q$ is

$$
C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad
d_{1,2} = \frac{\ln(S/K) + (r - q \pm \tfrac12\sigma^2)T}{\sigma\sqrt T}.
$$

One code path covers three markets:

```python
bs.price(100, 100, 1, 0.04, 0.2, "call", q=0.015)            # stock with dividend yield
bs.black76_price(19, 19, 0.75, 0.10, 0.28, "put")             # option on a future (q = r)
bs.garman_kohlhagen_price(1.56, 1.60, 0.5, 0.06, 0.08, 0.12)  # FX option (q = foreign rate)
```

When $\sigma\sqrt{T} = 0$ the price is the discounted forward intrinsic value, the exact
limit of the formula, rather than a division-by-zero error.

## Greeks

```python
g = bs.greeks(100, 105, 0.25, 0.04, 0.3, "put", q=0.01)
g.delta, g.gamma, g.vega, g.theta, g.rho     # first order (+ gamma)
g.vanna, g.volga, g.charm                    # second order
```

Every Greek is tested against a finite-difference derivative of the price.

## Implied volatility

```python
bs.implied_vol(price, S, K, T, r, "call", q)
```

The solver reduces each quote to the out-of-the-money option at the same strike, starts
Newton at the Manaster-Koehler point $\sigma_0 = \sqrt{2|\ln(F/K)|/T}$, and keeps a
bracket that always contains the root, falling back to bisection whenever Newton would
leave it. Quotes outside the no-arbitrage bounds return `nan`.

## American options and dividends

```python
res = bs.american_price(100, 110, 1.0, 0.08, 0.2, "put")
res.price, res.early_exercise_premium, res.delta, res.gamma, res.theta, res.vega, res.rho

# discrete cash dividends: (time in years, amount)
bs.american_price(100, 80, 0.5, 0.03, 0.2, "call", dividends=[(0.45, 5.0)])
bs.price_cash_dividends(100, 95, 1.0, 0.04, 0.25, [(0.25, 1.5), (0.75, 1.5)], "call")
```

The default engine is a Leisen-Reimer tree, whose error falls roughly as $1/n^2$ rather
than the oscillating $1/n$ of Cox-Ross-Rubinstein. Dividends use the escrowed model:
the tree is built on the share price net of the dividends' present value, and early
exercise is judged against the actual share price.

## Barrier options

```python
bs.barrier_price(S, K, H, T, r, sigma, "call", "down-and-out", q=0.0, rebate=0.0)
```

All eight single-barrier types (down/up, in/out, call/put) are priced in closed form
(Reiner-Rubinstein). `barrier.barrier_mc` provides an independent Monte Carlo check
with a Brownian-bridge correction for continuous monitoring.
