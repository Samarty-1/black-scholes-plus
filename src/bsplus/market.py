"""Turn a listed option chain into a clean implied-volatility surface.

Steps, as a volatility desk would do them:

1. **Clean** - drop quotes with no bid, crossed or very wide markets, and
   strikes nobody holds (zero open interest).
2. **Implied forward** - for each expiry, put-call parity
   ``C - P = D (F - K)`` is linear in the strike, so a regression of
   ``C - P`` on ``K`` over near-the-money pairs gives the discount factor
   ``D`` (minus the slope) and the forward ``F``. This removes any need to
   guess the dividend yield or the funding rate.
3. **OTM implied vols** - the IV of each strike is taken from the
   out-of-the-money option (puts below the forward, calls above), which
   carries the time value and, for American-style options, almost no
   early-exercise premium.

Listed equity/ETF options (e.g. SPY) are American; treating OTM quotes as
European is the standard approximation and is accurate to well under the
bid-ask spread for the strikes kept here.

``fetch_chain`` needs the optional ``yfinance`` package; everything else
works on any DataFrame with the documented columns, so it can be fed from a
CSV or another data vendor.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .american import american_implied_vol
from .black_scholes import price as bs_price
from .implied_vol import implied_vol

CHAIN_COLUMNS = ["expiry", "days", "type", "strike", "bid", "ask", "volume", "open_interest"]


def fetch_chain(ticker: str, min_days: int = 7, max_days: int = 400, max_expiries: int = 8):
    """Download an option chain from Yahoo Finance via ``yfinance``.

    Returns ``(chain, spot, as_of)`` where ``chain`` has :data:`CHAIN_COLUMNS`.
    Expiries are thinned to at most ``max_expiries`` spread across the range.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("fetch_chain needs yfinance: pip install yfinance") from exc

    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")
    if hist.empty:
        raise ValueError(f"no price history for {ticker!r}")
    spot = float(hist["Close"].iloc[-1])
    today = dt.date.today()

    expiries = []
    for e in tk.options:
        days = (dt.date.fromisoformat(e) - today).days
        if min_days <= days <= max_days:
            expiries.append((e, days))
    if not expiries:
        raise ValueError(f"no expiries for {ticker!r} between {min_days} and {max_days} days")
    if len(expiries) > max_expiries:
        idx = np.unique(np.linspace(0, len(expiries) - 1, max_expiries).round().astype(int))
        expiries = [expiries[i] for i in idx]

    frames = []
    for e, days in expiries:
        oc = tk.option_chain(e)
        for kind, df in (("call", oc.calls), ("put", oc.puts)):
            frames.append(pd.DataFrame({
                "expiry": e,
                "days": days,
                "type": kind,
                "strike": df["strike"].astype(float),
                "bid": df["bid"].astype(float),
                "ask": df["ask"].astype(float),
                "volume": df["volume"].fillna(0).astype(float),
                "open_interest": df["openInterest"].fillna(0).astype(float),
            }))
    return pd.concat(frames, ignore_index=True), spot, today.isoformat()


def clean_chain(chain: pd.DataFrame, max_rel_spread: float = 0.35, min_bid: float = 0.05):
    """Keep two-sided, reasonably tight, held quotes and add a ``mid`` column."""
    df = chain.copy()
    df = df[(df["bid"] >= min_bid) & (df["ask"] > df["bid"])]
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df = df[(df["ask"] - df["bid"]) / df["mid"] <= max_rel_spread]
    df = df[df["open_interest"] > 0]
    return df.reset_index(drop=True)


def fetch_rate(symbol: str = "^IRX") -> float:
    """Latest 13-week T-bill yield from Yahoo Finance, as a decimal.

    ``^IRX`` is quoted on a discount basis in percent; at these maturities
    the difference from a continuously compounded rate is a few basis
    points, far below the noise in any option-implied rate.
    """
    import yfinance as yf

    hist = yf.Ticker(symbol).history(period="5d")
    if hist.empty:
        raise ValueError(f"no data for {symbol!r}")
    return float(hist["Close"].iloc[-1]) / 100.0


def _parity_pairs(slice_: pd.DataFrame, n_pairs: int):
    calls = slice_[slice_["type"] == "call"].groupby("strike")["mid"].mean()
    puts = slice_[slice_["type"] == "put"].groupby("strike")["mid"].mean()
    both = calls.index.intersection(puts.index)
    if len(both) < 3:
        raise ValueError("need at least 3 strikes quoted on both sides for parity")
    diff = (calls[both] - puts[both]).to_numpy(dtype=float)
    near = np.argsort(np.abs(diff))[:n_pairs]
    K = both.to_numpy(dtype=float)[near]
    return K, calls[K].to_numpy(float), puts[K].to_numpy(float)


def implied_forward(slice_: pd.DataFrame, T: float, rate: float | None = None,
                    n_pairs: int = 12, spot: float | None = None, american: bool = False,
                    n_iter: int = 3):
    """Forward and discount factor of one expiry from put-call parity.

    * ``rate=None`` (European-style chains such as SPX): fit
      ``C - P = D F - D K`` by least squares over the ``n_pairs`` strikes
      nearest the money; the slope gives ``D`` and the intercept ``F``.
    * ``rate`` given (American-style chains such as SPY): ``D = exp(-rate T)``
      and only the forward is estimated, as the median of ``K + (C - P) / D``.

    Why two modes: an American put's early-exercise premium grows with the
    strike, which drags ``C - P`` down at high strikes and steepens the
    slope. On a SPY chain (2026-10-06) the regression produced ``D > 1`` -
    i.e. negative rates of -2% to -49% - on every expiry. The forward
    itself is far less affected, because near the money the premium is
    small and the median ignores the strikes where it is not.

    Even with the right rate, the near-the-money American put still carries
    some early-exercise premium, which biases the forward low (by ~0.2% for
    a 6-month option at r = 5%). ``american=True`` (needs ``rate`` and
    ``spot``) removes it by *de-Americanizing*: each quote's volatility is
    solved on the American tree, the quote is re-priced as a European
    option at that volatility, and parity is re-run on the European
    equivalents. The tree needs the carry ``q``, which depends on ``F``, so
    this is iterated ``n_iter`` times from the plain estimate.

    Returns ``(F, D)``.
    """
    K, c, p = _parity_pairs(slice_, n_pairs)
    y = c - p
    if rate is not None:
        D = float(np.exp(-rate * T))
        F = float(np.median(K + y / D))
        if american:
            if spot is None:
                raise ValueError("american=True needs spot")
            for _ in range(n_iter):
                q = rate - np.log(F / spot) / T
                c_eu = _de_americanize(c, spot, K, T, rate, q, "call")
                p_eu = _de_americanize(p, spot, K, T, rate, q, "put")
                ok = np.isfinite(c_eu) & np.isfinite(p_eu)
                if ok.sum() < 3:
                    break
                F = float(np.median(K[ok] + (c_eu[ok] - p_eu[ok]) / D))
        return F, D
    slope, intercept = np.polyfit(K, y, 1)
    D = float(-slope)
    if not (0.5 < D <= 1.0):
        raise ValueError(
            f"parity regression gave discount factor {D:.4f}; for American-style "
            "options pass a rate instead"
        )
    return float(intercept / D), D


def _de_americanize(mids, spot, strikes, T, r, q, kind):
    """European-equivalent prices of American quotes (nan where unsolvable)."""
    out = np.full(len(mids), np.nan)
    for i, (m, k) in enumerate(zip(mids, strikes, strict=True)):
        vol = american_implied_vol(m, spot, k, T, r, kind, q)
        if np.isfinite(vol):
            out[i] = bs_price(spot, k, T, r, vol, kind, q)
    return out


def build_surface(chain: pd.DataFrame, spot: float, rate: float | None = None,
                  k_range: float = 0.6, american: bool = False):
    """Implied-vol surface from a cleaned chain.

    Returns one row per (expiry, strike) with columns
    ``days, T, strike, k, iv, forward, discount, type, mid, rel_spread``.
    Only out-of-the-money quotes are used, and log-moneyness is restricted
    to ``|k| <= k_range * sqrt(max(T, 0.25))`` to drop illiquid far wings.
    ``rate`` is passed to :func:`implied_forward` (give it for American-style
    chains). With ``american=True`` the forward is de-Americanized and each
    quote's IV is solved on the American tree rather than with
    Black-Scholes (slower: one tree root-search per quote).
    """
    rows = []
    for days, sl in chain.groupby("days"):
        T = days / 365.0
        try:
            F, D = implied_forward(sl, T, rate, spot=spot, american=american)
        except ValueError:
            continue
        r = -np.log(D) / T
        q = r - np.log(F / spot) / T  # carry that reproduces the forward
        otm = sl[((sl["type"] == "put") & (sl["strike"] < F))
                 | ((sl["type"] == "call") & (sl["strike"] >= F))]
        if otm.empty:
            continue
        if american:
            iv = np.array([
                american_implied_vol(m, spot, k, T, r, kind, q)
                for m, k, kind in zip(otm["mid"], otm["strike"], otm["type"], strict=True)
            ])
        else:
            iv = implied_vol(otm["mid"].to_numpy(), spot, otm["strike"].to_numpy(), T, r,
                             otm["type"].to_numpy(), q)
        k = np.log(otm["strike"].to_numpy() / F)
        keep = np.isfinite(iv) & (np.abs(k) <= k_range * np.sqrt(max(T, 0.25)))
        rows.append(pd.DataFrame({
            "days": int(days),
            "T": T,
            "strike": otm["strike"].to_numpy()[keep],
            "k": k[keep],
            "iv": iv[keep],
            "forward": F,
            "discount": D,
            "type": otm["type"].to_numpy()[keep],
            "mid": otm["mid"].to_numpy()[keep],
            "rel_spread": ((otm["ask"] - otm["bid"]) / otm["mid"]).to_numpy()[keep],
        }))
    if not rows:
        raise ValueError("no expiry produced a usable smile")
    return pd.concat(rows, ignore_index=True).sort_values(["days", "strike"], ignore_index=True)
