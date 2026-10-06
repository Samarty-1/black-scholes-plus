"""Real-data study: fit SVI, SSVI and Heston to a SPY option chain.

Reproduces ``docs/spy-study.md`` and its figures from the bundled snapshot:

    python examples/spy_surface_study.py

Pass ``--live`` to rebuild the chain from Yahoo Finance first (results will
then differ from the published ones, which is expected: it is a new day).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bsplus import heston, market, svi  # noqa: E402
from bsplus.surface import fit_ssvi  # noqa: E402

DOCS = ROOT / "docs"
DATA = ROOT / "data"
SNAPSHOT = "2026-10-06"


def load(live: bool):
    if live:
        chain, spot, as_of = market.fetch_chain("SPY")
        rate = market.fetch_rate()
    else:
        chain = pd.read_csv(DATA / f"spy_chain_{SNAPSHOT}.csv")
        meta = dict(line.split("=", 1) for line in
                    (DATA / f"spy_chain_{SNAPSHOT}.meta").read_text().splitlines())
        spot, as_of = float(meta["spot"]), meta["as_of"]
        rate = float(meta["rate"].split()[0])
    return chain, spot, as_of, rate


def subsample(surf, per_expiry=25):
    """Evenly spaced quotes per expiry so every maturity weighs the same."""
    idx = []
    for _, s in surf.groupby("days"):
        idx += list(s.index[np.unique(np.linspace(0, len(s) - 1,
                                                  min(per_expiry, len(s))).astype(int))])
    return surf.loc[idx]


def main(live: bool = False):
    chain, spot, as_of, rate = load(live)
    cleaned = market.clean_chain(chain)

    t0 = time.time()
    euro = market.build_surface(cleaned, spot, rate=rate)
    surf = market.build_surface(cleaned, spot, rate=rate, american=True)
    t_build = time.time() - t0

    # --- de-Americanization impact ----------------------------------------
    deam = []
    for d in sorted(surf["days"].unique()):
        a, e = surf[surf["days"] == d], euro[euro["days"] == d]
        m = a.merge(e, on="strike", suffixes=("_a", "_e"))
        deam.append({
            "days": int(d),
            "forward shift %": 100 * (a["forward"].iloc[0] / e["forward"].iloc[0] - 1),
            "median IV change (vol pts)": 100 * np.median(m["iv_a"] - m["iv_e"]),
            "max |IV change| (vol pts)": 100 * np.abs(m["iv_a"] - m["iv_e"]).max(),
        })
    deam = pd.DataFrame(deam)

    # --- models ------------------------------------------------------------
    svi_fits = {}
    for d, s in surf.groupby("days"):
        svi_fits[int(d)] = svi.fit(s["k"].to_numpy(), s["iv"].to_numpy(), s["T"].iloc[0])
    ssvi = fit_ssvi(surf["k"], surf["T"], surf["iv"])
    slice_list = [(float(surf.loc[surf["days"] == d, "T"].iloc[0]), f.params)
                  for d, f in svi_fits.items()]
    quoted = svi.calendar_crossings(slice_list, np.linspace(surf["k"].quantile(0.25),
                                                            surf["k"].quantile(0.75), 200))
    wide = svi.calendar_crossings(slice_list, np.linspace(-0.6, 0.5, 400))
    n_bfly = sum(not f.arbitrage_free for f in svi_fits.values())
    n_cal_wide = sum(frac > 0 for _, frac in wide)
    worst_cal = max(wide, key=lambda x: x[1])

    sub = subsample(surf)
    q = (rate - np.log(sub["forward"] / spot) / sub["T"]).to_numpy()
    t0 = time.time()
    hes = heston.calibrate(spot, sub["strike"], sub["T"], sub["iv"], rate, q=q, n_starts=8)
    t_heston = time.time() - t0
    hes_err = (hes.model_vols - sub["iv"].to_numpy()) * 100

    rows = []
    for d, s in surf.groupby("days"):
        T = s["T"].iloc[0]
        f = svi_fits[int(d)]
        hs = hes_err[sub["days"].to_numpy() == d]
        rows.append({
            "days": int(d),
            "quotes": len(s),
            "ATM IV %": 100 * s["iv"].iloc[np.argmin(np.abs(s["k"]))],
            "SVI": 100 * f.rmse_vol,
            "SVI arb-free": "yes" if f.arbitrage_free else "**no**",
            "SSVI": 100 * ssvi.rmse_by_expiry[float(T)],
            "Heston": float(np.sqrt(np.mean(hs**2))),
        })
    table = pd.DataFrame(rows)
    svi_all = np.sqrt(np.mean(np.concatenate([
        (svi.implied_vol(s["k"], s["T"].iloc[0], svi_fits[int(d)].params) - s["iv"]) ** 2
        for d, s in surf.groupby("days")])))

    # --- figures -------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#0F172A", "axes.facecolor": "#1A2234", "axes.edgecolor": "#334155",
        "axes.labelcolor": "#CBD5E1", "xtick.color": "#94A3B8", "ytick.color": "#94A3B8",
        "text.color": "#F8FAFC", "grid.color": "#1E293B", "font.size": 9,
        "legend.facecolor": "#0F172A", "legend.edgecolor": "#334155",
    })
    DOCS.mkdir(exist_ok=True)
    expiries = sorted(surf["days"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), sharey=False)
    for ax, d in zip(axes.ravel(), expiries, strict=False):
        s = surf[surf["days"] == d]
        T = s["T"].iloc[0]
        kk = np.linspace(s["k"].min(), s["k"].max(), 200)
        fwd = s["forward"].iloc[0]
        ax.plot(s["strike"], 100 * s["iv"], "o", ms=2.5, mfc="none", mec="#F8FAFC",
                label="market (mid)")
        ax.plot(fwd * np.exp(kk), 100 * svi.implied_vol(kk, T, svi_fits[int(d)].params),
                color="#F59E0B", lw=1.6, label="SVI")
        ax.plot(fwd * np.exp(kk), 100 * ssvi.surface.implied_vol(kk, T), "--",
                color="#3B82F6", lw=1.6, label="SSVI")
        hv = heston.implied_vol_smile(spot, fwd * np.exp(kk), T, rate, hes.params,
                                      rate - np.log(fwd / spot) / T)
        ax.plot(fwd * np.exp(kk), 100 * hv, ":", color="#8B5CF6", lw=1.8, label="Heston")
        ax.set_title(f"{d} days", fontsize=10)
        ax.grid(True)
    axes[0, 0].set_ylabel("implied vol (%)")
    axes[1, 0].set_ylabel("implied vol (%)")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"SPY implied-volatility smiles, {as_of} (spot {spot:.2f})", fontsize=12)
    fig.tight_layout()
    fig.savefig(DOCS / "spy-smiles.png", dpi=140)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(table))
    for i, (col, color) in enumerate([("SVI", "#F59E0B"), ("SSVI", "#3B82F6"),
                                      ("Heston", "#8B5CF6")]):
        ax.bar(x + (i - 1) * 0.27, table[col], 0.27, label=col, color=color)
    ax.set_xticks(x, [f"{d}d" for d in table["days"]])
    ax.set_ylabel("RMSE (vol points)")
    ax.set_title("Fit error by expiry (lower is better)")
    ax.legend()
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(DOCS / "spy-fit-error.png", dpi=140)

    # --- report ------------------------------------------------------------
    def md(df, fmt="{:.3f}"):
        df = df.copy()
        for c in df.columns:
            if df[c].dtype.kind == "f":
                df[c] = df[c].map(fmt.format)
        head = "| " + " | ".join(df.columns) + " |"
        sep = "|" + "|".join("---" for _ in df.columns) + "|"
        body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.to_numpy())
        return f"{head}\n{sep}\n{body}"

    p = hes.params
    ssvi_s = ssvi.surface
    at_bound = p.xi > 2.99 or p.kappa > 14.9
    bound_note = (f"; a parameter at its bound (xi = {p.xi:.2f}, kappa = {p.kappa:.1f}) is "
                  "the usual symptom" if at_bound else "")
    best = min(hes.start_rmses)
    converged = sum(abs(x - best) < 1e-4 for x in hes.start_rmses)
    snapshot_note = "" if live else (
        "\n  (On this snapshot, before de-Americanization the 359-day slice also failed the "
        "butterfly\n  check: that \"arbitrage\" was the early-exercise premium, not the market.)")
    report = f"""# SPY volatility-surface study ({as_of})

*Generated by `examples/spy_surface_study.py` from the bundled snapshot
`data/spy_chain_{as_of}.csv` (Yahoo Finance via yfinance). Spot {spot:.2f}, rate
{rate:.3%} (13-week T-bill, `^IRX`). Re-run the script to reproduce every number.*

## Data

{len(chain):,} raw quotes over {chain['days'].nunique()} expiries ({chain['days'].min()}-{chain['days'].max()} days);
{len(cleaned):,} after removing no-bid, crossed, >35%-wide and zero-open-interest quotes; {len(surf):,}
out-of-the-money quotes in the final surface. Building it took {t_build:.0f} s, almost all of
it de-Americanization.

SPY options are **American** and SPY pays dividends. Two consequences found while
building this study:

1. **Put-call-parity regression fails.** Regressing `C - P` on strike to get the discount
   factor gave `D > 1` (a negative rate, -2% to -49%) on every expiry, because the
   early-exercise premium of in-the-money puts grows with strike. The library now
   refuses that regression for such chains and takes `D` from the T-bill rate instead.
2. **De-Americanizing matters for long expiries.** Each quote's vol is solved on the
   American (Leisen-Reimer) tree and the forward is re-estimated from European-equivalent
   prices. Compared with treating the quotes as European:

{md(deam)}

Under a month the difference is negligible; at one year it is up to
{deam['max |IV change| (vol pts)'].max():.1f} vol points - larger than the SVI fit error itself.

## Fits

| Model | What it is | Overall RMSE (vol pts) | Static-arbitrage free |
|---|---|---|---|
| SVI | 5 parameters per expiry, fitted independently | {svi_all * 100:.3f} | not guaranteed: {n_bfly} of {len(svi_fits)} slices fail the butterfly check; {n_cal_wide} of {len(wide)} adjacent pairs cross (calendar arbitrage) on a wide k grid |
| SSVI | 1 ATM variance per expiry + 3 global parameters | {ssvi.rmse_vol * 100:.3f} | **yes, by construction** |
| Heston | 5-parameter stochastic-vol model (dynamics, not just a fit) | {np.sqrt(np.mean(hes_err**2)):.3f} | yes (it is a model) |

{md(table)}

![smiles](spy-smiles.png)

![fit error](spy-fit-error.png)

**SSVI**: rho = {ssvi_s.rho:.3f}, eta = {ssvi_s.eta:.3f}, gamma = {ssvi_s.gamma:.3f}, so
eta(1+|rho|) = {ssvi_s.eta * (1 + abs(ssvi_s.rho)):.3f}. Both no-arbitrage bounds (gamma <= 0.5,
eta(1+|rho|) <= 2) are essentially *active*: the data wants more short-dated curvature
than the sufficient conditions allow. That is the price of the guarantee.

**Heston** (8-start calibration, {t_heston:.0f} s): v0 = {p.v0:.4f}, kappa = {p.kappa:.2f},
theta = {p.theta:.4f}, xi = {p.xi:.2f}, rho = {p.rho:.3f}; Feller condition
{'satisfied' if p.feller_satisfied() else 'violated'}. Short-pass RMSE from each start (vol pts):
{', '.join(f'{x * 100:.2f}' for x in hes.start_rmses)}.

## Reading the results

* **SVI fits best** because it has the most freedom - 5 parameters for every expiry.
  On this (de-Americanized) data {n_bfly} slice(s) fail the butterfly check. Slices
  {'never' if all(f == 0 for _, f in quoted) else 'do'} cross inside the quoted strikes,
  but in the extrapolated wings {n_cal_wide} adjacent pair(s) cross - worst
  {worst_cal[0][0] * 365:.0f}d/{worst_cal[0][1] * 365:.0f}d on {worst_cal[1]:.0%} of the
  grid - which is calendar arbitrage. Fine for interpolating quotes; unsafe to
  extrapolate or to feed into local vol, because nothing ties the slices together.{snapshot_note}
* **SSVI** gives up fit (mostly at the shortest expiries, where the skew is steepest) for
  a surface that is guaranteed arbitrage-free and smooth in maturity - the input a
  local-volatility model needs.
* **Heston** fits worst at short maturities. It is a diffusion, so the smile it can
  generate over a few days is limited{bound_note}. This is a known limitation of the
  model (jumps or rough volatility are the usual fixes), not a calibration bug:
  {converged} of {hes.n_starts} starting points reach the same optimum.

*Snapshot quotes are delayed and mid-prices are used; this is a methods demonstration,
not trading advice.*
"""
    (DOCS / "spy-study.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true", help="fetch a fresh chain first")
    main(ap.parse_args().live)
