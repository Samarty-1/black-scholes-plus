"""Interactive dashboard for the bsplus option-pricing library.

Run with:  streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import bsplus as bs
from bsplus import heston, market, svi
from bsplus.american import american_price, european_tree_price
from bsplus.barrier import KINDS, barrier_mc, barrier_price
from bsplus.surface import fit_ssvi

# --- Design tokens (design-system/bs-plus/pages/dashboard.md) ---------------
BG, CARD, BORDER, GRID = "#0F172A", "#1A2234", "#334155", "#1E293B"
FG, MUTED = "#F8FAFC", "#94A3B8"
AMBER, BLUE, VIOLET = "#F59E0B", "#3B82F6", "#8B5CF6"
GAIN, LOSS = "#26A69A", "#EF5350"
DASHES = ["solid", "dash", "dot", "dashdot"]

st.set_page_config(page_title="BS+ Option Lab", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextInput, .stNumberInput, .stSelectbox {{
  font-family: 'Fira Sans', system-ui, sans-serif;
}}
.block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
[data-testid="stMetric"] {{
  background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 12px 14px;
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.8rem; text-transform: uppercase;
  letter-spacing: 0.04em; }}
[data-testid="stMetricValue"] {{ font-family: 'Fira Code', monospace;
  font-variant-numeric: tabular-nums;
  font-size: 1.45rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ padding: 8px 14px; min-height: 44px; }}
.stButton button {{ min-height: 44px; cursor: pointer; transition: filter 150ms ease; }}
.stButton button:hover {{ filter: brightness(1.1); }}
.stButton button[kind="primary"] {{ color: {BG}; font-weight: 600; }}
*:focus-visible {{ outline: 2px solid {AMBER} !important; outline-offset: 2px; }}
.mono {{ font-family: 'Fira Code', monospace; font-variant-numeric: tabular-nums; }}
.caption {{ color: {MUTED}; font-size: 0.88rem; line-height: 1.5; }}
.badge {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem;
  border: 1px solid {BORDER}; color: {MUTED}; margin-right: 6px; }}
.ok {{ color: {GAIN}; border-color: {GAIN}; }}
.warn {{ color: {LOSS}; border-color: {LOSS}; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
""",
    unsafe_allow_html=True,
)


def style_fig(fig: go.Figure, title: str, x: str, y: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=FG), x=0.01),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CARD,
        font=dict(family="Fira Sans, sans-serif", color=MUTED, size=12),
        margin=dict(l=56, r=20, t=48, b=104),
        # Legend sits below the x-axis title: at the top it collided with the
        # chart title and Plotly's toolbar on narrow (two-column) charts.
        legend=dict(orientation="h", y=-0.26, x=0, xanchor="left", yanchor="top",
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=BG, bordercolor=BORDER, font=dict(family="Fira Code", color=FG)),
    )
    fig.update_xaxes(title=x, gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(title=y, gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER)
    return fig


def line(fig, x, y, name, color, dash=0, width=2.2):
    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines",
                             line=dict(color=color, dash=DASHES[dash], width=width)))


def fmt(x, d=4):
    return "—" if x is None or not np.isfinite(x) else f"{x:,.{d}f}"


# --- Sidebar: contract inputs ------------------------------------------------
with st.sidebar:
    st.markdown("### Contract")
    option_type = st.radio("Type", ["call", "put"], horizontal=True)
    S = st.number_input("Spot S", min_value=0.01, value=100.0, step=1.0)
    K = st.number_input("Strike K", min_value=0.01, value=100.0, step=1.0)
    days = st.number_input("Days to expiry", min_value=1, max_value=3650, value=90, step=1)
    sigma = st.slider("Volatility σ (%)", 1.0, 150.0, 25.0, 0.5) / 100
    r = st.slider("Risk-free rate r (%)", -2.0, 15.0, 4.0, 0.05) / 100
    q = st.slider("Dividend / carry yield q (%)", 0.0, 15.0, 0.0, 0.05) / 100
    st.markdown(
        '<p class="caption">Set q = r for options on futures (Black-76) or q = foreign '
        "rate for FX (Garman-Kohlhagen).</p>",
        unsafe_allow_html=True,
    )
    div_text = st.text_input("Cash dividends (days:amount, …)", "",
                             placeholder="e.g. 30:1.25, 120:1.25",
                             help="Discrete dividends for the Pricer tab (escrowed model).")
T = days / 365.0


def parse_dividends(text):
    divs, bad = [], []
    for part in filter(None, (p.strip() for p in text.split(","))):
        try:
            d_, a_ = (float(x) for x in part.split(":"))
            if d_ <= 0 or a_ <= 0:
                raise ValueError
            divs.append((d_ / 365.0, a_))
        except ValueError:
            bad.append(part)
    return divs, bad


dividends, bad_divs = parse_dividends(div_text)
if bad_divs:
    st.sidebar.error(f"Ignored (use days:amount): {', '.join(bad_divs)}")
pv_divs = sum(a_ * np.exp(-r * t_) for t_, a_ in dividends if t_ <= T)
if pv_divs >= S:
    st.sidebar.error("Dividends exceed the spot price; ignoring them.")
    dividends, pv_divs = [], 0.0

st.markdown("## BS+ Option Lab")
st.markdown(
    '<p class="caption">Black-Scholes and beyond: analytic Greeks, robust implied vol, '
    "American early exercise, Heston, SVI/SSVI surfaces on real SPY data, local vol and "
    "barrier options.</p>",
    unsafe_allow_html=True,
)

tabs = st.tabs(["Pricer", "Scenarios", "Strategy builder", "Implied vol",
                "Smile lab", "Exotics", "Convergence"])

# --- Tab 1: pricer -----------------------------------------------------------
with tabs[0]:
    # With cash dividends the European side uses the escrowed model: Black-Scholes
    # on S minus the PV of dividends paid before expiry.
    g = bs.greeks(S - pv_divs, K, T, r, sigma, option_type, q)
    amer = american_price(S, K, T, r, sigma, option_type, q, steps=301, dividends=dividends)

    c = st.columns(4)
    c[0].metric("European (escrowed divs)" if dividends else "European (BSM)", fmt(g.price))
    c[1].metric("American (LR tree)", fmt(amer.price))
    c[2].metric("Early-exercise premium", fmt(amer.early_exercise_premium))
    fwd = (S - pv_divs) * np.exp((r - q) * T)
    c[3].metric("Forward", fmt(fwd, 2))
    c[3].caption(f"{(fwd / K - 1) * 100:+.2f}% vs strike")

    st.markdown("#### Greeks")
    greek_rows = [
        ("Delta", g.delta, amer.delta, "∂V/∂S"),
        ("Gamma", g.gamma, amer.gamma, "∂²V/∂S²"),
        ("Vega (per 1 vol pt)", g.vega / 100, amer.vega / 100, "∂V/∂σ ÷ 100"),
        ("Theta (per day)", g.theta / 365, amer.theta / 365, "−∂V/∂T ÷ 365"),
        ("Rho (per 1%)", g.rho / 100, amer.rho / 100, "∂V/∂r ÷ 100"),
        ("Vanna", g.vanna, None, "∂Δ/∂σ"),
        ("Volga", g.volga, None, "∂²V/∂σ²"),
        ("Charm (per day)", g.charm / 365, None, "−∂Δ/∂T ÷ 365"),
    ]
    df = pd.DataFrame(
        [(n, fmt(e, 5), fmt(a, 5) if a is not None else "—", d) for n, e, a, d in greek_rows],
        columns=["Greek", "European (analytic)", "American (tree)", "Definition"],
    )
    st.dataframe(df, hide_index=True, width="stretch")

    spots = np.linspace(max(S * 0.4, 1.2 * pv_divs, 0.01), S * 1.6, 241)
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        line(fig, spots, bs.price(spots - pv_divs, K, T, r, sigma, option_type, q),
             "European today", AMBER)
        amer_curve = [american_price(s, K, T, r, sigma, option_type, q, steps=101,
                                     dividends=dividends).price
                      for s in spots[::6]]
        line(fig, spots[::6], amer_curve, "American today", BLUE, 1)
        payoff = np.maximum((spots - K) if option_type == "call" else (K - spots), 0)
        line(fig, spots, payoff, "Payoff at expiry", MUTED, 2, 1.6)
        fig.add_vline(x=S, line=dict(color=BORDER, dash="dot"))
        st.plotly_chart(style_fig(fig, "Value vs spot", "Spot", "Option value"),
                        width="stretch")
    with right:
        greek_name = st.selectbox("Greek profile", ["delta", "gamma", "vega", "theta",
                                                    "vanna", "volga", "charm"])
        fig = go.Figure()
        for i, frac in enumerate([1.0, 0.5, 0.1]):
            t_f = max(T * frac, 1 / 365)
            pv_f = sum(a_ * np.exp(-r * t_) for t_, a_ in dividends if t_ <= t_f)
            gg = bs.greeks(np.maximum(spots - pv_f, 1e-6), K, t_f, r, sigma, option_type, q)
            label = f"{max(round(days * frac), 1)}d to expiry"
            line(fig, spots, getattr(gg, greek_name), label, [AMBER, BLUE, VIOLET][i], i)
        fig.add_vline(x=S, line=dict(color=BORDER, dash="dot"))
        st.plotly_chart(style_fig(fig, f"{greek_name.capitalize()} vs spot across time",
                                  "Spot", greek_name), width="stretch")

# --- Tab 2: scenarios --------------------------------------------------------
with tabs[1]:
    st.markdown(
        '<p class="caption">Mark-to-model P&L of <b>one long option</b> under simultaneous '
        "spot and volatility shocks, after the chosen holding period.</p>",
        unsafe_allow_html=True,
    )
    a, b, c3 = st.columns(3)
    spot_range = a.slider("Spot shock range (±%)", 5, 50, 20)
    vol_range = b.slider("Vol shock range (± vol pts)", 2, 40, 10)
    hold = c3.slider("Holding period (days)", 0, int(days) - 1, 0) if days > 1 else 0

    ds = np.linspace(-spot_range, spot_range, 21) / 100
    dv = np.linspace(-vol_range, vol_range, 17) / 100
    SS, VV = np.meshgrid(S * (1 + ds), np.maximum(sigma + dv, 0.005))
    T_h = max(T - hold / 365, 1e-6)
    pnl = bs.price(SS, K, T_h, r, VV, option_type, q) - g.price
    fig = go.Figure(go.Heatmap(
        x=ds * 100, y=(sigma + dv) * 100, z=pnl, zmid=0,
        colorscale=[[0, LOSS], [0.5, CARD], [1, GAIN]],
        colorbar=dict(title="P&L", tickfont=dict(family="Fira Code")),
        hovertemplate="spot %{x:+.1f}%<br>vol %{y:.1f}%<br>P&L %{z:+.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=[0], y=[sigma * 100], mode="markers", name="Today",
                             marker=dict(symbol="x", size=12, color=FG)))
    fig = style_fig(fig, "P&L heatmap: spot shock × volatility", "Spot shock (%)",
                    "Volatility (%)", 460)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, width="stretch")

    with st.expander("Show P&L grid as a table"):
        tbl = pd.DataFrame(pnl, index=[f"{v * 100:.1f}%" for v in sigma + dv],
                           columns=[f"{x * 100:+.0f}%" for x in ds])
        st.dataframe(tbl.style.format("{:+.2f}"), width="stretch")

# --- Tab 3: strategy builder -------------------------------------------------
PRESETS = {
    "Long straddle": [("call", 0, 1), ("put", 0, 1)],
    "Long strangle": [("call", 10, 1), ("put", -10, 1)],
    "Bull call spread": [("call", 0, 1), ("call", 10, -1)],
    "Bear put spread": [("put", 0, 1), ("put", -10, -1)],
    "Iron condor": [("put", -15, 1), ("put", -5, -1), ("call", 5, -1), ("call", 15, 1)],
    "Long butterfly": [("call", -10, 1), ("call", 0, -2), ("call", 10, 1)],
    "Short OTM call": [("call", 10, -1)],
}

with tabs[2]:
    left, right = st.columns([1, 2])
    with left:
        preset = st.selectbox("Preset", list(PRESETS))
        legs = pd.DataFrame(
            [{"type": t, "strike": round(S * (1 + off / 100), 2), "qty": qty,
              "days": int(days), "vol %": round(sigma * 100, 2)}
             for t, off, qty in PRESETS[preset]]
        )
        legs = st.data_editor(
            legs, num_rows="dynamic", hide_index=True, width="stretch",
            key=f"legs_{preset}",
            column_config={
                "type": st.column_config.SelectboxColumn(options=["call", "put"], required=True),
                "strike": st.column_config.NumberColumn(min_value=0.01, format="%.2f"),
                "qty": st.column_config.NumberColumn(step=1, help="Negative = short"),
                "days": st.column_config.NumberColumn(min_value=1, step=1),
                "vol %": st.column_config.NumberColumn(min_value=0.5, format="%.2f"),
            },
        )
        legs = legs.dropna()
    with right:
        if legs.empty:
            st.info("Add at least one leg.")
        else:
            grid = np.linspace(S * 0.5, S * 1.5, 301)
            now = np.zeros_like(grid)
            expiry = np.zeros_like(grid)
            cost = 0.0
            agg = dict(delta=0.0, gamma=0.0, vega=0.0, theta=0.0)
            min_days = legs["days"].min()
            for _, lg in legs.iterrows():
                t_leg, vol = lg["days"] / 365, lg["vol %"] / 100
                premium = bs.price(S, lg["strike"], t_leg, r, vol, lg["type"], q)
                cost += lg["qty"] * premium
                now += lg["qty"] * bs.price(grid, lg["strike"], t_leg, r, vol, lg["type"], q)
                # Value at the first expiry: expiring legs at intrinsic, others at model.
                t_rem = (lg["days"] - min_days) / 365
                expiry += lg["qty"] * bs.price(grid, lg["strike"], t_rem, r, vol, lg["type"], q)
                gl = bs.greeks(S, lg["strike"], t_leg, r, vol, lg["type"], q)
                for k in agg:
                    agg[k] += lg["qty"] * getattr(gl, k)
            m = st.columns(5)
            m[0].metric("Net premium", fmt(cost, 3), "debit" if cost > 0 else "credit",
                        delta_color="off")
            m[1].metric("Delta", fmt(agg["delta"], 3))
            m[2].metric("Gamma", fmt(agg["gamma"], 4))
            m[3].metric("Vega / vol pt", fmt(agg["vega"] / 100, 3))
            m[4].metric("Theta / day", fmt(agg["theta"] / 365, 3))
            fig = go.Figure()
            pnl_exp = expiry - cost
            fig.add_trace(go.Scatter(x=grid, y=np.where(pnl_exp >= 0, pnl_exp, np.nan),
                                     fill="tozeroy", mode="none", fillcolor="rgba(38,166,154,0.18)",
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=grid, y=np.where(pnl_exp < 0, pnl_exp, np.nan),
                                     fill="tozeroy", mode="none", fillcolor="rgba(239,83,80,0.18)",
                                     showlegend=False, hoverinfo="skip"))
            line(fig, grid, pnl_exp, f"P&L at first expiry ({int(min_days)}d)", AMBER)
            line(fig, grid, now - cost, "P&L today (model)", BLUE, 1)
            fig.add_hline(y=0, line=dict(color=BORDER))
            fig.add_vline(x=S, line=dict(color=BORDER, dash="dot"))
            st.plotly_chart(style_fig(fig, f"{preset}: P&L profile", "Spot at horizon", "P&L",
                                      420), width="stretch")
            crossings = grid[:-1][np.diff(np.sign(pnl_exp)) != 0]
            be = ", ".join(f"{x:.2f}" for x in crossings) or "none in range"
            st.markdown(f'<p class="caption">Break-even spot(s) at expiry: '
                        f'<span class="mono">{be}</span> · max P&L in range '
                        f'<span class="mono">{pnl_exp.max():+.2f}</span> · min '
                        f'<span class="mono">{pnl_exp.min():+.2f}</span></p>',
                        unsafe_allow_html=True)

# --- Tab 4: implied vol ------------------------------------------------------
with tabs[3]:
    left, right = st.columns([1, 2])
    with left:
        st.markdown("#### Solve for implied volatility")
        mkt = st.number_input("Observed option price", min_value=0.0,
                              value=float(round(g.price, 6)), step=0.05, format="%.6f")
        iv = bs.implied_vol(mkt, S, K, T, r, option_type, q)
        if np.isfinite(iv):
            st.metric("Implied volatility", f"{iv * 100:.4f}%")
            st.markdown('<span class="badge ok">inside no-arbitrage bounds</span>',
                        unsafe_allow_html=True)
        else:
            st.metric("Implied volatility", "no solution")
            st.markdown('<span class="badge warn">price violates no-arbitrage bounds</span>',
                        unsafe_allow_html=True)
        lower = bs.price(S, K, T, r, 0.0, option_type, q)
        upper = S * np.exp(-q * T) if option_type == "call" else K * np.exp(-r * T)
        st.markdown(f'<p class="caption">Valid range: <span class="mono">({lower:.4f}, '
                    f'{upper:.4f})</span>. The solver keeps a bracket around the root and '
                    "falls back to bisection when Newton would overshoot, so deep ITM/OTM "
                    "quotes still converge.</p>", unsafe_allow_html=True)
    with right:
        vols = np.linspace(0.01, 1.5, 300)
        fig = go.Figure()
        line(fig, vols * 100, bs.price(S, K, T, r, vols, option_type, q), "Model price", AMBER)
        fig.add_hline(y=mkt, line=dict(color=BLUE, dash="dash"),
                      annotation_text="observed", annotation_font_color=BLUE)
        if np.isfinite(iv):
            fig.add_trace(go.Scatter(x=[iv * 100], y=[mkt], mode="markers", name="Solution",
                                     marker=dict(size=11, color=FG, symbol="circle-open",
                                                 line=dict(width=2))))
        st.plotly_chart(style_fig(fig, "Price is monotone in σ, so the root is unique",
                                  "Volatility (%)", "Option price"), width="stretch")

# --- Tab 5: smile lab --------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SNAPSHOT = "2026-10-06"


@st.cache_data(show_spinner=False)
def spy_snapshot():
    meta = dict(line.split("=", 1) for line in
                (DATA_DIR / f"spy_chain_{SNAPSHOT}.meta").read_text().splitlines())
    surf = pd.read_csv(DATA_DIR / f"spy_surface_{SNAPSHOT}.csv")
    return surf, float(meta["spot"]), float(meta["rate"].split()[0])


@st.cache_data(show_spinner=False, ttl=900)
def live_surface(ticker, american):
    chain, spot, as_of = market.fetch_chain(ticker)
    rate = market.fetch_rate()
    surf = market.build_surface(market.clean_chain(chain), spot, rate=rate, american=american)
    return surf, spot, rate, as_of


@st.cache_data(show_spinner=False)
def synthetic_quotes(p_tuple, noise, S, r, q):
    p = heston.HestonParams(*p_tuple)
    rng = np.random.default_rng(42)
    rows = []
    for d in (30, 90, 180, 365):
        t = d / 365
        ks = S * np.exp(np.linspace(-0.35, 0.25, 13) * np.sqrt(max(t, 0.25)))
        vols = heston.implied_vol_smile(S, ks, t, r, p, q)
        vols = vols + rng.normal(0, noise / 100, vols.size)
        rows += [(kk, d, v) for kk, v in zip(ks, vols, strict=True) if np.isfinite(v)]
    return pd.DataFrame(rows, columns=["strike", "days", "iv"])


def with_forwards(quotes, spot, r, q):
    """Add T, forward and log-moneyness assuming a flat carry r - q."""
    out = quotes.copy()
    out["T"] = out["days"] / 365.0
    out["forward"] = spot * np.exp((r - q) * out["T"])
    out["k"] = np.log(out["strike"] / out["forward"])
    return out


@st.cache_data(show_spinner=False)
def run_heston(quotes, spot, rate, n_starts):
    idx = []
    for _, s in quotes.groupby("days"):
        idx += list(s.index[np.unique(np.linspace(0, len(s) - 1, min(25, len(s))).astype(int))])
    sub = quotes.loc[idx]
    q_vec = (rate - np.log(sub["forward"] / spot) / sub["T"]).to_numpy()
    return heston.calibrate(spot, sub["strike"].to_numpy(), sub["T"].to_numpy(),
                            sub["iv"].to_numpy(), rate, q=q_vec, n_starts=n_starts)


def render_smile_lab():
    # A function so early exits can `return`: return would also halt the
    # tabs rendered after this one.
    st.markdown(
        '<p class="caption">Black-Scholes assumes one volatility for every strike; real markets '
        "show a smile. Fit <b>SVI</b> per expiry (best fit, no cross-expiry guarantee), "
        "<b>SSVI</b> across expiries (arbitrage-free by construction) and <b>Heston</b> "
        "(a stochastic-vol model), then view the <b>Dupire local vol</b> implied by SSVI.</p>",
        unsafe_allow_html=True,
    )
    source = st.radio("Quotes", ["Synthetic (Heston + noise)", f"SPY snapshot ({SNAPSHOT})",
                                 "Live chain (Yahoo Finance)", "Upload CSV"], horizontal=True)
    true_p = None
    if source.startswith("Synthetic"):
        cols = st.columns(6)
        v0 = cols[0].slider("v₀ (√ %)", 5.0, 60.0, 20.0, 0.5) / 100
        theta_h = cols[1].slider("θ (√ %)", 5.0, 60.0, 24.0, 0.5) / 100
        kappa = cols[2].slider("κ", 0.1, 10.0, 1.5, 0.1)
        xi = cols[3].slider("ξ vol of vol", 0.05, 2.0, 0.6, 0.05)
        rho = cols[4].slider("ρ", -0.95, 0.95, -0.65, 0.05)
        noise = cols[5].slider("Quote noise (vol pts)", 0.0, 2.0, 0.3, 0.1)
        true_p = heston.HestonParams(v0**2, kappa, theta_h**2, xi, rho)
        raw = synthetic_quotes(tuple(true_p.as_dict().values()), noise, S, r, q)
        quotes, spot_u, rate_u = with_forwards(raw, S, r, q), S, r
        st.markdown('<p class="caption">Spot, r and q come from the sidebar.</p>',
                    unsafe_allow_html=True)
    elif source.startswith("SPY"):
        quotes, spot_u, rate_u = spy_snapshot()
        st.markdown(
            f'<p class="caption">{len(quotes):,} out-of-the-money quotes, spot '
            f'<span class="mono">{spot_u:.2f}</span>, rate <span class="mono">{rate_u:.3%}</span>. '
            "Forwards come from put-call parity and every quote is de-Americanized (SPY options "
            "are American); see <code>docs/spy-study.md</code>. Sidebar inputs are not used.</p>",
            unsafe_allow_html=True)
    elif source.startswith("Live"):
        c1, c2, c3 = st.columns([1, 1, 2])
        ticker = c1.text_input("Ticker", "SPY").strip().upper()
        american = c2.checkbox("De-Americanize", value=False,
                               help="Accurate for American-style options but ~30 s per chain.")
        if c3.button("Fetch chain", type="primary"):
            try:
                with st.spinner(f"Downloading {ticker} options and building the surface…"):
                    st.session_state["live"] = live_surface(ticker, american)
            except Exception as exc:  # network / ticker errors are shown, not raised
                st.error(f"Could not build a surface for {ticker}: {exc}")
        if "live" not in st.session_state:
            st.info("Press **Fetch chain** to load live quotes.")
            return
        quotes, spot_u, rate_u, as_of = st.session_state["live"]
        st.markdown(f'<p class="caption">{len(quotes):,} quotes as of {as_of}, spot '
                    f'<span class="mono">{spot_u:.2f}</span>, rate '
                    f'<span class="mono">{rate_u:.3%}</span> (13-week T-bill).</p>',
                    unsafe_allow_html=True)
    else:
        upload = st.file_uploader("CSV with columns strike, days, iv (iv in %)", type="csv")
        if upload is None:
            st.info("Upload a CSV to continue. Spot, r and q come from the sidebar.")
            return
        raw = pd.read_csv(upload)
        missing = {"strike", "days", "iv"} - set(raw.columns)
        if missing:
            st.error(f"CSV is missing column(s): {', '.join(sorted(missing))}")
            return
        raw = raw[["strike", "days", "iv"]].dropna()
        raw = raw[(raw["strike"] > 0) & (raw["days"] > 0) & (raw["iv"] > 0)]
        raw["iv"] = raw["iv"] / 100
        quotes, spot_u, rate_u = with_forwards(raw, S, r, q), S, r

    quotes = quotes.groupby("days").filter(lambda s: len(s) >= 5).reset_index(drop=True)
    if quotes.empty:
        st.warning("No expiry has the 5 quotes SVI needs.")
        return

    ssvi_fit = fit_ssvi(quotes["k"], quotes["T"], quotes["iv"])
    surf_s = ssvi_fit.surface

    hc1, hc2 = st.columns([1, 3])
    n_starts = hc1.select_slider("Heston starts", [1, 2, 4, 8], value=4)
    if hc2.button("Calibrate Heston", type="primary"):
        with st.spinner(f"Calibrating Heston from {n_starts} starting point(s)…"):
            st.session_state["calib"] = (source, run_heston(quotes, spot_u, rate_u, n_starts))
    calib = None
    if st.session_state.get("calib") and st.session_state["calib"][0] == source:
        calib = st.session_state["calib"][1]

    expiries = sorted(quotes["days"].unique())
    plot_cols = st.columns(2)
    rows = []
    for i, d in enumerate(expiries):
        sl = quotes[quotes["days"] == d].sort_values("strike")
        t, fwd_t = sl["T"].iloc[0], sl["forward"].iloc[0]
        k = sl["k"].to_numpy()
        fit = svi.fit(k, sl["iv"].to_numpy(), t)
        pad = 0.15 * max(np.ptp(k), 0.05)
        k_in = np.linspace(k.min(), k.max(), 150)
        k_lo = np.linspace(k.min() - pad, k.min(), 30)
        k_hi = np.linspace(k.max(), k.max() + pad, 30)
        k_all = np.linspace(k.min() - pad, k.max() + pad, 200)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sl["strike"], y=sl["iv"] * 100, mode="markers", name="Quotes",
                                 marker=dict(color=FG, size=6, symbol="circle-open")))
        line(fig, fwd_t * np.exp(k_in), svi.implied_vol(k_in, t, fit.params) * 100, "SVI", AMBER)
        for kx, show in ((k_lo, True), (k_hi, False)):
            fig.add_trace(go.Scatter(
                x=fwd_t * np.exp(kx), y=svi.implied_vol(kx, t, fit.params) * 100,
                mode="lines", name="SVI (extrapolated)", showlegend=show,
                line=dict(color=AMBER, dash="dot", width=1.4)))
        line(fig, fwd_t * np.exp(k_all), surf_s.implied_vol(k_all, t) * 100, "SSVI", BLUE, 1)
        hs_rmse = np.nan
        q_t = rate_u - np.log(fwd_t / spot_u) / t
        if calib is not None:
            hv = heston.implied_vol_smile(spot_u, fwd_t * np.exp(k_all), t, rate_u,
                                          calib.params, q_t)
            line(fig, fwd_t * np.exp(k_all), hv * 100, "Heston", VIOLET, 2)
            hq = heston.implied_vol_smile(spot_u, sl["strike"].to_numpy(), t, rate_u,
                                          calib.params, q_t)
            hs_rmse = float(np.sqrt(np.nanmean((hq - sl["iv"].to_numpy()) ** 2))) * 100
        if true_p is not None:
            tv = heston.implied_vol_smile(spot_u, fwd_t * np.exp(k_all), t, rate_u, true_p, q)
            line(fig, fwd_t * np.exp(k_all), tv * 100, "True (generator)", MUTED, 3, 1.2)
        rows.append((int(d), len(sl), fit.rmse_vol * 100, "yes" if fit.arbitrage_free else "no",
                     ssvi_fit.rmse_by_expiry[float(t)] * 100, hs_rmse))
        with plot_cols[i % 2]:
            st.plotly_chart(style_fig(fig, f"{int(d)}-day smile", "Strike", "Implied vol (%)",
                                      340), width="stretch")

    st.markdown("#### Fit quality (RMSE in vol points)")
    table = pd.DataFrame(rows, columns=["days", "quotes", "SVI", "SVI butterfly-free", "SSVI",
                                        "Heston"])
    st.dataframe(table.style.format({"SVI": "{:.3f}", "SSVI": "{:.3f}", "Heston": "{:.3f}"},
                                    na_rep="—"), hide_index=True, width="stretch")
    k_wide = np.linspace(quotes["k"].min() - 0.3, quotes["k"].max() + 0.3, 400)
    slice_list = [(float(t), svi.fit(s["k"].to_numpy(), s["iv"].to_numpy(), float(t)).params)
                  for t, s in quotes.groupby("T")]
    crossed = [c for c in svi.calendar_crossings(slice_list, k_wide) if c[1] > 0]
    cal_txt = ("no SVI slices cross" if not crossed else
               f"{len(crossed)} adjacent SVI pair(s) cross in the wings (calendar arbitrage)")

    left, right = st.columns(2)
    with left:
        st.markdown("#### SSVI surface")
        cond = surf_s.no_arbitrage_conditions()
        st.markdown(
            f'<span class="badge">ρ {surf_s.rho:.3f}</span><span class="badge">η '
            f'{surf_s.eta:.3f}</span><span class="badge">γ {surf_s.gamma:.3f}</span>'
            + "".join(f'<span class="badge {"ok" if v else "warn"}">{n}</span>'
                      for n, v in cond.items()),
            unsafe_allow_html=True)
        st.markdown(f'<p class="caption">Overall RMSE <span class="mono">'
                    f'{ssvi_fit.rmse_vol * 100:.3f}</span> vol pts. For comparison, {cal_txt}.</p>',
                    unsafe_allow_html=True)
    with right:
        st.markdown("#### Heston")
        if calib is None:
            st.markdown('<p class="caption">Press <b>Calibrate Heston</b> above.</p>',
                        unsafe_allow_html=True)
        else:
            names = ["v₀", "κ", "θ", "ξ", "ρ"]
            vals = list(calib.params.as_dict().values())
            truth = list(true_p.as_dict().values()) if true_p is not None else [None] * 5
            st.dataframe(pd.DataFrame(
                [(n, fmt(v, 4), fmt(t_, 4) if t_ is not None else "—")
                 for n, v, t_ in zip(names, vals, truth, strict=True)],
                columns=["param", "calibrated", "generator"]), hide_index=True, width="stretch")
            feller = "satisfied" if calib.params.feller_satisfied() else "violated"
            st.markdown(f'<span class="badge">RMSE {calib.rmse_vol * 100:.3f} vol pts</span>'
                        f'<span class="badge">Feller {feller}</span>'
                        f'<span class="badge">{calib.n_starts} start(s)</span>',
                        unsafe_allow_html=True)

    st.markdown("#### Dupire local volatility from the SSVI surface")
    t_grid = np.linspace(quotes["T"].min(), quotes["T"].max(), 40)
    k_grid = np.linspace(np.percentile(quotes["k"], 5), np.percentile(quotes["k"], 95), 60)
    lv = np.array([surf_s.local_vol(k_grid, tt) for tt in t_grid]) * 100
    # Clip the colour range: a few extreme short-dated wing values would
    # otherwise flatten the rest of the surface into a single colour.
    lo_c, hi_c = np.nanpercentile(lv, [2, 98])
    fig = go.Figure(go.Heatmap(
        x=k_grid, y=t_grid * 365, z=lv, zmin=lo_c, zmax=hi_c,
        colorscale=[[0, "#1E3A8A"], [0.5, BLUE], [1, AMBER]],
        colorbar=dict(title="σ_loc %", tickfont=dict(family="Fira Code")),
        hovertemplate="k %{x:.3f}<br>%{y:.0f} days<br>local vol %{z:.2f}%<extra></extra>"))
    fig = style_fig(fig, "σ_loc(k, t) = √(∂w/∂t ÷ g(k, t))", "Log-moneyness k = ln(K/F)",
                    "Days", 420)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, width="stretch")
    st.markdown('<p class="caption">Local vol is what a single-factor model needs at each spot '
                "and time to reproduce every vanilla price; it is only well defined because "
                "the SSVI surface has no arbitrage (g > 0, w increasing in t).</p>",
                unsafe_allow_html=True)


with tabs[4]:
    render_smile_lab()

# --- Tab 6: exotics ----------------------------------------------------------
with tabs[5]:
    st.markdown(
        '<p class="caption">Continuously monitored single-barrier options (Reiner-Rubinstein '
        "closed form) on the sidebar contract, checked against a Brownian-bridge Monte Carlo."
        "</p>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    kind = b1.selectbox("Barrier type", list(KINDS), index=2)
    default_h = round(S * (0.9 if kind.startswith("down") else 1.1), 2)
    H = b2.number_input("Barrier H", min_value=0.01, value=default_h, step=1.0,
                        key=f"barrier_{kind}")
    rebate = b3.number_input("Cash rebate", min_value=0.0, value=0.0, step=0.5)
    breached = (kind.startswith("down") and S <= H) or (kind.startswith("up") and S >= H)
    bp = barrier_price(S, K, H, T, r, sigma, option_type, kind, q, rebate)
    vanilla = bs.price(S, K, T, r, sigma, option_type, q)
    twin = kind.replace("-in", "-out") if kind.endswith("-in") else kind.replace("-out", "-in")
    m = st.columns(4)
    m[0].metric(f"{kind} {option_type}", fmt(bp))
    m[1].metric("Vanilla", fmt(vanilla))
    m[2].metric(f"{twin} (no rebate)", fmt(barrier_price(S, K, H, T, r, sigma, option_type,
                                                         twin, q)))
    m[3].metric("Barrier discount", f"{(1 - bp / vanilla) * 100:.1f}%" if vanilla > 0 else "—")
    if breached:
        st.warning("The barrier is already breached at today's spot.")
    if rebate > 0:
        st.caption("The Monte Carlo check prices the option without a rebate, so it is "
                   "disabled while a rebate is set.")
    elif st.button("Check with Monte Carlo (200k paths)"):
        mc, se = barrier_mc(S, K, H, T, r, sigma, option_type, kind, q, seed=7)
        z = (mc - bp) / se if se > 0 else 0.0
        st.markdown(f'<span class="badge">MC {mc:.4f} ± {1.96 * se:.4f} (95%)</span>'
                    f'<span class="badge {"ok" if abs(z) < 3 else "warn"}">z = {z:+.2f}</span>',
                    unsafe_allow_html=True)
    spots = np.linspace(S * 0.6, S * 1.4, 161)
    fig = go.Figure()
    line(fig, spots, [barrier_price(s, K, H, T, r, sigma, option_type, kind, q, rebate)
                      for s in spots], f"{kind}", AMBER)
    line(fig, spots, bs.price(spots, K, T, r, sigma, option_type, q), "Vanilla", BLUE, 1)
    fig.add_vline(x=H, line=dict(color=LOSS, dash="dash"), annotation_text="barrier",
                  annotation_font_color=LOSS)
    fig.add_vline(x=S, line=dict(color=BORDER, dash="dot"))
    st.plotly_chart(style_fig(fig, "Price vs spot", "Spot", "Option value", 400),
                    width="stretch")

# --- Tab 7: convergence ------------------------------------------------------
with tabs[6]:
    st.markdown(
        '<p class="caption">Why the Leisen-Reimer tree is the default: pricing the current '
        "contract as a <b>European</b> option on each tree and comparing with the exact "
        "Black-Scholes value. CRR error oscillates and shrinks like 1/n; LR shrinks like "
        "1/n².</p>",
        unsafe_allow_html=True,
    )

    @st.cache_data(show_spinner=False)
    def convergence(S, K, T, r, sigma, option_type, q):
        exact = bs.price(S, K, T, r, sigma, option_type, q)
        ns = np.unique(np.geomspace(11, 1001, 40).astype(int) | 1)
        crr = [abs(european_tree_price(S, K, T, r, sigma, option_type, q, n, "crr") - exact)
               for n in ns]
        lr = [abs(european_tree_price(S, K, T, r, sigma, option_type, q, n, "lr") - exact)
              for n in ns]
        return ns, np.array(crr), np.array(lr)

    ns, crr_err, lr_err = convergence(S, K, T, r, sigma, option_type, q)
    fig = go.Figure()
    line(fig, ns, np.maximum(crr_err, 1e-16), "CRR", BLUE, 1)
    line(fig, ns, np.maximum(lr_err, 1e-16), "Leisen-Reimer", AMBER)
    fig = style_fig(fig, "Absolute pricing error vs number of steps", "Steps (log)",
                    "|tree − Black-Scholes| (log)", 420)
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log", exponentformat="e")
    st.plotly_chart(fig, width="stretch")
