"""
AI / Tech Sector Rotation — Relative Rotation Graph (RRG)

Run with:
    streamlit run rrg_app.py

Measures each custom sub-theme basket against QQQ on two axes:
  RS-Ratio     (x) — normalized relative strength vs the benchmark
  RS-Momentum  (y) — rate of change of that relative strength

Both are centered on 100, producing four quadrants. Baskets generally
travel clockwise: Improving -> Leading -> Weakening -> Lagging.

Note on the math: the original JdK RS-Ratio / RS-Momentum formulas are
proprietary. This is the standard open reconstruction (smoothed relative
strength, then a rolling z-score re-centered at 100). Tail shapes and
quadrant crossings match the commercial version closely; exact coordinates
will not.
"""

import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Sector Rotation — RRG", layout="wide")

BENCHMARK = "QQQ"

DEFAULT_BASKETS = {
    "Memory": ["MU", "WDC", "STX", "SNDK"],
    "AI Semis": ["NVDA", "AMD", "AVGO", "MRVL"],
    "Semicap": ["AMAT", "LRCX", "KLAC", "ASML", "TER"],
    "Foundry": ["TSM", "INTC", "GFS", "UMC"],
    "Hyperscalers": ["MSFT", "GOOGL", "AMZN", "META", "ORCL"],
    "Neoclouds": ["CRWV", "NBIS", "IREN", "APLD", "CIFR", "WULF"],
    "Networking & Optics": ["ANET", "CRDO", "ALAB", "COHR", "LITE", "CIEN"],
    "DC Hardware": ["DELL", "SMCI", "HPE", "PSTG", "NTAP"],
    "AI Power": ["VRT", "GEV", "CEG", "VST", "TLN", "ETN"],
    "AI Software": ["PLTR", "NOW", "SNOW", "DDOG", "MDB", "CRM"],
}

QUADRANTS = {
    "Leading": "#3f9c6a",
    "Weakening": "#c9a227",
    "Lagging": "#c05a52",
    "Improving": "#4a7fb5",
}

PALETTE = [
    "#e0a458", "#5aa9c9", "#c96a6a", "#7fb069", "#b48ead",
    "#d98e5a", "#6f9ec9", "#c0a76a", "#8f7fb0", "#6ab5a0",
]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _download_chunk(chunk: list, period: str) -> pd.DataFrame:
    """One sequential request. Returns only columns that came back with data."""
    raw = yf.download(
        chunk, period=period, interval="1d", auto_adjust=True,
        progress=False, threads=False,
    )
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        out = raw["Close"].copy()
    else:
        out = raw[["Close"]].copy()
        out.columns = [chunk[0]]

    return out.dropna(axis=1, how="all")


@st.cache_data(ttl=900, show_spinner=False)
def fetch_prices(tickers: tuple, period: str = "5y", chunk_size: int = 6):
    """Adjusted daily closes, fetched in small sequential batches with retries.

    Yahoo throttles concurrent requests and returns empty columns rather than
    raising, so a plain threaded yf.download silently drops a third of the
    universe. Chunking sequentially with backoff gets essentially everything.

    Returns (prices, failed_tickers).
    """
    collected: dict = {}
    outstanding = list(tickers)

    for attempt in range(3):
        if not outstanding:
            break
        still_missing = []
        for i in range(0, len(outstanding), chunk_size):
            chunk = outstanding[i:i + chunk_size]
            try:
                got = _download_chunk(chunk, period)
            except Exception:
                got = pd.DataFrame()
            for t in chunk:
                if t in got.columns and got[t].notna().any():
                    collected[t] = got[t]
                else:
                    still_missing.append(t)
            time.sleep(0.4)
        outstanding = still_missing
        if outstanding:
            time.sleep(2.0 * (attempt + 1))  # back off before retrying

    if not collected:
        return pd.DataFrame(), list(tickers)

    px = pd.DataFrame(collected).dropna(how="all")
    px.index = pd.to_datetime(px.index)
    return px, outstanding


def basket_index(px: pd.DataFrame, members: list) -> pd.Series:
    """Equal-weight total-return index, rebased to 100.

    Each day averages the returns of whichever members have data, so names
    that IPO'd mid-history (CRWV, NBIS) join the basket without wrecking it.
    """
    cols = [t for t in members if t in px.columns]
    if not cols:
        return pd.Series(dtype=float)

    rets = px[cols].pct_change()
    breadth = rets.notna().sum(axis=1)
    ew = rets.mean(axis=1, skipna=True)
    ew = ew.where(breadth > 0)

    first = ew.first_valid_index()
    if first is None:
        return pd.Series(dtype=float)

    ew = ew.loc[first:].fillna(0.0)
    return 100.0 * (1.0 + ew).cumprod()


# --------------------------------------------------------------------------
# RRG math
# --------------------------------------------------------------------------

def _warmup(window: int) -> int:
    """Half a window is enough to start. Without this, three stacked rolling
    windows eat ~3 years of weekly history and anything that listed recently
    (CRWV, NBIS) never plots at all."""
    return max(4, window // 2)


def _zscore(s: pd.Series, window: int) -> pd.Series:
    mp = _warmup(window)
    mean = s.rolling(window, min_periods=mp).mean()
    std = s.rolling(window, min_periods=mp).std(ddof=0)
    return (s - mean) / std.replace(0.0, np.nan)


def rrg_coordinates(basket: pd.Series, bench: pd.Series,
                    long_w: int, short_w: int, norm_w: int):
    """Return (rs_ratio, rs_momentum), both centered on 100."""
    joined = pd.concat([basket, bench], axis=1, join="inner").dropna()
    if joined.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    rs = 100.0 * joined.iloc[:, 0] / joined.iloc[:, 1]

    rs_ratio_raw = 100.0 * rs / rs.rolling(long_w, min_periods=_warmup(long_w)).mean()
    rs_ratio = 100.0 + _zscore(rs_ratio_raw, norm_w)

    rs_mom_raw = 100.0 * rs_ratio / rs_ratio.rolling(
        short_w, min_periods=_warmup(short_w)).mean()
    rs_mom = 100.0 + _zscore(rs_mom_raw, norm_w)

    return rs_ratio, rs_mom


def quadrant_of(x: float, y: float) -> str:
    if x >= 100 and y >= 100:
        return "Leading"
    if x >= 100:
        return "Weakening"
    if y >= 100:
        return "Improving"
    return "Lagging"


def heading_degrees(dx: float, dy: float) -> float:
    """Compass bearing of the tail's last step. 0 = due north, 90 = due east."""
    if dx == 0 and dy == 0:
        return float("nan")
    return (np.degrees(np.arctan2(dx, dy)) + 360) % 360


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.header("Setup")

timeframe = st.sidebar.radio(
    "Timeframe",
    ["Weekly", "Daily"],
    help="Weekly is the RRG standard — slower, cleaner rotation. "
         "Daily reacts within days but whipsaws.",
)

if timeframe == "Weekly":
    tail_len = st.sidebar.slider("Tail length (weeks)", 4, 26, 12)
    long_w, short_w, norm_w = 26, 13, 52
    period = "6y"
else:
    tail_len = st.sidebar.slider("Tail length (days)", 5, 40, 10)
    long_w, short_w, norm_w = 50, 20, 100
    period = "3y"

with st.sidebar.expander("Smoothing windows"):
    long_w = st.number_input("RS-Ratio lookback", 5, 260, long_w)
    short_w = st.number_input("RS-Momentum lookback", 3, 130, short_w)
    norm_w = st.number_input("Normalization window", 20, 500, norm_w)

with st.sidebar.expander("Edit baskets"):
    st.caption("JSON: basket name → list of tickers.")
    baskets_text = st.text_area(
        "Baskets", json.dumps(DEFAULT_BASKETS, indent=2), height=280,
        label_visibility="collapsed",
    )
    try:
        baskets = json.loads(baskets_text)
    except json.JSONDecodeError as exc:
        st.error(f"Can't parse that JSON: {exc.msg} (line {exc.lineno})")
        baskets = DEFAULT_BASKETS

selected = st.sidebar.multiselect(
    "Baskets on chart", list(baskets), default=list(baskets)
)

if st.sidebar.button("Refresh prices"):
    fetch_prices.clear()

st.sidebar.caption(
    "Prices are Yahoo daily adjusted closes, cached 15 minutes. "
    "Intraday quotes are delayed, so treat the last point as provisional "
    "until the session closes."
)


# --------------------------------------------------------------------------
# Load and compute
# --------------------------------------------------------------------------

st.title("Where the money is rotating")
st.caption(f"AI and tech sub-themes measured against {BENCHMARK} · "
           f"{timeframe.lower()} bars, {tail_len}-period tails")

if not selected:
    st.info("Pick at least one basket in the sidebar.")
    st.stop()

all_tickers = sorted({t for b in selected for t in baskets[b]} | {BENCHMARK})

with st.spinner(f"Pulling prices for {len(all_tickers)} tickers…"):
    px, failed = fetch_prices(tuple(all_tickers), period=period)

if px.empty or BENCHMARK not in px.columns:
    st.error(
        "No price data came back — Yahoo is most likely throttling this IP. "
        "Wait a few minutes before hitting Refresh prices; retrying immediately "
        "makes it worse."
    )
    st.stop()

if failed:
    hurt = {b: [t for t in baskets[b] if t in failed] for b in selected}
    hurt = {b: t for b, t in hurt.items() if t}
    st.warning(
        f"Still missing after 3 retries: {', '.join(sorted(failed))}. "
        "These are valid tickers, so this is Yahoo rate limiting rather than "
        "bad symbols — the affected baskets below are incomplete and their "
        "positions should not be read as meaningful."
    )
    for b, t in hurt.items():
        st.caption(f"　{b}: missing {', '.join(t)} of {len(baskets[b])} members")

if timeframe == "Weekly":
    px = px.resample("W-FRI").last().dropna(how="all")

bench = px[BENCHMARK].dropna()

coords = {}
for name in selected:
    idx = basket_index(px, baskets[name])
    if idx.empty:
        continue
    x, y = rrg_coordinates(idx, bench, long_w, short_w, norm_w)
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) >= 2:
        coords[name] = frame

if not coords:
    st.error(
        "Not enough history to normalize. Shorten the normalization window "
        "or switch to daily bars."
    )
    st.stop()

young = [n for n, f in coords.items() if len(f) < max(long_w, norm_w) // 2]
if young:
    st.info(
        f"Short history, so coordinates are still stabilizing: {', '.join(young)}. "
        "Their tails will drift more than the rest."
    )

# Union, not intersection — one recently-listed basket shouldn't clip everyone
# else's history down to its own.
common = sorted(set().union(*(set(f.index) for f in coords.values())))
if len(common) < 2:
    st.error("Not enough overlapping history to plot.")
    st.stop()

as_of = st.select_slider(
    "As of",
    options=common,
    value=common[-1],
    format_func=lambda d: f"{d:%d %b %y}",
    help="Drag back to replay how the rotation developed.",
)
as_of_idx = common.index(as_of)
st.caption(f"Showing {as_of:%d %b %Y}"
           + ("" if as_of_idx == len(common) - 1 else "  ·  historical replay"))


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------

tails = {}
for name, frame in coords.items():
    upto = frame.loc[:as_of]
    if len(upto) >= 2:
        tails[name] = upto.iloc[-tail_len:]

if not tails:
    st.warning("No basket has data as of that date. Drag the slider forward.")
    st.stop()

pad = 0.6
xs = np.concatenate([t["x"].values for t in tails.values()])
ys = np.concatenate([t["y"].values for t in tails.values()])
xr = [min(xs.min(), 100) - pad, max(xs.max(), 100) + pad]
yr = [min(ys.min(), 100) - pad, max(ys.max(), 100) + pad]

fig = go.Figure()

for label, (x0, x1, y0, y1) in {
    "Leading": (100, xr[1], 100, yr[1]),
    "Weakening": (100, xr[1], yr[0], 100),
    "Lagging": (xr[0], 100, yr[0], 100),
    "Improving": (xr[0], 100, 100, yr[1]),
}.items():
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                  fillcolor=QUADRANTS[label], opacity=0.07,
                  line_width=0, layer="below")
    fig.add_annotation(
        x=(x0 + x1) / 2, y=y1 - (y1 - y0) * 0.04 if y1 > 100 else y0 + (y1 - y0) * 0.04,
        text=label, showarrow=False,
        font=dict(size=13, color=QUADRANTS[label]), opacity=0.75,
    )

fig.add_hline(y=100, line_width=1, line_color="rgba(140,140,150,0.5)")
fig.add_vline(x=100, line_width=1, line_color="rgba(140,140,150,0.5)")

for i, (name, tail) in enumerate(tails.items()):
    color = PALETTE[i % len(PALETTE)]
    fig.add_trace(go.Scatter(
        x=tail["x"], y=tail["y"], mode="lines+markers",
        line=dict(color=color, width=1.6),
        marker=dict(size=4, color=color, opacity=0.55),
        name=name, legendgroup=name,
        hovertemplate=f"<b>{name}</b><br>%{{customdata|%d %b %Y}}"
                      "<br>RS-Ratio %{x:.2f}<br>RS-Mom %{y:.2f}<extra></extra>",
        customdata=tail.index,
    ))
    last = tail.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[last["x"]], y=[last["y"]], mode="markers+text",
        marker=dict(size=13, color=color, line=dict(color="white", width=1.5)),
        text=[f" {name}"], textposition="middle right",
        textfont=dict(size=12, color=color),
        legendgroup=name, showlegend=False, hoverinfo="skip",
    ))

fig.update_layout(
    height=680,
    xaxis=dict(title="RS-Ratio  →  relative strength", range=xr,
               zeroline=False, gridcolor="rgba(128,128,128,0.12)"),
    yaxis=dict(title="RS-Momentum  →  strength accelerating", range=yr,
               zeroline=False, gridcolor="rgba(128,128,128,0.12)"),
    margin=dict(l=60, r=90, t=20, b=50),
    legend=dict(orientation="h", y=-0.13, x=0),
    hovermode="closest",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)

st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------

rows = []
for name, tail in tails.items():
    last, prev = tail.iloc[-1], tail.iloc[-2]
    quad = quadrant_of(last["x"], last["y"])
    prev_quad = quadrant_of(prev["x"], prev["y"])
    dx, dy = last["x"] - prev["x"], last["y"] - prev["y"]
    rows.append({
        "Basket": name,
        "Quadrant": quad,
        "Just crossed": prev_quad if quad != prev_quad else "",
        "RS-Ratio": round(last["x"], 2),
        "RS-Mom": round(last["y"], 2),
        "Heading": "" if np.isnan(heading_degrees(dx, dy)) else f"{heading_degrees(dx, dy):.0f}°",
        "Speed": round(float(np.hypot(dx, dy)), 2),
        "Distance from center": round(float(np.hypot(last["x"] - 100, last["y"] - 100)), 2),
    })

table = pd.DataFrame(rows).sort_values(
    ["Quadrant", "RS-Mom"], ascending=[True, False]
)

left, right = st.columns([3, 2])

with left:
    st.subheader("Standings")
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(
        "Heading is the compass bearing of the last step: 0° is straight up, "
        "90° is due east. Speed is how far the basket moved this period — "
        "long tails moving fast are the ones actually being repositioned."
    )

with right:
    st.subheader("Rotating in")
    improving = table[table["Quadrant"] == "Improving"].sort_values(
        "RS-Mom", ascending=False
    )
    fresh = table[(table["Quadrant"] == "Leading") & (table["Just crossed"] == "Improving")]

    if len(fresh):
        for _, r in fresh.iterrows():
            st.success(f"**{r['Basket']}** crossed Improving → Leading")
    if len(improving):
        for _, r in improving.iterrows():
            st.write(f"**{r['Basket']}** — RS-Mom {r['RS-Mom']}, heading {r['Heading']}")
    if not len(fresh) and not len(improving):
        st.write("Nothing in the Improving quadrant. Money is sitting still, "
                 "or already fully rotated.")

    st.subheader("Rolling over")
    weakening = table[
        (table["Quadrant"] == "Weakening") & (table["Just crossed"] == "Leading")
    ]
    if len(weakening):
        for _, r in weakening.iterrows():
            st.warning(f"**{r['Basket']}** dropped Leading → Weakening")
    else:
        st.write("No fresh breakdowns out of Leading.")


with st.expander("How to read this"):
    st.markdown(
        """
Position tells you where a basket stands; the tail tells you where it's going.

- **Improving** (top left) — lagging QQQ but momentum has turned up. This is
  where a rotation shows up first, before the performance tables notice.
- **Leading** (top right) — outperforming and still accelerating. Crowded, but
  the trend is intact.
- **Weakening** (bottom right) — still outperforming, momentum fading. Money is
  leaving even though the returns still look fine.
- **Lagging** (bottom left) — underperforming with negative momentum.

Baskets normally travel clockwise. A basket that reverses back into the quadrant
it came from is a failed rotation, and those are common — treat a single-period
crossing as noise until the tail extends through it.

Because the benchmark is QQQ, everything is measured against the AI trade as a
whole, not the market. A basket in Lagging can still be up 20% on the year; it
just didn't keep up with the index it's part of.
        """
    )
