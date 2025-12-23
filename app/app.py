import streamlit as st

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Gender Gap in Sports Performance", layout="wide")

st.title("Women Do Better in Sports")
st.header("Exploring the narrowing performance gap between men and women")

st.markdown(
    """
Women’s performance has accelerated dramatically across many sports, especially as access, participation, coaching,
and professionalization have expanded. In several disciplines, women’s historical improvement rate has been steeper
than men’s, narrowing the performance gap over time.

This dashboard explores and **quantifies** that gap by combining record progressions with model-based forecasts, 
measuring how much faster women’s performance has improved relative to men’s across different sports.

**How to use the dashboard**  
The main chart focuses on a single discipline and allows you to toggle historical data, forecasts, regression trends,
and gender-gap indicators. Filters can be used to restrict the grid by category and subcategory, while sorting options
 highlight events where women’s performance has improved faster than men’s.

Below the main chart, the grid view provides a comparative overview across all disciplines.
"""
)

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -----------------------------
# Config
# -----------------------------
CURRENT_YEAR = 2025
PRED_START_YEAR = 2026

# Pick non-stereotypical, distinguishable colors
COLOR_WOMEN = "#2ca02c"      # green
COLOR_WOMEN_FILL = "rgba(44,160,44,0.15)"
COLOR_MEN = "#ff7f0e"    # orange
COLOR_MEN_FILL = "rgba(255,127,14,0.15)"
COLOR_COMPARE = "#9467bd"  # purple
COLOR_WOMEN_ANNOT = "#3ddc84"  # lighter green for annotations
COLOR_GAP_ANNOT = "#b07edf"  # lighter purple for annotations

# Rounding for matching prediction record values to df_combined record values
MATCH_ROUND_DECIMALS = 3

PLOTLY_CONFIG = {
    "displayModeBar": False,   # hides zoom/pan/download/etc
    "displaylogo": True,
}

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_combined(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Standardize
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.lower()
    if "event" in df.columns:
        df["event"] = df["event"].astype(str)
    if "measure" in df.columns:
        df["measure"] = df["measure"].astype(str).str.lower()

    return df


@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["sex"] = df["sex"].astype(str).str.lower()
    df["event"] = df["event"].astype(str)
    if "measure" in df.columns:
        df["measure"] = df["measure"].astype(str).str.lower()
    return df


def perf_col_from_measure(measure: str) -> str:
    m = (measure or "").lower().strip()
    return "time_seconds" if m == "time" else "mark_meters"


def raw_col_from_measure(measure: str) -> str:
    m = (measure or "").lower().strip()
    return "time_raw" if m == "time" else "mark_raw"


def build_yearly_record_series(dfp_event_sex: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a complete yearly series for historical best progression using y_hist,
    forward-filled, for clean plotting.

    Returns a DF with columns: year, y_hist_ffill, y_pred
    """
    d = dfp_event_sex.sort_values("year").copy()
    d["y_hist_ffill"] = d["y_hist"].ffill()
    return d[["year", "y_hist_ffill", "y_pred"]]


def build_record_lookup_from_combined(
    df_combined: pd.DataFrame,
    event: str,
    sex: str,
    measure: str,
) -> dict:
    """
    Build a lookup from rounded performance value -> (record_date, record_year, athletes_str, raw_string)
    """
    perf_col = perf_col_from_measure(measure)
    raw_col = raw_col_from_measure(measure)

    d = df_combined[
        (df_combined["event"] == event)
        & (df_combined["sex"] == sex)
        & (df_combined["measure"] == measure.lower())
    ].copy()

    d = d.dropna(subset=[perf_col, "date"])
    if d.empty:
        return {}

    d["perf_round"] = pd.to_numeric(d[perf_col], errors="coerce").round(MATCH_ROUND_DECIMALS)
    d = d.dropna(subset=["perf_round"])

    lookup = {}
    for perf_val, g in d.groupby("perf_round"):
        g_sorted = g.sort_values("date")
        first_date = g_sorted.iloc[0]["date"]
        same_date = g_sorted[g_sorted["date"] == first_date]

        athletes = sorted({str(a) for a in same_date["athlete"].dropna().tolist() if str(a).strip()})
        athletes_str = ", ".join(athletes) if athletes else "Unknown"

        raw_vals = [str(x) for x in same_date[raw_col].dropna().tolist() if str(x).strip()]
        raw_str = raw_vals[0] if raw_vals else "Unknown"

        lookup[float(perf_val)] = {
            "record_date": first_date.strftime("%Y-%m-%d"),
            "record_year": int(first_date.year),
            "athletes": athletes_str,
            "raw": raw_str,
        }

    return lookup


def attach_hover_metadata(
    yearly_series: pd.DataFrame,
    record_lookup: dict,
) -> pd.DataFrame:
    """
    Add hover columns: record_date, record_year_obtained, athletes, raw
    by matching rounded y_hist_ffill to df_combined lookup.
    """
    d = yearly_series.copy()
    d["perf_round"] = pd.to_numeric(d["y_hist_ffill"], errors="coerce").round(MATCH_ROUND_DECIMALS)

    dates, years, athletes, raw = [], [], [], []

    for v in d["perf_round"].tolist():
        info = record_lookup.get(float(v)) if pd.notna(v) else None
        if info:
            dates.append(info["record_date"])
            years.append(info["record_year"])
            athletes.append(info["athletes"])
            raw.append(info["raw"])
        else:
            dates.append("Unknown")
            years.append(np.nan)
            athletes.append("Unknown")
            raw.append("Unknown")

    d["record_date"] = dates
    d["record_year_obtained"] = years
    d["athletes"] = athletes
    d["raw"] = raw
    return d


def first_valid_year_and_value(d: pd.DataFrame):
    dd = d.dropna(subset=["y_hist_ffill"]).sort_values("year")
    if dd.empty:
        return None, None
    return int(dd.iloc[0]["year"]), float(dd.iloc[0]["y_hist_ffill"])


def year_value(d: pd.DataFrame, year: int):
    dd = d[d["year"] == year].dropna(subset=["y_hist_ffill"])
    if dd.empty:
        return None
    return float(dd.iloc[0]["y_hist_ffill"])


def find_crossing_year(men_hist: pd.DataFrame, women_value_2025: float, measure: str):
    """
    time: lower is better => women reached if men >= women_value
    mark: higher is better => women reached if men <= women_value
    """
    d = men_hist.dropna(subset=["y_hist_ffill"]).sort_values("year")
    if d.empty:
        return None

    if measure == "time":
        crossed = d[d["y_hist_ffill"] >= women_value_2025]
    else:
        crossed = d[d["y_hist_ffill"] <= women_value_2025]

    if crossed.empty:
        return None
    return int(crossed.iloc[-1]["year"])


def decade_ticks(min_year: int, max_year: int):
    start = (min_year // 10) * 10
    end = ((max_year + 9) // 10) * 10
    return list(range(start, end + 1, 10))

# Load data
df_combined = load_combined("data/processed/all_events_results_clean.csv")
df_predictions = load_predictions("data/predictions/predictions_normalized_near_ceiling.csv")


def sort_events_custom(events: list[str]) -> list[str]:
    events = [e for e in events if isinstance(e, str)]

    # 1) Custom athletics order (your explicit list)
    athletics_order = [
        "100m", "4x100m_relays", "200m", "400m", "4x400m_relays",
        "800m", "1500m", "3000m", "5000m", "10000m",
        "half_marathon", "marathon",
        "high_jump", "long_jump", "triple_jump", "pole_vault",
    ]
    athletics_rank = {e: i for i, e in enumerate(athletics_order)}

    def is_swim(e: str) -> bool:
        return e.lower().startswith("swimming_") or e.lower().startswith("swim")

    # 2) Swimming sort: stroke, then distances (50,100,200,400,800,1500), then relay last
    swim_dist_rank = {"50m": 0, "100m": 1, "200m": 2, "400m": 3, "800m": 4, "1500m": 5}
    stroke_rank = {
        "freestyle": 0,
        "backstroke": 1,
        "breaststroke": 2,
        "butterfly": 3,
        "medley": 4,
    }

    def swim_key(e: str):
        el = e.lower()

        # Try to extract distance token like "50m", "100m", ...
        dist = None
        for d in swim_dist_rank.keys():
            if d in el:
                dist = d
                break
        dist_i = swim_dist_rank.get(dist, 99)

        is_relay = 1 if "relay" in el else 0

        # Stroke
        stroke_i = 99
        for s, idx in stroke_rank.items():
            if s in el:
                stroke_i = idx
                break

        return (stroke_i, dist_i, is_relay, el)

    def key(e: str):
        el = e.lower()

        # athletics explicit order first
        if e in athletics_rank:
            return (0, athletics_rank[e], el)

        # then swimming
        if is_swim(e):
            return (1, ) + swim_key(e)

        # then everything else alphabetically
        return (2, el)

    return sorted(events, key=key)


available_events = sort_events_custom(df_predictions["event"].dropna().unique().tolist())

available_models = sorted(df_predictions["model"].dropna().unique().tolist()) if "model" in df_predictions.columns else ["model"]

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# Top filter "box"
# First row: event and model
with st.container():
    c1, c2 = st.columns([3, 3])
    with c1:
        event = st.selectbox(
            "Discipline",
            available_events,
            index=available_events.index("100m") if "100m" in available_events else 0,
        )
    with c2:
        model_name = st.selectbox("Prediction Model", available_models, index=0)

# Second row: checkboxes
with st.container():
    c3, c4, c5, c6 = st.columns([2, 2, 2, 2])
    with c3:
        show_gap_line = st.checkbox("Show men–women gap line", value=True)
    with c4:
        show_future_pred = st.checkbox("Show predictions (>2025)", value=True)
    with c5:
        show_full_pred = st.checkbox("Show model fit (≤2025)", value=False)
    with c6:
        show_regression = st.checkbox("Show regression slopes", value=False)


# Increase space
st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


# Filter predictions to event/model
dfp = df_predictions[df_predictions["event"] == event].copy()
if "model" in dfp.columns:
    dfp = dfp[dfp["model"] == model_name].copy()

# Guardrails
needed_cols = {"event", "sex", "year", "y_hist", "y_pred", "measure"}
missing = needed_cols - set(dfp.columns)
if missing:
    st.error(f"df_predictions is missing required columns: {sorted(missing)}")
    st.stop()

# Measure (assume consistent within event)
measure = str(dfp["measure"].dropna().iloc[0]).lower() if dfp["measure"].notna().any() else "time"

# Series
dfp_m = dfp[dfp["sex"] == "men"]
dfp_w = dfp[dfp["sex"] == "women"]

men_series = build_yearly_record_series(dfp_m)
women_series = build_yearly_record_series(dfp_w)

# Historical window up to current year
men_hist = men_series[men_series["year"] <= CURRENT_YEAR].copy()
women_hist = women_series[women_series["year"] <= CURRENT_YEAR].copy()

# Predictions from next year onward
men_pred_future = men_series[men_series["year"] >= PRED_START_YEAR].copy()
women_pred_future = women_series[women_series["year"] >= PRED_START_YEAR].copy()

# Lookups for hover (raw + athlete metadata)
men_lookup = build_record_lookup_from_combined(df_combined, event=event, sex="men", measure=measure)
women_lookup = build_record_lookup_from_combined(df_combined, event=event, sex="women", measure=measure)

men_hist_h = attach_hover_metadata(men_hist, men_lookup)
women_hist_h = attach_hover_metadata(women_hist, women_lookup)

# Crossing point based on women's 2025 value
women_2025_val = year_value(women_hist_h, CURRENT_YEAR)
men_first_year, _ = first_valid_year_and_value(men_hist_h)
cross_year = None
if women_2025_val is not None and men_first_year is not None:
    cross_year = find_crossing_year(men_hist_h, women_2025_val, measure=measure)

# X ticks every 10 years
min_year = int(dfp["year"].dropna().min())
max_year = int(dfp["year"].dropna().max())
tickvals = decade_ticks(min_year, max_year)

# --- Y-axis zoom to center the lines ---
all_y = pd.concat(
    [
        men_hist_h["y_hist_ffill"],
        women_hist_h["y_hist_ffill"],
        men_pred_future["y_pred"] if not men_pred_future.empty else pd.Series(dtype=float),
        women_pred_future["y_pred"] if not women_pred_future.empty else pd.Series(dtype=float),
    ]
).dropna()

y_min, y_max = all_y.min(), all_y.max()
padding = 0.15 * (y_max - y_min)  # adjust zoom strength here (e.g. 0.1–0.2)
y_low = y_min - padding
y_high = y_max + padding
y_range_zoomed = max(y_high - y_low, 1e-9)
y_offset = 0.06 * y_range_zoomed

# -----------------------------
# Plotly figure
# -----------------------------
fig = go.Figure()

# Women historical (filled)
fig.add_trace(
    go.Scatter(
        x=women_hist_h["year"].astype(int),
        y=women_hist_h["y_hist_ffill"],
        mode="lines",
        name=f"Women — record progression (to {CURRENT_YEAR})",
        line=dict(color=COLOR_WOMEN, width=3),
        fill="tozeroy",
        fillcolor=COLOR_WOMEN_FILL,
        customdata=np.stack(
            [
                women_hist_h["record_date"],
                women_hist_h["record_year_obtained"].fillna(-1).astype(int),
                women_hist_h["athletes"],
                women_hist_h["raw"],
            ],
            axis=1,
        ),
        hovertemplate=(
            "<b>Women</b><br>"
            # "Year: %{x}<br>"
            "Record: %{customdata[3]}<br>"
            "Obtained: %{customdata[0]} (year %{customdata[1]})<br>"
            "Athlete(s): %{customdata[2]}<br>"
            "<extra></extra>"
        ),
    )
)

# Men historical (filled)
fig.add_trace(
    go.Scatter(
        x=men_hist_h["year"].astype(int),
        y=men_hist_h["y_hist_ffill"],
        mode="lines",
        name=f"Men — record progression (to {CURRENT_YEAR})",
        line=dict(color=COLOR_MEN, width=3),
        fill="tozeroy",
        fillcolor=COLOR_MEN_FILL,
        customdata=np.stack(
            [
                men_hist_h["record_date"],
                men_hist_h["record_year_obtained"].fillna(-1).astype(int),
                men_hist_h["athletes"],
                men_hist_h["raw"],
            ],
            axis=1,
        ),
        hovertemplate=(
            "<b>Men</b><br>"
            # "Year: %{x}<br>"
            "Record: %{customdata[3]}<br>"
            "Obtained: %{customdata[0]} (year %{customdata[1]})<br>"
            "Athlete(s): %{customdata[2]}<br>"
            "<extra></extra>"
        ),
    )
)

# Forecasts (dashed)
# + Optional full prediction line (≤2025) dashed, but NON-interactive
if show_full_pred:
    # women pre-2025 (non-interactive)
    w_pre = women_series[women_series["year"] <= CURRENT_YEAR].copy()
    if not w_pre.empty:
        fig.add_trace(
            go.Scatter(
                x=w_pre["year"].astype(int),
                y=w_pre["y_pred"],
                mode="lines",
                name="Women — model fit (≤2025)",
                line=dict(color=COLOR_WOMEN, width=2, dash="dash"),
                hoverinfo="skip",
                opacity=0.7,
            )
        )

    # men pre-2025 (non-interactive)
    m_pre = men_series[men_series["year"] <= CURRENT_YEAR].copy()
    if not m_pre.empty:
        fig.add_trace(
            go.Scatter(
                x=m_pre["year"].astype(int),
                y=m_pre["y_pred"],
                mode="lines",
                name="Men — model fit (≤2025)",
                line=dict(color=COLOR_MEN, width=2, dash="dash"),
                hoverinfo="skip",
                opacity=0.7,
            )
        )

# Future forecasts (interactive)
if show_future_pred and not women_pred_future.empty:
    fig.add_trace(
        go.Scatter(
            x=women_pred_future["year"].astype(int),
            y=women_pred_future["y_pred"],
            mode="lines",
            name="Women — forecast (2026+)",
            line=dict(color=COLOR_WOMEN, width=2, dash="dash"),
            hovertemplate=(
                "<b>Women forecast</b><br>"
                # "Year: %{x}<br>"
                "Predicted: %{y:.3f}<br>"
                "<extra></extra>"),
            hoverinfo="text",
        )
    )
if show_future_pred and not men_pred_future.empty:
    fig.add_trace(
        go.Scatter(
            x=men_pred_future["year"].astype(int),
            y=men_pred_future["y_pred"],
            mode="lines",
            name="Men — forecast (2026+)",
            line=dict(color=COLOR_MEN, width=2, dash="dash"),
            hovertemplate=(
                "<b>Men forecast</b><br>"
                # "Year: %{x}<br>"
                "Predicted: %{y:.3f}<br>"
                "<extra></extra>"),
            hoverinfo="text",
        )
    )

# Vertical connector lines: first record -> x-axis (y=0)
def first_valid(d: pd.DataFrame):
    dd = d.dropna(subset=["y_hist_ffill"]).sort_values("year")
    if dd.empty:
        return None, None
    return int(dd.iloc[0]["year"]), float(dd.iloc[0]["y_hist_ffill"])

def add_vertical_connector(x_year: int, y_val: float, color: str):
    if x_year is None or y_val is None:
        return
    fig.add_shape(
        type="line",
        x0=x_year, x1=x_year,
        y0=0, y1=y_val,
        line=dict(color=color, width=1),
        opacity=0.6,
    )

men_first_y, men_first_v = first_valid(men_hist_h)
women_first_y, women_first_v = first_valid(women_hist_h)

add_vertical_connector(men_first_y, men_first_v, COLOR_MEN)
add_vertical_connector(women_first_y, women_first_v, COLOR_WOMEN)

# Vertical connector lines: 2025 record -> x-axis (y=0)
men_2025_val = year_value(men_hist_h, CURRENT_YEAR)
women_2025_val = year_value(women_hist_h, CURRENT_YEAR)

# Year when the current women record was set
women_record_year = None
if women_2025_val is not None:
    rec_row = women_hist_h[women_hist_h["y_hist_ffill"] == women_2025_val]
    if not rec_row.empty:
        women_record_year = int(rec_row.iloc[0]["record_year_obtained"])

add_vertical_connector(CURRENT_YEAR, men_2025_val, COLOR_MEN)
add_vertical_connector(CURRENT_YEAR, women_2025_val, COLOR_WOMEN)

# Visual-only regression slopes (first record -> 2025)
if show_regression:

    def add_regression_line(hist_df: pd.DataFrame, sex_label: str, color: str):
        # Use the "best-so-far" series up to 2025
        d = hist_df.dropna(subset=["year", "y_hist_ffill"]).sort_values("year")
        d = d[d["year"] <= CURRENT_YEAR]

        if d.empty:
            return

        # Fit a linear regression y = m*x + b using all available years up to 2025
        x = d["year"].astype(float).to_numpy()
        y = d["y_hist_ffill"].astype(float).to_numpy()

        # Need at least 2 points
        if len(x) < 2:
            return

        m, b = np.polyfit(x, y, 1)  # slope, intercept

        # Plot the fitted line over the historical range
        x0 = int(d["year"].min())
        x1 = CURRENT_YEAR
        x_line = [x0, x1]
        y_line = [m * x0 + b, m * x1 + b]

        y_2025 = m * CURRENT_YEAR + b
        slope = m  # slope per year

        # Add dashed regression line
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"{sex_label} — slope (visual)",
                line=dict(color=color, width=3, dash="dot"),
                hoverinfo="skip",  # visual aid only
                opacity=0.95,
                showlegend=True,
            )
        )

        # # Small label near 2025 endpoint with slope
        # # For time: negative slope means improving (going down); for marks: positive slope means improving (going up)
        # slope_text = f"Improvement slope:<br>{slope:+.4f} per year"

        # # Offset = a fraction of the visible range (works for sprints + marathon)
        # y_offset = 0.06 * y_range_zoomed   # tweak 0.05–0.08
        # y_lab_regression = (y_2025 - y_offset) if measure == "time" else (y_2025 + y_offset)
        
        # # Place label slightly left of CURRENT_YEAR
        # x_label = CURRENT_YEAR + 15
        # annot_color = COLOR_WOMEN_ANNOT if sex_label == "Women" else color

        # fig.add_annotation(
        #     x=x_label,
        #     y=y_lab_regression,
        #     text=slope_text,
        #     showarrow=False,
        #     font=dict(color=annot_color, size=14),
        #     opacity=0.9,
        # )

    add_regression_line(women_hist_h, "Women", COLOR_WOMEN)
    add_regression_line(men_hist_h, "Men", COLOR_MEN)



# Horizontal comparison line + crossing point marker + better label placement
if show_gap_line and women_2025_val is not None and men_first_y is not None:

    # Point where it reaches women's record level (at crossing year)
    if women_record_year is not None:
        fig.add_trace(
            go.Scatter(
                x=[women_record_year],
                y=[women_2025_val],
                mode="markers",
                marker=dict(size=10, color=COLOR_COMPARE),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Put label above the women's 2025 point
    # y_offset = (0.025 * abs(women_2025_val)) if measure == "time" else (-0.06 * abs(women_2025_val) if women_2025_val != 0 else 0.06)

    if cross_year is None:
        # No crossing: dashed line from first men's record year to 2025
        fig.add_shape(
            type="line",
            x0=men_first_y, x1=women_record_year,
            y0=women_2025_val, y1=women_2025_val,
            line=dict(color=COLOR_COMPARE, width=3, dash="dash"),
            opacity=0.9,
        )

        fig.add_annotation(
            x=women_record_year + 10,
            y=women_2025_val + y_offset,
            text=f"Women have not<br>surpassed men yet",
            showarrow=False,
            font=dict(color=COLOR_GAP_ANNOT, size=14),
        )
    else:
        # Crossing exists: solid line from crossing year to 2025
        fig.add_shape(
            type="line",
            x0=cross_year, x1=women_record_year,
            y0=women_2025_val, y1=women_2025_val,
            line=dict(color=COLOR_COMPARE, width=3),
            opacity=0.9,
        )

        # Point where it reaches men's record level (at crossing year)
        fig.add_trace(
            go.Scatter(
                x=[cross_year],
                y=[women_2025_val],
                mode="markers",
                marker=dict(size=10, color=COLOR_COMPARE),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Vertical purple line at crossing year down to y=0
        if cross_year is not None:
            fig.add_shape(
                type="line",
                x0=cross_year, x1=cross_year,
                y0=0, y1=women_2025_val,
                line=dict(color=COLOR_COMPARE, width=1),  # thin purple line
                opacity=0.8,
            )

        # Vertical purple line at crossing year down to y=0
        if cross_year is not None:
            fig.add_shape(
                type="line",
                x0=women_record_year, x1=women_record_year,
                y0=0, y1=women_2025_val,
                line=dict(color=COLOR_COMPARE, width=1),  # thin purple line
                opacity=0.8,
            )

        # Label: to the right and slightly higher to avoid intersection
        # For time, "higher" => slightly smaller; for marks, "higher" => slightly bigger
        if cross_year is not None:
            fig.add_annotation(
                x=women_record_year + 10,
                y=women_2025_val + y_offset,
                text=f"In {women_record_year} women surpass<br>men from {cross_year}",
                showarrow=False,
                font=dict(color=COLOR_GAP_ANNOT, size=14),
            )


# Layout: vertical gridlines + decade ticks
y_title = "Time (seconds)" if measure == "time" else "Mark (meters)"
fig.update_layout(
    height=600,
    margin=dict(l=40, r=40, t=40, b=40),
    xaxis_title="Year",
    yaxis_title=y_title,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=14)),
)

fig.update_xaxes(
    tickmode="array",
    tickvals=tickvals,
    showgrid=True,
    gridwidth=1,
    griddash="dot",
)

# --- Y-axis zoom to center the lines ---
fig.update_yaxes(range=[y_min - padding, y_max + padding])

# (Optional) keep y grid subtle too
# fig.update_yaxes(showgrid=True, gridwidth=1, griddash="dot")

fig.update_layout(
    title=dict(
        text=f"{event} record progression and forecasts",
        x=0.5,
        xanchor="center",
        y=0.97,
        yanchor="top",
        font=dict(size=24),
    ),
    margin=dict(t=120)  # ensure room for the title
)

st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG, key="main_plot")

st.caption(
    "Hover to see the raw record value (from the progression tables), the date it was obtained, and the athlete(s). "
    "The optional dashed 'model fit' (≤2025) is non-interactive."
)



# ============================================================
# GRID VIEW — all disciplines at once
# ============================================================

@st.cache_data(show_spinner=False)
def compute_event_metrics(df_predictions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    dfp = df_predictions.copy()
    if "model" in dfp.columns:
        dfp = dfp[dfp["model"] == model_name].copy()

    for ev in dfp["event"].dropna().unique():
        dfp_ev = dfp[dfp["event"] == ev]
        if dfp_ev.empty:
            continue

        measure_ev = str(dfp_ev["measure"].dropna().iloc[0]).lower() if dfp_ev["measure"].notna().any() else "time"

        dfp_m = dfp_ev[dfp_ev["sex"] == "men"]
        dfp_w = dfp_ev[dfp_ev["sex"] == "women"]

        men_series = build_yearly_record_series(dfp_m)
        women_series = build_yearly_record_series(dfp_w)

        men_hist = men_series[men_series["year"] <= CURRENT_YEAR].copy()
        women_hist = women_series[women_series["year"] <= CURRENT_YEAR].copy()

        men_slope_imp = compute_improvement_slope(men_hist, measure_ev)
        women_slope_imp = compute_improvement_slope(women_hist, measure_ev)

        cross_year, _, _ = compute_cross_year_for_event(dfp_ev, ev)

        women_better = None
        if men_slope_imp is not None and women_slope_imp is not None:
            women_better = women_slope_imp > men_slope_imp

        rel_adv = None
        if (
            men_slope_imp is not None
            and women_slope_imp is not None
            and not pd.isna(men_slope_imp)
            and abs(men_slope_imp) > 0
        ):
            rel_adv = (women_slope_imp - men_slope_imp) / abs(men_slope_imp) * 100


        rows.append(
            dict(
                event=ev,
                model=model_name,
                measure=measure_ev,
                men_improvement_slope=men_slope_imp,
                women_improvement_slope=women_slope_imp,
                improvement_diff=women_slope_imp - men_slope_imp,
                improvement_relative_pct=rel_adv,   # ← NEW
                women_better=women_better,
                cross_year=cross_year,
            )
        )

    return pd.DataFrame(rows)

st.divider()
st.header("How much faster do women improve compared to men?")

st.markdown(
    """
This grid view compares how much faster women improve compared to men, in proportion to men’s rate.
Each panel summarizes the historical record progression for a single discipline, shown for women and men using regression slopes. 
Optional elements (forecasts and gap indicators) can be toggled globally to support rapid comparison across events. 
Sorting by relative improvement highlights disciplines where women’s performance has accelerated faster than men’s, while crossing filters identify events where women have reached or surpassed historical men’s records.
"""
)

# -----------------------------
# Helpers for grid computations
# -----------------------------
def safe_event_meta(df_combined: pd.DataFrame, df_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Returns event-level metadata: category, subcategory, measure.
    Prefers df_combined if it contains category/subcategory; otherwise falls back to 'Unknown'.
    """
    meta_cols = ["event"]
    has_cat = "category" in df_combined.columns
    has_sub = "subcategory" in df_combined.columns
    has_measure_combined = "measure" in df_combined.columns

    # Start from df_predictions to guarantee we include all predicted events
    base = df_predictions[["event"]].dropna().drop_duplicates().copy()

    # Merge category/subcategory/measure from df_combined if available
    if has_cat or has_sub or has_measure_combined:
        keep = ["event"]
        if has_cat:
            keep.append("category")
        if has_sub:
            keep.append("subcategory")
        if has_measure_combined:
            keep.append("measure")
        comb_meta = df_combined[keep].dropna(subset=["event"]).drop_duplicates()

        base = base.merge(comb_meta, on="event", how="left")

    # Fill missing
    if "category" not in base.columns:
        base["category"] = "Unknown"
    else:
        base["category"] = base["category"].fillna("Unknown").astype(str)

    if "subcategory" not in base.columns:
        base["subcategory"] = "Unknown"
    else:
        base["subcategory"] = base["subcategory"].fillna("Unknown").astype(str)

    # Ensure measure exists: prefer df_predictions measure (assumed consistent per event)
    if "measure" not in base.columns or base["measure"].isna().all():
        pred_meas = (
            df_predictions[["event", "measure"]]
            .dropna(subset=["event"])
            .drop_duplicates()
        )
        base = base.drop(columns=["measure"], errors="ignore").merge(pred_meas, on="event", how="left")

    base["measure"] = base["measure"].fillna("time").astype(str).str.lower()
    return base


def compute_improvement_slope(hist_df: pd.DataFrame, measure: str) -> float | None:
    """
    Returns an *improvement* slope score:
      - time: improvement = -slope (since lower is better)
      - mark: improvement = +slope (since higher is better)
    Uses linear fit on y_hist_ffill up to CURRENT_YEAR.
    """
    d = hist_df.dropna(subset=["year", "y_hist_ffill"]).sort_values("year")
    d = d[d["year"] <= CURRENT_YEAR]
    if len(d) < 2:
        return None

    x = d["year"].astype(float).to_numpy()
    y = d["y_hist_ffill"].astype(float).to_numpy()
    slope, _ = np.polyfit(x, y, 1)

    if measure == "time":
        return float(-slope)  # more positive = faster improvement
    return float(slope)      # more positive = faster improvement


def compute_cross_year_for_event(dfp_event: pd.DataFrame, event: str) -> tuple[int | None, float | None, str]:
    """
    Computes cross_year for this event using the same logic as the main plot:
    based on women's CURRENT_YEAR value vs men's historical progression.
    Returns: (cross_year, women_2025_val, measure)
    """
    if dfp_event.empty:
        return None, None, "time"

    measure = str(dfp_event["measure"].dropna().iloc[0]).lower() if dfp_event["measure"].notna().any() else "time"

    dfp_m = dfp_event[dfp_event["sex"] == "men"]
    dfp_w = dfp_event[dfp_event["sex"] == "women"]

    men_series = build_yearly_record_series(dfp_m)
    women_series = build_yearly_record_series(dfp_w)

    men_hist = men_series[men_series["year"] <= CURRENT_YEAR].copy()
    women_hist = women_series[women_series["year"] <= CURRENT_YEAR].copy()

    men_lookup = build_record_lookup_from_combined(df_combined, event=event, sex="men", measure=measure)
    women_lookup = build_record_lookup_from_combined(df_combined, event=event, sex="women", measure=measure)

    men_hist_h = attach_hover_metadata(men_hist, men_lookup)
    women_hist_h = attach_hover_metadata(women_hist, women_lookup)

    women_2025_val = year_value(women_hist_h, CURRENT_YEAR)
    men_first_year, _ = first_valid_year_and_value(men_hist_h)

    if women_2025_val is None or men_first_year is None:
        return None, women_2025_val, measure

    cross_year = find_crossing_year(men_hist_h, women_2025_val, measure=measure)
    return cross_year, women_2025_val, measure


def ticks_every_25_years(min_year: int, max_year: int, must_include: int = CURRENT_YEAR):
    start = (min_year // 25) * 25
    end = ((max_year + 24) // 25) * 25
    ticks = list(range(start, end + 1, 25))
    if must_include not in ticks:
        ticks.append(must_include)
    return sorted(set(ticks))


def make_event_figure(
    event: str,
    model_name: str,
    show_gap_line: bool,
    show_future_pred: bool,
    show_full_pred: bool,
    show_regression: bool,
    show_history: bool,  # NEW
) -> go.Figure | None:
    """
    Mini-plot builder for the grid. Very similar to the main plot, but:
    - smaller height
    - simpler legend defaults
    """
    dfp_e = df_predictions[df_predictions["event"] == event].copy()
    if "model" in dfp_e.columns:
        dfp_e = dfp_e[dfp_e["model"] == model_name].copy()

    needed_cols = {"event", "sex", "year", "y_hist", "y_pred", "measure"}
    if (needed_cols - set(dfp_e.columns)):
        return None

    measure = str(dfp_e["measure"].dropna().iloc[0]).lower() if dfp_e["measure"].notna().any() else "time"

    dfp_m = dfp_e[dfp_e["sex"] == "men"]
    dfp_w = dfp_e[dfp_e["sex"] == "women"]

    men_series = build_yearly_record_series(dfp_m)
    women_series = build_yearly_record_series(dfp_w)

    men_hist = men_series[men_series["year"] <= CURRENT_YEAR].copy()
    women_hist = women_series[women_series["year"] <= CURRENT_YEAR].copy()

    men_pred_future = men_series[men_series["year"] >= PRED_START_YEAR].copy()
    women_pred_future = women_series[women_series["year"] >= PRED_START_YEAR].copy()

    men_lookup = build_record_lookup_from_combined(df_combined, event=event, sex="men", measure=measure)
    women_lookup = build_record_lookup_from_combined(df_combined, event=event, sex="women", measure=measure)

    men_hist_h = attach_hover_metadata(men_hist, men_lookup)
    women_hist_h = attach_hover_metadata(women_hist, women_lookup)

    women_2025_val = year_value(women_hist_h, CURRENT_YEAR)
    
    # Year when the current women record was set
    women_record_year = None
    if women_2025_val is not None:
        rec_row = women_hist_h[women_hist_h["y_hist_ffill"] == women_2025_val]
        if not rec_row.empty:
            women_record_year = int(rec_row.iloc[0]["record_year_obtained"])

    men_first_y, _ = first_valid_year_and_value(men_hist_h)
    cross_year = None
    if women_2025_val is not None and men_first_y is not None:
        cross_year = find_crossing_year(men_hist_h, women_2025_val, measure=measure)

    # Ticks
    min_year = int(dfp_e["year"].dropna().min())
    max_year = int(dfp_e["year"].dropna().max())
    tickvals = ticks_every_25_years(min_year, max_year, must_include=CURRENT_YEAR)

    fig = go.Figure()

    if show_history:
        # Women historical
        fig.add_trace(
            go.Scatter(
                x=women_hist_h["year"].astype(int),
                y=women_hist_h["y_hist_ffill"],
                mode="lines",
                name="Women",
                line=dict(color=COLOR_WOMEN, width=2),
                fill="tozeroy",
                fillcolor=COLOR_WOMEN_FILL,
                customdata=np.stack(
                    [
                        women_hist_h["record_date"],
                        women_hist_h["record_year_obtained"].fillna(-1).astype(int),
                        women_hist_h["athletes"],
                        women_hist_h["raw"],
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>Women</b><br>"
                    "Record: %{customdata[3]}<br>"
                    "Obtained: %{customdata[1]}<br>"
                    "<extra></extra>"
                ),
            )
        )

        # Men historical
        fig.add_trace(
            go.Scatter(
                x=men_hist_h["year"].astype(int),
                y=men_hist_h["y_hist_ffill"],
                mode="lines",
                name="Men",
                line=dict(color=COLOR_MEN, width=2),
                fill="tozeroy",
                fillcolor=COLOR_MEN_FILL,
                customdata=np.stack(
                    [
                        men_hist_h["record_date"],
                        men_hist_h["record_year_obtained"].fillna(-1).astype(int),
                        men_hist_h["athletes"],
                        men_hist_h["raw"],
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>Men</b><br>"
                    "Record: %{customdata[3]}<br>"
                    "Obtained: %{customdata[1]}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Optional full prediction (≤2025) non-interactive
    if show_full_pred:
        w_pre = women_series[women_series["year"] <= CURRENT_YEAR].copy()
        if not w_pre.empty:
            fig.add_trace(
                go.Scatter(
                    x=w_pre["year"].astype(int),
                    y=w_pre["y_pred"],
                    mode="lines",
                    name="Women fit",
                    line=dict(color=COLOR_WOMEN, width=1.5, dash="dash"),
                    hoverinfo="skip",
                    opacity=0.7,
                    showlegend=False,
                )
            )
        m_pre = men_series[men_series["year"] <= CURRENT_YEAR].copy()
        if not m_pre.empty:
            fig.add_trace(
                go.Scatter(
                    x=m_pre["year"].astype(int),
                    y=m_pre["y_pred"],
                    mode="lines",
                    name="Men fit",
                    line=dict(color=COLOR_MEN, width=1.5, dash="dash"),
                    hoverinfo="skip",
                    opacity=0.7,
                    showlegend=False,
                )
            )

    # Future forecasts
    if show_future_pred and not women_pred_future.empty:
        fig.add_trace(
            go.Scatter(
                x=women_pred_future["year"].astype(int),
                y=women_pred_future["y_pred"],
                mode="lines",
                name="Women forecast",
                line=dict(color=COLOR_WOMEN, width=1.5, dash="dash"),
                hovertemplate="<b>Women forecast</b><br>Predicted: %{y:.3f}<extra></extra>",
                showlegend=False,
            )
        )
    if show_future_pred and not men_pred_future.empty:
        fig.add_trace(
            go.Scatter(
                x=men_pred_future["year"].astype(int),
                y=men_pred_future["y_pred"],
                mode="lines",
                name="Men forecast",
                line=dict(color=COLOR_MEN, width=1.5, dash="dash"),
                hovertemplate="<b>Men forecast</b><br>Predicted: %{y:.3f}<extra></extra>",
                showlegend=False,
            )
        )

    # Optional regression (visual)
    if show_regression:

        def add_reg_line(hist_df: pd.DataFrame, color: str):
            d = hist_df.dropna(subset=["year", "y_hist_ffill"]).sort_values("year")
            d = d[d["year"] <= CURRENT_YEAR]
            if len(d) < 2:
                return
            x = d["year"].astype(float).to_numpy()
            y = d["y_hist_ffill"].astype(float).to_numpy()
            m, b = np.polyfit(x, y, 1)
            x0 = int(d["year"].min())
            x1 = CURRENT_YEAR
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[m * x0 + b, m * x1 + b],
                    mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                    opacity=0.95,
                )
            )

        add_reg_line(women_hist_h, COLOR_WOMEN)
        add_reg_line(men_hist_h, COLOR_MEN)

    # Optional gap line & crossing marker
    if show_gap_line and women_2025_val is not None and men_first_y is not None:
        if women_record_year is not None:
            fig.add_trace(
                go.Scatter(
                    x=[women_record_year],
                    y=[women_2025_val],
                        mode="markers",
                        marker=dict(size=7, color=COLOR_COMPARE),
                        hoverinfo="skip",
                        showlegend=False
                    )
            )
        if cross_year is None:
            if women_record_year is not None:
                fig.add_shape(
                    type="line",
                    x0=men_first_y, x1=women_record_year,
                    y0=women_2025_val, y1=women_2025_val,
                    line=dict(color=COLOR_COMPARE, width=2, dash="dash"),
                    opacity=0.9,
                )
        else:
            if women_record_year is not None:
                fig.add_shape(
                    type="line",
                    x0=cross_year, x1=women_record_year,
                    y0=women_2025_val, y1=women_2025_val,
                    line=dict(color=COLOR_COMPARE, width=2),
                    opacity=0.9,
                )
            fig.add_trace(
                go.Scatter(
                    x=[cross_year],
                    y=[women_2025_val],
                    mode="markers",
                    marker=dict(size=7, color=COLOR_COMPARE),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Zoom y
    all_y = pd.concat(
        [
            men_hist_h["y_hist_ffill"],
            women_hist_h["y_hist_ffill"],
            men_pred_future["y_pred"] if not men_pred_future.empty else pd.Series(dtype=float),
            women_pred_future["y_pred"] if not women_pred_future.empty else pd.Series(dtype=float),
        ]
    ).dropna()

    if not all_y.empty:
        y_min, y_max = all_y.min(), all_y.max()
        pad = 0.15 * (y_max - y_min) if (y_max - y_min) != 0 else 0.15 * abs(y_max if y_max != 0 else 1.0)
        fig.update_yaxes(range=[y_min - pad, y_max + pad])

    y_title = "Time (s)" if measure == "time" else "Mark (m)"
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=10, t=60, b=5),
        title=dict(text=event, x=0.02, xanchor="left", y=0.98, yanchor="top", font=dict(size=16)),
        xaxis_title="",
        yaxis_title=y_title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        showgrid=True,
        gridwidth=1,
        griddash="dot",
    )
    return fig


# -----------------------------
# Build metadata and controls
# -----------------------------
event_meta = safe_event_meta(df_combined, df_predictions)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# -----------------------------
# Row 1: category + subcategory + crossing
# -----------------------------
r1c1, r1c2, r1c3 = st.columns([3, 3, 4])

with r1c1:
    categories = ["All"] + sorted(event_meta["category"].dropna().unique().tolist())
    grid_category = st.selectbox("Category (grid)", categories, index=0)

with r1c2:
    if grid_category == "All":
        subcats = ["All"] + sorted(event_meta["subcategory"].dropna().unique().tolist())
    else:
        subcats = ["All"] + sorted(
            event_meta[event_meta["category"] == grid_category]["subcategory"].dropna().unique().tolist()
        )
    grid_subcategory = st.selectbox("Subcategory (grid)", subcats, index=0)

with r1c3:
    cross_filter = st.selectbox(
        "Crossing filter",
        ["All", "Only women reached men (cross_year exists)", "Only women NOT reached men (cross_year is None)"],
        index=0,
    )

# -----------------------------
# Row 2: grid toggles (show/hide)
# -----------------------------
r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns([2, 2, 2, 2, 2])

with r2c1:
    grid_show_history = st.checkbox("Grid: show historical data", value=True)
with r2c2:
    grid_show_regression = st.checkbox("Grid: show regression slopes", value=True)
with r2c3:
    grid_show_gap_line = st.checkbox("Grid: show gap line", value=False)
with r2c4:
    grid_show_future_pred = st.checkbox("Grid: show predictions (2026+)", value=False)
with r2c5:
    grid_show_full_pred = st.checkbox("Grid: show model fit (≤2025)", value=show_full_pred)

# -----------------------------
# Row 3: sorting + layout
# -----------------------------
r3c1, r3c2 = st.columns([3, 2])

with r3c1:
    sort_women_better = st.checkbox(
        "Sort disciplines by women improving faster than men",
        value=False,
        help="Uses linear regression slope on historical best-so-far up to 2025 (converted to an improvement score) and calculates the difference of improvement rates between women and men in percentage.",
    )

with r3c2:
    grid_cols = st.slider("Grid columns", min_value=2, max_value=5, value=4, step=1)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# -----------------------------
# Filter events list
# -----------------------------
meta_f = event_meta.copy()

if grid_category != "All":
    meta_f = meta_f[meta_f["category"] == grid_category]
if grid_subcategory != "All":
    meta_f = meta_f[meta_f["subcategory"] == grid_subcategory]

grid_events = meta_f["event"].dropna().unique().tolist()

# -----------------------------
# Compute per-event stats for sorting / filtering
# -----------------------------
stats = compute_event_metrics(df_predictions, model_name)

# Apply category/subcategory filters to stats
if grid_events:
    stats = stats[stats["event"].isin(grid_events)]
else:
    stats = stats.iloc[0:0]  # empty

# Crossing filter
if not stats.empty:
    if cross_filter == "Only women reached men (cross_year exists)":
        stats = stats[stats["cross_year"].notna()]
    elif cross_filter == "Only women NOT reached men (cross_year is None)":
        stats = stats[stats["cross_year"].isna()]

# Sorting
if sort_women_better:
    stats = stats.sort_values(
        by="improvement_relative_pct",
        ascending=False,
        na_position="last",
    )
else:
    # Default: keep your custom ordering if possible
    # (stats order is arbitrary, so re-apply your sorter)
    stats_events = stats["event"].tolist() if not stats.empty else []
    stats = stats.set_index("event").loc[sort_events_custom(stats_events)].reset_index()

# Final event list for rendering
render_events = stats["event"].tolist() if not stats.empty else []

# -----------------------------
# Render grid
# -----------------------------

if not render_events:
    st.info("No disciplines match the selected filters.")
else:
    st.caption(
        "Mini-plots use the same visual logic as the main chart (historical filled progression, optional forecasts, regression slopes, and gap indicators), "
        "with global toggles to show or hide each component. "
        "Sorting ranks disciplines by women’s relative improvement advantage compared to men, expressed as a percentage of men’s improvement rate "
        "(based on improvement-score slopes: time = −slope, mark = +slope)."
    )

    # Quick lookup: event -> stats row
    stats_by_event = stats.set_index("event").to_dict(orient="index")

    cols = st.columns(grid_cols)
    for i, ev in enumerate(render_events):
        col = cols[i % grid_cols]
        with col:
            fig_ev = make_event_figure(
                event=ev,
                model_name=model_name,
                show_gap_line=grid_show_gap_line,
                show_future_pred=grid_show_future_pred,
                show_full_pred=grid_show_full_pred,
                show_regression=grid_show_regression,
                show_history=grid_show_history,
            )
            if fig_ev is None:
                st.warning(f"Could not render {ev} (missing data).")
            else:
                st.plotly_chart(fig_ev, width='stretch', config=PLOTLY_CONFIG, key=f"grid_plot_{ev}_{i}",)

                # ---- slope label under each mini-plot ----
                row = stats_by_event.get(ev, {})

                w_slope = row.get("women_improvement_slope")
                m_slope = row.get("men_improvement_slope")
                women_better = row.get("women_better")
                measure_ev = row.get("measure")  # "time" or "mark" (you have it in stats)

                def format_slope_with_units(x, measure):
                    if x is None or pd.isna(x) or measure is None:
                        return "N/A"

                    x = float(x)
                    ax = abs(x)
                    sign = "+" if x >= 0 else "-"

                    if str(measure).lower() == "time":
                        # seconds per year, switch to ms if very small
                        if ax < 0.1:
                            return f"{sign}{ax*1000:.1f} ms/year"
                        return f"{sign}{ax:.3f} s/year"
                    else:
                        # meters per year, switch to cm if very small
                        if ax < 0.1:
                            return f"{sign}{ax*100:.1f} cm/year"
                        return f"{sign}{ax:.3f} m/year"

                measure_ev = row.get("measure")  # "time" or "mark"

                w_txt = format_slope_with_units(w_slope, measure_ev)
                m_txt = format_slope_with_units(m_slope, measure_ev)

                # Difference (women - men)
                diff = None
                if (w_slope is not None and not pd.isna(w_slope)) and (m_slope is not None and not pd.isna(m_slope)):
                    diff = float(w_slope) - float(m_slope)

                # ---- units helper for Δ ----
                def format_delta_with_units(d: float | None, measure: str | None) -> str:
                    if d is None or measure is None:
                        return "N/A"

                    ad = abs(d)
                    sign = "+" if d >= 0 else "-"

                    if str(measure).lower() == "time":
                        # use ms if < 0.1 s/year
                        if ad < 0.1:
                            return f"{sign}{ad*1000:.1f} ms/year"
                        return f"{sign}{ad:.3f} s/year"
                    else:
                        # mark: use cm if < 0.1 m/year
                        if ad < 0.1:
                            return f"{sign}{ad*100:.1f} cm/year"
                        return f"{sign}{ad:.3f} m/year"

                diff_txt_units = format_delta_with_units(diff, measure_ev)

                rel_pct = row.get("improvement_relative_pct")

                rel_txt = (
                    "N/A"
                    if rel_pct is None or pd.isna(rel_pct)
                    else f"{rel_pct:+.1f}%"
                )
                
                # Color + text for relative improvement (no +/- sign)
                if rel_pct is None or pd.isna(rel_pct):
                    rel_phrase = "N/A"
                else:
                    pct_abs = abs(rel_pct)
                    pct_txt = f"{pct_abs:.1f}%"

                    if rel_pct > 0:
                        color = COLOR_WOMEN  # green
                        rel_phrase = (
                            f"<span style='color:{color}; font-weight:650;'>"
                            f"{pct_txt} faster</span>"
                        )
                    else:
                        color = "#db3939"  # red
                        rel_phrase = (
                            f"<span style='color:{color}; font-weight:650;'>"
                            f"{pct_txt} slower</span>"
                        )

                if women_better is True:
                    badge = (
                        f"<span style='padding:2px 8px;border-radius:999px;"
                        f"border:1px solid rgba(255,255,255,0.16);"
                        f"background:{COLOR_WOMEN_FILL};font-weight:650;'>"
                        f"Women improving faster</span>"
                    )
                elif women_better is False:
                    badge = (
                        f"<span style='padding:2px 8px;border-radius:999px;"
                        f"border:1px solid rgba(255,255,255,0.16);"
                        f"background:{COLOR_MEN_FILL};font-weight:650;'>"
                        f"Men improving faster</span>"
                    )
                else:
                    badge = (
                        "<span style='padding:2px 8px;border-radius:999px;"
                        "border:1px solid rgba(255,255,255,0.12);"
                        "background:rgba(255,255,255,0.03);'>"
                        "Insufficient data</span>"
                    )

                st.markdown(
                    f"""
                    <div style="
                        font-size:0.86rem;
                        line-height:1.25;
                        opacity:0.92;
                        margin-top:-35px;        /* ↓ tighter to plot */
                        margin-bottom:30px;    /* ↑ more space before next plot */
                        padding-left: 15px;
                    ">
                    {badge}
                    <div style="height:6px;"></div>

                    <div>
                        <b>Improvement-score slope:</b><br>
                        · Women {w_txt}<br>
                        · Men {m_txt}<br>
                        · Women vs. Men: <b>{diff_txt_units}</b>
                    </div>

                    <div style="height:6px;"></div>

                    <div>Women improve {rel_phrase}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Show a small summary table (optional but handy)
    with st.expander("Show grid metrics table"):

        table = stats.copy()

        # Numeric improvement-score slopes (signed, sortable)
        table["Women improvement"] = table["women_improvement_slope"]
        table["Men improvement"] = table["men_improvement_slope"]

        # Numeric relative improvement percentage (signed)
        table["Women vs. Men (%)"] = table["improvement_relative_pct"]

        # Build unit-aware column names (same unit for entire table row-wise)
        def improvement_unit(measure):
            return "s/year" if str(measure).lower() == "time" else "m/year"

        table["Improvement unit"] = table["measure"].apply(improvement_unit)

        # Reorder / rename columns for display
        show_cols = [
            "event",
            "measure",
            "Women improvement",
            "Men improvement",
            "Women vs. Men (%)",
            "cross_year",
        ]

        display_table = table[show_cols].copy()

        # Rename columns to include units
        display_table = display_table.rename(
            columns={
                "Women improvement": "Women improvement (per year)",
                "Men improvement": "Men improvement (per year)",
            }
        )

        # Display
        st.dataframe(
            display_table,
            width="stretch",
            hide_index=True,
        )
