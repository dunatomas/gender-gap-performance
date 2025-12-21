import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# -----------------------------
# Config
# -----------------------------
CURRENT_YEAR = 2025
PRED_START_YEAR = 2026

# Pick non-stereotypical, distinguishable colors
COLOR_MEN = "#2ca02c"      # green
COLOR_WOMEN = "#ff7f0e"    # orange
COLOR_COMPARE = "#9467bd"  # purple
COLOR_MEN_ANNOT = "#3ddc84"  # lighter green for annotations

# Rounding for matching prediction record values to df_combined record values
MATCH_ROUND_DECIMALS = 3


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


# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Gender Gap in Sports Performance", layout="wide")

st.title("Gender Gap in Sports Performance")

st.markdown(
    """
Women’s performance has accelerated dramatically across many sports, especially as access, participation, coaching,
and professionalization have expanded. In several disciplines, women’s historical improvement rate has been steeper
than men’s, narrowing the performance gap over time.

This dashboard explores that gap by combining record progressions with model-based forecasts.
"""
)

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
        model_name = st.selectbox("Model", available_models, index=0)

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

st.subheader(f"{event} record progression and forecasts")

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
        fillcolor="rgba(255,127,14,0.15)",
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
        fillcolor="rgba(44,160,44,0.15)",
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
                line=dict(color=color, width=2, dash="dot"),
                hoverinfo="skip",  # visual aid only
                opacity=0.85,
                showlegend=True,
            )
        )

        # Small label near 2025 endpoint with slope
        # For time: negative slope means improving (going down); for marks: positive slope means improving (going up)
        slope_text = f"Improvement slope:<br>{slope:+.4f} per year"

        # Offset label a bit so it doesn't sit on the line
        y_offset = 0.03 * abs(y_2025) if y_2025 != 0 else 0.03
        y_lab = (y_2025 - y_offset) if measure == "time" else (y_2025 + y_offset)

        # Place label slightly left of CURRENT_YEAR
        x_label = CURRENT_YEAR + 15
        annot_color = COLOR_MEN_ANNOT if sex_label == "Men" else color

        fig.add_annotation(
            x=x_label,
            y=y_lab,
            text=slope_text,
            showarrow=False,
            font=dict(color=annot_color, size=14),
            opacity=0.9,
        )

    add_regression_line(women_hist_h, "Women", COLOR_WOMEN)
    add_regression_line(men_hist_h, "Men", COLOR_MEN)

# Horizontal comparison line + crossing point marker + better label placement
if show_gap_line and women_2025_val is not None and men_first_y is not None:

    # Point where it reaches women's record level (at crossing year)
    fig.add_trace(
        go.Scatter(
            x=[CURRENT_YEAR],
            y=[women_2025_val],
            mode="markers",
            marker=dict(size=10, color=COLOR_COMPARE),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Put label above the women's 2025 point
    y_offset = (0.025 * abs(women_2025_val)) if measure == "time" else (-0.06 * abs(women_2025_val) if women_2025_val != 0 else 0.06)

    if cross_year is None:
        # No crossing: dashed line from first men's record year to 2025
        fig.add_shape(
            type="line",
            x0=men_first_y, x1=CURRENT_YEAR,
            y0=women_2025_val, y1=women_2025_val,
            line=dict(color=COLOR_COMPARE, width=3, dash="dash"),
            opacity=0.9,
        )

        fig.add_annotation(
            x=CURRENT_YEAR,
            y=women_2025_val + y_offset,
            text=f"Women have not<br>surpassed men yet",
            showarrow=False,
            font=dict(color=COLOR_COMPARE, size=14),
        )
    else:
        # Crossing exists: solid line from crossing year to 2025
        fig.add_shape(
            type="line",
            x0=cross_year, x1=CURRENT_YEAR,
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

        # Label: to the right and slightly higher to avoid intersection
        # For time, "higher" => slightly smaller; for marks, "higher" => slightly bigger
        if cross_year is not None:
            fig.add_annotation(
                x=CURRENT_YEAR,
                y=women_2025_val + y_offset,
                text=f"Women surpass men<br>from {cross_year}",
                showarrow=False,
                font=dict(color=COLOR_COMPARE, size=14),
            )


# Layout: vertical gridlines + decade ticks
y_title = "Time (seconds)" if measure == "time" else "Mark (meters)"
fig.update_layout(
    height=650,
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

fig.update_yaxes(range=[y_min - padding, y_max + padding])

# (Optional) keep y grid subtle too
# fig.update_yaxes(showgrid=True, gridwidth=1, griddash="dot")

st.plotly_chart(fig, width='stretch')

st.caption(
    "Hover to see the raw record value (from the progression tables), the date it was obtained, and the athlete(s). "
    "The optional dashed 'model fit' (≤2025) is non-interactive."
)
