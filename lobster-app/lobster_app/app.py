"""Run the local LOBSTER research application.

Notes
-----
Data, cached calculations, and experiment history stay in memory. Files are
exported only through a user initiated browser download. No external assets,
analytics, cloud services, or update checks are used.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import threading
import time
from urllib.parse import urlparse
import uuid

os.environ["DASH_DISABLE_VERSION_CHECK"] = "true"
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP_DIR))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, State, MATCH, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
from flask import abort, jsonify, request

import lobster_exploration as lob
import research


BID = "#22e6b8"
ASK = "#ff667e"
INK = "#e7edf5"
MUTED = "#9aa7b8"
PANEL = "#171d27"
GRID = "#2b3545"
ACCENT = "#62a8ff"
LIGHT_CHART_COLORS = {
    BID: "#007f60", ASK: "#d82d55", ACCENT: "#205bd7",
    INK: "#182435", MUTED: "#526278", PANEL: "#ffffff", GRID: "#e1e7ef",
    "#202735": "#f1f5fa", "#445269": "#8798af",
    "#b5a0f5": "#7545c0", "#e9bd71": "#996100", "#80c5e8": "#007caa",
    "#df94cf": "#b3318a", "#aaca83": "#567a19",
    "#111a24": "#f2faf8", "#1d4e59": "#b3dcd1", "#298f89": "#3b9e87", "#8ce8be": "#00694f",
}
BOOK_SIZE_CAPS = (250, 500, 1000, 2500, 5000, 10000, 0)
BOOK_CHART_HEIGHT = 550
METHOD_LABELS = {"pca": "Depth structure", "response": "Imbalance and forward response", "pp": "Patnaik Pearson dimension"}
EMBEDDING_LABELS = {"custom": "selected features", "log_depth": "log depth", "relative_depth": "relative depth", "price_depth": "price and depth", "collaborator": "collaborator representation"}
GRAPH_CONFIG = {"displaylogo": False, "responsive": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
SETTING_HELP = {
    "Price differences": (r"Use each bid and ask price minus the best bid and best ask, omitting the two differences that are always zero.", r"4L-2\ \text{values}"),
    "Cumulative volumes": (r"Add displayed shares from level 1 through each level k, separately for bids and asks.", r"B_k=\sum_{j=1}^{k}q^b_j,\quad A_k=\sum_{j=1}^{k}q^a_j"),
    "Logarithms of cumulative volumes": (r"Take the logarithm of each cumulative bid and ask volume, flooring volumes at 0.00000001.", r"\log(\max(B_k,10^{-8})),\quad\log(\max(A_k,10^{-8}))"),
    "Volume imbalance": (r"Compare cumulative bid volume B with cumulative ask volume A through each level k.", r"I_k=(B_k-A_k)/(B_k+A_k+10^{-6})"),
    "Imbalance using logarithms of volumes": (r"Use the difference between log cumulative volumes divided by their sum, flooring volumes before taking logs.", r"J_k=(\log B_k-\log A_k)/(\log B_k+\log A_k+10^{-6})"),
    "Log volumes at each level": (r"Take the logarithm of one plus the displayed shares at each individual bid and ask level.", r"\log(1+q_k)"),
    "Volumes as proportions of total depth": (r"Divide each level's displayed volume by the total across all selected bid and ask levels.", r"q_k/\sum_{j=1}^{L}(q^b_j+q^a_j)"),
    "Distances from the midquote": (r"Measure each quote's distance from the midquote m in basis points, positive on both sides of a normal book.", r"10^4(a_k-m)/m,\quad10^4(m-b_k)/m"),
    "Burn in time (seconds)": (r"Exclude this duration from the start of the selected period before sampling or standardisation.", None),
    "Burn out time (seconds)": (r"Exclude this duration from the end of the selected period before calculating PP.", None),
    "Research question": (r"Choose which calculation to run on the selected data.", None),
    "Occupied levels per side": (r"Include the best $L$ bid prices and the best $L$ ask prices.", r"\ell=1,\ldots,L"),
    "Observation window": (r"Analyse the data between the selected start time $S$ and end time $E$.", r"S\leq t\leq E"),
    "Sampling interval": (r"Take a snapshot every $\Delta$ seconds.", r"t,\quad t+\Delta,\quad t+2\Delta,\ldots"),
    "Maximum snapshot age": (r"Use a snapshot only when its message time $\tau$ is at most $A$ seconds before the sampling time $t$.", r"0\leq t-\tau\leq A"),
    "Proportion used for training": (r"Fit PCA or regression on the earlier part of the selected period and evaluate it on the later part.", None),
    "Depth transformation": (r"Use volumes $q$ directly or take their logarithms to reduce the influence of very large volumes.", r"x=q\quad\text{or}\quad x=\log(1+q)"),
    "Standardise using training data": (r"Divide each variable by its training standard deviation $s$ after subtracting its training mean $\mu$.", r"z=(x-\mu)/s"),
    "Forward horizon": (r"Measure the return of the midquote $m$ over the next $h$ seconds.", r"r_{t,h}=10^4\log\left(\frac{m_{t+h}}{m_t}\right)"),
    "What forms each row of X?": (r"Make each row from one or more snapshots at different times, or from price levels within a single snapshot.", None),
    "Snapshot encoding": (r"Choose the variables describing each snapshot, including the collaborator’s price differences, cumulative volumes and their logarithms.", None),
    "Snapshots per observation": (r"Combine $K$ snapshots into one observation, separated by the chosen time or message interval.", None),
    "Time between snapshots": (r"Separate the snapshots in a combined observation by $\delta$ seconds, with no effect when only one snapshot is used.", r"t,\quad t-\delta,\quad t-2\delta,\ldots"),
    "Estimation window": (r"Use the preceding $W$ seconds of sampled snapshots for an estimate at time $T$.", r"T-W\leq t\leq T"),
    "Calculate dimension every": (r"Calculate a PP estimate every $\eta$ seconds.", r"T,\quad T+\eta,\quad T+2\eta,\ldots"),
    "Volatility window": (r"Use the preceding $W$ seconds of one second midquote returns, requiring a complete valid window and reporting volatility in basis points.", r"\mathrm{RV}_t=10^4\sqrt{\sum_{j=0}^{W-1}\left[\log\left(m_{t-j}/m_{t-j-1}\right)\right]^2}"),
    "Levels in each observation": (r"Combine bid and ask levels with the same level number, either one level on each side or three consecutive levels on each side.", None),
    "Sampling method": (r"Take snapshots at regular times or use the snapshot after every message, including messages with the same time.", None),
    "Messages per estimation window": (r"Use the last $M$ messages at each estimate, counting them before exclusions.", r"r-M+1,\ldots,r"),
    "Messages between snapshots": (r"Separate snapshots within a combined observation by $J$ messages, with no effect when only one snapshot is used.", r"r,\quad r-J,\quad r-2J,\ldots"),
    "Standardisation history": (r"Give older sampled observations exponentially less weight, reducing their weight to 1% after $H$ observations and discarding the first $H$ after the start or an invalid sample.", r"\lambda=0.01^{1/H}"),
    "Historical standardisation": (r"Standardise each sampled snapshot using earlier observations, replacing column standardisation if selected.", r"z_r=(x_r-\mu_r)/\sqrt{v_r+10^{-6}}"),
    "Column standardisation": (r"Subtract each column’s mean and optionally divide by its standard deviation, replacing historical standardisation if selected.", r"Z_{ij}=(X_{ij}-\mu_j)/s_j"),
    "Scaling": (r"Choose no standardisation, standardisation within each estimate, or standardisation using exponentially weighted past observations.", None),
}

PATHS = lob.discover_datasets(DATA_DIR)
EXPERIMENT_CACHE = OrderedDict()
CACHE_LOCK = threading.RLock()


def digest(path):
    """Compute a file fingerprint without loading the complete file into memory.

    Parameters
    ----------
    path : pathlib.Path
        File to read.

    Returns
    -------
    str
        SHA256 hex digest.
    """
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


IMPLEMENTATION_PATHS = (APP_DIR / "research.py", APP_DIR / "app.py", ROOT / "lobster_exploration.py")
IMPLEMENTATION_HASHES = {p.name: digest(p) for p in IMPLEMENTATION_PATHS}


def signature(index):
    """Identify the current source pair by sizes and modification times.

    Parameters
    ----------
    index : int
        Position in the discovered dataset list.

    Returns
    -------
    tuple
        File state used to invalidate the memory cache.
    """
    if not isinstance(index, int) or not 0 <= index < len(PATHS):
        raise ValueError("Choose an available dataset.")
    message = PATHS[index]
    book = message.with_name(message.name.replace("_message_", "_orderbook_"))
    return tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in (message, book))


@lru_cache(maxsize=2)
def _load(index, source_signature):
    """Read one source pair and its provenance into the bounded memory cache.

    Parameters
    ----------
    index : int
        Dataset position.
    source_signature : tuple
        Expected file state before loading.

    Returns
    -------
    tuple
        Loaded data and source metadata.
    """
    data = lob.load_lobster(PATHS[index])
    sources = [{"file": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
               for p in (data.message_path, data.orderbook_path)]
    if signature(index) != source_signature:
        raise ValueError("A source file changed during loading. Try again when it is stable.")
    return data, sources


def dataset(index):
    """Return the selected immutable source dataset and metadata.

    Parameters
    ----------
    index : int
        Dataset position.

    Returns
    -------
    tuple
        Loaded data and source metadata.
    """
    return _load(index, signature(index))


def clock(seconds):
    """Format seconds after midnight as an exchange clock time.

    Parameters
    ----------
    seconds : float
        Seconds after midnight.

    Returns
    -------
    str
        Clock label rounded down to the second.
    """
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def bounds(data, window):
    """Map an inclusive selected time interval to original event positions.

    Parameters
    ----------
    data : lobster_exploration.LobsterData
        Loaded source data.
    window : sequence of float
        Start and end in seconds after midnight.

    Returns
    -------
    tuple of int
        Inclusive first position and exclusive final position.
    """
    if not window or len(window) != 2 or not np.isfinite(window).all() or window[0] >= window[1]:
        raise ValueError("Choose a time window with a start before its end.")
    stamps = data.messages.time_ns.to_numpy()
    start = int(np.searchsorted(stamps, int(round(window[0] * 1e9)), side="left"))
    stop = int(np.searchsorted(stamps, int(round(window[1] * 1e9)), side="right"))
    if stop <= start:
        raise ValueError("There are no events in this time window.")
    return start, stop


def selected_data(data, window, levels):
    """Copy the selected event rows and recalculate depth within chosen levels.

    Parameters
    ----------
    data : lobster_exploration.LobsterData
        Loaded source data.
    window : sequence of float
        Inclusive exchange clock interval in seconds.
    levels : int
        Number of occupied levels per side.

    Returns
    -------
    lobster_exploration.LobsterData
        Selected event rows. Time means cover first through last selected event.
    """
    first, stop = bounds(data, window)
    levels = int(levels)
    if not 1 <= levels <= data.levels:
        raise ValueError("Choose a valid number of price levels.")
    m = data.messages.iloc[first:stop].copy()
    b = data.book.iloc[first:stop].copy()
    f = data.features.iloc[first:stop].copy()
    f.iloc[-1, f.columns.get_loc("duration_seconds")] = 0
    f.iloc[0, f.columns.get_loc("interarrival_ms")] = np.nan
    for side in ("bid", "ask"):
        f[f"{side}_depth"] = sum(b[f"{side}_size_{k}"].where(b[f"{side}_price_{k}"].notna(), 0)
                                  for k in range(1, levels + 1))
    denominator = f.bid_depth + f.ask_depth
    f["depth_imbalance"] = (f.bid_depth - f.ask_depth) / denominator.where(denominator.gt(0))
    return lob.LobsterData(m, b, f, data.quality, data.message_path, data.orderbook_path, levels, False)


def styled(fig, height=360):
    """Apply the application chart theme.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Chart to format.
    height : int, optional
        Chart height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Formatted chart.
    """
    fig.update_layout(template="plotly_dark", title=None, height=height,
                      font={"family": "Arial, sans-serif", "size": 12, "color": INK},
                      paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      colorway=[BID, ASK, ACCENT, "#b5a0f5", "#e9bd71", "#80c5e8", "#df94cf", "#aaca83"],
                      margin={"l": 55, "r": 25, "t": 72, "b": 50},
                      legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0,
                              "bgcolor": PANEL, "font_color": INK, "traceorder": "normal"},
                      hoverlabel={"bgcolor": "#202735", "bordercolor": "#445269", "font_color": INK},
                      modebar={"bgcolor": PANEL, "color": MUTED, "activecolor": ACCENT},
                      hovermode="x unified")
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, tickfont_color=MUTED, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, tickfont_color=MUTED, zerolinecolor="#445269")
    fig.update_traces(marker_line_width=0, selector={"type": "bar"})
    fig.update_traces(colorbar={"outlinewidth": 0, "tickfont_color": MUTED}, selector={"type": "heatmap"})
    return fig


def empty_figure(message="Select data to begin"):
    """Create an informative empty plot.

    Parameters
    ----------
    message : str, optional
        Message shown in the plot area.

    Returns
    -------
    plotly.graph_objects.Figure
        Placeholder chart.
    """
    fig = styled(go.Figure())
    fig.add_annotation(text=message, x=.5, y=.5, xref="paper", yref="paper", showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def graph(figure, graph_id=None):
    """Wrap a Plotly chart for the application.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        Chart to display.
    graph_id : str, optional
        Dash component identifier.

    Returns
    -------
    dash.html.Div
        Responsive graph and its original figure in browser memory. Theme
        changes recolor a copy without modifying data or rerunning analysis.
    """
    key = graph_id or uuid.uuid4().hex
    kwargs = {"id": graph_id} if graph_id else {}
    return html.Div([
        dcc.Store(id={"type": "chart-source", "key": key}, data=figure, storage_type="memory"),
        dcc.Graph(id={"type": "themed-chart", "key": key}, config=GRAPH_CONFIG,
                  figure={"data": [], "layout": {"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
                                                   "xaxis": {"visible": False}, "yaxis": {"visible": False}}},
                  style={"height": f"{figure.layout.height or 360}px"}),
    ], className="graph", **kwargs)


def card(title, content, note=None, wide=False):
    """Build a titled chart or table panel.

    Parameters
    ----------
    title : str
        Panel heading.
    content : component or list
        Panel content.
    note : str, optional
        Supporting interpretation.
    wide : bool, optional
        Span both grid columns.

    Returns
    -------
    dash.html.Div
        Panel component.
    """
    header = html.Div([html.H3(title, className="card-title"),
                       html.P(note, className="card-note") if note else None], className="card-header")
    return html.Div([header] + (content if isinstance(content, list) else [content]),
                    className="chart-card" + (" wide" if wide else ""))


def table(frame, digits=4):
    """Render a compact, escaped HTML data table.

    Parameters
    ----------
    frame : pandas.DataFrame
        Data to display.
    digits : int, optional
        Decimal places for floating point cells.

    Returns
    -------
    dash.html.Div
        Scrollable table.
    """
    def value(item):
        if pd.isna(item):
            return "—"
        if isinstance(item, (bool, np.bool_)):
            return "Yes" if item else "No"
        if isinstance(item, (float, np.floating)):
            return f"{item:,.{digits}f}"
        if isinstance(item, (int, np.integer)):
            return f"{item:,}"
        return str(item)
    return html.Div(html.Table([
        html.Thead(html.Tr([html.Th(str(c).replace("_", " ")) for c in frame.columns])),
        html.Tbody([html.Tr([html.Td(value(item)) for item in row]) for row in frame.itertuples(index=False, name=None)])
    ], className="data-table"), className="data-table-wrap")


def kpi(label, value, footnote):
    """Build one statistic card.

    Parameters
    ----------
    label : str
        Statistic name.
    value : str
        Formatted value.
    footnote : str
        Units or weighting definition.

    Returns
    -------
    dash.html.Div
        Statistic card.
    """
    return html.Div([html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value"),
                     html.Div(footnote, className="kpi-footnote")], className="kpi-card")


def price_chart(view, metric):
    """Plot a bounded display sample without changing analytical observations.

    Parameters
    ----------
    view : lobster_exploration.LobsterData
        Selected event rows.
    metric : str
        One of price, spread, or imbalance.

    Returns
    -------
    plotly.graph_objects.Figure
        Time series sampled at up to 3500 event positions.
    """
    ix = np.unique(np.linspace(0, len(view.messages) - 1, min(3500, len(view.messages)), dtype=int))
    f = view.features.iloc[ix]
    # Dash indexes customdata directly when constructing hoverData. A plain
    # list avoids Plotly's binary encoding of pandas and NumPy integer arrays.
    event_rows = f.index.tolist()
    fig = go.Figure()
    columns = ({"price": [("best_bid", "Best bid", BID), ("best_ask", "Best ask", ASK), ("midquote", "Midquote", ACCENT)],
                "spread": [("spread", "Spread", BID)],
                "imbalance": [("depth_imbalance", "Depth imbalance", BID)]})[metric]
    for col, label, color in columns:
        fig.add_trace(go.Scattergl(x=f.timestamp, y=f[col], name=label, mode="lines",
                                   line={"color": color, "width": 1.6}, customdata=event_rows,
                                   visible="legendonly" if col == "midquote" else True,
                                   hovertemplate="%{x}<br>%{y:.5f}<br>Original row %{customdata:,}<extra>%{fullData.name}</extra>"))
    if metric == "price":
        snapshots = view.book.iloc[ix]
        for level in range(2, view.levels + 1):
            opacity = .75 * .85 ** (level - 2)
            for side, color in (("bid", BID), ("ask", ASK)):
                label = f"{side.title()} level {level}"
                legend_entry = level == 2 and side == "bid"
                fig.add_trace(go.Scattergl(
                    x=f.timestamp, y=snapshots[f"{side}_price_{level}"],
                    name="Extra levels" if legend_entry else label,
                    mode="lines", line={"color": color, "width": 1.2},
                    opacity=opacity, connectgaps=False, customdata=event_rows,
                    legendgroup="extra-levels", showlegend=legend_entry,
                    visible=True,
                    hovertemplate="%{x}<br>%{y:.5f}<br>Original row %{customdata:,}<extra>" + label + "</extra>"))
        fig.update_layout(legend_groupclick="togglegroup")
    fig.update_layout(uirevision=f"{view.message_path.name}-{view.messages.index[0]}-{view.messages.index[-1]}-{metric}")
    fig.update_yaxes(title_text="USD" if metric != "imbalance" else "(Bid − Ask) / Total")
    styled(fig, 400)
    if metric == "price":
        # Keep every value tied to the same original event, even at equal times.
        prices = snapshots[[f"{side}_price_{level}" for side in ("bid", "ask")
                            for level in range(1, view.levels + 1)]]
        prices = prices.astype(object).where(np.isfinite(prices), None)
        fig.update_traces(hovertemplate=None, hoverinfo="none")
        fig.update_layout(hovermode="x", hoverdistance=-1, meta={"price_readout": {
            "levels": view.levels, "rows": event_rows,
            "times": f.timestamp.astype(str).tolist(), "prices": prices.values.tolist(),
            "midquotes": f.midquote.astype(object).where(np.isfinite(f.midquote), None).tolist(),
        }})
        fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="data",
                         spikecolor=MUTED, spikethickness=1)
    return fig


def depth_heatmap(view):
    """Draw visible depth on an evenly spaced display clock grid.

    Parameters
    ----------
    view : lobster_exploration.LobsterData
        Selected event rows.

    Returns
    -------
    plotly.graph_objects.Figure
        Heatmap of log1p displayed shares across at most 700 clock positions.
    """
    stamps = view.messages.time_ns.to_numpy()
    grid = np.linspace(stamps[0], stamps[-1], min(700, len(stamps))).astype("int64")
    positions = np.searchsorted(stamps, grid, side="right") - 1
    b = view.book.iloc[positions]
    labels, matrix = [], []
    for side, levels in (("bid", range(view.levels, 0, -1)), ("ask", range(1, view.levels + 1))):
        for level in levels:
            labels.append(f"{side.title()} {level}")
            matrix.append(np.log1p(b[f"{side}_size_{level}"].where(b[f"{side}_price_{level}"].notna(), 0)).to_numpy())
    times = view.messages.timestamp.iloc[0].normalize() + pd.to_timedelta(grid, unit="ns")
    fig = go.Figure(go.Heatmap(x=times, y=labels, z=matrix, colorscale=[[0, "#111a24"], [.3, "#1d4e59"], [.65, "#298f89"], [1, "#8ce8be"]],
                              colorbar={"title": "log(1+shares)", "thickness": 10},
                              hovertemplate="%{x}<br>%{y}<br>log(1 + shares) %{z:.3f}<extra></extra>"))
    fig.update_layout(hovermode="closest")
    return styled(fig, 360)


def activity_chart(view):
    """Aggregate selected execution volumes into minute bins.

    Parameters
    ----------
    view : lobster_exploration.LobsterData
        Selected event rows.

    Returns
    -------
    plotly.graph_objects.Figure
        Visible and hidden execution share volumes, excluding cross messages.
    """
    t = lob.intraday_summary(view, "1min")
    fig = go.Figure()
    for col, label, color in (("visible_execution_shares", "Visible", BID), ("hidden_execution_shares", "Hidden", ASK)):
        fig.add_bar(x=t.index, y=t[col], name=label, marker_color=color, marker_line_width=0)
    fig.update_layout(barmode="stack", bargap=0)
    fig.update_yaxes(title_text="Executed shares")
    return styled(fig)


def result_figures(result):
    """Build research plots from a completed experiment record.

    Parameters
    ----------
    result : dict
        PCA, response, or dimension experiment result.

    Returns
    -------
    list
        Chart cards for the experiment.
    """
    if result["method"] == "pp":
        records = result["series"]
        times = [r["time"] for r in records]
        dimension = go.Figure()
        dimension.add_scatter(x=times, y=[r["dimension"] for r in records], name="PP dimension", mode="lines+markers",
                              line_color=BID, marker_size=4, connectgaps=False,
                              customdata=[[r["point_count"], r["coordinate_count"], r["status"], i, result["run_id"]] for i, r in enumerate(records)],
                              hovertemplate="%{x}<br>PP %{y:.4f}<br>%{customdata[0]} points · %{customdata[1]} coordinates<br>%{customdata[2]}<extra></extra>")
        dimension.add_scatter(x=times, y=[r["rank_bound"] for r in records], name="Upper bound on dimension", mode="lines", line={"color": MUTED, "dash": "dot"},
                              customdata=[[r["point_count"], r["coordinate_count"], r["status"], i, result["run_id"]] for i, r in enumerate(records)])
        dimension.update_layout(hovermode="x unified")
        dimension.update_yaxes(title_text="Patnaik Pearson dimension", rangemode="tozero")
        dimension.update_xaxes(title_text="Snapshot time" if result["parameters"]["mode"] == "snapshot" else "History window ending time")
        return [card("Dimension through time", graph(styled(dimension), "pp-dimension-time"), "Hover over a time to inspect its estimate below. Gaps indicate undefined estimates. With N observations and d numbers per observation, dimension cannot exceed min(N − 1, d). Linear dependencies can lower this bound.", True)]
    if result["method"] == "pca":
        x = list(range(1, len(result["explained_variance"]) + 1))
        variance = go.Figure()
        variance.add_bar(x=x, y=np.array(result["explained_variance"]) * 100, name="Per component", marker_color=ACCENT)
        variance.add_scatter(x=x, y=np.array(result["cumulative_variance"]) * 100, name="Cumulative", line_color=BID, mode="lines+markers")
        variance.update_yaxes(title_text="Training variance (%)", range=[0, 102])
        variance.update_xaxes(title_text="Principal components", dtick=5 if len(x) > 10 else 1, range=[.4, len(x) + .6])
        error = go.Figure(go.Scatter(x=result["reconstruction_components"], y=result["test_reconstruction_rmse"], mode="lines+markers", line_color=ASK))
        error.update_yaxes(title_text="Evaluation RMSE")
        error.update_xaxes(title_text="Components retained", dtick=5 if len(x) > 10 else 1)
        component_matrix = np.asarray(result["components"])
        weights = go.Figure(go.Heatmap(z=component_matrix[:min(5, len(component_matrix))],
                                      x=result["feature_names"], y=[f"PC {i+1}" for i in range(min(5, len(component_matrix)))],
                                      colorscale=[[0, ASK], [.5, PANEL], [1, BID]], zmid=0, colorbar={"thickness": 10, "title": "Weight"}))
        return [card("Where the variance sits", graph(styled(variance)), "PCA is fitted on the training period only."),
                card("Reconstruction on later observations", graph(styled(error)), "Zero components is the training mean baseline. Units: " + result["parameters"].get("reconstruction_units", "transformed feature units") + "."),
                card("Component directions", graph(styled(weights, 300)), "First five directions. Component signs are arbitrary. These are linear variance patterns, not intrinsic dimension estimates.", True)]
    fig = go.Figure()
    count = go.Figure()
    for prefix, label, color in (("train", "Training", BID), ("test", "Evaluation", ASK)):
        fig.add_scatter(x=result["bin_centers"], y=result[f"{prefix}_means"], name=label, mode="lines+markers", line_color=color,
                        customdata=result[f"{prefix}_counts"], hovertemplate="Imbalance %{x:.2f}<br>Mean return %{y:.4f} bp<br>Samples %{customdata:,}<extra>%{fullData.name}</extra>")
        count.add_bar(x=result["bin_centers"], y=result[f"{prefix}_counts"], name=label, marker_color=color)
    fig.update_xaxes(title_text="Displayed depth imbalance")
    fig.update_yaxes(title_text="Mean forward log return (bp)")
    count.update_layout(barmode="group")
    count.update_xaxes(title_text="Displayed depth imbalance")
    count.update_yaxes(title_text="Valid response pairs")
    return [card("Forward response by imbalance", graph(styled(fig)), "Fixed bins. Lines summarize conditional means and do not establish predictive significance."),
            card("How much evidence is in each bin", graph(styled(count)), "Serial dependence and overlapping horizons prevent ordinary independent sample inference.")]


def run_record(method, index, window, levels, step, horizon, transform, standardize, fraction, max_age, dimension_settings=None):
    """Run a deterministic experiment and attach reproducibility metadata.

    Parameters
    ----------
    method : str
        Either pca, response, or pp.
    index : int
        Source dataset position.
    window : sequence of float
        Selected interval in exchange clock seconds.
    levels : int
        Occupied levels per side.
    step : int
        Sampling interval in seconds.
    horizon : int
        Response horizon in seconds.
    transform : str
        PCA transformation, raw or log.
    standardize : bool
        Whether to scale PCA features using training standard deviations.
    fraction : float
        Chronological training fraction.
    max_age : int
        Maximum snapshot age in seconds.
    dimension_settings : dict, optional
        Embedding, history, scaling, and evaluation settings for PP dimension.

    Returns
    -------
    dict
        JSON compatible result with data and implementation fingerprints.
    """
    if any(digest(p) != IMPLEMENTATION_HASHES[p.name] for p in IMPLEMENTATION_PATHS):
        raise ValueError("Application source files changed after startup. Restart the app before running another experiment.")
    data, sources = dataset(index)
    bounds(data, window)
    dimension_settings = dict(dimension_settings or {}) if method == "pp" else {}
    key = (method, tuple((s["file"], s["sha256"]) for s in sources), tuple(window), levels, step, horizon, transform, standardize, fraction, max_age,
           json.dumps(dimension_settings, sort_keys=True, allow_nan=False))
    began = time.perf_counter()
    with CACHE_LOCK:
        cached = EXPERIMENT_CACHE.get(key)
    if cached is None:
        common = {"start_seconds": float(window[0]), "end_seconds": float(window[1]), "step_seconds": int(step),
                  "levels": int(levels), "train_fraction": float(fraction), "max_age_seconds": float(max_age)}
        if method == "pca":
            raw = research.run_pca(data, transform=transform, standardize=bool(standardize), **common)
        elif method == "response":
            raw = research.run_response(data, horizon_seconds=int(horizon), **common)
        elif method == "pp":
            common.pop("train_fraction")
            raw = research.run_dimension(data, **dimension_settings, **common)
        else:
            raise ValueError("Choose a supported experiment.")
        cached = json.loads(json.dumps(raw, allow_nan=False))
        with CACHE_LOCK:
            EXPERIMENT_CACHE[key] = cached
            while len(EXPERIMENT_CACHE) > 12:
                EXPERIMENT_CACHE.popitem(last=False)
    result = dict(cached)
    result.update({"method": method, "method_label": cached["method"], "run_id": uuid.uuid4().hex[:8], "created_utc": datetime.now(timezone.utc).isoformat(),
                   "seconds": round(time.perf_counter() - began, 4), "dataset": data.message_path.name,
                   "sources": sources, "loaded_rows": len(data.messages), "preview": False,
                   "implementation": dict(IMPLEMENTATION_HASHES),
                   "software": {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in ("numpy", "pandas", "plotly", "dash")}}})
    return result


app = Dash(__name__, assets_folder=str(APP_DIR / "assets"), title="LOBSTER · Research studio", update_title=None, serve_locally=True)
app.index_string = app.index_string.replace("<html>", '<html data-theme="light">', 1)
server = app.server


app.clientside_callback(
    """function(mode) {
        const theme = mode === 'dark' ? 'dark' : 'light';
        document.documentElement.dataset.theme = theme;
        return theme;
    }""",
    Output("theme-applied", "data"), Input("color-mode", "value"),
)

# Every figure, including dynamically created research plots, follows the same
# browser-side theme callback. The original numerical figure stays untouched.
CHART_THEME_CALLBACK = """function(source, mode) {
    if (!source) return window.dash_clientside.no_update;
    const light = mode === 'light';
    const colors = __COLORS__;
    function copy(value) {
        if (Array.isArray(value)) return value.map(copy);
        if (value !== null && typeof value === 'object') {
            return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, copy(item)]));
        }
        return light && typeof value === 'string' && colors[value] ? colors[value] : value;
    }
    const figure = copy(source);
    if (light) {
        figure.layout.template = __LIGHT_TEMPLATE__;
        // White full-size labels keep their dark backing in both modes.
        (figure.layout.annotations || []).forEach(annotation => {
            if (annotation.font && annotation.font.color === '#ffffff') {
                annotation.bgcolor = '#171d27';
            }
        });
    }
    return figure;
}""".replace("__COLORS__", json.dumps(LIGHT_CHART_COLORS)).replace(
    "__LIGHT_TEMPLATE__", json.dumps(go.Figure(layout_template="plotly_white").layout.template.to_plotly_json()))

app.clientside_callback(
    CHART_THEME_CALLBACK,
    Output({"type": "themed-chart", "key": MATCH}, "figure"),
    Input({"type": "chart-source", "key": MATCH}, "data"), Input("theme-applied", "data"),
)


PRICE_READOUT_CALLBACK = """function(hover, source) {
    const data = source && source.layout && source.layout.meta && source.layout.meta.price_readout;
    if (!data) return [[], {display: 'none'}];
    const changed = window.dash_clientside.callback_context.triggered.some(
        item => item.prop_id.endsWith('.data'));
    const point = !changed && hover && hover.points && hover.points[0];
    const index = point && point.pointNumber;
    const valid = Number.isInteger(index) && index >= 0 && index < data.rows.length
        && point.customdata === data.rows[index];
    const prices = valid ? data.prices[index] : [];
    const component = (type, children, props = {}) => ({
        namespace: 'dash_html_components', type, props: {...props, children}
    });
    const price = (value, currency = false) => typeof value === 'number' && Number.isFinite(value)
        ? (currency ? '$' : '') + value.toFixed(4) : '—';
    const heading = valid
        ? data.times[index] + ' · Original row ' + data.rows[index].toLocaleString()
            + ' · Midquote ' + price(data.midquotes[index], true)
        : 'Hover over the chart to inspect prices at each level.';
    const headers = [component('Th', 'Level', {scope: 'col'})];
    const bids = [component('Th', 'Bid ($)', {scope: 'row'})];
    const asks = [component('Th', 'Ask ($)', {scope: 'row'})];
    for (let level = 0; level < data.levels; level++) {
        headers.push(component('Th', String(level + 1), {scope: 'col'}));
        bids.push(component('Td', price(prices[level])));
        asks.push(component('Td', price(prices[data.levels + level])));
    }
    return [[component('Div', heading, {className: 'small price-readout-time'}),
        component('Div', component('Table', [
            component('Thead', component('Tr', headers)),
            component('Tbody', [component('Tr', bids, {className: 'price-readout-bid'}),
                                component('Tr', asks, {className: 'price-readout-ask'})])
        ], {className: 'data-table price-readout-table', 'aria-label': 'Prices by occupied level'}),
        {className: 'data-table-wrap'})], {}];
}"""

app.clientside_callback(
    PRICE_READOUT_CALLBACK,
    Output("price-readout", "children"), Output("price-readout", "style"),
    Input({"type": "themed-chart", "key": "price-chart"}, "hoverData"),
    Input({"type": "chart-source", "key": "price-chart"}, "data"),
)


@server.before_request
def local_requests_only():
    """Reject foreign hosts and cross origin browser requests.

    Returns
    -------
    None
        Request processing continues only for local origins.
    """
    if request.host.split(":")[0] not in {"127.0.0.1", "localhost"}:
        abort(403)
    origin = request.headers.get("Origin")
    if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
        abort(403)
    if request.headers.get("Sec-Fetch-Site") == "cross-site":
        abort(403)


@server.after_request
def privacy_headers(response):
    """Apply local application response headers.

    Parameters
    ----------
    response : flask.Response
        Outgoing response.

    Returns
    -------
    flask.Response
        Response without persistent data caching.
    """
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response


@server.get("/health")
def health():
    """Identify the running local application.

    Returns
    -------
    flask.Response
        Non sensitive application identity.
    """
    return jsonify({"application": "lobster-research-studio", "status": "ready"})


def order_book_spinner():
    """Create an animated circle of bid and ask bars.

    Returns
    -------
    dash.html.Div
        Accessible loading status with decorative bars animated by CSS.
    """
    heights = (22, 32, 18, 27, 15, 34, 23, 16, 29, 20, 31, 17,
               28, 19, 33, 16, 25, 21, 30, 14, 27, 18, 32, 23)
    return html.Div([
        html.Div([
            html.Span(html.Span(className="book-loader-bar"), className="book-loader-spoke",
                      style={"--angle": f"{i * 15}deg", "--bar-height": f"{height}px",
                             "--delay": f"{-i * .075:.3f}s",
                             "--bar-colour": "var(--coral)" if i < 12 else "var(--positive)"})
            for i, height in enumerate(heights)
        ], className="book-loader-ring", **{"aria-hidden": "true"}),
        html.Span("Calculating experiment", className="sr-only"),
    ], className="book-loader", role="status", **{"aria-live": "polite"})


def setting_help(label):
    """Create a compact explanation next to a setting.

    Parameters
    ----------
    label : str
        Setting name matching its visible label.

    Returns
    -------
    dash.html.Details or None
        Accessible popover containing one sentence and an optional equation.
    """
    if label not in SETTING_HELP:
        return None
    sentence, equation = SETTING_HELP[label]
    return html.Details([
        html.Summary("?", className="control-help", **{"aria-label": f"Help for {label}"}),
        html.Div([
            html.Div([
                html.Strong(label),
                html.Button("×", type="button", className="help-close", **{"aria-label": "Close help"}),
            ], className="help-heading"),
            dcc.Markdown(sentence, mathjax=True, className="help-sentence"),
            dcc.Markdown("$$\n" + equation + "\n$$", mathjax=True, className="help-equation") if equation else None,
        ], className="help-popover", role="dialog", **{"aria-label": f"About {label}"}),
    ], className="setting-help", open=False)


def control(label, component):
    """Wrap a labeled form control.

    Parameters
    ----------
    label : str
        Visible control name.
    component : object
        Dash input control.

    Returns
    -------
    dash.html.Div
        Labeled control.
    """
    return html.Div([html.Div([html.Label(label, className="control-label"), setting_help(label)],
                             className="control-label-row"), component], className="control-group")


def experiment_documentation():
    """Keep the common notation and PP calculation below the controls.

    Returns
    -------
    dash.html.Section
        Two reference sections with mathematical notation.
    """
    notation = r"""
A **snapshot** records the order book at one time. An **observation** is a row of the
matrix used in a calculation and may combine several snapshots or several price levels.

For side $s\in\{b,a\}$, write $p^s_{t,\ell}$ for the price and $q^s_{t,\ell}$ for the
displayed volume at time $t$ and level $\ell$. Here $b$ means bid, $a$ means ask, and
$\ell=1$ is the best price on that side. The selected number of levels per side is $L$.

The midquote is $m_t$, and $\mathbf{q}_t$ collects the selected volumes, with bid volumes
first and ask volumes second.

$$
m_t=\frac{p^b_{t,1}+p^a_{t,1}}{2}.
$$

$$
\mathbf{q}_t=(q^b_{t,1},\ldots,q^b_{t,L},
q^a_{t,1},\ldots,q^a_{t,L})^{\mathsf T}.
$$

For clock sampling, if message $r$ has time $\tau_r$, the snapshot used at time $t$ is the one after
the final message at or before $t$.

$$
r(t)=\max\{r:\tau_r\leq t\}.
$$

$\mathbf{x}_t$ denotes the chosen vector representing one snapshot.
$\mathbf{y}_t$ combines $K$ such vectors when several snapshots form one observation.
All vectors are column vectors.
Message sampling uses every snapshot in its original row order, including equal timestamps.
Its estimation window and snapshot spacing are measured in messages.

The time settings use the following symbols.

- $[S,E]$ is the selected observation period.
- $\Delta$ is the sampling interval.
- $A$ is the maximum snapshot age.
- $\delta$ is the time between snapshots combined into one observation.
- $W$ is the amount of recent history used for one estimate.
- $\eta$ is the time between dimension estimates.
- $h$ is the period over which a forward return is measured.

Times are in seconds, volumes are in shares, and relative price distances and returns
are expressed in basis points.
The selected price differences use original LOBSTER units,
where one unit is $10^{-4}$ dollars.
"""
    dimension = r"""
Put the observations into the rows of a matrix $X\in\mathbb{R}^{N\times d}$.
There are $N$ observations and $d$ variables per observation.
The selected settings determine these rows and columns.
Feature groups set to **Standardised** use earlier observations. **Both** includes
original and historically standardised values as separate columns. **Column standardisation**
then determines whether every column is only centred or also divided by its standard deviation.

**1. Centre and, if selected, standardise.** For $N\geq2$, calculate each column's mean
$\mu_j$ and sample standard deviation $s_j$ using the observations in this estimate.

$$
\mu_j=\frac{1}{N}\sum_{i=1}^{N}X_{ij},\qquad
s_j^2=\frac{1}{N-1}\sum_{i=1}^{N}(X_{ij}-\mu_j)^2.
$$

Subtract the mean in every case, and divide by the standard deviation only for
**Standardise within each estimate**. A constant column becomes zero.

$$
Z_{ij}=
\begin{cases}
X_{ij}-\mu_j,&\text{no further standardisation},\\
(X_{ij}-\mu_j)/s_j,&\text{standardise, }s_j>0,\\
0,&\text{standardise, }s_j=0.
\end{cases}
$$

**2. Calculate singular values.** Let $\sigma_j$ be the nonnegative diagonal entries
of $\Sigma$ in the singular value decomposition of $Z$.

$$
Z=U\Sigma V^{\mathsf T}.
$$

**3. Calculate PP dimension.** Use the singular values themselves, rather than
covariance eigenvalues.

$$
d_{\mathrm{PP}}=
\frac{\left(\sum_j\sigma_j\right)^2}{\sum_j\sigma_j^2}.
$$

One nonzero singular value gives dimension one. Exactly $r$ equal nonzero singular
values give dimension $r$. The result need not be an integer.

For nonconstant observations, the bound and the normalised result are

$$
1\leq d_{\mathrm{PP}}\leq\min(N-1,d),\qquad
d_{\mathrm{normalised}}=\frac{d_{\mathrm{PP}}}{d}.
$$

Fewer than two observations or identical observations give an undefined estimate,
shown as a gap. Constant columns still count in $d$. The inspector rescales singular
values by their largest value for display, which leaves PP unchanged. Small nonzero
singular values remain in the calculation.

PP depends on the representation, scaling and observations chosen, and does not by
itself establish the dimension of an underlying manifold.
"""
    return html.Section([
        html.H2("Reference", id="experiment-docs-title", className="card-title"),
        *[html.Details([
            html.Summary(html.Span(title, className="method-doc-title")),
            dcc.Markdown(body.strip(), mathjax=True, className="method-doc-body"),
        ], className="method-doc", open=False)
          for title, body in (("Notation", notation), ("How PP dimension is calculated", dimension))],
    ], id="experiment-docs", className="experiment-docs",
        **{"aria-labelledby": "experiment-docs-title"})


def layout():
    """Create an independent browser tab layout and memory stores.

    Returns
    -------
    dash.html.Div
        Complete application layout.
    """
    options = [{"label": f"{p.name.split('_')[0]}  ·  {p.name.split('_')[1]}  ·  {p.stem.split('_')[-1]} levels", "value": i} for i, p in enumerate(PATHS)]
    return html.Div([
        dcc.Store(id="run-history", data=[], storage_type="memory"), dcc.Download(id="download-runs"),
        dcc.Store(id="dimension-result", data=None, storage_type="memory"),
        dcc.Store(id="pp-features", data=research.TOM_FEATURES),
        dcc.Store(id="book-axis-state", data=None, storage_type="memory"),
        dcc.Store(id="book-clock-value", data=None, storage_type="memory"),
        dcc.Store(id="theme-applied", data="light", storage_type="memory"),
        dcc.Store(id="help-ready", data=False, storage_type="memory"),
        dcc.Interval(id="play-timer", interval=200, disabled=True, n_intervals=0),
        html.Aside([
            html.Div([html.Div("L", className="brand-mark"), html.Div([html.Div("LOBSTER", className="brand-name"), html.Div("Research studio", className="brand-subtitle")])], className="brand"),
            html.P("DATA SOURCE", className="sidebar-label"),
            dcc.Dropdown(id="dataset", options=options, value=0 if options else None, clearable=False, searchable=False),
            html.Div(id="dataset-info", className="sidebar-note"),
            html.P("VISIBLE DEPTH", className="sidebar-label"),
            control("Occupied levels per side", dcc.Slider(id="levels", min=1, max=10, step=1, value=10, marks={1: "1", 5: "5", 10: "10"}, allow_direct_input=False, tooltip={"placement": "bottom", "always_visible": False})),
            html.Div([html.Span("●", style={"color": "var(--positive)"}), "  Local on this computer", html.P("Source CSVs are read only. Experiments stay in this tab until you download them.", className="small")], className="sidebar-status"),
        ], className="sidebar"),
        html.Main([
            html.Header([html.Div([html.H1("Investigate the order book", className="page-title")]),
                         html.Div([
                             dcc.RadioItems(id="color-mode", options=[
                                 {"label": html.Span([html.Span(className="theme-icon theme-moon", **{"aria-hidden": "true"}),
                                                       html.Span("Dark mode", className="sr-only")], title="Dark mode", className="theme-option"), "value": "dark"},
                                 {"label": html.Span([html.Span(className="theme-icon theme-sun", **{"aria-hidden": "true"}),
                                                       html.Span("Light mode", className="sr-only")], title="Light mode", className="theme-option"), "value": "light"}],
                                            value="light", inline=True, className="theme-switch"),
                         ], className="header-actions")], className="topbar"),
            html.Div([html.Div([html.Div([html.Label("Observation window", className="control-label"), setting_help("Observation window")], className="control-label-row"), html.Span(id="window-label", className="range-label")], className="card-header"),
                      dcc.RangeSlider(id="window", min=34140, max=57660, step=1, value=[34140, 57660], marks={}, allowCross=False, allow_direct_input=False,
                                      tooltip={"placement": "bottom", "always_visible": False}),
                      html.Div([html.Button("Full file", id="full-window", className="button-quiet"), html.Button("09:30 to 16:00", id="regular-window", className="button-quiet"),
                                html.Span("Exchange clock · all statistics use the selected observations", className="small")], className="button-row")], className="controls-card"),
            dcc.Tabs(id="view", value="overview", className="view-tabs", children=[
                dcc.Tab(label=label, value=value, className="tab", selected_className="tab--selected")
                for label, value in (("Overview", "overview"), ("Order book", "book"), ("Experiments", "experiments"), ("Compare runs", "compare"))]),
            html.Section([
                html.Div(id="overview-kpis", className="kpi-grid"),
                card("Market through time", [dcc.RadioItems(id="metric", options=[{"label": l, "value": v} for l, v in (("Quotes", "price"), ("Spread", "spread"), ("Imbalance", "imbalance"))], value="price", inline=True, className="button-row"), graph(empty_figure(), "price-chart"),
                     html.Div(id="price-readout", className="price-readout", style={"display": "none"})],
                     "Zoom by dragging. Click Midquote in the legend to show it. Extra levels are shown by default and toggle together when clicked in the legend, fading with each level. Lines draw at most 3,500 event rows and may omit brief excursions. Tables use every selected row."),
                card("Realised volatility", [
                    html.Div(control("Volatility window", dcc.Dropdown(
                        id="volatility-window", options=[{"label": f"{minutes} minute{'s' if minutes != 1 else ''}", "value": minutes * 60}
                                                          for minutes in (1, 5, 15, 30)],
                        value=300, clearable=False, searchable=False)), style={"maxWidth": "220px", "marginTop": "16px"}),
                    graph(empty_figure(), "volatility-chart")],
                    "Calculated from one second midquote log returns, without annualising. Gaps mark incomplete windows, invalid quotes or quotes more than 60 seconds old."),
                html.Div([card("Depth across occupied levels", graph(empty_figure(), "depth-chart"), "At most 700 display clock points. Color shows log(1 + displayed shares)."),
                          card("Execution activity", graph(empty_figure(), "activity-chart"), "One minute bins. Visible and hidden executions only. Edge bins may be partial.")], className="chart-grid"),
                html.Div(id="overview-tables", className="chart-grid"),
            ], id="overview-view"),
            html.Section([
                html.Div([html.Div([html.H3("Replay snapshots", className="card-title"), html.P("Each snapshot shows the order book after the selected event. Row numbers refer to the original file.", className="card-note")], className="card-header"),
                          dcc.Slider(id="book-row", min=0, max=1, step=1, value=0, marks={}, allow_direct_input=False, tooltip={"placement": "bottom"}),
                          html.Div([html.Button("← Previous", id="previous-row", className="button-secondary"), html.Button("Play", id="play-button", className="button-primary"),
                                    html.Button("Next →", id="next-row", className="button-secondary"),
                                    control("Event row", dcc.Input(id="book-row-number", type="number", min=0, step=1, value=0, debounce=True)),
                                    control("Events per step", dcc.Dropdown(id="row-step", options=[1, 10, 100, 1000], value=100, clearable=False, searchable=False)),
                                    control("Playback speed", dcc.Dropdown(id="play-speed", options=[{"label": f"{speed:g}×", "value": speed} for speed in (.25, .5, 1, 2, 4, 8)], value=4, clearable=False, searchable=False)),
                                    control("Vertical cap", dcc.Dropdown(id="book-size-cap", options=[{"label": f"{cap:,} shares" if cap else "Full sizes", "value": cap} for cap in BOOK_SIZE_CAPS], value=2500, clearable=False, searchable=False)),
                                    html.Button("Refit axes", id="refit-book-axes", n_clicks=0, className="button-secondary")], className="button-row"),
                          html.P("1× updates every 0.8 seconds. Events per step controls how many messages advance. This is event replay, not exchange clock speed.", className="small")], className="controls-card"),
                html.Div([
                    card("Displayed orders by price", graph(empty_figure().update_layout(height=BOOK_CHART_HEIGHT), "snapshot-chart"),
                         "Bars above the vertical cap are clipped and labelled in white with their full size. Hover and the ladder show exact sizes. Price limits expand only when needed. The dotted blue line is the midquote."),
                    html.Div([
                    html.Div([
                        html.Div("Exchange time", className="replay-clock-label"),
                        html.Div([
                            html.Span([
                                html.Span(":", className="replay-clock-colon") if i in (2, 5) else
                                html.Span(html.Span("–", id=f"replay-digit-{i}", className="replay-clock-digit"), className="replay-clock-slot")
                                for i in range(8)
                            ], className="replay-clock-whole"),
                            html.Span(".000000000", id="replay-clock-fraction", className="replay-clock-fraction"),
                        ], className="replay-clock-reading", **{"aria-hidden": "true"}),
                    ], id="replay-clock", className="replay-clock", role="img", **{"aria-label": "Exchange time unavailable"}),
                    card("Prices and volumes", html.Div(id="book-ladder"), "Displayed shares at the selected occupied levels, aggregated by price. Each bar can include multiple orders."),
                    ], className="book-replay-right"),
                ], className="chart-grid book-replay-grid"),
                html.Div(id="event-detail", className="info-note"),
            ], id="book-view", style={"display": "none"}),
            html.Section([
                html.Div([html.Div([
                    html.P("DESIGN AN EXPERIMENT", className="eyebrow"),
                    control("Research question", dcc.Dropdown(id="experiment-method", options=[{"label": "PCA of displayed depth", "value": "pca"}, {"label": "Imbalance and forward response", "value": "response"}, {"label": "Patnaik Pearson dimension", "value": "pp"}], value="pp", clearable=False, searchable=False)),
                    html.Div(control("Sampling interval", dcc.Dropdown(id="sampling", options=[{"label": f"{s} seconds", "value": s} for s in (1, 5, 10, 30, 60)], value=5, clearable=False, searchable=False)), id="sampling-controls"),
                    html.Div(control("Proportion used for training", dcc.Dropdown(id="train-fraction", options=[{"label": f"{round(v*100)}% training / {100-round(v*100)}% evaluation", "value": v} for v in (.5, .6, .7, .8)], value=.7, clearable=False, searchable=False)), id="training-controls"),
                    html.Div(control("Maximum snapshot age", dcc.Dropdown(id="max-age", options=[{"label": f"{v} seconds", "value": v} for v in (5, 15, 30, 60, 300)], value=60, clearable=False, searchable=False)), id="age-controls"),
                    html.Div([control("Depth transformation", dcc.Dropdown(id="transform", options=[{"label": "log(1 + shares)", "value": "log"}, {"label": "Raw shares", "value": "raw"}], value="log", clearable=False, searchable=False)),
                              html.Div([dcc.Checklist(id="standardize", options=[{"label": " Standardise using training data", "value": "yes"}], value=["yes"]), setting_help("Standardise using training data")], className="control-label-row")], id="pca-controls"),
                    html.Div(control("Forward horizon", dcc.Dropdown(id="horizon", options=[{"label": f"{v} seconds", "value": v} for v in (5, 10, 30, 60, 120, 300)], value=30, clearable=False, searchable=False)), id="response-controls", style={"display": "none"}),
                    html.Div([
                        html.Button("Use Tom’s setup", id="pp-collaborator-preset", n_clicks=0, className="button-secondary"),
                        html.Section([
                            html.H3("Features", className="pp-settings-title"),
                                html.P("Choose which measurements describe each snapshot. Standardised uses earlier observations. Both includes the original and standardised versions.", className="small"),
                                *[control(label, dcc.Dropdown(id="pp-feature-" + key,
                                    options=[{"label": label, "value": value} for label, value in
                                             (("Off", "omit"), ("Original", "original"), ("Standardised", "ewm"), ("Both", "both"))],
                                    value=research.TOM_FEATURES[key], clearable=False, searchable=False))
                                  for key, label in research.FEATURE_GROUPS.items()],
                                html.Div([
                                    dcc.Input(id="pp-embedding", value="custom"),
                                    dcc.Input(id="pp-snapshot-scaling", value="none"),
                                ], style={"display": "none"}),
                        ], id="pp-encoding-controls", className="pp-settings-section", **{"aria-label": "Features"}),
                        html.Section([
                            html.H3(["Snapshot ", html.Var(["x", html.Sub("t")]), " settings"], className="pp-settings-title"),
                            html.Div(control("Standardisation history", dcc.Input(id="pp-ewm-lookback", type="number", min=1, step=1, value=501, debounce=True)), id="pp-ewm-controls", style={"display": "none"}),
                            html.P("This mode uses log volumes and distances from the midquote at each price level.", id="pp-fixed-encoding", className="small", style={"display": "none"}),
                        ], className="pp-settings-section", **{"aria-label": "Snapshot x_t settings"}),
                        html.Section([
                            html.H3(["Representation ", html.Var("X"), " settings"], className="pp-settings-title"),
                            control("What forms each row of X?", dcc.Dropdown(id="pp-mode", options=[{"label": "Snapshots at different times", "value": "history"}, {"label": "Price levels within one snapshot", "value": "snapshot"}], value="history", clearable=False, searchable=False)),
                            html.Div([
                                control("Sampling method", dcc.Dropdown(id="pp-sampling", options=[{"label": "Regular clock intervals", "value": "clock"}, {"label": "Every message", "value": "messages"}], value="messages", clearable=False, searchable=False)),
                                html.Div(control("Sampling interval", dcc.Dropdown(id="pp-clock-interval", options=[{"label": f"{v} seconds", "value": v} for v in (1, 5, 10, 30, 60)], value=5, clearable=False, searchable=False)), id="pp-clock-sampling-controls"),
                                html.Div(control("Messages per estimation window", dcc.Dropdown(id="pp-window-messages", options=[100, 250, 500, 1000, 2500, 5000, 10000], value=1000, clearable=False, searchable=False)), id="pp-message-window-controls", style={"display": "none"}),
                                html.Div(control("Estimation window", dcc.Dropdown(id="pp-window", options=[{"label": label, "value": value} for label, value in (("1 minute", 60), ("5 minutes", 300), ("10 minutes", 600), ("30 minutes", 1800), ("60 minutes", 3600))], value=300, clearable=False, searchable=False)), id="pp-clock-window-controls"),
                                control("Snapshots per observation", dcc.Dropdown(id="pp-lags", options=[1, 2, 3, 5], value=1, clearable=False, searchable=False)),
                                html.Div(control("Time between snapshots", dcc.Dropdown(id="pp-lag-spacing", options=[{"label": f"{v} seconds", "value": v} for v in (1, 5, 10, 30, 60)], value=5, clearable=False, searchable=False)), id="pp-clock-lag-controls"),
                                html.Div(control("Messages between snapshots", dcc.Dropdown(id="pp-lag-messages", options=[1, 2, 5, 10, 50, 100], value=1, clearable=False, searchable=False)), id="pp-message-lag-controls", style={"display": "none"}),
                            ], id="pp-history-controls"),
                            html.Div(control("Maximum snapshot age", dcc.Dropdown(id="pp-max-age", options=[{"label": f"{v} seconds", "value": v} for v in (5, 15, 30, 60, 300)], value=60, clearable=False, searchable=False)), id="pp-age-controls"),
                            html.Div(control("Levels in each observation", dcc.Dropdown(id="pp-patch", options=[{"label": "One level on each side", "value": 1}, {"label": "Three consecutive levels on each side", "value": 3}], value=1, clearable=False, searchable=False)), id="pp-snapshot-controls", style={"display": "none"}),
                            control("Column standardisation", dcc.Dropdown(id="pp-matrix-scaling", options=[{"label": "Centre only", "value": "none"}, {"label": "Centre and divide by s", "value": "standardize"}], value="none", clearable=False, searchable=False)),
                        ], className="pp-settings-section", **{"aria-label": "Representation X settings"}),
                        html.Section([
                            html.H3("Time excluded", className="pp-settings-title"),
                            control("Burn in time (seconds)", dcc.Input(id="pp-burn-in", type="number", min=0, step="any", value=0, debounce=True)),
                            control("Burn out time (seconds)", dcc.Input(id="pp-burn-out", type="number", min=0, step="any", value=0, debounce=True)),
                            html.P("Measured from the selected window's start and end. Excluded data are not used, including for standardisation. Zero excludes nothing.", className="small"),
                        ], className="pp-settings-section"),
                        html.Div([
                            control("Calculate dimension every", dcc.Dropdown(id="pp-evaluation-step", options=[{"label": f"{v} seconds", "value": v} for v in (5, 10, 30, 60, 300)], value=60, clearable=False, searchable=False)),
                        ], className="pp-repeat-controls"),
                        html.Div(id="pp-construction", className="pp-summary info-note"),
                        html.Div([
                            control("Scaling", dcc.Dropdown(id="pp-scaling", options=[{"label": "Keep original units", "value": "none"}, {"label": "Standardise within each estimate", "value": "standardize"}, {"label": "Standardise using past observations", "value": "ewm"}], value="none", clearable=False, searchable=False)),
                        ], style={"display": "none"}),
                    ], id="pp-controls", className="pp-controls", style={"display": "none"}),
                    html.Button("Run experiment →", id="run-experiment", n_clicks=0, className="button-primary"),
                    html.P(id="run-status", className="small"),
                    html.Div("The dataset, time window and selected depth also apply here. Changing controls does not change a completed result. Press Run experiment to calculate a new one.", className="info-note"),
                ], className="experiment-controls"),
                html.Div([dcc.Loading(html.Div(id="experiment-output"), custom_spinner=order_book_spinner(),
                                     parent_className="experiment-loading", show_initially=False,
                                     delay_show=150, delay_hide=200),
                    dcc.Store(id="pp-selected-estimate"),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.H3("Inspect an individual estimate", className="card-title"),
                                html.P("Hover above or choose a time.", className="card-note"),
                            ]),
                            control("Estimate ending at", dcc.Dropdown(id="pp-inspect-time", options=[], value=None, clearable=False)),
                        ], className="pp-inspector-heading"),
                        html.Div([
                            html.Div(id="pp-inspect-summary", className="pp-summary"),
                            graph(empty_figure().update_layout(height=280), "pp-spectrum"),
                        ], className="pp-inspector-body"),
                        html.Div(id="pp-inspect-details"),
                    ], id="pp-inspector", className="pp-inspector chart-card", style={"display": "none"}),
                    html.Div(id="experiment-details"),
                ], className="experiment-results"),
                ], className="experiment-grid"),
                experiment_documentation(),
            ], id="experiments-view", style={"display": "none"}),
            html.Section([
                html.Div([html.Div([html.H3("A record of your questions", className="card-title"), html.P("The last eight runs are kept in memory in this browser tab. Download them to retain their results and provenance before refreshing or closing.", className="card-note")]),
                          html.Div([html.Button("Download run records", id="download-button", className="button-primary"), html.Button("Clear this tab’s history", id="clear-runs", className="button-secondary")], className="button-row")], className="controls-card"),
                dcc.Dropdown(id="compare-selection", options=[], value=[], multi=True, placeholder="Choose completed runs to compare"),
                html.Div(id="compare-output", className="empty-state", children="Run an experiment to begin comparing results."),
            ], id="compare-view", style={"display": "none"}),
        ], className="main"),
    ], className="app-shell")


app.layout = layout


app.clientside_callback(
    """
    function(theme) {
        if (window.lobsterSettingHelpInstalled) return true;
        function closeHelp(restoreFocus) {
            document.querySelectorAll('.setting-help[open]').forEach(function(item) {
                item.open = false;
                if (restoreFocus) item.querySelector('summary').focus({preventScroll: true});
            });
        }
        document.addEventListener('toggle', function(event) {
            const item = event.target;
            if (!item.matches('.setting-help') || !item.open) return;
            document.querySelectorAll('.setting-help[open]').forEach(function(other) {
                if (other !== item) other.open = false;
            });
            requestAnimationFrame(function() {
                if (!item.open) return;
                const panel = item.querySelector('.help-popover');
                const anchor = item.querySelector('summary').getBoundingClientRect();
                const box = panel.getBoundingClientRect();
                const width = document.documentElement.clientWidth;
                const height = document.documentElement.clientHeight;
                panel.style.left = Math.max(12, Math.min(anchor.left, width - box.width - 12)) + 'px';
                let top = anchor.bottom + 8;
                if (top + box.height > height - 12) top = anchor.top - box.height - 8;
                panel.style.top = Math.max(12, top) + 'px';
                panel.style.visibility = 'visible';
            });
        }, true);
        document.addEventListener('click', function(event) {
            if (event.target.closest('.help-close')) closeHelp(true);
            else if (!event.target.closest('.setting-help')) closeHelp(false);
        });
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape' && document.querySelector('.setting-help[open]')) {
                closeHelp(true);
                event.preventDefault();
            }
        });
        document.addEventListener('scroll', function(event) {
            if (!event.target.closest || !event.target.closest('.help-popover')) closeHelp(false);
        }, true);
        window.addEventListener('resize', function() { closeHelp(false); });
        window.lobsterSettingHelpInstalled = true;
        return true;
    }
    """,
    Output("help-ready", "data"), Input("theme-applied", "data"),
)


@app.callback(Output("overview-view", "style"), Output("book-view", "style"), Output("experiments-view", "style"), Output("compare-view", "style"), Input("view", "value"))
def switch_view(view):
    return tuple({"display": "block" if view == name else "none"} for name in ("overview", "book", "experiments", "compare"))


@app.callback(Output("dataset-info", "children"), Output("window", "min"), Output("window", "max"), Output("window", "value"), Output("window", "marks"),
              Output("levels", "max"), Output("levels", "value"), Output("levels", "marks"), Input("dataset", "value"), Input("full-window", "n_clicks"), Input("regular-window", "n_clicks"))
def choose_dataset(index, full_clicks, regular_clicks):
    if index is None:
        return "No extracted CSV pairs found. Place them in the data folder and restart.", 0, 1, [0, 1], {}, 1, 1, {1: "1"}
    try:
        data, _ = dataset(index)
        lo = int(data.messages.time_ns.iloc[0] // 10**9)
        hi = int(np.ceil(data.messages.time_ns.iloc[-1] / 1e9))
        if hi <= lo:
            hi = lo + 1
        window = [max(lo, 34200), min(hi, 57600)] if ctx.triggered_id == "regular-window" else [lo, hi]
        if window[0] >= window[1]:
            window = [lo, hi]
        marks = {t: clock(t)[:5] for t in range(((lo // 3600) + 1) * 3600, hi, 3600)}
        summary = [html.Strong(f"{len(data.messages):,} aligned events"), html.P(f"{clock(lo)} to {clock(hi)} · exchange clock"),
                   html.P("" if data.quality.drop(["message_rows", "snapshot_rows", "extra_message_columns", "equal_adjacent_timestamps"]).sum() == 0 else "Review the source checks below the overview.")]
        return summary, lo, hi, window, marks, data.levels, data.levels, {1: "1", data.levels: str(data.levels)}
    except (ValueError, OSError) as exc:
        return html.Div(str(exc), className="error-note"), 0, 1, [0, 1], {}, 1, 1, {1: "1"}


@app.callback(Output("window-label", "children"), Input("window", "value"))
def window_label(window):
    return f"{clock(window[0])} to {clock(window[1])}" if window else ""


@app.callback(Output("overview-kpis", "children"), Output({"type": "chart-source", "key": "price-chart"}, "data"), Output({"type": "chart-source", "key": "depth-chart"}, "data"), Output({"type": "chart-source", "key": "activity-chart"}, "data"), Output("overview-tables", "children"),
              Input("dataset", "value"), Input("window", "value"), Input("levels", "value"), Input("metric", "value"))
def overview(index, window, levels, metric):
    try:
        data, _ = dataset(index)
        view = selected_data(data, window, min(int(levels), data.levels))
        stats = lob.metric_summary(view)
        executions = lob.execution_summary(view)
        spread = stats.loc["spread", "time_mean"]
        imbalance = stats.loc["depth_imbalance", "time_mean"]
        cards = [kpi("Events in window", f"{len(view.messages):,}", "Every selected message row"),
                 kpi("Mean spread", f"${spread:.4f}" if np.isfinite(spread) else "—", "Weighted by elapsed time"),
                 kpi("Executed shares", f"{int(executions.loc['Combined', 'shares']):,}", "Visible + hidden · excludes auctions"),
                 kpi("Mean depth imbalance", f"{imbalance:+.3f}" if np.isfinite(imbalance) else "—", f"Time weighted · {view.levels} levels per side")]
        events = lob.event_summary(view).reset_index()[["event", "count", "reported_shares", "percent_of_events"]]
        checks = data.quality.rename_axis("check").reset_index()
        tables = [card("Event composition", table(events, 2), "Selected interval. Reported shares include nontrade events and must not be summed as traded volume."),
                  card("Source checks", table(checks, 0), "Complete loaded file. Times remain in original order, including equal timestamps."),
                  card("Snapshot statistics", table(stats.reset_index().rename(columns={"index": "metric"})[["metric", "event_mean", "time_mean", "50%", "min", "max", "covered_seconds"]]),
                       "Time means cover the first through last selected event, exclude known halts, and do not extrapolate beyond observations. Event means weight rows equally.", True)]
        return cards, price_chart(view, metric), depth_heatmap(view), activity_chart(view), tables
    except (ValueError, OSError, TypeError) as exc:
        message = str(exc)
        return [html.Div(message, className="error-note")], empty_figure(), empty_figure(), empty_figure(), []


@app.callback(Output({"type": "chart-source", "key": "volatility-chart"}, "data"),
              Input("dataset", "value"), Input("window", "value"), Input("volatility-window", "value"))
def overview_volatility(index, window, volatility_window):
    """Plot rolling realised volatility independently of the other overview charts.

    Parameters
    ----------
    index : int
        Selected dataset.
    window : sequence of float
        Selected start and end in seconds after midnight.
    volatility_window : int
        Trailing volatility window in seconds.

    Returns
    -------
    plotly.graph_objects.Figure
        Volatility in basis points on the complete one second sampling grid.
    """
    try:
        data, _ = dataset(index)
        bounds(data, window)
        result = research.rolling_realised_volatility(data, window[0], window[1],
                                                     window_seconds=volatility_window)
        if not result["realised_volatility_bps"].notna().any():
            return empty_figure("No complete valid window. Select a longer period or a shorter volatility window.")
        minutes = volatility_window / 60
        fig = go.Figure(go.Scattergl(
            x=result.index, y=result["realised_volatility_bps"], mode="lines",
            name=f"{minutes:g} minute realised volatility", line={"color": ACCENT, "width": 1.8},
            connectgaps=False, showlegend=False,
            hovertemplate="Window ending %{x}<br>Realised volatility %{y:.3f} bp<extra></extra>"))
        fig.update_xaxes(title_text="Exchange clock")
        fig.update_yaxes(title_text="Realised volatility (bp)", rangemode="tozero")
        fig.update_layout(uirevision=f"volatility-{data.message_path.name}-{window}-{volatility_window}")
        return styled(fig)
    except (ValueError, OSError, TypeError) as exc:
        return empty_figure(str(exc))


@app.callback(Output("book-row", "min"), Output("book-row", "max"), Output("book-row", "value"), Output("book-row-number", "value"), Output("book-row-number", "max"),
              Input("dataset", "value"), Input("window", "value"), Input("book-row", "value"), Input("book-row-number", "value"),
              Input("previous-row", "n_clicks"), Input("next-row", "n_clicks"), Input("play-timer", "n_intervals"), State("row-step", "value"))
def select_row(index, window, slider, number, previous, following, ticks, step):
    try:
        data, _ = dataset(index)
        first, stop = bounds(data, window)
        trigger = ctx.triggered_id
        row = slider if slider is not None else first
        if trigger in ("dataset", "window", None):
            row = first
        elif trigger == "book-row-number":
            row = number if number is not None else first
        elif trigger == "previous-row":
            row -= int(step or 1)
        elif trigger in ("next-row", "play-timer"):
            row += int(step or 1)
        row = max(first, min(stop - 1, int(row)))
        return first, stop - 1, row, row, stop - 1
    except (ValueError, OSError, TypeError):
        return 0, 1, 0, 0, 1


@app.callback(Output("play-timer", "interval"), Input("play-speed", "value"))
def playback_interval(speed):
    """Convert the selected playback speed to an update interval.

    Parameters
    ----------
    speed : float
        Playback multiplier relative to one update every 800 milliseconds.

    Returns
    -------
    int
        Requested interval in milliseconds. Rendering may limit actual speed.
        Unrecognised selections use the default 4x playback speed.
    """
    if speed not in (.25, .5, 1, 2, 4, 8):
        speed = 4
    return round(800 / speed)


@app.callback(Output("play-timer", "disabled"), Output("play-button", "children"), Input("play-button", "n_clicks"), Input("view", "value"), State("play-timer", "disabled"))
def play_pause(clicks, view, disabled):
    if view != "book":
        return True, "Play"
    if ctx.triggered_id != "play-button":
        return disabled, "Play" if disabled else "Pause"
    return not disabled, "Pause" if disabled else "Play"


def book_axis_ranges(prices, sizes, previous=None):
    """Keep replay axes stable, expanding in chunks when headroom runs out.

    Parameters
    ----------
    prices : array_like
        Displayed prices at the selected levels on both sides.
    sizes : array_like
        Corresponding displayed share counts.
    previous : dict, optional
        Earlier x and y ranges for this browser tab and selection.

    Returns
    -------
    dict
        Explicit x and y ranges. Ranges never shrink until reset. Invalid
        quotes are excluded, and the share axis always starts at zero.

    Raises
    ------
    ValueError
        If no displayed quote has a finite positive price and valid size.
    """
    prices, sizes = np.asarray(prices, dtype=float), np.asarray(sizes, dtype=float)
    valid = np.isfinite(prices) & (prices > 0) & np.isfinite(sizes) & (sizes >= 0)
    prices, sizes = prices[valid], sizes[valid]
    if not len(prices):
        raise ValueError("No valid displayed quotes in this snapshot.")
    low, high, peak = float(prices.min()), float(prices.max()), float(sizes.max())
    padding = max((high - low) * .2, .0001)
    if not previous:
        return {"x": [low - padding, high + padding], "y": [0., max(1., peak * 1.25)]}
    left, right = map(float, previous["x"])
    ceiling = float(previous["y"][1])
    span = right - left
    if low < left + .05 * span:
        left -= max(1, int(np.ceil((left + .2 * span - low) / (.5 * span)))) * .5 * span
    if high > right - .05 * span:
        right += max(1, int(np.ceil((high + .2 * span - right) / (.5 * span)))) * .5 * span
    if peak > .95 * ceiling:
        ceiling = max(2 * ceiling, peak * 1.25)
    return {"x": [left, right], "y": [0., ceiling]}


def replay_clock_value(time_ns, source):
    """Format an exact event timestamp for the replay clock.

    Parameters
    ----------
    time_ns : int
        Nanoseconds since midnight from the message file.
    source : str
        Dataset identifier used to suppress animation on a dataset change.

    Returns
    -------
    dict
        Whole seconds, nine fractional digits and the source identifier.
    """
    seconds, fraction = divmod(int(time_ns), 1_000_000_000)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return {"whole": f"{hours:02d}:{minutes:02d}:{seconds:02d}", "fraction": f"{fraction:09d}", "source": source}


app.clientside_callback(
    """
    function(value) {
        const root = document.getElementById('replay-clock');
        if (!root) return window.dash_clientside.no_update;
        const whole = value ? value.whole : '––:––:––';
        const previous = root.dataset.whole;
        const animate = value && previous && root.dataset.source === value.source &&
            !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        const direction = previous && whole < previous ? -1 : 1;
        for (const i of [0, 1, 3, 4, 6, 7]) {
            const digit = document.getElementById('replay-digit-' + i);
            if (digit.textContent === whole[i]) continue;
            digit.getAnimations().forEach(animation => animation.cancel());
            digit.textContent = whole[i];
            if (animate) digit.animate([
                {transform: 'translateY(' + (direction * 100) + '%)', opacity: 0.3},
                {transform: 'translateY(0)', opacity: 1}
            ], {duration: 160, easing: 'cubic-bezier(.2,.7,.2,1)'});
        }
        document.getElementById('replay-clock-fraction').textContent = value ? '.' + value.fraction : '.–––––––––';
        root.dataset.whole = value ? whole : '';
        root.dataset.source = value ? value.source : '';
        return value ? 'Exchange time ' + whole + '.' + value.fraction : 'Exchange time unavailable';
    }
    """,
    Output("replay-clock", "aria-label"), Input("book-clock-value", "data"),
)


@app.callback(Output({"type": "chart-source", "key": "snapshot-chart"}, "data"), Output("book-ladder", "children"), Output("event-detail", "children"), Output("book-axis-state", "data"), Output("book-clock-value", "data"),
              Input("dataset", "value"), Input("book-row", "value"), Input("levels", "value"), Input("window", "value"),
              Input("refit-book-axes", "n_clicks"), Input("book-size-cap", "value"), State("book-axis-state", "data"))
def show_book(index, row, levels, window=None, refits=0, size_cap=2500, axis_state=None):
    """Render an order book snapshot with persistent axis ranges and its midquote.

    Parameters
    ----------
    index : int
        Selected dataset index.
    row : int
        Requested original event row.
    levels : int
        Occupied levels to display on each side.
    window : sequence of float, optional
        Selected exchange clock interval.
    refits : int, optional
        Refit button count, used to reset saved ranges and manual zoom.
    size_cap : int, optional
        Maximum displayed bar height in shares. Zero displays full sizes.
        Original sizes remain in labels, tooltips, and the book ladder.
    axis_state : dict, optional
        Axis ranges and selection key held in this browser tab.

    Returns
    -------
    tuple
        Chart, ladder, event description, updated axis state and exact clock time.
    """
    try:
        data, _ = dataset(index)
        first, stop = bounds(data, window) if window else (0, len(data.messages))
        row = max(first, min(stop - 1, int(row or 0)))
        levels = min(int(levels), data.levels)
        b, m = data.book.iloc[row], data.messages.iloc[row]
        key = json.dumps([index, signature(index), levels, window, refits])
        previous = axis_state.get("ranges") if axis_state and axis_state.get("key") == key else None
        prices = [b[f"{side}_price_{k}"] for side in ("bid", "ask") for k in range(1, levels + 1)]
        sizes = [b[f"{side}_size_{k}"] for side in ("bid", "ask") for k in range(1, levels + 1)]
        ranges = book_axis_ranges(prices, sizes, previous)
        if size_cap not in BOOK_SIZE_CAPS:
            size_cap = 2500
        ceiling = float(size_cap) if size_cap else ranges["y"][1]
        fig = go.Figure()
        clipped = []
        for side, color in (("bid", BID), ("ask", ASK)):
            side_prices = [b[f"{side}_price_{k}"] for k in range(1, levels + 1)]
            actual_sizes = [int(b[f"{side}_size_{k}"]) for k in range(1, levels + 1)]
            displayed_sizes = [min(size, size_cap) if size_cap else size for size in actual_sizes]
            customdata = [[size, f"<br>Bar capped at {size_cap:,} shares" if size_cap and size > size_cap else ""] for size in actual_sizes]
            fig.add_bar(x=side_prices, y=displayed_sizes, customdata=customdata,
                        name=side.title(), marker_color=color, marker_line_width=0,
                        marker_opacity=[max(0.5, 0.95 ** (k - 1)) for k in range(1, levels + 1)],
                        hovertemplate="Price $%{x:.4f}<br>%{customdata[0]:,.0f} shares%{customdata[1]}<extra>%{fullData.name}</extra>")
            clipped.extend((float(price), size) for price, size in zip(side_prices, actual_sizes)
                           if size_cap and size > size_cap and np.isfinite(price) and price > 0)
        # Stagger nearby labels into rows so large sizes remain legible even
        # when several adjacent price levels exceed the cap.
        lane_ends = []
        span = ranges["x"][1] - ranges["x"][0]
        for price, size in sorted(clipped):
            label = f"{size:,}"
            half_width = span * (7 * len(label) + 4) / (2 * 180)
            lane = next((i for i, end in enumerate(lane_ends) if end < price - half_width), len(lane_ends))
            if lane == len(lane_ends):
                lane_ends.append(price + half_width)
            else:
                lane_ends[lane] = price + half_width
            fig.add_annotation(x=price, y=ceiling, text=label, showarrow=False,
                               xanchor="center", yanchor="top", yshift=-4 - 15 * lane,
                               font={"color": "#ffffff", "size": 11}, bgcolor=PANEL, borderpad=1)
        mid = (b["bid_price_1"] + b["ask_price_1"]) / 2
        if np.isfinite(mid) and b["bid_price_1"] > 0 and b["ask_price_1"] > 0:
            fig.add_vline(x=float(mid), line_color=ACCENT, line_dash="dot", line_width=2,
                          name="Midquote", showlegend=True, legendrank=1001)
        fig.update_xaxes(title_text="Price (USD)", range=ranges["x"], autorange=False)
        fig.update_yaxes(title_text="Shares (bars capped)" if size_cap else "Displayed shares",
                         range=[0, ceiling], autorange=False, uirevision=f"{key}:{size_cap}")
        fig.update_layout(uirevision=key)
        ladder = pd.DataFrame([{ "level": k, "bid_size": int(b[f"bid_size_{k}"]), "bid_price": b[f"bid_price_{k}"],
                                 "ask_price": b[f"ask_price_{k}"], "ask_size": int(b[f"ask_size_{k}"])} for k in range(1, levels + 1)])
        side = "buy" if m.direction == 1 else "sell" if m.direction == -1 else "not applicable"
        price = f"${m.price:.4f}" if pd.notna(m.price) else f"status {m.price_raw}"
        detail = [html.Strong(f"Row {row:,} · {m.timestamp} · {m.event_name}"), html.Br(),
                  f"Reported size {int(m['size']):,} shares · price {price} · resting side {side} · order ID {m.order_id}"]
        clock_value = replay_clock_value(m.time_ns, f"{index}:{data.message_path.name}")
        return styled(fig, BOOK_CHART_HEIGHT), table(ladder), detail, {"key": key, "ranges": ranges}, clock_value
    except (ValueError, OSError, TypeError) as exc:
        return empty_figure().update_layout(height=BOOK_CHART_HEIGHT), [], str(exc), None, None


@app.callback(Output("pca-controls", "style"), Output("response-controls", "style"), Output("pp-controls", "style"),
              Output("training-controls", "style"), Output("sampling-controls", "style"), Output("age-controls", "style"),
              Input("experiment-method", "value"), Input("pp-mode", "value"), Input("pp-sampling", "value"))
def method_controls(method, mode, sampling):
    show = {"display": "block"}
    hide = {"display": "none"}
    return (show if method == "pca" else hide, show if method == "response" else hide,
            show if method == "pp" else hide, hide if method == "pp" else show,
            hide if method == "pp" else show,
            hide if method == "pp" else show)


@app.callback(Output("pp-features", "data"),
              *[Input("pp-feature-" + key, "value") for key in research.FEATURE_GROUPS])
def selected_features(*choices):
    """Collect the selected snapshot feature transformations.

    Parameters
    ----------
    choices : tuple of str
        One transformation for each feature group.

    Returns
    -------
    dict
        Explicit choices for calculation and exports.
    """
    return dict(zip(research.FEATURE_GROUPS, choices))


@app.callback(Output("pp-history-controls", "style"), Output("pp-snapshot-controls", "style"), Output("pp-construction", "children"),
              Output("pp-encoding-controls", "style"), Output("pp-fixed-encoding", "style"),
              Input("pp-mode", "value"), Input("pp-embedding", "value"), Input("levels", "value"), Input("pp-lags", "value"),
              Input("pp-lag-spacing", "value"), Input("pp-window", "value"), Input("pp-patch", "value"), Input("pp-scaling", "value"),
              Input("pp-sampling", "value"), Input("pp-window-messages", "value"), Input("pp-lag-messages", "value"), Input("pp-ewm-lookback", "value"), Input("pp-features", "data"))
def dimension_controls(mode, embedding, levels, lags, spacing, window, patch, scaling, sampling, message_window, message_spacing, lookback, features):
    if mode == "snapshot":
        points = max(0, int(levels) - int(patch) + 1)
        construction = ("Each observation combines the volume and distance from the midquote at a bid price with those at the ask price of the same level number. "
                        if int(patch) == 1 else
                        "Each observation combines volumes and distances from the midquote for three consecutive levels on each side. The groups start at levels 1, 2, 3 and so on. ")
        text = (construction + f"There are at most {points} observations, each containing {4 * int(patch)} numbers. "
                "The calculation describes differences between price levels within this snapshot.")
    else:
        coordinates = research.feature_coordinate_count(features or {}, int(levels)) * int(lags)
        units, spacing, window = ("messages", message_spacing, message_window) if sampling == "messages" else ("seconds", spacing, window)
        text = (f"Each observation contains {coordinates} numbers from {lags} snapshot{'s' if int(lags) != 1 else ''}. "
                f"Each estimate uses the preceding {int(window):,} {units} before exclusions.")
        if int(lags) > 1:
            text += f" The snapshots in an observation are {int(spacing)} {units} apart."
        if embedding == "collaborator" and int(levels) != 10:
            text += " The repository uses ten levels per side."
    if scaling == "ewm" or any(v in ("ewm", "both") for v in (features or {}).values()):
        text += f" Standardisation uses earlier sampled observations and discards the first {lookback} after the start or an invalid sample."
    if scaling == "standardize":
        text += " Each variable is divided by its standard deviation within the observations used for that estimate. This changes the relative weight of the variables."
    return ({"display": "block" if mode == "history" else "none"}, {"display": "block" if mode == "snapshot" else "none"}, text,
            {"display": "block" if mode == "history" else "none"}, {"display": "none" if mode == "history" else "block"})


@app.callback(Output("pp-lag-spacing", "options"), Output("pp-lag-spacing", "value"),
              Output("pp-evaluation-step", "options"), Output("pp-evaluation-step", "value"),
              Input("pp-clock-interval", "value"), Input("pp-mode", "value"), Input("pp-sampling", "value"),
              Input("pp-collaborator-preset", "n_clicks"), State("pp-lag-spacing", "value"), State("pp-evaluation-step", "value"),
              State("window", "value"), State("pp-window-messages", "value"), State("levels", "max"))
def dimension_spacing_options(step, mode, sampling, preset_clicks, lag, evaluation, window, message_window, maximum_levels):
    step = int(step)
    lags = [step * multiplier for multiplier in (1, 2, 3, 6, 12)]
    preset = ctx.triggered_id == "pp-collaborator-preset" and preset_clicks
    if preset:
        sampling = "messages"
        # Choose a safe initial sweep frequency without reducing message samples.
        coordinates = research.feature_coordinate_count(research.TOM_FEATURES, min(10, int(maximum_levels)))
        for interval in (60, 300, 600, 1800, 3600, 7200):
            count = max(0, int(np.floor(window[1] / interval) - np.ceil(window[0] / interval) + 1))
            cost = count * int(message_window) * coordinates * min(int(message_window), coordinates)
            if cost <= research.MAX_DIMENSION_SVD_WORK and count > 0:
                evaluation = interval
                break
    evaluations = [5, 10, 30, 60, 300, 600, 1800, 3600, 7200] if mode == "snapshot" or sampling == "messages" else sorted({step, step * 2, step * 6, step * 12, 300})
    evaluations = [value for value in evaluations if mode == "snapshot" or sampling == "messages" or value % step == 0]
    return ([{"label": f"{s} seconds", "value": s} for s in lags], lag if lag in lags else step,
            [{"label": f"{s} seconds", "value": s} for s in evaluations], evaluation if evaluation in evaluations else next((s for s in evaluations if s >= 60), evaluations[-1]))


@app.callback(Output("pp-message-window-controls", "style"), Output("pp-clock-window-controls", "style"),
              Output("pp-message-lag-controls", "style"), Output("pp-clock-lag-controls", "style"), Output("pp-ewm-controls", "style"),
              Output("pp-clock-sampling-controls", "style"), Output("pp-age-controls", "style"),
              Input("pp-sampling", "value"), Input("pp-scaling", "value"), Input("pp-mode", "value"), Input("pp-lags", "value"), Input("pp-features", "data"))
def history_control_visibility(sampling, scaling, mode, lags, features):
    show, hide = {"display": "block"}, {"display": "none"}
    messages = sampling == "messages"
    return (show if messages else hide, hide if messages else show,
            show if messages and int(lags) > 1 else hide,
            show if not messages and int(lags) > 1 else hide,
            show if mode == "history" and any(v in ("ewm", "both") for v in (features or {}).values()) else hide,
            hide if messages else show, show if mode == "snapshot" or not messages else hide)


@app.callback(Output("pp-scaling", "value"), Output("pp-snapshot-scaling", "value"), Output("pp-matrix-scaling", "value"),
              Input("pp-snapshot-scaling", "value"), Input("pp-matrix-scaling", "value"),
              Input("pp-mode", "value"), Input("pp-collaborator-preset", "n_clicks"))
def dimension_scaling_controls(snapshot_scaling, matrix_scaling, mode, preset_clicks):
    """Place the existing, mutually exclusive scaling choices in their sections.

    Parameters
    ----------
    snapshot_scaling, matrix_scaling : str
        Visible choices for snapshot or matrix standardisation.
    mode : str
        Construction of rows from history or levels in one snapshot.
    preset_clicks : int
        Number of requests to apply Tom's setup.

    Returns
    -------
    tuple of str
        Research scaling value and the two visible choices.
    """
    trigger = ctx.triggered_id
    if trigger == "pp-collaborator-preset" and preset_clicks:
        return "none", "none", "none"
    if mode == "snapshot":
        snapshot_scaling = "none"
    if trigger == "pp-snapshot-scaling" and snapshot_scaling == "ewm":
        matrix_scaling = "none"
    elif trigger == "pp-matrix-scaling" and matrix_scaling == "standardize":
        snapshot_scaling = "none"
    scaling = "ewm" if snapshot_scaling == "ewm" else matrix_scaling
    return scaling, snapshot_scaling, matrix_scaling


@app.callback(Output("pp-mode", "value"), Output("pp-embedding", "value"), Output("pp-sampling", "value"),
              Output("pp-lags", "value"), Output("pp-ewm-lookback", "value"),
              Output("levels", "value", allow_duplicate=True),
              *[Output("pp-feature-" + key, "value") for key in research.FEATURE_GROUPS],
              Input("pp-collaborator-preset", "n_clicks"), State("levels", "max"), prevent_initial_call=True)
def collaborator_preset(clicks, maximum_levels):
    if not clicks:
        raise PreventUpdate
    return ("history", "custom", "messages", 1, 501, min(10, int(maximum_levels)), *research.TOM_FEATURES.values())


@app.callback(Output("experiment-output", "children"), Output("run-history", "data"), Output("run-status", "children"), Output("dimension-result", "data"), Output("experiment-details", "children"),
              Input("run-experiment", "n_clicks"), Input("clear-runs", "n_clicks"),
              State("experiment-method", "value"), State("dataset", "value"), State("window", "value"), State("levels", "value"),
              State("sampling", "value"), State("horizon", "value"), State("transform", "value"), State("standardize", "value"),
              State("train-fraction", "value"), State("max-age", "value"), State("run-history", "data"),
              State("pp-mode", "value"), State("pp-embedding", "value"), State("pp-patch", "value"), State("pp-lags", "value"),
              State("pp-lag-spacing", "value"), State("pp-window", "value"), State("pp-evaluation-step", "value"), State("pp-scaling", "value"),
              State("pp-sampling", "value"), State("pp-window-messages", "value"), State("pp-lag-messages", "value"), State("pp-ewm-lookback", "value"), State("pp-clock-interval", "value"), State("pp-max-age", "value"),
              State("pp-burn-in", "value"), State("pp-burn-out", "value"), State("pp-features", "data"), prevent_initial_call=True,
              running=[(Output("run-experiment", "disabled"), True, False)])
def execute(run_clicks, clear_clicks, method, index, window, levels, step, horizon, transform, standardize, fraction, max_age, history,
            pp_mode, pp_embedding, pp_patch, pp_lags, pp_lag_spacing, pp_window, pp_evaluation_step, pp_scaling, pp_sampling, pp_window_messages, pp_lag_messages, pp_ewm_lookback, pp_clock_interval, pp_max_age, pp_burn_in, pp_burn_out, pp_features):
    if ctx.triggered_id == "clear-runs":
        return no_update, [], "History cleared. The last displayed result remains visible.", no_update, no_update
    try:
        dimension_settings = {"mode": pp_mode, "embedding": pp_embedding, "patch_width": pp_patch, "lag_count": pp_lags,
                              "lag_spacing_seconds": pp_lag_spacing, "window_seconds": pp_window,
                              "evaluation_step_seconds": pp_evaluation_step, "scaling": pp_scaling,
                              "sampling": pp_sampling, "window_messages": pp_window_messages,
                              "lag_spacing_messages": pp_lag_messages, "ewm_lookback": pp_ewm_lookback,
                              "burn_in_seconds": pp_burn_in, "burn_out_seconds": pp_burn_out, "features": pp_features}
        if method == "pp":
            step, max_age = pp_clock_interval, pp_max_age
        result = run_record(method, index, window, levels, step, horizon, transform, "yes" in (standardize or []), fraction, max_age, dimension_settings)
        metric_labels = {"first_component_variance": "First component explained variance fraction",
                         "last_dimension": "Latest valid PP dimension", "last_normalized_dimension": "Latest valid dimension / coordinates",
                         "last_point_count": "Points in latest valid estimate"}
        metrics = pd.DataFrame([{"metric": metric_labels.get(key, key.replace("_", " ")), "value": val} for key, val in result["metrics"].items()])
        counts = pd.DataFrame([{"sample_check": key, "count": val} for key, val in result["counts"].items()])
        params = json.dumps(result["parameters"], indent=2)
        interpretation = ("The result depends on which volumes and prices are included and how they are scaled. Comparing levels within one snapshot and comparing snapshots through time measure different kinds of variation."
                          if method == "pp" else "The evaluation period is later in the same file. It does not demonstrate stability across trading days.")
        method_note = ("PP = (sum of singular values)² / sum of squared singular values, after subtracting each column's mean and applying the selected scaling. Each estimate uses only observations at or before its stated time."
                       if method == "pp" else "Each sampling time uses the snapshot after the last message at or before that time. Invalid quotes, locked or crossed quotes, trading halts and observations older than the chosen age limit are excluded. The training and evaluation periods are determined before exclusions. No significance tests or trading performance claims are made.")
        latest_dimension = next((point for point in reversed(result.get("series", [])) if point["dimension"] is not None), None)
        dimension_summary = (html.P(f"{result['counts']['successful_points']} valid estimates of {result['counts']['evaluation_points']} requested. " +
                                    (f"Latest valid PP {latest_dimension['dimension']:.3f} at {clock(latest_dimension['seconds'])}." if latest_dimension else "No defined estimate. Check how the observations are constructed and how many remain after exclusions."), className="pp-summary")
                             if method == "pp" else None)
        contents = [html.Div([html.P("COMPLETED EXPERIMENT", className="eyebrow"), html.H2(METHOD_LABELS[method]),
                               html.P(f"Run {result['run_id']} · {result['seconds']:.2f} seconds · {clock(window[0])} to {clock(window[1])}", className="run-meta"),
                               dimension_summary,
                               html.P(f"Excluded {result['parameters']['burn_in_seconds']:g} seconds at the start and {result['parameters']['burn_out_seconds']:g} seconds at the end before calculation.", className="run-meta") if method == "pp" else None,
                               html.P(interpretation, className="card-note")], className="run-card"),
                    html.Div(result_figures(result), className="chart-grid"),
                    html.Div([card("Measurements", table(metrics, 6)), card("Observations included and excluded", table(counts, 0))], className="chart-grid"),
                    html.Details([html.Summary("Methods and exact settings"), html.Pre(params, className="mono"),
                                  html.P(method_note),
                                  html.A("Estimator definition in the source paper", href="https://arxiv.org/html/2606.19268v4#S3.SS3", target="_blank", rel="noopener noreferrer") if method == "pp" else None,
                                  html.Ul([html.Li(note) for note in result.get("notes", [])]),
                                  html.P("Download the results to keep the software versions and file hashes identifying the data and code used.")], className="info-note")]
        return contents[:2], ((history or []) + [result])[-8:], f"Completed run {result['run_id']} in {result['seconds']:.2f}s", result if method == "pp" else None, contents[2:]
    except (ValueError, OSError, TypeError, np.linalg.LinAlgError) as exc:
        return html.Div([html.Strong("The experiment could not run."), html.P(str(exc))], className="error-note"), no_update, "Adjust the settings and try again.", None, []


@app.callback(Output("pp-inspector", "style"), Output("pp-inspect-time", "options"), Output("pp-inspect-time", "value"),
              Input("dimension-result", "data"))
def dimension_inspector_options(result):
    if not result:
        return {"display": "none"}, [], None
    records = result["series"]
    options = [{"label": f"{clock(r['seconds'])} · " + (f"PP {r['dimension']:.3f}" if r["dimension"] is not None else r["status"].replace("_", " ")), "value": i}
               for i, r in enumerate(records)]
    selected = next((i for i in range(len(records) - 1, -1, -1) if records[i]["dimension"] is not None), len(records) - 1)
    return {"display": "block"}, options, selected


app.clientside_callback(
    """
    function(hover, result, selected) {
        const unchanged = window.dash_clientside.no_update;
        if (!hover || !result || !hover.points) return unchanged;
        for (const point of hover.points) {
            const data = point.customdata;
            if (!data || data[4] !== result.run_id) continue;
            const index = data[3];
            if (Number.isInteger(index) && index >= 0 && index < result.series.length) {
                return index === selected ? unchanged : index;
            }
        }
        return unchanged;
    }
    """,
    Output("pp-inspect-time", "value", allow_duplicate=True),
    Input({"type": "themed-chart", "key": "pp-dimension-time"}, "hoverData", allow_optional=True),
    State("dimension-result", "data"), State("pp-inspect-time", "value"),
    prevent_initial_call=True,
)

# Select in the browser so hovering sends only one estimate to the server,
# rather than retransmitting every spectrum in the completed experiment.
app.clientside_callback(
    """
    function(position, result) {
        if (!result || !Number.isInteger(position) || position < 0 || position >= result.series.length) return null;
        return result.series[position];
    }
    """,
    Output("pp-selected-estimate", "data"),
    Input("pp-inspect-time", "value"), Input("dimension-result", "data"),
)


@app.callback(Output("pp-inspect-summary", "children"), Output({"type": "chart-source", "key": "pp-spectrum"}, "data"), Output("pp-inspect-details", "children"),
              Input("pp-selected-estimate", "data"))
def inspect_dimension(point):
    if point is None:
        return [], empty_figure("Run a dimension experiment to inspect its spectrum").update_layout(height=280), []
    dim = point["dimension"]
    normalized = point["normalized_dimension"]
    summary = html.Div([
        kpi("PP dimension", f"{dim:.3f}" if dim is not None else "Undefined", point["status"].replace("_", " ")),
        kpi("Points", f"{point['point_count']:,}", "Retained observations"),
        kpi("Coordinates", str(point["coordinate_count"]), "Numbers in each observation"),
        kpi("Dimension / coordinates", f"{normalized:.3f}" if normalized is not None else "—", f"Upper bound on dimension {point['rank_bound']}"),
    ], className="kpi-grid")
    values = point.get("singular_values", [])
    if values:
        spectrum = go.Figure(go.Bar(x=list(range(1, len(values) + 1)), y=values, marker_color=BID))
        spectrum.update_yaxes(title_text="Singular value / largest", rangemode="tozero")
        spectrum.update_xaxes(title_text="Singular value index", dtick=5 if len(values) > 10 else 1)
        spectrum = styled(spectrum, 280)
    else:
        spectrum = empty_figure("No nonzero spectrum for this estimate").update_layout(height=280)
    spectrum.update_layout(margin={"l": 50, "r": 12, "t": 12, "b": 48})
    detail = {key: value for key, value in point.items() if key not in {"singular_values"} and not isinstance(value, (dict, list))}
    details = [html.Details([html.Summary("Spectrum and observation details"),
                            html.P("PP uses singular values, rather than their squares. The plotted spectrum is divided by its largest value, which leaves PP unchanged.", className="card-note"),
                            table(pd.DataFrame([{"quantity": key.replace("_", " "), "value": value} for key, value in detail.items()]), 6)], className="info-note")]
    return summary, spectrum, details


@app.callback(Output("compare-selection", "options"), Output("compare-selection", "value"), Input("run-history", "data"))
def compare_options(history):
    options = []
    for record in history or []:
        params = record["parameters"]
        if record["method"] == "pp":
            description = (f"snapshot · {params['patch_width']} level{'s' if params['patch_width'] != 1 else ''} per side in each observation" if params["mode"] == "snapshot" else
                           f"history · {EMBEDDING_LABELS.get(params['embedding'], params['embedding'])} · {params['lag_count']} snapshots per observation")
        else:
            description = f"{params.get('step_seconds', '?')}s grid"
        options.append({"label": f"{record['run_id']} · {record['method'].upper()} · {description}", "value": record["run_id"]})
    return options, [r["run_id"] for r in (history or [])[-2:]]


@app.callback(Output("compare-output", "children"), Output("compare-output", "className"), Input("compare-selection", "value"), State("run-history", "data"))
def compare_runs(selection, history):
    runs = [r for r in (history or []) if r["run_id"] in (selection or [])]
    if not runs:
        return "Choose completed runs to compare. Each record includes its exact parameters.", "empty-state"
    summary = []
    for r in runs:
        p = r["parameters"]
        summary.append({"run": r["run_id"], "method": r["method"], "dataset": " · ".join(r["dataset"].split("_")[:2]),
                        "levels": p.get("levels"), "sampling_s": p.get("step_seconds"), "training_share": p.get("train_fraction"),
                        "horizon_s": p.get("horizon_seconds", "—"), "object": p.get("mode", "—"),
                        "embedding": p.get("embedding", "—"), "sampling": p.get("sampling", "clock"),
                        "window_messages": p.get("window_messages", "—"), "window_s": p.get("window_seconds", "—"),
                        "burn_in_s": p.get("burn_in_seconds", "—"), "burn_out_s": p.get("burn_out_seconds", "—"),
                        "lag_spacing_messages": p.get("lag_spacing_messages", "—"), "standardisation_history": p.get("ewm_lookback", "—"),
                        "snapshots_per_vector": p.get("lag_count", "—"), "lag_spacing_s": p.get("lag_spacing_seconds", "—"),
                        "patch_width": p.get("patch_width", "—"), "scaling": p.get("scaling", "—"),
                        "evaluation_step_s": p.get("evaluation_step_seconds", "—"), "elapsed_s": r["seconds"]})
    figures = []
    pca = [r for r in runs if r["method"] == "pca"]
    response = [r for r in runs if r["method"] == "response"]
    dimensions = [r for r in runs if r["method"] == "pp"]
    if pca:
        fig = go.Figure()
        for r in pca:
            fig.add_scatter(x=list(range(1, len(r["cumulative_variance"]) + 1)), y=np.array(r["cumulative_variance"]) * 100, name=r["run_id"], mode="lines+markers")
        fig.update_xaxes(title_text="Components")
        fig.update_yaxes(title_text="Training variance explained (%)")
        figures.append(card("PCA variance profiles", graph(styled(fig)), "Changing variables or scaling changes the meaning of explained variance."))
    if response:
        fig = go.Figure()
        for r in response:
            fig.add_scatter(x=r["bin_centers"], y=r["test_means"], name=r["run_id"], mode="lines+markers")
        fig.update_xaxes(title_text="Imbalance")
        fig.update_yaxes(title_text="Evaluation mean response (bp)")
        figures.append(card("Evaluation response profiles", graph(styled(fig)), "Compare horizons, sampling choices and sample counts before drawing conclusions."))
    for mode in ("snapshot", "history"):
        matched = [r for r in dimensions if r["parameters"]["mode"] == mode]
        if not matched:
            continue
        fig = go.Figure()
        for r in matched:
            records = r["series"]
            fig.add_scatter(x=[point["time"] for point in records], y=[point["dimension"] for point in records], name=r["run_id"], mode="lines+markers", connectgaps=False, marker_size=4)
        fig.update_xaxes(title_text="Exchange clock")
        fig.update_yaxes(title_text="Patnaik Pearson dimension", rangemode="tozero")
        figures.append(card("Dimension across price levels" if mode == "snapshot" else "Dimension through time", graph(styled(fig)),
                            "Changing the variables, their scaling, the times included or the number of observations can change the dimension. Check these settings when comparing results.", True))
    settings = pd.DataFrame(summary)
    settings = settings.loc[:, [col for col in settings if not settings[col].map(lambda value: pd.isna(value) or value == "—").all()]]
    return [card("Selected run settings", table(settings)), html.Div(figures, className="chart-grid"),
            html.Details([html.Summary("Full parameters for selected runs"), html.Pre(json.dumps([{ "run_id": r["run_id"], "parameters": r["parameters"], "counts": r["counts"]} for r in runs], indent=2), className="mono")], className="info-note")], ""


@app.callback(Output("download-runs", "data"), Input("download-button", "n_clicks"), State("run-history", "data"), prevent_initial_call=True)
def download_runs(clicks, history):
    if not history:
        raise PreventUpdate
    return dcc.send_string(json.dumps({"format": "lobster-research-runs-v1", "runs": history}, indent=2, allow_nan=False), "lobster_experiment_records.json")


def main():
    """Start the local application and optionally open the browser.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="LOBSTER Research studio")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if not PATHS:
        raise SystemExit("No extracted LOBSTER CSV pairs were found in the data folder. Add matching message and orderbook CSV files there, then restart.")
    if not 1024 <= args.port <= 65535:
        raise SystemExit("Choose a port between 1024 and 65535.")
    url = f"http://127.0.0.1:{args.port}"
    # Recognize an existing copy when the Finder launcher is opened twice.
    try:
        from urllib.request import urlopen
        with urlopen(url + "/health", timeout=.5) as response:
            existing = json.load(response).get("application") == "lobster-research-studio"
    except Exception:
        existing = False
    if existing:
        if args.open_browser:
            import webbrowser
            webbrowser.open(url)
        print(f"LOBSTER Research studio is already running at {url}")
        return
    print("Loading the first dataset. Source files are read only.", flush=True)
    dataset(0)
    if args.open_browser:
        import webbrowser
        timer = threading.Timer(1.0, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    print(f"Open {url}. Press Ctrl+C in this terminal to stop.", flush=True)
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False, dev_tools_disable_version_check=True)


if __name__ == "__main__":
    main()
