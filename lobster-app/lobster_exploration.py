"""Explore paired LOBSTER CSV files without changing the source data.

Dependencies
------------
numpy, pandas, plotly. The companion notebook also uses Jupyter and ipywidgets.

Examples
--------
python -B lobster_exploration.py --summary-only
python -B lobster_exploration.py

Notes
-----
The command line dashboard is served from memory on localhost. This module
does not save figures, reports, transformed data, or browser files.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd


EVENT_NAMES = {
    1: "Submission", 2: "Partial cancellation", 3: "Deletion",
    4: "Visible execution", 5: "Hidden execution", 6: "Cross / auction",
    7: "Trading halt indicator",
}
PRICE_SCALE = 10_000
DUMMY_PRICE = 9_999_999_999
BID_COLOR = "#087f8c"
ASK_COLOR = "#d65a45"


@dataclass
class LobsterData:
    """Aligned messages, visible book snapshots, and derived measurements.

    Attributes
    ----------
    messages : pandas.DataFrame
        Original message fields plus exact timestamps and interpreted labels.
    book : pandas.DataFrame
        Prices in dollars and displayed sizes in shares. Dummy prices are NaN.
    features : pandas.DataFrame
        One row per message, calculated from the snapshot after that message.
    quality : pandas.Series
        Counts of relevant data quality conditions.
    message_path : pathlib.Path
        Source message CSV.
    orderbook_path : pathlib.Path
        Source orderbook CSV.
    levels : int
        Number of occupied price levels per side in the source files.
    preview : bool
        Whether a row limit was requested. Limited loads are not full file audits.
    """

    messages: pd.DataFrame
    book: pd.DataFrame
    features: pd.DataFrame
    quality: pd.Series
    message_path: Path
    orderbook_path: Path
    levels: int
    preview: bool


def discover_datasets(directory):
    """Find extracted message files with matching orderbook files.

    Parameters
    ----------
    directory : str or pathlib.Path
        Directory to search recursively. Archives are not extracted.

    Returns
    -------
    list of pathlib.Path
        Sorted message paths whose corresponding orderbook files exist.
    """
    return [p for p in sorted(Path(directory).expanduser().rglob("*_message_*.csv"))
            if p.with_name(p.name.replace("_message_", "_orderbook_")).is_file()]


def _time_nanoseconds(values):
    """Parse seconds after midnight without floating point timestamp rounding.

    Parameters
    ----------
    values : pandas.Series
        Decimal strings with at most nine fractional digits.

    Returns
    -------
    pandas.Series
        Integer nanoseconds after midnight.
    """
    text = values.astype("string")
    if not text.str.fullmatch(r"\d+(?:\.\d{1,9})?").all():
        raise ValueError("Times must be nonnegative seconds with at most 9 decimal places.")
    parts = text.str.partition(".")
    result = parts[0].astype("int64") * 10**9 + parts[2].str.ljust(9, "0").astype("int64")
    if (result >= 86_400 * 10**9).any():
        raise ValueError("A message timestamp falls outside its calendar date.")
    return result


def load_lobster(message_path, orderbook_path=None, max_rows=None):
    """Load and validate a pair of LOBSTER files while preserving row order.

    Parameters
    ----------
    message_path : str or pathlib.Path
        Message CSV with a date and level count in its standard filename.
    orderbook_path : str or pathlib.Path, optional
        Matching book CSV. Inferred from the message filename when omitted.
    max_rows : int, optional
        Load only the first rows of both files for a partial preview.

    Returns
    -------
    LobsterData
        Aligned messages, cleaned book, derived features, and quality counts.

    Raises
    ------
    ValueError
        If file structure, row counts, timestamps, or sizes are invalid.

    Notes
    -----
    Snapshot k is the visible book AFTER event k. The sixth message column
    describes the resting order side, so execution aggressor sign is its negative.
    Columns beyond the documented six are retained as extra_1, extra_2, etc.
    No interpretation of those undocumented fields is required for the analysis.
    """
    message_path = Path(message_path).expanduser().resolve()
    orderbook_path = (Path(orderbook_path).expanduser().resolve() if orderbook_path
                      else message_path.with_name(message_path.name.replace("_message_", "_orderbook_")))
    match = re.search(r"_(\d{4}-\d{2}-\d{2})_.*_message_(\d+)\.csv$", message_path.name)
    if match is None:
        raise ValueError("Expected a standard LOBSTER message filename containing date and levels.")
    if max_rows is not None and (not isinstance(max_rows, int) or max_rows < 1):
        raise ValueError("max_rows must be a positive integer or None.")
    with message_path.open(newline="") as stream:
        first = next(csv.reader(stream), [])
    if len(first) < 6:
        raise ValueError("The message file must contain at least six columns.")
    names = ["time_raw", "event_type", "order_id", "size", "price_raw", "direction"]
    names += [f"extra_{i}" for i in range(1, len(first) - 5)]
    dtypes = {name: "int64" for name in names[1:6]}
    dtypes.update({name: "string" for name in ["time_raw"] + names[6:]})
    messages = pd.read_csv(message_path, header=None, names=names, dtype=dtypes,
                           nrows=max_rows, keep_default_na=False)
    if messages.empty:
        raise ValueError("The message file is empty.")
    for name in names[6:]:
        messages[name] = messages[name].replace({"null": pd.NA, "": pd.NA})
    raw_book = pd.read_csv(orderbook_path, header=None, dtype="int64", nrows=max_rows)
    levels = int(match.group(2))
    if levels < 1 or raw_book.shape[1] != 4 * levels:
        raise ValueError(f"Expected {4 * levels} book columns, found {raw_book.shape[1]}.")
    if len(raw_book) != len(messages):
        raise ValueError(f"Unaligned files: {len(messages):,} messages and {len(raw_book):,} snapshots.")
    messages["time_ns"] = _time_nanoseconds(messages["time_raw"])
    delta = messages["time_ns"].diff()
    if (delta < 0).any():
        raise ValueError("Message times decrease. Rows were not sorted because alignment must be preserved.")
    messages["timestamp"] = pd.Timestamp(match.group(1)) + pd.to_timedelta(messages["time_ns"], unit="ns")
    messages["event_name"] = messages["event_type"].map(EVENT_NAMES).fillna("Unknown")
    is_execution = messages["event_type"].isin([4, 5])
    price_valid = (messages["price_raw"] > 0) & (messages["price_raw"] < DUMMY_PRICE)
    messages["price"] = (messages["price_raw"] / PRICE_SCALE).where(price_valid & messages["event_type"].ne(7))
    messages["aggressor_sign"] = (-messages["direction"]).where(is_execution & messages["direction"].isin([-1, 1]))
    book_columns = [f"{side}_{field}_{level}" for level in range(1, levels + 1)
                    for side in ("ask", "bid") for field in ("price", "size")]
    raw_book.columns = book_columns
    size_columns = [c for c in book_columns if "_size_" in c]
    price_columns = [c for c in book_columns if "_price_" in c]
    if (raw_book[size_columns] < 0).any().any() or (messages["size"] < 0).any():
        raise ValueError("Negative share sizes require inspection before analysis.")
    book = raw_book.copy()
    for col in price_columns:
        valid = raw_book[col].between(1, DUMMY_PRICE - 1)
        book[col] = (raw_book[col] / PRICE_SCALE).where(valid)
    f = pd.DataFrame(index=messages.index)
    f["timestamp"] = messages["timestamp"]
    f["best_bid"] = book["bid_price_1"]
    f["best_ask"] = book["ask_price_1"]
    # Subtract integers first, avoiding cancellation error in dollar spreads.
    f["spread"] = ((raw_book["ask_price_1"] - raw_book["bid_price_1"]) / PRICE_SCALE).where(
        f["best_bid"].notna() & f["best_ask"].notna())
    quote_valid = (f["best_bid"].notna() & f["best_ask"].notna()
                   & book["bid_size_1"].gt(0) & book["ask_size_1"].gt(0))
    halt_status = messages["price_raw"].where(messages["event_type"].eq(7)).ffill()
    f["trading_halted"] = halt_status.isin([-1, 0])
    f["valid_quote"] = quote_valid & f["spread"].ge(0) & ~f["trading_halted"]
    f["midquote"] = ((f["best_bid"] + f["best_ask"]) / 2).where(f["valid_quote"])
    f["spread_bps"] = (10_000 * f["spread"] / f["midquote"]).where(f["valid_quote"])
    for side in ("bid", "ask"):
        depth = pd.concat([book[f"{side}_size_{k}"].where(book[f"{side}_price_{k}"].notna(), 0)
                           for k in range(1, levels + 1)], axis=1)
        f[f"{side}_depth"] = depth.sum(axis=1)
    total_depth = f["bid_depth"] + f["ask_depth"]
    f["depth_imbalance"] = ((f["bid_depth"] - f["ask_depth"]) / total_depth.where(total_depth > 0))
    touch_depth = book["bid_size_1"] + book["ask_size_1"]
    f["touch_imbalance"] = ((book["bid_size_1"] - book["ask_size_1"]) / touch_depth.where(touch_depth > 0)).where(f["valid_quote"])
    f["microprice"] = ((f["best_ask"] * book["bid_size_1"] + f["best_bid"] * book["ask_size_1"])
                       / touch_depth.where(touch_depth > 0)).where(f["valid_quote"])
    f["duration_seconds"] = (messages["time_ns"].shift(-1) - messages["time_ns"]).fillna(0) / 1e9
    f["interarrival_ms"] = delta / 1e6
    quality = pd.Series({
        "message_rows": len(messages), "snapshot_rows": len(book),
        "extra_message_columns": len(names) - 6,
        "equal_adjacent_timestamps": int(delta.eq(0).sum()),
        "unknown_event_types": int((~messages["event_type"].isin(EVENT_NAMES)).sum()),
        "invalid_order_directions": int((messages["event_type"].isin([1, 2, 3, 4, 5]) & ~messages["direction"].isin([-1, 1])).sum()),
        "invalid_execution_prices": int((is_execution & ~price_valid).sum()),
        "missing_or_invalid_best_quotes": int((~quote_valid).sum()),
        "crossed_quotes": int((quote_valid & f["spread"].lt(0)).sum()),
        "locked_quotes": int((quote_valid & f["spread"].eq(0)).sum()),
        "dummy_price_cells": int(raw_book[price_columns].abs().eq(DUMMY_PRICE).sum().sum()),
        "halt_indicators": int(messages["event_type"].eq(7).sum()),
        "snapshots_during_halt": int(f["trading_halted"].sum()),
    }, name="count", dtype="int64")
    return LobsterData(messages, book, f, quality, message_path, orderbook_path, levels, max_rows is not None)


def event_summary(data):
    """Summarize each message type separately.

    Parameters
    ----------
    data : LobsterData
        Loaded dataset.

    Returns
    -------
    pandas.DataFrame
        Counts and reported share sizes by event type. Sizes are not all trades.
    """
    table = data.messages.groupby("event_type").agg(
        count=("size", "size"), reported_shares=("size", "sum"),
        mean_size=("size", "mean"), median_size=("size", "median"))
    table.insert(0, "event", [EVENT_NAMES.get(i, "Unknown") for i in table.index])
    table["percent_of_events"] = 100 * table["count"] / len(data.messages)
    return table


def metric_summary(data):
    """Compare event averages with elapsed time weighted snapshot averages.

    Parameters
    ----------
    data : LobsterData
        Loaded dataset.

    Returns
    -------
    pandas.DataFrame
        Descriptive statistics plus time weighted means and covered seconds.

    Notes
    -----
    Snapshot k is weighted by time until event k+1. Equal timestamps have zero
    duration. The last snapshot is not extrapolated beyond the final event.
    Each statistic uses its own finite observations. Raw negative spreads are
    retained, while crossed or invalid quotes have no midquote or microprice.
    Halt intervals, including quotation only periods, have no midquote and are
    excluded from time means. Event descriptions retain displayed sizes/spreads.
    """
    cols = ["midquote", "spread", "spread_bps", "bid_depth", "ask_depth",
            "touch_imbalance", "depth_imbalance", "microprice"]
    table = data.features[cols].describe(percentiles=[.01, .25, .5, .75, .99]).T
    table = table.rename(columns={"mean": "event_mean"})
    weights = data.features["duration_seconds"].where(~data.features["trading_halted"], 0)
    for name in cols:
        valid = data.features[name].notna() & weights.gt(0)
        duration = weights[valid].sum()
        table.loc[name, "time_mean"] = (np.average(data.features.loc[valid, name], weights=weights[valid])
                                         if duration > 0 else np.nan)
        table.loc[name, "covered_seconds"] = duration
    return table


def execution_summary(data):
    """Summarize visible and hidden executions, excluding cross messages.

    Parameters
    ----------
    data : LobsterData
        Loaded dataset.

    Returns
    -------
    pandas.DataFrame
        Execution counts, volume, and VWAP with valid price coverage.
    """
    rows = []
    for label, codes in (("Visible", [4]), ("Hidden", [5]), ("Combined", [4, 5])):
        subset = data.messages[data.messages["event_type"].isin(codes)]
        valid = subset["price"].notna()
        volume = subset.loc[valid, "size"].sum()
        vwap = np.average(subset.loc[valid, "price"], weights=subset.loc[valid, "size"]) if volume else np.nan
        rows.append({"execution": label, "count": len(subset), "shares": subset["size"].sum(),
                     "priced_shares": volume, "vwap": vwap})
    return pd.DataFrame(rows).set_index("execution")


def intraday_summary(data, frequency="1min"):
    """Aggregate events and calculate regular clock grid midquote returns.

    Parameters
    ----------
    data : LobsterData
        Loaded dataset.
    frequency : str, optional
        Positive fixed pandas interval such as '1min' or '5min'.

    Returns
    -------
    pandas.DataFrame
        Left labelled event bins and boundary sampled midquotes and log returns.

    Notes
    -----
    Flow columns cover [bin start, bin end). grid_midquote is the final snapshot
    at or before the bin start. Returns end at that boundary. No earlier quote
    is invented for the first grid point. An invalid snapshot remains invalid
    until a subsequent valid snapshot. Returns are descriptive, not a forecast.
    """
    offset = pd.tseries.frequencies.to_offset(frequency)
    if offset.nanos <= 0:
        raise ValueError("frequency must be a positive fixed interval.")
    m = data.messages.set_index("timestamp")
    f = data.features.set_index("timestamp")
    result = m["event_type"].resample(frequency).size().to_frame("event_count")
    for code, label in ((1, "submissions"), (2, "partial_cancellations"), (3, "deletions")):
        result[label] = m["event_type"].eq(code).resample(frequency).sum().reindex(result.index, fill_value=0)
    for codes, label in (([4], "visible_execution_shares"), ([5], "hidden_execution_shares"), ([6], "cross_shares")):
        volume = m["size"].where(m["event_type"].isin(codes), 0)
        result[label] = volume.resample(frequency).sum().reindex(result.index, fill_value=0)
    result["event_mean_spread"] = f["spread"].resample(frequency).mean()
    result["event_mean_depth_imbalance"] = f["depth_imbalance"].resample(frequency).mean()
    # Keep the final original row at a tied timestamp before clock sampling.
    quotes = f["midquote"].loc[~f.index.duplicated(keep="last")]
    result["grid_midquote"] = quotes.reindex(result.index, method="ffill")
    result["log_return_bps"] = 10_000 * np.log(result["grid_midquote"]).diff()
    return result


def _style(figure, title, height=500):
    """Apply a consistent appearance to an exploratory figure.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        Figure to format.
    title : str
        Figure title.
    height : int, optional
        Height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Formatted figure.
    """
    figure.update_layout(template="plotly_white",
                         title=dict(text=title, x=.055, y=.985, xanchor="left", yanchor="top"), height=height,
                         font=dict(family="Arial, sans-serif", size=12),
                         margin=dict(l=65, r=35, t=125, b=55), hovermode="x unified",
                         legend=dict(orientation="h", y=1.04, yanchor="bottom"))
    return figure


def overview_figure(data, max_points=5000):
    """Plot quotes, spread, visible depth, and imbalance over the observed interval.

    Parameters
    ----------
    data : LobsterData
        Loaded dataset.
    max_points : int, optional
        Maximum evenly spaced event rows drawn. Statistics use every loaded row.

    Returns
    -------
    plotly.graph_objects.Figure
        Linked panels with zoom, hover, and a time range slider.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    if max_points < 2:
        raise ValueError("max_points must be at least two.")
    rows = np.unique(np.linspace(0, len(data.features) - 1, min(max_points, len(data.features)), dtype=int))
    f = data.features.iloc[rows]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.07,
                        subplot_titles=("Best quotes and midquote", "Quoted spread", "Displayed depth within loaded levels", "Displayed depth imbalance"))
    for name, label, color in (("best_bid", "Bid", BID_COLOR), ("best_ask", "Ask", ASK_COLOR), ("midquote", "Midquote", "#243b63")):
        fig.add_trace(go.Scatter(x=f["timestamp"], y=f[name], name=label, line=dict(color=color, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=f["timestamp"], y=f["spread"], name="Spread", line_color="#7957a6"), row=2, col=1)
    for side, color in (("bid", BID_COLOR), ("ask", ASK_COLOR)):
        fig.add_trace(go.Scatter(x=f["timestamp"], y=f[f"{side}_depth"], name=f"{side.title()} depth", line_color=color), row=3, col=1)
    fig.add_trace(go.Scatter(x=f["timestamp"], y=f["depth_imbalance"], name="Imbalance", line_color="#243b63"), row=4, col=1)
    for row, title in enumerate(("USD", "USD", "Shares", "(Bid − Ask) / Total"), 1):
        fig.update_yaxes(title_text=title, row=row, col=1)
    fig.update_yaxes(range=[-1, 1], row=4, col=1)
    fig.update_xaxes(title_text="Exchange clock", rangeslider_visible=True, rangeslider_thickness=.045, row=4, col=1)
    return _style(fig, f"{data.message_path.name.split('_')[0]} · {len(f):,} drawn snapshots of {len(data.messages):,} loaded", 1000)


def activity_figure(data, frequency="1min"):
    """Plot event counts, execution volume, and clock grid returns.

    Parameters
    ----------
    data : LobsterData
        Loaded dataset.
    frequency : str, optional
        Fixed aggregation interval.

    Returns
    -------
    plotly.graph_objects.Figure
        Three linked panels. Cross messages are excluded from execution volume.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    table = intraday_summary(data, frequency)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.09,
                        subplot_titles=("All event types", "Execution volume, types 4 and 5", "Midquote log returns at clock boundaries"))
    fig.add_trace(go.Bar(x=table.index, y=table["event_count"], name="Events", marker_color="#243b63"), row=1, col=1)
    for label, col, color in (("Visible", "visible_execution_shares", BID_COLOR), ("Hidden", "hidden_execution_shares", ASK_COLOR)):
        fig.add_trace(go.Bar(x=table.index, y=table[col], name=label, marker_color=color), row=2, col=1)
    fig.add_trace(go.Scatter(x=table.index, y=table["log_return_bps"], name="Log return", line_color="#7957a6"), row=3, col=1)
    fig.update_layout(barmode="stack")
    for row, title in enumerate(("Events", "Shares", "Basis points"), 1):
        fig.update_yaxes(title_text=title, row=row, col=1)
    fig.update_xaxes(title_text="Exchange clock · first and last flow bins may be partial", row=3, col=1)
    return _style(fig, f"Activity and returns · {frequency} intervals", 850)


def distributions_figure(data):
    """Plot distributions from every loaded observation.

    Parameters
    ----------
    data : LobsterData
        Loaded dataset.

    Returns
    -------
    plotly.graph_objects.Figure
        Histograms of spreads, imbalance, execution sizes, and positive gaps.

    Notes
    -----
    Histograms are aggregated in Python to keep browser payloads small.
    Size and positive time gap panels use logarithms to expose their tails.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    f, m = data.features, data.messages
    sizes = m.loc[m["event_type"].isin([4, 5]) & m["size"].gt(0), "size"]
    gaps = f.loc[f["interarrival_ms"].gt(0), "interarrival_ms"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Spread across event rows", "Depth imbalance across event rows", "Positive execution sizes", "Positive gaps between consecutive events"))
    series = [(f["spread"], "Spread (USD)"), (f["depth_imbalance"], "Imbalance"),
              (np.log10(sizes), "log10(size in shares)"), (np.log10(gaps), "log10(gap in milliseconds)")]
    for i, (values, label) in enumerate(series):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        counts, edges = np.histogram(values, bins=60)
        row, col = i // 2 + 1, i % 2 + 1
        fig.add_trace(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges),
                             marker_color=(BID_COLOR if i % 2 == 0 else ASK_COLOR), showlegend=False,
                             customdata=np.column_stack((edges[:-1], edges[1:])),
                             hovertemplate="Bin %{customdata[0]:.4g} to %{customdata[1]:.4g}<br>Count %{y:,}<extra></extra>"), row=row, col=col)
        fig.update_xaxes(title_text=label, row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)
    ties = int(f["interarrival_ms"].eq(0).sum())
    return _style(fig, f"Distributions · {ties:,} zero time gaps excluded from the log gap panel", 750)


def depth_profile_figure(data):
    """Compare event mean displayed depth at each occupied price level.

    Parameters
    ----------
    data : LobsterData
        Loaded dataset.

    Returns
    -------
    plotly.graph_objects.Figure
        Mean depth by side and level, using every loaded snapshot.
    """
    import plotly.graph_objects as go
    fig = go.Figure()
    for side, color in (("bid", BID_COLOR), ("ask", ASK_COLOR)):
        means = [data.book[f"{side}_size_{k}"].where(data.book[f"{side}_price_{k}"].notna(), 0).mean()
                 for k in range(1, data.levels + 1)]
        fig.add_bar(x=list(range(1, data.levels + 1)), y=means, name=side.title(), marker_color=color)
    fig.update_layout(barmode="group", xaxis_title="Occupied price level (not tick distance)", yaxis_title="Event mean displayed shares")
    return _style(fig, "Average visible depth by level")


def snapshot_figure(data, row=0):
    """Inspect the visible book immediately after one selected message.

    Parameters
    ----------
    data : LobsterData
        Loaded dataset.
    row : int, optional
        Zero based event position.

    Returns
    -------
    plotly.graph_objects.Figure
        Displayed sizes at actual dollar prices for both sides.
    """
    import plotly.graph_objects as go
    if not isinstance(row, (int, np.integer)) or not 0 <= row < len(data.messages):
        raise ValueError(f"row must be between 0 and {len(data.messages) - 1}.")
    book = data.book.iloc[row]
    message = data.messages.iloc[row]
    fig = go.Figure()
    for side, color in (("bid", BID_COLOR), ("ask", ASK_COLOR)):
        prices = [book[f"{side}_price_{k}"] for k in range(1, data.levels + 1)]
        sizes = [book[f"{side}_size_{k}"] for k in range(1, data.levels + 1)]
        fig.add_bar(x=prices, y=sizes, name=side.title(), marker_color=color,
                    customdata=list(range(1, data.levels + 1)),
                    hovertemplate="Price $%{x:.4f}<br>Shares %{y:,}<br>Level %{customdata}<extra>%{fullData.name}</extra>")
    fig.update_layout(xaxis_title="Price (USD)", yaxis_title="Displayed shares", barmode="overlay")
    return _style(fig, f"After row {row:,} · {message['timestamp']} · {message['event_name']}")


def print_summary(data):
    """Print compact source, quality, event, execution, and quote summaries.

    Parameters
    ----------
    data : LobsterData
        Loaded dataset.

    Returns
    -------
    None
    """
    m = data.messages
    print(f"\n{data.message_path.name}\n{len(m):,} events · {data.levels} levels per side")
    print(f"Observed interval: {m['timestamp'].iloc[0]} to {m['timestamp'].iloc[-1]}")
    if data.preview:
        print("PARTIAL PREVIEW: a row limit was requested. File tails were not audited.")
    for title, table in (("Data checks", data.quality), ("Event types", event_summary(data)),
                         ("Executions (types 4 and 5 only)", execution_summary(data)),
                         ("Snapshot metrics", metric_summary(data))):
        print(f"\n{title}\n{table.to_string()}")


def dashboard_html(data):
    """Build an offline dashboard document entirely in memory.

    Parameters
    ----------
    data : LobsterData
        Loaded dataset.

    Returns
    -------
    str
        HTML including Plotly itself, with no external network dependencies.
    """
    import plotly.io as pio
    figures = [overview_figure(data), activity_figure(data), distributions_figure(data), depth_profile_figure(data)]
    charts = "".join(pio.to_html(fig, full_html=False, include_plotlyjs=(i == 0)) for i, fig in enumerate(figures))
    snapshot = pio.to_html(snapshot_figure(data), full_html=False, include_plotlyjs=False, div_id="snapshot")
    preview = "<p class='warning'>Partial preview. File tails were not audited.</p>" if data.preview else ""
    tables = "".join(f"<h2>{title}</h2><div class='table'>{table.to_html()}</div>" for title, table in (
        ("Data checks", data.quality.to_frame()), ("Event types", event_summary(data)),
        ("Executions", execution_summary(data)), ("Snapshot statistics", metric_summary(data))))
    return """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LOBSTER exploration</title><style>
body{max-width:1250px;margin:35px auto;padding:0 24px;font:15px/1.6 Arial,sans-serif;color:#24334b;background:#f8fafc}
h1{font-size:32px;margin-bottom:6px}h2{margin-top:32px}p{max-width:950px}.table{overflow:auto;background:white;padding:12px}
table{border-collapse:collapse;font-size:13px}th,td{padding:7px 12px;border:1px solid #dce3ec;text-align:right}
.warning{color:#a33}input[type=range]{width:75%;vertical-align:middle}a{color:#087f8c}
</style></head><body><h1>LOBSTER exploration</h1>""" + f"""
<p>{escape(data.message_path.name)}<br>{len(data.messages):,} events · {data.levels} levels per side<br>
{data.messages['timestamp'].iloc[0]} to {data.messages['timestamp'].iloc[-1]} · exchange clock</p>{preview}
<p>Drag to zoom, double click to reset, and click legend entries to hide traces. Overview lines draw at most 5,000 evenly spaced event rows.
All tables, histograms, and activity totals use every loaded row. Selected snapshots use the exact original row.</p>
{charts}<h2>Explore individual snapshots</h2>
<label for="row">Event row </label><input id="row" type="range" min="0" max="{len(data.messages)-1}" value="0">
<input id="rowNumber" aria-label="Event row number" type="number" min="0" max="{len(data.messages)-1}" value="0" style="width:95px">
<p id="status">The snapshot is the visible book after the selected event.</p>{snapshot}{tables}
<h2>Interpretation</h2><p>Prices are dollars and sizes are shares. Depth covers the loaded levels and excludes hidden liquidity.
Execution counts are order executions, which need not equal economic trade counts. Cross and auction messages stay separate from execution totals.
Snapshot means weight event rows equally. Time means weight each snapshot until the next event and stop at the final timestamp.
Known halt and quotation only intervals have no midquote and are excluded from time means.
Clock grid returns use the final available snapshot at each boundary. First and last activity bins may be partial.
The observed interval can include data before 09:30 and after 16:00. Historical prices are not adjusted for later stock splits.
Extra message columns are preserved without assigning them an undocumented meaning.</p>
<p>References: <a href="https://php.lobsterdata.com/info/DataStructure.php">LOBSTER output specification</a> ·
<a href="https://php.lobsterdata.com/info/help_faq_general.php">LOBSTER FAQ</a> ·
<a href="https://php.lobsterdata.com/info/help_reconstruction.php">Reconstruction notes</a></p>
""" + """<script>
let requestId=0;
const rowSlider=document.getElementById('row'), rowNumber=document.getElementById('rowNumber');
async function updateSnapshot(value){
 const row=Number(value), current=++requestId;
 rowSlider.value=row; rowNumber.value=row;
 try {
  const response=await fetch('/snapshot?row='+row);
  if(!response.ok) throw new Error(await response.text());
  const fig=await response.json();
  if(current!==requestId) return;
  await Plotly.react('snapshot',fig.data,fig.layout,{responsive:true});
  document.getElementById('status').textContent='Visible book after event row '+row.toLocaleString();
 } catch(error){document.getElementById('status').textContent=error.message;}
}
rowSlider.addEventListener('change',event=>updateSnapshot(event.target.value));
rowSlider.addEventListener('input',event=>{rowNumber.value=event.target.value;});
rowNumber.addEventListener('change',event=>updateSnapshot(event.target.value));
</script></body></html>"""


def serve_dashboard(data, port=0):
    """Serve the dashboard from memory until interrupted.

    Parameters
    ----------
    data : LobsterData
        Loaded dataset.
    port : int, optional
        Local port. Zero asks the operating system for an available port.

    Returns
    -------
    None

    Notes
    -----
    Only the generated dashboard and snapshot responses are served. The server
    binds to 127.0.0.1 and does not expose directories or write browser files.
    """
    import plotly.io as pio
    document = dashboard_html(data).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlparse(self.path)
            try:
                if request.path == "/":
                    body, content_type = document, "text/html; charset=utf-8"
                elif request.path == "/snapshot":
                    row = int(parse_qs(request.query).get("row", ["0"])[0])
                    body = pio.to_json(snapshot_figure(data, row)).encode("utf-8")
                    content_type = "application/json"
                else:
                    self.send_error(404)
                    return
            except (ValueError, IndexError) as exc:
                self.send_error(400, str(exc))
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"\nOpen http://127.0.0.1:{server.server_port} in your browser. Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard stopped.")


def main():
    """Load the selected dataset and print summaries or start the dashboard.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent / "data", help="Folder containing extracted CSV pairs")
    parser.add_argument("--dataset", type=int, default=0, help="Zero based index in the sorted dataset list")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional first row preview limit")
    parser.add_argument("--port", type=int, default=0, help="Dashboard port, default chooses an available port")
    parser.add_argument("--summary-only", action="store_true", help="Print summary tables and exit")
    args = parser.parse_args()
    paths = discover_datasets(args.directory)
    if not paths:
        parser.error("No extracted message and orderbook CSV pairs found.")
    if not 0 <= args.dataset < len(paths):
        parser.error(f"Choose --dataset between 0 and {len(paths)-1}.")
    print("Available datasets:")
    for i, path in enumerate(paths):
        print(f"  {i}: {path.name}")
    data = load_lobster(paths[args.dataset], max_rows=args.max_rows)
    print_summary(data)
    if not args.summary_only:
        serve_dashboard(data, args.port)


if __name__ == "__main__":
    main()
