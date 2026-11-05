"""Deterministic research calculations for locally loaded LOBSTER observations.

Clock samples preserve invalid snapshots and original ordering at tied times.
The functions return data rather than figures and never write files.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN

import numpy as np
import pandas as pd


MAX_GRID_POINTS = 100_000
MAX_MESSAGE_SAMPLES = 500_000
NANOSECONDS = 1_000_000_000
CONSTANT_RELATIVE_TOLERANCE = 64 * np.finfo(float).eps
MAX_DIMENSION_ENDPOINTS = 5000
# Full-file benchmarks on the local machine put representative requests near
# this budget in the tens of seconds. This is not a wall-clock timeout.
MAX_DIMENSION_SVD_WORK = 300_000_000_000
MAX_DIMENSION_MEMORY_BYTES = 1024 ** 3
DIMENSION_FEATURE_CHUNK_ROWS = 4096
DIMENSION_FEATURE_CHUNK_BYTES = 8 * 1024 ** 2


def _seconds_ns(value, name, *, allow_zero=True):
    """Convert a finite seconds value to the nearest integer nanosecond.

    Parameters
    ----------
    value : float or str
        Seconds represented numerically or as a decimal string.
    name : str
        Parameter name to include in validation errors.
    allow_zero : bool, optional
        Whether zero is an admissible value.

    Returns
    -------
    int
        Integer nanoseconds, using decimal rounding with ties to even.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number of seconds.")
    try:
        seconds = Decimal(str(value))
        if not seconds.is_finite():
            raise ValueError
        ns = int((seconds * NANOSECONDS).to_integral_value(rounding=ROUND_HALF_EVEN))
    except (InvalidOperation, ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number of seconds.") from exc
    if ns < 0 or (ns == 0 and not allow_zero):
        requirement = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {requirement} at nanosecond resolution.")
    return ns


def _level_count(data, levels):
    """Validate the requested number of occupied levels on each side.

    Parameters
    ----------
    data : LobsterData
        Loaded observations and their available depth.
    levels : int
        Requested number of levels per side.

    Returns
    -------
    int
        Validated level count.
    """
    if isinstance(levels, (bool, np.bool_)) or not isinstance(levels, (int, np.integer)):
        raise ValueError("levels must be an integer.")
    if not 1 <= levels <= data.levels:
        raise ValueError(f"levels must be between 1 and {data.levels}.")
    return int(levels)


def _quote_validity(data):
    """Identify event snapshots with usable, strictly positive spreads.

    Parameters
    ----------
    data : LobsterData
        Loaded observations with quote validity and halt indicators.

    Returns
    -------
    numpy.ndarray
        Boolean validity in original event order. Locked quotes are excluded.
    """
    f = data.features
    midquote = f["midquote"].to_numpy(dtype=float)
    return (f["valid_quote"].to_numpy(dtype=bool)
            & ~f["trading_halted"].to_numpy(dtype=bool)
            & f["spread"].gt(0).to_numpy(dtype=bool)
            & np.isfinite(midquote) & (midquote > 0))


def sample_grid(data, start_seconds, end_seconds, step_seconds, levels,
                max_age_seconds=60):
    """Select snapshots on a regular exchange clock grid without looking ahead.

    Parameters
    ----------
    data : LobsterData
        Aligned messages, cleaned snapshots, and derived quote measurements.
    start_seconds, end_seconds : float
        Requested interval in exchange clock seconds after midnight. Its
        intersection with the loaded observations is used without extrapolation.
    step_seconds : float
        Positive interval between grid points, anchored to midnight.
    levels : int
        Occupied levels per side used for depth and imbalance.
    max_age_seconds : float or None, optional
        Maximum age of the source snapshot. None disables the age restriction.

    Returns
    -------
    pandas.DataFrame
        Timestamp indexed samples with original row position, integer clock
        timestamp, midquote, displayed imbalance, depth features, quote age,
        exclusion flags, and combined validity. Invalid rows remain in the grid.

    Raises
    ------
    ValueError
        If parameters are invalid, the interval has no grid points, or the grid
        exceeds the explicit analytical size limit.

    Notes
    -----
    Each sample uses the final original event at or before its clock time.
    Invalid snapshots are never replaced with an earlier valid quote. Halts,
    locked quotes, crossed quotes, and invalid best quotes are excluded.
    A missing deeper price with zero size contributes zero displayed shares.
    A missing price with nonzero size invalidates the sample. The cleaned loader
    cannot distinguish a malformed missing price with zero size from a documented
    dummy price with zero size. Grid reduction is never applied silently.
    """
    levels = _level_count(data, levels)
    start_ns = _seconds_ns(start_seconds, "start_seconds")
    end_ns = _seconds_ns(end_seconds, "end_seconds")
    step_ns = _seconds_ns(step_seconds, "step_seconds", allow_zero=False)
    age_ns = (None if max_age_seconds is None
              else _seconds_ns(max_age_seconds, "max_age_seconds"))
    if start_ns > end_ns:
        raise ValueError("The start time must not follow the end time.")
    times = data.messages["time_ns"].to_numpy(dtype=np.int64)
    if len(times) == 0 or np.any(np.diff(times) < 0):
        raise ValueError("Research sampling requires nonempty, ordered message times.")
    lower = max(start_ns, int(times[0]))
    upper = min(end_ns, int(times[-1]))
    first = ((lower + step_ns - 1) // step_ns) * step_ns
    count = 0 if first > upper else (upper - first) // step_ns + 1
    if count == 0:
        raise ValueError("This interval contains no clock grid points. Widen it or use a shorter sampling interval.")
    if count > MAX_GRID_POINTS:
        raise ValueError(f"The interval creates {count:,} samples. Increase the sampling interval to use at most {MAX_GRID_POINTS:,} samples.")
    grid_ns = first + np.arange(count, dtype=np.int64) * step_ns
    rows = np.searchsorted(times, grid_ns, side="right") - 1
    day = pd.Timestamp(data.messages["timestamp"].iloc[0]).normalize()
    frame = pd.DataFrame(index=pd.DatetimeIndex(day + pd.to_timedelta(grid_ns, unit="ns"), name="timestamp"))
    frame["row"] = rows
    frame["timestamp_ns"] = grid_ns
    frame["quote_age_seconds"] = (grid_ns - times[rows]) / NANOSECONDS
    frame["invalid_quote"] = ~_quote_validity(data)[rows]
    frame["stale"] = False if age_ns is None else (grid_ns - times[rows] > age_ns)
    invalid_depth = np.zeros(count, dtype=bool)
    totals = {}
    for side in ("bid", "ask"):
        totals[side] = np.zeros(count, dtype=float)
        for level in range(1, levels + 1):
            size_name = f"{side}_size_{level}"
            sizes = data.book[size_name].to_numpy(dtype=float)[rows]
            prices = data.book[f"{side}_price_{level}"].to_numpy(dtype=float)[rows]
            price_available = np.isfinite(prices) & (prices > 0)
            invalid_depth |= ~np.isfinite(sizes) | (sizes < 0) | (~price_available & (sizes != 0))
            sizes = np.where(price_available, sizes, 0.0)
            frame[size_name] = sizes
            totals[side] += sizes
    total = totals["bid"] + totals["ask"]
    invalid_depth |= ~np.isfinite(total) | (total <= 0)
    frame["invalid_depth"] = invalid_depth
    frame["valid"] = ~(frame["invalid_quote"] | frame["invalid_depth"] | frame["stale"])
    frame["midquote"] = data.features["midquote"].to_numpy(dtype=float)[rows]
    frame.loc[frame["invalid_quote"], "midquote"] = np.nan
    frame["imbalance"] = np.divide(totals["bid"] - totals["ask"], total,
                                   out=np.full(count, np.nan), where=total > 0)
    frame.loc[frame["invalid_depth"], "imbalance"] = np.nan
    frame.attrs.update({"step_ns": step_ns, "start_ns": start_ns, "end_ns": end_ns,
                        "max_age_ns": age_ns, "levels": levels})
    return frame


def _partition(frame, train_fraction):
    """Split the complete regular grid before excluding invalid samples.

    Parameters
    ----------
    frame : pandas.DataFrame
        Complete regular clock grid returned by sample_grid.
    train_fraction : float
        Fraction strictly between zero and one allocated before the split.

    Returns
    -------
    tuple of int and int
        Split row position and integer nanoseconds after midnight.
    """
    if isinstance(train_fraction, (bool, np.bool_)):
        raise ValueError("train_fraction must be strictly between zero and one.")
    try:
        fraction = float(train_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("train_fraction must be strictly between zero and one.") from exc
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("train_fraction must be strictly between zero and one.")
    position = int(np.floor(len(frame) * fraction))
    if position < 1 or position >= len(frame):
        raise ValueError("The selected interval needs more grid points for a chronological split.")
    return position, int(frame["timestamp_ns"].iloc[position])


def _parameters(frame, split_position, train_fraction):
    """Describe the shared sampling and chronological partition policy.

    Parameters
    ----------
    frame : pandas.DataFrame
        Clock samples and their integer sampling metadata.
    split_position : int
        First grid position in the evaluation partition.
    train_fraction : float
        Requested chronological training fraction.

    Returns
    -------
    dict
        JSON compatible research parameters.
    """
    attrs = frame.attrs
    return {
        "start_seconds": attrs["start_ns"] / NANOSECONDS,
        "end_seconds": attrs["end_ns"] / NANOSECONDS,
        "first_grid_time": frame.index[0].isoformat(),
        "last_grid_time": frame.index[-1].isoformat(),
        "step_seconds": attrs["step_ns"] / NANOSECONDS,
        "levels": attrs["levels"],
        "max_age_seconds": (None if attrs["max_age_ns"] is None
                            else attrs["max_age_ns"] / NANOSECONDS),
        "train_fraction": float(train_fraction),
        "split_time": frame.index[split_position].isoformat(),
        "split_seconds": int(frame["timestamp_ns"].iloc[split_position]) / NANOSECONDS,
        "sampling": "Midnight anchored clock grid. Final original event at or before each grid time.",
        "quote_policy": "Exclude halts, missing best quotes, crossed quotes, and locked quotes.",
        "depth_policy": "Missing price with zero size is zero depth. Missing price with nonzero size invalidates the sample.",
    }


def _counts(frame):
    """Count grid observations and independently overlapping exclusion flags.

    Parameters
    ----------
    frame : pandas.DataFrame
        Complete clock samples.

    Returns
    -------
    dict
        Counts of samples, validity, and exclusions.
    """
    return {"grid_points": len(frame), "valid_grid_points": int(frame["valid"].sum()),
            "invalid_grid_points": int((~frame["valid"]).sum()),
            "invalid_quote_grid_points": int(frame["invalid_quote"].sum()),
            "invalid_depth_grid_points": int(frame["invalid_depth"].sum()),
            "stale_grid_points": int(frame["stale"].sum())}


def run_pca(data, start_seconds, end_seconds, step_seconds=5, levels=10,
            transform="log", standardize=True, train_fraction=.7,
            max_age_seconds=60):
    """Fit depth PCA on training observations and evaluate reconstruction later.

    Parameters
    ----------
    data : LobsterData
        Loaded, aligned LOBSTER observations.
    start_seconds, end_seconds : float
        Requested exchange clock interval in seconds after midnight.
    step_seconds : float, optional
        Regular clock sampling interval.
    levels : int, optional
        Occupied levels per side included as separate depth features.
    transform : {'log', 'raw'}, optional
        Apply log1p to displayed shares or retain share units.
    standardize : bool, optional
        Divide each retained, centered feature by its training sample standard
        deviation. Centering always uses training means.
    train_fraction : float, optional
        Chronological fraction of the complete clock grid used for training.
    max_age_seconds : float or None, optional
        Maximum source snapshot age, or None to disable the restriction.

    Returns
    -------
    dict
        JSON compatible parameters, counts, metrics, training variance fractions,
        component axes, loadings, feature names, and evaluation reconstruction
        errors for zero through all fitted components. Component rows are axes
        in retained feature order. Loading rows are features and columns are
        components, with axes scaled by training eigenvalue square roots.

    Notes
    -----
    Training constant columns are discarded using a relative range tolerance of
    64 machine epsilons. Evaluation data never affect scaling or fitted axes.
    Component signs are fixed by making each axis's largest absolute entry
    positive. PCA describes linear variance structure, not intrinsic dimension.
    Reconstruction RMSE uses the transformed, optionally standardized units.
    """
    if transform not in ("log", "raw"):
        raise ValueError("transform must be 'log' or 'raw'.")
    if not isinstance(standardize, (bool, np.bool_)):
        raise ValueError("standardize must be True or False.")
    grid = sample_grid(data, start_seconds, end_seconds, step_seconds, levels, max_age_seconds)
    split_position, split_ns = _partition(grid, train_fraction)
    train_mask = grid["valid"].to_numpy() & (grid["timestamp_ns"].to_numpy() < split_ns)
    test_mask = grid["valid"].to_numpy() & (grid["timestamp_ns"].to_numpy() >= split_ns)
    train_count, test_count = int(train_mask.sum()), int(test_mask.sum())
    if train_count < 3 or test_count < 1:
        raise ValueError("PCA needs at least three valid training samples and one valid evaluation sample. Widen the interval or shorten sampling.")
    all_names = [f"{side}_size_{level}" for side in ("bid", "ask")
                 for level in range(1, int(levels) + 1)]
    values = grid[all_names].to_numpy(dtype=float)
    if transform == "log":
        values = np.log1p(values)
    train = values[train_mask]
    magnitude = np.maximum(1.0, np.max(np.abs(train), axis=0))
    retained = np.ptp(train, axis=0) > CONSTANT_RELATIVE_TOLERANCE * magnitude
    if not np.any(retained):
        raise ValueError("Every selected depth feature is constant in the training interval. Choose another interval or feature set.")
    names = [name for name, keep in zip(all_names, retained) if keep]
    dropped = [name for name, keep in zip(all_names, retained) if not keep]
    train = train[:, retained]
    test = values[test_mask][:, retained]
    means = np.mean(train, axis=0)
    scales = np.std(train, axis=0, ddof=1) if standardize else np.ones(len(names))
    z_train = (train - means) / scales
    z_test = (test - means) / scales
    _, singular_values, components = np.linalg.svd(z_train, full_matrices=False)
    component_count = min(len(names), train_count - 1)
    singular_values, components = singular_values[:component_count], components[:component_count]
    for component in components:
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            component *= -1
    eigenvalues = np.square(singular_values) / (train_count - 1)
    explained = eigenvalues / eigenvalues.sum()
    cumulative = np.minimum(np.cumsum(explained), 1.0)
    scores = z_test @ components.T
    baseline_sse = float(np.square(z_test).sum())
    captured_sse = np.cumsum(np.square(scores).sum(axis=0))
    residual_sse = np.maximum(baseline_sse - captured_sse, 0.0)
    reconstruction = np.sqrt(np.r_[baseline_sse, residual_sse] / z_test.size)
    k90 = min(int(np.searchsorted(cumulative, .90)) + 1, component_count)
    k95 = min(int(np.searchsorted(cumulative, .95)) + 1, component_count)
    params = _parameters(grid, split_position, train_fraction)
    params.update({"transform": transform, "standardize": bool(standardize),
                   "constant_relative_tolerance": CONSTANT_RELATIVE_TOLERANCE,
                   "depth_feature_order": "Bid levels in order, then ask levels in order.",
                   "reconstruction_units": "standardized features" if standardize else ("log1p shares" if transform == "log" else "shares")})
    counts = _counts(grid)
    counts.update({"train_samples": train_count, "test_samples": test_count,
                   "input_features": len(all_names), "retained_features": len(names),
                   "dropped_features": len(dropped), "fitted_components": component_count})
    return {
        "method": "Depth PCA", "parameters": params, "counts": counts,
        "metrics": {"components_for_90_percent": k90, "components_for_95_percent": k95,
                    "first_component_variance": float(explained[0]),
                    "test_baseline_rmse": float(reconstruction[0]),
                    "test_rmse_at_90_percent": float(reconstruction[k90]),
                    "test_rmse_at_95_percent": float(reconstruction[k95]),
                    "test_full_reconstruction_rmse": float(reconstruction[-1])},
        "explained_variance": explained.tolist(), "cumulative_variance": cumulative.tolist(),
        "components": components.tolist(),
        "loadings": (components.T * np.sqrt(eigenvalues)).tolist(),
        "feature_names": names, "dropped_feature_names": dropped,
        "training_mean": means.tolist(), "training_scale": scales.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "reconstruction_components": list(range(component_count + 1)),
        "test_reconstruction_rmse": reconstruction.tolist(),
        "notes": ["PCA measures linear variance structure and does not estimate intrinsic dimension.",
                  "Chronological evaluation within this file does not establish stability across days.",
                  "Observations remain temporally dependent. No significance claims are made.",
                  "Unavailable deeper prices with zero size are treated as zero displayed depth. The cleaned snapshot cannot identify every malformed zero size price."]
    }


def _overlapping_intervals(starts, ends, left, right):
    """Detect positive duration invalid intervals intersecting each horizon.

    Parameters
    ----------
    starts, ends : numpy.ndarray
        Ordered interval starts and ends. Each end must exceed its start.
    left, right : numpy.ndarray
        Ordered or unordered query horizon starts and ends.

    Returns
    -------
    numpy.ndarray
        Whether each open horizon interior intersects an invalid interval.
        Endpoint validity is checked separately by the caller.
    """
    return ((np.searchsorted(starts, right, side="left")
             - np.searchsorted(ends, left, side="right")) > 0)


def rolling_realised_volatility(data, start_seconds, end_seconds, window_seconds=300,
                                step_seconds=1, max_age_seconds=60):
    """Calculate rolling realised volatility from sampled midquote log returns.

    Parameters
    ----------
    data : LobsterData
        Aligned messages, snapshots and quote measurements.
    start_seconds, end_seconds : float
        Selected interval in exchange clock seconds after midnight.
    window_seconds : float, optional
        Length of each trailing window. Must be a positive integer multiple
        of step_seconds. A full window of valid returns is required.
    step_seconds : float, optional
        Interval between sampled midquotes, anchored to midnight.
    max_age_seconds : float or None, optional
        Maximum source quote age. None disables the age restriction.

    Returns
    -------
    pandas.DataFrame
        Timestamp indexed midquotes, log returns, return validity, valid return
        counts per trailing window and realised_volatility_bps. Missing values
        denote incomplete or invalid windows. Attributes retain the settings.

    Notes
    -----
    Volatility is 10000 times the square root of the sum of squared log returns.
    Returns are not demeaned and volatility is not annualised. No quotes before
    the selected start or after an estimate are used. Equal timestamps use the
    last original message. Invalid, locked, crossed, halted or stale quotes
    invalidate returns, including positive duration gaps between valid sampled
    endpoints. Displayed volumes and the selected depth do not enter the result.
    """
    step_ns = _seconds_ns(step_seconds, "step_seconds", allow_zero=False)
    window_ns = _seconds_ns(window_seconds, "window_seconds", allow_zero=False)
    if window_ns % step_ns:
        raise ValueError("The volatility window must be an integer multiple of the sampling interval.")
    window_returns = window_ns // step_ns
    grid = sample_grid(data, start_seconds, end_seconds, step_seconds, 1, max_age_seconds)
    times = data.messages["time_ns"].to_numpy(dtype=np.int64)
    source_rows = grid["row"].to_numpy(dtype=np.int64)
    clock_ns = grid["timestamp_ns"].to_numpy(dtype=np.int64)
    valid_quote = (~grid["invalid_quote"].to_numpy(dtype=bool)
                   & ~grid["stale"].to_numpy(dtype=bool)
                   & (times[source_rows] >= grid.attrs["start_ns"]))
    pair_valid = valid_quote[:-1] & valid_quote[1:]
    invalid_intervals = ~_quote_validity(data)[:-1] & (np.diff(times) > 0)
    pair_valid &= ~_overlapping_intervals(times[:-1][invalid_intervals], times[1:][invalid_intervals],
                                         clock_ns[:-1], clock_ns[1:])
    age_ns = grid.attrs["max_age_ns"]
    if age_ns is not None:
        stale_starts = times[:-1] + age_ns
        stale_intervals = stale_starts < times[1:]
        pair_valid &= ~_overlapping_intervals(stale_starts[stale_intervals], times[1:][stale_intervals],
                                             clock_ns[:-1], clock_ns[1:])
    midquotes = grid["midquote"].to_numpy(dtype=float)
    returns = np.full(len(grid), np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        changes = np.diff(np.log(midquotes))
    pair_valid &= np.isfinite(changes)
    returns[1:] = np.where(pair_valid, changes, np.nan)
    squared = pd.Series(np.square(returns), index=grid.index)
    rolling_sum = squared.rolling(window_returns, min_periods=window_returns).sum()
    result = pd.DataFrame({"midquote": np.where(valid_quote, midquotes, np.nan),
                           "log_return": returns, "valid_return": np.isfinite(returns),
                           "valid_returns_in_window": squared.rolling(window_returns, min_periods=1).count().fillna(0).astype(int),
                           "realised_volatility_bps": 10_000 * np.sqrt(rolling_sum.clip(lower=0))}, index=grid.index)
    result.attrs.update({"window_seconds": window_ns / NANOSECONDS,
                         "step_seconds": step_ns / NANOSECONDS,
                         "max_age_seconds": None if age_ns is None else age_ns / NANOSECONDS,
                         "returns_per_window": int(window_returns), "annualised": False,
                         "formula": "10000 * sqrt(sum(log(mid_t / mid_previous)**2))"})
    return result


def _binned_response(x, y, edges):
    """Aggregate response means, medians, and counts within fixed bins.

    Parameters
    ----------
    x, y : numpy.ndarray
        Finite imbalance observations and corresponding response values.
    edges : numpy.ndarray
        Fixed increasing bin edges spanning minus one through plus one.

    Returns
    -------
    tuple of list
        Means, medians, and integer counts. Empty summaries are None.
    """
    assignments = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    means, medians, counts = [], [], []
    for number in range(len(edges) - 1):
        values = y[assignments == number]
        counts.append(int(len(values)))
        means.append(float(np.mean(values)) if len(values) else None)
        medians.append(float(np.median(values)) if len(values) else None)
    return means, medians, counts


def run_response(data, start_seconds, end_seconds, step_seconds=5, levels=10,
                 horizon_seconds=30, train_fraction=.7, max_age_seconds=60,
                 bins=10):
    """Compare future midquote response with current displayed imbalance.

    Parameters
    ----------
    data : LobsterData
        Loaded, aligned LOBSTER observations.
    start_seconds, end_seconds : float
        Requested exchange clock interval in seconds after midnight.
    step_seconds : float, optional
        Regular clock sampling interval.
    levels : int, optional
        Occupied levels per side used in displayed imbalance. One uses the touch.
    horizon_seconds : float, optional
        Positive response horizon, an integer multiple of the sampling interval.
    train_fraction : float, optional
        Chronological fraction of the complete grid allocated before the split.
    max_age_seconds : float or None, optional
        Maximum source snapshot age. Horizons crossing an older snapshot interval
        are also excluded. None disables the age restriction.
    bins : int, optional
        Number of fixed equal width imbalance bins, from two through fifty.

    Returns
    -------
    dict
        JSON compatible parameters, counts, descriptive response curves, fixed
        bin edges, and an ordinary least squares line fitted on training data.
        Responses, intercept, and prediction errors use basis points. The slope
        is basis points per unit imbalance. No confidence intervals or p values
        are calculated.

    Notes
    -----
    The target is 10000 times log of the future midquote divided by the starting
    midquote. Training targets must end strictly before the split. Evaluation
    observations start at or after the split and end within the selected grid.
    Horizons crossing a positive duration invalid quote or stale interval are
    excluded even when both sampled endpoints are valid. Intermediate snapshots
    with zero duration at a tied timestamp do not define an invalid interval.
    Training and evaluation targets can overlap within their own partitions.
    Dependence precludes interpreting observation counts as independent samples.
    """
    if (isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer))
            or not 2 <= bins <= 50):
        raise ValueError("bins must be an integer between 2 and 50.")
    horizon_ns = _seconds_ns(horizon_seconds, "horizon_seconds", allow_zero=False)
    grid = sample_grid(data, start_seconds, end_seconds, step_seconds, levels, max_age_seconds)
    step_ns = grid.attrs["step_ns"]
    if horizon_ns % step_ns:
        raise ValueError("The response horizon must be an integer multiple of the sampling interval.")
    shift = horizon_ns // step_ns
    split_position, split_ns = _partition(grid, train_fraction)
    ns = grid["timestamp_ns"].to_numpy(dtype=np.int64)
    complete = np.arange(len(grid)) + shift < len(grid)
    if not np.any(complete):
        raise ValueError("The selected interval is shorter than the response horizon. Widen it or shorten the horizon.")
    source_positions = np.flatnonzero(complete)
    target_positions = source_positions + shift
    valid = grid["valid"].to_numpy(dtype=bool)
    endpoints_valid = valid[source_positions] & valid[target_positions]
    starts, ends = ns[source_positions], ns[target_positions]
    event_ns = data.messages["time_ns"].to_numpy(dtype=np.int64)
    positive_duration = np.diff(event_ns) > 0
    invalid_intervals = ~_quote_validity(data)[:-1] & positive_duration
    crosses_invalid = _overlapping_intervals(event_ns[:-1][invalid_intervals],
                                             event_ns[1:][invalid_intervals], starts, ends)
    age_ns = grid.attrs["max_age_ns"]
    crosses_stale = np.zeros(len(starts), dtype=bool)
    if age_ns is not None:
        stale_starts = event_ns[:-1] + age_ns
        stale_intervals = stale_starts < event_ns[1:]
        crosses_stale = _overlapping_intervals(stale_starts[stale_intervals],
                                               event_ns[1:][stale_intervals], starts, ends)
    interval_valid = endpoints_valid & ~crosses_invalid & ~crosses_stale
    train_mask = interval_valid & (ends < split_ns)
    test_mask = interval_valid & (starts >= split_ns)
    train_count, test_count = int(train_mask.sum()), int(test_mask.sum())
    if train_count < 3 or test_count < 1:
        raise ValueError("Response analysis needs at least three valid training targets and one valid evaluation target after excluding boundary and invalid intervals. Widen the interval or shorten the horizon.")
    mids = grid["midquote"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        response = 10_000 * np.log(mids[target_positions] / mids[source_positions])
    imbalance = grid["imbalance"].to_numpy(dtype=float)[source_positions]
    train_x, train_y = imbalance[train_mask], response[train_mask]
    test_x, test_y = imbalance[test_mask], response[test_mask]
    x_mean, y_mean = float(train_x.mean()), float(train_y.mean())
    centered_x = train_x - x_mean
    scale = max(1.0, float(np.max(np.abs(train_x))))
    slope_identified = bool(np.ptp(train_x) > CONSTANT_RELATIVE_TOLERANCE * scale)
    slope = (float(np.dot(centered_x, train_y - y_mean) / np.dot(centered_x, centered_x))
             if slope_identified else 0.0)
    intercept = float(y_mean - slope * x_mean)
    train_prediction = intercept + slope * train_x
    test_prediction = intercept + slope * test_x
    baseline_sse = float(np.square(test_y - y_mean).sum())
    model_sse = float(np.square(test_y - test_prediction).sum())
    edges = np.linspace(-1.0, 1.0, int(bins) + 1)
    train_means, train_medians, train_bins = _binned_response(train_x, train_y, edges)
    test_means, test_medians, test_bins = _binned_response(test_x, test_y, edges)
    params = _parameters(grid, split_position, train_fraction)
    params.update({"horizon_seconds": horizon_ns / NANOSECONDS, "bins": int(bins),
                   "imbalance": "(bid displayed shares minus ask displayed shares) divided by their sum across selected occupied levels",
                   "target": "10000 * log(midquote(t + horizon) / midquote(t))",
                   "split_policy": "Training target ends strictly before split. Evaluation starts at or after split.",
                   "interval_policy": "Discard horizons crossing positive duration invalid quote or stale intervals.",
                   "bin_policy": "Fixed equal width bins on [-1, 1]. Rightmost bin includes +1.",
                   "regression": "Training ordinary least squares with intercept. Constant training imbalance uses the training target mean."})
    counts = _counts(grid)
    counts.update({"complete_horizon_rows": int(complete.sum()),
                   "incomplete_horizon_rows": int((~complete).sum()),
                   "invalid_endpoint_rows": int((~endpoints_valid).sum()),
                   "invalid_interval_rows": int(crosses_invalid.sum()),
                   "stale_interval_rows": int(crosses_stale.sum()),
                   "crossing_split_rows": int(((starts < split_ns) & (ends >= split_ns)).sum()),
                   "train_samples": train_count, "test_samples": test_count})
    return {
        "method": "Forward midquote response", "parameters": params, "counts": counts,
        "metrics": {"slope_bps": slope, "intercept_bps": intercept,
                    "slope_identified": slope_identified,
                    "train_mean_response_bps": y_mean,
                    "test_mean_response_bps": float(test_y.mean()),
                    "train_rmse_bps": float(np.sqrt(np.mean(np.square(train_y - train_prediction)))),
                    "test_rmse_bps": float(np.sqrt(model_sse / test_count)),
                    "baseline_rmse_bps": float(np.sqrt(baseline_sse / test_count)),
                    "test_r2_vs_train_mean": (float(1 - model_sse / baseline_sse) if baseline_sse > 0 else None)},
        "bin_edges": edges.tolist(), "bin_centers": ((edges[:-1] + edges[1:]) / 2).tolist(),
        "train_means": train_means, "test_means": test_means,
        "train_medians": train_medians, "test_medians": test_medians,
        "train_counts": train_bins, "test_counts": test_bins,
        "slope": slope, "intercept": intercept,
        "train_start_time": grid.index[source_positions[train_mask][0]].isoformat(),
        "train_last_target_time": grid.index[target_positions[train_mask][-1]].isoformat(),
        "test_start_time": grid.index[source_positions[test_mask][0]].isoformat(),
        "test_last_target_time": grid.index[target_positions[test_mask][-1]].isoformat(),
        "notes": ["Curves are descriptive conditional responses, not a trading strategy or causal estimate.",
                  "Overlapping targets and serial dependence mean counts are not independent sample sizes. No p values or confidence intervals are reported.",
                  "The evaluation baseline is the training mean response. Negative evaluation R squared means the fitted line is worse than that baseline.",
                  "Changing settings after viewing evaluation results makes that evaluation exploratory.",
                  "Chronological evaluation within one file does not establish performance on other dates.",
                  "Exclusion counts can overlap and should not be added together."]
    }


def patnaik_pearson(points, scaling="none"):
    """Calculate Patnaik Pearson dimension from centered singular values.

    Parameters
    ----------
    points : array_like
        Finite matrix with one point per row and one coordinate per column.
    scaling : {'none', 'standardize'}, optional
        Retain the supplied coordinate metric or divide centered coordinates by
        their sample standard deviations within this cloud. Constant coordinates
        remain zero and are retained in the embedding coordinate count.

    Returns
    -------
    dict
        Dimension, dimension divided by coordinate count, normalized singular
        spectrum, point and coordinate counts, numerical rank, theoretical rank
        bound, and calculation status. Undefined dimensions are None. Singular
        values are divided by the largest singular value for numerical safety.

    Notes
    -----
    Implements Definition 3.3 and equation (21) of arXiv:2606.19268v4. If s are
    the singular values of the column centered point matrix, the dimension is
    sum(s)**2 / sum(s**2). Covariance eigenvalues must not be substituted for s.
    Numerical rank is diagnostic only and does not truncate small singular
    values. A single point or a cloud of identical points has undefined dimension.
    """
    if scaling not in ("none", "standardize"):
        raise ValueError("scaling must be 'none' or 'standardize'.")
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("A point cloud must be a matrix with at least one coordinate.")
    if not np.isfinite(values).all():
        raise ValueError("Point cloud coordinates must all be finite.")
    n, p = values.shape
    bound = max(0, min(n - 1, p))
    result = {"dimension": None, "normalized_dimension": None,
              "point_count": int(n), "coordinate_count": int(p),
              "rank_bound": int(bound), "numerical_rank": 0,
              "constant_coordinates": [], "singular_values": [],
              "singular_values_normalization": "Largest singular value equals one.",
              "sample_limited": bool(n <= p), "status": "insufficient_points"}
    if n < 2:
        return result
    # Subtract one row before taking means so repeated nonzero points center
    # exactly to zero. Fall back to an initial common scale for extreme inputs.
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = values - values[0]
    if not np.isfinite(shifted).all():
        if scaling == "standardize":
            for col in np.flatnonzero(~np.isfinite(shifted).all(axis=0)):
                scaled = values[:, col] / np.max(np.abs(values[:, col]))
                shifted[:, col] = scaled - scaled[0]
        else:
            magnitude = float(np.max(np.abs(values)))
            scaled = values / magnitude
            shifted = scaled - scaled[0]
    constant = np.max(np.abs(shifted), axis=0) == 0
    result["constant_coordinates"] = np.flatnonzero(constant).tolist()
    if constant.all():
        result["status"] = "zero_variation"
        return result
    if scaling == "standardize":
        # Column normalization before the standard deviation also handles very
        # small but nonzero coordinates without imposing a variance cutoff.
        centered = np.zeros_like(shifted)
        for col in range(p):
            if constant[col]:
                continue
            shifted_col = shifted[:, col] / np.max(np.abs(shifted[:, col]))
            centered_col = shifted_col - shifted_col.mean()
            normalized = centered_col / np.max(np.abs(centered_col))
            deviation = float(np.sqrt(np.sum(np.square(normalized)) / (n - 1)))
            centered[:, col] = normalized / deviation
    else:
        shifted = shifted / np.max(np.abs(shifted))
        centered = shifted - shifted.mean(axis=0)
        centered[:, constant] = 0
    center_scale = float(np.max(np.abs(centered)))
    if center_scale == 0:
        result["status"] = "zero_variation"
        return result
    singular = np.linalg.svd(centered / center_scale, compute_uv=False, full_matrices=False)
    if len(singular) == 0 or singular[0] == 0:
        result["status"] = "zero_variation"
        return result
    relative = singular / singular[0]
    dimension = float(np.square(relative.sum()) / np.square(relative).sum())
    # Only enforce the exact finite point bound, never the numerical rank.
    dimension = float(np.clip(dimension, 1.0, bound))
    numerical_tolerance = max(n, p) * np.finfo(float).eps
    result.update({"dimension": dimension, "normalized_dimension": dimension / p,
                   "numerical_rank": int(np.count_nonzero(relative > numerical_tolerance)),
                   "singular_values": relative.tolist(), "status": "ok"})
    return result


def _dimension_event_validity(data, levels, embedding):
    """Identify valid source snapshots for history embeddings and delay gaps.

    Parameters
    ----------
    data : LobsterData
        Loaded observations.
    levels : int
        Selected occupied levels on each side.
    embedding : str
        History embedding name. Price depth requires every selected price.

    Returns
    -------
    numpy.ndarray
        Validity in original event order.
    """
    valid = _quote_validity(data).copy()
    total = np.zeros(len(valid), dtype=float)
    for side in ("bid", "ask"):
        for level in range(1, levels + 1):
            prices = data.book[f"{side}_price_{level}"].to_numpy(dtype=float)
            sizes = data.book[f"{side}_size_{level}"].to_numpy(dtype=float)
            available = np.isfinite(prices) & (prices > 0)
            valid &= np.isfinite(sizes) & (sizes >= 0)
            valid &= available if embedding in ("price_depth", "collaborator") else (available | (sizes == 0))
            total += np.where(available, sizes, 0.0)
    return valid & np.isfinite(total) & (total > 0)


def _dimension_history_features(data, grid, levels, embedding):
    """Build vectors representing snapshots at sampled clock times.

    Parameters
    ----------
    data : LobsterData
        Loaded source observations.
    grid : pandas.DataFrame
        Complete clock grid with validity flags and original row positions.
    levels : int
        Selected levels per side.
    embedding : {'log_depth', 'relative_depth', 'price_depth', 'collaborator'}
        Chosen representation of each snapshot.

    Returns
    -------
    tuple
        Feature matrix and coordinate names. Validity is checked separately.
    """
    size_names = [f"{side}_size_{level}" for side in ("bid", "ask")
                  for level in range(1, levels + 1)]
    sizes = grid[size_names].to_numpy(dtype=float)
    if embedding == "collaborator":
        rows = grid["row"].to_numpy(dtype=np.int64)
        # Recover LOBSTER integer price units before taking differences. The
        # variance regulariser in historical scaling makes these units matter.
        prices = {side: np.rint(data.book.iloc[rows][[f"{side}_price_{k}" for k in range(1, levels + 1)]].to_numpy(dtype=float) * 10_000)
                  for side in ("ask", "bid")}
        ask, bid = prices["ask"], prices["bid"]
        differences = [ask[:, 1:] - ask[:, :1], ask - bid[:, :1],
                       bid[:, 1:] - bid[:, :1], bid - ask[:, :1]]
        names = ([f"ask_{k}_minus_ask_1" for k in range(2, levels + 1)]
                 + [f"ask_{k}_minus_bid_1" for k in range(1, levels + 1)]
                 + [f"bid_{k}_minus_bid_1" for k in range(2, levels + 1)]
                 + [f"bid_{k}_minus_ask_1" for k in range(1, levels + 1)])
        cumulative = np.column_stack((np.cumsum(sizes[:, levels:], axis=1),
                                      np.cumsum(sizes[:, :levels], axis=1)))
        volume_names = [f"cumulative_{side}_size_{k}" for side in ("ask", "bid")
                        for k in range(1, levels + 1)]
        return (np.column_stack((*differences, cumulative, np.log(np.maximum(cumulative, 1e-8)))),
                names + volume_names + [f"log_{name}" for name in volume_names])
    if embedding == "relative_depth":
        total = sizes.sum(axis=1, keepdims=True)
        values = np.divide(sizes, total, out=np.zeros_like(sizes), where=total > 0)
        return values, [f"share_{name}" for name in size_names]
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.log1p(sizes)
    names = [f"log1p_{name}" for name in size_names]
    if embedding == "price_depth":
        rows = grid["row"].to_numpy(dtype=np.int64)
        mids = grid["midquote"].to_numpy(dtype=float)
        distances = []
        for side, sign in (("bid", -1), ("ask", 1)):
            for level in range(1, levels + 1):
                prices = data.book[f"{side}_price_{level}"].to_numpy(dtype=float)[rows]
                with np.errstate(invalid="ignore", divide="ignore"):
                    distances.append(sign * 10_000 * (prices - mids) / mids)
                names.append(f"{side}_mid_distance_bps_{level}")
        values = np.column_stack((values, *distances))
    return values, names


FEATURE_GROUPS = {
    "price_differences": "Price differences",
    "cumulative_volume": "Cumulative volumes",
    "log_cumulative_volume": "Logarithms of cumulative volumes",
    "imbalance": "Volume imbalance",
    "log_imbalance": "Imbalance using logarithms of volumes",
    "total_volume": "Combined cumulative volumes",
    "log_total_volume": "Logarithms of combined cumulative volumes",
    "log_depth": "Log volumes at each level",
    "relative_depth": "Volumes as proportions of total depth",
    "mid_distances": "Distances from the midquote",
}
TOM_FEATURES = {
    "price_differences": "ewm",
    "cumulative_volume": "ewm",
    "log_cumulative_volume": "ewm",
    "imbalance": "both",
    "log_imbalance": "both",
    "total_volume": "ewm",
    "log_total_volume": "ewm",
    "log_depth": "omit",
    "relative_depth": "omit",
    "mid_distances": "omit",
}


def feature_coordinate_count(features, levels):
    """Count selected snapshot coordinates and validate their transformations.

    Parameters
    ----------
    features : dict
        Feature group names mapped to omit, original, ewm or both.
    levels : int
        Number of occupied levels on each side.

    Returns
    -------
    int
        Number of selected coordinates before combining snapshots.
    """
    if not isinstance(features, dict) or set(features) - set(FEATURE_GROUPS):
        raise ValueError("Choose valid snapshot feature groups.")
    count = 0
    for key, choice in features.items():
        if choice not in ("omit", "original", "ewm", "both"):
            raise ValueError("Choose an available feature transformation.")
        width = 4 * levels - 2 if key == "price_differences" else levels if key in ("imbalance", "log_imbalance", "total_volume", "log_total_volume") else 2 * levels
        count += width * (2 if choice == "both" else 0 if choice == "omit" else 1)
    return count


def _selected_history_features(data, grid, levels, features):
    """Build selected coordinates and identify those requiring historical scaling.

    Parameters
    ----------
    data : LobsterData
        Loaded snapshots.
    grid : pandas.DataFrame
        Sampled rows and volumes.
    levels : int
        Occupied levels on each side.
    features : dict
        Selected feature groups and transformations.

    Returns
    -------
    tuple
        Matrix, unique coordinate names and historical scaling mask.
    """
    blocks, names, historical = [], [], []
    cumulative = None
    for key in FEATURE_GROUPS:
        choice = features.get(key, "omit")
        if choice == "omit":
            continue
        if key in ("price_differences", "cumulative_volume", "log_cumulative_volume", "imbalance", "log_imbalance", "total_volume", "log_total_volume"):
            if cumulative is None:
                cumulative = {side: np.cumsum(grid[[f"{side}_size_{k}" for k in range(1, levels + 1)]].to_numpy(dtype=float), axis=1)
                              for side in ("ask", "bid")}
            if key == "price_differences":
                values, labels = _dimension_history_features(data, grid, levels, "collaborator")
                values, labels = values[:, :4 * levels - 2], labels[:4 * levels - 2]
            elif key in ("cumulative_volume", "log_cumulative_volume"):
                values = np.column_stack((cumulative["ask"], cumulative["bid"]))
                if key == "log_cumulative_volume":
                    values = np.log(np.maximum(values, 1e-8))
                labels = [f"{key}_{side}_{k}" for side in ("ask", "bid") for k in range(1, levels + 1)]
            elif key in ("total_volume", "log_total_volume"):
                values = cumulative["bid"] + cumulative["ask"]
                if key == "log_total_volume":
                    values = np.log(np.maximum(values, 1e-8))
                labels = [f"{key}_{k}" for k in range(1, levels + 1)]
            else:
                bid, ask = cumulative["bid"], cumulative["ask"]
                if key == "log_imbalance":
                    bid, ask = np.log(np.maximum(bid, 1e-8)), np.log(np.maximum(ask, 1e-8))
                values = (bid - ask) / (bid + ask + 1e-6)
                labels = [f"{key}_{k}" for k in range(1, levels + 1)]
        else:
            values, labels = _dimension_history_features(data, grid, levels, "price_depth" if key == "mid_distances" else key)
            if key == "mid_distances":
                values, labels = values[:, 2 * levels:], labels[2 * levels:]
        for scaled in ([False, True] if choice == "both" else [choice == "ewm"]):
            blocks.append(values)
            names.extend([name + ("_ewm" if scaled else "") for name in labels])
            historical.extend([scaled] * values.shape[1])
    return np.column_stack(blocks), names, np.asarray(historical, dtype=bool)


def _historical_standardize(values, valid, lookback):
    """Standardise observations using strictly earlier exponential moments.

    Parameters
    ----------
    values : numpy.ndarray
        Matrix in chronological sample order.
    valid : numpy.ndarray
        Usable sample flags. Invalid samples restart the moment estimates.
    lookback : int
        Initial observations discarded in each valid run. Exponential weights
        decay to 0.01 after this many observations, without a hard cutoff.

    Returns
    -------
    tuple of numpy.ndarray
        Standardised matrix and flags for observations discarded during initialisation.
    """
    alpha = 1.0 - np.exp(np.log(.01) / lookback)
    output = np.full_like(values, np.nan, dtype=float)
    warmup = np.zeros(len(values), dtype=bool)
    edges = np.diff(np.r_[False, valid, False].astype(int))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        warmup[start:min(stop, start + lookback)] = True
        if stop - start <= lookback:
            continue
        # Process columns separately to avoid several full matrix copies.
        for column in range(values.shape[1]):
            series = pd.Series(values[start:stop, column])
            mean = series.ewm(alpha=alpha, adjust=False).mean().shift(1)
            mean.iloc[0] = series.iloc[0]
            variance = ((series - mean) ** 2).ewm(alpha=alpha, adjust=False).mean().shift(1)
            variance.iloc[0] = 0.0
            scaled = ((series - mean) / np.sqrt(variance + 1e-6)).to_numpy()
            output[start + lookback:stop, column] = scaled[lookback:]
    return output, warmup


def _historical_standardize_chunk(values, valid, lookback, state):
    """Carry causal exponential moments across consecutive chunks.

    Parameters
    ----------
    values : numpy.ndarray
        Chronological coordinates requiring historical standardisation.
    valid : numpy.ndarray
        Raw observation validity, before historical warmup exclusions.
    lookback : int
        Number of valid observations excluded after each initialisation.
    state : tuple or None
        Next mean, next variance and valid run length from the preceding chunk.

    Returns
    -------
    tuple
        Standardised values, warmup mask and continuation state. An invalid
        observation resets the moments, including at chunk boundaries.
    """
    alpha = 1.0 - np.exp(np.log(.01) / lookback)
    output = np.full_like(values, np.nan, dtype=float)
    warmup = np.zeros(len(values), dtype=bool)
    edges = np.diff(np.r_[False, valid, False].astype(int))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if start:
            state = None
        mean, variance, count = (state if state is not None else
                                 (values[start].copy(), np.zeros(values.shape[1]), 0))
        segment = values[start:stop]
        # Seed EWM with the moments after the previous chunk. The first row
        # then uses those moments, before its own observation updates them.
        means = pd.DataFrame(np.vstack((mean, segment))).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        residuals = segment - means[:-1]
        variances = pd.DataFrame(np.vstack((variance, residuals ** 2))).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        output[start:stop] = residuals / np.sqrt(variances[:-1] + 1e-6)
        warmup[start:min(stop, start + max(0, lookback - count))] = True
        state = (means[-1].copy(), variances[-1].copy(), count + stop - start)
    if len(valid) and not valid[-1]:
        state = None
    output[warmup] = np.nan
    return output, warmup, state


def _feature_chunk_rows(coordinates):
    """Bound both the row count and byte size of one feature chunk.

    Parameters
    ----------
    coordinates : int
        Coordinates per snapshot, including temporary feature construction.

    Returns
    -------
    int
        Positive number of rows allowed in a chunk.
    """
    return max(1, min(DIMENSION_FEATURE_CHUNK_ROWS,
                      DIMENSION_FEATURE_CHUNK_BYTES // (8 * max(1, coordinates))))


class _HistoryFeatureStream:
    """Build consecutive feature chunks and retain a bounded rolling history.

    Parameters
    ----------
    grid : pandas.DataFrame
        Ordered sampled observations, including invalid rows.
    builder : callable
        Build values, coordinate names and historical mask for a grid slice.
    valid : numpy.ndarray
        Raw validity copied before warmup and finite feature exclusions.
    capacity : int
        Maximum rows retained for an estimation window, including delay history.
    chunk_rows : int
        Maximum rows constructed at once.
    lookback : int or None
        Historical warmup length, or None to retain original features.
    """

    def __init__(self, grid, builder, valid, capacity, chunk_rows, lookback):
        self.grid, self.builder = grid, builder
        probe, self.names, self.historical = builder(grid.iloc[:1])
        self.capacity = max(1, min(len(grid), capacity))
        self.buffer = np.empty((self.capacity, probe.shape[1]), dtype=float)
        self.valid = valid.copy()
        self.warmup = np.zeros(len(grid), dtype=bool)
        self.chunk_rows, self.lookback = chunk_rows, lookback
        self.position, self.state = 0, None

    def advance(self, stop):
        """Process all observations through an exclusive sampled position.

        Parameters
        ----------
        stop : int
            Exclusive ending position, never earlier than the previous call.

        Returns
        -------
        None
        """
        for start in range(self.position, stop, self.chunk_rows):
            end = min(stop, start + self.chunk_rows)
            values, _, _ = self.builder(self.grid.iloc[start:end])
            valid = self.valid[start:end] & np.isfinite(values).all(axis=1)
            if self.lookback is not None:
                scaled, warmup, self.state = _historical_standardize_chunk(
                    values[:, self.historical], valid, self.lookback, self.state)
                values[:, self.historical] = scaled
                self.warmup[start:end] = warmup
            self.valid[start:end] = valid & np.isfinite(values).all(axis=1)
            keep = min(self.capacity, end - start)
            positions = np.arange(end - keep, end) % self.capacity
            self.buffer[positions] = values[-keep:]
        self.position = stop

    def take(self, positions):
        """Read retained rows in the requested order.

        Parameters
        ----------
        positions : numpy.ndarray
            Original sampled positions within the retained window.

        Returns
        -------
        numpy.ndarray
            Feature rows in the requested order.
        """
        if len(positions) and (positions.min() < max(0, self.position - self.capacity)
                               or positions.max() >= self.position):
            raise RuntimeError("Requested feature rows are outside the rolling buffer.")
        return self.buffer[positions % self.capacity]


def _message_grid(data, start_ns, end_ns, levels):
    """Select every message snapshot, retaining original order at tied times.

    Parameters
    ----------
    data : LobsterData
        Aligned observations.
    start_ns, end_ns : int
        Inclusive selected bounds in nanoseconds after midnight.
    levels : int
        Number of levels per side.

    Returns
    -------
    pandas.DataFrame
        Samples in original row order, including invalid observations.
    """
    times = data.messages["time_ns"].to_numpy(dtype=np.int64)
    lo = np.searchsorted(times, start_ns, side="left")
    hi = np.searchsorted(times, end_ns, side="right")
    if hi == lo:
        raise ValueError("No messages fall inside the selected period.")
    if hi - lo > MAX_MESSAGE_SAMPLES:
        raise ValueError(f"This period contains {hi - lo:,} messages. Shorten the observation window to at most {MAX_MESSAGE_SAMPLES:,} messages, or choose clock sampling.")
    rows = np.arange(lo, hi, dtype=np.int64)
    frame = pd.DataFrame(index=pd.DatetimeIndex(data.messages["timestamp"].iloc[rows], name="timestamp"))
    frame["row"] = rows
    frame["timestamp_ns"] = times[rows]
    frame["valid"] = _quote_validity(data)[rows]
    frame["midquote"] = data.features["midquote"].to_numpy(dtype=float)[rows]
    for side in ("bid", "ask"):
        for level in range(1, levels + 1):
            name = f"{side}_size_{level}"
            frame[name] = data.book[name].to_numpy(dtype=float)[rows]
    return frame


def _dimension_snapshot_points(data, row, levels, patch_width):
    """Construct observations from one or three levels per side in one snapshot.

    Parameters
    ----------
    data : LobsterData
        Loaded source observations.
    row : int
        Original position of the source snapshot.
    levels : int
        Selected occupied levels on each side.
    patch_width : int
        One paired level or three consecutive paired levels per point.

    Returns
    -------
    tuple
        Point matrix, valid patch starting levels, and invalid level count.
    """
    snapshot = data.book.iloc[row]
    mid = float(data.features["midquote"].iloc[row])
    base = np.full((levels, 4), np.nan)
    valid = np.ones(levels, dtype=bool)
    for level in range(1, levels + 1):
        bid = float(snapshot[f"bid_price_{level}"])
        ask = float(snapshot[f"ask_price_{level}"])
        bid_size = float(snapshot[f"bid_size_{level}"])
        ask_size = float(snapshot[f"ask_size_{level}"])
        good = (np.isfinite([bid, ask, bid_size, ask_size]).all()
                and bid > 0 and ask > 0 and bid_size >= 0 and ask_size >= 0)
        valid[level - 1] = good
        if good:
            base[level - 1] = [np.log1p(bid_size), np.log1p(ask_size),
                               10_000 * (mid - bid) / mid, 10_000 * (ask - mid) / mid]
    starts = [i for i in range(levels - patch_width + 1) if valid[i:i + patch_width].all()]
    points = (np.stack([base[i:i + patch_width].reshape(-1) for i in starts])
              if starts else np.empty((0, 4 * patch_width)))
    return points, [i + 1 for i in starts], int((~valid).sum())


def _dimension_endpoint_grid(data, start_ns, end_ns, interval_ns):
    """Construct bounded midnight anchored evaluation endpoints.

    Parameters
    ----------
    data : LobsterData
        Loaded source observations.
    start_ns, end_ns : int
        Requested interval in nanoseconds after midnight.
    interval_ns : int
        Positive separation between evaluation endpoints.

    Returns
    -------
    numpy.ndarray
        Integer nanosecond endpoints within observed and requested bounds.
    """
    events = data.messages["time_ns"].to_numpy(dtype=np.int64)
    if len(events) == 0 or np.any(np.diff(events) < 0):
        raise ValueError("Dimension analysis requires nonempty, ordered message times.")
    lower, upper = max(start_ns, int(events[0])), min(end_ns, int(events[-1]))
    first = ((lower + interval_ns - 1) // interval_ns) * interval_ns
    count = 0 if first > upper else (upper - first) // interval_ns + 1
    if count == 0:
        raise ValueError("No dimension evaluation times fall inside this interval. Widen the selection or shorten the evaluation interval.")
    if count > MAX_DIMENSION_ENDPOINTS:
        raise ValueError(f"This creates {count:,} evaluation times. Increase the evaluation interval to use at most {MAX_DIMENSION_ENDPOINTS:,} endpoints.")
    return first + np.arange(count, dtype=np.int64) * interval_ns


def _dimension_memory_estimate(sample_count, base_coordinates, coordinates,
                               candidate_points, endpoint_count, *, history_samples=None,
                               levels=0, event_count=0):
    """Estimate additional memory for a dimension sweep before allocation.

    Parameters
    ----------
    sample_count : int
        Number of sampled snapshots across the selected period.
    base_coordinates : int
        Variables describing one snapshot before combining snapshots.
    coordinates : int
        Columns in each matrix passed to PP.
    candidate_points : int
        Maximum rows in a matrix that will actually be calculated.
    endpoint_count : int
        Number of requested estimates.
    history_samples : int or None, optional
        Retained history rows including delay constituents. Defaults to the
        candidate point count for callers without delayed observations.
    levels : int, optional
        Levels per side in the sampled metadata and size grid.
    event_count : int, optional
        Source rows requiring validity and interval bookkeeping.

    Returns
    -------
    int
        Approximate bytes for feature arrays, matrix copies used during SVD,
        coordinate names and retained spectra and constant coordinate indices.
        Loaded source data, prior runs and browser memory are not included.
    """
    samples, base, columns, rows, endpoints = map(
        int, (sample_count, base_coordinates, coordinates, candidate_points, endpoint_count))
    history = min(samples, rows if history_samples is None else int(history_samples))
    chunk = min(samples, _feature_chunk_rows(base))
    features = 8 * base * (history + 8 * chunk)
    bookkeeping = samples * (128 + 32 * levels) + 64 * event_count
    matrix_copies = 8 * 8 * rows * columns
    results = 64 * endpoints * (min(rows, columns) + columns)
    names = 160 * columns
    return features + bookkeeping + matrix_copies + results + names


def _check_dimension_memory(estimated_bytes):
    """Reject a sweep whose estimated additional memory is too large.

    Parameters
    ----------
    estimated_bytes : int
        Estimate returned by `_dimension_memory_estimate`.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the estimate exceeds the configured memory allowance.
    """
    if estimated_bytes > MAX_DIMENSION_MEMORY_BYTES:
        raise ValueError(
            f"This request is estimated to need {estimated_bytes / 1024**3:.2f} GiB of additional memory "
            f"(allowance {MAX_DIMENSION_MEMORY_BYTES / 1024**3:.2f} GiB). "
            "Shorten the selected period or estimation window, or reduce levels or snapshots per observation.")


def run_dimension(data, start_seconds, end_seconds, step_seconds=5, levels=10,
                  mode="history", embedding="log_depth", patch_width=1,
                  lag_count=1, lag_spacing_seconds=5, window_seconds=300,
                  evaluation_step_seconds=60, scaling="none", max_age_seconds=60,
                  sampling="clock", window_messages=1000, lag_spacing_messages=1,
                  ewm_lookback=501, burn_in_seconds=0, burn_out_seconds=0, features=None):
    """Sweep snapshot geometry or trailing history Patnaik Pearson dimension.

    Parameters
    ----------
    data : LobsterData
        Loaded, aligned LOBSTER observations.
    start_seconds, end_seconds : float
        Selected exchange clock interval in seconds after midnight.
    step_seconds : float, optional
        History clock sampling interval. Snapshot mode does not use it.
    levels : int, optional
        Occupied levels per side.
    mode : {'history', 'snapshot'}, optional
        Treat vectors representing snapshots over time or price levels within one snapshot
        as points. The two modes describe different mathematical objects.
    embedding : {'log_depth', 'relative_depth', 'price_depth', 'collaborator'}, optional
        History vectors use log1p displayed sizes, sizes divided by total depth
        on both sides, or log1p sizes plus outward midquote distances in basis
        points. Collaborator features use best quote differences, cumulative
        volumes and their logarithms. Snapshot mode uses level price and size data.
    patch_width : {1, 3}, optional
        Snapshot points concatenate one or three adjacent paired levels.
    lag_count : int, optional
        Number of snapshots combined into each observation, including earlier times.
    lag_spacing_seconds : float, optional
        Separation of lag vectors for clock sampling, a positive multiple of
        step_seconds. Ignored for message sampling.
    window_seconds : float, optional
        Width of the complete trailing history window for clock sampling. Every
        constituent sample must lie within that window and the selected interval.
        Ignored for message sampling.
    evaluation_step_seconds : float, optional
        Separation of evaluation endpoints, anchored to midnight. With clock
        sampling this must be an integer multiple of the sampling interval.
    scaling : {'none', 'standardize', 'ewm'}, optional
        Keep chosen coordinate units or use coordinate standard deviations
        calculated within each cloud, or apply causal historical standardisation
        to sampled snapshots before constructing observations and centring.
        There is no training partition.
    max_age_seconds : float or None, optional
        Maximum source snapshot age. None disables the age restriction.
        Message sampling does not use this setting.
    sampling : {'clock', 'messages'}, optional
        Sample at regular times or retain every message in original order.
    window_messages : int, optional
        Number of messages per estimation window when sampling messages.
    lag_spacing_messages : int, optional
        Message separation of constituent snapshots when sampling messages.
    ewm_lookback : int, optional
        Initialisation length for historical scaling. Weights decay to 0.01
        after this many sampled observations. Used only with scaling='ewm'.
    burn_in_seconds, burn_out_seconds : float, optional
        Durations removed from the start and end of the selected interval.
        Excluded observations do not enter PP matrices or initialise historical
        standardisation. The two remaining interval boundaries are inclusive.
    features : dict, optional
        Groups and original or historical transformations for embedding='custom'.

    Returns
    -------
    dict
        Strict JSON compatible method, parameters, counts, metrics, notes,
        coordinate names, and a series of time, dimension, normalized spectrum,
        point count, coordinate count, rank bound, status, and source metadata.

    Notes
    -----
    A snapshot is not treated as a single vector whose dimension could be
    estimated. Its points are paired occupied levels or their adjacent patches.
    History observations represent one snapshot or combine several snapshots. Invalid
    grid observations are retained until lag construction, and no delay vector
    bridges invalid or stale source intervals. Windows use no observations from
    before the selected interval. Snapshot evaluation uses the final original
    row at or before its clock time, including all events tied at that time.
    """
    if mode not in ("history", "snapshot"):
        raise ValueError("mode must be 'history' or 'snapshot'.")
    if scaling not in ("none", "standardize", "ewm"):
        raise ValueError("scaling must be 'none', 'standardize', or 'ewm'.")
    if sampling not in ("clock", "messages"):
        raise ValueError("sampling must be 'clock' or 'messages'.")
    if mode == "snapshot" and scaling == "ewm":
        raise ValueError("Historical standardisation applies only to recent histories of snapshots.")
    custom_historical = embedding == "custom" and isinstance(features, dict) and any(v in ("ewm", "both") for v in features.values())
    if embedding == "custom" and scaling == "ewm":
        raise ValueError("Choose historical standardisation for each feature group.")
    if (scaling == "ewm" or custom_historical) and (isinstance(ewm_lookback, (bool, np.bool_))
                            or not isinstance(ewm_lookback, (int, np.integer)) or ewm_lookback < 1):
        raise ValueError("The standardisation history must be a positive integer.")
    estimator_scaling = "none" if scaling == "ewm" else scaling
    levels = _level_count(data, levels)
    start_ns = _seconds_ns(start_seconds, "start_seconds")
    end_ns = _seconds_ns(end_seconds, "end_seconds")
    selected_start_ns, selected_end_ns = start_ns, end_ns
    burn_in_ns = _seconds_ns(burn_in_seconds, "Burn in time")
    burn_out_ns = _seconds_ns(burn_out_seconds, "Burn out time")
    start_ns += burn_in_ns
    end_ns -= burn_out_ns
    if start_ns >= end_ns:
        raise ValueError("Burn in and burn out must leave a positive duration within the selected observation window.")
    start_seconds, end_seconds = start_ns / NANOSECONDS, end_ns / NANOSECONDS
    step_ns = _seconds_ns(step_seconds, "step_seconds", allow_zero=False)
    evaluation_ns = _seconds_ns(evaluation_step_seconds, "evaluation_step_seconds", allow_zero=False)
    age_ns = None if max_age_seconds is None else _seconds_ns(max_age_seconds, "max_age_seconds")
    if start_ns >= end_ns:
        raise ValueError("The start time must be before the end time.")
    endpoints = _dimension_endpoint_grid(data, start_ns, end_ns, evaluation_ns)
    event_ns = data.messages["time_ns"].to_numpy(dtype=np.int64)
    day = pd.Timestamp(data.messages["timestamp"].iloc[0]).normalize()
    records = []
    params = {"start_seconds": start_ns / NANOSECONDS, "end_seconds": end_ns / NANOSECONDS,
              "selected_start_seconds": selected_start_ns / NANOSECONDS,
              "selected_end_seconds": selected_end_ns / NANOSECONDS,
              "burn_in_seconds": burn_in_ns / NANOSECONDS,
              "burn_out_seconds": burn_out_ns / NANOSECONDS,
              "burn_policy": "Discard observations outside the inclusive retained interval before sampling, standardisation and matrix construction. Historical standardisation starts afresh within the retained interval.",
              "mode": mode, "levels": levels, "scaling": scaling,
              "evaluation_step_seconds": evaluation_ns / NANOSECONDS,
              "max_age_seconds": None if age_ns is None else age_ns / NANOSECONDS,
              "centering": "Subtract the coordinate mean within each point cloud.",
              "estimator": "(sum of centered singular values)^2 / sum of their squares",
              "normalization": "Dimension divided by full embedding coordinate count, including constant coordinates.",
              "spectrum_normalization": "Largest singular value equals one. No singular values are discarded for dimension.",
              "paper": "https://arxiv.org/html/2606.19268v4#S3.SS3",
              "author_implementation": "https://github.com/tdhadfield/PatnaikPearson/blob/main/PatnaikPearson/helper.py",
              "quote_policy": "Exclude missing, locked, crossed, halted, and stale source quotes.",
              "clock_policy": "Midnight anchored grids. Final original row at or before each time, including final row at tied timestamps.",
              "scaling_policy": "Within cloud coordinate standard deviations change the metric separately in each cloud." if scaling == "standardize" else "No coordinate standardization. Relative coordinate units affect the result."}
    if mode == "snapshot":
        if (isinstance(patch_width, (bool, np.bool_)) or not isinstance(patch_width, (int, np.integer))
                or patch_width not in (1, 3) or patch_width > levels):
            raise ValueError("Snapshot patch width must be 1 or 3 and must not exceed the selected levels.")
        patch_width = int(patch_width)
        coordinate_count = 4 * patch_width
        candidate_points = levels - patch_width + 1
        work = len(endpoints) * candidate_points * coordinate_count * min(candidate_points, coordinate_count)
        memory_estimate = _dimension_memory_estimate(
            0, 0, coordinate_count, candidate_points, len(endpoints))
        _check_dimension_memory(memory_estimate)
        feature_names = [f"{name}_offset_{offset}" for offset in range(patch_width)
                         for name in ("log1p_bid_size", "log1p_ask_size", "bid_mid_distance_bps", "ask_mid_distance_bps")]
        params.update({"embedding": "paired_level_price_depth", "patch_width": patch_width,
                       "point_definition": "One paired occupied level, or one consecutive patch of paired levels, within a single snapshot.",
                       "window_policy": "Use only source rows at or after the selected start and at or before each evaluation time. Earlier source rows remain explicit gaps.",
                       "price_coordinates": "Outward distances are 10000*(mid-bid)/mid and 10000*(ask-mid)/mid.",
                       "missing_depth_policy": "Remove a paired level with any missing price or invalid size. Remove every patch containing that level."})
        if work > MAX_DIMENSION_SVD_WORK:
            raise ValueError("This snapshot sweep requests too much matrix work. Increase the evaluation interval or reduce the selected levels.")
        quotes_valid = _quote_validity(data)
        rows = np.searchsorted(event_ns, endpoints, side="right") - 1
        for endpoint, row in zip(endpoints, rows):
            timestamp = day + pd.Timedelta(int(endpoint), unit="ns")
            source_time = pd.Timestamp(data.messages["timestamp"].iloc[row])
            age = int(endpoint - event_ns[row])
            bad_quote = not quotes_valid[row]
            stale = age_ns is not None and age > age_ns
            before_start = event_ns[row] < start_ns
            if before_start or bad_quote or stale:
                record = patnaik_pearson(np.empty((0, coordinate_count)), scaling)
                status = ("source_before_selected_start" if before_start
                          else "invalid_snapshot" if bad_quote else "stale_snapshot")
                record.update({"status": status,
                               "invalid_levels": 0, "valid_patch_start_levels": []})
            else:
                points, starts, invalid_levels = _dimension_snapshot_points(data, int(row), levels, patch_width)
                record = patnaik_pearson(points, scaling)
                record.update({"invalid_levels": invalid_levels, "valid_patch_start_levels": starts})
            record.update({"time": timestamp.isoformat(), "seconds": int(endpoint) / NANOSECONDS,
                           "source_row": int(row), "source_time": source_time.isoformat(),
                           "source_seconds": int(event_ns[row]) / NANOSECONDS,
                           "quote_age_seconds": age / NANOSECONDS,
                           "candidate_points": candidate_points,
                           "excluded_points": candidate_points - record["point_count"]})
            records.append(record)
    else:
        if embedding not in ("log_depth", "relative_depth", "price_depth", "collaborator", "custom"):
            raise ValueError("Choose log_depth, relative_depth, price_depth, or collaborator.")
        if (isinstance(lag_count, (bool, np.bool_)) or not isinstance(lag_count, (int, np.integer))
                or lag_count < 1):
            raise ValueError("lag_count must be a positive integer.")
        lag_count = int(lag_count)
        spacing_ns = _seconds_ns(lag_spacing_seconds, "lag_spacing_seconds", allow_zero=False)
        window_ns = _seconds_ns(window_seconds, "window_seconds", allow_zero=False)
        if sampling == "messages":
            for name, value in (("window_messages", window_messages), ("lag_spacing_messages", lag_spacing_messages)):
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                    raise ValueError(f"{name} must be a positive integer.")
            window_messages, lag_spacing_messages = int(window_messages), int(lag_spacing_messages)
            lag_steps = lag_spacing_messages if lag_count > 1 else 0
        else:
            if evaluation_ns % step_ns:
                raise ValueError("The evaluation interval must be an integer multiple of the history sampling interval.")
            if lag_count > 1 and spacing_ns % step_ns:
                raise ValueError("Lag spacing must be an integer multiple of the history sampling interval.")
            lag_steps = spacing_ns // step_ns if lag_count > 1 else 0
        lag_span = (lag_count - 1) * lag_steps
        window_samples = window_messages if sampling == "messages" else window_ns // step_ns + 1
        if window_samples < lag_span + 2:
            raise ValueError("The history window must contain at least two complete delay vectors. Increase the window or reduce the lag count or spacing.")
        base_coordinates = 8 * levels - 2 if embedding == "collaborator" else (4 if embedding == "price_depth" else 2) * levels
        if embedding == "custom":
            base_coordinates = feature_coordinate_count(features, levels)
            if not base_coordinates:
                raise ValueError("Select at least one snapshot feature group.")
        coordinate_count = base_coordinates * lag_count
        candidate_points = window_samples - lag_span
        if sampling == "messages":
            first_source = np.searchsorted(event_ns, start_ns, side="left")
            complete_windows = int(np.count_nonzero(np.searchsorted(event_ns, endpoints, side="right") - first_source >= window_messages))
            sample_count = int(np.searchsorted(event_ns, end_ns, side="right") - first_source)
        else:
            complete_windows = int(np.count_nonzero(endpoints - window_ns >= max(start_ns, int(event_ns[0]))))
            lower, upper = max(start_ns, int(event_ns[0])), min(end_ns, int(event_ns[-1]))
            first_sample = ((lower + step_ns - 1) // step_ns) * step_ns
            sample_count = max(0, (upper - first_sample) // step_ns + 1)
        feature_workspace = max(base_coordinates, 8 * levels - 2) if embedding == "custom" else base_coordinates
        history_samples = min(window_samples, sample_count) if complete_windows else 1
        memory_estimate = _dimension_memory_estimate(
            sample_count, feature_workspace, coordinate_count,
            min(candidate_points, sample_count) if complete_windows else 0, len(endpoints),
            history_samples=history_samples, levels=levels, event_count=len(event_ns))
        _check_dimension_memory(memory_estimate)
        work = complete_windows * candidate_points * coordinate_count * min(candidate_points, coordinate_count)
        if work > MAX_DIMENSION_SVD_WORK:
            raise ValueError("This history sweep requests too much matrix work. Calculate dimension less often, shorten the estimation window or selected period, or reduce levels or snapshots per observation.")
        grid = (_message_grid(data, start_ns, end_ns, levels) if sampling == "messages" else
                sample_grid(data, start_seconds, end_seconds, step_seconds, levels, max_age_seconds))
        grid_ns = grid["timestamp_ns"].to_numpy(dtype=np.int64)
        source_rows = grid["row"].to_numpy(dtype=np.int64)
        validity_embedding = ("collaborator" if any(features.get(k, "omit") != "omit" for k in ("price_differences", "mid_distances")) else "log_depth") if embedding == "custom" else embedding
        event_valid = _dimension_event_validity(data, levels, validity_embedding)
        selected_sources = event_ns[source_rows] >= start_ns
        base_valid = (grid["valid"].to_numpy(dtype=bool) & event_valid[source_rows] & selected_sources)
        if embedding == "custom":
            builder = lambda part: _selected_history_features(data, part, levels, features)
            params["features"] = dict(features)
            params["feature_source"] = "https://github.com/RobertGBoyce/PP_LOB/blob/main/LOB/helper.py"
            params["imbalance_epsilon"] = 1e-6
            params["log_cumulative_volume_floor"] = 1e-8
            params["log_total_volume_floor"] = 1e-8
        else:
            builder = lambda part: (*_dimension_history_features(data, part, levels, embedding),
                                    np.ones(base_coordinates, dtype=bool))
        stream = _HistoryFeatureStream(
            grid, builder, base_valid, history_samples, _feature_chunk_rows(feature_workspace),
            ewm_lookback if scaling == "ewm" or custom_historical else None)
        base_valid, warmup = stream.valid, stream.warmup
        params.update({"feature_processing": "Bounded chunks with a rolling feature buffer. Historical moments continue across chunks and reset only at invalid observations.",
                       "feature_chunk_rows": stream.chunk_rows, "feature_buffer_rows": stream.capacity})
        if scaling == "ewm" or custom_historical:
            params.update({"ewm_lookback": int(ewm_lookback), "ewm_decay": float(np.exp(np.log(.01) / ewm_lookback)),
                           "ewm_variance_epsilon": 1e-6,
                           "scaling_policy": "Subtract the exponentially weighted mean of strictly earlier sampled observations and divide by sqrt(previous variance + 1e-6). Variance is the exponentially weighted squared error from the previous mean. No additional within estimate standardisation.",
                           "ewm_initialisation": "Initial mean is the first valid observation and variance is zero. Discard the first lookback observations. Restart after each invalid sampled observation. No observations before the selected start are used.",
                           "ewm_warmup_observations": 0})
            if embedding == "custom":
                params["scaling_policy"] = "Historical standardisation applies only to selected coordinates. Original coordinates are retained unchanged. All columns are then centred within each estimate" + (" and divided by their within estimate standard deviations." if scaling == "standardize" else ".")
        feature_names = [f"{name}_lag_{lag}" for lag in range(lag_count) for name in stream.names]
        bad_interval = np.zeros(len(grid), dtype=bool)
        stale_interval = np.zeros(len(grid), dtype=bool)
        delay_positions = np.arange(lag_span, len(grid), dtype=np.int64)
        if len(delay_positions):
            if lag_span:
                left = grid_ns[delay_positions - lag_span]
                right = grid_ns[delay_positions]
                invalid_events = ~event_valid[:-1] & (np.diff(event_ns) > 0)
                bad_interval[delay_positions] = _overlapping_intervals(
                    event_ns[:-1][invalid_events], event_ns[1:][invalid_events], left, right)
                if age_ns is not None:
                    stale_starts = event_ns[:-1] + age_ns
                    stale_events = stale_starts < event_ns[1:]
                    stale_interval[delay_positions] = _overlapping_intervals(
                        stale_starts[stale_events], event_ns[1:][stale_events], left, right)
                if sampling == "messages":
                    # Include invalid events even when their timestamps are tied.
                    bad_prefix = np.r_[0, np.cumsum(~event_valid[source_rows])]
                    bad_interval[delay_positions] |= (bad_prefix[delay_positions + 1]
                                                       - bad_prefix[delay_positions - lag_span]) > 0
                    stale_interval[:] = False
        params.update({"embedding": embedding, "step_seconds": step_ns / NANOSECONDS,
                       "window_seconds": window_ns / NANOSECONDS, "lag_count": lag_count,
                       "lag_spacing_seconds": spacing_ns / NANOSECONDS,
                       "lag_order": "Current vector first, then progressively earlier vectors.",
                       "point_definition": "One vector representing a snapshot at each valid sampling time, optionally combined with vectors representing earlier snapshots.",
                       "window_policy": "Inclusive complete trailing window wholly inside selected and observed coverage. No source rows before the selected start.",
                       "delay_policy": "Every constituent time must be valid. No delay vector bridges any positive duration invalid or stale source interval.",
                       "missing_depth_policy": "Price depth requires every selected price. Size embeddings allow a missing price only when its size is zero.",
                       "price_coordinates": "Outward distances are 10000*(mid-bid)/mid and 10000*(ask-mid)/mid.",
                       "relative_depth_denominator": "Total displayed shares across both sides and all selected levels."})
        params["sampling"] = sampling
        if sampling == "messages":
            params.update({"step_seconds": None, "window_seconds": None, "lag_spacing_seconds": None,
                           "max_age_seconds": None, "window_messages": window_messages,
                           "lag_spacing_messages": lag_spacing_messages,
                           "quote_policy": "Exclude missing, locked, crossed and halted quotes. Every sampled snapshot is taken at its own message time.",
                           "clock_policy": "Keep every message in original row order, including equal timestamps. Evaluation remains on a regular clock grid and includes all messages at the endpoint.",
                           "window_policy": "Use the last window_messages original rows at or before each endpoint, before exclusions. Require a complete window inside the selected period.",
                           "delay_policy": "No combined observation crosses an invalid message, including messages with tied timestamps. Snapshot age does not apply to message sampling."})
        if embedding == "collaborator":
            params.update({"representation_source": "https://github.com/RobertGBoyce/PP_LOB/blob/main/LOB/helper.py",
                           "representation_coordinates": "Ask minus best ask (levels 2..L), ask minus best bid (1..L), bid minus best bid (2..L), bid minus best ask (1..L), cumulative ask volumes, cumulative bid volumes, and logarithms of both cumulative volumes.",
                           "price_coordinates": "Differences in original LOBSTER integer price units (one ten thousandth of a dollar).",
                           "log_volume_floor": 1e-8,
                           "missing_depth_policy": "Every selected price must be present and every size finite and nonnegative. Invalid quotes and zero total volume are excluded.",
                           "representation_compatibility": "Generalises the repository's ten levels to L levels with 8L-2 coordinates. Matches its feature and historical scaling formulas on valid aligned rows. Uses the app's headerless loader without the repository's first and last row omissions."})
        for endpoint in endpoints:
            timestamp = day + pd.Timedelta(int(endpoint), unit="ns")
            hi = int(np.searchsorted(grid_ns, endpoint, side="right"))
            stream.advance(hi)
            if sampling == "messages":
                lo = max(0, hi - window_messages)
                left_ns = int(grid_ns[lo]) if hi else max(start_ns, int(event_ns[0]))
                incomplete = hi < window_messages
            else:
                left_ns = int(endpoint) - window_ns
                lo = int(np.searchsorted(grid_ns, left_ns, side="left"))
                incomplete = left_ns < max(start_ns, int(event_ns[0]))
            positions = np.arange(min(lo + lag_span, hi), hi, dtype=np.int64)
            valid_delays = ~(bad_interval[positions] | stale_interval[positions])
            for lag in range(lag_count):
                valid_delays &= base_valid[positions - lag * lag_steps]
            retained = positions[valid_delays]
            if incomplete:
                record = patnaik_pearson(np.empty((0, coordinate_count)), estimator_scaling)
                record["status"] = "incomplete_window"
                retained = np.empty(0, dtype=np.int64)
            else:
                points = (np.concatenate([stream.take(retained - lag * lag_steps)
                                          for lag in range(lag_count)], axis=1)
                          if len(retained) else np.empty((0, coordinate_count)))
                record = patnaik_pearson(points, estimator_scaling)
                if not len(retained) and warmup[lo:hi].any():
                    record["status"] = "standardisation_warmup"
            row = int(np.searchsorted(event_ns, endpoint, side="right") - 1)
            record.update({"time": timestamp.isoformat(), "seconds": int(endpoint) / NANOSECONDS,
                           "window_start_time": (day + pd.Timedelta(left_ns, unit="ns")).isoformat(),
                           "window_start_seconds": left_ns / NANOSECONDS,
                           "source_row": row, "source_time": pd.Timestamp(data.messages["timestamp"].iloc[row]).isoformat(),
                           "source_seconds": int(event_ns[row]) / NANOSECONDS,
                           "candidate_points": int(len(positions)), "excluded_points": int(len(positions) - len(retained)),
                           "window_grid_points": hi - lo,
                           "standardisation_warmup_points": int(warmup[lo:hi].sum()),
                           "invalid_grid_points": int((~base_valid[lo:hi]).sum()),
                           "sources_before_selected_start": int((~selected_sources[lo:hi]).sum()),
                           "invalid_interval_vectors": int(bad_interval[positions].sum()),
                           "stale_interval_vectors": int(stale_interval[positions].sum()),
                           "first_point_time": grid.index[retained[0]].isoformat() if len(retained) else None,
                           "last_point_time": grid.index[retained[-1]].isoformat() if len(retained) else None,
                           "earliest_constituent_time": grid.index[retained[0] - lag_span].isoformat() if len(retained) else None,
                           "earliest_source_row": int(source_rows[retained[0] - lag_span]) if len(retained) else None,
                           "latest_source_row": int(source_rows[retained[-1]]) if len(retained) else None})
            records.append(record)
        # Include trailing observations in the historical exclusion count,
        # even when no evaluation endpoint occurs at the end of the selection.
        stream.advance(len(grid))
        if scaling == "ewm" or custom_historical:
            params["ewm_warmup_observations"] = int(warmup.sum())
    successes = [r for r in records if r["status"] == "ok"]
    values = [r["dimension"] for r in successes]
    latest = successes[-1] if successes else None
    status_counts = {name: sum(r["status"] == name for r in records)
                     for name in sorted({r["status"] for r in records})}
    params.update({"first_evaluation_time": records[0]["time"], "last_evaluation_time": records[-1]["time"],
                   "svd_work_estimate": int(work), "coordinate_order": feature_names,
                   "estimated_additional_memory_bytes": int(memory_estimate),
                   "calculation_limits": {"svd_work": MAX_DIMENSION_SVD_WORK,
                                          "estimated_additional_memory_bytes": MAX_DIMENSION_MEMORY_BYTES,
                                          "evaluation_points": MAX_DIMENSION_ENDPOINTS}})
    return {
        "method": "Patnaik Pearson dimension", "parameters": params,
        "counts": {"evaluation_points": len(records), "successful_points": len(successes),
                   "undefined_points": len(records) - len(successes),
                   **{f"status_{name}": count for name, count in status_counts.items()}},
        "metrics": {"last_dimension": latest["dimension"] if latest else None,
                    "last_normalized_dimension": latest["normalized_dimension"] if latest else None,
                    "minimum_dimension": min(values) if values else None,
                    "maximum_dimension": max(values) if values else None,
                    "mean_dimension": float(np.mean(values)) if values else None,
                    "last_point_count": latest["point_count"] if latest else None,
                    "coordinate_count": coordinate_count},
        "series": records, "feature_names": feature_names,
        "notes": ["This is the Patnaik Pearson spectral measure of the specified point cloud. It is not a general guarantee of a manifold's true geometric dimension.",
                  "Snapshot mode measures geometry across occupied levels. History mode measures variation between vectors representing snapshots at different times. Their values describe different objects.",
                  "For a nonconstant cloud with N points and d coordinates, dimension lies between 1 and min(N-1, d). Few points impose a low ceiling.",
                  "A single point or identical point cloud has undefined dimension, shown as a gap rather than zero.",
                  "The measure uses singular values, not covariance eigenvalues. Small nonzero singular values are retained even when numerical rank treats them as negligible.",
                  "Changing embeddings, coordinate scales, patch width, lag count, or sample frequency changes the measured object. Within cloud standardization changes the metric separately at each endpoint.",
                  "Overlapping windows, overlapping level patches, and serial dependence do not supply independent observations. No confidence intervals or significance claims are made.",
                  "Relative depth vectors sum to one. This imposes an affine constraint. Repeated observations are retained according to the selected clock or message sampling.",
                  "Early history endpoints without the complete requested past window remain undefined. All delay constituents stay within that window and the selected interval."]
    }
