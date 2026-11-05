"""In memory checks for chronological LOBSTER research calculations.

Run from the project folder with Python's bytecode generation disabled.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from lobster_exploration import LobsterData
from lobster_app.research import (
    MAX_GRID_POINTS, _binned_response, patnaik_pearson, run_dimension,
    run_pca, run_response, sample_grid, _dimension_history_features,
    _historical_standardize, _message_grid, MAX_DIMENSION_ENDPOINTS,
    MAX_DIMENSION_SVD_WORK, _dimension_memory_estimate, rolling_realised_volatility,
    FEATURE_GROUPS, TOM_FEATURES, feature_coordinate_count, _selected_history_features,
    _historical_standardize_chunk, _HistoryFeatureStream, _feature_chunk_rows,
)


def make_data(times=None, bid=None, ask=None, midquotes=None):
    """Create aligned synthetic LOBSTER observations entirely in memory.

    Parameters
    ----------
    times : array_like, optional
        Sorted exchange clock seconds. Defaults to integers zero through sixty.
    bid, ask : array_like, optional
        Displayed size matrices with matching observation and level dimensions.
    midquotes : array_like, optional
        Positive midquotes corresponding to the supplied event times.

    Returns
    -------
    LobsterData
        Synthetic messages, order book, and quote measurements. The source paths
        identify imaginary inputs and are never opened or written.
    """
    times = np.arange(61, dtype=float) if times is None else np.asarray(times, dtype=float)
    n = len(times)
    t = np.arange(n, dtype=float)
    bid = (np.column_stack((100 + 2 * t, 200 + 4 * t)) if bid is None
           else np.asarray(bid, dtype=float))
    ask = (np.column_stack((100 + 3 * t, 50 + t)) if ask is None
           else np.asarray(ask, dtype=float))
    midquotes = (100 * np.exp(.0001 * times) if midquotes is None
                 else np.asarray(midquotes, dtype=float))
    levels = bid.shape[1]
    ns = np.rint(times * 1_000_000_000).astype(np.int64)
    timestamps = pd.Timestamp("2026-01-02") + pd.to_timedelta(ns, unit="ns")
    messages = pd.DataFrame({"time_ns": ns, "timestamp": timestamps,
                             "event_type": np.ones(n, dtype=int)})
    book = pd.DataFrame()
    for level in range(1, levels + 1):
        book[f"bid_price_{level}"] = midquotes - .01 * level
        book[f"bid_size_{level}"] = bid[:, level - 1]
        book[f"ask_price_{level}"] = midquotes + .01 * level
        book[f"ask_size_{level}"] = ask[:, level - 1]
    features = pd.DataFrame({"midquote": midquotes, "valid_quote": True,
                             "trading_halted": False, "spread": .02})
    return LobsterData(messages, book, features, pd.Series(dtype=int),
                       Path("synthetic_message_2.csv"), Path("synthetic_orderbook_2.csv"),
                       levels, False)


class SamplingTests(unittest.TestCase):
    def test_final_original_row_wins_at_tied_timestamps(self):
        data = make_data(times=[0, 1, 1, 2, 3])
        grid = sample_grid(data, 0, 3, 1, 2)
        self.assertEqual(grid["row"].tolist(), [0, 2, 3, 4])

    def test_invalid_final_tie_does_not_use_earlier_valid_snapshot(self):
        data = make_data(times=[0, 1, 1, 2, 3])
        data.features.loc[2, "valid_quote"] = False
        data.features.loc[2, "midquote"] = np.nan
        grid = sample_grid(data, 0, 3, .5, 2)
        invalid = grid.loc[grid["timestamp_ns"].between(1_000_000_000, 1_500_000_000)]
        self.assertTrue((invalid["row"] == 2).all())
        self.assertTrue((~invalid["valid"]).all())
        self.assertTrue(invalid["midquote"].isna().all())

    def test_grid_clips_loaded_coverage_and_stays_midnight_anchored(self):
        grid = sample_grid(make_data(times=[10.2, 12.3, 20.3]), 0, 30, 1, 2)
        self.assertEqual(grid["timestamp_ns"].iloc[0], 11_000_000_000)
        self.assertEqual(grid["timestamp_ns"].iloc[-1], 20_000_000_000)
        self.assertEqual(len(grid), 10)

    def test_age_limit_is_exact_and_no_future_snapshot_is_used(self):
        grid = sample_grid(make_data(times=[0, 10, 20]), 0, 20, 1, 2, max_age_seconds=2)
        self.assertTrue(grid["valid"].iloc[2])
        self.assertFalse(grid["valid"].iloc[3])
        self.assertEqual(grid["row"].iloc[9], 0)
        self.assertTrue(grid["valid"].iloc[10])
        unlimited = sample_grid(make_data(times=[0, 10, 20]), 0, 20, 1, 2, max_age_seconds=None)
        self.assertTrue(unlimited["valid"].all())

    def test_missing_depth_policy_distinguishes_zero_and_nonzero_size(self):
        data = make_data()
        data.book.loc[0, ["bid_price_2", "bid_size_2"]] = [np.nan, 0]
        data.book.loc[1, "bid_price_2"] = np.nan
        grid = sample_grid(data, 0, 60, 1, 2)
        self.assertTrue(grid["valid"].iloc[0])
        self.assertEqual(grid["bid_size_2"].iloc[0], 0)
        self.assertFalse(grid["valid"].iloc[1])
        self.assertTrue(grid["invalid_depth"].iloc[1])

    def test_locked_and_halted_quotes_are_excluded(self):
        data = make_data()
        data.features.loc[1, "spread"] = 0
        data.features.loc[2, "trading_halted"] = True
        grid = sample_grid(data, 0, 60, 1, 2)
        self.assertFalse(grid["valid"].iloc[1])
        self.assertFalse(grid["valid"].iloc[2])

    def test_zero_depth_and_invalid_parameters_are_rejected(self):
        data = make_data()
        for side in ("bid", "ask"):
            for level in (1, 2):
                data.book.loc[0, f"{side}_size_{level}"] = 0
        grid = sample_grid(data, 0, 60, 1, 2)
        self.assertFalse(grid["valid"].iloc[0])
        with self.assertRaisesRegex(ValueError, "no clock grid points"):
            sample_grid(data, 61, 80, 1, 2)
        with self.assertRaises(ValueError):
            sample_grid(data, 0, 60, 0, 2)
        with self.assertRaises(ValueError):
            sample_grid(data, 0, 60, 1, 3)
        with self.assertRaises(ValueError):
            sample_grid(data, 0, 60, 60 / (MAX_GRID_POINTS + 2), 2)


class RealisedVolatilityTests(unittest.TestCase):
    def test_matches_sum_of_squared_log_returns_without_demeaning(self):
        returns = np.array([.001, -.002, .003, .001, .001, .001])
        prices = 100 * np.exp(np.r_[0, np.cumsum(returns)])
        result = rolling_realised_volatility(make_data(times=np.arange(7), midquotes=prices),
                                             0, 6, window_seconds=3)
        expected = [10_000 * np.sqrt(np.sum(returns[i - 3:i] ** 2)) for i in range(3, 7)]
        self.assertTrue(result.realised_volatility_bps.iloc[:3].isna().all())
        np.testing.assert_allclose(result.realised_volatility_bps.iloc[3:], expected, rtol=1e-11)
        self.assertEqual(result.valid_returns_in_window.iloc[-1], 3)
        self.assertFalse(result.attrs["annualised"])
        self.assertGreater(result.realised_volatility_bps.iloc[-1], 0)

    def test_constant_quotes_have_zero_volatility_after_a_complete_window(self):
        data = make_data(times=np.arange(10), midquotes=np.full(10, 100.))
        result = rolling_realised_volatility(data, 0, 9, window_seconds=3)
        self.assertTrue(result.realised_volatility_bps.iloc[:3].isna().all())
        np.testing.assert_array_equal(result.realised_volatility_bps.iloc[3:], 0)

    def test_invalid_quotes_leave_a_gap_until_the_window_is_valid_again(self):
        data = make_data(times=np.arange(7))
        data.features.loc[2, "valid_quote"] = False
        result = rolling_realised_volatility(data, 0, 6, window_seconds=2)
        self.assertTrue(result.realised_volatility_bps.iloc[2:5].isna().all())
        self.assertFalse(result.valid_return.iloc[2])
        self.assertFalse(result.valid_return.iloc[3])
        self.assertTrue(np.isfinite(result.realised_volatility_bps.iloc[5]))

    def test_halts_between_clock_samples_are_not_bridged(self):
        times = np.sort(np.r_[np.arange(7), 2.3, 2.6])
        data = make_data(times=times)
        data.features.loc[np.flatnonzero(times == 2.3), "trading_halted"] = True
        result = rolling_realised_volatility(data, 0, 6, window_seconds=1)
        self.assertTrue(np.isfinite(result.midquote.iloc[2:4]).all())
        self.assertTrue(np.isnan(result.realised_volatility_bps.iloc[3]))
        self.assertTrue(np.isfinite(result.realised_volatility_bps.iloc[4]))

    def test_stale_periods_between_fresh_endpoints_are_not_bridged(self):
        data = make_data(times=[0, .75, 1, 1.5, 2])
        result = rolling_realised_volatility(data, 0, 2, window_seconds=1, max_age_seconds=.5)
        self.assertTrue(np.isfinite(result.midquote).all())
        self.assertFalse(result.valid_return.iloc[1])
        self.assertTrue(result.valid_return.iloc[2])

    def test_selected_start_and_future_changes_do_not_leak_into_estimates(self):
        data = make_data(times=[0, 2, 3, 4, 5, 6])
        result = rolling_realised_volatility(data, .4, 6, window_seconds=2)
        self.assertTrue(np.isnan(result.midquote.iloc[0]))
        self.assertTrue(result.realised_volatility_bps.iloc[:3].isna().all())
        self.assertTrue(np.isfinite(result.realised_volatility_bps.iloc[3]))
        data.features.loc[data.messages.time_ns > 4_000_000_000, "midquote"] *= 2
        other = rolling_realised_volatility(data, .4, 6, window_seconds=2)
        pd.testing.assert_frame_equal(result.iloc[:4], other.iloc[:4])

    def test_final_tied_quote_is_used_and_volumes_do_not_enter_calculation(self):
        data = make_data(times=[0, 1, 1, 2], midquotes=[100, 101, 102, 103])
        result = rolling_realised_volatility(data, 0, 2, window_seconds=1)
        self.assertAlmostEqual(result.realised_volatility_bps.iloc[1], 10_000 * np.log(102 / 100))
        data.book.loc[:, ["bid_size_1", "ask_size_1"]] = 0
        data.book.loc[:, "ask_price_2"] = np.nan
        other = rolling_realised_volatility(data, 0, 2, window_seconds=1)
        pd.testing.assert_frame_equal(result, other)

    def test_window_validation_and_short_selections(self):
        data = make_data()
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            rolling_realised_volatility(data, 0, 60, window_seconds=3, step_seconds=2)
        for window in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                rolling_realised_volatility(data, 0, 60, window_seconds=window)
        result = rolling_realised_volatility(data, 0, 0, window_seconds=60)
        self.assertEqual(len(result), 1)
        self.assertTrue(result.realised_volatility_bps.isna().all())


class PCATests(unittest.TestCase):
    def test_known_rank_one_matrix_has_one_variance_component(self):
        result = run_pca(make_data(), 0, 60, step_seconds=1, levels=2,
                         transform="raw", standardize=False)
        self.assertAlmostEqual(result["explained_variance"][0], 1, places=12)
        self.assertEqual(result["metrics"]["components_for_90_percent"], 1)
        self.assertLess(result["test_reconstruction_rmse"][1], 1e-4)
        self.assertEqual(np.asarray(result["components"]).shape, (4, 4))
        self.assertEqual(np.asarray(result["loadings"]).shape, (4, 4))
        json.dumps(result, allow_nan=False)

    def test_evaluation_changes_do_not_change_training_fit(self):
        data = make_data()
        original = run_pca(data, 0, 60, step_seconds=1, levels=2)
        split = int(np.floor(61 * .7))
        data.book.loc[split:, "bid_size_1"] *= 100
        changed = run_pca(data, 0, 60, step_seconds=1, levels=2)
        np.testing.assert_array_equal(original["components"], changed["components"])
        np.testing.assert_array_equal(original["training_mean"], changed["training_mean"])
        np.testing.assert_array_equal(original["training_scale"], changed["training_scale"])
        self.assertNotEqual(original["test_reconstruction_rmse"][0], changed["test_reconstruction_rmse"][0])

    def test_training_constant_feature_is_dropped_even_if_evaluation_varies(self):
        data = make_data()
        split = int(np.floor(61 * .7))
        data.book.loc[:split - 1, "bid_size_2"] = 200
        result = run_pca(data, 0, 60, step_seconds=1, levels=2)
        self.assertIn("bid_size_2", result["dropped_feature_names"])
        self.assertNotIn("bid_size_2", result["feature_names"])

    def test_split_is_chosen_before_invalid_samples_are_removed(self):
        data = make_data()
        data.features.loc[:20, "valid_quote"] = False
        result = run_pca(data, 0, 60, step_seconds=1, levels=2)
        self.assertEqual(result["parameters"]["split_seconds"], 42)
        self.assertEqual(result["counts"]["train_samples"], 21)

    def test_all_constants_and_insufficient_samples_fail_clearly(self):
        data = make_data(bid=np.full((61, 2), 100), ask=np.full((61, 2), 100))
        with self.assertRaisesRegex(ValueError, "constant"):
            run_pca(data, 0, 60, step_seconds=1, levels=2)
        with self.assertRaisesRegex(ValueError, "three valid training"):
            run_pca(make_data(), 0, 3, step_seconds=1, levels=2)

    def test_input_data_are_not_mutated(self):
        data = make_data()
        before_book = data.book.copy(deep=True)
        before_features = data.features.copy(deep=True)
        run_pca(data, 0, 60, step_seconds=1, levels=2)
        pd.testing.assert_frame_equal(data.book, before_book)
        pd.testing.assert_frame_equal(data.features, before_features)


class ResponseTests(unittest.TestCase):
    def test_known_log_response_and_strict_split_boundary(self):
        result = run_response(make_data(), 0, 60, step_seconds=1, levels=2, horizon_seconds=5)
        self.assertEqual(result["parameters"]["split_seconds"], 42)
        self.assertEqual(result["counts"]["train_samples"], 37)
        self.assertEqual(result["counts"]["test_samples"], 14)
        self.assertEqual(result["counts"]["crossing_split_rows"], 5)
        self.assertLess(pd.Timestamp(result["train_last_target_time"]), pd.Timestamp(result["parameters"]["split_time"]))
        self.assertGreaterEqual(pd.Timestamp(result["test_start_time"]), pd.Timestamp(result["parameters"]["split_time"]))
        self.assertAlmostEqual(result["metrics"]["train_mean_response_bps"], 5, places=9)
        self.assertAlmostEqual(result["metrics"]["test_mean_response_bps"], 5, places=9)
        self.assertEqual(sum(result["train_counts"]), result["counts"]["train_samples"])
        json.dumps(result, allow_nan=False)

    def test_halt_between_valid_grid_endpoints_excludes_target(self):
        times = np.sort(np.r_[np.arange(61), 2.4, 2.6])
        data = make_data(times=times)
        halted = np.flatnonzero(times == 2.4)[0]
        data.features.loc[halted, ["trading_halted", "valid_quote"]] = [True, False]
        result = run_response(data, 0, 60, step_seconds=1, levels=2, horizon_seconds=1)
        self.assertEqual(result["counts"]["invalid_endpoint_rows"], 0)
        self.assertEqual(result["counts"]["invalid_interval_rows"], 1)
        self.assertEqual(result["counts"]["train_samples"], 40)

    def test_zero_duration_invalid_snapshot_at_tie_does_not_create_interval(self):
        times = np.sort(np.r_[np.arange(61), 2, 2])
        data = make_data(times=times)
        tied = np.flatnonzero(times == 2)
        data.features.loc[tied[0], "valid_quote"] = False
        result = run_response(data, 0, 60, step_seconds=1, levels=2, horizon_seconds=1)
        self.assertEqual(result["counts"]["invalid_interval_rows"], 0)
        self.assertEqual(result["counts"]["invalid_endpoint_rows"], 0)

    def test_stale_interval_between_fresh_endpoints_excludes_target(self):
        times = np.sort(np.r_[np.arange(0, 21, 2), np.arange(21, 61)])
        data = make_data(times=times)
        result = run_response(data, 0, 60, step_seconds=2, levels=2,
                              horizon_seconds=2, max_age_seconds=1)
        self.assertEqual(result["counts"]["invalid_endpoint_rows"], 0)
        self.assertEqual(result["counts"]["stale_interval_rows"], 10)

    def test_constant_imbalance_uses_training_mean_and_zero_variance_baseline(self):
        data = make_data(bid=np.full((61, 2), 100), ask=np.full((61, 2), 100),
                         midquotes=np.full(61, 100))
        result = run_response(data, 0, 60, step_seconds=1, levels=2, horizon_seconds=5)
        self.assertFalse(result["metrics"]["slope_identified"])
        self.assertEqual(result["slope"], 0)
        self.assertEqual(result["intercept"], 0)
        self.assertEqual(result["metrics"]["test_rmse_bps"], 0)
        self.assertIsNone(result["metrics"]["test_r2_vs_train_mean"])
        json.dumps(result, allow_nan=False)

    def test_evaluation_changes_do_not_change_training_regression(self):
        data = make_data()
        original = run_response(data, 0, 60, step_seconds=1, levels=2, horizon_seconds=5)
        data.features.loc[42:, "midquote"] *= 1.1
        data.book.loc[42:, "bid_size_1"] *= 100
        changed = run_response(data, 0, 60, step_seconds=1, levels=2, horizon_seconds=5)
        self.assertEqual(original["slope"], changed["slope"])
        self.assertEqual(original["intercept"], changed["intercept"])
        self.assertEqual(original["train_means"], changed["train_means"])

    def test_response_bin_endpoints_and_empty_bins(self):
        means, medians, counts = _binned_response(np.array([-1., 1.]), np.array([-2., 3.]), np.linspace(-1, 1, 5))
        self.assertEqual(counts, [1, 0, 0, 1])
        self.assertEqual(means, [-2., None, None, 3.])
        self.assertEqual(medians, means)

    def test_horizon_validation_and_incomplete_window(self):
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            run_response(make_data(), 0, 60, step_seconds=2, levels=2, horizon_seconds=3)
        with self.assertRaisesRegex(ValueError, "shorter than the response horizon"):
            run_response(make_data(), 0, 60, step_seconds=1, levels=2, horizon_seconds=70)
        with self.assertRaisesRegex(ValueError, "three valid training targets"):
            run_response(make_data(), 0, 10, step_seconds=1, levels=2, horizon_seconds=5)


class PatnaikPearsonTests(unittest.TestCase):
    def test_uses_singular_values_instead_of_covariance_eigenvalues(self):
        points = np.array([[3., 0.], [-3., 0.], [0., 1.], [0., -1.]])
        result = patnaik_pearson(points)
        self.assertAlmostEqual(result["dimension"], 1.6, places=13)
        self.assertNotAlmostEqual(result["dimension"], 100 / 82, places=5)
        self.assertAlmostEqual(result["normalized_dimension"], .8, places=13)
        self.assertEqual(result["rank_bound"], 2)
        json.dumps(result, allow_nan=False)

    def test_rank_one_and_equal_axes_have_expected_dimensions(self):
        line = np.array([[-1., -2.], [0., 0.], [1., 2.]])
        plane = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        self.assertAlmostEqual(patnaik_pearson(line)["dimension"], 1, places=13)
        self.assertAlmostEqual(patnaik_pearson(plane)["dimension"], 2, places=13)

    def test_translation_rotation_and_uniform_scale_invariance(self):
        points = np.array([[3., 0.], [-3., 0.], [0., 1.], [0., -1.]])
        angle = .7
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for changed in (points + [100, -1000], points @ rotation,
                        points * 1e-250, points * 1e250):
            self.assertAlmostEqual(patnaik_pearson(changed)["dimension"], 1.6, places=12)

    def test_identical_points_and_single_vector_are_undefined(self):
        constant = patnaik_pearson(np.tile([.1, .2, .3], (100, 1)))
        single = patnaik_pearson([[2., 5.]])
        self.assertEqual(constant["status"], "zero_variation")
        self.assertEqual(single["status"], "insufficient_points")
        self.assertIsNone(constant["dimension"])
        self.assertIsNone(single["dimension"])
        json.dumps([constant, single], allow_nan=False)

    def test_standardization_preserves_extremely_small_varying_coordinates(self):
        points = np.array([[1., 0.], [-1., 0.], [0., .5], [0., -.5]]) * [1e300, 1e-300]
        result = patnaik_pearson(points, scaling="standardize")
        self.assertAlmostEqual(result["dimension"], 2, places=12)
        self.assertEqual(result["constant_coordinates"], [])

    def test_numerical_rank_does_not_remove_tiny_singular_values(self):
        points = np.array([[1., 0.], [-1., 0.], [0., 1e-17], [0., -1e-17]])
        result = patnaik_pearson(points)
        self.assertEqual(result["numerical_rank"], 1)
        self.assertEqual(len(result["singular_values"]), 2)
        self.assertGreater(result["singular_values"][1], 0)

    def test_constant_coordinates_remain_in_normalization_and_invalid_input_fails(self):
        points = np.array([[1., 5.], [2., 5.], [3., 5.]])
        result = patnaik_pearson(points, scaling="standardize")
        self.assertEqual(result["coordinate_count"], 2)
        self.assertEqual(result["constant_coordinates"], [1])
        self.assertAlmostEqual(result["normalized_dimension"], .5)
        with self.assertRaisesRegex(ValueError, "finite"):
            patnaik_pearson([[1., np.nan], [2., 3.]])
        with self.assertRaisesRegex(ValueError, "matrix"):
            patnaik_pearson([1., 2.])


class DimensionSweepTests(unittest.TestCase):
    def test_all_138_features_have_unique_names_and_correct_formulas(self):
        bid = np.arange(1, 611).reshape(61, 10).astype(float)
        ask = bid * 2 + 7
        data = make_data(bid=bid, ask=ask)
        grid = _message_grid(data, 0, 60 * 10**9, 10)
        values, names, historical = _selected_history_features(data, grid, 10, TOM_FEATURES)
        self.assertEqual(values.shape, (61, 138))
        self.assertEqual(len(set(names)), 138)
        self.assertEqual(historical.sum(), 118)
        self.assertEqual(feature_coordinate_count(TOM_FEATURES, 10), 138)
        b, a = bid.cumsum(axis=1), ask.cumsum(axis=1)
        np.testing.assert_allclose(values[:, 78:88], (b-a)/(b+a+1e-6))
        np.testing.assert_allclose(values[:, 98:108], (np.log(b)-np.log(a))/(np.log(b)+np.log(a)+1e-6))
        np.testing.assert_allclose(values[:, 118:128], b + a)
        np.testing.assert_allclose(values[:, 128:138], np.log(b + a))
        self.assertTrue(historical[118:].all())
        for key in FEATURE_GROUPS:
            for choice in ("original", "ewm", "both"):
                selection = {key: choice}
                matrix, labels, mask = _selected_history_features(data, grid, 10, selection)
                self.assertEqual(matrix.shape[1], feature_coordinate_count(selection, 10))
                self.assertEqual(len(set(labels)), len(labels))
                self.assertEqual(mask.sum(), matrix.shape[1] // 2 if choice == "both" else matrix.shape[1] if choice == "ewm" else 0)

    def test_total_volume_groups_are_independent_and_log_the_combined_volume(self):
        bid = np.array([[0., 2.], [3., 5.], [1e-12, 7.]])
        ask = np.array([[0., 4.], [7., 11.], [2e-12, 13.]])
        data = make_data(times=[0, 1, 2], bid=bid, ask=ask)
        for levels in (1, 2):
            grid = _message_grid(data, 0, 2 * 10**9, levels)
            total = np.column_stack((bid[:, 0] + ask[:, 0], bid.sum(axis=1) + ask.sum(axis=1)))[:, :levels]
            for key, expected in (("total_volume", total),
                                  ("log_total_volume", np.log(np.maximum(total, 1e-8)))):
                for choice in ("original", "ewm", "both"):
                    with self.subTest(levels=levels, key=key, choice=choice):
                        selection = {"total_volume": "omit", "log_total_volume": "omit", key: choice}
                        values, names, mask = _selected_history_features(data, grid, levels, selection)
                        np.testing.assert_allclose(values, np.tile(expected, (1, 2 if choice == "both" else 1)))
                        self.assertEqual(feature_coordinate_count(selection, levels), levels * (2 if choice == "both" else 1))
                        self.assertEqual(len(set(names)), values.shape[1])
                        np.testing.assert_array_equal(mask, [False] * levels + [True] * levels if choice == "both" else [choice == "ewm"] * levels)
            self.assertEqual(feature_coordinate_count(TOM_FEATURES, levels), 14 * levels - 2)

    def test_updated_tom_preset_runs_with_delays_and_exports_new_coordinates(self):
        t = np.arange(650)[:, None]
        data = make_data(times=np.arange(650) / 10,
                         bid=100 + (t * np.arange(1, 11)) % 73,
                         ask=200 + (t * np.arange(2, 12)) % 97)
        result = run_dimension(data, 0, 64.9, levels=10, sampling="messages",
                               window_messages=100, embedding="custom", features=TOM_FEATURES,
                               lag_count=2, lag_spacing_messages=1, evaluation_step_seconds=10)
        self.assertEqual(result["metrics"]["coordinate_count"], 276)
        self.assertEqual(result["parameters"]["ewm_warmup_observations"], 501)
        self.assertEqual(result["parameters"]["features"], TOM_FEATURES)
        self.assertEqual(result["parameters"]["log_total_volume_floor"], 1e-8)
        names = result["parameters"]["coordinate_order"]
        self.assertEqual(len(set(names)), 276)
        self.assertIn("total_volume_10_ewm_lag_1", names)
        self.assertIn("log_total_volume_10_ewm_lag_1", names)
        last = result["series"][-1]
        self.assertEqual(last["status"], "ok")
        self.assertEqual(last["point_count"], 99)
        self.assertAlmostEqual(last["normalized_dimension"], last["dimension"] / 276)
        json.dumps(result, allow_nan=False)

    def test_custom_78_features_match_previous_collaborator_calculation(self):
        args = dict(levels=2, step_seconds=1, window_seconds=10, evaluation_step_seconds=10, ewm_lookback=3)
        features = {key: "ewm" for key in list(FEATURE_GROUPS)[:3]}
        old = run_dimension(make_data(), 0, 60, embedding="collaborator", scaling="ewm", **args)
        new = run_dimension(make_data(), 0, 60, embedding="custom", features=features, **args)
        self.assertEqual(old["series"], new["series"])

    def test_mixed_original_and_standardised_features_are_not_scaled_twice(self):
        data = make_data()
        grid = _message_grid(data, 0, 60 * 10**9, 2)
        raw, _, mask = _selected_history_features(data, grid, 2, {"imbalance": "both"})
        scaled, _ = _historical_standardize(raw[:, mask], np.ones(61, dtype=bool), 3)
        expected = patnaik_pearson(np.column_stack((raw[50:61, :2], scaled[50:61])))
        result = run_dimension(data, 0, 60, levels=2, step_seconds=1, window_seconds=10,
                               evaluation_step_seconds=10, ewm_lookback=3, embedding="custom", features={"imbalance": "both"})
        self.assertAlmostEqual(result["series"][-1]["dimension"], expected["dimension"])
        data.book.loc[51:, "bid_size_1"] *= 100
        changed = run_dimension(data, 0, 60, levels=2, step_seconds=1, window_seconds=10,
                                evaluation_step_seconds=10, ewm_lookback=3, embedding="custom", features={"imbalance": "both"})
        self.assertEqual(result["series"][:-1], changed["series"][:-1])

    def test_custom_features_reject_empty_or_unknown_choices(self):
        for selection in ({}, {"imbalance": "omit"}, {"unknown": "original"}, {"imbalance": "invalid"}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                run_dimension(make_data(), 0, 60, levels=2, embedding="custom", features=selection)

    def test_burn_periods_match_trimmed_window_and_ignore_excluded_values(self):
        cases = [dict(mode="snapshot"), dict(sampling="clock"), dict(sampling="messages"),
                 dict(sampling="clock", scaling="ewm", ewm_lookback=3),
                 dict(sampling="messages", scaling="ewm", ewm_lookback=3)]
        for settings in cases:
            with self.subTest(settings=settings):
                data = make_data()
                args = dict(levels=2, step_seconds=1, evaluation_step_seconds=5,
                            window_seconds=5, window_messages=6, **settings)
                trimmed = run_dimension(data, 10, 50, **args)
                burned = run_dimension(data, 0, 60, burn_in_seconds=10,
                                       burn_out_seconds=10, **args)
                self.assertEqual(trimmed["series"], burned["series"])
                self.assertEqual(burned["parameters"]["start_seconds"], 10)
                self.assertEqual(burned["parameters"]["end_seconds"], 50)
                self.assertEqual(burned["parameters"]["selected_start_seconds"], 0)
                outside = (data.messages.time_ns < 10 * 10**9) | (data.messages.time_ns > 50 * 10**9)
                data.book.loc[outside, "bid_size_1"] *= 1000
                data.book.loc[outside, "ask_size_2"] *= 2000
                changed = run_dimension(data, 0, 60, burn_in_seconds=10,
                                        burn_out_seconds=10, **args)
                self.assertEqual(burned["series"], changed["series"])
                self.assertTrue(all(10 <= p["seconds"] <= 50 for p in burned["series"]))

    def test_invalid_burn_periods_are_rejected(self):
        for start, end in [(-1, 0), (0, -1), (30, 30), (61, 0), (0, float("nan")), (None, 0)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                run_dimension(make_data(), 0, 60, levels=2,
                              burn_in_seconds=start, burn_out_seconds=end)

    def test_complete_history_windows_and_delay_dimensions(self):
        result = run_dimension(make_data(), 0, 60, step_seconds=1, levels=2,
                               window_seconds=10, evaluation_step_seconds=10,
                               lag_count=3, lag_spacing_seconds=2)
        first = result["series"][0]
        at_twenty = next(r for r in result["series"] if r["seconds"] == 20)
        self.assertEqual(first["status"], "incomplete_window")
        self.assertIsNone(first["dimension"])
        self.assertEqual(at_twenty["point_count"], 7)
        self.assertEqual(at_twenty["coordinate_count"], 12)
        self.assertEqual(at_twenty["rank_bound"], 6)
        self.assertEqual(pd.Timestamp(at_twenty["earliest_constituent_time"]).second, 10)
        self.assertEqual(pd.Timestamp(at_twenty["first_point_time"]).second, 14)
        self.assertEqual(pd.Timestamp(at_twenty["last_point_time"]).second, 20)
        json.dumps(result, allow_nan=False)

    def test_future_book_changes_cannot_affect_earlier_history_estimate(self):
        data = make_data()
        args = dict(step_seconds=1, levels=2, window_seconds=10,
                    evaluation_step_seconds=10, lag_count=2, lag_spacing_seconds=1)
        original = run_dimension(data, 0, 60, **args)
        data.book.loc[21:, "bid_size_1"] *= 100
        data.book.loc[21:, "ask_price_2"] += 10
        changed = run_dimension(data, 0, 60, **args)
        self.assertEqual(original["series"][2], changed["series"][2])

    def test_history_does_not_take_source_observations_before_selected_start(self):
        data = make_data(times=np.r_[0., 10., np.arange(12., 31.)])
        result = run_dimension(data, 10.5, 30, step_seconds=1, levels=2,
                               window_seconds=3, evaluation_step_seconds=1)
        at_fourteen = next(r for r in result["series"] if r["seconds"] == 14)
        self.assertEqual(at_fourteen["sources_before_selected_start"], 1)
        earliest = at_fourteen["earliest_source_row"]
        self.assertGreaterEqual(data.messages.time_ns.iloc[earliest], 10_500_000_000)
        self.assertEqual(at_fourteen["point_count"], 3)

    def test_delay_vectors_do_not_bridge_a_halt_between_valid_clock_samples(self):
        times = np.sort(np.r_[np.arange(61), 5.4, 5.6])
        data = make_data(times=times)
        halted = np.flatnonzero(times == 5.4)[0]
        data.features.loc[halted, ["trading_halted", "valid_quote"]] = [True, False]
        result = run_dimension(data, 0, 60, step_seconds=1, levels=2,
                               window_seconds=10, evaluation_step_seconds=10,
                               lag_count=2, lag_spacing_seconds=2)
        at_ten = result["series"][1]
        self.assertEqual(at_ten["invalid_grid_points"], 0)
        self.assertEqual(at_ten["invalid_interval_vectors"], 2)
        self.assertEqual(at_ten["candidate_points"], 9)
        self.assertEqual(at_ten["point_count"], 7)

    def test_delay_vectors_do_not_bridge_intervening_zero_depth(self):
        times = np.sort(np.r_[np.arange(61), 5.4, 5.6])
        data = make_data(times=times)
        empty = np.flatnonzero(times == 5.4)[0]
        for col in ("bid_size_1", "bid_size_2", "ask_size_1", "ask_size_2"):
            data.book.loc[empty, col] = 0
        result = run_dimension(data, 0, 60, step_seconds=1, levels=2,
                               window_seconds=10, evaluation_step_seconds=10,
                               lag_count=2, lag_spacing_seconds=2)
        self.assertEqual(result["series"][1]["invalid_grid_points"], 0)
        self.assertEqual(result["series"][1]["invalid_interval_vectors"], 2)
        self.assertEqual(result["series"][1]["point_count"], 7)

    def test_invalid_clock_samples_are_not_compressed_before_lag_construction(self):
        data = make_data()
        data.features.loc[5, "valid_quote"] = False
        result = run_dimension(data, 0, 60, step_seconds=1, levels=2,
                               window_seconds=10, evaluation_step_seconds=10,
                               lag_count=2, lag_spacing_seconds=2)
        self.assertEqual(result["series"][1]["candidate_points"], 9)
        self.assertEqual(result["series"][1]["point_count"], 6)

    def test_price_embedding_requires_prices_even_for_zero_depth(self):
        data = make_data()
        data.book.loc[5, ["bid_price_2", "bid_size_2"]] = [np.nan, 0]
        args = dict(step_seconds=1, levels=2, window_seconds=10, evaluation_step_seconds=10)
        size_result = run_dimension(data, 0, 60, embedding="log_depth", **args)
        price_result = run_dimension(data, 0, 60, embedding="price_depth", **args)
        self.assertEqual(size_result["series"][1]["point_count"], 11)
        self.assertEqual(price_result["series"][1]["point_count"], 10)
        self.assertEqual(price_result["series"][1]["coordinate_count"], 8)

    def test_relative_depth_discards_common_volume_multipliers(self):
        multiplier = np.arange(1., 62.)[:, None]
        data = make_data(bid=multiplier * [1., 2.], ask=multiplier * [2., 3.])
        result = run_dimension(data, 0, 60, step_seconds=1, levels=2,
                               embedding="relative_depth", window_seconds=10,
                               evaluation_step_seconds=10)
        self.assertEqual(result["series"][1]["status"], "zero_variation")
        self.assertIsNone(result["series"][1]["dimension"])
        json.dumps(result, allow_nan=False)

    def test_snapshot_uses_level_points_and_final_tied_source_row(self):
        data = make_data(times=[0, 1, 1, 2, 3, 4],
                         bid=np.full((6, 4), 100), ask=np.full((6, 4), 200))
        result = run_dimension(data, 0, 4, levels=4, mode="snapshot",
                               evaluation_step_seconds=1)
        tied = result["series"][1]
        self.assertEqual(tied["source_row"], 2)
        self.assertEqual(tied["point_count"], 4)
        self.assertEqual(tied["coordinate_count"], 4)
        self.assertEqual(tied["rank_bound"], 3)
        self.assertAlmostEqual(tied["dimension"], 1, places=10)

    def test_invalid_snapshot_levels_remove_every_containing_patch(self):
        data = make_data(bid=np.full((61, 7), 100), ask=np.full((61, 7), 200))
        data.book.loc[10, ["bid_price_4", "bid_size_4"]] = [np.nan, 0]
        result = run_dimension(data, 0, 60, levels=7, mode="snapshot",
                               patch_width=3, evaluation_step_seconds=10)
        record = result["series"][1]
        self.assertEqual(record["valid_patch_start_levels"], [1, 5])
        self.assertEqual(record["invalid_levels"], 1)
        self.assertEqual(record["point_count"], 2)
        self.assertEqual(record["coordinate_count"], 12)
        self.assertEqual(record["excluded_points"], 3)
        self.assertEqual(record["rank_bound"], 1)

    def test_single_level_snapshot_is_undefined_and_stale_snapshot_is_a_gap(self):
        data = make_data(times=[0, 10, 20])
        result = run_dimension(data, 0, 20, levels=1, mode="snapshot",
                               evaluation_step_seconds=1, max_age_seconds=2)
        self.assertEqual(result["series"][0]["status"], "insufficient_points")
        self.assertEqual(result["series"][3]["status"], "stale_snapshot")
        self.assertIsNone(result["series"][3]["dimension"])
        self.assertEqual(result["counts"]["successful_points"], 0)
        json.dumps(result, allow_nan=False)

    def test_snapshot_source_before_selected_start_remains_an_explicit_gap(self):
        result = run_dimension(make_data(times=[0, 10, 20]), 5, 20, levels=2,
                               mode="snapshot", evaluation_step_seconds=5,
                               max_age_seconds=60)
        early, valid = result["series"][:2]
        self.assertEqual(early["seconds"], 5)
        self.assertEqual(early["status"], "source_before_selected_start")
        self.assertIsNone(early["dimension"])
        self.assertEqual(early["point_count"], 0)
        self.assertEqual(early["source_row"], 0)
        self.assertEqual(early["source_seconds"], 0)
        self.assertEqual(pd.Timestamp(early["source_time"]).second, 0)
        self.assertEqual(valid["seconds"], 10)
        self.assertEqual(valid["status"], "ok")
        self.assertEqual(valid["source_row"], 1)
        self.assertIn("window_policy", result["parameters"])
        json.dumps(result, allow_nan=False)

    def test_delay_vectors_do_not_bridge_stale_intervals(self):
        times = np.r_[np.arange(0, 21, 2), np.arange(21, 61)]
        result = run_dimension(make_data(times=times), 0, 60, levels=2,
                               step_seconds=2, evaluation_step_seconds=10,
                               window_seconds=10, lag_count=2,
                               lag_spacing_seconds=2, max_age_seconds=1)
        record = result["series"][1]
        self.assertEqual(record["invalid_grid_points"], 0)
        self.assertEqual(record["stale_interval_vectors"], 5)
        self.assertEqual(record["point_count"], 0)

    def test_alignment_and_endpoint_limits_are_explicit(self):
        data = make_data()
        with self.assertRaisesRegex(ValueError, "evaluation interval"):
            run_dimension(data, 0, 60, step_seconds=2, levels=2, evaluation_step_seconds=3)
        with self.assertRaisesRegex(ValueError, "Lag spacing"):
            run_dimension(data, 0, 60, step_seconds=2, levels=2,
                          evaluation_step_seconds=10, lag_count=2, lag_spacing_seconds=3)
        with self.assertRaisesRegex(ValueError, "evaluation times"):
            run_dimension(make_data(times=[0, MAX_DIMENSION_ENDPOINTS + 1]),
                          0, MAX_DIMENSION_ENDPOINTS + 1, levels=2,
                          mode="snapshot", evaluation_step_seconds=1)
        with self.assertRaisesRegex(ValueError, "patch width"):
            run_dimension(data, 0, 60, levels=2, mode="snapshot", patch_width=3)

    def test_excessive_svd_work_is_rejected_before_computation(self):
        data = make_data(times=np.arange(0, 10001, 100),
                         bid=np.full((101, 10), 100), ask=np.full((101, 10), 200))
        with patch("lobster_app.research.MAX_DIMENSION_SVD_WORK", 200_000_000):
            with patch("lobster_app.research.sample_grid", side_effect=AssertionError("Unexpected allocation")):
                with self.assertRaisesRegex(ValueError, "too much matrix work"):
                    run_dimension(data, 0, 10000, levels=10, step_seconds=1,
                                  window_seconds=5000, evaluation_step_seconds=20)

    def test_more_than_200_coordinates_produces_the_expected_pp(self):
        data = make_data(times=np.arange(700), bid=np.full((700, 10), 100) + np.arange(700)[:, None],
                         ask=np.full((700, 10), 200) + (np.arange(700)[:, None] % 13))
        result = run_dimension(data, 0, 699, levels=10, embedding="collaborator", sampling="messages",
                               lag_count=5, window_messages=500, evaluation_step_seconds=50)
        point = result["series"][-1]
        self.assertEqual(point["coordinate_count"], 390)
        self.assertEqual(point["point_count"], 496)
        grid = _message_grid(data, 0, 699_000_000_000, 10)
        base, _ = _dimension_history_features(data, grid, 10, "collaborator")
        expected_matrix = np.concatenate([base[155 - lag:651 - lag] for lag in range(5)], axis=1)
        expected = patnaik_pearson(expected_matrix)
        self.assertAlmostEqual(point["dimension"], expected["dimension"], places=12)
        self.assertGreater(result["parameters"]["svd_work_estimate"], 200_000_000)
        self.assertEqual(result["parameters"]["calculation_limits"]["svd_work"], MAX_DIMENSION_SVD_WORK)
        json.dumps(result, allow_nan=False)

    def test_more_than_1000_estimates_are_allowed(self):
        result = run_dimension(make_data(times=[0, 1200]), 0, 1200, levels=2,
                               mode="snapshot", evaluation_step_seconds=1, max_age_seconds=None)
        self.assertEqual(result["counts"]["evaluation_points"], 1201)
        self.assertEqual(result["counts"]["successful_points"], 1201)

    def test_memory_limit_is_checked_before_allocating_features(self):
        with patch("lobster_app.research.MAX_DIMENSION_MEMORY_BYTES", 1):
            with patch("lobster_app.research.sample_grid", side_effect=AssertionError("Unexpected allocation")):
                with self.assertRaisesRegex(ValueError, "additional memory"):
                    run_dimension(make_data(), 0, 60, levels=2, step_seconds=1,
                                  window_seconds=10, evaluation_step_seconds=10)

    def test_memory_estimate_accounts_for_history_matrices_and_results(self):
        small = _dimension_memory_estimate(1000, 78, 78, 100, 10)
        self.assertGreater(_dimension_memory_estimate(2000, 78, 78, 100, 10), small)
        self.assertGreater(_dimension_memory_estimate(1000, 78, 390, 100, 10), small)
        self.assertGreater(_dimension_memory_estimate(1000, 78, 78, 1000, 10), small)
        self.assertGreater(_dimension_memory_estimate(1000, 78, 78, 100, 100), small)


class StreamingDimensionTests(unittest.TestCase):
    def test_chunked_moments_match_full_history_across_invalid_boundaries(self):
        values = np.random.default_rng(21).normal(size=(110, 5)) * [1, 100, .1, 0, 1e-8]
        valid = np.ones(110, dtype=bool)
        valid[[0, 1, 7, 8, 9, 30, 63, 64, 65, 109]] = False
        values[~valid] = np.nan
        expected, expected_warmup = _historical_standardize(values, valid, 11)
        for chunk in (1, 7, 11, 32, 200):
            state, parts, warmups = None, [], []
            for start in range(0, len(values), chunk):
                part, warmup, state = _historical_standardize_chunk(
                    values[start:start + chunk], valid[start:start + chunk], 11, state)
                parts.append(part)
                warmups.append(warmup)
            with self.subTest(chunk=chunk):
                np.testing.assert_allclose(np.concatenate(parts), expected, rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(np.concatenate(warmups), expected_warmup)
                self.assertIsNone(state)

    def test_rolling_buffer_matches_eager_features_without_restarting_history(self):
        data = make_data()
        grid = _message_grid(data, 0, 60 * 10**9, 2)
        raw, names, mask = _selected_history_features(data, grid, 2, TOM_FEATURES)
        valid = np.ones(61, dtype=bool)
        valid[[0, 7, 8, 29, 58]] = False
        expected = raw.copy()
        expected[:, mask], warmup = _historical_standardize(raw[:, mask], valid, 3)
        for chunk in (1, 4, 13):
            sizes = []

            def builder(part):
                """Record allocation sizes while building the reference features.

                Parameters
                ----------
                part : pandas.DataFrame
                    Consecutive sampled rows.

                Returns
                -------
                tuple
                    Feature values, names and historical mask.
                """
                sizes.append(len(part))
                return _selected_history_features(data, part, 2, TOM_FEATURES)

            stream = _HistoryFeatureStream(grid, builder, valid, 5, chunk, 3)
            for end in (1, 7, 20, 21, 60, 61):
                stream.advance(end)
                positions = np.arange(max(0, end - 5), end)
                np.testing.assert_allclose(stream.take(positions), expected[positions], rtol=1e-12, atol=1e-12)
                self.assertEqual(stream.buffer.shape, (5, 26))
            self.assertLessEqual(max(sizes), chunk)
            self.assertEqual(stream.names, names)
            np.testing.assert_array_equal(stream.valid, valid & np.isfinite(expected).all(axis=1))
            np.testing.assert_array_equal(stream.warmup, warmup)
            with self.assertRaisesRegex(RuntimeError, "outside the rolling buffer"):
                stream.take(np.array([0]))

    def test_sweeps_are_independent_of_chunks_with_ties_gaps_and_delays(self):
        times = np.repeat(np.arange(50), 2).astype(float)
        times[60:] += 10
        data = make_data(times=times)
        data.features.loc[[0, 13, 14, 29, 30, 61, 75, 99], "valid_quote"] = False
        for sampling in ("clock", "messages"):
            args = dict(levels=2, sampling=sampling, step_seconds=1,
                        window_seconds=10, window_messages=12, lag_count=3,
                        lag_spacing_seconds=2, lag_spacing_messages=2,
                        evaluation_step_seconds=7, max_age_seconds=2,
                        ewm_lookback=3, embedding="custom", features=TOM_FEATURES)
            expected = run_dimension(data, 0, 59, **args)
            for chunk in (1, 7):
                with patch("lobster_app.research.DIMENSION_FEATURE_CHUNK_ROWS", chunk):
                    actual = run_dimension(data, 0, 59, **args)
                self.assertEqual(actual["parameters"]["ewm_warmup_observations"], expected["parameters"]["ewm_warmup_observations"])
                for a, b in zip(actual["series"], expected["series"]):
                    for key in a:
                        if key in ("dimension", "normalized_dimension", "singular_values") and a[key] is not None:
                            np.testing.assert_allclose(a[key], b[key], rtol=1e-11, atol=1e-11)
                        else:
                            self.assertEqual(a[key], b[key], key)

    def test_large_period_budget_retains_only_window_and_chunk_features(self):
        small = _dimension_memory_estimate(250_000, 138, 138, 5000, 391,
                                          history_samples=5000, levels=10, event_count=250_000)
        large = _dimension_memory_estimate(500_000, 138, 138, 5000, 391,
                                          history_samples=5000, levels=10, event_count=500_000)
        self.assertLess(small, 256 * 1024**2)
        # Extra duration adds only grid and interval metadata, not full feature matrices.
        self.assertEqual(large - small, 250_000 * (128 + 32 * 10 + 64))
        self.assertLessEqual(_feature_chunk_rows(5000) * 5000 * 8, 8 * 1024**2)


class CollaboratorRepresentationTests(unittest.TestCase):
    def test_explicit_coordinates_order_units_and_volume_floor(self):
        data = make_data(times=[0, 1], bid=[[100, 200], [0, 200]],
                         ask=[[100, 50], [0, 50]], midquotes=[100, 100])
        grid = _message_grid(data, 0, 1_000_000_000, 2)
        values, names = _dimension_history_features(data, grid, 2, "collaborator")
        expected = [100, 200, 300, -100, -200, -300, 100, 150, 100, 300,
                    np.log(100), np.log(150), np.log(100), np.log(300)]
        np.testing.assert_allclose(values[0], expected, rtol=1e-14)
        self.assertEqual(len(names), 14)
        self.assertEqual(names[0], "ask_2_minus_ask_1")
        self.assertEqual(names[-1], "log_cumulative_bid_size_2")
        self.assertEqual(values[1, 10], np.log(1e-8))
        self.assertEqual(values[1, 12], np.log(1e-8))

    def test_historical_scaling_matches_independent_recurrence(self):
        values = np.random.default_rng(12).normal(size=(620, 4)) * [1, 100, .1, 0]
        decay = .01 ** (1 / 501)
        mean, variance = values[0].copy(), np.zeros(4)
        expected = []
        for i, value in enumerate(values):
            residual = value - mean
            if i >= 501:
                expected.append(residual / np.sqrt(variance + 1e-6))
            mean = decay * mean + (1 - decay) * value
            variance = decay * variance + (1 - decay) * residual ** 2
        actual, warmup = _historical_standardize(values, np.ones(620, dtype=bool), 501)
        np.testing.assert_allclose(actual[501:], expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(warmup.sum(), 501)
        self.assertTrue(np.isnan(actual[:501]).all())
        changed = values.copy()
        changed[550:] *= 1000
        other, _ = _historical_standardize(changed, np.ones(620, dtype=bool), 501)
        np.testing.assert_array_equal(actual[:550], other[:550])

    def test_historical_scaling_restarts_after_invalid_sample(self):
        values = np.arange(18, dtype=float).reshape(9, 2)
        valid = np.ones(9, dtype=bool)
        valid[4] = False
        scaled, warmup = _historical_standardize(values, valid, 2)
        self.assertEqual(np.flatnonzero(warmup).tolist(), [0, 1, 5, 6])
        self.assertTrue(np.isnan(scaled[4]).all())
        isolated, _ = _historical_standardize(values[5:], np.ones(4, dtype=bool), 2)
        np.testing.assert_array_equal(scaled[5:], isolated)

    def test_message_windows_keep_ties_and_original_positions(self):
        data = make_data(times=[0, 1, 1, 1, 2, 3])
        result = run_dimension(data, 0, 3, levels=2, sampling="messages", window_messages=3,
                               evaluation_step_seconds=1, embedding="collaborator")
        at_one = result["series"][1]
        self.assertEqual(at_one["point_count"], 3)
        self.assertEqual(at_one["earliest_source_row"], 1)
        self.assertEqual(at_one["latest_source_row"], 3)
        self.assertEqual(at_one["window_start_seconds"], 1)
        self.assertEqual(result["series"][0]["status"], "incomplete_window")
        self.assertIsNone(result["parameters"]["step_seconds"])
        self.assertIsNone(result["parameters"]["window_seconds"])
        json.dumps(result, allow_nan=False)

    def test_message_lags_do_not_bridge_an_invalid_tied_event(self):
        data = make_data(times=[0, 1, 1, 1, 2, 3])
        data.features.loc[2, "valid_quote"] = False
        result = run_dimension(data, 0, 3, levels=2, sampling="messages", window_messages=4,
                               lag_count=2, lag_spacing_messages=2, evaluation_step_seconds=1)
        at_three = result["series"][-1]
        self.assertEqual(at_three["candidate_points"], 2)
        self.assertEqual(at_three["point_count"], 1)
        self.assertEqual(at_three["invalid_interval_vectors"], 1)

    def test_ten_levels_and_historical_scaling_in_full_run(self):
        times = np.arange(650) / 10
        t = np.arange(650)[:, None]
        data = make_data(times=times, bid=100 + (t * np.arange(1, 11)) % 73,
                         ask=200 + (t * np.arange(2, 12)) % 97)
        args = dict(levels=10, sampling="messages", window_messages=100,
                    embedding="collaborator", scaling="ewm", evaluation_step_seconds=10)
        result = run_dimension(data, 0, 64.9, **args)
        self.assertEqual(result["metrics"]["coordinate_count"], 78)
        self.assertEqual(result["parameters"]["ewm_warmup_observations"], 501)
        self.assertEqual(result["series"][3]["status"], "standardisation_warmup")
        self.assertEqual(result["series"][-1]["point_count"], 100)
        grid = _message_grid(data, 0, 64_900_000_000, 10)
        raw, _ = _dimension_history_features(data, grid, 10, "collaborator")
        scaled, _ = _historical_standardize(raw, np.ones(650, dtype=bool), 501)
        expected = patnaik_pearson(scaled[501:601])
        self.assertAlmostEqual(result["series"][-1]["dimension"], expected["dimension"], places=12)
        data.book.loc[601:, "bid_size_1"] *= 100
        changed = run_dimension(data, 0, 64.9, **args)
        self.assertEqual(result["series"][-1], changed["series"][-1])
        json.dumps(result, allow_nan=False)

    def test_selected_start_resets_history_and_missing_price_resets_again(self):
        data = make_data()
        data.book.loc[25, ["ask_price_2", "ask_size_2"]] = [np.nan, 0]
        result = run_dimension(data, 20, 60, levels=2, sampling="messages", window_messages=10,
                               embedding="collaborator", scaling="ewm", ewm_lookback=3,
                               evaluation_step_seconds=10)
        record = result["series"][1]
        self.assertEqual(record["standardisation_warmup_points"], 5)
        self.assertEqual(record["point_count"], 4)
        self.assertEqual(record["earliest_source_row"], 23)
        data.book.loc[:19, "ask_size_1"] *= 1e6
        changed = run_dimension(data, 20, 60, levels=2, sampling="messages", window_messages=10,
                                embedding="collaborator", scaling="ewm", ewm_lookback=3,
                                evaluation_step_seconds=10)
        self.assertEqual(result["series"], changed["series"])

    def test_clock_sampling_supports_collaborator_and_historical_scaling(self):
        result = run_dimension(make_data(), 0, 60, step_seconds=2, levels=2,
                               embedding="collaborator", scaling="ewm", ewm_lookback=3,
                               window_seconds=10, evaluation_step_seconds=10)
        self.assertEqual(result["series"][1]["point_count"], 3)
        self.assertEqual(result["series"][2]["point_count"], 6)
        self.assertEqual(result["parameters"]["sampling"], "clock")

    def test_new_settings_validation_and_limits(self):
        data = make_data()
        for settings in ({"sampling": "other"}, {"scaling": "ewm", "ewm_lookback": 0},
                         {"scaling": "ewm", "ewm_lookback": 2.5},
                         {"scaling": "ewm", "mode": "snapshot"},
                         {"sampling": "messages", "window_messages": 1},
                         {"sampling": "messages", "lag_spacing_messages": 1.5}):
            with self.assertRaises(ValueError):
                run_dimension(data, 0, 60, levels=2, **settings)
        with patch("lobster_app.research.MAX_MESSAGE_SAMPLES", 10):
            with self.assertRaisesRegex(ValueError, "Shorten the observation window"):
                run_dimension(data, 0, 60, levels=2, sampling="messages", window_messages=5)


class PriceChartTests(unittest.TestCase):
    def test_serialized_hover_rows_can_be_indexed_by_dash(self):
        from lobster_app.app import price_chart

        data = make_data(times=[0, 1, 1], bid=np.ones((3, 10)), ask=np.ones((3, 10)))
        data.features["timestamp"] = data.messages.timestamp
        data.features["best_bid"] = data.book.bid_price_1
        data.features["best_ask"] = data.book.ask_price_1
        for frame in (data.messages, data.book, data.features):
            frame.index = pd.Index([100, 200, 300])
        figure = json.loads(price_chart(data, "price").to_json())
        readout = figure["layout"]["meta"]["price_readout"]
        for trace in figure["data"]:
            # Dash reads trace.customdata[pointNumber] from the serialized
            # input, so a Plotly binary array object loses the original row.
            self.assertIsInstance(trace["customdata"], list)
            for point_number in range(3):
                self.assertEqual(trace["customdata"][point_number], readout["rows"][point_number])
        self.assertEqual(readout["rows"], [100, 200, 300])
        self.assertEqual(readout["prices"][2][0], data.book.loc[300, "bid_price_1"])
        self.assertEqual(readout["prices"][2][-1], data.book.loc[300, "ask_price_10"])


if __name__ == "__main__":
    unittest.main()
