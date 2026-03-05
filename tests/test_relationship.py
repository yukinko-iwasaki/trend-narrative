"""Unit tests for trend_narrative.relationship."""

import numpy as np
import pytest

from trend_narrative.relationship import (
    get_relationship_narrative,
    get_direction,
    get_correlation_strength,
    compute_yoy_changes,
    analyze_segment_comovement,
    interpolate_at_years,
    compute_lagged_correlation,
    compute_all_lagged_correlations,
    find_best_lag,
    DEFAULT_CORRELATION_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(start_year, end_year, slope, start_value=100.0, end_value=None):
    if end_value is None:
        end_value = start_value + slope * (end_year - start_year)
    return {
        "start_year": float(start_year),
        "end_year": float(end_year),
        "start_value": float(start_value),
        "end_value": float(end_value),
        "slope": float(slope),
        "p_value": 0.01,
    }


# ---------------------------------------------------------------------------
# get_direction
# ---------------------------------------------------------------------------

class TestGetDirection:
    def test_increasing(self):
        assert get_direction(np.array([100, 150])) == "increased"

    def test_decreasing(self):
        assert get_direction(np.array([150, 100])) == "decreased"

    def test_stable_small_change(self):
        assert get_direction(np.array([100, 104])) == "remained stable"

    def test_single_value(self):
        assert get_direction(np.array([100])) == "unknown"

    def test_empty(self):
        assert get_direction(np.array([])) == "unknown"


# ---------------------------------------------------------------------------
# get_correlation_strength
# ---------------------------------------------------------------------------

class TestGetCorrelationStrength:
    def test_no_correlation(self):
        assert get_correlation_strength(0.05) == "no"

    def test_weak(self):
        assert get_correlation_strength(0.25) == "weak"

    def test_moderate(self):
        assert get_correlation_strength(0.45) == "moderate"

    def test_strong(self):
        assert get_correlation_strength(0.65) == "strong"

    def test_very_strong(self):
        assert get_correlation_strength(0.85) == "very strong"

    def test_negative_same_as_positive(self):
        assert get_correlation_strength(-0.65) == "strong"


# ---------------------------------------------------------------------------
# compute_yoy_changes
# ---------------------------------------------------------------------------

class TestComputeYoyChanges:
    def test_consecutive_years(self):
        years = np.array([2010, 2011, 2012, 2013])
        values = np.array([100, 110, 120, 130])
        changes = compute_yoy_changes(years, values)
        np.testing.assert_array_equal(changes, [10, 10, 10])

    def test_with_gaps_annualized(self):
        years = np.array([2010, 2012, 2015])
        values = np.array([100, 120, 150])
        changes = compute_yoy_changes(years, values)
        np.testing.assert_array_almost_equal(changes, [10, 10])

    def test_single_value(self):
        years = np.array([2010])
        values = np.array([100])
        changes = compute_yoy_changes(years, values)
        assert len(changes) == 0

    def test_unsorted_input(self):
        years = np.array([2012, 2010, 2011])
        values = np.array([120, 100, 110])
        changes = compute_yoy_changes(years, values)
        np.testing.assert_array_equal(changes, [10, 10])


# ---------------------------------------------------------------------------
# interpolate_at_years
# ---------------------------------------------------------------------------

class TestInterpolateAtYears:
    def test_exact_years(self):
        source_years = np.array([2010, 2012, 2014, 2016])
        source_values = np.array([100, 120, 140, 160])
        target_years = np.array([2010, 2014, 2016])
        result = interpolate_at_years(source_years, source_values, target_years)
        np.testing.assert_array_almost_equal(result, [100, 140, 160])

    def test_interpolated_years(self):
        source_years = np.array([2010, 2014])
        source_values = np.array([100, 140])
        target_years = np.array([2011, 2012, 2013])
        result = interpolate_at_years(source_years, source_values, target_years)
        np.testing.assert_array_almost_equal(result, [110, 120, 130])

    def test_outside_range_returns_nan(self):
        source_years = np.array([2012, 2014, 2016])
        source_values = np.array([120, 140, 160])
        target_years = np.array([2010, 2013, 2018])
        result = interpolate_at_years(source_years, source_values, target_years)
        assert np.isnan(result[0])
        assert result[1] == 130
        assert np.isnan(result[2])

    def test_unsorted_source(self):
        source_years = np.array([2014, 2010, 2012])
        source_values = np.array([140, 100, 120])
        target_years = np.array([2011, 2013])
        result = interpolate_at_years(source_years, source_values, target_years)
        np.testing.assert_array_almost_equal(result, [110, 130])


# ---------------------------------------------------------------------------
# compute_lagged_correlation
# ---------------------------------------------------------------------------

class TestComputeLaggedCorrelation:
    # Shared test data
    sparse_years = np.array([2010, 2011, 2012, 2013, 2014])
    sparse_values = np.array([50, 55, 62, 68, 75], dtype=float)
    dense_years = np.array([2009, 2010, 2011, 2012, 2013, 2014])
    dense_values = np.array([95, 100, 110, 125, 138, 150], dtype=float)

    def test_lag_zero(self):
        result = compute_lagged_correlation(
            self.sparse_years, self.sparse_values,
            self.dense_years, self.dense_values,
            lag=0
        )
        assert result is not None
        assert result["n_pairs"] == 4
        assert result["correlation"] == pytest.approx(0.753, rel=0.01)
        assert result["p_value"] == pytest.approx(0.247, rel=0.01)

    def test_lag_one(self):
        result = compute_lagged_correlation(
            self.sparse_years, self.sparse_values,
            self.dense_years, self.dense_values,
            lag=1
        )
        assert result is not None
        assert result["n_pairs"] == 4
        assert result["correlation"] == pytest.approx(0.580, rel=0.01)
        assert result["p_value"] == pytest.approx(0.420, rel=0.01)

    def test_insufficient_overlap(self):
        result = compute_lagged_correlation(
            self.sparse_years[:2], self.sparse_values[:2],
            self.dense_years[:2], self.dense_values[:2],
            lag=0
        )
        assert result is None


# ---------------------------------------------------------------------------
# compute_all_lagged_correlations
# ---------------------------------------------------------------------------

class TestComputeAllLaggedCorrelations:
    def test_multiple_lags(self):
        sparse_years = np.array([2010, 2011, 2012, 2013, 2014, 2015, 2016])
        sparse_values = np.array([50, 58, 62, 72, 75, 88, 90], dtype=float)
        dense_years = np.arange(2005, 2020)
        dense_values = np.array([50, 60, 65, 78, 85, 100, 108, 125, 130, 148, 155, 175, 180, 200, 210], dtype=float)

        results = compute_all_lagged_correlations(
            sparse_years, sparse_values,
            dense_years, dense_values,
            max_lag=3
        )
        assert len(results) == 4
        lags = [r["lag"] for r in results]
        assert lags == [0, 1, 2, 3]

    def test_empty_if_no_valid_lags(self):
        sparse_years = np.array([2020, 2021])
        sparse_values = np.array([50, 55], dtype=float)
        dense_years = np.array([2010, 2011])
        dense_values = np.array([100, 110], dtype=float)

        results = compute_all_lagged_correlations(
            sparse_years, sparse_values,
            dense_years, dense_values,
            max_lag=2
        )
        assert len(results) == 0


# ---------------------------------------------------------------------------
# find_best_lag
# ---------------------------------------------------------------------------

class TestFindBestLag:
    def test_selects_strongest_significant_correlation(self):
        lag_results = [
            {"lag": 0, "correlation": 0.3, "p_value": 0.08, "n_pairs": 5},
            {"lag": 1, "correlation": 0.8, "p_value": 0.01, "n_pairs": 4},
            {"lag": 2, "correlation": 0.5, "p_value": 0.05, "n_pairs": 3},
        ]
        best = find_best_lag(lag_results)
        assert best["lag"] == 1
        assert best["correlation"] == 0.8

    def test_prefers_significant_over_stronger_insignificant(self):
        lag_results = [
            {"lag": 0, "correlation": 0.6, "p_value": 0.05, "n_pairs": 5},
            {"lag": 2, "correlation": 0.9, "p_value": 0.30, "n_pairs": 3},
        ]
        best = find_best_lag(lag_results)
        assert best["lag"] == 0
        assert best["correlation"] == 0.6

    def test_falls_back_to_strongest_if_none_significant(self):
        lag_results = [
            {"lag": 0, "correlation": 0.5, "p_value": 0.20, "n_pairs": 4},
            {"lag": 1, "correlation": 0.8, "p_value": 0.15, "n_pairs": 3},
        ]
        best = find_best_lag(lag_results)
        assert best["lag"] == 1
        assert best["correlation"] == 0.8

    def test_considers_absolute_value(self):
        lag_results = [
            {"lag": 0, "correlation": 0.5, "p_value": 0.05, "n_pairs": 5},
            {"lag": 1, "correlation": -0.9, "p_value": 0.001, "n_pairs": 4},
        ]
        best = find_best_lag(lag_results)
        assert best["lag"] == 1
        assert best["correlation"] == -0.9

    def test_empty_returns_none(self):
        assert find_best_lag([]) is None


# ---------------------------------------------------------------------------
# analyze_segment_comovement
# ---------------------------------------------------------------------------

class TestAnalyzeSegmentComovement:
    def test_both_increasing(self):
        segment = _seg(2010, 2015, slope=5)
        comp_years = np.array([2011, 2014])
        comp_values = np.array([50, 70])
        result = analyze_segment_comovement(segment, comp_years, comp_values)
        assert result["reference_direction"] == "increased"
        assert result["comparison_direction"] == "increased"
        assert result["comparison_n_points"] == 2

    def test_opposite_directions(self):
        segment = _seg(2010, 2015, slope=5)
        comp_years = np.array([2011, 2014])
        comp_values = np.array([70, 50])
        result = analyze_segment_comovement(segment, comp_years, comp_values)
        assert result["reference_direction"] == "increased"
        assert result["comparison_direction"] == "decreased"

    def test_no_comparison_data_in_segment(self):
        segment = _seg(2010, 2015, slope=5)
        comp_years = np.array([2016, 2017])
        comp_values = np.array([50, 60])
        result = analyze_segment_comovement(segment, comp_years, comp_values)
        assert result["comparison_direction"] is None
        assert result["comparison_n_points"] == 0

    def test_single_comparison_point(self):
        segment = _seg(2010, 2015, slope=5)
        comp_years = np.array([2012])
        comp_values = np.array([55])
        result = analyze_segment_comovement(segment, comp_years, comp_values)
        assert result["comparison_direction"] is None
        assert result["comparison_n_points"] == 1
        assert result["comparison_start"] == 55


# ---------------------------------------------------------------------------
# get_relationship_narrative - insufficient data
# ---------------------------------------------------------------------------

class TestRelationshipNarrativeInsufficientData:
    def test_too_few_comparison_points(self):
        segments = [_seg(2010, 2020, slope=5)]
        comp_years = np.array([2012, 2015])
        comp_values = np.array([50, 60])
        result = get_relationship_narrative(
            reference_segments=segments,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["method"] == "insufficient_data"
        assert "limited data" in result["narrative"].lower()

    def test_empty_segments(self):
        comp_years = np.array([2012, 2015, 2018, 2020])
        comp_values = np.array([50, 60, 70, 80])
        result = get_relationship_narrative(
            reference_segments=[],
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["method"] == "insufficient_data"


# ---------------------------------------------------------------------------
# get_relationship_narrative - comovement path
# ---------------------------------------------------------------------------

class TestRelationshipNarrativeComovement:
    def test_single_segment_both_increasing(self):
        segments = [_seg(2010, 2020, slope=5)]
        comp_years = np.array([2012, 2015, 2018])
        comp_values = np.array([50, 65, 80])
        result = get_relationship_narrative(
            reference_segments=segments,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="health spending",
            comparison_name="UHC index",
        )
        assert result["method"] == "comovement"
        assert "health spending" in result["narrative"]
        assert "UHC index" in result["narrative"]
        assert "same direction" in result["narrative"]

    def test_single_segment_opposite_directions(self):
        segments = [_seg(2010, 2020, slope=5)]
        comp_years = np.array([2012, 2015, 2018])
        comp_values = np.array([80, 65, 50])
        result = get_relationship_narrative(
            reference_segments=segments,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["method"] == "comovement"
        assert "opposite" in result["narrative"]

    def test_multiple_segments(self):
        segments = [
            _seg(2010, 2015, slope=5),
            _seg(2015, 2020, slope=-3),
        ]
        comp_years = np.array([2011, 2014, 2016, 2019])
        comp_values = np.array([50, 60, 65, 70])
        result = get_relationship_narrative(
            reference_segments=segments,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["method"] == "comovement"
        assert len(result["segment_details"]) == 2

    def test_caveat_about_limited_data(self):
        segments = [_seg(2010, 2020, slope=5)]
        comp_years = np.array([2012, 2015, 2018])
        comp_values = np.array([50, 65, 80])
        result = get_relationship_narrative(
            reference_segments=segments,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert "statistical relationship cannot be established" in result["narrative"]

    def test_segment_with_no_comparison_data(self):
        segments = [
            _seg(2010, 2015, slope=5),
            _seg(2015, 2020, slope=-3),
        ]
        comp_years = np.array([2011, 2013, 2014])
        comp_values = np.array([50, 55, 60])
        result = get_relationship_narrative(
            reference_segments=segments,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["method"] == "comovement"
        assert "unavailable" in result["narrative"]


# ---------------------------------------------------------------------------
# get_relationship_narrative - lagged correlation path
# ---------------------------------------------------------------------------

class TestRelationshipNarrativeLaggedCorrelation:
    def test_positive_correlation(self):
        segments = [_seg(2010, 2020, slope=5)]
        years = np.arange(2010, 2020)
        ref_values = np.array([100, 108, 112, 125, 128, 140, 145, 155, 162, 175], dtype=float)
        comp_values = np.array([50, 55, 58, 65, 68, 75, 78, 85, 90, 98], dtype=float)

        result = get_relationship_narrative(
            reference_segments=segments,
            comparison_years=years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
            reference_years=years,
            reference_values=ref_values,
            correlation_threshold=8,
        )
        assert result["method"] == "lagged_correlation"
        assert result["best_lag"]["lag"] == 0
        assert result["best_lag"]["correlation"] == pytest.approx(0.968, rel=0.01)
        assert result["best_lag"]["p_value"] == pytest.approx(0.0, abs=0.001)
        assert result["best_lag"]["n_pairs"] == 9
        assert result["narrative"] == (
            "Year-on-year changes in spending and outcome show the strongest association "
            "contemporaneously (no lag), with a very strong positive correlation "
            "(r=0.97, p=0.000, n=9 change pairs), which is statistically significant. "
            "When spending increases, outcome tends to increase in the same year. "
            "This analysis is based on 10 observations and assumes approximately "
            "linear change between data points."
        )

    def test_negative_correlation(self):
        segments = [_seg(2010, 2020, slope=5)]
        years = np.arange(2010, 2020)
        ref_values = np.array([100, 108, 112, 125, 128, 140, 145, 155, 162, 175], dtype=float)
        comp_values = np.array([100, 95, 92, 82, 80, 70, 68, 58, 52, 40], dtype=float)

        result = get_relationship_narrative(
            reference_segments=segments,
            comparison_years=years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
            reference_years=years,
            reference_values=ref_values,
            correlation_threshold=8,
        )
        assert result["method"] == "lagged_correlation"
        assert result["best_lag"]["lag"] == 0
        assert result["best_lag"]["correlation"] == pytest.approx(-0.959, rel=0.01)
        assert result["best_lag"]["p_value"] == pytest.approx(0.0, abs=0.001)
        assert result["best_lag"]["n_pairs"] == 9
        assert result["narrative"] == (
            "Year-on-year changes in spending and outcome show the strongest association "
            "contemporaneously (no lag), with a very strong negative correlation "
            "(r=-0.96, p=0.000, n=9 change pairs), which is statistically significant. "
            "When spending increases, outcome tends to decrease in the same year. "
            "This analysis is based on 10 observations and assumes approximately "
            "linear change between data points."
        )

    def test_insignificant_correlation(self):
        segments = [_seg(2010, 2020, slope=5)]
        years = np.arange(2010, 2020)
        ref_values = np.array([100, 105, 102, 108, 104, 110, 106, 112, 108, 114], dtype=float)
        comp_values = np.array([50, 48, 52, 49, 51, 47, 53, 50, 48, 52], dtype=float)

        result = get_relationship_narrative(
            reference_segments=segments,
            comparison_years=years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
            reference_years=years,
            reference_values=ref_values,
            correlation_threshold=8,
        )
        assert result["method"] == "lagged_correlation"
        assert result["best_lag"]["lag"] == 0
        assert result["best_lag"]["correlation"] == pytest.approx(-0.559, rel=0.01)
        assert result["best_lag"]["p_value"] == pytest.approx(0.118, rel=0.01)
        assert result["best_lag"]["n_pairs"] == 9
        assert result["narrative"] == (
            "Year-on-year changes in spending and outcome show the strongest association "
            "contemporaneously (no lag), with a strong negative correlation "
            "(r=-0.56, p=0.118, n=9 change pairs), which is not statistically significant. "
            "When spending increases, outcome tends to decrease in the same year. "
            "This analysis is based on 10 observations and assumes approximately "
            "linear change between data points."
        )

    def test_falls_back_to_comovement_below_threshold(self):
        segments = [_seg(2010, 2020, slope=5)]
        years = np.array([2010, 2012, 2015, 2018])
        ref_values = np.array([100, 110, 125, 140], dtype=float)
        comp_values = np.array([50, 55, 62, 70], dtype=float)

        result = get_relationship_narrative(
            reference_segments=segments,
            comparison_years=years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
            reference_years=years,
            reference_values=ref_values,
            correlation_threshold=8,
        )
        assert result["method"] == "comovement"

    def test_symmetric_reference_sparser_than_comparison(self):
        """When reference has fewer points than comparison, use reference to define periods."""
        segments = [_seg(2010, 2020, slope=5)]
        ref_years = np.array([2010, 2011, 2013, 2014, 2016, 2017, 2019, 2020])
        ref_values = np.array([100, 108, 120, 128, 145, 152, 168, 175], dtype=float)
        comp_years = np.arange(2010, 2021)
        comp_values = np.array([50, 53, 56, 60, 64, 68, 72, 76, 81, 86, 92], dtype=float)

        result = get_relationship_narrative(
            reference_segments=segments,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
            reference_years=ref_years,
            reference_values=ref_values,
            correlation_threshold=8,
        )
        assert result["method"] == "lagged_correlation"

    def test_lagged_effect_detection(self):
        """Test that lagged effects can be detected."""
        segments = [_seg(2010, 2025, slope=5)]
        ref_years = np.arange(2010, 2025)
        comp_years = np.arange(2010, 2025)
        # Reference increases in specific years
        ref_values = np.array([100, 100, 110, 110, 120, 120, 130, 130, 140, 140, 150, 150, 160, 160, 170], dtype=float)
        # Comparison follows reference pattern with 2-year lag
        comp_values = np.array([50, 50, 50, 50, 55, 55, 60, 60, 65, 65, 70, 70, 75, 75, 80], dtype=float)

        result = get_relationship_narrative(
            reference_segments=segments,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
            reference_years=ref_years,
            reference_values=ref_values,
            correlation_threshold=8,
            max_lag_cap=5,
        )
        assert result["method"] == "lagged_correlation"
        assert result["best_lag"]["lag"] == 2
        assert result["best_lag"]["correlation"] == pytest.approx(1.0, rel=0.01)
        assert result["best_lag"]["p_value"] == pytest.approx(0.0, abs=0.001)
        assert result["best_lag"]["n_pairs"] == 12
        assert result["all_lags"] is not None
        assert len(result["all_lags"]) == 6
        assert result["max_lag_tested"] == 5
        assert result["narrative"] == (
            "Year-on-year changes in spending and outcome show the strongest association "
            "with a 2-year lag, with a very strong positive correlation "
            "(r=1.00, p=0.000, n=12 change pairs), which is statistically significant. "
            "When spending increases, outcome tends to increase about 2 years later. "
            "This analysis is based on 15 observations and assumes approximately "
            "linear change between data points."
        )


# ---------------------------------------------------------------------------
# get_relationship_narrative - NaN handling
# ---------------------------------------------------------------------------

class TestRelationshipNarrativeNanHandling:
    def test_removes_nan_from_comparison(self):
        segments = [_seg(2010, 2020, slope=5)]
        comp_years = np.array([2012, 2015, 2018, 2019])
        comp_values = np.array([50, np.nan, 80, 90])
        result = get_relationship_narrative(
            reference_segments=segments,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["n_points"] == 3


# ---------------------------------------------------------------------------
# get_relationship_narrative - return structure
# ---------------------------------------------------------------------------

class TestRelationshipNarrativeReturnStructure:
    def test_comovement_return_keys(self):
        segments = [_seg(2010, 2020, slope=5)]
        comp_years = np.array([2012, 2015, 2018])
        comp_values = np.array([50, 65, 80])
        result = get_relationship_narrative(
            reference_segments=segments,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert "narrative" in result
        assert "method" in result
        assert "n_points" in result
        assert "segment_details" in result
        assert "best_lag" in result
        assert "all_lags" in result
        assert "max_lag_tested" in result
        # For comovement, lag-related fields should be None
        assert result["best_lag"] is None
        assert result["all_lags"] is None

    def test_lagged_correlation_return_keys(self):
        segments = [_seg(2010, 2020, slope=5)]
        years = np.arange(2010, 2020)
        ref_values = np.array([100, 108, 112, 125, 128, 140, 145, 155, 162, 175], dtype=float)
        comp_values = np.array([50, 55, 58, 65, 68, 75, 78, 85, 90, 98], dtype=float)

        result = get_relationship_narrative(
            reference_segments=segments,
            comparison_years=years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
            reference_years=years,
            reference_values=ref_values,
            correlation_threshold=8,
        )
        assert "narrative" in result
        assert "method" in result
        assert "n_points" in result
        assert "best_lag" in result
        assert "all_lags" in result
        assert "max_lag_tested" in result
        # For lagged correlation, these should be populated
        assert result["best_lag"] is not None
        assert "lag" in result["best_lag"]
        assert "correlation" in result["best_lag"]
        assert "p_value" in result["best_lag"]
        assert "n_pairs" in result["best_lag"]

    def test_insufficient_data_return_keys(self):
        segments = [_seg(2010, 2020, slope=5)]
        comp_years = np.array([2012, 2015])
        comp_values = np.array([50, 60])
        result = get_relationship_narrative(
            reference_segments=segments,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert result["method"] == "insufficient_data"
        assert result["segment_details"] is None
        assert result["best_lag"] is None
        assert result["all_lags"] is None
        assert result["max_lag_tested"] is None
