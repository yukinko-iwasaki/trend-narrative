"""Unit tests for trend_narrative.relationship_analysis."""

import numpy as np
import pytest

from trend_narrative.relationship_analysis import (
    analyze_relationship,
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
        assert get_direction(np.array([0, 150])) == "increased"

    def test_decreasing(self):
        assert get_direction(np.array([150, 100])) == "decreased"
        assert get_direction(np.array([0, -150])) == "decreased"

    def test_stable_small_change(self):
        assert get_direction(np.array([100, 104])) == "remained_stable"
        assert get_direction(np.array([0, 0])) == "remained_stable"

    def test_single_value(self):
        assert get_direction(np.array([100])) == "unknown"

    def test_empty(self):
        assert get_direction(np.array([])) == "unknown"


# ---------------------------------------------------------------------------
# get_correlation_strength
# ---------------------------------------------------------------------------

class TestGetCorrelationStrength:
    def test_no_correlation(self):
        assert get_correlation_strength(0.05) == "strength_no"

    def test_weak(self):
        assert get_correlation_strength(0.25) == "strength_weak"

    def test_moderate(self):
        assert get_correlation_strength(0.45) == "strength_moderate"

    def test_strong(self):
        assert get_correlation_strength(0.65) == "strength_strong"

    def test_very_strong(self):
        assert get_correlation_strength(0.85) == "strength_very_strong"

    def test_negative_same_as_positive(self):
        assert get_correlation_strength(-0.65) == "strength_strong"


# ---------------------------------------------------------------------------
# compute_yoy_changes
# ---------------------------------------------------------------------------

class TestComputeYoyChanges:
    def test_consecutive_years(self):
        years = np.array([2010, 2011, 2012, 2013])
        values = np.array([100, 110, 120, 130])
        changes = compute_yoy_changes(years, values)
        np.testing.assert_array_almost_equal(changes, [0.1, 0.0909, 0.0833], decimal=3)

    def test_with_gaps_annualized(self):
        years = np.array([2010, 2012, 2015])
        values = np.array([100, 120, 150])
        changes = compute_yoy_changes(years, values)
        np.testing.assert_array_almost_equal(changes, [0.1, 0.0833], decimal=3)

    def test_single_value(self):
        years = np.array([2010])
        values = np.array([100])
        changes = compute_yoy_changes(years, values)
        assert len(changes) == 0

    def test_unsorted_input(self):
        years = np.array([2012, 2010, 2011])
        values = np.array([120, 100, 110])
        changes = compute_yoy_changes(years, values)
        np.testing.assert_array_almost_equal(changes, [0.1, 0.0909], decimal=3)


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
        assert result["correlation"] == pytest.approx(0.863, rel=0.01)
        assert result["p_value"] == pytest.approx(0.137, rel=0.01)

    def test_lag_one(self):
        result = compute_lagged_correlation(
            self.sparse_years, self.sparse_values,
            self.dense_years, self.dense_values,
            lag=1
        )
        assert result is not None
        assert result["n_pairs"] == 4
        assert result["correlation"] == pytest.approx(-0.040, abs=0.01)
        assert result["p_value"] == pytest.approx(0.960, rel=0.01)

    def test_insufficient_overlap(self):
        result = compute_lagged_correlation(
            self.sparse_years[:2], self.sparse_values[:2],
            self.dense_years[:2], self.dense_values[:2],
            lag=0
        )
        assert result is None

    def test_near_zero_base_values_filtered(self):
        """Near-zero base values produce NaN changes; these should be filtered."""
        sparse_years = np.array([2010, 2011, 2012, 2013, 2014, 2015])
        sparse_values = np.array([0.0, 50, 60, 70, 80, 90], dtype=float)
        dense_years = np.array([2010, 2011, 2012, 2013, 2014, 2015])
        dense_values = np.array([100, 110, 120, 130, 140, 150], dtype=float)
        result = compute_lagged_correlation(
            sparse_years, sparse_values,
            dense_years, dense_values,
            lag=0
        )
        assert result is not None
        assert result["n_pairs"] == 4

    def test_all_near_zero_returns_none(self):
        """If all changes are NaN/inf, should return None."""
        sparse_years = np.array([2010, 2011, 2012])
        sparse_values = np.array([0.0, 0.0, 0.0], dtype=float)
        dense_years = np.array([2010, 2011, 2012])
        dense_values = np.array([100, 110, 120], dtype=float)
        result = compute_lagged_correlation(
            sparse_years, sparse_values,
            dense_years, dense_values,
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
    def test_both_increasing_fallback(self):
        """Falls back to actual data points when interpolation fails."""
        segment = _seg(2010, 2015, slope=5)
        comp_years = np.array([2011, 2014])
        comp_values = np.array([50, 70])
        result = analyze_segment_comovement(segment, comp_years, comp_values)
        assert result["reference_direction"] == "increased"
        assert result["comparison_direction"] == "increased"
        assert result["comparison_n_points"] == 2
        assert result["comparison_start"] == 50
        assert result["comparison_end"] == 70
        assert result["interpolated"] is False

    def test_both_increasing_interpolated(self):
        """Uses interpolation when comparison data spans segment boundaries."""
        segment = _seg(2010, 2015, slope=5)
        comp_years = np.array([2008, 2012, 2018])
        comp_values = np.array([40, 60, 100])
        result = analyze_segment_comovement(segment, comp_years, comp_values)
        assert result["reference_direction"] == "increased"
        assert result["comparison_direction"] == "increased"
        assert result["comparison_n_points"] == 1
        assert result["comparison_start"] == pytest.approx(50, rel=0.01)
        assert result["comparison_end"] == pytest.approx(80, rel=0.01)
        assert result["interpolated"] is True

    def test_exact_boundary_matches(self):
        """When comparison data exists at exact segment boundaries, not interpolated."""
        segment = _seg(2010, 2015, slope=5)
        comp_years = np.array([2010, 2012, 2015])
        comp_values = np.array([50, 60, 70])
        result = analyze_segment_comovement(segment, comp_years, comp_values)
        assert result["reference_direction"] == "increased"
        assert result["comparison_direction"] == "increased"
        assert result["comparison_start"] == 50
        assert result["comparison_end"] == 70
        assert result["interpolated"] is False

    def test_opposite_directions(self):
        segment = _seg(2010, 2015, slope=5)
        comp_years = np.array([2011, 2014])
        comp_values = np.array([70, 50])
        result = analyze_segment_comovement(segment, comp_years, comp_values)
        assert result["reference_direction"] == "increased"
        assert result["comparison_direction"] == "decreased"
        assert result["interpolated"] is False

    def test_no_comparison_data_in_segment(self):
        segment = _seg(2010, 2015, slope=5)
        comp_years = np.array([2016, 2017])
        comp_values = np.array([50, 60])
        result = analyze_segment_comovement(segment, comp_years, comp_values)
        assert result["comparison_direction"] is None
        assert result["comparison_n_points"] == 0
        assert result["interpolated"] is False

    def test_single_comparison_point(self):
        segment = _seg(2010, 2015, slope=5)
        comp_years = np.array([2012])
        comp_values = np.array([55])
        result = analyze_segment_comovement(segment, comp_years, comp_values)
        assert result["comparison_direction"] is None
        assert result["comparison_n_points"] == 1
        assert result["comparison_start"] == 55
        assert result["interpolated"] is False


# ---------------------------------------------------------------------------
# analyze_relationship
# ---------------------------------------------------------------------------

class TestAnalyzeRelationship:
    years_3pt = np.array([2010, 2015, 2020])
    ref_increasing = np.array([100, 125, 150], dtype=float)
    comp_increasing = np.array([50, 65, 80], dtype=float)
    periods_10 = np.arange(1, 11)
    ref_lag0_pos = np.array([100, 108, 112, 125, 128, 140, 145, 155, 162, 175], dtype=float)
    comp_lag0_pos = np.array([50, 55, 58, 65, 68, 75, 78, 85, 90, 98], dtype=float)

    def test_returns_comovement_insights(self):
        """analyze_relationship returns structured insights for comovement case."""
        result = analyze_relationship(
            reference_years=self.years_3pt,
            reference_values=self.ref_increasing,
            comparison_years=self.years_3pt,
            comparison_values=self.comp_increasing,
        )
        assert result["method"] == "comovement"
        assert result["n_points"] == 3
        assert result["segment_details"] is not None
        assert result["best_lag"] is None
        assert result["all_lags"] is None
        assert "reference_leads" in result

    def test_returns_lagged_correlation_insights(self):
        """analyze_relationship returns structured insights for correlation case."""
        result = analyze_relationship(
            reference_years=self.periods_10,
            reference_values=self.ref_lag0_pos,
            comparison_years=self.periods_10,
            comparison_values=self.comp_lag0_pos,
            correlation_threshold=8,
        )
        assert result["method"] == "lagged_correlation"
        assert result["n_points"] == 10
        assert result["segment_details"] is None
        assert result["best_lag"] is not None
        assert result["all_lags"] is not None
        assert "reference_leads" in result

    def test_returns_insufficient_data_insights(self):
        """analyze_relationship returns structured insights for insufficient data."""
        result = analyze_relationship(
            reference_years=np.array([2010, 2020]),
            reference_values=np.array([100, 150]),
            comparison_years=np.array([2010, 2020]),
            comparison_values=np.array([50, 75]),
        )
        assert result["method"] == "insufficient_data"
        assert result["n_points"] == 2
        assert result["segment_details"] is None
        assert result["best_lag"] is None
        assert "reference_leads" in result

    def test_no_narrative_key(self):
        """analyze_relationship should not include narrative in result."""
        result = analyze_relationship(
            reference_years=self.years_3pt,
            reference_values=self.ref_increasing,
            comparison_years=self.years_3pt,
            comparison_values=self.comp_increasing,
        )
        assert "narrative" not in result

    def test_reference_leads_when_comparison_sparser(self):
        """When comparison is sparser, reference_leads should be True."""
        result = analyze_relationship(
            reference_years=self.periods_10,
            reference_values=self.ref_lag0_pos,
            comparison_years=self.periods_10[:5],
            comparison_values=self.comp_lag0_pos[:5],
            correlation_threshold=5,
        )
        assert result["reference_leads"] is True

    def test_reference_leads_false_when_reference_sparser(self):
        """When reference is sparser, reference_leads should be False."""
        result = analyze_relationship(
            reference_years=self.periods_10[:5],
            reference_values=self.ref_lag0_pos[:5],
            comparison_years=self.periods_10,
            comparison_values=self.comp_lag0_pos,
            correlation_threshold=5,
        )
        assert result["reference_leads"] is False

    def test_handles_nan_values(self):
        """NaN values should be filtered out."""
        ref_years = np.array([2010, 2015, 2020])
        ref_values = np.array([100, np.nan, 150])
        comp_years = np.array([2012, 2015, 2018])
        comp_values = np.array([50, 65, 80])
        result = analyze_relationship(
            reference_years=ref_years,
            reference_values=ref_values,
            comparison_years=comp_years,
            comparison_values=comp_values,
        )
        assert result["n_points"] == 2

    def test_handles_unsorted_input(self):
        """Input data should be sorted internally."""
        sorted_result = analyze_relationship(
            reference_years=self.years_3pt,
            reference_values=self.ref_increasing,
            comparison_years=self.years_3pt,
            comparison_values=self.comp_increasing,
        )
        unsorted_result = analyze_relationship(
            reference_years=self.years_3pt[::-1],
            reference_values=self.ref_increasing[::-1],
            comparison_years=self.years_3pt[::-1],
            comparison_values=self.comp_increasing[::-1],
        )
        assert sorted_result["method"] == unsorted_result["method"]
        assert sorted_result["n_points"] == unsorted_result["n_points"]
