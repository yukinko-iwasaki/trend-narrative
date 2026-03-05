"""Unit tests for trend_narrative.relationship."""

import numpy as np
import pytest

from trend_narrative.relationship import (
    get_relationship_narrative,
    get_direction,
    get_correlation_strength,
    compute_yoy_changes,
    analyze_segment_comovement,
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
# get_relationship_narrative - correlation path
# ---------------------------------------------------------------------------

class TestRelationshipNarrativeCorrelation:
    def test_positive_correlation(self):
        segments = [_seg(2010, 2020, slope=5)]
        years = np.arange(2010, 2020)
        # Add variation to avoid constant changes
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
            threshold_high=8,
        )
        assert result["method"] == "correlation"
        assert result["correlation"] is not None
        assert result["correlation"] > 0
        assert "positive" in result["narrative"]

    def test_negative_correlation(self):
        segments = [_seg(2010, 2020, slope=5)]
        years = np.arange(2010, 2020)
        # Add variation to avoid constant changes
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
            threshold_high=8,
        )
        assert result["method"] == "correlation"
        assert result["correlation"] < 0
        assert "negative" in result["narrative"]

    def test_no_correlation(self):
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
            threshold_high=8,
        )
        assert result["method"] == "correlation"

    def test_falls_back_to_comovement_below_threshold(self):
        segments = [_seg(2010, 2020, slope=5)]
        years = np.array([2010, 2012, 2015, 2018, 2020])
        ref_values = np.array([100, 110, 125, 140, 150], dtype=float)
        comp_values = np.array([50, 55, 62, 70, 75], dtype=float)

        result = get_relationship_narrative(
            reference_segments=segments,
            comparison_years=years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
            reference_years=years,
            reference_values=ref_values,
            threshold_high=8,
        )
        assert result["method"] == "comovement"

    def test_includes_p_value(self):
        segments = [_seg(2010, 2020, slope=5)]
        years = np.arange(2010, 2020)
        # Add variation to avoid constant changes
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
            threshold_high=8,
        )
        assert result["p_value"] is not None
        assert 0 <= result["p_value"] <= 1


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
        assert "correlation" in result
        assert "p_value" in result

    def test_correlation_return_keys(self):
        segments = [_seg(2010, 2020, slope=5)]
        years = np.arange(2010, 2020)
        ref_values = 100 + 5 * np.arange(10).astype(float)
        comp_values = 50 + 3 * np.arange(10).astype(float)

        result = get_relationship_narrative(
            reference_segments=segments,
            comparison_years=years,
            comparison_values=comp_values,
            reference_name="spending",
            comparison_name="outcome",
            reference_years=years,
            reference_values=ref_values,
            threshold_high=8,
        )
        assert "narrative" in result
        assert "method" in result
        assert "n_points" in result
        assert "correlation" in result
        assert "p_value" in result
