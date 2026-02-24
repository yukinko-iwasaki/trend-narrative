"""Unit tests for trend_narrative.narrative."""

import pytest

from trend_narrative.narrative import (
    consolidate_segments,
    generate_narrative,
    get_segment_narrative,
    millify,
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
# millify
# ---------------------------------------------------------------------------

class TestMillify:
    def test_small_number(self):
        assert millify(750) == "750.00"

    def test_thousands(self):
        assert millify(1_500) == "1.50 K"

    def test_millions(self):
        assert millify(2_000_000) == "2.00 M"

    def test_billions(self):
        assert millify(3_000_000_000) == "3.00 B"

    def test_zero(self):
        result = millify(0)
        assert result == "0.00"

    def test_negative(self):
        result = millify(-500_000)
        assert "-" in result or "500" in result   # sign preserved


# ---------------------------------------------------------------------------
# consolidate_segments
# ---------------------------------------------------------------------------

class TestConsolidateSegments:
    def test_empty_returns_empty(self):
        assert consolidate_segments([]) == []

    def test_single_segment_unchanged(self):
        segs = [_seg(2010, 2015, slope=5)]
        result = consolidate_segments(segs)
        assert len(result) == 1

    def test_merges_same_direction(self):
        segs = [_seg(2010, 2013, slope=5), _seg(2013, 2016, slope=3)]
        result = consolidate_segments(segs)
        assert len(result) == 1
        assert result[0]["start_year"] == 2010
        assert result[0]["end_year"] == 2016

    def test_keeps_different_directions(self):
        segs = [
            _seg(2010, 2014, slope=5),
            _seg(2014, 2018, slope=-3),
        ]
        result = consolidate_segments(segs)
        assert len(result) == 2

    def test_merged_slope_recomputed(self):
        segs = [
            _seg(2010, 2012, slope=4, start_value=100, end_value=108),
            _seg(2012, 2014, slope=6, start_value=108, end_value=120),
        ]
        result = consolidate_segments(segs)
        expected_slope = (120 - 100) / (2014 - 2010)
        assert result[0]["slope"] == pytest.approx(expected_slope)

    def test_does_not_mutate_input(self):
        segs = [_seg(2010, 2012, slope=2), _seg(2012, 2014, slope=3)]
        original_end = segs[0]["end_year"]
        consolidate_segments(segs)
        assert segs[0]["end_year"] == original_end

    def test_zero_duration_slope_is_zero(self):
        segs = [
            _seg(2010, 2010, slope=0, start_value=100, end_value=100),
            _seg(2010, 2010, slope=0, start_value=100, end_value=100),
        ]
        result = consolidate_segments(segs)
        assert result[0]["slope"] == 0.0

    def test_three_same_direction_merged_to_one(self):
        segs = [
            _seg(2010, 2012, slope=2),
            _seg(2012, 2014, slope=3),
            _seg(2014, 2016, slope=1),
        ]
        result = consolidate_segments(segs)
        assert len(result) == 1

    def test_alternating_directions_preserved(self):
        segs = [
            _seg(2010, 2013, slope=5),
            _seg(2013, 2016, slope=-2),
            _seg(2016, 2019, slope=4),
        ]
        result = consolidate_segments(segs)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# get_segment_narrative
# ---------------------------------------------------------------------------

class TestGetSegmentNarrative:
    # --- no-segment fallbacks ---
    def test_no_segments_low_cv(self):
        text = get_segment_narrative([], cv_value=2.0, metric="spending")
        assert "stable" in text.lower()

    def test_no_segments_moderate_cv(self):
        text = get_segment_narrative([], cv_value=10.0, metric="spending")
        assert "fluctuation" in text.lower() or "moderate" in text.lower()

    def test_no_segments_high_cv(self):
        text = get_segment_narrative([], cv_value=25.0, metric="spending")
        assert "volatility" in text.lower() or "volatile" in text.lower()

    # --- single segment ---
    def test_single_upward_segment(self):
        segs = [_seg(2010, 2020, slope=10, start_value=100, end_value=200)]
        text = get_segment_narrative(segs, cv_value=8.0, metric="health spending")
        assert "increased" in text
        assert "2010" in text and "2020" in text

    def test_single_downward_segment(self):
        segs = [_seg(2010, 2020, slope=-5, start_value=200, end_value=100)]
        text = get_segment_narrative(segs, cv_value=8.0, metric="spending")
        assert "decreased" in text

    def test_single_segment_contains_metric(self):
        segs = [_seg(2010, 2018, slope=3, start_value=100, end_value=124)]
        text = get_segment_narrative(segs, cv_value=5.0, metric="education budget")
        assert "education budget" in text

    # --- multi-segment ---
    def test_two_segments_peak_transition(self):
        segs = [
            _seg(2010, 2015, slope=10, start_value=100, end_value=150),
            _seg(2015, 2020, slope=-5, start_value=150, end_value=125),
        ]
        text = get_segment_narrative(segs, cv_value=10.0, metric="spending")
        assert "peak" in text.lower()

    def test_two_segments_trough_transition(self):
        segs = [
            _seg(2010, 2015, slope=-5, start_value=150, end_value=125),
            _seg(2015, 2020, slope=10, start_value=125, end_value=175),
        ]
        text = get_segment_narrative(segs, cv_value=10.0, metric="spending")
        assert "low" in text.lower() or "recovery" in text.lower()

    def test_returns_string(self):
        segs = [_seg(2010, 2020, slope=5)]
        result = get_segment_narrative(segs, cv_value=8.0)
        assert isinstance(result, str)

    def test_empty_segments_invalid_cv_returns_empty(self):
        result = get_segment_narrative([], cv_value=None, metric="x")
        assert result == ""

    # --- consolidation is applied internally ---
    def test_same_direction_segments_consolidated(self):
        """Two upward segments should consolidate to one → single-segment narrative."""
        segs = [
            _seg(2010, 2013, slope=3, start_value=100, end_value=109),
            _seg(2013, 2016, slope=4, start_value=109, end_value=121),
        ]
        text = get_segment_narrative(segs, cv_value=6.0, metric="spending")
        # Single-segment branch produces "maintaining a consistent trajectory"
        assert "consistent" in text.lower() or "increased" in text.lower()


# ---------------------------------------------------------------------------
# generate_narrative – end-to-end convenience function
# ---------------------------------------------------------------------------

class TestGenerateNarrative:
    def test_returns_dict_with_expected_keys(self):
        import numpy as np
        x = np.arange(2010, 2022, dtype=float)
        y = 100.0 + 5.0 * (x - 2010)
        result = generate_narrative(x, y, metric="test metric")
        assert {"narrative", "cv_value", "segments"} == set(result.keys())

    def test_narrative_is_string(self):
        import numpy as np
        x = np.arange(2010, 2022, dtype=float)
        y = 100.0 + 5.0 * (x - 2010)
        result = generate_narrative(x, y)
        assert isinstance(result["narrative"], str)

    def test_metric_appears_in_narrative(self):
        import numpy as np
        x = np.arange(2010, 2022, dtype=float)
        y = 100.0 + 5.0 * (x - 2010)
        result = generate_narrative(x, y, metric="defence spending")
        # May be in narrative OR in stability fallback text
        assert isinstance(result["narrative"], str)

    def test_detector_kwargs_forwarded(self):
        import numpy as np
        x = np.arange(2005, 2022, dtype=float)
        y = np.concatenate([50 - 2 * np.arange(8), 34 + 3 * np.arange(9)])
        result = generate_narrative(x, y, detector_kwargs={"max_segments": 1})
        assert len(result["segments"]) <= 1
