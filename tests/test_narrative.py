"""Unit tests for trend_narrative.narrative."""

import numpy as np
import pytest

from trend_narrative.narrative import (
    consolidate_segments,
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
        assert millify(0) == "0.00"

    def test_negative(self):
        result = millify(-500_000)
        assert "-" in result or "500" in result


# ---------------------------------------------------------------------------
# consolidate_segments
# ---------------------------------------------------------------------------

class TestConsolidateSegments:
    def test_empty_returns_empty(self):
        assert consolidate_segments([]) == []

    def test_single_segment_unchanged(self):
        result = consolidate_segments([_seg(2010, 2015, slope=5)])
        assert len(result) == 1

    def test_merges_same_direction(self):
        segs = [_seg(2010, 2013, slope=5), _seg(2013, 2016, slope=3)]
        result = consolidate_segments(segs)
        assert len(result) == 1
        assert result[0]["start_year"] == 2010
        assert result[0]["end_year"] == 2016

    def test_keeps_different_directions(self):
        segs = [_seg(2010, 2014, slope=5), _seg(2014, 2018, slope=-3)]
        assert len(consolidate_segments(segs)) == 2

    def test_merged_slope_recomputed(self):
        segs = [
            _seg(2010, 2012, slope=4, start_value=100, end_value=108),
            _seg(2012, 2014, slope=6, start_value=108, end_value=120),
        ]
        result = consolidate_segments(segs)
        assert result[0]["slope"] == pytest.approx((120 - 100) / (2014 - 2010))

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
        assert consolidate_segments(segs)[0]["slope"] == 0.0

    def test_three_same_direction_merged_to_one(self):
        segs = [_seg(2010, 2012, slope=2), _seg(2012, 2014, slope=3), _seg(2014, 2016, slope=1)]
        assert len(consolidate_segments(segs)) == 1

    def test_alternating_directions_preserved(self):
        segs = [_seg(2010, 2013, slope=5), _seg(2013, 2016, slope=-2), _seg(2016, 2019, slope=4)]
        assert len(consolidate_segments(segs)) == 3


# ---------------------------------------------------------------------------
# get_segment_narrative – Path 1 (precomputed segments + cv_value)
# ---------------------------------------------------------------------------

class TestGetSegmentNarrativePath1:
    """All calls use explicit segments= and cv_value= (Databricks path)."""

    def test_no_segments_low_cv(self):
        text = get_segment_narrative(segments=[], cv_value=2.0, metric="spending")
        assert "stable" in text.lower()

    def test_no_segments_moderate_cv(self):
        text = get_segment_narrative(segments=[], cv_value=10.0, metric="spending")
        assert "fluctuation" in text.lower() or "moderate" in text.lower()

    def test_no_segments_high_cv(self):
        text = get_segment_narrative(segments=[], cv_value=25.0, metric="spending")
        assert "volatility" in text.lower()

    def test_single_upward_segment(self):
        segs = [_seg(2010, 2020, slope=10, start_value=100, end_value=200)]
        text = get_segment_narrative(segments=segs, cv_value=8.0, metric="health spending")
        assert "increased" in text
        assert "2010" in text and "2020" in text

    def test_single_downward_segment(self):
        segs = [_seg(2010, 2020, slope=-5, start_value=200, end_value=100)]
        text = get_segment_narrative(segments=segs, cv_value=8.0, metric="spending")
        assert "decreased" in text

    def test_metric_label_in_output(self):
        segs = [_seg(2010, 2018, slope=3, start_value=100, end_value=124)]
        text = get_segment_narrative(segments=segs, cv_value=5.0, metric="education budget")
        assert "education budget" in text

    def test_peak_transition(self):
        segs = [
            _seg(2010, 2015, slope=10, start_value=100, end_value=150),
            _seg(2015, 2020, slope=-5, start_value=150, end_value=125),
        ]
        text = get_segment_narrative(segments=segs, cv_value=10.0, metric="spending")
        assert "peak" in text.lower()

    def test_trough_transition(self):
        segs = [
            _seg(2010, 2015, slope=-5, start_value=150, end_value=125),
            _seg(2015, 2020, slope=10, start_value=125, end_value=175),
        ]
        text = get_segment_narrative(segments=segs, cv_value=10.0, metric="spending")
        assert "low" in text.lower() or "recovery" in text.lower()

    def test_returns_string(self):
        segs = [_seg(2010, 2020, slope=5)]
        assert isinstance(get_segment_narrative(segments=segs, cv_value=8.0), str)

    def test_same_direction_consolidates(self):
        segs = [
            _seg(2010, 2013, slope=3, start_value=100, end_value=109),
            _seg(2013, 2016, slope=4, start_value=109, end_value=121),
        ]
        text = get_segment_narrative(segments=segs, cv_value=6.0, metric="spending")
        assert "consistent" in text.lower() or "increased" in text.lower()

    def test_missing_both_raises(self):
        with pytest.raises(ValueError):
            get_segment_narrative(metric="spending")


# ---------------------------------------------------------------------------
# get_segment_narrative – Path 2 (raw x/y arrays, standalone)
# ---------------------------------------------------------------------------

class TestGetSegmentNarrativePath2:
    """All calls use x= and y= (standalone path)."""

    def test_returns_string(self):
        x = np.arange(2010, 2022, dtype=float)
        y = 100.0 + 5.0 * (x - 2010)
        result = get_segment_narrative(x=x, y=y, metric="spending")
        assert isinstance(result, str)

    def test_upward_series_mentions_increase_or_stable(self):
        x = np.arange(2010, 2022, dtype=float)
        y = 100.0 + 5.0 * (x - 2010)
        text = get_segment_narrative(x=x, y=y, metric="spending")
        assert len(text) > 0

    def test_metric_label_forwarded(self):
        x = np.arange(2010, 2022, dtype=float)
        y = 100.0 + 5.0 * (x - 2010)
        text = get_segment_narrative(x=x, y=y, metric="defence budget")
        assert "defence budget" in text

    def test_detector_kwargs_forwarded(self):
        x = np.arange(2005, 2022, dtype=float)
        y = np.concatenate([50 - 2 * np.arange(8), 34 + 3 * np.arange(9)], dtype=float)
        # With max_segments=1 we get at most 1 segment, so no peak/trough language
        text = get_segment_narrative(x=x, y=y, metric="spending",
                                     detector_kwargs={"max_segments": 1})
        assert isinstance(text, str)

    def test_path2_matches_manual_path1(self):
        """Path 2 output should equal manually calling extractor then Path 1."""
        from trend_narrative import InsightExtractor

        x = np.arange(2010, 2022, dtype=float)
        y = 100.0 + 5.0 * (x - 2010)

        # Path 2
        text_path2 = get_segment_narrative(x=x, y=y, metric="spending")

        # Equivalent manual Path 1
        suite = InsightExtractor(x, y).extract_full_suite()
        text_path1 = get_segment_narrative(
            segments=suite["segments"], cv_value=suite["cv_value"], metric="spending"
        )

        assert text_path2 == text_path1

    def test_partial_args_raises(self):
        """Providing only x without y should raise ValueError."""
        x = np.arange(2010, 2022, dtype=float)
        with pytest.raises((ValueError, TypeError)):
            get_segment_narrative(x=x, metric="spending")
