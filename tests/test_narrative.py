"""Unit tests for trend_narrative.narrative."""

import numpy as np
import pytest

from trend_narrative import InsightExtractor, TrendDetector
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


def _linear_extractor(n=12, slope=5.0):
    """Return an InsightExtractor wrapping a clean linear series."""
    x = np.arange(2010, 2010 + n, dtype=float)
    y = 100.0 + slope * np.arange(n, dtype=float)
    return InsightExtractor(x, y)


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
        assert len(consolidate_segments([_seg(2010, 2015, slope=5)])) == 1

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
    """Caller holds precomputed segments and cv_value (e.g. from Delta table)."""

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

    def test_missing_inputs_raises(self):
        with pytest.raises(ValueError):
            get_segment_narrative(metric="spending")

    def test_segments_without_cv_raises(self):
        segs = [_seg(2010, 2020, slope=5)]
        with pytest.raises(ValueError):
            get_segment_narrative(segments=segs, metric="spending")

    def test_cv_without_segments_raises(self):
        with pytest.raises(ValueError):
            get_segment_narrative(cv_value=5.0, metric="spending")


# ---------------------------------------------------------------------------
# get_segment_narrative – Path 2 (InsightExtractor object)
# ---------------------------------------------------------------------------

class TestGetSegmentNarrativePath2:
    """Caller builds an InsightExtractor (choosing detector) and passes it in."""

    def test_returns_string(self):
        text = get_segment_narrative(extractor=_linear_extractor(), metric="spending")
        assert isinstance(text, str)

    def test_non_empty_output(self):
        text = get_segment_narrative(extractor=_linear_extractor(), metric="spending")
        assert len(text) > 0

    def test_metric_label_forwarded(self):
        text = get_segment_narrative(extractor=_linear_extractor(), metric="defence budget")
        assert "defence budget" in text

    def test_custom_detector_respected(self):
        """A max_segments=1 detector should never produce peak/trough language."""
        x = np.arange(2005, 2022, dtype=float)
        y = np.concatenate([50 - 2 * np.arange(8), 34 + 3 * np.arange(9)]).astype(float)
        detector = TrendDetector(max_segments=1)
        extractor = InsightExtractor(x, y, detector=detector)
        text = get_segment_narrative(extractor=extractor, metric="spending")
        assert "peak" not in text.lower() and "low" not in text.lower()

    def test_default_detector_used_when_none_specified(self):
        """InsightExtractor without explicit detector should still work."""
        x = np.arange(2010, 2022, dtype=float)
        y = 100.0 + 5.0 * np.arange(12, dtype=float)
        extractor = InsightExtractor(x, y)
        text = get_segment_narrative(extractor=extractor, metric="spending")
        assert isinstance(text, str)

    def test_path2_matches_path1_for_same_data(self):
        """Path 2 output should equal the equivalent manual Path 1 call."""
        x = np.arange(2010, 2022, dtype=float)
        y = 100.0 + 5.0 * np.arange(12, dtype=float)
        extractor = InsightExtractor(x, y)

        # Path 2
        text_path2 = get_segment_narrative(extractor=extractor, metric="spending")

        # Equivalent Path 1
        suite = extractor.extract_full_suite()
        text_path1 = get_segment_narrative(
            segments=suite["segments"], cv_value=suite["cv_value"], metric="spending"
        )

        assert text_path2 == text_path1

    def test_extractor_and_segments_extractor_wins(self):
        """When both extractor and segments are provided, extractor takes precedence."""
        extractor = _linear_extractor(slope=5.0)
        # Pass a conflicting segments list — it should be ignored
        dummy_segs = [_seg(1900, 1910, slope=-99, start_value=1, end_value=1)]
        text = get_segment_narrative(
            extractor=extractor, segments=dummy_segs, cv_value=99.0, metric="spending"
        )
        # Should not mention 1900 or the dummy negative slope
        assert "1900" not in text
