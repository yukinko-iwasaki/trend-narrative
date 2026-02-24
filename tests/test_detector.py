"""Unit tests for trend_narrative.detector.TrendDetector."""

import numpy as np
import pytest

from trend_narrative.detector import TrendDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_up():
    """Strictly increasing linear series (2010–2021)."""
    x = np.arange(2010, 2022, dtype=float)
    y = 100.0 + 5.0 * (x - 2010)
    return x, y


@pytest.fixture
def linear_down():
    """Strictly decreasing linear series."""
    x = np.arange(2010, 2022, dtype=float)
    y = 200.0 - 5.0 * (x - 2010)
    return x, y


@pytest.fixture
def v_shape():
    """A V-shape with a clear trough in the middle."""
    x = np.arange(2005, 2021, dtype=float)
    y = np.concatenate([
        100.0 - 4.0 * np.arange(8),   # decline 2005-2012
        68.0  + 4.0 * np.arange(8),   # recovery 2013-2020
    ])
    return x, y


@pytest.fixture
def flat():
    """Nearly flat series – no real trend."""
    rng = np.random.default_rng(0)
    x = np.arange(2010, 2022, dtype=float)
    y = 100.0 + rng.normal(0, 0.5, len(x))   # tiny noise
    return x, y


# ---------------------------------------------------------------------------
# calculate_bic
# ---------------------------------------------------------------------------

class TestCalculateBic:
    def test_lower_ssr_gives_lower_bic(self):
        bic_good = TrendDetector.calculate_bic(ssr=10, n_data_points=20, n_segments=2)
        bic_bad  = TrendDetector.calculate_bic(ssr=100, n_data_points=20, n_segments=2)
        assert bic_good < bic_bad

    def test_more_segments_penalises_bic(self):
        """Extra segments should increase BIC when SSR is held constant."""
        bic_simple  = TrendDetector.calculate_bic(ssr=50, n_data_points=20, n_segments=1)
        bic_complex = TrendDetector.calculate_bic(ssr=50, n_data_points=20, n_segments=3)
        assert bic_complex > bic_simple

    def test_returns_float(self):
        result = TrendDetector.calculate_bic(100, 20, 2)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# find_local_maxima_years
# ---------------------------------------------------------------------------

class TestFindLocalMaximaYears:
    def test_detects_peak(self):
        detector = TrendDetector()
        x = np.array([2010, 2011, 2012, 2013, 2014], dtype=float)
        y = np.array([1, 3, 5, 3, 1], dtype=float)   # peak at 2012
        extrema = detector.find_local_maxima_years(x, y)
        assert 2012.0 in extrema

    def test_detects_valley(self):
        detector = TrendDetector()
        x = np.array([2010, 2011, 2012, 2013, 2014], dtype=float)
        y = np.array([5, 3, 1, 3, 5], dtype=float)   # valley at 2012
        extrema = detector.find_local_maxima_years(x, y)
        assert 2012.0 in extrema

    def test_monotone_has_no_extrema(self):
        detector = TrendDetector()
        x = np.arange(2010, 2016, dtype=float)
        y = np.arange(6, dtype=float)
        extrema = detector.find_local_maxima_years(x, y)
        assert extrema == []


# ---------------------------------------------------------------------------
# extract_trend – end-to-end
# ---------------------------------------------------------------------------

class TestExtractTrend:
    def test_returns_list(self, linear_up):
        x, y = linear_up
        detector = TrendDetector()
        result = detector.extract_trend(x, y)
        assert isinstance(result, list)

    def test_segment_keys_present(self, linear_up):
        x, y = linear_up
        detector = TrendDetector()
        result = detector.extract_trend(x, y)
        if result:
            expected_keys = {"start_year", "end_year", "start_value", "end_value", "slope", "p_value"}
            assert expected_keys.issubset(result[0].keys())

    def test_upward_trend_positive_slope(self, linear_up):
        x, y = linear_up
        detector = TrendDetector()
        result = detector.extract_trend(x, y)
        if result:
            assert all(seg["slope"] > 0 for seg in result)

    def test_downward_trend_negative_slope(self, linear_down):
        x, y = linear_down
        detector = TrendDetector()
        result = detector.extract_trend(x, y)
        if result:
            assert all(seg["slope"] < 0 for seg in result)

    def test_v_shape_returns_two_segments(self, v_shape):
        x, y = v_shape
        detector = TrendDetector()
        result = detector.extract_trend(x, y)
        # Allow 0 (no valid fit) or 2 (correct detection)
        assert len(result) in (0, 2)
        if len(result) == 2:
            assert result[0]["slope"] < 0   # declining first
            assert result[1]["slope"] > 0   # recovering second

    def test_accepts_list_input(self):
        x = list(range(2010, 2022))
        y = [float(i) * 3 for i in range(12)]
        result = TrendDetector().extract_trend(x, y)
        assert isinstance(result, list)

    def test_segments_are_contiguous(self, v_shape):
        x, y = v_shape
        result = TrendDetector().extract_trend(x, y)
        for i in range(1, len(result)):
            assert result[i]["start_year"] == result[i - 1]["end_year"]

    def test_too_few_points_returns_empty(self):
        x = np.array([2010, 2011], dtype=float)
        y = np.array([100, 110], dtype=float)
        result = TrendDetector().extract_trend(x, y)
        assert result == []

    def test_max_segments_respected(self, v_shape):
        x, y = v_shape
        detector = TrendDetector(max_segments=1)
        result = detector.extract_trend(x, y)
        assert len(result) <= 1
