"""Unit tests for French language support (lang='fr')."""

import numpy as np
import pytest

from trend_narrative import InsightExtractor, SUPPORTED_LANGUAGES
from trend_narrative.narrative import get_segment_narrative
from trend_narrative.relationship_narrative import get_relationship_narrative
from trend_narrative.relationship_analysis import get_direction, get_correlation_strength
from trend_narrative.translations import get_translations


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
# translations module
# ---------------------------------------------------------------------------

class TestTranslationsModule:
    def test_supported_languages(self):
        assert "en" in SUPPORTED_LANGUAGES
        assert "fr" in SUPPORTED_LANGUAGES

    def test_get_translations_en(self):
        t = get_translations("en")
        assert t["increased"] == "increased"

    def test_get_translations_fr(self):
        t = get_translations("fr")
        assert t["increased"] == "a augmenté"

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            get_translations("xx")


# ---------------------------------------------------------------------------
# Direction / correlation strength keys resolved via the French catalog.
#
# get_direction / get_correlation_strength are language-neutral — they return
# stable keys. These tests verify the end-to-end catalog lookup a narrative
# layer would perform.
# ---------------------------------------------------------------------------

class TestDirectionFrench:
    def _fr(self, values):
        return get_translations("fr")[get_direction(np.array(values))]

    def test_increased(self):
        assert self._fr([100.0, 200.0]) == "a augmenté"

    def test_decreased(self):
        assert self._fr([200.0, 100.0]) == "a diminué"

    def test_stable(self):
        assert self._fr([100.0, 100.0]) == "est resté stable"

    def test_unknown(self):
        assert self._fr([100.0]) == "inconnu"


class TestCorrelationStrengthFrench:
    def _fr(self, corr):
        return get_translations("fr")[get_correlation_strength(corr)]

    def test_no_correlation(self):
        assert self._fr(0.05) == "aucune"

    def test_weak(self):
        assert self._fr(0.2) == "faible"

    def test_moderate(self):
        assert self._fr(0.4) == "modérée"

    def test_strong(self):
        assert self._fr(0.6) == "forte"

    def test_very_strong(self):
        assert self._fr(0.9) == "très forte"


# ---------------------------------------------------------------------------
# get_segment_narrative with lang='fr'
# ---------------------------------------------------------------------------

class TestSegmentNarrativeFrench:
    def test_no_segments_low_cv(self):
        text = get_segment_narrative(segments=[], cv_value=2.0, metric="dépenses", lang="fr")
        assert "stable" in text.lower()

    def test_no_segments_moderate_cv(self):
        text = get_segment_narrative(segments=[], cv_value=10.0, metric="dépenses", lang="fr")
        assert "fluctuations" in text.lower()

    def test_no_segments_high_cv(self):
        text = get_segment_narrative(segments=[], cv_value=25.0, metric="dépenses", lang="fr")
        assert "volatilité" in text.lower()

    def test_single_upward_segment(self):
        segs = [_seg(2010, 2020, slope=10, start_value=100, end_value=200)]
        text = get_segment_narrative(segments=segs, cv_value=8.0, metric="dépenses de santé", lang="fr")
        assert "a augmenté" in text
        assert "2010" in text and "2020" in text

    def test_single_downward_segment(self):
        segs = [_seg(2010, 2020, slope=-5, start_value=200, end_value=100)]
        text = get_segment_narrative(segments=segs, cv_value=8.0, metric="dépenses", lang="fr")
        assert "a diminué" in text

    def test_metric_label_in_output(self):
        segs = [_seg(2010, 2018, slope=3, start_value=100, end_value=124)]
        text = get_segment_narrative(segments=segs, cv_value=5.0, metric="budget éducation", lang="fr")
        assert "budget éducation" in text

    def test_peak_transition(self):
        segs = [
            _seg(2010, 2015, slope=10, start_value=100, end_value=150),
            _seg(2015, 2020, slope=-5, start_value=150, end_value=125),
        ]
        text = get_segment_narrative(segments=segs, cv_value=10.0, metric="dépenses", lang="fr")
        assert "pic" in text.lower()

    def test_trough_transition(self):
        segs = [
            _seg(2010, 2015, slope=-5, start_value=150, end_value=125),
            _seg(2015, 2020, slope=10, start_value=125, end_value=175),
        ]
        text = get_segment_narrative(segments=segs, cv_value=10.0, metric="dépenses", lang="fr")
        assert "creux" in text.lower() or "reprise" in text.lower()

    def test_default_lang_is_english(self):
        """Omitting lang should produce English output (backward compat)."""
        segs = [_seg(2010, 2020, slope=10, start_value=100, end_value=200)]
        text = get_segment_narrative(segments=segs, cv_value=8.0, metric="spending")
        assert "increased" in text

    def test_extractor_path_with_french(self):
        x = np.arange(2010, 2022, dtype=float)
        y = 100.0 + 5.0 * np.arange(12, dtype=float)
        extractor = InsightExtractor(x, y)
        text = get_segment_narrative(extractor=extractor, metric="dépenses", lang="fr")
        assert isinstance(text, str)
        assert len(text) > 0


# ---------------------------------------------------------------------------
# get_relationship_narrative with lang='fr'
# ---------------------------------------------------------------------------

class TestRelationshipNarrativeFrench:
    years = np.array([2010, 2020])
    values = np.array([100, 150])
    enough_years = np.array([2010, 2011, 2012, 2013, 2014, 2015, 2016])
    enough_values = np.array([50, 58, 62, 72, 75, 88, 90], dtype=float)

    def test_insufficient_data_french(self):
        result = get_relationship_narrative(
            reference_years=self.years,
            reference_values=self.values,
            comparison_years=self.years,
            comparison_values=self.values,
            reference_name="dépenses",
            comparison_name="résultat",
            lang="fr",
        )
        assert result["method"] == "insufficient_data"
        assert "ne peut être déterminée" in result["narrative"]

    def test_comovement_french(self):
        years_3pt = np.array([2010, 2015, 2020])
        ref_increasing = np.array([100, 125, 150], dtype=float)
        comp_years_3pt = np.array([2012, 2015, 2018])
        comp_increasing = np.array([50, 65, 80], dtype=float)

        result = get_relationship_narrative(
            reference_years=years_3pt,
            reference_values=ref_increasing,
            comparison_years=comp_years_3pt,
            comparison_values=comp_increasing,
            reference_name="dépenses de santé",
            comparison_name="indice CSU",
            lang="fr",
        )
        assert result["method"] == "comovement"
        assert "dépenses de santé" in result["narrative"]

    def test_lagged_correlation_french(self):
        np.random.seed(42)
        n = 15
        ref_years = np.arange(2000, 2000 + n, dtype=float)
        ref_values = 100 + 5 * np.arange(n, dtype=float) + np.random.normal(0, 2, n)
        comp_years = ref_years.copy()
        comp_values = 50 + 3 * np.arange(n, dtype=float) + np.random.normal(0, 2, n)

        result = get_relationship_narrative(
            reference_years=ref_years,
            reference_values=ref_values,
            comparison_years=comp_years,
            comparison_values=comp_values,
            reference_name="dépenses",
            comparison_name="résultat",
            lang="fr",
        )
        assert result["method"] == "lagged_correlation"
        # Should contain French text
        narrative = result["narrative"]
        assert isinstance(narrative, str)
        assert len(narrative) > 0

    def test_default_lang_is_english(self):
        """Omitting lang should produce English (backward compat)."""
        result = get_relationship_narrative(
            reference_years=self.years,
            reference_values=self.values,
            comparison_years=self.years,
            comparison_values=self.values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert "cannot be determined" in result["narrative"]
