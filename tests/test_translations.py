"""Tests for French language support (lang='fr').

Structured in three layers to avoid duplication:

* **Layer 1 – Primitives**: pure helpers (icu_format, _unpack_metric,
  _genitive, _format_percent, millify). Parametrized for coverage density.
* **Layer 2 – Catalog**: spot-checks that analysis keys (e.g. "increased",
  "strength_weak") resolve to correctly-inflected French strings.
* **Layer 3 – Integration**: per-narrative-function smoke tests. Grammar
  correctness is covered in Layers 1/2; these verify wiring only.
"""

import numpy as np
import pytest

from trend_narrative import InsightExtractor, SUPPORTED_LANGUAGES
from trend_narrative.narrative import (
    _format_percent,
    get_segment_narrative,
    millify,
)
from trend_narrative.relationship_analysis import (
    get_correlation_strength,
    get_direction,
)
from trend_narrative.relationship_narrative import (
    _genitive,
    _resolve_time_unit,
    _time_unit_comparison,
    get_relationship_narrative,
)
from trend_narrative.translations import _unpack_metric, get_translations, icu_format


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


def _strong_signal_series(n=20, seed=42):
    """Two correlated series with enough points for the lagged-correlation path."""
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n, dtype=float)
    ref = 100 + 5 * np.arange(n, dtype=float) + rng.normal(0, 0.5, n)
    comp = 50 + 3 * np.arange(n, dtype=float) + rng.normal(0, 0.5, n)
    return years, ref, comp


# ===========================================================================
# Layer 1: Primitives
# ===========================================================================


class TestUnpackMetric:
    """``_unpack_metric`` returns ICU-ready kwargs directly — the
    user-facing ``plural``/``feminine`` booleans become ``number``/``gender``."""

    @pytest.mark.parametrize("metric, name, icu_kwargs", [
        ("real expenditure", "real expenditure",
         {"number": "singular", "gender": "masculine"}),
        ({"name": "les dépenses", "plural": True, "feminine": True},
         "les dépenses", {"number": "plural", "gender": "feminine"}),
        ({"name": "les prix", "plural": True},
         "les prix", {"number": "plural", "gender": "masculine"}),
        ({"name": "spending"}, "spending",
         {"number": "singular", "gender": "masculine"}),
        ({"name": "x", "plural": True, "extra": "ignored"},
         "x", {"number": "plural", "gender": "masculine"}),
    ])
    def test_unpack(self, metric, name, icu_kwargs):
        assert _unpack_metric(metric) == (name, icu_kwargs)

    def test_dict_missing_name_raises(self):
        with pytest.raises(TypeError, match="must include a 'name' key"):
            _unpack_metric({"plural": True})

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="must be a str or dict"):
            _unpack_metric(42)


class TestIcuFormat:
    @pytest.mark.parametrize("lang, key, kwargs, expected", [
        # English has no select blocks → returned unchanged.
        ("en", "increased", {"number": "plural"}, "increased"),
        # French avoir verbs: number only.
        ("fr", "increased", {"number": "singular"}, "a augmenté"),
        ("fr", "increased", {"number": "plural"}, "ont augmenté"),
        ("fr", "decreased", {"number": "plural"}, "ont diminué"),
        # French être verbs: nested number × gender.
        ("fr", "remained_stable",
         {"number": "singular", "gender": "masculine"}, "est resté stable"),
        ("fr", "remained_stable",
         {"number": "singular", "gender": "feminine"}, "est restée stable"),
        ("fr", "remained_stable",
         {"number": "plural", "gender": "masculine"}, "sont restés stables"),
        ("fr", "remained_stable",
         {"number": "plural", "gender": "feminine"}, "sont restées stables"),
    ])
    def test_inflected_strings_resolve(self, lang, key, kwargs, expected):
        assert icu_format(get_translations(lang)[key], **kwargs) == expected

    def test_plain_string_passthrough(self):
        assert icu_format("plain {metric}", number="plural") == "plain {metric}"

    def test_unknown_select_value_falls_back_to_other(self):
        t = get_translations("fr")
        assert icu_format(t["increased"], number="bogus") == "ont augmenté"

    def test_preserves_format_placeholders(self):
        """ICU select resolves; {metric} stays for subsequent .format()."""
        t = get_translations("fr")
        assert "{metric}" in icu_format(t["vol_moderate"], number="singular")

    def test_preserves_format_specs(self):
        """{corr:.2f} format spec must survive select resolution."""
        template = "{number, select, singular {r={corr:.2f}} other {R={corr:.2f}}}"
        assert icu_format(template, number="singular") == "r={corr:.2f}"


class TestGenitive:
    """French article contractions (de+le→du, de+les→des) and elision."""

    @pytest.mark.parametrize("lang, name, expected", [
        # French article contractions
        ("fr", "les dépenses", "des dépenses"),
        ("fr", "le taux", "du taux"),
        ("fr", "la relation", "de la relation"),
        ("fr", "l'indice", "de l'indice"),
        # French bare-noun elision before vowel
        ("fr", "économies", "d'économies"),
        ("fr", "prix", "de prix"),
        # English
        ("en", "spending", "of spending"),
        # Unimplemented language: passthrough
        ("de", "die Ausgaben", "die Ausgaben"),
        # Edge: empty
        ("fr", "", ""),
        ("en", "", ""),
    ])
    def test_genitive(self, lang, name, expected):
        assert _genitive(lang, name) == expected


class TestResolveTimeUnit:
    """Known time_unit keys come from the catalog; unknown keys use the
    catalog's fallback suffix — never a hardcoded English ``s``."""

    @pytest.mark.parametrize("lang, unit, count, expected", [
        # Catalogued units: catalog wins.
        ("en", "year", 1, "year"),
        ("en", "year", 5, "years"),
        ("fr", "year", 1, "année"),
        ("fr", "year", 5, "années"),
        # Unknown units: English appends "s", French passes through.
        # This prevents "fortnights" appearing inside French prose.
        ("en", "fortnight", 1, "fortnight"),
        ("en", "fortnight", 5, "fortnights"),
        ("fr", "quinzaine", 1, "quinzaine"),
        ("fr", "quinzaine", 5, "quinzaine"),
    ])
    def test_resolve(self, lang, unit, count, expected):
        assert _resolve_time_unit(get_translations(lang), unit, count) == expected


class TestTimeUnitComparison:
    @pytest.mark.parametrize("lang, unit, expected", [
        ("fr", "année", "d'année en année"),       # vowel → elision
        ("fr", "mois", "de mois en mois"),          # consonant
        ("fr", "semaine", "de semaine en semaine"),
        ("fr", "trimestre", "de trimestre en trimestre"),
        ("en", "year", "year-over-year"),
        ("en", "month", "month-over-month"),
    ])
    def test_time_unit_comparison(self, lang, unit, expected):
        assert _time_unit_comparison(lang, unit) == expected


class TestFormatPercent:
    @pytest.mark.parametrize("value, lang, expected", [
        (250.0, "en", "+250.00%"),
        (-50.0, "en", "-50.00%"),
        (0.0, "en", "+0.00%"),
        (250.0, "fr", "+250,00 %"),
        (-50.0, "fr", "-50,00 %"),
        (0.0, "fr", "+0,00 %"),
    ])
    def test_format_percent(self, value, lang, expected):
        assert _format_percent(value, lang=lang) == expected


class TestMillify:
    @pytest.mark.parametrize("n, lang, expected", [
        # English
        (750, "en", "750.00"),
        (1_500, "en", "1.50 K"),
        (2_000_000, "en", "2.00 M"),
        (3_000_000_000, "en", "3.00 B"),
        (0, "en", "0.00"),
        # French: comma decimal, "Md" for 10^9 (NOT "B" — "billion" in
        # French means 10^12, a false friend with English).
        (1_500, "fr", "1,50 k"),
        (2_000_000, "fr", "2,00 M"),
        (3_000_000_000, "fr", "3,00 Md"),
        (750, "fr", "750,00"),
        (0, "fr", "0,00"),
    ])
    def test_millify(self, n, lang, expected):
        assert millify(n, lang=lang) == expected

    def test_default_lang_is_english(self):
        """Backward compat: existing callers omit lang."""
        assert millify(1_500_000) == "1.50 M"

    def test_billions_french_never_uses_B(self):
        """Regression guard for the milliard/billion false friend."""
        assert "B" not in millify(3_000_000_000, lang="fr")


# ===========================================================================
# Layer 2: Catalog
# ===========================================================================


class TestCatalogModule:
    def test_supported_languages(self):
        assert {"en", "fr"} <= set(SUPPORTED_LANGUAGES)

    def test_get_translations_en(self):
        assert get_translations("en")["increased"] == "increased"

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            get_translations("xx")


class TestDirectionFrench:
    """End-to-end: get_direction returns a key → catalog → ICU resolves to French."""

    @pytest.mark.parametrize("values, number, gender, expected", [
        # avoir verbs (number only)
        ([100, 200], "singular", "masculine", "a augmenté"),
        ([100, 200], "plural", "masculine", "ont augmenté"),
        ([200, 100], "singular", "masculine", "a diminué"),
        # être verbs (number + gender)
        ([100, 100], "singular", "masculine", "est resté stable"),
        ([100, 100], "singular", "feminine", "est restée stable"),
        ([100, 100], "plural", "feminine", "sont restées stables"),
        # Edge: not enough data → unknown
        ([100], "singular", "masculine", "inconnu"),
    ])
    def test_direction_resolves(self, values, number, gender, expected):
        t = get_translations("fr")
        key = get_direction(np.array(values))
        assert icu_format(t[key], number=number, gender=gender) == expected


class TestCorrelationStrengthFrench:
    @pytest.mark.parametrize("corr, expected", [
        (0.05, "aucune"),
        (0.2, "faible"),
        (0.4, "modérée"),
        (0.6, "forte"),
        (0.9, "très forte"),
    ])
    def test_strength_resolves(self, corr, expected):
        assert get_translations("fr")[get_correlation_strength(corr)] == expected


# ===========================================================================
# Layer 3: Integration
# Smoke tests verifying each narrative path wires through the localization
# correctly. Grammar correctness is Layer 1/2's job.
# ===========================================================================


class TestSegmentNarrativeIntegration:
    def test_default_lang_is_english(self):
        segs = [_seg(2010, 2020, slope=10, start_value=100, end_value=200)]
        text = get_segment_narrative(segments=segs, cv_value=8.0, metric="spending")
        assert "increased" in text

    @pytest.mark.parametrize("cv, expected_substring", [
        (2.0, "stable"),         # vol_low
        (10.0, "fluctuations"),  # vol_moderate
        (25.0, "volatilité"),    # vol_high
    ])
    def test_volatility_fallback_french(self, cv, expected_substring):
        text = get_segment_narrative(
            segments=[], cv_value=cv, metric="dépenses", lang="fr",
        )
        assert expected_substring in text.lower()

    def test_single_segment_localized_numbers_french(self):
        """End-to-end: comma decimal, space before %, 'Md' for billions."""
        segs = [_seg(2010, 2020, slope=3e8, start_value=1e9, end_value=4e9)]
        text = get_segment_narrative(
            segments=segs, cv_value=8.0,
            metric={"name": "les dépenses", "plural": True, "feminine": True},
            lang="fr",
        )
        assert "ont augmenté" in text
        assert " Md" in text
        assert "+300,00 %" in text

    def test_multi_segment_peak_transition_french(self):
        segs = [
            _seg(2010, 2015, slope=10, start_value=100, end_value=150),
            _seg(2015, 2020, slope=-5, start_value=150, end_value=125),
        ]
        text = get_segment_narrative(
            segments=segs, cv_value=10.0,
            metric={"name": "les dépenses", "plural": True},
            lang="fr",
        )
        assert "ont affiché" in text
        assert "pic" in text.lower()

    def test_metric_dict_safe_for_english(self):
        """Grammar flags on the dict are silently ignored in English."""
        segs = [_seg(2010, 2020, slope=10, start_value=100, end_value=200)]
        text = get_segment_narrative(
            segments=segs, cv_value=8.0,
            metric={"name": "expenditures", "plural": True}, lang="en",
        )
        assert "increased" in text and "expenditures" in text

    def test_extractor_path_with_french(self):
        x = np.arange(2010, 2022, dtype=float)
        y = 100.0 + 5.0 * np.arange(12, dtype=float)
        text = get_segment_narrative(
            extractor=InsightExtractor(x, y), metric="dépenses", lang="fr",
        )
        assert text and isinstance(text, str)


class TestRelationshipNarrativeIntegration:
    SHORT_YEARS = np.array([2010, 2020])
    SHORT_VALUES = np.array([100, 150])

    def test_default_lang_is_english(self):
        result = get_relationship_narrative(
            reference_years=self.SHORT_YEARS, reference_values=self.SHORT_VALUES,
            comparison_years=self.SHORT_YEARS, comparison_values=self.SHORT_VALUES,
            reference_name="spending", comparison_name="outcome",
        )
        assert "cannot be determined" in result["narrative"]

    def test_insufficient_data_french(self):
        result = get_relationship_narrative(
            reference_years=self.SHORT_YEARS, reference_values=self.SHORT_VALUES,
            comparison_years=self.SHORT_YEARS, comparison_values=self.SHORT_VALUES,
            reference_name="dépenses", comparison_name="résultat", lang="fr",
        )
        assert result["method"] == "insufficient_data"
        assert "ne peut être déterminée" in result["narrative"]

    def test_comovement_french_with_dict_metrics(self):
        result = get_relationship_narrative(
            reference_years=np.array([2010, 2015, 2020]),
            reference_values=np.array([100, 125, 150], dtype=float),
            comparison_years=np.array([2012, 2015, 2018]),
            comparison_values=np.array([50, 65, 80], dtype=float),
            reference_name={"name": "les dépenses de santé",
                            "plural": True, "feminine": True},
            comparison_name={"name": "les indices", "plural": True},
            lang="fr",
        )
        assert result["method"] == "comovement"
        assert "ont augmenté" in result["narrative"]

    @pytest.mark.parametrize("ref, comp, leader_verb, follower_verb", [
        # significant_finding has two independent subject-verb pairs;
        # each verb must agree with its own subject's number.
        (
            {"name": "les dépenses", "plural": True, "feminine": True},
            {"name": "l'indice", "plural": False},
            "les dépenses augmentent", "l'indice tend",
        ),
        (
            {"name": "l'indice", "plural": False},
            {"name": "les taux", "plural": True},
            "l'indice augmente", "les taux tendent",
        ),
    ])
    def test_significant_finding_subject_verb_agreement(
        self, ref, comp, leader_verb, follower_verb,
    ):
        years, ref_values, comp_values = _strong_signal_series()
        result = get_relationship_narrative(
            reference_years=years, reference_values=ref_values,
            comparison_years=years, comparison_values=comp_values,
            reference_name=ref, comparison_name=comp,
            time_unit="year", max_lag_cap=0, lang="fr",
        )
        assert leader_verb in result["narrative"]
        assert follower_verb in result["narrative"]

    def test_no_de_les_or_de_le_leaks_in_french_output(self):
        """Regression: French genitive contractions must always fire when an
        article-prefixed metric flows into the no_reliable_relationship path."""
        rng = np.random.default_rng(42)
        n = 15
        years = np.arange(2000, 2000 + n, dtype=float)
        result = get_relationship_narrative(
            reference_years=years, reference_values=rng.normal(100, 10, n),
            comparison_years=years, comparison_values=rng.normal(50, 5, n),
            reference_name={"name": "les dépenses", "plural": True, "feminine": True},
            comparison_name={"name": "le taux", "plural": False},
            lang="fr",
        )
        narrative = result["narrative"]
        assert "de les" not in narrative
        assert "de le " not in narrative

    def test_timing_same_gender_agreement_french(self):
        """time_unit='year' is feminine → 'la même année', not 'du même année'."""
        years, ref, comp = _strong_signal_series(n=15)
        result = get_relationship_narrative(
            reference_years=years, reference_values=ref,
            comparison_years=years, comparison_values=comp,
            reference_name="dépenses", comparison_name="résultat",
            time_unit="year", lang="fr",
        )
        narrative = result["narrative"]
        if "même" in narrative:
            assert "la même année" in narrative
            assert "du même" not in narrative
