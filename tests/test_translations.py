"""Unit tests for French language support (lang='fr')."""

import numpy as np
import pytest

from trend_narrative import InsightExtractor, SUPPORTED_LANGUAGES
from trend_narrative.narrative import get_segment_narrative
from trend_narrative.relationship_narrative import get_relationship_narrative
from trend_narrative.relationship_analysis import get_direction, get_correlation_strength
from trend_narrative.translations import get_translations, icu_format
from trend_narrative.translations import _unpack_metric


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
        # FR direction words use ICU select; default singular form
        assert icu_format(t["increased"], number="singular") == "a augmenté"

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
    def _fr(self, values, number="singular", gender="masculine"):
        t = get_translations("fr")
        key = get_direction(np.array(values))
        return icu_format(t[key], number=number, gender=gender)

    def test_increased(self):
        assert self._fr([100.0, 200.0]) == "a augmenté"

    def test_decreased(self):
        assert self._fr([200.0, 100.0]) == "a diminué"

    def test_stable(self):
        assert self._fr([100.0, 100.0]) == "est resté stable"

    def test_unknown(self):
        assert self._fr([100.0]) == "inconnu"

    def test_increased_plural(self):
        assert self._fr([100.0, 200.0], number="plural") == "ont augmenté"

    def test_decreased_plural(self):
        assert self._fr([200.0, 100.0], number="plural") == "ont diminué"

    def test_stable_feminine_singular(self):
        assert self._fr([100.0, 100.0], gender="feminine") == "est restée stable"

    def test_stable_masculine_plural(self):
        assert self._fr([100.0, 100.0], number="plural") == "sont restés stables"

    def test_stable_feminine_plural(self):
        assert self._fr(
            [100.0, 100.0], number="plural", gender="feminine",
        ) == "sont restées stables"


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

    def test_single_upward_segment_plural(self):
        segs = [_seg(2010, 2020, slope=10, start_value=100, end_value=200)]
        text = get_segment_narrative(
            segments=segs, cv_value=8.0,
            metric={"name": "les dépenses", "plural": True},
            lang="fr",
        )
        assert "ont augmenté" in text

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

    def test_volatility_low_plural(self):
        text = get_segment_narrative(
            segments=[], cv_value=2.0,
            metric={"name": "les prix", "plural": True},
            lang="fr",
        )
        assert "sont restés" in text

    def test_volatility_low_feminine_plural(self):
        text = get_segment_narrative(
            segments=[], cv_value=2.0,
            metric={"name": "les dépenses", "plural": True, "feminine": True},
            lang="fr",
        )
        assert "sont restées" in text

    def test_volatility_moderate_plural(self):
        text = get_segment_narrative(
            segments=[], cv_value=10.0,
            metric={"name": "les prix", "plural": True},
            lang="fr",
        )
        assert "ont montré" in text

    def test_multi_segment_plural(self):
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

    def test_metric_dict_ignored_for_english(self):
        """A dict metric should still work in English — grammar flags are ignored."""
        segs = [_seg(2010, 2020, slope=10, start_value=100, end_value=200)]
        text = get_segment_narrative(
            segments=segs, cv_value=8.0,
            metric={"name": "expenditures", "plural": True},
            lang="en",
        )
        assert "increased" in text
        assert "expenditures" in text

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


# ---------------------------------------------------------------------------
# Elision & time_unit_comparison
# ---------------------------------------------------------------------------

class TestFrenchElision:
    """Verify correct preposition elision (d' vs de) in French narratives."""

    def test_elision_with_vowel_unit(self):
        """'année' starts with a vowel → d'année en année."""
        from trend_narrative.relationship_narrative import _time_unit_comparison
        assert _time_unit_comparison("fr", "année") == "d'année en année"

    def test_no_elision_with_consonant_unit(self):
        """'mois' starts with a consonant → de mois en mois."""
        from trend_narrative.relationship_narrative import _time_unit_comparison
        assert _time_unit_comparison("fr", "mois") == "de mois en mois"

    def test_english_always_uses_over(self):
        from trend_narrative.relationship_narrative import _time_unit_comparison
        assert _time_unit_comparison("en", "year") == "year-over-year"
        assert _time_unit_comparison("en", "month") == "month-over-month"

    def test_elision_with_semaine(self):
        """'semaine' starts with a consonant → de semaine en semaine."""
        from trend_narrative.relationship_narrative import _time_unit_comparison
        assert _time_unit_comparison("fr", "semaine") == "de semaine en semaine"

    def test_elision_with_trimestre(self):
        from trend_narrative.relationship_narrative import _time_unit_comparison
        assert _time_unit_comparison("fr", "trimestre") == "de trimestre en trimestre"


# ---------------------------------------------------------------------------
# Genitive ("of X") form — French contractions and language dispatch
# ---------------------------------------------------------------------------

class TestGenitive:
    """The _genitive() helper dispatches per language.

    French: handles article contractions (de+le→du, de+les→des) and vowel
    elision. English: prepends "of ". Unknown languages: passthrough.
    """

    def test_fr_de_plus_les_contracts_to_des(self):
        from trend_narrative.relationship_narrative import _genitive
        assert _genitive("fr", "les dépenses") == "des dépenses"

    def test_fr_de_plus_le_contracts_to_du(self):
        from trend_narrative.relationship_narrative import _genitive
        assert _genitive("fr", "le taux") == "du taux"

    def test_fr_de_plus_la_no_contraction(self):
        from trend_narrative.relationship_narrative import _genitive
        assert _genitive("fr", "la relation") == "de la relation"

    def test_fr_de_plus_elided_article_no_contraction(self):
        from trend_narrative.relationship_narrative import _genitive
        assert _genitive("fr", "l'indice") == "de l'indice"

    def test_fr_de_elides_before_vowel(self):
        from trend_narrative.relationship_narrative import _genitive
        assert _genitive("fr", "économies") == "d'économies"

    def test_fr_de_stays_before_consonant(self):
        from trend_narrative.relationship_narrative import _genitive
        assert _genitive("fr", "prix") == "de prix"

    def test_en_prepends_of(self):
        from trend_narrative.relationship_narrative import _genitive
        assert _genitive("en", "spending") == "of spending"
        assert _genitive("en", "the rate") == "of the rate"

    def test_unknown_language_passthrough(self):
        from trend_narrative.relationship_narrative import _genitive
        # Languages with no implementation return the name unchanged.
        assert _genitive("de", "die Ausgaben") == "die Ausgaben"

    def test_empty_string(self):
        from trend_narrative.relationship_narrative import _genitive
        assert _genitive("fr", "") == ""
        assert _genitive("en", "") == ""


# ---------------------------------------------------------------------------
# "de les" contractions in full narratives (integration-style)
# ---------------------------------------------------------------------------

class TestFrenchContractionsInNarratives:
    """Verify the affected templates produce properly contracted French."""

    def _weak_correlation_data(self):
        """Generate two uncorrelated series to trigger no_reliable_relationship."""
        np.random.seed(42)
        n = 15
        years = np.arange(2000, 2000 + n, dtype=float)
        # Pure noise series → no meaningful correlation
        ref_values = np.random.normal(100, 10, n)
        comp_values = np.random.normal(50, 5, n)
        return years, ref_values, comp_values

    def test_no_reliable_relationship_plural_metrics(self):
        """Plural feminine metrics should contract de+les → des."""
        years, ref_values, comp_values = self._weak_correlation_data()

        result = get_relationship_narrative(
            reference_years=years,
            reference_values=ref_values,
            comparison_years=years,
            comparison_values=comp_values,
            reference_name={
                "name": "les dépenses",
                "plural": True,
                "feminine": True,
            },
            comparison_name={
                "name": "les indices",
                "plural": True,
                "feminine": False,
            },
            lang="fr",
        )
        narrative = result["narrative"]
        # Only check contraction if the template actually fired
        if "variations" in narrative:
            assert "de les" not in narrative
            assert "des dépenses" in narrative or "des indices" in narrative

    def test_no_reliable_relationship_masculine_singular(self):
        """Masculine singular metric with 'le' should contract de+le → du."""
        years, ref_values, comp_values = self._weak_correlation_data()

        result = get_relationship_narrative(
            reference_years=years,
            reference_values=ref_values,
            comparison_years=years,
            comparison_values=comp_values,
            reference_name={"name": "le taux", "plural": False, "feminine": False},
            comparison_name={"name": "l'indice", "plural": False, "feminine": False},
            lang="fr",
        )
        narrative = result["narrative"]
        # "de le" should never leak through regardless of which template fires
        assert "de le " not in narrative
        # If the no_reliable_relationship template fired, check contractions
        if "variations" in narrative:
            assert "du taux" in narrative
            assert "de l'indice" in narrative

    def test_de_helper_directly_on_templates(self):
        """Direct test: format the affected templates with contracted values."""
        from trend_narrative.translations import get_translations
        from trend_narrative.relationship_narrative import _genitive
        t = get_translations("fr")

        metric_plural_fem = "les dépenses"
        metric_singular_masc = "le taux"

        # no_reliable_relationship
        text = t["no_reliable_relationship"].format(
            x=metric_plural_fem, y=metric_singular_masc,
            x_gen=_genitive("fr", metric_plural_fem),
            y_gen=_genitive("fr", metric_singular_masc),
        )
        assert "des dépenses" in text
        assert "du taux" in text
        assert "de les" not in text
        assert "de le " not in text

        # no_data_available
        text = t["no_data_available"].format(
            x=metric_plural_fem, y=metric_singular_masc,
            x_gen=_genitive("fr", metric_plural_fem),
            y_gen=_genitive("fr", metric_singular_masc),
        )
        assert "du taux" in text  # "les données du taux"


# ---------------------------------------------------------------------------
# significant_finding — leader and follower can have independent grammar
# ---------------------------------------------------------------------------

class TestSignificantFindingIndependentGrammar:
    """The template has two subject-verb pairs whose subjects can differ
    in number. Each verb must agree with its own subject."""

    def _sig_narrative(self, ref_name, comp_name):
        """Build a significant-finding narrative with lag=0, year."""
        np.random.seed(42)
        n = 20
        years = np.arange(2000, 2000 + n, dtype=float)
        ref_values = 100 + 5 * np.arange(n, dtype=float) + np.random.normal(0, 0.5, n)
        comp_values = 50 + 3 * np.arange(n, dtype=float) + np.random.normal(0, 0.5, n)
        return get_relationship_narrative(
            reference_years=years, reference_values=ref_values,
            comparison_years=years, comparison_values=comp_values,
            reference_name=ref_name, comparison_name=comp_name,
            time_unit="year", max_lag_cap=0, lang="fr",
        )["narrative"]

    def test_plural_leader_singular_follower(self):
        """Leader plural → 'augmentent'; follower singular → 'tend' (not 'tendent')."""
        narrative = self._sig_narrative(
            ref_name={"name": "les dépenses", "plural": True, "feminine": True},
            comp_name={"name": "l'indice", "plural": False, "feminine": False},
        )
        assert "les dépenses augmentent" in narrative
        assert "l'indice tend à" in narrative
        assert "l'indice tendent" not in narrative

    def test_singular_leader_plural_follower(self):
        """Leader singular → 'augmente'; follower plural → 'tendent'."""
        narrative = self._sig_narrative(
            ref_name={"name": "l'indice", "plural": False, "feminine": False},
            comp_name={"name": "les taux", "plural": True, "feminine": False},
        )
        assert "l'indice augmente" in narrative
        assert "les taux tendent" in narrative
        assert "les taux tend à" not in narrative

    def test_both_plural(self):
        narrative = self._sig_narrative(
            ref_name={"name": "les dépenses", "plural": True, "feminine": True},
            comp_name={"name": "les taux", "plural": True, "feminine": False},
        )
        assert "les dépenses augmentent" in narrative
        assert "les taux tendent" in narrative

    def test_both_singular(self):
        narrative = self._sig_narrative(
            ref_name={"name": "l'indice", "plural": False, "feminine": False},
            comp_name={"name": "la production", "plural": False, "feminine": True},
        )
        assert "l'indice augmente" in narrative
        assert "la production tend à" in narrative


# ---------------------------------------------------------------------------
# timing_same — gender agreement with time_unit
# ---------------------------------------------------------------------------

class TestTimingSameGender:
    """timing_same must use 'la même' for feminine and 'du même' for masculine."""

    def _make_lagged_narrative(self, time_unit):
        """Helper: produce a lag=0 significant-correlation narrative."""
        np.random.seed(42)
        n = 15
        years = np.arange(2000, 2000 + n, dtype=float)
        # Strong positive correlation, no lag
        ref_values = 100 + 5 * np.arange(n, dtype=float) + np.random.normal(0, 1, n)
        comp_values = 50 + 3 * np.arange(n, dtype=float) + np.random.normal(0, 1, n)
        return get_relationship_narrative(
            reference_years=years, reference_values=ref_values,
            comparison_years=years, comparison_values=comp_values,
            reference_name="dépenses", comparison_name="résultat",
            time_unit=time_unit, lang="fr",
        )["narrative"]

    def test_feminine_year_uses_la_meme(self):
        narrative = self._make_lagged_narrative("year")
        # Should use "la même" for feminine année
        if "la même" in narrative or "du même" in narrative:
            # Only one or the other should appear — check feminine
            assert "la même année" in narrative
            assert "du même" not in narrative

    def test_masculine_month_uses_du_meme(self):
        narrative = self._make_lagged_narrative("month")
        # Should use "du même" for masculine mois
        if "la même" in narrative or "du même" in narrative:
            assert "du même mois" in narrative
            assert "la même mois" not in narrative

    def test_masculine_quarter_uses_du_meme(self):
        narrative = self._make_lagged_narrative("quarter")
        if "la même" in narrative or "du même" in narrative:
            assert "du même trimestre" in narrative
            assert "la même trimestre" not in narrative

    def test_english_unaffected(self):
        """English timing_same has no gender select and should render normally."""
        np.random.seed(42)
        n = 15
        years = np.arange(2000, 2000 + n, dtype=float)
        ref_values = 100 + 5 * np.arange(n, dtype=float) + np.random.normal(0, 1, n)
        comp_values = 50 + 3 * np.arange(n, dtype=float) + np.random.normal(0, 1, n)
        narrative = get_relationship_narrative(
            reference_years=years, reference_values=ref_values,
            comparison_years=years, comparison_values=comp_values,
            reference_name="spending", comparison_name="outcome",
            time_unit="year", lang="en",
        )["narrative"]
        # English produces "in the same year"
        if "same" in narrative:
            assert "in the same year" in narrative


# ---------------------------------------------------------------------------
# resolve_inflected
# ---------------------------------------------------------------------------

class TestIcuFormat:
    def test_plain_string_passthrough(self):
        assert icu_format("plain {metric} string", number="plural") == "plain {metric} string"

    def test_english_no_select(self):
        t = get_translations("en")
        assert icu_format(t["increased"], number="plural") == "increased"

    def test_singular_select(self):
        t = get_translations("fr")
        assert icu_format(t["increased"], number="singular") == "a augmenté"

    def test_plural_select(self):
        t = get_translations("fr")
        assert icu_format(t["increased"], number="plural") == "ont augmenté"

    def test_nested_select_all_forms(self):
        t = get_translations("fr")
        assert icu_format(t["remained_stable"], number="singular", gender="masculine") == "est resté stable"
        assert icu_format(t["remained_stable"], number="singular", gender="feminine") == "est restée stable"
        assert icu_format(t["remained_stable"], number="plural", gender="masculine") == "sont restés stables"
        assert icu_format(t["remained_stable"], number="plural", gender="feminine") == "sont restées stables"

    def test_default_falls_to_other(self):
        """Unknown value for select variable should fall back to 'other'."""
        t = get_translations("fr")
        assert icu_format(t["increased"], number="unknown_value") == "ont augmenté"

    def test_preserves_format_placeholders(self):
        """ICU select should resolve but leave {metric} etc. for .format()."""
        t = get_translations("fr")
        result = icu_format(t["vol_moderate"], number="singular")
        assert "{metric}" in result
        assert "a montré" in result

    def test_select_with_format_spec(self):
        """Placeholders with format specs like {corr:.2f} should be preserved."""
        template = "{number, select, singular {r={corr:.2f}} other {R={corr:.2f}}}"
        assert icu_format(template, number="singular") == "r={corr:.2f}"


# ---------------------------------------------------------------------------
# _unpack_metric
# ---------------------------------------------------------------------------

class TestUnpackMetric:
    def test_plain_string(self):
        name, grammar = _unpack_metric("real expenditure")
        assert name == "real expenditure"
        assert grammar == {"plural": False, "feminine": False}

    def test_dict_with_all_fields(self):
        name, grammar = _unpack_metric(
            {"name": "les dépenses", "plural": True, "feminine": True}
        )
        assert name == "les dépenses"
        assert grammar == {"plural": True, "feminine": True}

    def test_dict_partial_fields_default_false(self):
        name, grammar = _unpack_metric({"name": "les prix", "plural": True})
        assert name == "les prix"
        assert grammar == {"plural": True, "feminine": False}

    def test_dict_name_only(self):
        name, grammar = _unpack_metric({"name": "spending"})
        assert name == "spending"
        assert grammar == {"plural": False, "feminine": False}

    def test_dict_missing_name_raises(self):
        with pytest.raises(TypeError, match="must include a 'name' key"):
            _unpack_metric({"plural": True})

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="must be a str or dict"):
            _unpack_metric(42)

    def test_extra_keys_ignored(self):
        name, grammar = _unpack_metric(
            {"name": "x", "plural": True, "extra": "ignored"}
        )
        assert name == "x"
        assert grammar == {"plural": True, "feminine": False}


# ---------------------------------------------------------------------------
# Dict metric in get_relationship_narrative
# ---------------------------------------------------------------------------

class TestRelationshipNarrativeDictMetrics:
    def test_comovement_with_dict_metrics_french(self):
        years_3pt = np.array([2010, 2015, 2020])
        ref_increasing = np.array([100, 125, 150], dtype=float)
        comp_years_3pt = np.array([2012, 2015, 2018])
        comp_increasing = np.array([50, 65, 80], dtype=float)

        result = get_relationship_narrative(
            reference_years=years_3pt,
            reference_values=ref_increasing,
            comparison_years=comp_years_3pt,
            comparison_values=comp_increasing,
            reference_name={
                "name": "les dépenses de santé",
                "plural": True,
                "feminine": True,
            },
            comparison_name={
                "name": "les indices",
                "plural": True,
                "feminine": False,
            },
            lang="fr",
        )
        assert "les dépenses de santé" in result["narrative"]
        # Plural verbs should appear
        assert "ont augmenté" in result["narrative"]

    def test_insufficient_data_with_dict_metrics(self):
        years = np.array([2010, 2020])
        values = np.array([100, 150])
        result = get_relationship_narrative(
            reference_years=years,
            reference_values=values,
            comparison_years=years,
            comparison_values=values,
            reference_name={"name": "spending"},
            comparison_name={"name": "outcome"},
        )
        assert "spending" in result["narrative"]
        assert "outcome" in result["narrative"]

    def test_string_still_works(self):
        """Backward compat: plain strings still work."""
        years = np.array([2010, 2020])
        values = np.array([100, 150])
        result = get_relationship_narrative(
            reference_years=years,
            reference_values=values,
            comparison_years=years,
            comparison_values=values,
            reference_name="spending",
            comparison_name="outcome",
        )
        assert "spending" in result["narrative"]
