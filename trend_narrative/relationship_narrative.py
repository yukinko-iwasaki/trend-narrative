"""
trend_narrative.relationship_narrative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate narratives from relationship analysis results.

This module handles the text generation layer, converting structured
analysis outputs into descriptive text.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

from .relationship_analysis import (
    analyze_relationship,
    get_correlation_strength,
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_MAX_LAG_CAP,
    P_THRESHOLD,
)
from .translations import (
    MetricLike,
    _genitive,
    _resolve_time_unit,
    _time_unit_comparison,
    _unpack_metric,
    get_translations,
    icu_format,
)

Formatter = Union[str, Callable[[float], str]]


def _format_value(value: float, fmt: Formatter, lang: str = "en") -> str:
    """Format a numeric value, applying the language's decimal separator.

    String format specs are localized by swapping ``.`` for the catalog's
    ``decimal_sep``. Callable formats pass through unchanged so callers
    retain full control (e.g. custom currency rendering).

    Note: only handles the decimal separator. Format specs with thousand
    separators (e.g. ``",.2f"``) may produce ambiguous output in languages
    where ``,`` is the decimal separator; pass a callable for those cases.
    """
    if callable(fmt):
        return fmt(value)
    formatted = f"{value:{fmt}}"
    decimal_sep = get_translations(lang)["number_format"]["decimal_sep"]
    if decimal_sep != ".":
        formatted = formatted.replace(".", decimal_sep)
    return formatted


_DEFAULT_ICU = {"number": "singular", "gender": "masculine"}


def _build_comovement_narrative(
    segment_details: list[dict],
    reference_name: str,
    comparison_name: str,
    reference_format: Formatter = ".2f",
    comparison_format: Formatter = ".2f",
    lang: str = "en",
    reference_icu: Optional[dict[str, str]] = None,
    comparison_icu: Optional[dict[str, str]] = None,
) -> str:
    """Build narrative from segment-level co-movement analysis."""
    t = get_translations(lang)
    reference_icu = reference_icu or _DEFAULT_ICU
    comparison_icu = comparison_icu or _DEFAULT_ICU

    if not segment_details:
        return t["unable_to_analyze"].format(x=reference_name, y=comparison_name)

    total_comparison_points = sum(seg["comparison_n_points"] for seg in segment_details)
    if total_comparison_points == 0:
        return t["no_data_available"].format(
            x=reference_name, y=comparison_name,
            x_gen=_genitive(lang, reference_name),
            y_gen=_genitive(lang, comparison_name),
        )

    # Only build narratives for segments with comparison data
    narratives = []

    for seg in segment_details:
        comp_n = seg["comparison_n_points"]
        if comp_n == 0:
            continue

        period = t["period_from_to"].format(
            start=seg["start_year"], end=seg["end_year"]
        )
        # Language-neutral keys from the analysis layer
        ref_dir_key = seg["reference_direction"]
        comp_dir_key = seg["comparison_direction"]

        ref_start = _format_value(seg["reference_start"], reference_format, lang=lang)
        ref_end = _format_value(seg["reference_end"], reference_format, lang=lang)

        # Override direction if formatted values are the same
        if ref_start == ref_end:
            ref_dir_key = "remained_stable"

        ref_dir = icu_format(t[ref_dir_key], **reference_icu)

        if comp_dir_key is None:
            if comp_n == 1:
                comp_start = _format_value(seg["comparison_start"], comparison_format, lang=lang)
                seg_narrative = t["single_observation"].format(
                    period=period, ref_name=reference_name, ref_dir=ref_dir,
                    ref_start=ref_start, ref_end=ref_end,
                    comp_name=comparison_name, comp_start=comp_start,
                    comp_name_gen=_genitive(lang, comparison_name),
                )
            else:
                comp_start = _format_value(seg["comparison_start"], comparison_format, lang=lang)
                stable_tpl = icu_format(t["stable_comparison"], **comparison_icu)
                seg_narrative = stable_tpl.format(
                    period=period, ref_name=reference_name, ref_dir=ref_dir,
                    ref_start=ref_start, ref_end=ref_end,
                    comp_name=comparison_name, comp_start=comp_start,
                )
        else:
            comp_start = _format_value(seg["comparison_start"], comparison_format, lang=lang)
            comp_end = _format_value(seg["comparison_end"], comparison_format, lang=lang)

            # Override direction if formatted values are the same
            if comp_start == comp_end:
                comp_dir_key = "remained_stable"

            # Describe co-movement (compare against keys, not translated strings)
            if ref_dir_key == comp_dir_key:
                relationship = t["both_same_direction"]
            elif ref_dir_key == "remained_stable" or comp_dir_key == "remained_stable":
                relationship = None
            else:
                relationship = t["opposite_directions"]

            comp_dir = icu_format(t[comp_dir_key], **comparison_icu)

            if relationship:
                seg_narrative = t["comovement_with_rel"].format(
                    period=period, ref_name=reference_name, ref_dir=ref_dir,
                    ref_start=ref_start, ref_end=ref_end,
                    comp_name=comparison_name, comp_dir=comp_dir,
                    comp_start=comp_start, comp_end=comp_end,
                    relationship=relationship,
                )
            else:
                seg_narrative = t["comovement_no_rel"].format(
                    period=period, ref_name=reference_name, ref_dir=ref_dir,
                    ref_start=ref_start, ref_end=ref_end,
                    comp_name=comparison_name, comp_dir=comp_dir,
                    comp_start=comp_start, comp_end=comp_end,
                )

        # Capitalize first letter
        seg_narrative = seg_narrative[0].upper() + seg_narrative[1:]
        narratives.append(seg_narrative)

    # Join segments
    if len(narratives) == 1:
        narrative = narratives[0] + "."
    else:
        narrative = ". ".join(narratives) + "."

    # Add caveat about limited data (separator owned by composition, not the template)
    narrative += " " + t["limited_data_caveat"].format(comp_name=comparison_name)

    return narrative


def _build_lagged_correlation_narrative(
    best_lag: dict,
    all_lags: list[dict],
    n_sparse: int,
    max_lag_tested: int,
    reference_name: str,
    comparison_name: str,
    time_unit: str = "year",
    reference_leads: bool = True,
    lang: str = "en",
    leader_icu: Optional[dict[str, str]] = None,
    follower_icu: Optional[dict[str, str]] = None,
) -> str:
    """Build narrative from lagged correlation analysis.

    When reference_leads=True: "When reference increases, comparison follows"
    When reference_leads=False: "When comparison increases, reference follows"
    """
    t = get_translations(lang)
    leader_icu = leader_icu or _DEFAULT_ICU
    follower_icu = follower_icu or _DEFAULT_ICU

    correlation = best_lag["correlation"]
    p_value = best_lag["p_value"]
    lag = best_lag["lag"]
    n_pairs = best_lag["n_pairs"]

    strength_key = get_correlation_strength(correlation)
    strength = t[strength_key]
    is_significant = p_value < P_THRESHOLD

    # Resolve localized forms of time_unit. These are the ONLY strings that
    # should flow into time_unit template slots — never the raw English arg.
    time_unit_sg = _resolve_time_unit(t, time_unit, 1)

    # Determine which series leads based on computation
    if reference_leads:
        leader_name, follower_name = reference_name, comparison_name
    else:
        leader_name, follower_name = comparison_name, reference_name

    # Build lag timing description
    if lag == 0:
        # Resolve grammatical gender of the time unit so French can switch
        # between "la même {unit}" (feminine) and "du même {unit}"
        # (masculine). English has no gender select so the key is ignored.
        unit_gender = t.get("time_unit_genders", {}).get(time_unit, "other")
        timing_tpl = icu_format(t["timing_same"], gender=unit_gender)
        timing = timing_tpl.format(time_unit=time_unit_sg)
    else:
        timing = t["timing_lagged"].format(
            lag=lag, time_unit_form=_resolve_time_unit(t, time_unit, lag)
        )

    tu_comparison = _time_unit_comparison(lang, time_unit_sg)

    # Not significant: lead with uncertainty (compare against the key, not the translated value)
    if strength_key == "strength_no" or not is_significant:
        narrative = t["no_reliable_relationship"].format(
            x=reference_name, y=comparison_name,
            x_gen=_genitive(lang, reference_name),
            y_gen=_genitive(lang, comparison_name),
        )
        if strength_key != "strength_no":
            sign = t["positive"] if correlation > 0 else t["negative"]
            narrative += t["weak_pattern"].format(
                strength=strength, sign=sign,
                corr=correlation, n_pairs=n_pairs, p_val=p_value,
            )
        elif max_lag_tested == 0:
            narrative += t["no_association"].format(
                n_pairs=n_pairs, time_unit_comparison=tu_comparison,
            )
        else:
            narrative += t["no_association_with_lag"].format(
                max_lag=max_lag_tested,
                time_unit_form=_resolve_time_unit(t, time_unit, max_lag_tested),
                n_pairs=n_pairs,
                time_unit_comparison=tu_comparison,
            )
    else:
        # Significant: lead with the finding.
        # The template has two verbs — "Lorsque {leader} augmente/augmentent"
        # (agreeing with the leader) and "{follower} tend/tendent"
        # (agreeing with the follower). Each uses its own number select.
        direction_word = t["increase"] if correlation > 0 else t["decrease"]
        sig_tpl = icu_format(
            t["significant_finding"],
            leader_number=leader_icu["number"],
            follower_number=follower_icu["number"],
        )
        narrative = sig_tpl.format(
            leader=leader_name, follower=follower_name,
            direction_word=direction_word, timing=timing,
            strength=strength, corr=correlation, p_val=p_value,
            n_pairs=n_pairs, time_unit_comparison=tu_comparison,
        )

    return narrative


def get_relationship_narrative(
    reference_years: "array-like" = None,
    reference_values: "array-like" = None,
    comparison_years: "array-like" = None,
    comparison_values: "array-like" = None,
    reference_name: MetricLike = "",
    comparison_name: MetricLike = "",
    reference_segments: Optional[list[dict]] = None,
    correlation_threshold: int = DEFAULT_CORRELATION_THRESHOLD,
    max_lag_cap: int = DEFAULT_MAX_LAG_CAP,
    reference_format: Formatter = ".2f",
    comparison_format: Formatter = ".2f",
    time_unit: str = "year",
    reference_leads: Optional[bool] = None,
    insights: Optional[dict] = None,
    lang: str = "en",
) -> dict:
    """
    Analyze relationship between two time series and generate narrative.

    Supports two calling paths:

    **Path 1 – from raw data** (analysis computed on the fly):

    .. code-block:: python

        get_relationship_narrative(
            reference_years=years1,
            reference_values=values1,
            comparison_years=years2,
            comparison_values=values2,
            reference_name="spending",
            comparison_name="outcome",
        )

    **Path 2 – precomputed insights** (e.g. insights already stored in a
    Delta table — no re-analysis required):

    .. code-block:: python

        get_relationship_narrative(
            insights=row["relationship_insights"],
            reference_name="spending",
            comparison_name="outcome",
        )

    Parameters
    ----------
    reference_years : array-like, optional
        Year values for reference series (Path 1).
    reference_values : array-like, optional
        Data values for reference series (Path 1).
    comparison_years : array-like, optional
        Year values for the comparison series (Path 1).
    comparison_values : array-like, optional
        Data values for the comparison series (Path 1).
    reference_name : str or dict
        Display name for the reference series. May be either a plain
        string or a dict bundling the name with grammatical properties
        for French inflection::

            {"name": "les dépenses", "plural": True, "feminine": True}

        The ``plural`` / ``feminine`` keys default to ``False`` and are
        ignored for English.
    comparison_name : str or dict
        Display name for the comparison series. Same accepted shapes as
        *reference_name*.
    reference_segments : list[dict], optional
        Pre-computed segments from InsightExtractor for the reference series.
        Each dict should contain: start_year, end_year, start_value, end_value.
        If not provided, computed from reference_years/reference_values.
    correlation_threshold : int
        Minimum points to use correlation analysis (default 5).
        Below this, comovement analysis is used.
    max_lag_cap : int
        Maximum lag to test in years (default 5). Actual max lag may be
        lower if data is insufficient.
    reference_format : str or callable
        Format spec (e.g., ".2f") or callable (e.g., lambda x: f"${x:,.0f}")
        for reference series values in narratives. Default ".2f".
    comparison_format : str or callable
        Format spec or callable for comparison series values. Default ".2f".
    time_unit : str
        Time unit label for narratives (default "year"). Use "month", "quarter", etc.
    reference_leads : bool, optional
        Controls narrative direction for lagged correlation:
        - True: "When reference increases, comparison follows"
        - False: "When comparison increases, reference follows"
        - None (default): inferred from sparsity (sparser series is the follower)
    insights : dict, optional
        Pre-computed insights from analyze_relationship() (Path 2).
        If provided, raw data arrays are ignored. Insights are language-neutral
        and can be rendered in any supported language.
    lang : str
        Language code for the generated narrative (default ``"en"``).
        Supported values are listed in
        :data:`trend_narrative.SUPPORTED_LANGUAGES` (currently ``"en"``, ``"fr"``).
        Only the rendered narrative text depends on this; the returned
        analysis fields (``method``, ``segment_details``, ``best_lag``, …)
        are language-neutral.

    Returns
    -------
    dict
        Keys:
        - narrative: str, human-readable description
        - method: str, one of "insufficient_data", "comovement", "lagged_correlation"
        - n_points: int, number of data points in sparser series
        - segment_details: list[dict], per-segment analysis (comovement only)
        - best_lag: dict with lag, correlation, p_value, n_pairs (correlation only)
        - all_lags: list[dict], results for all tested lags (correlation only)
        - max_lag_tested: int, maximum lag that was tested (correlation only)

    Raises
    ------
    ValueError
        If neither insights nor data arrays are provided.
    """
    t = get_translations(lang)

    # Unpack metric names — each may be a plain string or a dict bundling
    # the display name with grammatical properties (plural/feminine).
    # _unpack_metric returns (name, icu_kwargs) ready for icu_format.
    reference_name, reference_icu = _unpack_metric(reference_name)
    comparison_name, comparison_icu = _unpack_metric(comparison_name)

    if insights is not None:
        analysis = insights
    elif reference_years is not None and comparison_years is not None:
        analysis = analyze_relationship(
            reference_years=reference_years,
            reference_values=reference_values,
            comparison_years=comparison_years,
            comparison_values=comparison_values,
            reference_segments=reference_segments,
            correlation_threshold=correlation_threshold,
            max_lag_cap=max_lag_cap,
        )
    else:
        raise ValueError(
            "Provide either insights= or data arrays "
            "(reference_years, reference_values, comparison_years, comparison_values)"
        )

    if reference_leads is None:
        reference_leads = analysis.get("reference_leads", True)

    method = analysis["method"]
    n_points = analysis["n_points"]

    if method == "insufficient_data":
        narrative = t["insufficient_data"].format(
            x=reference_name, y=comparison_name
        )
    elif method == "lagged_correlation":
        # The significant_finding template has two subject-verb pairs —
        # "{leader} augmente/augmentent" and "{follower} tend/tendent" —
        # each needing its own grammar for correct French agreement.
        if reference_leads:
            leader_icu, follower_icu = reference_icu, comparison_icu
        else:
            leader_icu, follower_icu = comparison_icu, reference_icu
        narrative = _build_lagged_correlation_narrative(
            analysis["best_lag"],
            analysis["all_lags"],
            n_points,
            analysis["max_lag_tested"],
            reference_name,
            comparison_name,
            time_unit,
            reference_leads=reference_leads,
            lang=lang,
            leader_icu=leader_icu,
            follower_icu=follower_icu,
        )
    else:
        narrative = _build_comovement_narrative(
            analysis["segment_details"],
            reference_name,
            comparison_name,
            reference_format=reference_format,
            comparison_format=comparison_format,
            lang=lang,
            reference_icu=reference_icu,
            comparison_icu=comparison_icu,
        )

    return {
        "narrative": narrative,
        "method": method,
        "n_points": n_points,
        "segment_details": analysis["segment_details"],
        "best_lag": analysis["best_lag"],
        "all_lags": analysis["all_lags"],
        "max_lag_tested": analysis["max_lag_tested"],
    }
