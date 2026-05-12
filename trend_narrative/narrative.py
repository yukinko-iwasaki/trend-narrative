"""
trend_narrative.narrative
~~~~~~~~~~~~~~~~~~~~~~~~~
Convert structured trend-segment data into human-readable text narratives.

Two calling paths are supported:

  Path 1 – precomputed (Databricks / Delta table):
      get_segment_narrative(
          segments=row["segments"],
          cv_value=row["cv_value"],
          metric="health spending",
      )

  Path 2 – standalone (user controls extraction logic via InsightExtractor):
      extractor = InsightExtractor(x, y, detector=MyCustomDetector())
      get_segment_narrative(extractor=extractor, metric="health spending")

Ported from dime-worldbank/rpf-country-dash components/narrative_generator.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .translations import (
    MetricLike,
    _format_percent,
    _unpack_metric,
    get_translations,
    icu_format,
    millify,
)

if TYPE_CHECKING:
    from .extractor import InsightExtractor

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

CV_LOW_THRESHOLD = 5
CV_MODERATE_THRESHOLD = 15


# ------------------------------------------------------------------
# Segment consolidation
# ------------------------------------------------------------------


def consolidate_segments(segments: list[dict]) -> list[dict]:
    """Merge consecutive segments that share the same slope direction.

    Two segments are merged when both slopes are non-negative or both
    are negative.  The merged segment spans from the first ``start_year``
    to the last ``end_year``, and its slope is recomputed as the
    rise-over-run of the combined span.

    Parameters
    ----------
    segments : list[dict]
        Raw segment list as returned by :meth:`TrendDetector.extract_trend`.

    Returns
    -------
    list[dict]
        Consolidated segment list (may be shorter than *segments*).
    """
    if not segments:
        return []

    # Deep-copy so we don't mutate the caller's data
    consolidated = [dict(segments[0])]

    for seg in segments[1:]:
        last = consolidated[-1]
        same_direction = (last["slope"] >= 0 and seg["slope"] >= 0) or (
            last["slope"] < 0 and seg["slope"] < 0
        )
        if same_direction:
            last["end_year"] = seg["end_year"]
            last["end_value"] = seg["end_value"]
            duration = last["end_year"] - last["start_year"]
            last["slope"] = (
                (last["end_value"] - last["start_value"]) / duration
                if duration != 0
                else 0.0
            )
        else:
            consolidated.append(dict(seg))

    return consolidated


# ------------------------------------------------------------------
# Narrative generation
# ------------------------------------------------------------------


def get_segment_narrative(
    segments: Optional[list[dict]] = None,
    cv_value: Optional[float] = None,
    metric: MetricLike = "expenditure",
    extractor: Optional["InsightExtractor"] = None,
    n_points: Optional[int] = None,
    lang: str = "en",
) -> str:
    """Generate a plain-language narrative from trend data.

    Supports two calling paths:

    **Path 1 – precomputed** (e.g. segments and cv already stored in a
    Delta table — no re-fitting required):

    .. code-block:: python

        get_segment_narrative(
            segments=row["segments"],
            cv_value=row["cv_value"],
            metric="health spending",
        )

    **Path 2 – standalone** (user creates an :class:`InsightExtractor`
    with the desired detection logic, then passes it in):

    .. code-block:: python

        from trend_narrative import InsightExtractor, TrendDetector

        extractor = InsightExtractor(x, y, detector=TrendDetector(max_segments=2))
        get_segment_narrative(extractor=extractor, metric="health spending")

    Keeping the extractor construction separate means you can swap in any
    custom detector without touching the narrative layer.

    Parameters
    ----------
    segments : list[dict], optional
        Precomputed segment list (Path 1). Each dict must contain
        ``start_year``, ``end_year``, ``start_value``, ``end_value``,
        ``slope``.
    cv_value : float, optional
        Precomputed Coefficient of Variation % (Path 1).
    metric : str or dict
        Human-readable metric label used in the generated text
        (default ``"expenditure"``). May be either:

        * a plain string (e.g. ``"real expenditure"``), or
        * a dict bundling the display name with grammatical properties
          used for French inflection (ignored for English)::

              {"name": "les dépenses", "plural": True, "feminine": True}

        The ``plural`` / ``feminine`` keys default to ``False``.
    extractor : InsightExtractor, optional
        A configured :class:`InsightExtractor` instance (Path 2).
        ``extract_full_suite()`` is called internally.
    n_points : int, optional
        Number of data points (Path 1). If fewer than 2, returns empty
        string. Automatically inferred when using extractor path.
    lang : str
        Language code for the generated narrative (default ``"en"``).
        Supported: ``"en"``, ``"fr"``.

    Returns
    -------
    str
        A multi-sentence narrative string.

    Raises
    ------
    ValueError
        If neither an extractor nor (segments + cv_value) are provided.
    """
    # --- Path 2: extractor object supplied → run extraction now ---
    if extractor is not None:
        suite = extractor.extract_full_suite()
        segments = suite["segments"]
        cv_value = suite["cv_value"]
        n_points = suite["n_points"]

    # --- Validate Path 1 inputs ---
    elif segments is None or cv_value is None:
        raise ValueError(
            "Provide either an InsightExtractor via extractor=, "
            "or precomputed data via segments= and cv_value=."
        )

    # --- Insufficient data: return empty narrative ---
    if n_points is not None and n_points < 2:
        return ""

    return _build_narrative(segments, cv_value, metric, lang=lang)


# ------------------------------------------------------------------
# Internal narrative builder (pure text logic, no data concerns)
# ------------------------------------------------------------------


def _build_narrative(
    segments: list[dict],
    cv_value: float,
    metric: MetricLike,
    lang: str = "en",
) -> str:
    """Core narrative logic shared by both calling paths."""
    t = get_translations(lang)
    segments = consolidate_segments(segments)
    metric, icu_kw = _unpack_metric(metric)

    # --- No detectable trend: fall back to volatility description ---
    if len(segments) == 0:
        if cv_value < CV_LOW_THRESHOLD:
            return icu_format(t["vol_low"], **icu_kw).format(metric=metric)
        elif cv_value <= CV_MODERATE_THRESHOLD:
            return icu_format(t["vol_moderate"], **icu_kw).format(metric=metric)
        else:
            return icu_format(t["vol_high"], **icu_kw).format(metric=metric)

    # --- Single monotone trend ---
    if len(segments) == 1:
        seg = segments[0]
        total_change = seg["end_value"] - seg["start_value"]
        pct_change = (total_change / seg["start_value"]) * 100 if seg["start_value"] != 0 else 0.0
        dir_key = "increased" if total_change > 0 else "decreased"
        direction = icu_format(t[dir_key], **icu_kw)
        return t["single_segment"].format(
            start_year=int(seg["start_year"]),
            end_year=int(seg["end_year"]),
            metric=metric,
            direction=direction,
            change=millify(abs(total_change), lang=lang),
            pct_change=_format_percent(pct_change, lang=lang),
        )

    # --- Multi-segment narrative ---
    narrative: list[str] = []
    transition_prefixes = t["transition_prefixes"]
    for i, seg in enumerate(segments):
        is_upward = seg["slope"] > 0
        trend_phrase = t["trend_upward"] if is_upward else t["trend_downward"]
        path_phrase = t["path_upward"] if is_upward else t["path_downward"]

        if i == 0:
            first_seg_tpl = icu_format(t["first_segment"], **icu_kw)
            narrative.append(
                first_seg_tpl.format(
                    start_year=int(seg["start_year"]),
                    end_year=int(seg["end_year"]),
                    metric=metric,
                    trend_phrase=trend_phrase,
                )
            )
        else:
            prev_slope = segments[i - 1]["slope"]
            prefix = transition_prefixes[min(i - 1, len(transition_prefixes) - 1)]

            if prev_slope > 0 and seg["slope"] < 0:
                transition = t["peak_reversal"].format(year=int(seg["start_year"]))
            elif prev_slope < 0 and seg["slope"] > 0:
                transition = t["low_recovery"].format(year=int(seg["start_year"]))
            else:
                transition = t["continuing"].format(
                    path_phrase=path_phrase, year=int(seg["end_year"])
                )

            narrative.append(f"{prefix} {transition}")

    return " ".join(narrative)
