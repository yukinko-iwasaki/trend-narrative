"""
trend_narrative.narrative
~~~~~~~~~~~~~~~~~~~~~~~~~
Convert structured trend-segment data into human-readable text narratives.

Ported from dime-worldbank/rpf-country-dash components/narrative_generator.py
"""

from __future__ import annotations

import math
from typing import Optional

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

CV_LOW_THRESHOLD = 5
CV_MODERATE_THRESHOLD = 15

_MILLNAMES = ["", " K", " M", " B", " T"]

_TRANSITION_PREFIXES = [
    "Trend then shifted,",
    "This trajectory pivoted again,",
    "Then,",
]


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------


def millify(n: float) -> str:
    """Format a large number into a human-readable string with suffix.

    Examples
    --------
    >>> millify(1_500_000)
    '1.50 M'
    >>> millify(750)
    '750.00'
    """
    n = float(n)
    idx = max(
        0,
        min(
            len(_MILLNAMES) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
        ),
    )
    return f"{n / 10 ** (3 * idx):.2f}{_MILLNAMES[idx]}"


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
    segments: list[dict],
    cv_value: float,
    metric: str = "expenditure",
) -> str:
    """Generate a plain-English narrative from piecewise trend segments.

    Parameters
    ----------
    segments : list[dict]
        Segment list as returned by :func:`consolidate_segments` or
        :meth:`TrendDetector.extract_trend`.  Each dict must contain
        ``start_year``, ``end_year``, ``start_value``, ``end_value``,
        ``slope``.
    cv_value : float
        Coefficient of Variation (%) from :class:`InsightExtractor`.
        Used when no segments exist to describe overall volatility.
    metric : str
        Human-readable metric label used inside the generated text
        (default ``"expenditure"``).

    Returns
    -------
    str
        A multi-sentence narrative string, or an empty string when
        inputs are invalid.

    Examples
    --------
    >>> segs = [{"start_year": 2010, "end_year": 2020,
    ...          "start_value": 100, "end_value": 200, "slope": 10}]
    >>> text = get_segment_narrative(segs, cv_value=8.0, metric="health spending")
    >>> "increased" in text
    True
    """
    if not segments and cv_value is None:
        return ""

    segments = consolidate_segments(segments)

    # --- No detectable trend: fall back to volatility description ---
    if len(segments) == 0:
        if cv_value < CV_LOW_THRESHOLD:
            return f"{metric} remained highly stable and range-bound."
        elif cv_value <= CV_MODERATE_THRESHOLD:
            return f"{metric} showed moderate fluctuations around a consistent mean."
        else:
            return f"{metric} exhibited significant volatility without a clear direction."

    # --- Single monotone trend ---
    if len(segments) == 1:
        seg = segments[0]
        total_change = seg["end_value"] - seg["start_value"]
        pct_change = (total_change / seg["start_value"]) * 100 if seg["start_value"] != 0 else 0.0
        direction = "increased" if total_change > 0 else "decreased"
        return (
            f"between {int(seg['start_year'])} and {int(seg['end_year'])}, "
            f"the {metric} {direction} by {millify(abs(total_change))} "
            f"({pct_change:+.2f}%), maintaining a consistent trajectory."
        )

    # --- Multi-segment narrative ---
    narrative: list[str] = []
    for i, seg in enumerate(segments):
        direction = "an upward" if seg["slope"] > 0 else "a downward"

        if i == 0:
            narrative.append(
                f"From {int(seg['start_year'])} to {int(seg['end_year'])}, "
                f"the {metric} showed {direction} trend."
            )
        else:
            prev_slope = segments[i - 1]["slope"]
            prefix = _TRANSITION_PREFIXES[min(i - 1, len(_TRANSITION_PREFIXES) - 1)]

            if prev_slope > 0 and seg["slope"] < 0:
                transition = (
                    f"reaching a peak in {int(seg['start_year'])} "
                    f"before reversing into a decline."
                )
            elif prev_slope < 0 and seg["slope"] > 0:
                transition = (
                    f"hitting a low in {int(seg['start_year'])} "
                    f"followed by a recovery."
                )
            else:
                transition = (
                    f"continuing its {direction} path through {int(seg['end_year'])}."
                )

            narrative.append(f"{prefix} {transition}")

    return " ".join(narrative)


# ------------------------------------------------------------------
# Convenience: run the full pipeline in one call
# ------------------------------------------------------------------


def generate_narrative(
    x,
    y,
    metric: str = "expenditure",
    detector_kwargs: Optional[dict] = None,
) -> dict:
    """End-to-end helper: detect trends, extract insights, generate text.

    Parameters
    ----------
    x : array-like
        1-D array of x-values (e.g. years).
    y : array-like
        1-D array of metric values.
    metric : str
        Human-readable label for the metric.
    detector_kwargs : dict, optional
        Keyword arguments forwarded to :class:`~trend_narrative.TrendDetector`.

    Returns
    -------
    dict
        ``{"narrative": str, "cv_value": float, "segments": list[dict]}``

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(2010, 2022)
    >>> y = np.array([100,105,110,108,115,130,125,120,118,122,130,140], dtype=float)
    >>> result = generate_narrative(x, y, metric="health spending")
    >>> isinstance(result["narrative"], str)
    True
    """
    from .extractor import InsightExtractor
    from .detector import TrendDetector

    detector = TrendDetector(**(detector_kwargs or {}))
    extractor = InsightExtractor(x, y, detector=detector)
    suite = extractor.extract_full_suite()

    narrative = get_segment_narrative(
        segments=suite["segments"],
        cv_value=suite["cv_value"],
        metric=metric,
    )

    return {
        "narrative": narrative,
        "cv_value": suite["cv_value"],
        "segments": suite["segments"],
    }
