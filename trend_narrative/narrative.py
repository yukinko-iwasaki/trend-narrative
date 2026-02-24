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

import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .extractor import InsightExtractor

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
    segments: Optional[list[dict]] = None,
    cv_value: Optional[float] = None,
    metric: str = "expenditure",
    extractor: Optional["InsightExtractor"] = None,
) -> str:
    """Generate a plain-English narrative from trend data.

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
    metric : str
        Human-readable metric label used in the generated text
        (default ``"expenditure"``).
    extractor : InsightExtractor, optional
        A configured :class:`InsightExtractor` instance (Path 2).
        ``extract_full_suite()`` is called internally.

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

    # --- Validate Path 1 inputs ---
    elif segments is None or cv_value is None:
        raise ValueError(
            "Provide either an InsightExtractor via extractor=, "
            "or precomputed data via segments= and cv_value=."
        )

    return _build_narrative(segments, cv_value, metric)


# ------------------------------------------------------------------
# Internal narrative builder (pure text logic, no data concerns)
# ------------------------------------------------------------------


def _build_narrative(
    segments: list[dict],
    cv_value: float,
    metric: str,
) -> str:
    """Core narrative logic shared by both calling paths."""
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
