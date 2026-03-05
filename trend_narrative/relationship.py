"""
trend_narrative.relationship
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Analyze relationships between two time series, using pre-computed segments
from a reference series to align analysis of a comparison series.

Supports two analysis modes based on data availability:
- Co-movement narrative: When data points < threshold, describes directional
  alignment within each segment of the reference series.
- Correlation analysis: When sufficient data points exist, computes
  year-on-year change correlation to account for shared trends.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats


CORRELATION_THRESHOLDS = {
    0.1: "no",
    0.3: "weak",
    0.5: "moderate",
    0.7: "strong",
    1.0: "very strong",
}

DEFAULT_THRESHOLD_HIGH = 8
DEFAULT_THRESHOLD_LOW = 3


def get_direction(values: np.ndarray) -> str:
    """Determine direction from start to end value."""
    if len(values) < 2:
        return "unknown"

    start, end = values[0], values[-1]
    if start == 0:
        return "increased" if end > 0 else "stable"

    pct_change = (end - start) / abs(start)

    if abs(pct_change) < 0.05:
        return "remained stable"
    return "increased" if pct_change > 0 else "decreased"


def get_direction_from_slope(slope: float) -> str:
    """Convert slope to direction string."""
    if slope > 0:
        return "increased"
    elif slope < 0:
        return "decreased"
    return "remained stable"


def get_correlation_strength(corr: float) -> str:
    """Map correlation coefficient to descriptive strength."""
    abs_corr = abs(corr)
    for threshold, strength in sorted(CORRELATION_THRESHOLDS.items()):
        if abs_corr <= threshold:
            return strength
    return "very strong"


def compute_yoy_changes(
    years: np.ndarray,
    values: np.ndarray
) -> np.ndarray:
    """
    Compute year-on-year changes, handling gaps by annualizing.

    Returns array of annualized changes (change per year).
    """
    if len(years) < 2:
        return np.array([])

    sorted_idx = np.argsort(years)
    years_sorted = years[sorted_idx]
    values_sorted = values[sorted_idx]

    year_gaps = np.diff(years_sorted)
    value_diffs = np.diff(values_sorted)

    return value_diffs / year_gaps


def analyze_segment_comovement(
    segment: dict,
    comparison_years: np.ndarray,
    comparison_values: np.ndarray,
) -> dict:
    """Analyze comparison series within a single reference segment."""
    start_year = segment["start_year"]
    end_year = segment["end_year"]

    mask = (comparison_years >= start_year) & (comparison_years <= end_year)
    seg_years = comparison_years[mask]
    seg_values = comparison_values[mask]

    n_points = len(seg_values)

    result = {
        "start_year": int(start_year),
        "end_year": int(end_year),
        "reference_direction": get_direction_from_slope(segment["slope"]),
        "comparison_n_points": n_points,
    }

    if n_points == 0:
        result["comparison_direction"] = None
        result["comparison_start"] = None
        result["comparison_end"] = None
    elif n_points == 1:
        result["comparison_direction"] = None
        result["comparison_start"] = float(seg_values[0])
        result["comparison_end"] = float(seg_values[0])
    else:
        sorted_idx = np.argsort(seg_years)
        seg_values_sorted = seg_values[sorted_idx]
        result["comparison_direction"] = get_direction(seg_values_sorted)
        result["comparison_start"] = float(seg_values_sorted[0])
        result["comparison_end"] = float(seg_values_sorted[-1])

    return result


def _build_comovement_narrative(
    segment_details: list[dict],
    reference_name: str,
    comparison_name: str,
) -> str:
    """Build narrative from segment-level co-movement analysis."""
    if not segment_details:
        return f"Unable to analyze relationship between {reference_name} and {comparison_name}."

    narratives = []

    for i, seg in enumerate(segment_details):
        period = f"from {seg['start_year']} to {seg['end_year']}"
        ref_dir = seg["reference_direction"]
        comp_dir = seg["comparison_direction"]
        comp_n = seg["comparison_n_points"]

        if comp_dir is None:
            if comp_n == 0:
                seg_narrative = (
                    f"{period}, {reference_name} {ref_dir}, "
                    f"but {comparison_name} data is unavailable for this period"
                )
            else:
                # n == 1
                seg_narrative = (
                    f"{period}, {reference_name} {ref_dir}, "
                    f"with only one {comparison_name} observation ({seg['comparison_start']:.2f})"
                )
        else:
            # Describe co-movement
            if ref_dir == comp_dir:
                relationship = "both moving in the same direction"
            elif ref_dir == "remained stable" or comp_dir == "remained stable":
                relationship = None
            else:
                relationship = "moving in opposite directions"

            if relationship:
                seg_narrative = (
                    f"{period}, {reference_name} {ref_dir} while {comparison_name} {comp_dir} "
                    f"({seg['comparison_start']:.2f} to {seg['comparison_end']:.2f}), {relationship}"
                )
            else:
                seg_narrative = (
                    f"{period}, {reference_name} {ref_dir} while {comparison_name} {comp_dir} "
                    f"({seg['comparison_start']:.2f} to {seg['comparison_end']:.2f})"
                )

        # Capitalize first letter for first segment
        if i == 0:
            seg_narrative = seg_narrative[0].upper() + seg_narrative[1:]

        narratives.append(seg_narrative)

    # Join segments
    if len(narratives) == 1:
        narrative = narratives[0] + "."
    else:
        narrative = ". ".join(narratives) + "."

    # Add caveat about limited data
    total_comparison_points = sum(seg["comparison_n_points"] for seg in segment_details)
    if total_comparison_points < DEFAULT_THRESHOLD_HIGH:
        narrative += (
            f" With only {total_comparison_points} {comparison_name} observations, "
            "a statistical relationship cannot be established."
        )

    return narrative


def _build_correlation_narrative(
    correlation: float,
    p_value: float,
    n_points: int,
    reference_name: str,
    comparison_name: str,
) -> str:
    """Build narrative from year-on-year change correlation analysis."""
    strength = get_correlation_strength(correlation)
    direction = "positive" if correlation > 0 else "negative"

    if strength == "no":
        narrative = (
            f"Year-on-year changes in {reference_name} and {comparison_name} "
            f"show no significant correlation (r={correlation:.2f}, n={n_points}), "
            "suggesting changes in one are not associated with changes in the other."
        )
    else:
        if p_value < 0.05:
            significance = "statistically significant"
        elif p_value < 0.10:
            significance = "marginally significant"
        else:
            significance = "not statistically significant"

        narrative = (
            f"Year-on-year changes in {reference_name} and {comparison_name} "
            f"show a {strength} {direction} correlation (r={correlation:.2f}, p={p_value:.3f}, n={n_points}), "
            f"which is {significance}. "
        )

        if correlation > 0:
            narrative += f"This suggests that increases in {reference_name} tend to coincide with increases in {comparison_name}."
        else:
            narrative += f"This suggests that increases in {reference_name} tend to coincide with decreases in {comparison_name}."

    return narrative


def get_relationship_narrative(
    reference_segments: list[dict],
    comparison_years: "array-like",
    comparison_values: "array-like",
    reference_name: str,
    comparison_name: str,
    reference_years: Optional["array-like"] = None,
    reference_values: Optional["array-like"] = None,
    threshold_high: int = DEFAULT_THRESHOLD_HIGH,
    threshold_low: int = DEFAULT_THRESHOLD_LOW,
) -> dict:
    """
    Analyze relationship between two time series.

    Uses pre-computed segments from the reference series to align
    analysis of the comparison series. Chooses analysis method based
    on data availability.

    Parameters
    ----------
    reference_segments : list[dict]
        Pre-computed segments from InsightExtractor for the reference series.
        Each dict should contain: start_year, end_year, slope.
    comparison_years : array-like
        Year values for the comparison series.
    comparison_values : array-like
        Data values for the comparison series.
    reference_name : str
        Display name for the reference series.
    comparison_name : str
        Display name for the comparison series.
    reference_years : array-like, optional
        Year values for reference series. Required for correlation analysis.
    reference_values : array-like, optional
        Data values for reference series. Required for correlation analysis.
    threshold_high : int
        Minimum overlapping data points to use correlation analysis (default 8).
    threshold_low : int
        Minimum data points to produce any analysis (default 3).

    Returns
    -------
    dict
        Keys:
        - narrative: str, human-readable description
        - method: str, one of "insufficient_data", "comovement", "correlation"
        - n_points: int, number of overlapping data points
        - segment_details: list[dict], per-segment analysis (comovement only)
        - correlation: float, correlation coefficient (correlation only)
        - p_value: float, p-value for correlation (correlation only)
    """
    comparison_years = np.asarray(comparison_years, dtype=float)
    comparison_values = np.asarray(comparison_values, dtype=float)

    # Remove NaN values
    valid_mask = ~np.isnan(comparison_values)
    comparison_years = comparison_years[valid_mask]
    comparison_values = comparison_values[valid_mask]

    n_comparison = len(comparison_values)

    # Determine overlapping points if reference data provided
    if reference_years is not None and reference_values is not None:
        reference_years = np.asarray(reference_years, dtype=float)
        reference_values = np.asarray(reference_values, dtype=float)

        valid_mask = ~np.isnan(reference_values)
        reference_years = reference_years[valid_mask]
        reference_values = reference_values[valid_mask]

        # Find overlapping years
        common_years = np.intersect1d(reference_years, comparison_years)
        n_overlap = len(common_years)
    else:
        n_overlap = n_comparison

    # Insufficient data
    if n_comparison < threshold_low or not reference_segments:
        return {
            "narrative": (
                f"The relationship between {reference_name} and {comparison_name} "
                "cannot be determined due to limited data availability."
            ),
            "method": "insufficient_data",
            "n_points": n_comparison,
            "segment_details": None,
            "correlation": None,
            "p_value": None,
        }

    # Correlation analysis path (sufficient overlapping data)
    if (n_overlap >= threshold_high and
        reference_years is not None and
        reference_values is not None):

        # Get values for common years
        ref_mask = np.isin(reference_years, common_years)
        comp_mask = np.isin(comparison_years, common_years)

        ref_common_years = reference_years[ref_mask]
        ref_common_values = reference_values[ref_mask]
        comp_common_years = comparison_years[comp_mask]
        comp_common_values = comparison_values[comp_mask]

        # Sort both by year
        ref_sort_idx = np.argsort(ref_common_years)
        comp_sort_idx = np.argsort(comp_common_years)

        ref_common_values = ref_common_values[ref_sort_idx]
        comp_common_values = comp_common_values[comp_sort_idx]
        common_years_sorted = ref_common_years[ref_sort_idx]

        # Compute year-on-year changes
        ref_changes = compute_yoy_changes(common_years_sorted, ref_common_values)
        comp_changes = compute_yoy_changes(common_years_sorted, comp_common_values)

        if len(ref_changes) >= 2 and len(comp_changes) >= 2:
            # Pearson correlation on changes
            correlation, p_value = stats.pearsonr(ref_changes, comp_changes)

            narrative = _build_correlation_narrative(
                correlation, p_value, len(ref_changes),
                reference_name, comparison_name
            )

            return {
                "narrative": narrative,
                "method": "correlation",
                "n_points": n_overlap,
                "segment_details": None,
                "correlation": float(correlation),
                "p_value": float(p_value),
            }

    # Co-movement analysis path (limited data)
    segment_details = [
        analyze_segment_comovement(seg, comparison_years, comparison_values)
        for seg in reference_segments
    ]

    narrative = _build_comovement_narrative(
        segment_details, reference_name, comparison_name
    )

    return {
        "narrative": narrative,
        "method": "comovement",
        "n_points": n_comparison,
        "segment_details": segment_details,
        "correlation": None,
        "p_value": None,
    }
