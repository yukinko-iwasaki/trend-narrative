"""
trend_narrative.relationship
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Analyze relationships between two time series, using pre-computed segments
from a reference series to align analysis of a comparison series.

Supports two analysis modes based on data availability:
- Co-movement narrative: When data points < threshold, describes directional
  alignment within each segment of the reference series.
- Lagged correlation analysis: When sufficient data points exist, computes
  year-on-year change correlation at multiple lags to detect delayed effects.
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

THRESHOLD_LOW = 3  # Minimum for any analysis (need 2+ changes to compare)
DEFAULT_CORRELATION_THRESHOLD = 5  # Minimum for correlation (need enough change pairs)
DEFAULT_MAX_LAG_CAP = 5  # Domain-appropriate limit for policy effects
P_THRESHOLD = 0.10  # Threshold for statistical significance


def get_direction(values: np.ndarray) -> str:
    """Determine direction from start to end value."""
    if len(values) < 2:
        return "unknown"

    start, end = values[0], values[-1]
    if start == 0:
        return "increased" if end > 0 else "stable"

    pct_change = (end - start) / abs(start)

    # Less than 5% change is considered stable to avoid overstating minor fluctuations
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
    Compute annualized year-on-year percentage changes.

    Returns array of annualized percentage changes. For a 2-year gap with
    20% total change, returns 10% (annualized). Returns NaN for changes
    from near-zero base values.
    """
    if len(years) < 2:
        return np.array([])

    sorted_idx = np.argsort(years)
    years_sorted = years[sorted_idx]
    values_sorted = values[sorted_idx]

    year_gaps = np.diff(years_sorted)
    value_diffs = np.diff(values_sorted)
    base_values = values_sorted[:-1]

    # Use percentage change, annualized
    # Handle near-zero base values to avoid division issues
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_changes = (value_diffs / base_values) / year_gaps
        # Set to NaN where base value is effectively zero
        pct_changes[np.abs(base_values) < 1e-10] = np.nan

    return pct_changes


def interpolate_at_years(
    source_years: np.ndarray,
    source_values: np.ndarray,
    target_years: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolate source series to get values at target years.

    Years outside source range return NaN.
    """
    sorted_idx = np.argsort(source_years)
    source_years_sorted = source_years[sorted_idx]
    source_values_sorted = source_values[sorted_idx]

    return np.interp(
        target_years,
        source_years_sorted,
        source_values_sorted,
        left=np.nan,
        right=np.nan,
    )


def _get_boundary_value(
    target_year: float,
    comparison_years: np.ndarray,
    comparison_values: np.ndarray,
    seg_values_sorted: np.ndarray,
    use_first: bool,
) -> tuple[Optional[float], bool]:
    """Get comparison value at a segment boundary.

    Returns (value, is_interpolated). Tries in order:
    1. Exact match in data
    2. Interpolation from full series
    3. Fallback to first/last point within segment
    """
    if target_year in comparison_years:
        idx = np.where(comparison_years == target_year)[0][0]
        return float(comparison_values[idx]), False

    interpolated = interpolate_at_years(
        comparison_years, comparison_values, np.array([target_year])
    )[0]
    if not np.isnan(interpolated):
        return float(interpolated), True

    if len(seg_values_sorted) > 0:
        return float(seg_values_sorted[0] if use_first else seg_values_sorted[-1]), False

    return None, False


def analyze_segment_comovement(
    segment: dict,
    comparison_years: np.ndarray,
    comparison_values: np.ndarray,
) -> dict:
    """Analyze comparison series within a single reference segment.

    Uses interpolation to estimate comparison values at segment boundaries,
    ensuring direction is measured over the same time span as the reference.
    Falls back to actual data points if interpolation fails.
    """
    start_year = segment["start_year"]
    end_year = segment["end_year"]

    # Get actual observations within segment (sorted)
    mask = (comparison_years >= start_year) & (comparison_years <= end_year)
    seg_values_sorted = comparison_values[mask][np.argsort(comparison_years[mask])]
    n_points = len(seg_values_sorted)

    # Get boundary values
    comp_start, start_interp = _get_boundary_value(
        start_year, comparison_years, comparison_values, seg_values_sorted, use_first=True
    )
    comp_end, end_interp = _get_boundary_value(
        end_year, comparison_years, comparison_values, seg_values_sorted, use_first=False
    )

    # Need two distinct values to determine direction
    can_calc_direction = (
        comp_start is not None and
        comp_end is not None and
        comp_start != comp_end
    )

    return {
        "start_year": int(start_year),
        "end_year": int(end_year),
        "reference_direction": get_direction_from_slope(segment["slope"]),
        "reference_start": segment["start_value"],
        "reference_end": segment["end_value"],
        "comparison_n_points": n_points,
        "comparison_direction": get_direction(np.array([comp_start, comp_end])) if can_calc_direction else None,
        "comparison_start": comp_start,
        "comparison_end": comp_end,
        "interpolated": start_interp and end_interp,
    }


def compute_lagged_correlation(
    sparse_years: np.ndarray,
    sparse_values: np.ndarray,
    dense_years: np.ndarray,
    dense_values: np.ndarray,
    lag: int,
) -> Optional[dict]:
    """
    Compute correlation between changes at a specific lag.

    Correlates sparse series changes with dense series changes from `lag` years prior.
    Uses interpolation to get dense values at (sparse_years - lag).

    Returns dict with correlation, p_value, n_pairs, or None if insufficient data.
    """
    # Interpolate dense series at lagged years
    target_years = sparse_years - lag
    dense_at_lag = interpolate_at_years(dense_years, dense_values, target_years)

    # Remove any NaN (years outside dense range)
    valid_mask = ~np.isnan(dense_at_lag)
    if np.sum(valid_mask) < 2:
        return None

    valid_sparse_years = sparse_years[valid_mask]
    valid_sparse_values = sparse_values[valid_mask]
    valid_dense_values = dense_at_lag[valid_mask]

    # Sort by year
    sort_idx = np.argsort(valid_sparse_years)
    years_sorted = valid_sparse_years[sort_idx]
    sparse_sorted = valid_sparse_values[sort_idx]
    dense_sorted = valid_dense_values[sort_idx]

    # Compute changes
    sparse_changes = compute_yoy_changes(years_sorted, sparse_sorted)
    dense_changes = compute_yoy_changes(years_sorted, dense_sorted)

    n_pairs = len(sparse_changes)
    if n_pairs < 2:
        return None

    correlation, p_value = stats.pearsonr(dense_changes, sparse_changes)
    if np.isnan(correlation):
        return None

    return {
        "correlation": float(correlation),
        "p_value": float(p_value),
        "n_pairs": n_pairs,
    }


def compute_all_lagged_correlations(
    sparse_years: np.ndarray,
    sparse_values: np.ndarray,
    dense_years: np.ndarray,
    dense_values: np.ndarray,
    max_lag: int,
) -> list[dict]:
    """
    Compute correlations at all lags from 0 to max_lag.

    Returns list of lag results (with lag field added), excluding any that failed.
    """
    results = []
    for lag in range(max_lag + 1):
        result = compute_lagged_correlation(
            sparse_years, sparse_values,
            dense_years, dense_values,
            lag
        )
        if result is not None:
            result["lag"] = lag
            results.append(result)
    return results


def find_best_lag(lag_results: list[dict]) -> Optional[dict]:
    """Find the lag with strongest correlation, preferring significant results."""
    if not lag_results:
        return None

    # Prefer significant results
    significant = [r for r in lag_results if r["p_value"] < P_THRESHOLD]
    candidates = significant if significant else lag_results

    return max(candidates, key=lambda x: abs(x["correlation"]))


def _insufficient_data_result(
    reference_name: str,
    comparison_name: str,
    n_points: int,
) -> dict:
    """Return result dict for insufficient data cases."""
    return {
        "narrative": (
            f"The relationship between {reference_name} and {comparison_name} "
            "cannot be determined due to limited data availability."
        ),
        "method": "insufficient_data",
        "n_points": n_points,
        "segment_details": None,
        "best_lag": None,
        "all_lags": None,
        "max_lag_tested": None,
    }


def _format_value(value: float, fmt: str) -> str:
    """Format a numeric value using the given format spec."""
    return f"{value:{fmt}}"


def _build_comovement_narrative(
    segment_details: list[dict],
    reference_name: str,
    comparison_name: str,
    reference_format: str = ".2f",
    comparison_format: str = ".2f",
) -> str:
    """Build narrative from segment-level co-movement analysis."""
    if not segment_details:
        return f"Unable to analyze relationship between {reference_name} and {comparison_name}."

    total_comparison_points = sum(seg["comparison_n_points"] for seg in segment_details)
    if total_comparison_points == 0:
        return (
            f"The relationship between {reference_name} and {comparison_name} "
            f"cannot be determined because {comparison_name} data is not available."
        )

    narratives = []

    for i, seg in enumerate(segment_details):
        period = f"from {seg['start_year']} to {seg['end_year']}"
        ref_dir = seg["reference_direction"]
        comp_dir = seg["comparison_direction"]
        comp_n = seg["comparison_n_points"]

        ref_start = _format_value(seg["reference_start"], reference_format)
        ref_end = _format_value(seg["reference_end"], reference_format)

        # Override direction if formatted values are the same
        if ref_start == ref_end:
            ref_dir = "remained stable"

        if comp_dir is None:
            if comp_n == 0:
                seg_narrative = (
                    f"{period}, {reference_name} {ref_dir} "
                    f"({ref_start} to {ref_end}), "
                    f"but {comparison_name} data is unavailable for this period"
                )
            elif comp_n == 1:
                comp_start = _format_value(seg["comparison_start"], comparison_format)
                seg_narrative = (
                    f"{period}, {reference_name} {ref_dir} "
                    f"({ref_start} to {ref_end}), "
                    f"with only one {comparison_name} observation ({comp_start})"
                )
            else:
                # Multiple observations but all same value - remained stable
                comp_start = _format_value(seg["comparison_start"], comparison_format)
                seg_narrative = (
                    f"{period}, {reference_name} {ref_dir} ({ref_start} to {ref_end}) "
                    f"while {comparison_name} remained stable ({comp_start})"
                )
        else:
            comp_start = _format_value(seg["comparison_start"], comparison_format)
            comp_end = _format_value(seg["comparison_end"], comparison_format)

            # Override direction if formatted values are the same
            if comp_start == comp_end:
                comp_dir = "remained stable"

            # Describe co-movement
            if ref_dir == comp_dir:
                relationship = "both moving in the same direction"
            elif ref_dir == "remained stable" or comp_dir == "remained stable":
                relationship = None
            else:
                relationship = "moving in opposite directions"

            if relationship:
                seg_narrative = (
                    f"{period}, {reference_name} {ref_dir} ({ref_start} to {ref_end}) "
                    f"while {comparison_name} {comp_dir} "
                    f"({comp_start} to {comp_end}), {relationship}"
                )
            else:
                seg_narrative = (
                    f"{period}, {reference_name} {ref_dir} ({ref_start} to {ref_end}) "
                    f"while {comparison_name} {comp_dir} "
                    f"({comp_start} to {comp_end})"
                )

        # Capitalize first letter (each segment becomes a sentence)
        seg_narrative = seg_narrative[0].upper() + seg_narrative[1:]

        narratives.append(seg_narrative)

    # Join segments
    if len(narratives) == 1:
        narrative = narratives[0] + "."
    else:
        narrative = ". ".join(narratives) + "."

    # Add caveat about limited data
    total_comparison_points = sum(seg["comparison_n_points"] for seg in segment_details)
    narrative += (
        f" With only {total_comparison_points} {comparison_name} observations, "
        "a statistical relationship cannot be established."
    )

    return narrative


def _pluralize(word: str, count: int) -> str:
    """Return plural form if count != 1."""
    return word if count == 1 else f"{word}s"


def _build_lagged_correlation_narrative(
    best_lag: dict,
    all_lags: list[dict],
    n_sparse: int,
    max_lag_tested: int,
    reference_name: str,
    comparison_name: str,
    time_unit: str = "year",
) -> str:
    """Build narrative from lagged correlation analysis."""
    correlation = best_lag["correlation"]
    p_value = best_lag["p_value"]
    lag = best_lag["lag"]
    n_pairs = best_lag["n_pairs"]

    strength = get_correlation_strength(correlation)
    is_significant = p_value < P_THRESHOLD

    # Build lag timing description
    if lag == 0:
        timing = f"in the same {time_unit}"
    else:
        timing = f"about {lag} {_pluralize(time_unit, lag)} later"

    if strength == "no" or not is_significant:
        # Not significant: lead with uncertainty
        narrative = (
            f"No reliable relationship was detected between changes in {reference_name} "
            f"and {comparison_name}. "
        )
        if strength != "no":
            narrative += (
                f"While the data suggests a {strength} {'positive' if correlation > 0 else 'negative'} "
                f"pattern (r={correlation:.2f}), this could be due to chance "
                f"given the limited sample size (n={n_pairs} change pairs, p={p_value:.2f})."
            )
        else:
            lag_info = (
                "" if max_lag_tested == 0
                else f" at any lag tested (0-{max_lag_tested} {_pluralize(time_unit, max_lag_tested)})"
            )
            narrative += (
                f"Changes in one do not appear to be associated with changes in the other{lag_info}, "
                f"based on {n_pairs} {time_unit}-over-{time_unit} comparisons."
            )
    else:
        # Significant: lead with the finding
        direction_word = "increase" if correlation > 0 else "decrease"
        narrative = (
            f"When {reference_name} increases, {comparison_name} tends to "
            f"{direction_word} {timing}. "
            f"This is a {strength} relationship (r={correlation:.2f}) "
            f"and is statistically reliable (p={p_value:.3f}), "
            f"based on {n_pairs} {time_unit}-over-{time_unit} comparisons."
        )

    return narrative


def get_relationship_narrative(
    reference_years: "array-like",
    reference_values: "array-like",
    comparison_years: "array-like",
    comparison_values: "array-like",
    reference_name: str,
    comparison_name: str,
    reference_segments: Optional[list[dict]] = None,
    correlation_threshold: int = DEFAULT_CORRELATION_THRESHOLD,
    max_lag_cap: int = DEFAULT_MAX_LAG_CAP,
    reference_format: str = ".2f",
    comparison_format: str = ".2f",
    time_unit: str = "year",
) -> dict:
    """
    Analyze relationship between two time series.

    Segments can be provided directly or computed from reference data.
    Chooses analysis method based on data availability:
    - Insufficient data: < 3 points
    - Comovement: >= 3 but < correlation_threshold points
    - Lagged correlation: >= correlation_threshold points

    Parameters
    ----------
    reference_years : array-like
        Year values for reference series (the "driver" or spending series).
    reference_values : array-like
        Data values for reference series.
    comparison_years : array-like
        Year values for the comparison series (the outcome series).
    comparison_values : array-like
        Data values for the comparison series.
    reference_name : str
        Display name for the reference series.
    comparison_name : str
        Display name for the comparison series.
    reference_segments : list[dict], optional
        Pre-computed segments from InsightExtractor for the reference series.
        Each dict should contain: start_year, end_year, slope.
        If not provided, computed from reference_years/reference_values.
    correlation_threshold : int
        Minimum points to use correlation analysis (default 5).
        Below this, comovement analysis is used.
    max_lag_cap : int
        Maximum lag to test in years (default 5). Actual max lag may be
        lower if data is insufficient.
    reference_format : str
        Format spec for reference series values in narratives (default ".2f").
    comparison_format : str
        Format spec for comparison series values in narratives (default ".2f").
    time_unit : str
        Time unit label for narratives (default "year"). Use "month", "quarter", etc.

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
    """
    reference_years = np.asarray(reference_years, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    comparison_years = np.asarray(comparison_years, dtype=float)
    comparison_values = np.asarray(comparison_values, dtype=float)

    # Remove NaN values
    valid_mask = ~np.isnan(reference_values)
    reference_years = reference_years[valid_mask]
    reference_values = reference_values[valid_mask]

    valid_mask = ~np.isnan(comparison_values)
    comparison_years = comparison_years[valid_mask]
    comparison_values = comparison_values[valid_mask]

    # Sort by year
    ref_sort = np.argsort(reference_years)
    reference_years = reference_years[ref_sort]
    reference_values = reference_values[ref_sort]

    comp_sort = np.argsort(comparison_years)
    comparison_years = comparison_years[comp_sort]
    comparison_values = comparison_values[comp_sort]

    n_reference = len(reference_values)
    n_comparison = len(comparison_values)

    # Correlation is limited by the sparser series; identify which is which
    # so we can interpolate the dense series at sparse series years
    if n_comparison <= n_reference:
        sparse_years, sparse_values = comparison_years, comparison_values
        dense_years, dense_values = reference_years, reference_values
    else:
        sparse_years, sparse_values = reference_years, reference_values
        dense_years, dense_values = comparison_years, comparison_values

    n_sparse = len(sparse_years)

    # Insufficient data: too few points
    if n_sparse < THRESHOLD_LOW:
        return _insufficient_data_result(reference_name, comparison_name, n_sparse)

    # Try correlation analysis first if we have enough points
    if n_sparse >= correlation_threshold:
        # Higher lags reduce usable data points; limit max lag to ensure
        # we still have enough change pairs for meaningful correlation
        n_changes = n_sparse - 1
        min_pairs_needed = correlation_threshold - 1
        max_testable_lag = max(0, n_changes - min_pairs_needed)
        max_lag = min(max_testable_lag, max_lag_cap)

        # Compute correlations at all lags
        lag_results = compute_all_lagged_correlations(
            sparse_years, sparse_values,
            dense_years, dense_values,
            max_lag
        )

        if lag_results:
            best_lag = find_best_lag(lag_results)

            narrative = _build_lagged_correlation_narrative(
                best_lag, lag_results, n_sparse, max_lag,
                reference_name, comparison_name, time_unit
            )

            return {
                "narrative": narrative,
                "method": "lagged_correlation",
                "n_points": n_sparse,
                "segment_details": None,
                "best_lag": best_lag,
                "all_lags": lag_results,
                "max_lag_tested": max_lag,
            }

    # Fall back to comovement analysis - compute segments if not provided
    if reference_segments is None:
        from .extractor import InsightExtractor
        extractor = InsightExtractor(reference_years, reference_values)
        reference_segments = extractor.get_structural_segments()

    if not reference_segments:
        return _insufficient_data_result(reference_name, comparison_name, n_sparse)

    segment_details = [
        analyze_segment_comovement(seg, comparison_years, comparison_values)
        for seg in reference_segments
    ]

    narrative = _build_comovement_narrative(
        segment_details, reference_name, comparison_name,
        reference_format=reference_format, comparison_format=comparison_format
    )

    return {
        "narrative": narrative,
        "method": "comovement",
        "n_points": n_sparse,
        "segment_details": segment_details,
        "best_lag": None,
        "all_lags": None,
        "max_lag_tested": None,
    }
