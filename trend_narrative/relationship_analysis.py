"""
trend_narrative.relationship_analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Analyze relationships between two time series, using pre-computed segments
from a reference series to align analysis of a comparison series.

Supports two analysis modes based on data availability:
- Co-movement analysis: When data points < threshold, analyzes directional
  alignment within each segment of the reference series.
- Lagged correlation analysis: When sufficient data points exist, computes
  year-on-year change correlation at multiple lags to detect delayed effects.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats


CORRELATION_THRESHOLDS = {
    0.1: "strength_no",
    0.3: "strength_weak",
    0.5: "strength_moderate",
    0.7: "strength_strong",
    1.0: "strength_very_strong",
}

THRESHOLD_LOW = 3  # Minimum for any analysis (need 2+ changes to compare)
DEFAULT_CORRELATION_THRESHOLD = 5  # Minimum for correlation (need enough change pairs)
DEFAULT_MAX_LAG_CAP = 5  # Domain-appropriate limit for policy effects
P_THRESHOLD = 0.10  # Threshold for statistical significance


def get_direction(values: np.ndarray) -> str:
    """Determine direction from start to end value.

    Returns a language-neutral key: ``"increased"``, ``"decreased"``,
    ``"remained_stable"``, or ``"unknown"``. Translation into display text
    is the narrative layer's responsibility.
    """
    if len(values) < 2:
        return "unknown"

    start, end = values[0], values[-1]

    # NaN/inf would propagate silently through the comparisons below
    # (NaN > 0 is False, NaN != 0 is True) and produce "decreased" for any
    # non-finite input — caller pre-filtering can't always be assumed.
    if not (np.isfinite(start) and np.isfinite(end)):
        return "unknown"

    # Use the non-zero value as denominator; if both zero, stable
    if start == 0 and end == 0:
        return "remained_stable"
    denominator = abs(start) if start != 0 else abs(end)
    pct_change = (end - start) / denominator

    # Less than 5% change is considered stable to avoid overstating minor fluctuations
    if abs(pct_change) < 0.05:
        return "remained_stable"
    return "increased" if pct_change > 0 else "decreased"


def get_correlation_strength(corr: float) -> str:
    """Map correlation coefficient to a descriptive strength key.

    Returns a language-neutral key from :data:`CORRELATION_THRESHOLDS`
    (e.g. ``"strength_weak"``). Translation into display text is the
    narrative layer's responsibility.
    """
    abs_corr = abs(corr)
    for threshold, key in sorted(CORRELATION_THRESHOLDS.items()):
        if abs_corr <= threshold:
            return key
    return "strength_very_strong"


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
        "reference_direction": get_direction(np.array([segment["start_value"], segment["end_value"]])),
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

    # Filter to finite pairs only (handles NaN/inf from near-zero bases or zero year gaps)
    finite_mask = np.isfinite(sparse_changes) & np.isfinite(dense_changes)
    sparse_changes = sparse_changes[finite_mask]
    dense_changes = dense_changes[finite_mask]

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

    significant = [r for r in lag_results if r["p_value"] < P_THRESHOLD]
    candidates = significant if significant else lag_results

    return max(candidates, key=lambda x: abs(x["correlation"]))


def analyze_relationship(
    reference_years: "array-like",
    reference_values: "array-like",
    comparison_years: "array-like",
    comparison_values: "array-like",
    reference_segments: Optional[list[dict]] = None,
    correlation_threshold: int = DEFAULT_CORRELATION_THRESHOLD,
    max_lag_cap: int = DEFAULT_MAX_LAG_CAP,
) -> dict:
    """
    Analyze relationship between two time series.

    Returns structured, **language-neutral** insights without generating
    narrative text. Direction and strength fields are stable keys (e.g.
    ``"increased"``, ``"strength_weak"``) — translation is performed by the
    narrative layer. This makes the returned dict safe to store and reuse
    across languages.

    Use this function when you want to inspect or store the analysis results
    separately from narrative generation.

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

    Returns
    -------
    dict
        Keys:
        - method: str, one of "insufficient_data", "comovement", "lagged_correlation"
        - n_points: int, number of data points in sparser series
        - segment_details: list[dict], per-segment analysis (comovement only)
        - best_lag: dict with lag, correlation, p_value, n_pairs (correlation only)
        - all_lags: list[dict], results for all tested lags (correlation only)
        - max_lag_tested: int, maximum lag that was tested (correlation only)
        - reference_leads: bool, whether reference series leads comparison
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
    # so we can interpolate the dense series at sparse series years.
    if n_comparison <= n_reference:
        sparse_years, sparse_values = comparison_years, comparison_values
        dense_years, dense_values = reference_years, reference_values
        reference_leads = True  # reference is dense, so "reference leads comparison"
    else:
        sparse_years, sparse_values = reference_years, reference_values
        dense_years, dense_values = comparison_years, comparison_values
        reference_leads = False  # comparison is dense, so "comparison leads reference"

    n_sparse = len(sparse_years)

    # Insufficient data: too few points
    if n_sparse < THRESHOLD_LOW:
        return {
            "method": "insufficient_data",
            "n_points": n_sparse,
            "segment_details": None,
            "best_lag": None,
            "all_lags": None,
            "max_lag_tested": None,
            "reference_leads": reference_leads,
        }

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
            return {
                "method": "lagged_correlation",
                "n_points": n_sparse,
                "segment_details": None,
                "best_lag": find_best_lag(lag_results),
                "all_lags": lag_results,
                "max_lag_tested": max_lag,
                "reference_leads": reference_leads,
            }

    # Fall back to comovement analysis - compute segments if not provided
    if reference_segments is None:
        from .extractor import InsightExtractor
        extractor = InsightExtractor(reference_years, reference_values)
        reference_segments = extractor.get_structural_segments()

    if not reference_segments:
        return {
            "method": "insufficient_data",
            "n_points": n_sparse,
            "segment_details": None,
            "best_lag": None,
            "all_lags": None,
            "max_lag_tested": None,
            "reference_leads": reference_leads,
        }

    segment_details = [
        analyze_segment_comovement(seg, comparison_years, comparison_values)
        for seg in reference_segments
    ]

    return {
        "method": "comovement",
        "n_points": n_sparse,
        "segment_details": segment_details,
        "best_lag": None,
        "all_lags": None,
        "max_lag_tested": None,
        "reference_leads": reference_leads,
    }
