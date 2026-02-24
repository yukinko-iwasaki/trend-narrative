"""
trend_narrative.detector
~~~~~~~~~~~~~~~~~~~~~~~~
Piecewise-linear trend detection using BIC-optimised segment count and
breakpoint snapping to local extrema.

Ported from dime-worldbank/mega-boost analytics/insight_extractor.py
"""

from __future__ import annotations

import math
from itertools import product
from typing import Optional

import numpy as np
import pwlf
from scipy.signal import find_peaks


class TrendDetector:
    """Fit a piecewise-linear model to a 1-D time series.

    Parameters
    ----------
    max_segments : int
        Maximum number of linear segments to consider (default 3).
    threshold : float
        p-value threshold for slope significance (default 0.05).
    """

    def __init__(self, max_segments: int = 3, threshold: float = 0.05) -> None:
        self.max_segments = max_segments
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_bic(ssr: float, n_data_points: int, n_segments: int) -> float:
        """Bayesian Information Criterion for a piecewise-linear fit.

        Parameters
        ----------
        ssr : float
            Sum of squared residuals from the fit.
        n_data_points : int
            Number of observations.
        n_segments : int
            Number of linear segments in the model.

        Returns
        -------
        float
            BIC score (lower is better).
        """
        k = (2 * n_segments) + (n_segments - 1)
        return n_data_points * np.log(ssr / n_data_points) + k * np.log(n_data_points)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def find_local_maxima_years(
        self, x: np.ndarray, y: np.ndarray
    ) -> list[float]:
        """Return the x-values (years) at local peaks and valleys of *y*."""
        peaks, _ = find_peaks(y)
        valleys, _ = find_peaks(-y)
        extrema_indices = set(peaks) | set(valleys)
        return [x[i] for i in range(len(x)) if i in extrema_indices]

    def is_valid_fit(self, model: pwlf.PiecewiseLinFit) -> bool:
        """Return True when every slope knot is statistically significant
        and no two breakpoints coincide."""
        breakpoints = [int(b) for b in model.fit_breaks]
        if len(breakpoints) != len(set(breakpoints)):
            return False
        for p_val in model.p_values()[1:]:
            if p_val > self.threshold:
                return False
        return True

    # ------------------------------------------------------------------
    # Two-pass fitting
    # ------------------------------------------------------------------

    def _find_preliminary_model(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[Optional[pwlf.PiecewiseLinFit], int]:
        """Pass 1 – find the best segment count and rough breakpoints."""
        best_model: Optional[pwlf.PiecewiseLinFit] = None
        best_seg_count = 0

        for n_seg in range(1, self.max_segments + 1):
            if len(x) - n_seg <= 1:
                break
            try:
                model = pwlf.PiecewiseLinFit(x, y)
                model.fit(n_seg, seed=42)
                if self.is_valid_fit(model):
                    best_model = model
                    best_seg_count = n_seg
            except Exception as exc:  # noqa: BLE001
                print(f"Fit failed for {n_seg} segments: {exc}")
                continue

        return best_model, best_seg_count

    def _refine_to_narrative_milestones(
        self,
        x: np.ndarray,
        y: np.ndarray,
        prelim_model: pwlf.PiecewiseLinFit,
        seg_count: int,
    ) -> Optional[pwlf.PiecewiseLinFit]:
        """Pass 2 – snap breakpoints to integer years / local extrema
        and choose the candidate with the lowest BIC."""
        local_extrema = self.find_local_maxima_years(x, y)
        raw_breaks = prelim_model.fit_breaks
        neighbor_sets: list[list[float]] = []

        for b in raw_breaks:
            low, high = math.floor(b), math.ceil(b)
            if low == high:
                neighbor_sets.append([low])
            elif low in local_extrema:
                neighbor_sets.append([low])
            elif high in local_extrema:
                neighbor_sets.append([high])
            else:
                neighbor_sets.append([low, high])

        best_final_model: Optional[pwlf.PiecewiseLinFit] = None
        min_bic = np.inf

        for candidate in product(*neighbor_sets):
            if len(candidate) != len(set(candidate)):
                continue
            model = pwlf.PiecewiseLinFit(x, y)
            ssr = model.fit_with_breaks(candidate)
            current_bic = self.calculate_bic(ssr, len(x), seg_count)
            if current_bic < min_bic:
                min_bic = current_bic
                best_final_model = model

        return best_final_model

    def fit_best_model(
        self, x: np.ndarray, y: np.ndarray
    ) -> Optional[pwlf.PiecewiseLinFit]:
        """Run both passes and return the best-fitting piecewise model,
        or *None* if no valid fit was found."""
        prelim_model, seg_count = self._find_preliminary_model(x, y)
        if prelim_model is None:
            return None
        return self._refine_to_narrative_milestones(x, y, prelim_model, seg_count)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_trend(
        self,
        x: np.ndarray,
        y: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> list[dict]:
        """Fit the model and return per-segment statistics.

        Parameters
        ----------
        x : array-like
            1-D array of x-values (e.g. years), must be numeric.
        y : array-like
            1-D array of observed values aligned with *x*.
        metadata : dict, optional
            Unused placeholder for caller-provided context (kept for
            API compatibility).

        Returns
        -------
        list[dict]
            One dict per segment with keys:
            ``start_year``, ``end_year``, ``start_value``, ``end_value``,
            ``slope``, ``p_value``.
            Empty list when no valid fit is found.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        model = self.fit_best_model(x, y)
        if model is None:
            return []

        p_values = model.p_values()
        slopes = np.cumsum(model.beta[1:])
        breaks = model.fit_breaks
        segments: list[dict] = []

        for i in range(len(breaks) - 1):
            start_year = breaks[i]
            end_year = breaks[i + 1]
            start_index = min(np.searchsorted(x, start_year), len(y) - 1)
            end_index = min(np.searchsorted(x, end_year), len(y) - 1)
            segments.append(
                {
                    "start_year": start_year,
                    "end_year": end_year,
                    "start_value": y[start_index],
                    "end_value": y[end_index],
                    "slope": slopes[i],
                    "p_value": p_values[i],
                }
            )

        return segments
