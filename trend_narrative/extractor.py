"""
trend_narrative.extractor
~~~~~~~~~~~~~~~~~~~~~~~~~
High-level facade that combines volatility measurement with piecewise-linear
trend detection.

Ported from dime-worldbank/mega-boost analytics/insight_extractor.py
"""

from __future__ import annotations

import numpy as np

from .detector import TrendDetector


class InsightExtractor:
    """Extract statistical insights from a univariate time series.

    Parameters
    ----------
    x : array-like
        1-D array of x-values (e.g. integer years).
    y : array-like
        1-D array of observed metric values aligned with *x*.
    detector : TrendDetector, optional
        Custom detector instance.  A default :class:`TrendDetector` is
        used when not provided.

    Examples
    --------
    >>> import numpy as np
    >>> from trend_narrative import InsightExtractor
    >>> x = np.arange(2010, 2022)
    >>> y = np.array([100, 105, 110, 108, 115, 130, 125, 120, 118, 122, 130, 140])
    >>> extractor = InsightExtractor(x, y)
    >>> result = extractor.extract_full_suite()
    >>> "cv_value" in result and "segments" in result
    True
    """

    def __init__(
        self,
        x: "array-like",
        y: "array-like",
        detector: TrendDetector | None = None,
    ) -> None:
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.trend_detector = detector if detector is not None else TrendDetector()

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def get_volatility(self) -> float:
        """Coefficient of Variation (%) – a measure of relative spread.

        Returns
        -------
        float
            ``(std / mean) * 100``.  Returns *NaN* when the mean is zero.
        """
        mean = self.y.mean()
        if mean == 0:
            return float("nan")
        return float((self.y.std() / mean) * 100)

    def get_structural_segments(self) -> list[dict]:
        """Run the trend detector and return per-segment statistics.

        Returns
        -------
        list[dict]
            See :meth:`TrendDetector.extract_trend` for the dict schema.
        """
        return self.trend_detector.extract_trend(self.x, self.y)

    # ------------------------------------------------------------------
    # Convenience bundle
    # ------------------------------------------------------------------

    def extract_full_suite(self) -> dict:
        """Return all insights as a single flat dictionary.

        Keys
        ----
        cv_value : float
            Coefficient of Variation (%).
        segments : list[dict]
            Piecewise-linear segment statistics.
        """
        return {
            "cv_value": self.get_volatility(),
            "segments": self.get_structural_segments(),
        }
