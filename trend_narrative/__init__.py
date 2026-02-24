"""
trend_narrative
~~~~~~~~~~~~~~~
A standalone Python package for piecewise-linear trend detection and
plain-English narrative generation.

Two calling paths are supported:

Path 1 – precomputed data (e.g. from a Delta table):

    from trend_narrative import get_segment_narrative

    narrative = get_segment_narrative(
        segments=row["segments"],
        cv_value=row["cv_value"],
        metric="health spending",
    )

Path 2 – raw data (standalone):

    from trend_narrative import get_segment_narrative

    narrative = get_segment_narrative(
        x=years_array,
        y=values_array,
        metric="health spending",
    )

Or use InsightExtractor directly for the extraction layer:

    from trend_narrative import InsightExtractor, get_segment_narrative

    extractor = InsightExtractor(x, y)
    suite = extractor.extract_full_suite()
    narrative = get_segment_narrative(
        segments=suite["segments"],
        cv_value=suite["cv_value"],
        metric="health spending",
    )
"""

from .detector import TrendDetector
from .extractor import InsightExtractor
from .narrative import (
    consolidate_segments,
    get_segment_narrative,
    millify,
)

__all__ = [
    "TrendDetector",
    "InsightExtractor",
    "consolidate_segments",
    "get_segment_narrative",
    "millify",
]

__version__ = "0.1.0"
