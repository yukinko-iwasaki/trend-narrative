"""
trend_narrative
~~~~~~~~~~~~~~~
A standalone Python package for piecewise-linear trend detection and
plain-English narrative generation.

Two calling paths are supported:

Path 1 – precomputed data (e.g. segments already stored in a Delta table):

    from trend_narrative import get_segment_narrative

    narrative = get_segment_narrative(
        segments=row["segments"],
        cv_value=row["cv_value"],
        metric="health spending",
    )

Path 2 – standalone (create an InsightExtractor with your chosen detector,
then pass it to the narrative function):

    from trend_narrative import InsightExtractor, TrendDetector, get_segment_narrative

    extractor = InsightExtractor(x, y, detector=TrendDetector(max_segments=2))
    narrative = get_segment_narrative(extractor=extractor, metric="health spending")

Keeping the extractor construction separate means you can plug in any
custom detector without changing the narrative layer.
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
