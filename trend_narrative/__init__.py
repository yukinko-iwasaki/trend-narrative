"""
trend_narrative
~~~~~~~~~~~~~~~
A standalone Python package for piecewise-linear trend detection and
plain-English narrative generation.

Typical usage
-------------
>>> from trend_narrative import generate_narrative
>>> import numpy as np
>>> x = np.arange(2010, 2022)
>>> y = np.array([100,105,110,108,115,130,125,120,118,122,130,140], dtype=float)
>>> result = generate_narrative(x, y, metric="health spending")
>>> print(result["narrative"])

Or step-by-step:

>>> from trend_narrative import InsightExtractor, get_segment_narrative
>>> extractor = InsightExtractor(x, y)
>>> suite = extractor.extract_full_suite()
>>> print(get_segment_narrative(suite["segments"], suite["cv_value"], metric="spending"))
"""

from .detector import TrendDetector
from .extractor import InsightExtractor
from .narrative import (
    consolidate_segments,
    generate_narrative,
    get_segment_narrative,
    millify,
)

__all__ = [
    "TrendDetector",
    "InsightExtractor",
    "consolidate_segments",
    "generate_narrative",
    "get_segment_narrative",
    "millify",
]

__version__ = "0.1.0"
