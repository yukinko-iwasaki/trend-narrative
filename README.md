# trend-narrative

A standalone Python package that combines **piecewise-linear trend detection** and **plain-English narrative generation** for time-series data.

Originally developed across two repos:
- Trend detection logic → `dime-worldbank/mega-boost` (`feature/add_trend_detection_model`)
- Narrative generation → `dime-worldbank/rpf-country-dash` (`enhancement/trend_narrative`)

---

## Installation

```bash
pip install -e ".[dev]"   # editable install with dev dependencies
```

Dependencies: `numpy`, `scipy`, `pwlf`

---

## Quick start

### One-liner

```python
import numpy as np
from trend_narrative import generate_narrative

x = np.arange(2010, 2022, dtype=float)
y = np.array([100, 105, 112, 108, 115, 130, 125, 120, 118, 122, 135, 148], dtype=float)

result = generate_narrative(x, y, metric="total real expenditure")
print(result["narrative"])
# → "From 2010 to 2015, the total real expenditure showed an upward trend.
#    Trend then shifted, reaching a peak in 2015 before reversing into a decline. ..."
print(f"CV: {result['cv_value']:.1f}%")
print(result["segments"])
```

### Step-by-step

```python
from trend_narrative import InsightExtractor, get_segment_narrative

extractor = InsightExtractor(x, y)
suite = extractor.extract_full_suite()
# suite = {"cv_value": 14.2, "segments": [...]}

narrative = get_segment_narrative(
    suite["segments"],
    cv_value=suite["cv_value"],
    metric="health spending",
)
print(narrative)
```

---

## API reference

### `TrendDetector(max_segments=3, threshold=0.05)`

Fits a piecewise-linear model using BIC-optimised segment count and snaps
breakpoints to integer years / local extrema.

| Method | Returns | Description |
|---|---|---|
| `extract_trend(x, y)` | `list[dict]` | Fit model; return per-segment stats |
| `fit_best_model(x, y)` | `pwlf model \| None` | Run both fitting passes |
| `calculate_bic(ssr, n, k)` | `float` | Static BIC helper |

Each segment dict contains: `start_year`, `end_year`, `start_value`,
`end_value`, `slope`, `p_value`.

---

### `InsightExtractor(x, y, detector=None)`

High-level facade combining volatility and trend detection.

| Method | Returns | Description |
|---|---|---|
| `get_volatility()` | `float` | Coefficient of Variation (%) |
| `get_structural_segments()` | `list[dict]` | Delegates to `TrendDetector` |
| `extract_full_suite()` | `dict` | `{cv_value, segments}` |

---

### `get_segment_narrative(segments, cv_value, metric="expenditure")`

Convert segment data into a multi-sentence English narrative.

- No segments + low CV → *"remained highly stable"*
- No segments + high CV → *"exhibited significant volatility"*
- Single segment → direction + % change sentence
- Multi-segment → transition phrases (peak / trough / continuation)

---

### `consolidate_segments(segments)`

Merge consecutive segments sharing the same slope direction. Useful for
simplifying noisy multi-segment fits before narrative generation.

---

### `millify(n)`

Format large numbers with suffix: `1_500_000 → "1.50 M"`.

---

## Running tests

```bash
pytest
# or with coverage:
pytest --cov=trend_narrative --cov-report=term-missing
```

---

## Project structure

```
trend-narrative/
├── trend_narrative/
│   ├── __init__.py        # Public API
│   ├── detector.py        # TrendDetector – piecewise-linear fitting
│   ├── extractor.py       # InsightExtractor – volatility + trend facade
│   └── narrative.py       # Narrative generation + millify helper
├── tests/
│   ├── test_detector.py
│   ├── test_extractor.py
│   └── test_narrative.py
├── pyproject.toml
└── README.md
```
