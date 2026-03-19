# trend-narrative

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://worldbank.github.io/trend-narrative/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

## Overview

The **trend-narrative** package is a standalone Python library that combines **piecewise-linear trend detection**, **relationship analysis**, and **plain-English narrative generation** for time-series data.

Given a time series — such as annual health spending or GDP figures — this package automatically identifies meaningful trends (e.g., "rising from 2010 to 2015, then declining") and produces a ready-to-use English sentence describing them. It can also compare two time series and explain how they move together or apart over time.

This is useful for analysts, researchers, and developers who need to turn numeric data into human-readable summaries without writing custom text logic each time.

## Documentation

Full documentation is available at **[https://worldbank.github.io/trend-narrative/](https://worldbank.github.io/trend-narrative/)**.

## Getting Started

### Prerequisites

- Python >= 3.9
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
uv add trend-narrative
```

**For development** (editable install with test dependencies):

```bash
git clone https://github.com/worldbank/trend-narrative.git
cd trend-narrative
uv sync --extra dev
```

Dependencies: `numpy`, `scipy`, `pwlf`

### Quick Example

```python
import numpy as np
from trend_narrative import InsightExtractor, TrendDetector, get_segment_narrative

x = np.arange(2010, 2022, dtype=float)
y = np.array([100, 105, 112, 108, 115, 130, 125, 120, 118, 122, 135, 148], dtype=float)

extractor = InsightExtractor(x, y, detector=TrendDetector(max_segments=2))
narrative = get_segment_narrative(extractor=extractor, metric="health spending")
print(narrative)
# → "From 2010 to 2015, the health spending showed an upward trend.
#    Trend then shifted, reaching a peak in 2015 before reversing into a decline."
```

---

## Usage

### Trend Narratives

#### Path 1 — from raw data

Create an `InsightExtractor` with your chosen detector, then pass it to the
narrative function. Keeping the two steps separate means you can swap in any
custom detector without touching the narrative layer:

```python
import numpy as np
from trend_narrative import InsightExtractor, TrendDetector, get_segment_narrative

x = np.arange(2010, 2022, dtype=float)
y = np.array([100, 105, 112, 108, 115, 130, 125, 120, 118, 122, 135, 148], dtype=float)

extractor = InsightExtractor(x, y, detector=TrendDetector(max_segments=2))
narrative = get_segment_narrative(extractor=extractor, metric="health spending")
```

You can also call the extraction step separately if you need the raw numbers:

```python
suite = extractor.extract_full_suite()
# {"cv_value": 14.2, "segments": [...], "n_points": 12}
```

#### Path 2 — from precomputed data

If you already have segments and a CV value stored (e.g. from a database or
a previous extraction run), pass them directly — no re-fitting required:

```python
from trend_narrative import get_segment_narrative

narrative = get_segment_narrative(
    segments=row["segments"],
    cv_value=row["cv_value"],
    metric="health spending",
)
```

### Relationship Narratives

Analyze the relationship between two time series (e.g., spending vs outcomes).

#### Path 1 — from raw data

```python
import numpy as np
from trend_narrative import get_relationship_narrative

result = get_relationship_narrative(
    reference_years=np.array([2010, 2012, 2014, 2016, 2018]),
    reference_values=np.array([100, 120, 140, 160, 180]),
    comparison_years=np.array([2010, 2012, 2014, 2016, 2018]),
    comparison_values=np.array([50, 55, 62, 70, 78]),
    reference_name="spending",
    comparison_name="outcome",
)
print(result["narrative"])
# → "When spending increases, outcome tends to increase in the same year..."
print(result["method"])  # "lagged_correlation", "comovement", or "insufficient_data"
```

#### Path 2 — from precomputed insights

```python
from trend_narrative import get_relationship_narrative

narrative = get_relationship_narrative(
    insights=row["relationship_insights"],
    reference_name="spending",
    comparison_name="outcome",
)
print(narrative["narrative"])
```

#### Separate analysis and narrative generation

Use `analyze_relationship()` when you want to inspect or store the analysis
results separately from narrative generation:

```python
from trend_narrative import analyze_relationship, get_relationship_narrative

insights = analyze_relationship(
    reference_years=years1,
    reference_values=values1,
    comparison_years=years2,
    comparison_values=values2,
)
# Store insights in database, inspect programmatically, etc.
print(insights["method"])  # "lagged_correlation", "comovement", or "insufficient_data"
print(insights["best_lag"])  # lag details for correlation path

# Generate narrative later from stored insights
result = get_relationship_narrative(
    insights=insights,
    reference_name="spending",
    comparison_name="outcome",
)
```

The function automatically chooses the analysis method based on data availability:
- **Lagged correlation**: >= 5 points, tests correlations at various lags
- **Comovement**: 3-4 points, describes directional movement within segments
- **Insufficient data**: < 3 points

---

## API Reference

### `get_segment_narrative(segments, cv_value, metric="expenditure")`
### `get_segment_narrative(extractor, metric="expenditure")`

Generates a plain-English narrative for a single time series. Accepts either
precomputed data or an `InsightExtractor` instance.

- No segments + low CV → *"remained highly stable"*
- No segments + high CV → *"exhibited significant volatility"*
- Single segment → direction + % change sentence
- Multi-segment → transition phrases (peak / trough / continuation)

### `analyze_relationship(...)`

Analyzes the relationship between two time series and returns structured
insights without generating narrative text.

```python
analyze_relationship(
    reference_years,           # array-like, the "driver" series years
    reference_values,          # array-like, the "driver" series values
    comparison_years,          # array-like, the "outcome" series years
    comparison_values,         # array-like, the "outcome" series values
    reference_segments=None,   # optional pre-computed segments
    correlation_threshold=5,   # min points for correlation analysis
    max_lag_cap=5,             # max lag to test in years
)
```

Returns a dict with:
- `method`: "lagged_correlation", "comovement", or "insufficient_data"
- `n_points`: int, number of points in sparser series
- `segment_details`: list[dict], per-segment analysis (comovement only)
- `best_lag`: dict with lag, correlation, p_value, n_pairs (correlation only)
- `all_lags`: list of all tested lags (correlation only)
- `max_lag_tested`: int, maximum lag tested (correlation only)
- `reference_leads`: bool, whether reference series leads comparison

### `get_relationship_narrative(...)`

Generates a narrative from relationship analysis. Accepts either precomputed
insights or raw data arrays.

```python
get_relationship_narrative(
    # Raw data (optional if insights provided)
    reference_years=None,      # array-like, the "driver" series years
    reference_values=None,     # array-like, the "driver" series values
    comparison_years=None,     # array-like, the "outcome" series years
    comparison_values=None,    # array-like, the "outcome" series values
    # Required for narrative
    reference_name="",         # str, display name for reference
    comparison_name="",        # str, display name for comparison
    # Optional parameters
    reference_segments=None,   # optional pre-computed segments
    correlation_threshold=5,   # min points for correlation analysis
    max_lag_cap=5,             # max lag to test in years
    reference_format=".2f",    # format spec or callable for reference values
    comparison_format=".2f",   # format spec or callable for comparison values
    time_unit="year",          # "year", "month", "quarter" for narratives
    reference_leads=None,      # True/False to override, None to infer
    # Precomputed insights
    insights=None,             # dict from analyze_relationship()
)
```

Returns a dict with:
- `narrative`: str, human-readable description
- `method`: "lagged_correlation", "comovement", or "insufficient_data"
- `n_points`: int, number of points in sparser series
- `segment_details`: list[dict], per-segment analysis (comovement only)
- `best_lag`: dict with lag details (correlation path only)
- `all_lags`: list of all tested lags (correlation path only)
- `max_lag_tested`: int, maximum lag tested (correlation only)

### `TrendDetector(max_segments=3, threshold=0.05)`

Fits a piecewise-linear model using BIC-optimised segment count, snapping
breakpoints to integer years and local extrema.

| Method | Returns | Description |
|---|---|---|
| `extract_trend(x, y)` | `list[dict]` | Fit model; return per-segment stats |
| `fit_best_model(x, y)` | `pwlf model \| None` | Run both fitting passes |
| `calculate_bic(ssr, n, k)` | `float` | Static BIC helper |

Each segment dict contains: `start_year`, `end_year`, `start_value`,
`end_value`, `slope`, `p_value`.

### `InsightExtractor(x, y, detector=None)`

Combines volatility measurement with trend detection. Pass a custom detector
to control the fitting logic.

| Method | Returns | Description |
|---|---|---|
| `get_volatility()` | `float` | Coefficient of Variation (%) |
| `get_structural_segments()` | `list[dict]` | Delegates to the detector |
| `extract_full_suite()` | `dict` | `{cv_value, segments, n_points}` |

### `consolidate_segments(segments)`

Merges consecutive segments that share the same slope direction. Applied
automatically inside `get_segment_narrative`.

### `millify(n)`

Formats large numbers with a human-readable suffix: `1_500_000 → "1.50 M"`.

---

## Running Tests

```bash
uv run pytest
# or with coverage:
uv run pytest --cov=trend_narrative --cov-report=term-missing
```

---

## Project Structure

```
trend-narrative/
├── trend_narrative/
│   ├── __init__.py              # Public API
│   ├── detector.py              # TrendDetector – piecewise-linear fitting
│   ├── extractor.py             # InsightExtractor – volatility + trend facade
│   ├── narrative.py             # Narrative generation + millify helper
│   ├── relationship_analysis.py # Relationship analysis between two series
│   └── relationship_narrative.py # Relationship narrative generation
├── tests/
│   ├── test_detector.py
│   ├── test_extractor.py
│   ├── test_narrative.py
│   ├── test_relationship_analysis.py
│   └── test_relationship_narrative.py
├── pyproject.toml
└── README.md
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## Contact

For questions, feedback, or collaboration enquiries, please reach out to [ysuzuki2@worldbank.org](mailto:ysuzuki2@worldbank.org) and [wlu4@worldbank.org](mailto:wlu4@worldbank.org).

## License

This project is licensed under the MIT License together with the [World Bank IGO Rider](WB-IGO-RIDER.md). The Rider is purely procedural: it reserves all privileges and immunities enjoyed by the World Bank, without adding restrictions to the MIT permissions. Please review both files before using, distributing or contributing.
