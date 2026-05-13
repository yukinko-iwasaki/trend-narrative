# trend-narrative

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://worldbank.github.io/trend-narrative/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

## Overview

The **trend-narrative** package is a standalone Python library that combines **piecewise-linear trend detection**, **relationship analysis**, and **multilingual narrative generation** for time-series data.

Given a time series — such as annual health spending or GDP figures — this package automatically identifies meaningful trends (e.g., "rising from 2010 to 2015, then declining") and produces a ready-to-use sentence describing them. It can also compare two time series and explain how they move together or apart over time.

Narratives can be generated in **English** and **French**, with an extensible architecture for adding more languages.

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
y = np.array([100, 110, 120, 130, 140, 150, 140, 130, 120, 110, 100, 90], dtype=float)

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
y = np.array([100, 110, 120, 130, 140, 150, 140, 130, 120, 110, 100, 90], dtype=float)

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

### Multilingual Support

All narrative functions accept a `lang` parameter. The default is `"en"` (English), so existing code works unchanged.

```python
# English — plain strings work for any metric
narrative = get_segment_narrative(extractor=extractor, metric="health spending")

# French — see "Grammatical agreement" below for non-trivial metrics
narrative = get_segment_narrative(
    extractor=extractor,
    metric={"name": "les dépenses de santé", "plural": True, "feminine": True},
    lang="fr",
)
# → "De 2010 à 2015, les dépenses de santé ont affiché une tendance à la hausse.
#    La tendance s'est ensuite inversée, atteignant un pic en 2015 avant de s'inverser en déclin."
```

Currently supported: `"en"` (English), `"fr"` (French).

#### Grammatical agreement (French)

French verbs and adjectives must agree with the metric's grammatical **number** (singular/plural) and **gender** (masculine/feminine). When the metric isn't singular masculine, pass it as a dict:

```python
{"name": "les dépenses",   "plural": True,  "feminine": True}   # plural feminine
{"name": "les taux",       "plural": True,  "feminine": False}  # plural masculine
{"name": "la production",  "plural": False, "feminine": True}   # singular feminine
{"name": "le taux",        "plural": False, "feminine": False}  # singular masculine
```

The `plural` / `feminine` keys default to `False`. A plain string (e.g. `metric="les dépenses"`) is accepted, but defaults to singular masculine — silently producing wrong agreement like `dépenses **a augmenté**` instead of `dépenses **ont augmenté**`. **The dict form is strongly recommended for any French metric that isn't singular masculine.**

The same applies to `reference_name` and `comparison_name` in `get_relationship_narrative`:

```python
import numpy as np
from trend_narrative import get_relationship_narrative

years = np.array([2010, 2012, 2014, 2016, 2018, 2020], dtype=float)
spending = np.array([100, 120, 140, 160, 180, 200], dtype=float)
inflation = np.array([2.0, 2.3, 2.7, 3.0, 3.4, 3.8], dtype=float)

result = get_relationship_narrative(
    reference_years=years, reference_values=spending,
    comparison_years=years, comparison_values=inflation,
    reference_name={"name": "les dépenses", "plural": True, "feminine": True},
    comparison_name={"name": "le taux d'inflation"},   # singular masculine defaults
    lang="fr",
)
```

Grammar flags on the dict are silently ignored for languages that don't need them (e.g. English), so the same call shape works across languages.

See [Adding a new language](#adding-a-new-language) below.

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
import numpy as np
from trend_narrative import analyze_relationship, get_relationship_narrative

years = np.array([2010, 2012, 2014, 2016, 2018, 2020], dtype=float)
spending = np.array([100, 120, 140, 160, 180, 200], dtype=float)
outcome = np.array([50, 55, 62, 70, 78, 85], dtype=float)

insights = analyze_relationship(
    reference_years=years,
    reference_values=spending,
    comparison_years=years,
    comparison_values=outcome,
)
# Store insights in a database, inspect programmatically, etc.
print(insights["method"])    # → "lagged_correlation"
print(insights["best_lag"])  # → {"lag": 1, "correlation": 0.85, "p_value": 0.15, "n_pairs": 4}
print(insights["n_points"])  # → 6

# Generate narrative later from stored insights — no re-analysis needed
result = get_relationship_narrative(
    insights=insights,
    reference_name="spending",
    comparison_name="outcome",
)
print(result["narrative"])
```

The function automatically chooses the analysis method based on data availability:
- **Lagged correlation**: >= 5 points, tests correlations at various lags
- **Comovement**: 3-4 points, describes directional movement within segments
- **Insufficient data**: < 3 points

---

## API Reference

### `get_segment_narrative(segments, cv_value, metric="expenditure", lang="en")`
### `get_segment_narrative(extractor, metric="expenditure", lang="en")`

Generates a narrative for a single time series. Accepts either
precomputed data or an `InsightExtractor` instance. Set `lang="fr"` for French.

`metric` accepts either a plain string or a dict with grammatical
properties: `{"name": str, "plural": bool, "feminine": bool}`. For French,
use the dict form when the metric isn't singular masculine — see
[Grammatical agreement](#grammatical-agreement-french).

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
    # Required for narrative — str or dict (see "Grammatical agreement")
    reference_name="",         # str | dict, display name for reference
    comparison_name="",        # str | dict, display name for comparison
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
    # Language
    lang="en",                 # "en" or "fr"
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

### `millify(n, lang="en")`

Formats large numbers with a human-readable suffix. The decimal separator
and magnitude suffixes come from the language catalog:

- `millify(1_500_000)` → `"1.50 M"`
- `millify(1_500_000, lang="fr")` → `"1,50 M"`
- `millify(3_000_000_000, lang="fr")` → `"3,00 Md"` (milliard — NOT `"B"`,
  which in French means 10¹², a false friend with English)

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
│   ├── narrative.py             # Segment narrative composition
│   ├── relationship_analysis.py # Relationship analysis (language-neutral)
│   ├── relationship_narrative.py # Relationship narrative composition
│   └── translations/            # All localization lives here
│       ├── __init__.py          # Catalog access, ICU engine, _unpack_metric,
│       │                        #   millify, _format_percent, _genitive,
│       │                        #   _resolve_time_unit, _time_unit_comparison
│       ├── en.py                # English catalog (data only)
│       └── fr.py                # French catalog (data only)
├── tests/
│   ├── test_detector.py
│   ├── test_extractor.py
│   ├── test_narrative.py
│   ├── test_relationship_analysis.py
│   ├── test_relationship_narrative.py
│   └── test_translations.py     # i18n primitives + catalog + integration
├── pyproject.toml
└── README.md
```

---

## Adding a New Language

To add a new language (e.g. Spanish):

1. **Copy** `trend_narrative/translations/en.py` to `trend_narrative/translations/es.py`.
2. **Translate** every value in the `STRINGS` dict. Take care with nested keys:
   - `number_format` (`decimal_sep`, `percent_template`, `suffixes`) — Spanish uses `,` for decimals, like French.
   - `time_units` and `time_unit_genders` — singular/plural pairs and grammatical genders for each unit.
   - `time_unit_fallback_plural_suffix` — what to append to unknown units when `count > 1` (English uses `"s"`; languages without a one-size-fits-all rule should use `""`).
3. **Register** the new module in `trend_narrative/translations/__init__.py`:

   ```python
   from . import en, fr, es  # add the new import

   _REGISTRY: dict[str, dict[str, object]] = {
       "en": en.STRINGS,
       "fr": fr.STRINGS,
       "es": es.STRINGS,  # add the new entry
   }
   ```

4. **Implement `_genitive_<lang>`** in `translations/__init__.py` if your language has article contractions or elision (Spanish: `de + el → del`, `de + la` stays). Dispatch in `_genitive`:

   ```python
   if lang == "es":
       return _genitive_es(name)
   ```

`SUPPORTED_LANGUAGES` updates automatically. A catalog-parity check fires at import time (`_assert_catalog_parity`) and raises `ImportError` if your catalog is missing any top-level keys present in English — so half-finished catalogs fail loud rather than producing wrong output in production.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## Contact

For questions, feedback, or collaboration enquiries, please reach out to [ysuzuki2@worldbank.org](mailto:ysuzuki2@worldbank.org) and [wlu4@worldbank.org](mailto:wlu4@worldbank.org).

## License

This project is licensed under the MIT License together with the [World Bank IGO Rider](WB-IGO-RIDER.md). The Rider is purely procedural: it reserves all privileges and immunities enjoyed by the World Bank, without adding restrictions to the MIT permissions. Please review both files before using, distributing or contributing.
