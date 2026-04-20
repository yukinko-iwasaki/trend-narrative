"""
English string catalog for narrative generation.
"""

STRINGS: dict[str, object] = {
    # Time unit singular/plural forms, keyed by the `time_unit` arg passed
    # to relationship narrative APIs. Unknown keys fall back to the raw
    # string + naive English "s" suffix.
    "time_units": {
        "year": ("year", "years"),
        "month": ("month", "months"),
        "quarter": ("quarter", "quarters"),
        "week": ("week", "weeks"),
        "day": ("day", "days"),
    },

    # Grammatical gender of time units. English has no grammatical gender,
    # but this key must exist to satisfy catalog parity; values are unused
    # by the English "timing_same" template (which has no gender select).
    "time_unit_genders": {
        "year": "neuter",
        "month": "neuter",
        "quarter": "neuter",
        "week": "neuter",
        "day": "neuter",
    },

    # Direction words (used by relationship_analysis + narratives)
    "unknown": "unknown",
    "remained_stable": "remained stable",
    "increased": "increased",
    "decreased": "decreased",

    # Correlation strength labels
    "strength_no": "no",
    "strength_weak": "weak",
    "strength_moderate": "moderate",
    "strength_strong": "strong",
    "strength_very_strong": "very strong",

    # narrative.py — volatility fallbacks (no segments)
    "vol_low": "{metric} remained highly stable and range-bound.",
    "vol_moderate": "{metric} showed moderate fluctuations around a consistent mean.",
    "vol_high": "{metric} exhibited significant volatility without a clear direction.",

    # narrative.py — single segment
    "single_segment": (
        "between {start_year} and {end_year}, "
        "the {metric} {direction} by {change} "
        "({pct_change}%), maintaining a consistent trajectory."
    ),

    # narrative.py — multi-segment
    # Trend phrase: complete noun phrase used with the "first segment" template.
    "trend_upward": "an upward trend",
    "trend_downward": "a downward trend",
    # Path phrase: bare adjective form used with "continuing its {…} path".
    "path_upward": "upward",
    "path_downward": "downward",
    "first_segment": (
        "From {start_year} to {end_year}, "
        "the {metric} showed {trend_phrase}."
    ),
    "transition_prefixes": [
        "Trend then shifted,",
        "This trajectory pivoted again,",
        "Then,",
    ],
    "peak_reversal": (
        "reaching a peak in {year} "
        "before reversing into a decline."
    ),
    "low_recovery": (
        "hitting a low in {year} "
        "followed by a recovery."
    ),
    "continuing": "continuing its {path_phrase} path through {year}.",

    # relationship_narrative.py — comovement
    "period_from_to": "from {start} to {end}",
    "unable_to_analyze": (
        "Unable to analyze relationship between {x} and {y}."
    ),
    "no_data_available": (
        "The relationship between {x} and {y} "
        "cannot be determined because {y} data is not available."
    ),
    "single_observation": (
        "{period}, {ref_name} {ref_dir} "
        "({ref_start} to {ref_end}), "
        "with only one {comp_name} observation ({comp_start})"
    ),
    "stable_comparison": (
        "{period}, {ref_name} {ref_dir} ({ref_start} to {ref_end}) "
        "while {comp_name} remained stable ({comp_start})"
    ),
    "both_same_direction": "both moving in the same direction",
    "opposite_directions": "moving in opposite directions",
    "comovement_with_rel": (
        "{period}, {ref_name} {ref_dir} ({ref_start} to {ref_end}) "
        "while {comp_name} {comp_dir} "
        "({comp_start} to {comp_end}), {relationship}"
    ),
    "comovement_no_rel": (
        "{period}, {ref_name} {ref_dir} ({ref_start} to {ref_end}) "
        "while {comp_name} {comp_dir} "
        "({comp_start} to {comp_end})"
    ),
    "limited_data_caveat": (
        " With limited {comp_name} data, "
        "a statistical relationship cannot be established."
    ),

    # relationship_narrative.py — lagged correlation
    "timing_same": "in the same {time_unit}",
    "timing_lagged": "about {lag} {time_unit_pl} later",
    "no_reliable_relationship": (
        "No reliable relationship was detected between changes in {x} "
        "and {y}. "
    ),
    "weak_pattern": (
        "While the data suggests a {strength} {sign} "
        "pattern (r={corr:.2f}), this could be due to chance "
        "given the limited sample size (n={n_pairs} change pairs, p={p_val:.2f})."
    ),
    "no_association": (
        "Changes in one do not appear to be associated with changes in the other, "
        "based on {n_pairs} {time_unit_comparison} comparisons."
    ),
    "no_association_with_lag": (
        "Changes in one do not appear to be associated with changes in the other "
        "at any lag tested (0-{max_lag} {time_unit_pl}), "
        "based on {n_pairs} {time_unit_comparison} comparisons."
    ),
    "significant_finding": (
        "When {leader} increases, {follower} tends to "
        "{direction_word} {timing}. "
        "This is a {strength} relationship (r={corr:.2f}) "
        "and is statistically reliable (p={p_val:.3f}), "
        "based on {n_pairs} {time_unit_comparison} comparisons."
    ),

    # relationship_narrative.py — insufficient data
    "insufficient_data": (
        "The relationship between {x} and {y} "
        "cannot be determined due to limited data availability."
    ),

    # Positive / negative labels for correlation sign
    "positive": "positive",
    "negative": "negative",
    "increase": "increase",
    "decrease": "decrease",
}
