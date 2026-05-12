"""
French string catalog for narrative generation.
Catalogue de chaînes françaises pour la génération de récits.

Inflected strings use ICU MessageFormat ``select`` syntax so that the
correct verb/participle form is chosen at runtime based on the metric's
grammatical number and gender::

    {number, select, singular {a augmenté} other {ont augmenté}}

Nesting is used when both number *and* gender matter (past participles
with *être*)::

    {number, select,
        singular {{gender, select, feminine {est restée stable} other {est resté stable}}}
        other    {{gender, select, feminine {sont restées stables} other {sont restés stables}}}}
"""

STRINGS: dict[str, object] = {
    # Number formatting. French uses ',' as decimal separator and inserts
    # a space before '%'. Suffix for 10^9 is "Md" (milliard), NOT "B" —
    # "B" / "billion" in French refers to 10^12 (faux ami with English).
    "number_format": {
        "decimal_sep": ",",
        "percent_template": "{value} %",
        "suffixes": ["", " k", " M", " Md", " Bn"],
    },

    # French has no one-size-fits-all plural rule (mois→mois, jour→jours,
    # œil→yeux, …), so unknown units are passed through unchanged rather
    # than guessed at — see _resolve_time_unit.
    "time_unit_fallback_plural_suffix": "",

    # Time unit singular/plural forms, keyed by the English `time_unit` arg
    # passed to relationship narrative APIs. Unknown keys are returned
    # unchanged (see ``time_unit_fallback_plural_suffix`` above).
    "time_units": {
        "year": ("année", "années"),
        "month": ("mois", "mois"),
        "quarter": ("trimestre", "trimestres"),
        "week": ("semaine", "semaines"),
        "day": ("jour", "jours"),
    },

    # Grammatical gender of each time unit (used by "timing_same" to pick
    # between "la même {unit}" (feminine) and "le même {unit}" (masculine),
    # and to contract "de" → "du" correctly).
    "time_unit_genders": {
        "year": "feminine",
        "month": "masculine",
        "quarter": "masculine",
        "week": "feminine",
        "day": "masculine",
    },

    # Direction words — ICU select on number (singular/plural).
    # For *avoir* verbs only number matters; for *être* verbs both
    # number and gender matter (nested select).
    "unknown": "inconnu",
    "remained_stable": (
        "{number, select, "
        "singular {{gender, select, feminine {est restée stable} other {est resté stable}}} "
        "other {{gender, select, feminine {sont restées stables} other {sont restés stables}}}}"
    ),
    "increased": "{number, select, singular {a augmenté} other {ont augmenté}}",
    "decreased": "{number, select, singular {a diminué} other {ont diminué}}",

    # Correlation strength labels
    "strength_no": "aucune",
    "strength_weak": "faible",
    "strength_moderate": "modérée",
    "strength_strong": "forte",
    "strength_very_strong": "très forte",

    # narrative.py — volatility fallbacks
    "vol_low": (
        "{number, select, "
        "singular {{gender, select, "
        "feminine {{metric} est restée très stable et dans une fourchette étroite.} "
        "other {{metric} est resté très stable et dans une fourchette étroite.}}} "
        "other {{gender, select, "
        "feminine {{metric} sont restées très stables et dans une fourchette étroite.} "
        "other {{metric} sont restés très stables et dans une fourchette étroite.}}}}"
    ),
    "vol_moderate": (
        "{number, select, "
        "singular {{metric} a montré des fluctuations modérées autour d'une moyenne constante.} "
        "other {{metric} ont montré des fluctuations modérées autour d'une moyenne constante.}}"
    ),
    "vol_high": (
        "{number, select, "
        "singular {{metric} a présenté une volatilité significative sans direction claire.} "
        "other {{metric} ont présenté une volatilité significative sans direction claire.}}"
    ),

    # narrative.py — single segment
    # {pct_change} is pre-formatted (sign, decimals, %) by _format_percent.
    "single_segment": (
        "entre {start_year} et {end_year}, "
        "{metric} {direction} de {change} "
        "({pct_change}), maintenant une trajectoire constante."
    ),

    # narrative.py — multi-segment
    "trend_upward": "une tendance à la hausse",
    "trend_downward": "une tendance à la baisse",
    "path_upward": "ascendante",
    "path_downward": "descendante",
    "first_segment": (
        "{number, select, "
        "singular {De {start_year} à {end_year}, {metric} a affiché {trend_phrase}.} "
        "other {De {start_year} à {end_year}, {metric} ont affiché {trend_phrase}.}}"
    ),
    "transition_prefixes": [
        "La tendance s'est ensuite inversée,",
        "Cette trajectoire a de nouveau pivoté,",
        "Puis,",
    ],
    "peak_reversal": (
        "atteignant un pic en {year} "
        "avant de s'inverser en déclin."
    ),
    "low_recovery": (
        "atteignant un creux en {year} "
        "suivi d'une reprise."
    ),
    "continuing": "poursuivant sa trajectoire {path_phrase} jusqu'en {year}.",

    # relationship_narrative.py — comovement
    "period_from_to": "de {start} à {end}",
    "unable_to_analyze": (
        "Impossible d'analyser la relation entre {x} et {y}."
    ),
    "no_data_available": (
        "La relation entre {x} et {y} "
        "ne peut être déterminée car les données {y_gen} ne sont pas disponibles."
    ),
    "single_observation": (
        "{period}, {ref_name} {ref_dir} "
        "({ref_start} à {ref_end}), "
        "avec une seule observation {comp_name_gen} ({comp_start})"
    ),
    "stable_comparison": (
        "{number, select, "
        "singular {{gender, select, "
        "feminine {{period}, {ref_name} {ref_dir} ({ref_start} à {ref_end}) tandis que {comp_name} est restée stable ({comp_start})} "
        "other {{period}, {ref_name} {ref_dir} ({ref_start} à {ref_end}) tandis que {comp_name} est resté stable ({comp_start})}}} "
        "other {{gender, select, "
        "feminine {{period}, {ref_name} {ref_dir} ({ref_start} à {ref_end}) tandis que {comp_name} sont restées stables ({comp_start})} "
        "other {{period}, {ref_name} {ref_dir} ({ref_start} à {ref_end}) tandis que {comp_name} sont restés stables ({comp_start})}}}}"
    ),
    "both_same_direction": "évoluant dans la même direction",
    "opposite_directions": "évoluant dans des directions opposées",
    "comovement_with_rel": (
        "{period}, {ref_name} {ref_dir} ({ref_start} à {ref_end}) "
        "tandis que {comp_name} {comp_dir} "
        "({comp_start} à {comp_end}), {relationship}"
    ),
    "comovement_no_rel": (
        "{period}, {ref_name} {ref_dir} ({ref_start} à {ref_end}) "
        "tandis que {comp_name} {comp_dir} "
        "({comp_start} à {comp_end})"
    ),
    "limited_data_caveat": (
        "Avec des données limitées sur {comp_name}, "
        "une relation statistique ne peut être établie."
    ),

    # relationship_narrative.py — lagged correlation
    "timing_same": (
        "{gender, select, "
        "feminine {au cours de la même {time_unit}} "
        "other {au cours du même {time_unit}}}"
    ),
    "timing_lagged": "environ {lag} {time_unit_pl} plus tard",
    "no_reliable_relationship": (
        "Aucune relation fiable n'a été détectée entre les variations {x_gen} "
        "et {y_gen}. "
    ),
    "weak_pattern": (
        "Bien que les données suggèrent une tendance {sign} {strength} "
        "(r={corr:.2f}), cela pourrait être dû au hasard "
        "compte tenu de la taille limitée de l'échantillon (n={n_pairs} paires, p={p_val:.2f})."
    ),
    "no_association": (
        "Les variations de l'un ne semblent pas être associées aux variations de l'autre, "
        "sur la base de {n_pairs} comparaisons {time_unit_comparison}."
    ),
    "no_association_with_lag": (
        "Les variations de l'un ne semblent pas être associées aux variations de l'autre "
        "à tout décalage testé (0-{max_lag} {time_unit_pl}), "
        "sur la base de {n_pairs} comparaisons {time_unit_comparison}."
    ),
    "significant_finding": (
        "Lorsque {leader} "
        "{leader_number, select, singular {augmente} other {augmentent}}, "
        "{follower} "
        "{follower_number, select, singular {tend} other {tendent}} à "
        "{direction_word} {timing}. "
        "Il s'agit d'une relation {strength} (r={corr:.2f}) "
        "et statistiquement fiable (p={p_val:.3f}), "
        "sur la base de {n_pairs} comparaisons {time_unit_comparison}."
    ),

    # relationship_narrative.py — insufficient data
    "insufficient_data": (
        "La relation entre {x} et {y} "
        "ne peut être déterminée en raison de données insuffisantes."
    ),

    # Positive / negative labels for correlation sign
    "positive": "positive",
    "negative": "négative",
    "increase": "augmenter",
    "decrease": "diminuer",
}
