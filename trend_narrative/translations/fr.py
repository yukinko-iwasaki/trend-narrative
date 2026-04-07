"""
French string catalog for narrative generation.
Catalogue de chaînes françaises pour la génération de récits.
"""

STRINGS: dict[str, object] = {
    # Time unit singular/plural forms, keyed by the English `time_unit` arg
    # passed to relationship narrative APIs. Unknown keys fall back to the
    # raw string unchanged.
    "time_units": {
        "year": ("année", "années"),
        "month": ("mois", "mois"),
        "quarter": ("trimestre", "trimestres"),
        "week": ("semaine", "semaines"),
        "day": ("jour", "jours"),
    },

    # Direction words
    "unknown": "inconnu",
    "remained_stable": "est resté stable",
    "increased": "a augmenté",
    "decreased": "a diminué",

    # Correlation strength labels
    "strength_no": "aucune",
    "strength_weak": "faible",
    "strength_moderate": "modérée",
    "strength_strong": "forte",
    "strength_very_strong": "très forte",

    # narrative.py — volatility fallbacks
    "vol_low": "{metric} est resté très stable et dans une fourchette étroite.",
    "vol_moderate": "{metric} a montré des fluctuations modérées autour d'une moyenne constante.",
    "vol_high": "{metric} a présenté une volatilité significative sans direction claire.",

    # narrative.py — single segment
    "single_segment": (
        "entre {start_year} et {end_year}, "
        "{metric} {direction} de {change} "
        "({pct_change}%), maintenant une trajectoire constante."
    ),

    # narrative.py — multi-segment
    # Trend phrase: complete noun phrase used with the "first segment" template.
    "trend_upward": "une tendance à la hausse",
    "trend_downward": "une tendance à la baisse",
    # Path phrase: adjective form used with "poursuivant sa trajectoire {…}".
    "path_upward": "ascendante",
    "path_downward": "descendante",
    "first_segment": (
        "De {start_year} à {end_year}, "
        "{metric} a affiché {trend_phrase}."
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
        "ne peut être déterminée car les données de {y} ne sont pas disponibles."
    ),
    "single_observation": (
        "{period}, {ref_name} {ref_dir} "
        "({ref_start} à {ref_end}), "
        "avec une seule observation de {comp_name} ({comp_start})"
    ),
    "stable_comparison": (
        "{period}, {ref_name} {ref_dir} ({ref_start} à {ref_end}) "
        "tandis que {comp_name} est resté stable ({comp_start})"
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
        " Avec des données limitées sur {comp_name}, "
        "une relation statistique ne peut être établie."
    ),

    # relationship_narrative.py — lagged correlation
    "timing_same": "au cours de la même {time_unit}",
    "timing_lagged": "environ {lag} {time_unit_pl} plus tard",
    "no_reliable_relationship": (
        "Aucune relation fiable n'a été détectée entre les variations de {x} "
        "et de {y}. "
    ),
    "weak_pattern": (
        "Bien que les données suggèrent une tendance {sign} {strength} "
        "(r={corr:.2f}), cela pourrait être dû au hasard "
        "compte tenu de la taille limitée de l'échantillon (n={n_pairs} paires, p={p_val:.2f})."
    ),
    "no_association": (
        "Les variations de l'un ne semblent pas être associées aux variations de l'autre, "
        "sur la base de {n_pairs} comparaisons d'{time_unit} en {time_unit}."
    ),
    "no_association_with_lag": (
        "Les variations de l'un ne semblent pas être associées aux variations de l'autre "
        "à tout décalage testé (0-{max_lag} {time_unit_pl}), "
        "sur la base de {n_pairs} comparaisons d'{time_unit} en {time_unit}."
    ),
    "significant_finding": (
        "Lorsque {leader} augmente, {follower} tend à "
        "{direction_word} {timing}. "
        "Il s'agit d'une relation {strength} (r={corr:.2f}) "
        "et statistiquement fiable (p={p_val:.3f}), "
        "sur la base de {n_pairs} comparaisons d'{time_unit} en {time_unit}."
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
