"""
trend_narrative.translations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lightweight translation catalog for narrative generation.

Each language has its own module (``en.py``, ``fr.py``, …) exporting a
``STRINGS`` dict.  To add a new language:

1. Copy an existing file (e.g. ``en.py``) to ``<code>.py``.
2. Translate every value in the ``STRINGS`` dict.
3. Add the language code to ``_REGISTRY`` below.

That's it — ``get_translations("<code>")`` will just work, and the new
language will appear in ``SUPPORTED_LANGUAGES``.
"""

from __future__ import annotations

import math
from typing import Union

from . import en, fr

# Public type alias: anything accepted by `_unpack_metric` as a metric name.
MetricLike = Union[str, dict]

# English is the reference catalog. Every other language MUST have exactly
# the same set of keys — validated at import time (see _assert_catalog_parity
# below). This turns a "missing key crashes French users in production" bug
# into an ImportError that fires as soon as the package is loaded.
_REFERENCE_LANG = "en"

# Maps language codes to their string catalogs.
# To add a new language, import the module above and add it here.
_REGISTRY: dict[str, dict[str, object]] = {
    "en": en.STRINGS,
    "fr": fr.STRINGS,
}

SUPPORTED_LANGUAGES = tuple(_REGISTRY.keys())


def _assert_catalog_parity() -> None:
    """Verify every non-reference catalog has exactly the same keys as English.

    Runs at import time. Raises ``ImportError`` listing any missing or extra
    keys so catalog drift is caught in CI rather than at render time in
    production. Matches keys only — does not validate that format-string
    placeholders agree across languages (that would require parsing every
    template).
    """
    reference_keys = set(_REGISTRY[_REFERENCE_LANG].keys())
    problems: list[str] = []
    for lang, catalog in _REGISTRY.items():
        if lang == _REFERENCE_LANG:
            continue
        lang_keys = set(catalog.keys())
        missing = reference_keys - lang_keys
        extra = lang_keys - reference_keys
        if missing:
            problems.append(f"  {lang!r} missing keys: {sorted(missing)}")
        if extra:
            problems.append(f"  {lang!r} has extra keys not in {_REFERENCE_LANG!r}: {sorted(extra)}")
    if problems:
        raise ImportError(
            "Translation catalog drift detected:\n" + "\n".join(problems)
        )


_assert_catalog_parity()


# ---------------------------------------------------------------------------
# Metric unpacking
#
# Metric / series names can be passed to narrative functions as either:
#
# 1. A plain string (English and simple cases):
#        metric="real expenditure"
#
# 2. A dict bundling the display name with grammatical properties
#    (needed for French plural/gender agreement, ignored otherwise):
#        metric={"name": "les dépenses", "plural": True, "feminine": True}
#
# ``_unpack_metric`` normalizes both forms to ``(name, icu_kwargs)`` where
# icu_kwargs is ready to splat directly into ``icu_format(**kwargs)`` — the
# user-facing "plural"/"feminine" booleans are translated to ICU's
# "number"/"gender" select keys in one step.
# ---------------------------------------------------------------------------


def _unpack_metric(metric: object) -> tuple[str, dict[str, str]]:
    """Normalize *metric* to ``(display_name, icu_kwargs)``.

    Parameters
    ----------
    metric : str or dict
        Either a plain display string, or a dict shaped
        ``{"name": str, "plural": bool, "feminine": bool}``.
        Extra keys are ignored.

    Returns
    -------
    (name, icu_kwargs) : tuple[str, dict]
        ``name`` — the string to interpolate into narratives.
        ``icu_kwargs`` — ``{"number": "singular"|"plural",
        "gender": "masculine"|"feminine"}`` ready to splat into
        :func:`icu_format`. Plain strings yield singular/masculine defaults.

    Raises
    ------
    TypeError
        If *metric* is neither a str nor a dict, or if the dict is missing
        a ``name`` key.
    """
    if isinstance(metric, str):
        return metric, {"number": "singular", "gender": "masculine"}
    if isinstance(metric, dict):
        if "name" not in metric:
            raise TypeError(
                "metric dict must include a 'name' key, got keys: "
                f"{sorted(metric.keys())}"
            )
        return str(metric["name"]), {
            "number": "plural" if metric.get("plural") else "singular",
            "gender": "feminine" if metric.get("feminine") else "masculine",
        }
    raise TypeError(
        f"metric must be a str or dict, got {type(metric).__name__}"
    )


def _parse_select(template: str, pos: int, kwargs: dict) -> tuple[str, int]:
    """Parse a ``{var, select, case1 {text} ... other {text}}`` block.

    Returns the resolved text and the index after the closing ``}``.
    Supports nesting: case bodies may contain further ``{var, select, ...}``
    blocks.  Regular ``{var}`` placeholders (no ``, select,``) are left
    untouched so that Python ``.format()`` can fill them in later.
    """
    # Find the variable name and check for ", select,"
    comma = template.index(",", pos)
    var = template[pos:comma].strip()
    rest = template[comma + 1:].lstrip()

    if not rest.startswith("select,"):
        # Not a select — leave as a plain {var} placeholder for .format()
        # Find matching closing brace (skip nested braces)
        depth = 1
        i = pos
        while depth > 0:
            if template[i] == "{":
                depth += 1
            elif template[i] == "}":
                depth -= 1
            i += 1
        # Return the original text including braces so .format() can use it
        return template[pos - 1:i], i

    # Skip past "select," — account for whitespace between first comma and "select,"
    select_start = template.index("select,", comma)
    idx = select_start + len("select,")

    # Parse case_key {body} pairs
    cases: dict[str, str] = {}
    while idx < len(template):
        c = template[idx]
        if c in " \t\n":
            idx += 1
            continue
        if c == "}":
            # End of select block
            idx += 1
            break

        # Read case key (e.g. "singular", "plural", "other")
        key_start = idx
        while idx < len(template) and template[idx] not in " \t\n{":
            idx += 1
        case_key = template[key_start:idx]

        # Skip whitespace before body
        while idx < len(template) and template[idx] in " \t\n":
            idx += 1

        # Read body in braces, respecting nesting
        if template[idx] != "{":
            raise ValueError(
                f"ICU select: expected '{{' after case key '{case_key}' "
                f"at position {idx}"
            )
        idx += 1  # skip opening brace
        depth = 1
        body_start = idx
        while depth > 0:
            if template[idx] == "{":
                depth += 1
            elif template[idx] == "}":
                depth -= 1
            if depth > 0:
                idx += 1
        body = template[body_start:idx]
        idx += 1  # skip closing brace
        cases[case_key] = body

    # Pick the matching case
    value = str(kwargs.get(var, "other"))
    chosen = cases.get(value, cases.get("other", ""))

    # Recursively resolve any nested selects
    return _resolve_selects(chosen, kwargs), idx


def _resolve_selects(template: str, kwargs: dict) -> str:
    """Walk *template* and resolve all ``{var, select, ...}`` blocks.

    Plain ``{var}`` placeholders (no ``select``) are preserved for later
    ``.format()`` substitution.
    """
    result: list[str] = []
    i = 0
    while i < len(template):
        if template[i] == "{":
            # Peek ahead: is this {var, select, ...} or a plain {var}?
            # Find the first non-identifier character after '{'
            j = i + 1
            while j < len(template) and template[j] not in ",} \t\n":
                # Allow collon in format specs like {corr:.2f}
                if template[j] == ":":
                    break
                j += 1
            # Check if next non-space char is a comma (potential select)
            peek = j
            while peek < len(template) and template[peek] in " \t":
                peek += 1
            if peek < len(template) and template[peek] == ",":
                # Check if it's ", select,"
                after_comma = template[peek + 1:].lstrip()
                if after_comma.startswith("select"):
                    # It's a select block — parse it
                    resolved, end = _parse_select(template, i + 1, kwargs)
                    result.append(resolved)
                    i = end
                    continue
            # Plain placeholder — copy as-is including braces
            depth = 1
            k = i + 1
            while k < len(template) and depth > 0:
                if template[k] == "{":
                    depth += 1
                elif template[k] == "}":
                    depth -= 1
                k += 1
            result.append(template[i:k])
            i = k
        else:
            result.append(template[i])
            i += 1
    return "".join(result)


def icu_format(template: str, **kwargs: str) -> str:
    """Resolve ICU MessageFormat ``select`` blocks in *template*.

    Handles ``{variable, select, case1 {text} case2 {text} other {text}}``
    with arbitrary nesting.  Regular ``{variable}`` and ``{var:.2f}``
    placeholders are left intact for subsequent Python ``.format()`` calls.

    Parameters
    ----------
    template : str
        An ICU MessageFormat string (or a plain string with no select
        blocks — returned unchanged).
    **kwargs : str
        Values for select variables, e.g. ``number="singular"``,
        ``gender="feminine"``.

    Returns
    -------
    str
        The template with all ``select`` blocks resolved.

    Examples
    --------
    >>> icu_format("{number, select, singular {a augmenté} other {ont augmenté}}", number="plural")
    'ont augmenté'
    >>> icu_format("plain {metric} string", number="plural")
    'plain {metric} string'
    """
    if "select" not in template:
        return template
    return _resolve_selects(template, kwargs)


def get_translations(lang: str = "en") -> dict[str, object]:
    """Return the string catalog for the requested language.

    Parameters
    ----------
    lang : str
        ISO 639-1 language code (default ``"en"``).

    Raises
    ------
    ValueError
        If *lang* is not a supported language code.
    """
    try:
        return _REGISTRY[lang]
    except KeyError:
        raise ValueError(
            f"Unsupported language '{lang}'. "
            f"Supported: {', '.join(SUPPORTED_LANGUAGES)}"
        )


# ---------------------------------------------------------------------------
# Language-specific text helpers
#
# These produce inflected/contracted forms that depend on the target
# language: time-unit pluralization, "of X" (genitive), and "X-over-X"
# comparison phrasing.
# ---------------------------------------------------------------------------


def _resolve_time_unit(t: dict, time_unit: str, count: int) -> str:
    """Return the singular or plural form of ``time_unit`` in the target language.

    Looks up ``time_unit`` in the catalog's ``time_units`` table (e.g.
    ``"year" -> ("year", "years")`` or ``("année", "années")``) and returns
    the form matching ``count``. For unknown keys, appends the catalog's
    ``time_unit_fallback_plural_suffix`` when ``count > 1`` — English uses
    ``"s"`` (so ``"fortnight" → "fortnights"``); French uses ``""`` (passes
    through unchanged, since French has no general plural rule). This avoids
    silently producing wrong-language plurals like ``"fortnights"`` inside
    French prose.
    """
    units = t.get("time_units", {})
    entry = units.get(time_unit)
    if entry is None:
        if count == 1:
            return time_unit
        suffix = t.get("time_unit_fallback_plural_suffix", "")
        return f"{time_unit}{suffix}"
    singular, plural = entry
    return singular if count == 1 else plural


_FRENCH_VOWELS = frozenset("aeiouyàâéèêëïîôùûüœæ")


def _genitive_fr(name: str) -> str:
    """French genitive: prefix *name* with the right form of "de".

    Handles article contractions and vowel elision:

    * ``de + le`` → ``du`` (e.g. "du taux")
    * ``de + les`` → ``des`` (e.g. "des dépenses")
    * ``de + la`` → ``de la`` (no contraction; feminine article stays)
    * ``de + l'`` → ``de l'`` (no contraction)
    * ``de`` + consonant-initial bare noun → ``de …``
    * ``de`` + vowel-initial bare noun → ``d'…`` (elision)
    """
    if not name:
        return name
    # Articles with a trailing space come first (more specific prefixes)
    if name.startswith("les "):
        return "des " + name[4:]
    if name.startswith("le "):
        return "du " + name[3:]
    if name.startswith(("la ", "l'")):
        return "de " + name
    first = name[0].lower()
    if first in _FRENCH_VOWELS:
        return "d'" + name
    return "de " + name


def _genitive(lang: str, name: str) -> str:
    """Return the genitive ("of X") form of *name* in the given language.

    The genitive role expresses attribution / origin ("of X", "X's").
    Every language produces this differently:

    * English: ``of {name}``
    * French: ``de/du/des/d' {name}`` (article contraction + vowel elision)
    * Italian: ``di/del/della/...`` (contraction like French; not yet implemented)
    * Spanish: ``de/del`` (``de + el → del``; not yet implemented)

    Add a new language by implementing ``_genitive_<lang>(name)`` and
    dispatching here.  Unknown languages return *name* unchanged.
    """
    if not name:
        return name
    if lang == "fr":
        return _genitive_fr(name)
    if lang == "en":
        return "of " + name
    return name


def _time_unit_comparison(lang: str, time_unit_sg: str) -> str:
    """Build the 'year-over-year' / 'd'année en année' phrase.

    Handles French elision: ``d'année en année`` (vowel) vs
    ``de mois en mois`` (consonant).
    """
    if lang == "fr":
        first_char = time_unit_sg[0].lower() if time_unit_sg else ""
        prep = "d'" if first_char in _FRENCH_VOWELS else "de "
        return f"{prep}{time_unit_sg} en {time_unit_sg}"
    return f"{time_unit_sg}-over-{time_unit_sg}"


# ---------------------------------------------------------------------------
# Number formatting
#
# Decimal separator and magnitude suffixes come from the catalog's
# ``number_format`` entry, so e.g. 10^9 renders as " B" in English and
# " Md" (milliard) in French — "B" in French would mean 10^12.
# ---------------------------------------------------------------------------


def millify(n: float, lang: str = "en") -> str:
    """Format a large number into a human-readable string with suffix.

    The decimal separator and magnitude suffixes come from the language
    catalog. French uses "," and "Md" for milliard (10^9), since "B" /
    "billion" in French refers to 10^12 (false friend with English).

    Examples
    --------
    >>> millify(1_500_000)
    '1.50 M'
    >>> millify(1_500_000, lang="fr")
    '1,50 M'
    >>> millify(3_000_000_000, lang="fr")
    '3,00 Md'
    """
    fmt = get_translations(lang)["number_format"]
    suffixes = fmt["suffixes"]
    decimal_sep = fmt["decimal_sep"]

    n = float(n)
    idx = max(
        0,
        min(
            len(suffixes) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
        ),
    )
    value_str = f"{n / 10 ** (3 * idx):.2f}"
    if decimal_sep != ".":
        value_str = value_str.replace(".", decimal_sep)
    return f"{value_str}{suffixes[idx]}"


def _format_percent(value: float, lang: str = "en") -> str:
    """Format a percent value with localized decimal separator and spacing.

    Includes the sign (``+``/``-``) and the ``%`` symbol. French uses ","
    as decimal separator and a space before ``%``.
    """
    fmt = get_translations(lang)["number_format"]
    s = f"{value:+.2f}"
    if fmt["decimal_sep"] != ".":
        s = s.replace(".", fmt["decimal_sep"])
    return fmt["percent_template"].format(value=s)
