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

from . import en, fr

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
# ``_unpack_metric`` normalizes both forms to a (name, grammar) pair so the
# internal narrative builders can work with a single representation.
# ---------------------------------------------------------------------------


def _unpack_metric(metric: object) -> tuple[str, dict[str, bool]]:
    """Normalize *metric* to ``(display_name, grammar_dict)``.

    Parameters
    ----------
    metric : str or dict
        Either a plain display string, or a dict shaped
        ``{"name": str, "plural": bool, "feminine": bool}``.
        Extra keys are ignored.

    Returns
    -------
    (name, grammar) : tuple[str, dict]
        ``name`` — the string to interpolate into narratives.
        ``grammar`` — ``{"plural": bool, "feminine": bool}`` with
        ``False`` defaults.

    Raises
    ------
    TypeError
        If *metric* is neither a str nor a dict, or if the dict is missing
        a ``name`` key.
    """
    if isinstance(metric, str):
        return metric, {"plural": False, "feminine": False}
    if isinstance(metric, dict):
        if "name" not in metric:
            raise TypeError(
                "metric dict must include a 'name' key, got keys: "
                f"{sorted(metric.keys())}"
            )
        return str(metric["name"]), {
            "plural": bool(metric.get("plural", False)),
            "feminine": bool(metric.get("feminine", False)),
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
