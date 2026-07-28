"""Tests for the schema pre-pass. SPEC §4.2.

Every case here is something `outlines_core` accepts and then silently
under-constrains. The whole point of the module is that these raise.
"""

from __future__ import annotations

import pytest

from diffgemma_fa.compile.schema import (
    BFCL_TYPE_MAP,
    UnsupportedSchemaError,
    check_supported,
    normalize_bfcl_schema,
)


# --------------------------------------------------------------------------
# BFCL dialect translation
# --------------------------------------------------------------------------

def test_bfcl_dict_becomes_object():
    """BFCL uses Python type names, not JSON Schema ones. Measured across
    BFCL v4: 9,464 `dict` and 0 `object`."""
    out = normalize_bfcl_schema({"type": "dict", "properties": {}})
    assert out["type"] == "object"


def test_bfcl_float_becomes_number():
    assert normalize_bfcl_schema({"type": "float"})["type"] == "number"


def test_bfcl_tuple_becomes_array():
    assert normalize_bfcl_schema({"type": "tuple"})["type"] == "array"


@pytest.mark.parametrize("java", ["String", "HashMap", "ArrayList", "long",
                                  "double", "Boolean", "Array", "char"])
def test_java_javascript_types_are_mapped(java: str):
    """These leak in from the simple_java / simple_javascript splits."""
    assert normalize_bfcl_schema({"type": java})["type"] == BFCL_TYPE_MAP[java]


def test_unknown_type_raises_rather_than_guessing():
    with pytest.raises(UnsupportedSchemaError, match="unrecognised BFCL type"):
        normalize_bfcl_schema({"type": "Frobnicator"})


def test_normalization_is_recursive():
    out = normalize_bfcl_schema({
        "type": "dict",
        "properties": {
            "a": {"type": "float"},
            "b": {"type": "array", "items": {"type": "dict",
                                             "properties": {"c": {"type": "long"}}}},
        },
    })
    assert out["properties"]["a"]["type"] == "number"
    assert out["properties"]["b"]["items"]["type"] == "object"
    assert out["properties"]["b"]["items"]["properties"]["c"]["type"] == "integer"


def test_default_and_optional_are_dropped():
    """Neither is a JSON Schema constraint; keeping them only grows the regex."""
    out = normalize_bfcl_schema({"type": "string", "default": "none",
                                 "optional": True, "description": "x"})
    assert out == {"type": "string"}


# --------------------------------------------------------------------------
# Fail-loud pre-pass
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kw,val", [
    ("minimum", 0), ("maximum", 10), ("exclusiveMinimum", 0),
    ("exclusiveMaximum", 10), ("multipleOf", 2),
])
def test_numeric_bounds_raise(kw: str, val):
    """SPEC §4.2: outlines ignores ALL numeric bounds, silently."""
    with pytest.raises(UnsupportedSchemaError, match=kw):
        check_supported({"type": "integer", kw: val})


@pytest.mark.parametrize("kw", ["patternProperties", "propertyNames",
                                "uniqueItems", "not", "if", "then", "else"])
def test_ignored_keywords_raise(kw: str):
    with pytest.raises(UnsupportedSchemaError, match=kw):
        check_supported({"type": "object", kw: {}})


@pytest.mark.parametrize("kw", ["minLength", "maxLength", "minItems", "maxItems"])
def test_length_keywords_are_enforced_and_must_not_raise(kw: str):
    """**SPEC §4.2 is wrong on these four.** It lists them among the silently
    dropped keywords; measured against outlines-core 0.2.14 all four are
    genuinely enforced — the generated regex rejects a too-short string and a
    too-long array.

    Raising on them would be worse than cosmetic: callers would have to
    `allow`-list constraints that actually work, desensitising the very signal
    this pre-pass exists to give for the bounds that really are dropped.
    """
    check_supported({"type": "object", kw: 1})


def test_min_length_really_forbids_the_empty_string():
    """The measurement behind `require_nonempty_strings` — and the reason it is
    the right lever for Phase 5's empty-argument collapse."""
    import re

    from diffgemma_fa.compile.schema import build_regex

    rx = build_regex({"type": "object", "required": ["s"],
                      "properties": {"s": {"type": "string", "minLength": 1}}},
                     whitespace_pattern="")
    assert re.fullmatch(rx, '{"s":"a"}')
    assert not re.fullmatch(rx, '{"s":""}')


def test_require_nonempty_strings_targets_only_required_free_strings():
    from diffgemma_fa.compile.schema import require_nonempty_strings

    out = require_nonempty_strings({
        "type": "object",
        "required": ["a", "c", "d"],
        "properties": {
            "a": {"type": "string"},                       # -> minLength 1
            "b": {"type": "string"},                       # optional, untouched
            "c": {"type": "string", "enum": ["x"]},        # enum, untouched
            "d": {"type": "integer"},                      # not a string
        },
    })
    assert out["properties"]["a"]["minLength"] == 1
    assert "minLength" not in out["properties"]["b"]
    assert "minLength" not in out["properties"]["c"]
    assert "minLength" not in out["properties"]["d"]


def test_additional_properties_false_raises():
    with pytest.raises(UnsupportedSchemaError, match="additionalProperties"):
        check_supported({"type": "object", "additionalProperties": False})


def test_first_match_wins_sibling_drop_raises():
    """`properties` beats `anyOf`; the `anyOf` is silently discarded."""
    with pytest.raises(UnsupportedSchemaError, match="first-match-wins"):
        check_supported({"properties": {"a": {"type": "string"}},
                         "anyOf": [{"type": "string"}]})


def test_type_alongside_properties_is_benign():
    """The ubiquitous case: `{"type": "object", "properties": {...}}` is fine
    and must not raise, or nothing would compile."""
    check_supported({"type": "object", "properties": {"a": {"type": "string"}}})


def test_allow_list_permits_a_known_loss():
    """Tolerating a dropped keyword must be explicit, and reportable."""
    check_supported({"type": "integer", "minimum": 0}, allow=["minimum"])


def test_wildcard_raises_by_default():
    """Phase 0 measured this shape at 16.7 s / 3,261 states / 36.2M
    transitions, versus 0.13-0.65 s for everything else."""
    with pytest.raises(UnsupportedSchemaError, match="wildcard"):
        check_supported({"type": "any"})


def test_wildcard_allowed_on_request():
    check_supported({"type": "any"}, allow_wildcard=True)


def test_additional_properties_true_is_a_wildcard():
    with pytest.raises(UnsupportedSchemaError, match="7-way alternation"):
        check_supported({"type": "object", "additionalProperties": True})


def test_nested_violations_are_found_and_located():
    with pytest.raises(UnsupportedSchemaError) as ei:
        check_supported({
            "type": "object",
            "properties": {"a": {"type": "object",
                                 "properties": {"b": {"type": "integer",
                                                      "minimum": 3}}}},
        })
    assert "minimum" in str(ei.value)
    assert "properties.a" in str(ei.value)


def test_violations_inside_array_items_are_found():
    with pytest.raises(UnsupportedSchemaError, match="uniqueItems"):
        check_supported({"type": "array", "items": {"type": "array",
                                                    "uniqueItems": True}})


def test_clean_schema_passes():
    check_supported({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["c", "f"]},
            "n": {"type": "integer"},
        },
        "required": ["city"],
    })


def test_bfcl_any_becomes_a_typeless_wildcard_not_a_literal():
    """BFCL's `any` has no JSON Schema spelling — the equivalent is *omitting*
    `type`. Passing the literal string to outlines raises
    `ValueError: Unsupported type: any`; measured on BFCL-Live that killed 11
    of 4,549 schemas."""
    out = normalize_bfcl_schema({"type": "any"})
    assert "type" not in out
    assert out.get("__wildcard__") is True


def test_any_still_refused_by_default_and_allowed_on_request():
    n = normalize_bfcl_schema({"type": "dict",
                               "properties": {"p": {"type": "any"}}})
    with pytest.raises(UnsupportedSchemaError, match="wildcard"):
        check_supported(n)
    check_supported(n, allow_wildcard=True)


def test_any_compiles_end_to_end_to_the_seven_way_alternation():
    from diffgemma_fa.compile.schema import build_regex

    rx = build_regex({"type": "dict", "required": ["p"],
                      "properties": {"p": {"type": "any"}}},
                     from_bfcl=True, allow_wildcard=True, whitespace_pattern="")
    assert len(rx) > 5000, "the wildcard should expand to the big alternation"
    assert "__wildcard__" not in rx, "internal marker must not leak into the regex"


def test_array_level_enum_is_pushed_down_into_items():
    """BFCL routinely writes an *element* enum at the array level:

        {"type": "array", "items": {"type": "string"}, "enum": ["view", ...]}

    Read literally that demands the whole array equal one of those strings —
    unsatisfiable — and outlines' precedence (`enum` above `type`) compiles
    exactly that, silently requiring `"metrics":"view"` where the reference
    answer is `"metrics":["view"]`. Measured on live_simple, this one pattern
    caused 21 of 23 remaining ground-truth rejections.
    """
    out = normalize_bfcl_schema({
        "type": "array",
        "items": {"type": "string"},
        "enum": ["view", "trust"],
    })
    assert out["type"] == "array"
    assert "enum" not in out
    assert out["items"]["enum"] == ["view", "trust"]
    assert out["items"]["type"] == "string"


def test_array_level_enum_does_not_clobber_an_existing_item_enum():
    out = normalize_bfcl_schema({
        "type": "array",
        "items": {"type": "string", "enum": ["a"]},
        "enum": ["b"],
    })
    assert out["items"]["enum"] == ["a"]


def test_scalar_enum_plus_type_stays_benign():
    """The common, harmless case must keep compiling."""
    check_supported({"type": "string", "enum": ["c", "f"]})
    check_supported({"type": "integer", "enum": [1, 2]})


def test_container_enum_plus_type_is_flagged():
    """After normalization nothing should reach here, but a hand-written schema
    with a container-level enum is genuinely ambiguous and must not pass."""
    with pytest.raises(UnsupportedSchemaError, match="first-match-wins"):
        check_supported({"type": "array", "enum": ["x"]})
