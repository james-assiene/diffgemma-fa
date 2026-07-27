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
                                "uniqueItems", "not", "if", "then", "else",
                                "minItems", "maxItems", "minLength", "maxLength"])
def test_ignored_keywords_raise(kw: str):
    with pytest.raises(UnsupportedSchemaError, match=kw):
        check_supported({"type": "object", kw: {}})


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
    with pytest.raises(UnsupportedSchemaError, match="maxItems"):
        check_supported({"type": "array", "items": {"type": "array",
                                                    "maxItems": 3}})


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
