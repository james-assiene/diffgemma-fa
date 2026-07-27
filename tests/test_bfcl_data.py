"""Tests for BFCL record loading and ground-truth materialisation.

`possible_answer` is deceptively shaped, and getting it wrong makes the
*grammar* look over-constrained when the fault is in the harness. Both mistakes
below were made during Phase 1 and each cost several points of apparent
ground-truth acceptance.
"""

from __future__ import annotations

from diffgemma_fa.compile.bfcl_data import materialize_ground_truth

SCHEMA = {
    "type": "dict",
    "required": ["command"],
    "properties": {
        "command": {"type": "string"},
        "unit": {"type": "string", "enum": ["seconds", "milliseconds"]},
        "tags": {"type": "array", "items": {"type": "string"}},
        "body": {
            "type": "dict",
            "properties": {
                "fabric": {"type": "string"},
                "group": {"type": "string"},
            },
        },
    },
}


def test_unwraps_the_acceptable_values_list():
    out = materialize_ground_truth({"command": ["docker ps"]}, SCHEMA)
    assert out == {"command": "docker ps"}


def test_empty_string_means_omitted_and_omission_wins():
    """`{"unit": ["", "N/A"]}` reads as "omitted, or N/A". With an enum of
    [seconds, milliseconds], omission is the ONLY valid reading — taking the
    second entry produced 12 of 24 apparent grammar rejections on live_simple.
    """
    out = materialize_ground_truth(
        {"command": ["docker ps"], "unit": ["", "N/A"]}, SCHEMA)
    assert out == {"command": "docker ps"}
    assert "unit" not in out


def test_empty_string_wins_even_when_listed_second():
    out = materialize_ground_truth(
        {"command": ["x"], "unit": ["seconds", ""]}, SCHEMA)
    assert "unit" not in out


def test_null_is_omission_not_a_value():
    out = materialize_ground_truth({"command": ["x"], "unit": [None]}, SCHEMA)
    assert out == {"command": "x"}


def test_nested_dicts_are_unwrapped_recursively():
    """The wrapper recurs at EVERY level. Unwrapping only the top leaves
    `["network222"]` as the value of a field declared `string`."""
    out = materialize_ground_truth(
        {"command": ["x"],
         "body": [{"fabric": ["network222"], "group": ["", "default"]}]},
        SCHEMA)
    assert out == {"command": "x", "body": {"fabric": "network222"}}


def test_genuine_arrays_keep_their_list():
    out = materialize_ground_truth({"command": ["x"], "tags": [["a", "b"]]},
                                   SCHEMA)
    assert out == {"command": "x", "tags": ["a", "b"]}


def test_null_elements_are_dropped_from_arrays():
    out = materialize_ground_truth({"command": ["x"], "tags": [["a", None]]},
                                   SCHEMA)
    assert out == {"command": "x", "tags": ["a"]}


def test_flat_array_without_double_wrapping():
    out = materialize_ground_truth({"command": ["x"], "tags": ["a"]}, SCHEMA)
    assert out["tags"] == ["a"]


def test_missing_schema_still_works():
    out = materialize_ground_truth({"a": ["v"]}, None)
    assert out == {"a": "v"}
