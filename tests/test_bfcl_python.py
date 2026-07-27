"""Tests for the authored BFCL Python grammar. SPEC §4.8.

The load-bearing property is that **positional arguments are impossible**.
BFCL's scorer parses with `ast` and reads only `elem.keywords`, so a positional
argument is silently discarded and guarantees a `missing_required` failure —
a grammar that permits one is worse than useless, because the output still
looks valid.
"""

from __future__ import annotations

import ast
import re

import pytest

from diffgemma_fa.compile.tasks.bfcl_python import (
    PythonGrammarConfig,
    build_call_regex,
    build_calls_regex,
    measure_depth,
    value_regex,
)

WEATHER = {
    "name": "get_weather",
    "parameters": {
        "type": "dict",
        "required": ["city"],
        "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            "days": {"type": "integer"},
        },
    },
}


def full(rx: str, s: str) -> bool:
    return re.fullmatch(rx, s) is not None


# --------------------------------------------------------------------------
# The scorer-driven constraint
# --------------------------------------------------------------------------

def test_keyword_arguments_are_accepted():
    rx = build_call_regex(WEATHER)
    assert full(rx, 'get_weather(city="Paris")')
    assert full(rx, 'get_weather(city="Paris", unit="celsius")')


def test_positional_arguments_are_rejected():
    """The whole point (SPEC §4.8)."""
    rx = build_call_regex(WEATHER)
    assert not full(rx, 'get_weather("Paris")')
    assert not full(rx, 'get_weather("Paris", "celsius")')


def test_accepted_calls_parse_as_keyword_only_under_ast():
    """Cross-check against the actual scoring mechanism rather than our regex:
    every string the grammar accepts must expose all its arguments in
    `elem.keywords` and none in `elem.args`."""
    rx = build_call_regex(WEATHER)
    for s in ['get_weather(city="Paris")',
              'get_weather(city="Paris", unit="celsius")',
              'get_weather(city="Paris", unit="fahrenheit", days=3)']:
        assert full(rx, s)
        node = ast.parse(s, mode="eval").body
        assert isinstance(node, ast.Call)
        assert node.args == [], "positional args would be discarded by BFCL"
        assert {k.arg for k in node.keywords}


def test_required_argument_cannot_be_omitted():
    rx = build_call_regex(WEATHER)
    assert not full(rx, "get_weather()")
    assert not full(rx, 'get_weather(unit="celsius")')


def test_optional_arguments_may_be_omitted():
    rx = build_call_regex(WEATHER)
    assert full(rx, 'get_weather(city="Paris")')
    assert full(rx, 'get_weather(city="Paris", days=2)')


def test_enum_values_are_constrained():
    rx = build_call_regex(WEATHER)
    assert full(rx, 'get_weather(city="X", unit="celsius")')
    assert not full(rx, 'get_weather(city="X", unit="kelvin")')


def test_wrong_function_name_rejected():
    rx = build_call_regex(WEATHER)
    assert not full(rx, 'get_forecast(city="Paris")')


def test_integer_argument_rejects_a_string():
    rx = build_call_regex(WEATHER)
    assert full(rx, 'get_weather(city="X", days=5)')
    assert not full(rx, 'get_weather(city="X", days="5")')


# --------------------------------------------------------------------------
# Value grammar and the depth bound
# --------------------------------------------------------------------------

def test_scalar_values():
    assert full(value_regex({"type": "integer"}, depth=0, max_depth=2), "-42")
    assert full(value_regex({"type": "number"}, depth=0, max_depth=2), "3.5e-2")
    assert full(value_regex({"type": "boolean"}, depth=0, max_depth=2), "True")
    assert full(value_regex({"type": "null"}, depth=0, max_depth=2), "None")


def test_python_booleans_not_json_ones():
    """Python format, so `True`/`False`/`None`, never `true`/`false`/`null`."""
    rx = value_regex({"type": "boolean"}, depth=0, max_depth=2)
    assert full(rx, "True") and not full(rx, "true")


def test_array_of_strings():
    rx = value_regex({"type": "array", "items": {"type": "string"}},
                     depth=0, max_depth=2)
    assert full(rx, '["a", "b"]')
    assert full(rx, "[]")
    assert not full(rx, "[1]")


def test_depth_bound_collapses_deeper_containers():
    """SPEC §4.8: bound the nesting, and report what the bound costs."""
    schema = {"type": "array", "items": {"type": "array",
                                         "items": {"type": "integer"}}}
    rx = value_regex(schema, depth=0, max_depth=1)
    assert full(rx, "[[]]")
    assert not full(rx, "[[1]]"), "depth 2 must be outside a max_depth=1 grammar"

    rx2 = value_regex(schema, depth=0, max_depth=2)
    assert full(rx2, "[[1, 2]]")


def test_measure_depth():
    assert measure_depth(5) == 0
    assert measure_depth([1, 2]) == 1
    assert measure_depth([[1]]) == 2
    assert measure_depth({"a": [1]}) == 2
    assert measure_depth([]) == 1


# --------------------------------------------------------------------------
# Call lists
# --------------------------------------------------------------------------

def test_bracketed_and_bare_calls_both_accepted():
    """BFCL auto-adds brackets when missing."""
    rx = build_calls_regex([WEATHER])
    assert full(rx, '[get_weather(city="Paris")]')
    assert full(rx, 'get_weather(city="Paris")')


def test_parallel_calls():
    rx = build_calls_regex([WEATHER])
    assert full(rx, '[get_weather(city="Paris"), get_weather(city="Rome")]')


def test_max_calls_is_enforced():
    rx = build_calls_regex([WEATHER], max_calls=2)
    two = '[get_weather(city="A"), get_weather(city="B")]'
    three = '[get_weather(city="A"), get_weather(city="B"), get_weather(city="C")]'
    assert full(rx, two)
    assert not full(rx, three)


def test_multiple_candidate_functions():
    """A `multiple` record offers several functions; the grammar is their
    union. BFCL v4 records carry 1-8+ functions."""
    other = {"name": "get_time",
             "parameters": {"type": "dict", "required": ["tz"],
                            "properties": {"tz": {"type": "string"}}}}
    rx = build_calls_regex([WEATHER, other])
    assert full(rx, 'get_weather(city="Paris")')
    assert full(rx, 'get_time(tz="UTC")')
    assert not full(rx, 'get_date(tz="UTC")')


def test_no_parameters_function():
    fn = {"name": "ping", "parameters": {"type": "dict", "properties": {}}}
    assert full(build_call_regex(fn), "ping()")


def test_dotted_names_are_escaped_not_wildcards():
    fn = {"name": "mod.sub.f",
          "parameters": {"type": "dict", "required": ["x"],
                         "properties": {"x": {"type": "integer"}}}}
    rx = build_call_regex(fn)
    assert full(rx, "mod.sub.f(x=1)")
    assert not full(rx, "modXsubXf(x=1)"), "the dot must be literal"


@pytest.mark.parametrize("space", [True, False])
def test_space_after_comma_config(space: bool):
    cfg = PythonGrammarConfig(space_after_comma=space)
    rx = build_call_regex(WEATHER, config=cfg)
    assert full(rx, 'get_weather(city="X", days=1)') is True
    assert full(rx, 'get_weather(city="X",days=1)') is (not space)
