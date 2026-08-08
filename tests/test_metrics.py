"""Tests for the eval scoring. SPEC §7.2, §3.8, §4.8.

A wrong metric is worse than no metric — it produces a table nobody questions.
Two mistakes already made once in this project are pinned here: scoring more
strictly than BFCL does, and failing to parse a channel-tagged emission.
"""

from __future__ import annotations

from diffgemma_fa.eval.metrics import (
    Scores,
    extract_json,
    normalise,
    score_arguments,
    schema_valid,
)


# --------------------------------------------------------------------------
# BFCL's normalisation
# --------------------------------------------------------------------------

def test_normalise_matches_bfcls_own_rule():
    r"""BFCL `ast_checker.py::standardize_string`:

        regex_string = r"[ \,\.\/\-\_\*\^]"
        return re.sub(regex_string, "", input_string).lower().replace("'", '"')

    The set is SPACE plus `,./-_*^`. SPEC §4.8's `",./-_*^` is a quotation of
    BFCL's docstring; the outer `"` is the delimiter, not a member.

    **[AUDIT-A] This test previously asserted the opposite on both counts** —
    that `"` is stripped and the space is not — and so blessed a rule that was
    simultaneously more lenient than the leaderboard (a literal `"black"`
    scored as `black`) and stricter than it (`NewYork` scored wrong against a
    ground truth of `New York`). See
    `tests/test_audit_measurement.py::test_normalise_agrees_with_bfcls_own_standardize_string`,
    which differential-tests against the vendored BFCL source.
    """
    assert normalise("Black") == normalise("black")
    assert normalise('"black"') != normalise("black")     # `"` is NOT stripped
    assert normalise("New York") == normalise("newyork")  # space IS stripped
    assert normalise("it's") == normalise('it"s')         # `'` folds to `"`
    assert normalise("Divinópolis, MG") == normalise("divinópolis mg")
    assert normalise(600) == "600"


def test_colon_is_NOT_in_bfcls_strip_set():
    """A correction to an earlier claim of mine.

    BFCL strips `",./-_*^` — **`:` is not in that set**. I previously asserted
    that a predicted `":black"` against a ground truth `black` would be a BFCL
    match and used that to argue strict equality was understating accuracy. It
    is not a match, and switching to BFCL's normalisation changed no number
    (1/19 before and after). The leading-`:` artifact had to be *fixed* at the
    grammar level, not normalised away.
    """
    assert normalise(":black") != normalise("black")


def test_normalise_does_not_collapse_genuinely_different_values():
    assert normalise("7890") != normalise("77890")
    assert normalise("comfort") != normalise("plus")


# --------------------------------------------------------------------------
# Extraction from a channel-tagged emission
# --------------------------------------------------------------------------

def test_extract_json_after_a_channel_header():
    """The grammar carries SPEC §3.6's header, so the emission legitimately
    *begins* with `<|channel>…<channel|>`. Splitting on `<` — which an earlier
    version did — yields the empty string and scores everything unparsable."""
    text = '<|channel>thought\n<channel|>{"user_id": 7890, "special": "black"}'
    assert extract_json(text) == {"user_id": 7890, "special": "black"}


def test_extract_json_handles_a_markdown_fence_name():
    text = '<|channel>```json_{\n<channel|>{"a": 1}<turn|>'
    assert extract_json(text) == {"a": 1}


def test_extract_json_stops_at_the_matching_brace():
    text = '<channel|>{"a": 1} trailing {"b": 2}'
    assert extract_json(text) == {"a": 1}


def test_extract_json_respects_braces_inside_strings():
    text = '<channel|>{"a": "}{"}'
    assert extract_json(text) == {"a": "}{"}


def test_extract_json_respects_escaped_quotes():
    text = r'<channel|>{"a": "say \"hi\""}'
    assert extract_json(text) == {"a": 'say "hi"'}


def test_extract_json_returns_none_on_garbage():
    assert extract_json("no json here") is None
    assert extract_json('<channel|>{"unclosed": ') is None


def test_extract_json_rejects_a_non_object():
    assert extract_json("<channel|>[1, 2, 3]") is None


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def test_score_arguments_counts_per_key():
    ok, tot, det = score_arguments({"a": "X", "b": 2}, {"a": "x", "b": 3})
    assert (ok, tot) == (1, 2)
    assert [d["ok"] for d in det] == [True, False]


def test_missing_key_is_a_miss_not_an_error():
    ok, tot, _ = score_arguments({}, {"a": "x"})
    assert (ok, tot) == (0, 1)


def test_extra_predicted_keys_do_not_count():
    ok, tot, _ = score_arguments({"a": "x", "z": "junk"}, {"a": "x"})
    assert (ok, tot) == (1, 1)


# --------------------------------------------------------------------------
# The reason CS is never reported alone (SPEC §3.8)
# --------------------------------------------------------------------------

def test_cs_can_be_perfect_while_content_is_empty():
    """SPEC §3.8, sharpened by Phase 4: a run emitting `{"location":""}` every
    time scores CS = 100% and is useless. The content columns are what expose
    it, which is why `Scores` reports them together."""
    s = Scores()
    for _ in range(10):
        s.add(accepted=True, parsed_obj={"location": ""},
              want={"location": "Tel Aviv, Israel"})
    d = s.as_dict()
    assert d["cs_rate"] == 1.0, "constraint satisfaction is genuinely perfect"
    assert d["nonempty_rate"] == 0.0, "and the output is entirely empty"
    assert d["arg_accuracy"] == 0.0


def test_exact_call_rate_requires_every_argument():
    s = Scores()
    s.add(accepted=True, parsed_obj={"a": "x", "b": "y"},
          want={"a": "x", "b": "y"})
    s.add(accepted=True, parsed_obj={"a": "x", "b": "WRONG"},
          want={"a": "x", "b": "y"})
    d = s.as_dict()
    assert d["exact_calls"] == 1
    assert d["arg_correct"] == 3 and d["arg_total"] == 4


def test_unparsable_output_counts_against_n():
    s = Scores()
    s.add(accepted=False, parsed_obj=None, want={"a": "x"})
    d = s.as_dict()
    assert d["n"] == 1 and d["parsed"] == 0 and d["cs"] == 0


# --------------------------------------------------------------------------
# Schema validity — the header-independent cross-arm column
# --------------------------------------------------------------------------

def test_schema_validity_is_independent_of_our_channel_header():
    """`cs_rate` asks whether the emitted **token sequence** is accepted by the
    compiled automaton, and that automaton carries SPEC §3.6's channel header —
    which is *our* addition. The stock model has no reason to emit it, so
    `cs_rate` scores the `unconstrained` and `mask` baselines at ~0 partly for a
    convention they were never asked to follow. A 0% -> 100% CS jump reported on
    that basis alone would overstate the result.

    `schema_valid` asks the question the benchmark cares about, in a form every
    arm can be asked fairly.
    """
    schema = {"type": "object",
              "properties": {"city": {"type": "string"}},
              "required": ["city"]}
    with_header = '<|channel>thought\n<channel|>{"city": "Paris"}'
    without = '{"city": "Paris"}'
    assert schema_valid(extract_json(with_header), schema)
    assert schema_valid(extract_json(without), schema)


def test_schema_validity_actually_rejects():
    schema = {"type": "object",
              "properties": {"n": {"type": "integer"}},
              "required": ["n"]}
    assert not schema_valid({"n": "not an integer"}, schema)
    assert not schema_valid({}, schema), "a missing required key must fail"
    assert not schema_valid(None, schema), "unparsable output must fail"


def test_schema_validity_understands_the_normalised_dialect():
    """BFCL writes `dict`/`float`/`any`; no JSON Schema validator understands
    those, which is why this consumes the schema **after**
    `normalize_bfcl_schema`."""
    from diffgemma_fa.compile.schema import normalize_bfcl_schema
    norm = normalize_bfcl_schema(
        {"type": "dict", "properties": {"x": {"type": "float"}},
         "required": ["x"]})
    assert schema_valid({"x": 1.5}, norm)
    assert not schema_valid({"x": "a"}, norm)


def test_cs_and_schema_validity_are_reported_together():
    """Neither substitutes for the other, so `Scores` carries both."""
    s = Scores()
    s.add(accepted=False, parsed_obj={"a": "x"}, want={"a": "x"},
          schema_ok=True)
    d = s.as_dict()
    assert d["cs_rate"] == 0.0 and d["schema_valid_rate"] == 1.0
