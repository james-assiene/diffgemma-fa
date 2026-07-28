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
)


# --------------------------------------------------------------------------
# BFCL's normalisation
# --------------------------------------------------------------------------

def test_normalise_matches_bfcls_own_rule():
    """SPEC §4.8: "scoring lowercases and strips `",./-_*^`". Scoring more
    strictly than the benchmark understates accuracy and is not comparable to
    the leaderboard."""
    assert normalise("Black") == normalise("black")
    assert normalise('"black"') == normalise("black")      # quote IS stripped
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
