"""Tests for the compiled automaton artifact. SPEC §3.1b, §3.5, §3.6, §5.5.

The two constructions here are the ones CLAUDE.md says cost a day each: the
unscored post-stop tail, and a `d(s)` computed at the wrong point in the
pipeline.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from diffgemma_fa.compile import automaton
from diffgemma_fa.compile.automaton import (
    INF_DISTANCE,
    augment_with_stop_tokens,
    bucket_size,
    compile_automaton,
    distance_to_final,
)
from diffgemma_fa.compile.minimize import Dfa

V = 32
END_TOKENS = (1, 106, 50)


def line_dfa(k: int) -> Dfa:
    """A chain accepting exactly one word of length k over label 7."""
    return Dfa(
        n_states=k + 1,
        transitions=tuple((i, 7, i + 1) for i in range(k)),
        start=0,
        finals=frozenset({k}),
    )


# --------------------------------------------------------------------------
# Stop-token augmentation
# --------------------------------------------------------------------------

def test_all_end_tokens_are_wired():
    """SPEC §3.5 trap 3: handling only EOS is the most likely cause of an empty
    state set at a block boundary."""
    aug, acc, _ = augment_with_stop_tokens(line_dfa(2), end_tokens=END_TOKENS,
                                           vocab_size=V)
    for t in END_TOKENS:
        assert (2, t, acc) in aug.transitions


def test_empty_end_tokens_raises():
    with pytest.raises(ValueError, match="end_tokens is empty"):
        augment_with_stop_tokens(line_dfa(1), end_tokens=(), vocab_size=V)


def test_post_stop_tail_is_unscored_full_sigma():
    """SPEC §3.5 trap 4. The tail must accept EVERY token, not just PAD.

    If it were PAD-only, a joint decode would pay (255-j)*log p(PAD) to
    terminate at position j and would place the stop token at 255 or never.
    """
    aug, acc, _ = augment_with_stop_tokens(line_dfa(1), end_tokens=(1,), vocab_size=V)
    out = {lbl for s, lbl, d in aug.transitions if s == acc and d == acc}
    assert out == set(range(V)), "ACC --Sigma--> ACC must span the whole vocab"


def test_acc_is_the_only_accepting_state():
    aug, acc, _ = augment_with_stop_tokens(line_dfa(3), end_tokens=(1,), vocab_size=V)
    assert aug.finals == frozenset({acc})


def test_unscored_tail_has_unit_emission_mass():
    """The point of Sigma-labelling: sum_{v in label} p_i(v) == 1, i.e. free."""
    a = compile_automaton(line_dfa(2), name="t", end_tokens=(1,), vocab_size=V)
    rng = np.random.default_rng(0)
    p = rng.random(V)
    p /= p.sum()

    acc = 3  # states 0,1,2 from line_dfa(2) plus ACC
    e = [i for i in range(a.n_edges)
         if a.edge_src[i] == acc and a.edge_dst[i] == acc]
    assert len(e) == 1
    c = a.edge_class[e[0]]
    t = a.tables
    stored = t.sum_indices[t.sum_indptr[c]:t.sum_indptr[c + 1]]
    w = p.sum() - p[stored].sum() if t.sum_is_neg[c] else p[stored].sum()
    assert w == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# d(s)
# --------------------------------------------------------------------------

def test_distance_is_zero_exactly_on_finals():
    """SPEC §3.1b: `d(s) <= 0 <=> s in F` is what removes the final-block
    special case."""
    d = distance_to_final(4, [(0, 1), (1, 2), (2, 3)], finals=[3])
    assert list(d) == [3, 2, 1, 0]


def test_distance_counts_tokens_not_labels():
    """An edge carrying 200 labels is still one token of distance."""
    d = distance_to_final(2, [(0, 1)], finals=[1])
    assert d[0] == 1


def test_unreachable_states_get_inf():
    d = distance_to_final(3, [(0, 1)], finals=[1])
    assert d[2] == INF_DISTANCE


def test_distance_is_computed_after_augmentation():
    """SPEC §3.1b: d must be computed AFTER the stop-token augmentation, or it
    is a completely different function. Before augmentation the grammar-final
    state has d=0; after, it is 1 - it still owes a stop token."""
    a = compile_automaton(line_dfa(2), name="t", end_tokens=(1,), vocab_size=V)
    assert a.d[2] == 1, "grammar-final still needs one stop token"
    assert a.d[3] == 0, "ACC is the only d==0 state"
    assert a.d[0] == 3
    assert a.is_final[3] and not a.is_final[2]


def test_budget_semantics():
    """b_L(s) = 1[d(s) <= R] must admit exactly the states that can still
    finish within the remaining budget."""
    a = compile_automaton(line_dfa(4), name="t", end_tokens=(1,), vocab_size=V)
    d = a.d[: a.n_states]
    assert [int(x) for x in d] == [5, 4, 3, 2, 1, 0]
    assert (d <= 2).sum() == 3  # states 3, 4, ACC


def test_dead_padding_states_are_infinite():
    """Bucketing pads with an absorbing dead state; it must never be viable."""
    a = compile_automaton(line_dfa(2), name="t", end_tokens=(1,), vocab_size=V)
    assert a.n_states_bucket > a.n_states
    assert (a.d[a.n_states:] == INF_DISTANCE).all()
    assert not a.is_final[a.n_states:].any()


# --------------------------------------------------------------------------
# Start vector, bucketing, DFA detection
# --------------------------------------------------------------------------

def test_start_is_a_vector_not_a_scalar():
    """SPEC §5.7: hardcoding a point mass is a bug that first appears on block
    2 or on an NFA."""
    a = compile_automaton(line_dfa(2), name="t", end_tokens=(1,), vocab_size=V)
    assert a.start_vector.dtype == bool
    assert a.start_vector.shape == (a.n_states_bucket,)
    assert a.start_vector.sum() == 1


def test_bucket_rounds_up_to_powers_of_two():
    assert bucket_size(1) == 16
    assert bucket_size(16) == 16
    assert bucket_size(17) == 32
    assert bucket_size(1025) == 2048


def test_bucket_refuses_beyond_the_ladder():
    """SPEC §5.5: do not warm a bucket you can never dispatch to."""
    with pytest.raises(ValueError, match="chain sampler"):
        bucket_size(5000)


def test_tree_bytes_keeps_the_factor_two():
    """SPEC §5.6: '(2L-1)', not L. Dropping the 2 flips the dispatch decision."""
    a = compile_automaton(line_dfa(2), name="t", end_tokens=(1,), vocab_size=V)
    assert a.tree_bytes == 511 * a.n_states_bucket ** 2 * 4


def test_dfa_flag_detects_nondeterminism():
    """eq (8)'s cheap `exists` token draw is only valid on a DFA (SPEC §2.6)."""
    nfa = Dfa(
        n_states=3,
        transitions=((0, 7, 1), (0, 7, 2), (1, 8, 2)),
        start=0,
        finals=frozenset({2}),
    )
    a = compile_automaton(nfa, name="t", end_tokens=(1,), vocab_size=V)
    assert not a.is_dfa

    a2 = compile_automaton(line_dfa(3), name="t", end_tokens=(1,), vocab_size=V)
    assert a2.is_dfa


def test_edges_group_by_state_pair():
    """M_i(s,s') sums over edges between the pair, so a DFA needs exactly one
    edge per (src, dst) carrying the union of its labels."""
    dfa = Dfa(
        n_states=2,
        transitions=((0, 3, 1), (0, 4, 1), (0, 5, 1)),
        start=0,
        finals=frozenset({1}),
    )
    a = compile_automaton(dfa, name="t", end_tokens=(1,), vocab_size=V)
    pairs = list(zip(a.edge_src.tolist(), a.edge_dst.tolist()))
    assert pairs.count((0, 1)) == 1


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    """The artifact must survive a save/load cycle bit-for-bit — Phase 3
    dispatches on these arrays and a silent dtype change would be invisible."""
    from diffgemma_fa.compile import automaton as am

    a = compile_automaton(line_dfa(3), name="rt", end_tokens=END_TOKENS,
                          vocab_size=V, schema_hash="deadbeef")
    p = str(tmp_path / "a.npz")
    am.save(a, p)
    b = am.load(p)

    assert b.name == a.name and b.schema_hash == a.schema_hash
    assert b.n_states == a.n_states and b.n_states_bucket == a.n_states_bucket
    assert b.is_dfa == a.is_dfa
    for field in ("edge_src", "edge_dst", "edge_class", "d", "is_final",
                  "start_vector"):
        x, y = getattr(a, field), getattr(b, field)
        assert x.dtype == y.dtype, field
        assert np.array_equal(x, y), field
    for field in ("sum_is_neg", "sum_indptr", "sum_indices",
                  "max_is_neg", "max_indptr", "max_indices", "class_id"):
        x, y = getattr(a.tables, field), getattr(b.tables, field)
        assert x.dtype == y.dtype, field
        assert np.array_equal(x, y), field
    assert b.tables.k_max == a.tables.k_max
    assert b.tables.max_neg_size == a.tables.max_neg_size


def test_loaded_automaton_simulates_identically(tmp_path):
    from diffgemma_fa.compile import automaton as am
    from diffgemma_fa.compile.validate import Simulator

    a = compile_automaton(line_dfa(2), name="rt", end_tokens=(1,), vocab_size=V)
    p = str(tmp_path / "b.npz")
    am.save(a, p)
    b = am.load(p)
    assert Simulator(a).accepts([7, 7, 1])
    assert Simulator(b).accepts([7, 7, 1])
    assert not Simulator(b).accepts([7, 1])


def test_oversize_bucket_flags_the_chain_path_instead_of_crashing():
    """SPEC §5.6: anything above the top usable bucket goes to the chain
    sampler. Measured on BFCL-Live the largest raw lifted automaton has 3,573
    states — above both this ladder and the 2,459 the paper quotes — so this
    path is real and must not abort a whole compile run."""
    assert bucket_size(5000, allow_oversize=True) == 8192
    with pytest.raises(ValueError, match="chain sampler"):
        bucket_size(5000, allow_oversize=False)


def test_small_automaton_is_not_flagged_for_the_chain_path():
    a = compile_automaton(line_dfa(2), name="t", end_tokens=(1,), vocab_size=V)
    assert not a.needs_chain_path


# --------------------------------------------------------------------------
# SPEC §3.6's channel header
# --------------------------------------------------------------------------

def test_channel_header_admits_the_models_own_first_token():
    """Phase 0 measured that **every** generation opens with
    `<|channel>NAME\\n<channel|>`. Without the header the grammar admits only
    `{` at canvas position 0 — measured 3 tokens, all `{` variants, with token
    100 forbidden — so the model's habitual first token is impossible and the
    whole canvas is decoded from an off-distribution prefix."""
    from diffgemma_fa.compile.automaton import prepend_channel_header

    base = line_dfa(2)
    hdr = prepend_channel_header(base, vocab_size=V, open_token=10,
                                 close_token=11, newline_token=12,
                                 reserved=END_TOKENS, max_name_tokens=3)
    out = {lbl for s, lbl, _ in hdr.transitions if s == hdr.start}
    assert out == {10}, "position 0 must admit exactly the channel-open token"


def test_channel_header_name_repetition_is_BOUNDED():
    """**The name must not be a Σ* self-loop.**

    A self-loop over the whole vocabulary has emission mass ~1.0 — exactly like
    the unscored `ACC --Σ--> ACC` tail — so staying in it is free and a joint
    MAP will sit there for the entire canvas rather than pay a specific token's
    probability to leave. Measured when this was a self-loop: the MAP consumed
    all 64 positions inside the name loop, never closed the header, and the
    emission was rejected.
    """
    from diffgemma_fa.compile.automaton import prepend_channel_header

    hdr = prepend_channel_header(line_dfa(2), vocab_size=V, open_token=10,
                                 close_token=11, newline_token=12,
                                 reserved=END_TOKENS, max_name_tokens=3)
    for src, lbl, dst in hdr.transitions:
        assert not (src == dst), (
            f"self-loop at state {src} on {lbl} — the name repetition must be "
            "bounded, or the MAP never leaves it"
        )


def test_channel_header_accepts_one_to_max_name_tokens():
    from diffgemma_fa.compile.automaton import prepend_channel_header
    from diffgemma_fa.compile.validate import Simulator

    n = 3
    hdr = prepend_channel_header(line_dfa(1), vocab_size=V, open_token=10,
                                 close_token=11, newline_token=12,
                                 reserved=END_TOKENS, max_name_tokens=n)
    a = compile_automaton(hdr, name="h", end_tokens=(1,), vocab_size=V)
    sim = Simulator(a)
    for k in range(1, n + 1):
        word = [10] + [5] * k + [12, 11, 7, 1]
        assert sim.accepts(word), f"a {k}-token name must be accepted"
    too_long = [10] + [5] * (n + 1) + [12, 11, 7, 1]
    assert not sim.accepts(too_long), "the name bound must actually bind"
    empty_name = [10, 12, 11, 7, 1]
    assert not sim.accepts(empty_name), (
        "the name is Sigma_name PLUS, not star -- a zero-token name would let "
        "the header collapse to `<|channel>\\n<channel|>`"
    )


# ==========================================================================
# The structural invariants are re-proved on load, not just at build time
# ==========================================================================

def test_load_rejects_a_duplicated_state_pair(tmp_path):
    """`_group_edges` emits one edge per `(src, dst)` carrying the union of its
    labels, which is what makes eq (8)'s multiplicity 0/1 by construction. That
    held at build time and was then never re-established — a hand-made or stale
    `.npz` with a duplicated pair loaded silently.

    The damage would be a wrong *distribution*: valid strings drawn with the
    wrong probabilities. No acceptance check can see that, which is exactly why
    it has to be an invariant rather than a test on outputs.
    """
    a = compile_automaton(line_dfa(3), name="t", end_tokens=(1,), vocab_size=V)
    p = tmp_path / "a.npz"
    automaton.save(a, str(p))

    z = dict(np.load(p, allow_pickle=False))
    z["edge_src"] = np.append(z["edge_src"], z["edge_src"][0])
    z["edge_dst"] = np.append(z["edge_dst"], z["edge_dst"][0])
    z["edge_class"] = np.append(z["edge_class"], z["edge_class"][0])
    forged = tmp_path / "forged.npz"
    np.savez(forged, **z)

    with pytest.raises(ValueError, match="duplicated"):
        automaton.load(str(forged))


def test_load_rejects_an_is_dfa_flag_that_over_claims(tmp_path):
    """`is_dfa` gates eq (8)'s cheap `∃` token draw (SPEC §2.6), so a flag that
    over-claims is a correctness bug, not a performance one."""
    nfa = Dfa(n_states=3, transitions=((0, 7, 1), (0, 8, 2), (1, 9, 2)),
              start=0, finals=frozenset({2}))
    a = compile_automaton(nfa, name="t", end_tokens=(1,), vocab_size=V)
    p = tmp_path / "a.npz"
    automaton.save(a, str(p))

    z = dict(np.load(p, allow_pickle=False))
    meta = json.loads(bytes(z["meta"]).decode())
    # Force two destinations for one (src, class), then claim determinism.
    z["edge_src"] = np.array([0, 0], np.int32)
    z["edge_dst"] = np.array([1, 2], np.int32)
    z["edge_class"] = np.array([0, 0], np.int32)
    meta["is_dfa"] = True
    z["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    forged = tmp_path / "forged.npz"
    np.savez(forged, **z)

    with pytest.raises(ValueError, match="is_dfa"):
        automaton.load(str(forged))


def test_a_real_automaton_round_trips_through_the_new_checks(tmp_path):
    a = compile_automaton(line_dfa(4), name="t", end_tokens=(1,), vocab_size=V)
    p = tmp_path / "a.npz"
    automaton.save(a, str(p))
    b = automaton.load(str(p))
    assert b.n_states == a.n_states and b.n_edges == a.n_edges


def test_the_cache_key_separates_grammars_that_differ_only_by_options():
    """SPEC §4.7(3)'s cache key used to hash the schema alone.

    The *same* schema compiles to materially different grammars under
    `whitespace_pattern`, `nonempty_required_strings`, `channel_header`,
    `allow` and `from_bfcl` — two of those change the accepted language
    outright. The key collided, so a cache hit would return the wrong
    automaton, and since every returned automaton is internally consistent
    nothing downstream could tell.
    """
    sch = {"type": "object", "properties": {"a": {"type": "string"}},
           "required": ["a"]}
    base = automaton.schema_fingerprint(sch)
    assert automaton.schema_fingerprint(sch) == base, "must be stable"

    differing = [
        {"whitespace_pattern": None},
        {"whitespace_pattern": ""},
        {"nonempty_required_strings": True},
        {"channel_header": False},
        {"from_bfcl": True},
        {"allow": ("minimum",)},
    ]
    keys = {automaton.schema_fingerprint(sch, **o) for o in differing}
    assert len(keys) == len(differing), "two option sets share a cache key"
    assert base not in keys


def test_the_cache_key_ignores_the_order_of_the_allow_list():
    """`allow` is a set of keywords: its order must not change the key, but its
    content must."""
    sch = {"type": "object", "properties": {"a": {"type": "string"}}}
    k1 = automaton.schema_fingerprint(sch, allow=("minimum", "maximum"))
    k2 = automaton.schema_fingerprint(sch, allow=("maximum", "minimum"))
    k3 = automaton.schema_fingerprint(sch, allow=("minimum",))
    assert k1 == k2
    assert k1 != k3
