"""Phase 0 / SPEC §1.3(3), §4.3 — can outlines-core consume Gemma's vocabulary?

`Vocabulary.from_pretrained` needs the gated HF repo, so this builds the
Vocabulary directly from Gemma's own SentencePiece model (which we have
locally) and checks that `Index(regex, vocabulary).get_transitions()` — the
only shipped path that materialises an explicit state -> {token -> state}
table (SPEC §4.3) — actually works on it.
"""

from __future__ import annotations

import json
import sys
import time

import outlines_core as oc
from gemma import gm


def build_vocabulary(tok) -> tuple[oc.Vocabulary, dict[str, object]]:
    """Byte-level Vocabulary from Gemma's SentencePiece pieces.

    SentencePiece normalisation: `<0xXX>` byte-fallback pieces map to the raw
    byte; every other piece has U+2581 (lower one-eighth block) standing in for
    a leading space. Control/special pieces are excluded — they carry no bytes
    and must not appear in any automaton alphabet (SPEC §3.6).
    """
    sp = tok._sp  # noqa: SLF001 - the underlying sentencepiece processor
    n = sp.GetPieceSize()

    stats = {
        "piece_size": int(n),
        "byte_fallback": 0,
        "control_or_unused": 0,
        "mapped": 0,
    }

    vocab = oc.Vocabulary(int(tok.special_tokens.EOS), {})
    for i in range(n):
        piece = sp.IdToPiece(i)
        # Control tokens (BOS/EOS/PAD/UNK and the <...> specials) emit no bytes.
        if sp.IsControl(i) or sp.IsUnknown(i):
            stats["control_or_unused"] += 1
            continue
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            b = bytes([int(piece[3:5], 16)])
            stats["byte_fallback"] += 1
        else:
            b = piece.replace("▁", " ").encode("utf-8")
            if not b:
                stats["control_or_unused"] += 1
                continue
        vocab.insert(b, i)
        stats["mapped"] += 1

    return vocab, stats


def main() -> None:
    out: dict[str, object] = {}
    tok = gm.text.Gemma4Tokenizer()

    t0 = time.perf_counter()
    vocab, stats = build_vocabulary(tok)
    out["build_seconds"] = round(time.perf_counter() - t0, 2)
    out["vocab_stats"] = stats
    out["vocab_len"] = len(vocab)
    out["eos_token_id"] = vocab.get_eos_token_id()

    # --- json schema -> regex (SPEC §4.2) ---------------------------------
    schema = json.dumps({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["name", "count"],
    })
    t0 = time.perf_counter()
    regex = oc.json_schema.build_regex_from_schema(schema)
    out["schema_to_regex_seconds"] = round(time.perf_counter() - t0, 3)
    out["regex_len"] = len(regex)
    out["regex_head"] = regex[:200]

    # --- token lift (SPEC §4.3) -------------------------------------------
    # This is the step SPEC §4.7 estimates at 4-8 minutes per schema at 262k
    # vocab. Measure it, because the whole Phase 1 schedule depends on it.
    t0 = time.perf_counter()
    index = oc.Index(regex, vocab)
    out["index_build_seconds"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    transitions = index.get_transitions()
    out["get_transitions_seconds"] = round(time.perf_counter() - t0, 2)

    out["num_states"] = len(transitions)
    state_ids = sorted(transitions.keys())
    out["state_id_min"] = int(state_ids[0])
    out["state_id_max"] = int(state_ids[-1])
    out["state_ids_dense"] = state_ids == list(range(len(state_ids)))
    out["state_id_sample"] = [int(s) for s in state_ids[:12]]
    gaps = {
        int(state_ids[i + 1] - state_ids[i]) for i in range(min(50, len(state_ids) - 1))
    }
    out["state_id_gap_set_first50"] = sorted(gaps)
    out["nnz_transitions"] = int(sum(len(v) for v in transitions.values()))
    out["mean_tokens_per_state"] = round(
        out["nnz_transitions"] / max(1, out["num_states"]), 1
    )

    out["index_attrs"] = [a for a in dir(index) if not a.startswith("_")]

    json.dump(out, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
