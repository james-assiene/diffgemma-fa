"""Phase 0 / SPEC §1.3 — tokenizer fact verification.

Answers, by measurement:
  1. vocab_size, and whether it exceeds the real token count (lm_head padding).
  2. Whether a <mask> token exists (and confirm it is NOT the diffusion mechanism).
  3. Whether outlines_core.Vocabulary.from_pretrained accepts Gemma's decoder config.
  4. Whether the end-of-thought marker is a single dedicated token id (SPEC §3.6).
  5. The full end_tokens tuple used by the diffusion sampler (SPEC §3.5 trap 3).
"""

from __future__ import annotations

import json
import sys

from gemma import gm


def main() -> None:
    out: dict[str, object] = {}

    tok = gm.text.Gemma4Tokenizer()
    st = tok.special_tokens

    out["vocab_size"] = int(tok.vocab_size)
    out["vocab_size_is_pow2"] = (int(tok.vocab_size) & (int(tok.vocab_size) - 1)) == 0

    # --- 1. real vs padded token count -------------------------------------
    # Pad slots decode to "" and must be excluded from every automaton alphabet.
    empty_ids, byte_fallback_ids = [], []
    decoded: list[str] = []
    for i in range(int(tok.vocab_size)):
        try:
            s = tok.decode([i])
        except Exception:  # noqa: BLE001 - we want the id, not the traceback
            s = None
        decoded.append(s if s is not None else "\x00<ERR>")
        if s == "":
            empty_ids.append(i)
    out["num_empty_decoding_ids"] = len(empty_ids)
    out["first_empty_ids"] = empty_ids[:10]
    out["last_empty_ids"] = empty_ids[-10:]
    # Contiguous tail of empties = the lm_head power-of-two padding.
    tail = 0
    for i in range(int(tok.vocab_size) - 1, -1, -1):
        if decoded[i] == "":
            tail += 1
        else:
            break
    out["contiguous_empty_tail"] = tail
    out["highest_nonempty_id"] = int(tok.vocab_size) - tail - 1

    # --- 2. special tokens --------------------------------------------------
    specials = {}
    for name in dir(st):
        if name.startswith("_"):
            continue
        v = getattr(st, name)
        if isinstance(v, int):
            specials[name] = v
    out["special_tokens"] = dict(sorted(specials.items(), key=lambda kv: kv[1]))

    # end_tokens as the diffusion Sampler builds them (no user stop_tokens).
    out["end_tokens_default"] = [
        int(st.EOS),
        int(st.END_OF_TURN),
        int(st.BEGIN_OF_TOOL_RESPONSE),
    ]

    # --- 3. mask token ------------------------------------------------------
    mask_like = {
        name: v for name, v in specials.items() if "mask" in name.lower()
    }
    out["mask_like_special_tokens"] = mask_like
    probe_mask = {}
    for lit in ("<mask>", "[MASK]", "<unused0>"):
        try:
            ids = tok.encode(lit)
            probe_mask[lit] = {"ids": [int(x) for x in ids], "n": len(ids)}
        except Exception as e:  # noqa: BLE001
            probe_mask[lit] = {"error": repr(e)}
    out["mask_literal_probe"] = probe_mask

    # --- 4. end-of-thought marker (SPEC §3.6) -------------------------------
    thought = {}
    for name, v in specials.items():
        if "thought" in name.lower() or "think" in name.lower():
            thought[name] = v
    out["thought_special_tokens"] = thought
    marker_probe = {}
    for lit in (
        "<end_of_thought>",
        "</thought>",
        "<start_of_thought>",
        "</think>",
        "<think>",
    ):
        try:
            ids = [int(x) for x in tok.encode(lit)]
            marker_probe[lit] = {
                "ids": ids,
                "n_tokens": len(ids),
                "single_dedicated": len(ids) == 1,
                "roundtrip": tok.decode(ids),
            }
        except Exception as e:  # noqa: BLE001
            marker_probe[lit] = {"error": repr(e)}
    out["thought_marker_probe"] = marker_probe

    # --- 5. outlines-core acceptance ---------------------------------------
    try:
        import outlines_core  # noqa: PLC0415

        out["outlines_core_version"] = getattr(outlines_core, "__version__", "?")
        try:
            v = outlines_core.Vocabulary.from_pretrained("google/gemma-3-4b-it")
            out["outlines_from_pretrained"] = {
                "ok": True,
                "len": len(v),
            }
        except Exception as e:  # noqa: BLE001
            out["outlines_from_pretrained"] = {"ok": False, "error": repr(e)[:400]}
    except ImportError as e:
        out["outlines_core_version"] = None
        out["outlines_from_pretrained"] = {"ok": False, "error": repr(e)}

    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main()
