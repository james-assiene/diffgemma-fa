"""Scoring. SPEC §7.2, §3.8, §4.8.

Three things are reported together and none substitutes for another:

- **CS** — constraint satisfaction, checked by an *independent* simulator over
  the compiled automaton. This is the paper's headline column.
- **accuracy** — BFCL's own notion, using **BFCL's own normalisation**
  (lowercase, strip `",./-_*^`, per SPEC §4.8). Scoring more strictly than the
  benchmark understates the result and is not comparable to the leaderboard.
- **content** — parse rate and non-empty-argument rate.

That last one exists because of SPEC §3.8's warning, sharpened by Phase 4:

> with a free-text branch … the model can spend its whole budget outside
> `FA_grammar`, making CS trivially 100% while measuring nothing.

A run that emits `{"location":""}` every time scores CS = 100% and is useless.
**Never report CS without the content columns beside it.**
"""

from __future__ import annotations

import dataclasses
import json

__all__ = ["Scores", "normalise", "extract_json", "score_arguments"]

#: BFCL's value normalisation (SPEC §4.8): "scoring lowercases and strips
#: `",./-_*^`".
_STRIP = '",./-_*^'


def normalise(value) -> str:
    s = str(value).strip().lower()
    for ch in _STRIP:
        s = s.replace(ch, "")
    return s


def extract_json(text: str) -> dict | None:
    """Pull the argument object out of a channel-tagged completion.

    The grammar carries SPEC §3.6's `<|channel>NAME\\n<channel|>` header, so the
    emission legitimately *begins* with it — naive splitting on `<` yields the
    empty string and scores everything unparsable.
    """
    body = text
    if "<channel|>" in body:
        body = body.split("<channel|>", 1)[1]
    start = body.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(body)):
        c = body[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(body[start:i + 1])
                except Exception:  # noqa: BLE001
                    return None
                return obj if isinstance(obj, dict) else None
    return None


@dataclasses.dataclass
class Scores:
    """Running totals for one configuration."""

    n: int = 0
    cs: int = 0                 # accepted by the independent simulator
    parsed: int = 0
    nonempty: int = 0
    arg_correct: int = 0
    arg_total: int = 0
    exact_calls: int = 0        # every argument correct
    detail: list = dataclasses.field(default_factory=list)

    def add(self, *, accepted: bool, parsed_obj: dict | None,
            want: dict | None) -> None:
        self.n += 1
        self.cs += int(accepted)
        if parsed_obj is None:
            return
        self.parsed += 1
        if any(v not in ("", None, [], {}) for v in parsed_obj.values()):
            self.nonempty += 1
        if want is None:
            return
        n_ok, n_tot, det = score_arguments(parsed_obj, want)
        self.arg_correct += n_ok
        self.arg_total += n_tot
        self.exact_calls += int(n_tot > 0 and n_ok == n_tot)
        self.detail.extend(det)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "cs": self.cs,
            "cs_rate": round(self.cs / max(1, self.n), 4),
            "parsed": self.parsed,
            "nonempty": self.nonempty,
            "nonempty_rate": round(self.nonempty / max(1, self.n), 4),
            "arg_correct": self.arg_correct,
            "arg_total": self.arg_total,
            "arg_accuracy": round(self.arg_correct / max(1, self.arg_total), 4),
            "exact_calls": self.exact_calls,
            "exact_call_rate": round(self.exact_calls / max(1, self.n), 4),
        }


def score_arguments(pred: dict, want: dict) -> tuple[int, int, list]:
    """`(n_correct, n_expected, per_key)` under BFCL's normalisation."""
    detail, n_ok = [], 0
    for key, value in want.items():
        got = pred.get(key)
        ok = key in pred and normalise(got) == normalise(value)
        n_ok += int(ok)
        detail.append({"key": key, "want": str(value)[:80],
                       "got": str(got)[:80], "ok": ok})
    return n_ok, len(want), detail
