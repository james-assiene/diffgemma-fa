"""Significance for the SPEC §7.2 baseline table.

Two tests, chosen for what the table actually claims:

- **Wilson score intervals** on every rate. At `n = 130` a normal-approximation
  interval on a rate near 1.0 extends past 1.0 and is meaningless exactly where
  the constrained arms live; Wilson does not.
- **McNemar's exact test** for arm-vs-arm differences. The arms are run on the
  **same records**, so they are paired — an unpaired two-proportion test throws
  away that pairing and is both wrong and less powerful. McNemar looks only at
  the discordant records, which is the right conditioning.

No new dependency: both are a few lines of `math`, and `scipy` is not in the
intended stack.

**Multiple comparisons.** The table has 6 arms, so 15 pairwise comparisons per
metric. `bonferroni` here is deliberately explicit rather than a default — SPEC
§6.1's warning that an uncorrected suite of dozens of tests goes red on its own
applies just as much to reading a results table as to running one.
"""

from __future__ import annotations

import math

__all__ = ["wilson", "mcnemar_exact", "bonferroni", "fmt_rate"]


def wilson(k: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for `k` successes in `n` trials.

    Defaults to 95% (`z = 1.96`). Correct at the boundaries: `k = n` gives an
    upper limit of exactly 1.0 and a lower limit strictly inside (0, 1), where
    the normal approximation gives the degenerate `[1.0, 1.0]`.

    Returns:
      `(lo, hi)`. `(0.0, 1.0)` for `n == 0`.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar `p`-value from the two discordant counts.

    Args:
      b: records where arm A succeeded and arm B failed.
      c: records where arm B succeeded and arm A failed.

    The concordant records carry no information about the *difference* and are
    correctly ignored — that is the whole point of pairing. The exact binomial
    form is used rather than the chi-square approximation because `b + c` is
    routinely below 25 here, where the approximation is not trustworthy.

    Returns:
      `p`, clipped to 1.0. `1.0` when there are no discordant pairs.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def bonferroni(p: float, n_comparisons: int) -> float:
    """`p` corrected for `n_comparisons`, clipped to 1.0.

    Stated explicitly at every call site rather than applied silently, so a
    reader can see how many comparisons the table actually makes.
    """
    return min(1.0, p * max(1, n_comparisons))


def fmt_rate(k: int, n: int) -> str:
    """`0.923 [0.865, 0.958]` — a rate is never printed without its interval."""
    lo, hi = wilson(k, n)
    return f"{k / max(1, n):.3f} [{lo:.3f}, {hi:.3f}]"
