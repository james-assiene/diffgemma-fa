"""The significance machinery behind `docs/RESULTS.md`.

A wrong interval is worse than no interval: it makes a table look settled when
it is not. Both functions are checked against values computable by hand or from
the published definition, not against another implementation of themselves.
"""

from __future__ import annotations

import math

import pytest

from diffgemma_fa.eval.stats import (
    bonferroni,
    fmt_rate,
    mcnemar_exact,
    wilson,
)


# --------------------------------------------------------------------------
# Wilson
# --------------------------------------------------------------------------

def test_wilson_does_not_degenerate_at_a_perfect_rate():
    """The reason this is not a normal-approximation interval. At `k == n` the
    normal interval is `[1.0, 1.0]` — it claims certainty from 130 samples,
    exactly where every constrained arm sits."""
    lo, hi = wilson(130, 130)
    assert hi == 1.0
    assert 0.95 < lo < 1.0, f"lower limit {lo} is degenerate or absurd"


def test_wilson_does_not_exceed_the_unit_interval_at_zero():
    lo, hi = wilson(0, 130)
    assert lo == 0.0
    assert 0.0 < hi < 0.05


def test_wilson_matches_the_published_value():
    """Textbook worked example: 8 successes in 10 trials, 95% -> about
    [0.490, 0.943] (Wilson 1927; Brown, Cai & DasGupta 2001 Table 1)."""
    lo, hi = wilson(8, 10)
    assert lo == pytest.approx(0.490, abs=2e-3)
    assert hi == pytest.approx(0.943, abs=2e-3)


def test_wilson_is_centred_near_the_point_estimate_for_large_n():
    lo, hi = wilson(65, 130)
    assert (lo + hi) / 2 == pytest.approx(0.5, abs=1e-3)


def test_wilson_narrows_with_n():
    w_small = wilson(13, 26)
    w_large = wilson(65, 130)
    assert (w_large[1] - w_large[0]) < (w_small[1] - w_small[0])


def test_wilson_on_no_data_is_uninformative_not_an_error():
    assert wilson(0, 0) == (0.0, 1.0)


# --------------------------------------------------------------------------
# McNemar
# --------------------------------------------------------------------------

def test_mcnemar_ignores_concordant_records_by_construction():
    """The signature takes only the two discordant counts. Concordant records
    carry no information about the *difference*, and conditioning on them is
    what makes this the right test for paired arms."""
    assert mcnemar_exact(10, 0) == mcnemar_exact(10, 0)


def test_mcnemar_is_a_two_sided_binomial_tail():
    """`b = 10, c = 0` is 10 straight successes: `2 * 0.5**10`."""
    assert mcnemar_exact(10, 0) == pytest.approx(2 * 0.5 ** 10)
    assert mcnemar_exact(0, 10) == pytest.approx(2 * 0.5 ** 10), "symmetric"


def test_mcnemar_on_a_perfectly_balanced_split_is_not_significant():
    assert mcnemar_exact(5, 5) == 1.0


def test_mcnemar_with_no_discordant_pairs_is_one():
    """Two arms that agree on every record. Not "significantly identical" —
    simply no evidence of a difference."""
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_never_exceeds_one():
    for b, c in [(1, 1), (2, 3), (7, 6), (1, 0), (60, 61)]:
        assert 0.0 < mcnemar_exact(b, c) <= 1.0


def test_mcnemar_matches_a_hand_computed_case():
    """`b = 3, c = 0`: `2 * (C(3,0)) / 2**3 = 0.25`."""
    assert mcnemar_exact(3, 0) == pytest.approx(0.25)


def test_a_realistic_arm_comparison_is_significant():
    """The shape the table will actually report: the constrained arm fixes 20
    records the baseline failed and breaks none."""
    assert mcnemar_exact(20, 0) < 1e-5
    # ...and it survives correcting for all 15 pairwise comparisons.
    assert bonferroni(mcnemar_exact(20, 0), 15) < 1e-4


# --------------------------------------------------------------------------
# Correction and formatting
# --------------------------------------------------------------------------

def test_bonferroni_clips_at_one():
    assert bonferroni(0.5, 15) == 1.0
    assert bonferroni(0.001, 15) == pytest.approx(0.015)


def test_bonferroni_of_a_single_comparison_is_the_identity():
    assert bonferroni(0.03, 1) == pytest.approx(0.03)
    assert bonferroni(0.03, 0) == pytest.approx(0.03), "n=0 must not divide"


def test_fmt_rate_always_carries_an_interval():
    s = fmt_rate(120, 130)
    assert s.startswith("0.923")
    assert "[" in s and "]" in s
    lo = float(s.split("[")[1].split(",")[0])
    hi = float(s.split(", ")[1].rstrip("]"))
    assert lo < 120 / 130 < hi


def test_the_interval_is_the_reason_130_records_was_chosen():
    """A power sanity check on the sample size, so the number is not folklore.

    At `n = 130` a rate of 0.85 has a 95% Wilson half-width under 0.07, which is
    tight enough to separate the baseline from a constrained arm at 1.0.
    """
    lo, hi = wilson(int(round(0.85 * 130)), 130)
    assert (hi - lo) / 2 < 0.07
    assert hi < 1.0, "0.85 at n=130 must be distinguishable from a perfect rate"
