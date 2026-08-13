Artifacts displaced by the 16-arm queue of 2026-08-12 (PID 3405194), which
archived twelve tags and then correctly self-aborted at its third arm.

THE HAZARD: eleven of these have NO current version in artifacts/. The queue
archives before running, so a stop leaves the tag with nothing in place.
`scripts/phase5_report.py` globs artifacts/ and would SILENTLY OMIT those rows
from a regenerated docs/RESULTS.md rather than failing.

    eval_bfcl_live_simple_mask_sample   exp_marmap_b0.01
    exp_f64_ws_only_j1                  exp_marmap_b0.1
    exp_h2_grammar_j1                   exp_marmap_b0.5
    exp_h2_marmap                       exp_oomfix_j0
    exp_h2_stock_j1                     task_countdown_mask
                                        task_sudoku_j1_sample

DO NOT restore these to artifacts/ to "fix" the gap. They are pre-2c9f9d9 and
pre-15cdc47: their mask/mar arms were measured through a prefix_suffix that
silently erased live transitions, and their BFCL arms decoded records 117 and
122 under a `type: any` grammar that rejects the correct object. That is why
they were archived.

The correct repair is to re-run them. Until then, docs/RESULTS.md must not be
regenerated from artifacts/ -- its committed text is the record.

task_sudoku_mask IS present and current: it is the one arm the queue completed,
and it returned cs = 1/250 against a predicted >= 2, which is what stopped the
queue. The prefix_suffix fix did not repair that closure and the cause is open.
