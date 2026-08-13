#!/usr/bin/env bash
# =====================================================================
# Re-run queue at 1aa8ccf.  REVIEWED, NOT YET LAUNCHED.
#
# Supersedes scratchpad/rerun_2c9f9d9_LOCKED.sh, which ran ONE arm
# (smoke_2c9f9d9_mask, 04:14-04:18) and then stopped itself at 04:19 because
# HEAD moved: docs/LOG.md was committed one minute after launch.  The guard was
# right to fire on the condition it was given and wrong about which condition
# mattered; both are fixed below (section 6A).  Its design -- gate structure,
# per-arm config readback out of the artifact, archive-before-overwrite,
# idempotent [skip] -- is reused nearly verbatim, because it was reviewed and
# it was right.
#
# THREE commits have landed since that script was written:
#   2c9f9d9  prefix_suffix: log-space forward-backward with a widening floor
#   15cdc47  `type: any`: expand typeless nodes to a parenthesised anyOf
#   1aa8ccf  audit infer/: the arbiter had the defects it was certifying against
# and 15cdc47 MOVES THE DENOMINATOR, which the old script was written around.
# Section 2 is the whole of that change and it is not optional reading.
#
# ---------------------------------------------------------------------
# 1. WHAT IS INVALIDATED, AND WHAT IS NOT -- RE-DERIVED FROM THE CODE
# ---------------------------------------------------------------------
# (a) 2c9f9d9.  `infer/scans.prefix_suffix` returned linear a, b normalised PER
#     VECTOR, bounding neither factor's within-vector range nor their product.
#     At the shipped sharpness it erased structurally live edges: 16 of 256
#     Sudoku positions and 7 of 256 uber.ride positions had EMPTY support.
#     Its callers are exactly two, by grep at THIS commit, not by inheritance:
#       model/sampler.py:433  the `--confidence mar` branch -> constrained_entropy_streamed
#       model/sampler.py:502  the `--variant mask` branch   -> scatter_edge_mass_to_tokens
#     Nothing else in model/ calls it.  Damage was directional and silent:
#       mask: an all -1e30 float32 row, Gumbel noise swamped, `categorical`
#             returns token 0 DETERMINISTICALLY.  No exception, no Z==0.
#       mar : H = -708.4 nats against a live median of 1.6e-4, i.e. below every
#             entropy_bound, so `_accept_from_entropy` sorts it first and ALWAYS
#             accepts.  The accept mask itself moved -> fresh baseline, not a
#             rescore.  A rescore cannot recover a decode that never happened.
#
# (b) 1aa8ccf touched THREE files under infer/, and only one of the three
#     changes can reach a number.  Checked, not assumed:
#       * infer/reference.py -- the float64 arbiter.  Not on any GPU path.
#       * infer/scans.py -- `_normalize` now also returns a `[n] bool` "this
#         division took a strictly positive entry to exactly 0.0", `up_sweep`
#         ORs it into a new `TreeLevels.underflow`, and `prefix_suffix` ANDs
#         `~underflow` into `representable`.  `levels` and `log_scales` are
#         byte-for-byte what they were, so no consumer of the tree moves; the
#         only behavioural change is that `feasible` can now go False where it
#         previously did not, which raises ZeroPartitionError (sampler.py:646)
#         instead of returning a silently degraded draw.  `scans.up_sweep` --
#         the linear sweep, the only one that normalises -- is called at
#         sampler.py:432 and 501 and NOWHERE ELSE, i.e. in the same two
#         branches 2c9f9d9 hit.  Every arm that can see this change is re-run
#         below anyway.  Predicted firing rate: 0 (the commit measures 0/256
#         positions on bfcl, countdown and sudoku at T = 1.0/0.4/0.2/0.1), so a
#         nonzero `zero_partition` on a mask/mar arm now has two candidate
#         causes -- 2c9f9d9's widening floor and this -- and the artifact's
#         zero_partition_records are what tell them apart.
#       * infer/tree.py -- `map_states_and_tokens` now breaks ties by lowest
#         TOKEN id (SPEC 2.7) instead of lowest EDGE index.  A no-op on the
#         compiled path and I verified the premise rather than quoting it:
#         `compile/automaton._group_edges` builds `grouped[(src,dst)]` as a
#         dict keyed on the PAIR, so exactly one edge exists per (src,dst),
#         `sel` selects one edge, and `min(tok over tied)` reduces to the old
#         `argmax`.  So every `--emission map` arm is unaffected.
#
# (c) NOT re-run, on a traced code path rather than caution:
#       * every j0 / j1 / j2 / unconstrained arm at `--confidence mf`.
#         `_accept_mask` (sampler.py:659) reads the UNCONSTRAINED shaped logits
#         -- `jax.nn.softmax(out.logits)` -- and never calls prefix_suffix;
#         `rep_ok` is initialised to all-ones and is only narrowed inside the
#         `mar` and `mask` branches.  joint_map / joint_draw call neither
#         class_weights nor scatter_edge_mass_to_tokens nor prefix_suffix.
#       * `--emission map` with `--confidence mf`: same, plus (b)'s tie-break
#         no-op above.
#       * SPEC 7.3's overhead table.  prefix_suffix is off the J0/J1 hot path,
#         so the tree cost it measures is unchanged; the fix's own cost is
#         +2.3/+5.6/+14.8% of the MASK closure at S=64/128/256, which is not in
#         that table, and `while == 0` / `scan == 0` still hold.
#     THE EXEMPTION IS NUMERICAL AND IT IS NOT A DENOMINATOR EXEMPTION.  See
#     section 2: 15cdc47 changes the grammar of 2 records for EVERY bfcl
#     offset-0 arm, exempt or not.
#
# ---------------------------------------------------------------------
# 2. THE DENOMINATOR.  THREE CATEGORIES, AND CONFLATING THEM IS THE TRAP
# ---------------------------------------------------------------------
# BFCL's `"type": "any"` has no JSON Schema spelling and was normalised by
# OMITTING `type`; outlines-core 0.2.14 expands an omitted type into a 7-way
# alternation IT DOES NOT PARENTHESISE, so the object's braces attach to the
# first and last branch only.  The grammar then rejects the correct object and
# accepts a bare `1`.  15cdc47 expands typeless nodes to a parenthesised anyOf.
#
# MEASURED HERE, not inherited: `normalize_bfcl_schema(p, expand_wildcard=True)`
# differs from `expand_wildcard=False` on EXACTLY 2 of the 258 live_simple
# records -- index 117 (`live_simple_117-73-0`) and index 122
# (`live_simple_122-78-0`) -- and on 0 of records 130..257.  Under JSON_WS the
# build gate's own predicate goes bad=['compact','spaced','indent2','indent4',
# 'tabs'] -> bad=[] for both.  That is checked again in the preflight, in code,
# before any GPU time (hard stop A5).
#
#   category                                            effect on a re-run
#   ---------------------------------------------------------------------
#   1. records_available=130 AND skipped {compile:ValueError: 2}
#      eval_bfcl_live_simple_mask_sample, exp_marmap_b{0.01,0.1,0.5}
#      (and the three unexempt siblings j0/j1/j2_sample, not in this queue)
#                                                       n 128 -> 130
#   2. records_available=128, no skips  (every --offset 130 arm)
#      exp_h2_marmap, exp_h2_stock_j1, exp_h2_grammar_j1
#                                                       stays 128, DIFFERENT cut
#   3. PRE-GATE records_available=130, no skips
#      exp_oomfix_j0, exp_f64_ws_only_j1                n stays 130, BUT cs /
#      schema_valid / arg_accuracy were measured on 2 of 130 under a grammar
#      that cannot emit a well-formed object.  A post-fix 130 is NOT comparable
#      to a pre-gate 130 either.
#
# DISCRIMINATE ON `records_available`, NEVER ON `n`.  The old script's section 3
# predicted exp_oomfix_j0 and exp_f64_ws_only_j1 would "return n=128" because
# the build gate refuses two records; 15cdc47 retires that prediction and they
# return n=130.  Every `require_n` gate below encodes the post-fix expectation,
# so if the gate refuses anything the queue stops instead of quietly publishing
# the old denominator.
#
# WHAT THIS DOES TO THE COMPARISONS (state it in docs/RESULTS.md; the queue
# cannot fix it and must not pretend to):
#   * docs/RESULTS.md's headline table and its McNemar pairs are the PRE-GATE
#     n=130 artifacts (now in artifacts/stale_pre_da1294a/), which include
#     records 117 and 122 scored under the broken grammar -- RESULTS.md already
#     says so, in the "Known defect inside this cut" note, and asks for the
#     table to be regenerated once the compiler fix lands.  This queue re-runs
#     the `mask-sample` row post-fix at n=130 and does NOT re-run
#     unconstrained / j0-sample / j1-sample / j2-sample / j0-map.  So after this
#     queue that table would straddle two grammars, and `mask` vs `j*` -- SPEC
#     2.8's headline evidence -- is the pair that breaks.  Two honest ways out,
#     both post-queue: (i) CPU only, restrict every paired comparison to the
#     128 records common to both grammars, which is computable from the `rows`
#     list each artifact already stores (it carries per-record `id`, `parsed`,
#     `accepted`, `schema_ok`); or (ii) ~6.4 more GPU-hours to re-run the five
#     partner arms post-fix (81 + 81 + 80 + 81 + 59 min from their own
#     elapsed_seconds).  Choose deliberately; do not average the two grammars.
#   * exp_h2_marmap (re-run) vs exp_h2_mfmap (exempt, not re-run) stays clean on
#     BOTH axes: records 130..257 contain 0 wildcard schemas (measured above),
#     so the grammar is identical, and both are records_available=128.
#   * The mar sweep's mf control is exp_e5_grammar130_map, a category-3 pre-gate
#     n=130 arm.  Post-queue the mar points are post-fix n=130 and the control
#     is not; restrict to the common 128 or re-run the control (103 min).
#
# ---------------------------------------------------------------------
# 3. PER-ARM CONFIGURATION, READ OUT OF EACH EXISTING ARTIFACT
# ---------------------------------------------------------------------
# Every measurement-affecting flag is passed EXPLICITLY, including the ones
# that equal today's default.  eval/run.py defaults to --variant j1 --emission
# sample (contradicting SPEC 3.9 and eval/run_tasks.py) and to --whitespace
# json -- and NO artifact in this repo has ever been measured at json.  An
# omitted flag is not the default you assume.
#
#  tag                               var  emis conf bound ws     fnc ci  dtyp seed off  n    avail skips        -> new n
#  eval_bfcl_live_simple_mask_sample mask samp mf   0.1   stock  no  no  f64  0    0    130  130   {CVE: 2}        130
#  exp_marmap_b0.003                 j0   map  mar  0.003 pretty yes yes f64  0    0    130  130   {CVE: 2}*       130
#  exp_marmap_b0.01 / .1 / .5        j0   map  mar  ...   pretty yes yes f64  0    0    130  130   {CVE: 2}        130
#  exp_marmap_b1.0                   j0   map  mar  1.0   pretty yes yes f64  0    0    130  --    (no artifact)   130
#  exp_h2_marmap                     j0   map  mar  0.1   pretty yes yes f64  0    130  128  128   {}              128
#  exp_h2_stock_j1                   j1   samp mf#  0.1   stock  no  no  f64  0    130  128  128   {}              128
#  exp_h2_grammar_j1                 j1   samp mf#  0.1   pretty yes yes f64  0    130  128  128   {}              128
#  exp_oomfix_j0                     j0   samp mf#  0.1   pretty yes yes f64  0    0#   130  130   {}              130
#  exp_f64_ws_only_j1                j1   samp mf#  0.1   pretty no  no  f64  0    0#   130  130   {}              130
#  task_sudoku_mask                  mask samp mf   0.1   (run_tasks has no grammar flags)  250                    250
#  task_countdown_mask               mask samp mf   0.1   (run_tasks has no grammar flags)  250                    250
#  task_sudoku_j1_sample             j1   samp mf   0.1   (run_tasks has no grammar flags)  250                    250
#
# All twelve config rows above were re-read out of artifacts/*.json at 1aa8ccf,
# not copied from the previous script.
#
# * exp_marmap_b0.003's row is read from the QUARANTINED artifact
#   (artifacts/unattributable_dirty_tree/exp_marmap_b0.003.json).  Legitimate:
#   the quarantine is about the NUMBERS being unattributable, not about the
#   invocation, and the invocation is independently fixed by the tag.
#   exp_marmap_b1.0 has no artifact at all -- it was killed mid-run; its bundle
#   is the sweep's and its bound is its name.
#
# # WHAT AN ABSENT FIELD MEANS (the starred/hashed rows).  exp_f64_ws_only_j1,
#   exp_oomfix_j0, exp_h2_grammar_j1 and exp_h2_stock_j1 all ran on 2026-08-02
#   and store NO `confidence` and NO `temp` key; the first two store no `offset`
#   either.  Absence is "the flag did not exist yet", not "it ran at None" --
#   checked against git, not guessed:
#       --confidence  added 66042d9  2026-08-02 19:40:31
#       --temp        added e4e7144  2026-08-02 21:43:14
#       --offset      added 8c3be4c  2026-08-02 16:32:48
#   and the four artifacts have mtimes 11:57, 14:40, 18:46 and 19:52, each
#   BEFORE the flag missing from it.  Each new flag defaulted to the
#   pre-existing behaviour (`mf` was the only accept rule, `stock` the only
#   temperature schedule, offset 0 the only cut), so
#   `--confidence mf --temp stock --offset 0` reproduces them exactly.  The
#   artifacts written AFTER each landing do carry the key -- exp_mar_ctrl_mf
#   (20:03) has `confidence`, exp_h2_mfmap (2026-08-03) has `temp` -- which is
#   what makes "the field did not exist" a reading rather than a guess.
#
# --expand-wildcard is NEW at 15cdc47, defaults to True, and lands in the
# artifact as `expand_wildcard`.  Every published artifact has it as None
# ("this run predates the flag" = the broken grammar).  It is passed
# explicitly on every eval.run arm below and printed back out, because a
# grammar-shaping boolean that is invisible in the console log is precisely
# the defect the readback exists to catch.  eval/run_tasks.py has no such flag
# and needs none: sudoku and countdown are compiled from regexes, not schemas.
#
# --fence and --ci-enums are store_true with NO negated form, so False is
# expressible only by omission -- which is the other reason every arm prints
# its own flags back OUT OF THE ARTIFACT.
#
# ONE CONFOUND THAT CANNOT BE FLAGGED AWAY, beyond section 2's:
#   The 2026-08-08 scorer changes (e2be58c) postdate every Sudoku and Countdown
#   artifact.  The Sudoku `solved` / `solve_rate` FIELDS in the existing
#   artifacts are the OLD scorer (task_sudoku_mask stores solved=51, 0.204)
#   while docs/RESULTS.md's row is the rescore (0.012).  Compare a new Sudoku
#   arm against RESULTS.md, never against the artifact it replaces.  `cs` is
#   computed by compile.validate.Simulator, is scorer-independent, and is why
#   every hard stop below is written on `cs`.
#
# ---------------------------------------------------------------------
# 4. WHY --n 130 AND NOT --n 128, STILL
# ---------------------------------------------------------------------
# --n selects a PREFIX of the split before anything compiles.  --n 130 pins the
# same 130-record prefix every published arm used.  Pre-15cdc47 the build gate
# then refused 2 and the artifact reported n=128; post-fix it refuses 0 and the
# artifact reports n=130.  --n 128 would select a DIFFERENT 128-record prefix
# and lose records 128 and 129.  exp_h2_* is `--offset 130 --n 128` and its 128
# means something else entirely -- "all 128 of records 130..257 ran".  The two
# senses of 128 must never be pooled.
#
# ---------------------------------------------------------------------
# 5. ORDER, BY HOW BADLY THE CURRENT NUMBER MISLEADS
# ---------------------------------------------------------------------
#  1 task_sudoku_mask       The only row 2c9f9d9's own message names as
#                           INVALIDATED, and the failure is mechanical, not
#                           statistical: 249 of 250 emissions are missing the
#                           whole channel-header region because positions 0..15
#                           were dead and token 0 decodes to nothing.  It is
#                           also the queue's falsifiable hard stop (6C).
#  2 eval_..._mask_sample   The mask row of RESULTS.md's headline table and the
#                           left side of five significant McNemar pairs; its CS
#                           column is SPEC 2.8's headline evidence.  Also the
#                           first arm that can test the n=130 prediction on the
#                           real pipeline (6E).
#  3 task_countdown_mask    Confirm-don't-assume.  Countdown lost 5,432 live
#                           edges but 0 legal (position,token) pairs at the one
#                           measured p -- and sharpness is per record, so 0 on
#                           one p is not 0 on 250.  45 min to convert an
#                           assumption into a measurement.
#  4 task_sudoku_j1_sample  The da1294a decider.  Front-loaded out of order on
#                           purpose: its outcome decides whether
#                           task_sudoku_j0_sample and task_sudoku_j2 need 90
#                           more GPU-minutes, and that is worth knowing at hour
#                           4 rather than hour 20.  Not low-stakes either --
#                           "constrained to the grid the model solves 1.2%" is
#                           the whole 7.1 cross-task story.
#  5 the six mar points     A sweep is misleading as a SET.  Pre-fix a dead
#                           position returned H = -708 and was always accepted
#                           first, making acceptance nearly bound-INDEPENDENT --
#                           so the published flatness (0.6655 / 0.6899 / 0.6725
#                           across 1.7 decades) is exactly what a corrupted rule
#                           would produce, and is the single most likely
#                           artifact in RESULTS.md.  The endpoints 0.003 and 1.0
#                           are where signal would appear if there is any.
#  6 exp_h2_marmap          Same defect, second-half replication, and the one
#                           pair in this queue that is clean on both the
#                           grammar and the denominator axis (section 2).
#  7 the four da1294a arms  Stale for the older reason and already carrying a
#                           "do not cite" in docs/LOG.md, so they mislead least;
#                           each has provable Z==0 records to recover
#                           (5, 1, 1, 2 of 128/128/130/130).
#
# SPEC 3.4's grid is {0.003, 0.01, 0.03, 0.1, 0.3, 1.0}.  After this queue the
# mar sweep covers 0.003, 0.01, 0.1, 1.0 plus an off-ladder 0.5 -- 0.03 and 0.3
# are still missing, and entropy_threshold has never been swept at all.  Say so
# in the table; do not call this "the SPEC 3.4 sweep".
#
# WALL CLOCK, from these arms' own elapsed_seconds (b1.0 has none; estimated
# from b0.5.  b0.003's 123 min is from the quarantined artifact -- an
# unattributable NUMBER is still a usable stopwatch):
#   smokes 9 | sudoku_mask 45 | bfcl mask 84 | countdown_mask 45
#   | sudoku_j1 45 | marmap 0.003 123 + 0.01 102 + 0.1 102 + 0.5 103
#   + 1.0 ~103 | h2_marmap 81 | h2_stock_j1 66 | h2_grammar_j1 72
#   | oomfix_j0 76 | f64_ws_only_j1 105
#   = 1,161 min ~= 19.4 h.  The eight offset-0 bfcl arms now decode 2 more
#   records each (+~1.5 min) and compile those two schemas ~1.3x FASTER
#   (331.5->248.8 s and 370.1->280.5 s, 15cdc47's own measurement), so the
#   denominator change is roughly time-neutral.  Call it 19.5-21 h with compile
#   jitter and the fix's +2.3-14.8% on the mask/mar closure.  If the Sudoku
#   decider fails, +1.5 h for j0_sample and j2, queued separately.
#
# THE ONE ARM TO KILL IF THE GPU IS NEEDED: exp_f64_ws_only_j1, last for that
# reason.  Most expensive of the four da1294a leftovers (105 min), recovers the
# fewest facts (2 Z==0 records), and its configuration (pretty, no fence, no
# ci-enums, j1, sample) has no partner arm at the current gate, so it cannot
# enter a paired comparison with anything.  Dropping it costs one uncitable
# singleton and saves 1.75 h.
#
# ---------------------------------------------------------------------
# 6. THE HARD STOPS -- CODE, NOT PROSE
# ---------------------------------------------------------------------
#  A. Preflight.  (1) this file is not writable; (2) diffgemma_fa/ at HEAD is
#     IDENTICAL to 1aa8ccf's -- not `HEAD == 1aa8ccf`, see the code for why and
#     for the two commits that would have false-stopped on it; (3)
#     `git status --porcelain` empty; (4) GPU present and idle; (5) THE
#     DENOMINATOR PREMISE, checked on CPU in milliseconds before any GPU time:
#     records 117 and 122 are still at those indices, their build-gate
#     predicate under JSON_WS is clean with expand_wildcard=True, AND still
#     dirty with expand_wildcard=False.  The second half is what stops the
#     check being vacuous -- it is the same assertion that failed pre-15cdc47.
#     (6) artifacts/fa/bfcl/ absent or empty: `compile/tasks/bfcl.py` caches
#     automata as `{rec_id}__{fn_index}.npz` keyed by FILENAME, and although
#     `expand_wildcard` is in `schema_fingerprint` the fingerprint does not gate
#     that path.  eval/run.py does not read that directory today (grepped: the
#     only readers in the tree are the compile CLI that writes it and
#     tests/test_audit_partition.py, which reads artifacts/fa/countdown), so
#     this is a latent hazard, not a live one -- and it is asserted rather than
#     deleted because silently removing hours of compiled automata is the worse
#     failure.
#  B. Tree fingerprint over diffgemma_fa/**/*.py, re-checked before EVERY arm.
#     If an agent edits the tree mid-queue the queue stops rather than produce
#     another unattributable number.  HEAD is RECORDED, not enforced: the
#     previous queue died on a docs/LOG.md commit, and a docs commit cannot
#     change what is being measured, while a code commit necessarily changes a
#     .py mtime and trips the fingerprint whether or not it is committed.  Same
#     reasoning bounds the mid-queue dirty check to `diffgemma_fa/` and
#     `env.sh`; the preflight still demands the whole tree clean, per CLAUDE.md.
#     Every abort names WHICH condition tripped -- the old message printed only
#     `fingerprint X -> X` for a HEAD change, which reads like a bug in the
#     guard and cost real diagnosis time.
#  C. task_sudoku_mask must report cs >= 2, against a published 1 of 250.  The
#     queue's falsifiable prediction, in its cheapest form.  Pre-fix, 249 of 250
#     emissions look like '<|channel>124\n4321\n2413\n12422\n\n1' -- the entire
#     header-name region absent, because positions 0..15 had empty support and
#     `categorical` returned token 0 for all of them; the single accepted record
#     instead reads '<|channel>thought\n<channel|>4132\n2314\n3241\n1423<turn|>'.
#     That failure has ONE mechanical cause and the fix removes it, so a correct
#     mask closure cannot reproduce 1/250.  Threshold 2 is the weakest strictly
#     directional claim available: a false stop needs the fix to have changed
#     nothing at all.  If it fires, the remaining ~17 h are measuring something
#     other than this fix.
#  D. task_sudoku_j1_sample must satisfy cs + zero_partition == n.  On J1 a
#     record either satisfies the automaton or died with Z==0; published
#     250 + 0 = 250.  Anything else is a guarantee violation.  Written this way
#     rather than as `cs == 250` precisely so a legitimate Z==0 does not
#     false-stop it.
#  E. Every bfcl arm asserts its own (records_available, n) pair -- 130/130 for
#     the offset-0 arms, 128/128 for the offset-130 ones -- and requires
#     `expand_wildcard` to be true in the artifact.  This is 15cdc47's
#     prediction ("4,538 -> 4,549 schemas pass the gate") tested on the real
#     pipeline: if the gate still refuses those two records, n silently returns
#     to 128, every downstream comparison in section 2 is wrong, and the queue
#     must stop rather than publish it.  It also catches a mistyped --offset,
#     which is the most likely way to get a plausible artifact measuring the
#     wrong cut.
#  F. zero_partition: 0 is [ok]; nonzero but under 10% of n prints [!!] and
#     CONTINUES on the long arms; >= 10% of n stops.  On the n=4 smokes any
#     nonzero stops, because one record there is already 25%.  Post-2c9f9d9 the
#     widening floor's `rep` rides `feasible` -> ZeroPartitionError, and
#     post-1aa8ccf so does up_sweep's underflow flag, so this is a failure mode
#     that did not exist when these arms were first measured.  It is deliberately
#     NOT all-or-nothing on the long arms: one record above the bound is a
#     finding to report -- it already has its own column -- not a reason to
#     discard the other 127 and the 14 hours behind them.
#
# Every gate is idempotent: on a relaunch the arm is [skip]ped (only if its
# artifact exists AND parses, so a killed run re-runs) and the gate re-reads the
# artifact and passes.
#
# ---------------------------------------------------------------------
# 7. LAUNCH
# ---------------------------------------------------------------------
#   D=/tmp/claude-1000/-home-ubuntu-diffgemma-fa/29da7bd9-0534-4f90-a9d4-d456e1f98acd/scratchpad
#   cp $D/rerun_1aa8ccf.sh $D/rerun_1aa8ccf_LOCKED.sh
#   chmod 444 $D/rerun_1aa8ccf_LOCKED.sh
#   md5sum $D/rerun_1aa8ccf_LOCKED.sh
#   nohup $D/rerun_1aa8ccf_LOCKED.sh \
#         > /home/ubuntu/diffgemma_fa/logs/rerun_1aa8ccf.log 2>&1 &
#   then record the PID, the md5 and the log path in docs/LOG.md.
# The script refuses to run if it is writable: editing a live bash script
# corrupts the running instance, and that has already happened here.
# =====================================================================
set -u

# ---------- preflight A1: immutability ----------
# RESOLVE $0 BEFORE THE cd.  Caught by testing this block rather than reading
# it: with the immutability check placed after `cd /home/ubuntu/diffgemma_fa`,
# a relative invocation leaves $0 pointing at a path that no longer resolves,
# `[ -w ]` is false on a nonexistent file, and the guard silently passes on a
# WRITABLE script.  It has to be absolute, and it has to be here.
SELF=$(readlink -f "$0" 2>/dev/null || echo "")
if [ -z "$SELF" ] || [ ! -f "$SELF" ]; then
  echo "!! cannot resolve this script's own path from '$0'. The immutability"
  echo "!! check cannot be performed, so it is not being skipped. STOPPING."
  exit 1
fi
if [ -w "$SELF" ]; then
  echo "!! $SELF is writable. Launch from a chmod 444 copy -- editing a live"
  echo "!! bash script corrupts the running instance, and that has already"
  echo "!! happened on this box."
  exit 1
fi

cd /home/ubuntu/diffgemma_fa || exit 1

# `source env.sh` under `set -u` is safe: the venv activate script guards every
# optional variable, and two previous queues ran nine arms this way.  It sets
# PREALLOCATE=true / MEM_FRACTION=.90 on this dedicated box, so the queue must
# stay SERIAL -- two of these processes cannot coexist.
source env.sh

EXPECT_COMMIT=1aa8ccf634934dbbe4b5ad9820b24e77f2baaa19
QUEUE=rerun_1aa8ccf
ARCHIVE=artifacts/stale_pre_1aa8ccf

# ---------- preflight A2/A3: code identity and clean tree ----------
# NOT `HEAD == EXPECT_COMMIT`.  What every arm below is attributed to is the
# CONTENT of diffgemma_fa/, and a commit that does not touch it cannot change a
# number.  Written this way after watching it happen twice in eighteen hours:
# the predecessor queue died on a docs/LOG.md commit made one minute after
# launch, and while this script was being reviewed 4f9a6e3 ("Document the
# grammar's RFC 8259 gaps") landed on README.md and docs/LIMITATIONS.md --
# `git diff --name-only 1aa8ccf 4f9a6e3 -- diffgemma_fa/` is empty.  A HEAD
# equality check would refuse to launch on that, which is a false stop costing
# GPU-hours; this check passes it and records the real HEAD in every artifact.
# A code commit still stops the queue -- it changes a tracked file under
# diffgemma_fa/, so the diff below is non-empty (and, mid-queue, the edit
# changes an mtime before the commit even exists: hard stop B).
HEAD_AT_LAUNCH=$(git rev-parse HEAD 2>/dev/null || echo none)
if ! git diff --quiet "$EXPECT_COMMIT" HEAD -- diffgemma_fa/ 2>/dev/null; then
  echo "!! STOP (preflight A2): diffgemma_fa/ at HEAD ($HEAD_AT_LAUNCH)"
  echo "!! differs from $EXPECT_COMMIT, which is the commit every arm below is"
  echo "!! written against and whose exemptions were traced through the code:"
  git diff --name-only "$EXPECT_COMMIT" HEAD -- diffgemma_fa/ | sed 's/^/   /'
  echo "!! Re-review the arm set against the new code before launching."
  exit 1
fi
if [ "$HEAD_AT_LAUNCH" != "$EXPECT_COMMIT" ]; then
  echo "[note] HEAD is $HEAD_AT_LAUNCH, not $EXPECT_COMMIT, but diffgemma_fa/"
  echo "       is identical between them -- commits since are docs/tests only."
  echo "       Recorded in every artifact as git_commit; the code identity is"
  echo "       recorded as code_base_commit."
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "!! STOP (preflight A3): the working tree is dirty. An unattributable"
  echo "!! number is worse than no number, because it looks like evidence."
  git status --porcelain | sed 's/^/   /'
  exit 1
fi
HEAD_SEEN="$HEAD_AT_LAUNCH"

tree_fp () { find diffgemma_fa -name '*.py' -printf '%T@ %s %p\n' | sort | md5sum | cut -d' ' -f1; }
tree_cfp () { find diffgemma_fa -name '*.py' -print0 | sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1; }
FP0=$(tree_fp)
CFP0=$(tree_cfp)

# ---------- the mid-queue guard (hard stop B) ----------
# Named conditions.  The predecessor tested a three-way OR and printed only the
# fingerprint pair, so a HEAD change rendered as `fingerprint X -> X`.
check_tree () {
  local now cnow dirty code_dirty head_now stop=0
  now=$(tree_fp)
  dirty=$(git status --porcelain)
  head_now=$(git rev-parse HEAD 2>/dev/null || echo none)

  if [ "$now" != "$FP0" ]; then
    cnow=$(tree_cfp)
    echo "!! STOP -- CONDITION 1 of 2: THE CODE TREE MOVED UNDER THE QUEUE."
    echo "!!   test:  md5 of (mtime, size, path) over diffgemma_fa/**/*.py"
    echo "!!   launch: $FP0"
    echo "!!   now:    $now"
    if [ "$cnow" != "$CFP0" ]; then
      echo "!!   CONTENT also changed ($CFP0 -> $cnow): an agent edited"
      echo "!!   diffgemma_fa/ while this queue was measuring it."
    else
      echo "!!   content is unchanged ($cnow): the files were rewritten"
      echo "!!   byte-identically (checkout, formatter, touch). Still a stop --"
      echo "!!   a measurement must not straddle a write to the tree, and the"
      echo "!!   window between the write and the read is not auditable."
    fi
    stop=1
  fi

  # Scoped deliberately.  The preflight already demanded a fully clean tree;
  # mid-queue, only dirt that can change what is being measured is fatal.  A
  # docs/LOG.md write is exactly what killed the previous queue and it cannot
  # move a number.
  code_dirty=$(printf '%s\n' "$dirty" | grep -E '(diffgemma_fa/|env\.sh)' || true)
  if [ -n "$code_dirty" ]; then
    echo "!! STOP -- CONDITION 2 of 2: UNCOMMITTED CHANGES UNDER diffgemma_fa/"
    echo "!!   test:  git status --porcelain, filtered to diffgemma_fa/ and env.sh"
    printf '%s\n' "$code_dirty" | sed 's/^/     /'
    stop=1
  elif [ -n "$dirty" ]; then
    echo "[note] tree is dirty OUTSIDE diffgemma_fa/ -- not a stop, recorded:"
    printf '%s\n' "$dirty" | sed 's/^/       /'
  fi

  if [ "$head_now" != "$HEAD_SEEN" ]; then
    echo "[note] HEAD moved $HEAD_SEEN -> $head_now. NOT a stop: attribution"
    echo "       rides the .py fingerprint above, which a code commit cannot"
    echo "       avoid tripping (the edit changes an mtime before the commit"
    echo "       exists). Recorded into every artifact as git_head_at_write."
    HEAD_SEEN="$head_now"
  fi

  if [ "$stop" -ne 0 ]; then
    echo "!! Everything measured from here would be unattributable. STOPPING."
    exit 1
  fi
}

echo "=== $QUEUE at $(git rev-parse --short HEAD)  $(date -u '+%Y-%m-%d %H:%M') UTC"
echo "=== tree fingerprint $FP0  (content $CFP0)"
echo "=== DGFA_DEDICATED=${DGFA_DEDICATED:-unset}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv || { echo "!! nvidia-smi failed"; exit 1; }
echo

# ---------- preflight A6: the automaton cache hazard ----------
if [ -d artifacts/fa/bfcl ] && [ -n "$(ls -A artifacts/fa/bfcl 2>/dev/null)" ]; then
  echo "!! STOP (preflight A6): artifacts/fa/bfcl/ is non-empty."
  echo "!! Those .npz files are keyed by FILENAME ({rec_id}__{fn_index}.npz),"
  echo "!! not by schema_hash, so a pre-15cdc47 automaton for a wildcard schema"
  echo "!! would be reused verbatim under the fixed grammar. Delete the"
  echo "!! directory by hand (it is not deleted here -- silently discarding"
  echo "!! hours of compiled automata is the worse failure) and relaunch."
  ls -la artifacts/fa/bfcl | sed 's/^/   /'
  exit 1
fi
echo "[ok] artifacts/fa/bfcl/ absent or empty -- no stale automaton can be reused"

# ---------- preflight A5: the denominator premise, on CPU ----------
# 15cdc47's claim, tested rather than trusted, in milliseconds, before 19 hours
# of GPU time are spent on a queue whose every comparison assumes it.  The
# expand_wildcard=False leg is the anti-vacuity control: it is the assertion
# that FAILED before the fix, so a probe that cannot fail is a probe that has
# been broken by a refactor.
python3 - <<'PY' || exit 1
import sys
from diffgemma_fa.compile import bfcl_data, schema as S
from diffgemma_fa.compile.tasks.bfcl import ALLOW

WANT = {117: "live_simple_117-73-0", 122: "live_simple_122-78-0"}

def gate_bad(params, expand):
    """The build gate's own predicate, as pipeline.compile_json_schema runs it
    for eval/run.py's arms: --nonempty and --whitespace stock|pretty, so
    verify_strict is False and the deciding recompile is under JSON_WS with the
    caller's expand_wildcard. 'reordered-keys' is not a build failure on an
    ordered grammar (pipeline.py) and is filtered there too."""
    norm = S.normalize_bfcl_schema(params, expand_wildcard=expand)
    inst = S.synthesize_instance(norm)
    prepared = S.require_nonempty_strings(norm)
    rgx = S.build_regex(prepared, from_bfcl=False, allow=ALLOW,
                        allow_wildcard=True, whitespace_pattern=S.JSON_WS,
                        expand_wildcard=expand)
    _, bad = S.accepts_all_renderings(rgx, inst)
    return [b for b in bad if b != "reordered-keys"]

recs = list(bfcl_data.load(("BFCL_v4_live_simple.json",)))
fail = False
if len(recs) != 258:
    print(f"!! live_simple has {len(recs)} records, expected 258"); fail = True
for i, want in WANT.items():
    if i >= len(recs) or recs[i].id != want:
        got = recs[i].id if i < len(recs) else "<out of range>"
        print(f"!! index {i} is {got}, expected {want} -- the split moved, so"
              f" every offset/n in this queue means something else")
        fail = True
        continue
    p = recs[i].functions[0].get("parameters") or {}
    fixed, broken = gate_bad(p, True), gate_bad(p, False)
    print(f"[gate] {want}: expand=True bad={fixed}  expand=False bad={broken}")
    if fixed:
        print(f"!! {want} STILL fails the build gate with the wildcard fix on."
              f" Every arm below would report n=128, and section 2's whole"
              f" denominator argument is wrong.")
        fail = True
    if not broken:
        print(f"!! {want} passes the gate even WITHOUT the fix. The probe"
              f" cannot fail, so it is proving nothing -- do not read its"
              f" green as evidence.")
        fail = True
if fail:
    print("!! STOP (preflight A5): the n=130 premise is not established.")
    sys.exit(1)
print("[ok] preflight A5: both wildcard records compile clean at 1aa8ccf and"
      " still fail without --expand-wildcard (control non-vacuous)")
PY
echo

mkdir -p logs "$ARCHIVE" artifacts/unattributable_dirty_tree

# ---------- archiving ----------
# `logs/` is gitignored and `run` opens logs/<tag>.log with `>`, so an
# unarchived log is destroyed unrecoverably by its own re-run.  Every tag below
# has an existing log; all of them are moved with their artifact.  Idempotent:
# on a relaunch the archive copy exists and the original is left alone.
archive_to () {                # archive_to <dir> <tag>...
  local dir="$1"; shift
  local t
  for t in "$@"; do
    if [ -f "artifacts/$t.json" ] && [ ! -e "$dir/$t.json" ]; then
      mv "artifacts/$t.json" "$dir/$t.json" && echo "[archived] $t.json -> $dir"
    fi
    if [ -f "logs/$t.log" ] && [ ! -e "$dir/$t.log" ]; then
      mv "logs/$t.log" "$dir/$t.log" && echo "[archived] $t.log -> $dir"
    fi
  done
}

archive_invalidated () {
  if [ ! -e "$ARCHIVE/README.txt" ]; then
    cat > "$ARCHIVE/README.txt" <<'TXT'
Artifacts and console logs replaced by the rerun_1aa8ccf queue.

They are stale for one of two reasons, both recorded here so the directory is
not just a heap:

  2c9f9d9  every --variant mask and --confidence mar arm. prefix_suffix's
           per-vector normalisation erased structurally live edges; mask then
           sampled from an all-sentinel row (token 0, deterministically) and
           mar accepted every dead position first (H = -708.4 nats).
           -> eval_bfcl_live_simple_mask_sample, task_sudoku_mask,
              task_countdown_mask, exp_marmap_b{0.01,0.1,0.5}, exp_h2_marmap

  da1294a  four j0/j1 sample arms with provable pre-fix zero_partition records
           (5, 1, 1, 2), plus the Sudoku J1 decider.
           -> task_sudoku_j1_sample, exp_h2_stock_j1, exp_h2_grammar_j1,
              exp_oomfix_j0, exp_f64_ws_only_j1

Do not cite these. Their replacements in artifacts/ carry git_commit,
git_tree_fingerprint and queue fields; these do not. The offset-0 bfcl ones
also predate 15cdc47, so their n=128 (or their pre-gate n=130) is a different
denominator from their replacement's n=130 -- see docs/RESULTS.md.
TXT
    echo "[archived] wrote $ARCHIVE/README.txt"
  fi
  archive_to "$ARCHIVE" \
      eval_bfcl_live_simple_mask_sample task_sudoku_mask task_countdown_mask \
      exp_marmap_b0.01 exp_marmap_b0.1 exp_marmap_b0.5 exp_h2_marmap \
      task_sudoku_j1_sample exp_h2_stock_j1 exp_h2_grammar_j1 \
      exp_oomfix_j0 exp_f64_ws_only_j1
  # The two dirty-tree console logs.  Their artifacts are already quarantined
  # (or were never written -- b1.0 was killed mid-run); the logs are the only
  # surviving record of what those runs did, and the tags below would open them
  # with `>` and destroy them.
  local t
  for t in exp_marmap_b0.003 exp_marmap_b1.0; do
    if [ -f "logs/$t.log" ] && [ ! -e "artifacts/unattributable_dirty_tree/$t.log" ]; then
      mv "logs/$t.log" "artifacts/unattributable_dirty_tree/$t.log" \
        && echo "[archived] $t.log -> unattributable_dirty_tree"
    fi
  done
}

# ---------- the runner ----------
# The config is printed back OUT OF THE ARTIFACT, on the [skip] path too: on a
# relaunch the readback is still the only record of what actually ran, and a
# wrong flag that is invisible in the console log is exactly the defect this is
# here to catch.
show_cfg () {                  # show_cfg <artifact path>
  python3 - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
g = d.get
# run_tasks.py stores none of ws/fence/ci/dtype/offset/temp/prompt/nonempty/
# expand_wildcard/records_available, so the Sudoku and Countdown lines print
# None for all of them. That is "the field was never written", not "the arm ran
# at None".
print("    cfg: variant=%s emission=%s conf=%s bound=%s ws=%s fence=%s ci=%s "
      "expand_wildcard=%s dtype=%s seed=%s offset=%s temp=%s prompt=%s "
      "nonempty=%s think=%s commit=%s code_base=%s" % (
          g("variant"), g("emission"), g("confidence"), g("entropy_bound"),
          g("whitespace"), g("fence"), g("ci_enums"), g("expand_wildcard"),
          g("dtype"), g("seed"), g("offset"), g("temp"), g("prompt_style"),
          g("nonempty_strings"), g("think"), (g("git_commit") or "?")[:7],
          (g("code_base_commit") or "?")[:7]))
print("    n=%s avail=%s skips=%s cs=%s solved=%s zp=%s oom=%s schema=%s "
      "arg=%s (%s/%s) exact=%s  %.1f min" % (
          g("n"), g("records_available"), g("skipped_by_reason"), g("cs_rate"),
          g("solve_rate"), g("zero_partition"), g("oom"),
          g("schema_valid_rate"), g("arg_accuracy"), g("arg_correct"),
          g("arg_total"), g("exact_call_rate"),
          (g("elapsed_seconds") or 0) / 60))
for r in (g("skipped_records") or []):
    print("      excluded: %s (%s)" % (r.get("id"), r.get("reason")))
for r in (g("zero_partition_records") or []):
    print("      Z==0: %s" % (r.get("id") if isinstance(r, dict) else r))
for r in (g("oom_records") or []):
    print("      oom: %s (bucket %s)" % (r.get("id"), r.get("n_states_bucket")))
PY
  return 0
}

run () {                       # run <module> <tag> <flags...>
  local mod="$1" tag="$2"; shift 2
  local out="artifacts/${tag}.json"
  check_tree
  # [skip] requires the artifact to PARSE, not merely to exist: a run killed
  # while writing leaves a truncated file, and skipping that would publish a
  # half-artifact as a measurement.
  if [ -f "$out" ] && python3 -c 'import json,sys; json.load(open(sys.argv[1]))' \
       "$out" 2>/dev/null; then
    echo "[skip] $tag (exists and parses)"
    show_cfg "$out"
    return 0
  fi
  echo "=== $(date -u +%H:%M) $tag"
  echo "    $mod $*"
  python -m "$mod" --out "$out" "$@" > "logs/${tag}.log" 2>&1
  if [ ! -f "$out" ]; then
    echo "  !! $tag DIED -- no artifact"
    tail -20 "logs/${tag}.log" | sed 's/^/     /'
    return 1
  fi
  # Stamp attribution INTO the artifact so nothing here is ever unattributable
  # again.  Extra keys only; scripts/phase5_report.py reads named fields.
  # Written to a temp file and renamed: an interrupted in-place rewrite would
  # truncate an arm that already cost 80 GPU-minutes, and os.replace is atomic
  # on the same filesystem.
  QUEUE="$QUEUE" BASE="$EXPECT_COMMIT" LAUNCH="$HEAD_AT_LAUNCH" \
  FP="$FP0" CFP="$CFP0" \
  HEADNOW="$(git rev-parse HEAD 2>/dev/null || echo none)" \
  python3 - "$out" <<'PY'
import json, os, sys
p = sys.argv[1]
d = json.load(open(p))
d["git_commit"] = os.environ["LAUNCH"]          # HEAD when the queue started
d["code_base_commit"] = os.environ["BASE"]      # diffgemma_fa/ is identical to this
d["git_head_at_write"] = os.environ["HEADNOW"]  # may differ: a docs commit
d["git_tree_fingerprint"] = os.environ["FP"]
d["git_tree_content_hash"] = os.environ["CFP"]
d["queue"] = os.environ["QUEUE"]
tmp = p + ".stamping"
with open(tmp, "w") as f:
    json.dump(d, f)
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp, p)
PY
  show_cfg "$out"
  return 0
}

# ---------- gates ----------
field () {                     # field <tag> <key>  -> value or ERR
  python3 -c 'import json,sys
try:
    v = json.load(open(sys.argv[1]))[sys.argv[2]]
    print("ERR" if v is None else v)
except Exception:
    print("ERR")' "artifacts/$1.json" "$2" 2>/dev/null || echo ERR
}

# hard stop E: the denominator, per arm.  15cdc47's prediction on the real
# pipeline, and the cheapest possible check that --offset/--n did what was
# meant.  bfcl arms only (run_tasks writes no records_available).
require_n () {                 # require_n <tag> <expect_avail> <expect_n>
  local tag="$1" ea="$2" en="$3" avail n skips ew
  avail=$(field "$tag" records_available)
  n=$(field "$tag" n)
  ew=$(field "$tag" expand_wildcard)
  skips=$(field "$tag" skipped_by_reason)
  if [ "$avail" = ERR ] && [ "$n" = ERR ]; then
    echo "!! $tag: no readable artifact -- the arm died before writing one."
    echo "!! See logs/$tag.log (its tail is above). Nothing behind this can be"
    echo "!! attributed, so STOPPING rather than running the rest against a"
    echo "!! hole in the table."
    exit 1
  fi
  if [ "$avail" != "$ea" ] || [ "$n" != "$en" ]; then
    echo "!! $tag: records_available=$avail n=$n, expected $ea and $en."
    echo "!! skipped_by_reason=$skips"
    if [ "$avail" = "$ea" ]; then
      echo "!! The cut is right and the BUILD GATE refused records. 15cdc47"
      echo "!! predicts 0 refusals over all 4,549 live schemas, and preflight"
      echo "!! A5 confirmed it on records 117 and 122 minutes ago -- so this is"
      echo "!! a THIRD record, or the fix is not reaching eval/run.py."
    else
      echo "!! records_available is the size of the CUT, so --offset or --n is"
      echo "!! not what this arm was published at. Every comparison behind this"
      echo "!! would be against a different set of records."
    fi
    echo "!! STOPPING."
    exit 1
  fi
  if [ "$ew" != "True" ]; then
    echo "!! $tag: expand_wildcard=$ew in the artifact, expected True. The"
    echo "!! grammar is the pre-15cdc47 one and n=$n is a coincidence."
    echo "!! STOPPING."
    exit 1
  fi
  echo "[ok] $tag avail=$avail n=$n skips=$skips expand_wildcard=True"
}

# hard stop F, strict form: any nonzero stops.  Used on the n=4 smokes, where a
# single affected record is >= 25% of the arm and therefore already systematic.
require_zp0 () {
  local tag="$1" zp
  zp=$(field "$tag" zero_partition)
  if [ "$zp" != "0" ]; then
    echo "!! $tag zero_partition=$zp, predicted 0."
    echo "!! Two independent post-fix causes ride the feasible flag: 2c9f9d9's"
    echo "!! widening floor (viable-restricted misalignment <= 474 nats across"
    echo "!! 8 real grammars at T=0.4, against a 623.4-nat threshold) and"
    echo "!! 1aa8ccf's up_sweep underflow flag (0/256 positions on bfcl,"
    echo "!! countdown and sudoku at four temperatures). Neither should fire."
    echo "!! At n=4 a nonzero is systematic: the support is degraded and every"
    echo "!! mask/mar arm behind this would be measuring damaged marginals."
    echo "!! STOPPING."
    exit 1
  fi
  echo "[ok] $tag zero_partition=0"
}

# hard stop F, graded form: 0 is clean, a tail record is reported and kept,
# 10% or more of n stops the queue.  See section 6F for why the long arms are
# not all-or-nothing.
zp_gate () {
  local tag="$1" zp n lim
  zp=$(field "$tag" zero_partition)
  n=$(field "$tag" n)
  if [ "$zp" = ERR ] || [ "$n" = ERR ] || ! [ "$n" -gt 0 ] 2>/dev/null; then
    echo "!! $tag: zero_partition or n unreadable (zp=$zp n=$n). STOPPING."
    exit 1
  fi
  if [ "$zp" -eq 0 ]; then
    echo "[ok] $tag zero_partition=0 -- neither the widening floor nor the"
    echo "     up_sweep underflow flag fired"
    return 0
  fi
  lim=$(( (n + 9) / 10 ))
  if [ "$zp" -ge "$lim" ]; then
    echo "!! $tag zero_partition=$zp of n=$n, at or above the 10% systematic"
    echo "!! threshold ($lim). This is not a tail record: the two-factor"
    echo "!! validity bound is being exceeded across the corpus and every"
    echo "!! mask/mar arm behind this is measuring damaged support. STOPPING."
    exit 1
  fi
  echo "[!!] $tag zero_partition=$zp of n=$n, predicted 0. Below the 10%"
  echo "     systematic threshold, so the queue continues and the artifact"
  echo "     keeps its own column. Read zero_partition_records above: these are"
  echo "     records whose gamma_i exceeded the 623.4-nat two-factor bound, or"
  echo "     whose leaf lost a live entry to up_sweep's max normalisation."
  echo "     Report them by id in docs/RESULTS.md; do not average them away."
}

# ================= 0. the smokes, ~9 min =================
# Both repaired consumers, before the 19-hour tail.  0a is the mask branch
# (prefix_suffix -> scatter_edge_mass_to_tokens, sampler.py:502) at the stock
# grammar; 0b is the mar branch (prefix_suffix -> constrained_entropy_streamed,
# sampler.py:433) at the pretty+fence+ci grammar, whose larger |S| is where the
# 2048 bucket and the one historical OOM live.  A different call site each.
# Records 0..3 contain no wildcard schema, so these smoke the KERNELS, not the
# denominator -- that is preflight A5's job and hard stop E's.
run diffgemma_fa.eval.run smoke_1aa8ccf_mask \
    --task bfcl_live_simple --variant mask --emission sample \
    --confidence mf --entropy-bound 0.1 --whitespace stock --dtype float64 \
    --expand-wildcard --prompt-style stock --temp stock --seed 0 --offset 0 \
    --n 4 --max-new-tokens 256 --nonempty
require_zp0 smoke_1aa8ccf_mask
require_n   smoke_1aa8ccf_mask 4 4

run diffgemma_fa.eval.run smoke_1aa8ccf_mar \
    --task bfcl_live_simple --variant j0 --emission map \
    --confidence mar --entropy-bound 0.1 \
    --whitespace pretty --fence --ci-enums --dtype float64 \
    --expand-wildcard --prompt-style stock --temp stock --seed 0 --offset 0 \
    --n 4 --max-new-tokens 256 --nonempty
require_zp0 smoke_1aa8ccf_mar
require_n   smoke_1aa8ccf_mar 4 4

# Only now displace what is being replaced: a smoke failure must not leave
# twelve artifacts missing and scripts/phase5_report.py emitting a table with
# holes in it.
archive_invalidated
echo

# ================= 1. task_sudoku_mask, 45 min =================
# The row 2c9f9d9 names as INVALIDATED, and the queue's falsifiable stop.
run diffgemma_fa.eval.run_tasks task_sudoku_mask \
    --task sudoku --variant mask --emission sample \
    --confidence mf --entropy-bound 0.1 --n 250 --seed 0 --max-new-tokens 256

# ---- HARD STOP C ----
sk_cs=$(field task_sudoku_mask cs)
case "$sk_cs" in
  ''|*[!0-9]*)
    echo "!! task_sudoku_mask cs unreadable ($sk_cs). STOPPING."; exit 1 ;;
esac
if [ "$sk_cs" -lt 2 ]; then
  echo "!! task_sudoku_mask cs=$sk_cs of 250, predicted >= 2 (published 1)."
  echo "!! The 249 pre-fix failures had ONE mechanical cause -- positions"
  echo "!! 0..15 empty, categorical returning token 0 -- and the fix removes"
  echo "!! it. Reproducing 1/250 means the mask closure is still degenerate"
  echo "!! and the remaining ~17 GPU-hours would measure something else."
  exit 1
fi
echo "[ok] task_sudoku_mask cs=$sk_cs of 250 (published 1) -- the dead"
echo "     front-of-canvas positions are gone."
echo "     NOTE: compare solve_rate against docs/RESULTS.md's rescored 0.012,"
echo "     NOT against this artifact's predecessor (0.204, pre-e2be58c scorer)."
zp_gate task_sudoku_mask
echo

# ================= 2. eval_bfcl_live_simple_mask_sample, 84 min =================
# --whitespace stock is not a preference: it is what the other rows of that
# table were measured at, and mixing grammars across rows of one table breaks
# every paired comparison in it.  (Section 2 explains why this row is now at a
# different WILDCARD grammar from its partners regardless, and what to do about
# it -- which is a reporting decision, not a flag.)
run diffgemma_fa.eval.run eval_bfcl_live_simple_mask_sample \
    --task bfcl_live_simple --variant mask --emission sample \
    --confidence mf --entropy-bound 0.1 --whitespace stock --dtype float64 \
    --expand-wildcard --prompt-style stock --temp stock --seed 0 --offset 0 \
    --n 130 --max-new-tokens 256 --nonempty
require_n eval_bfcl_live_simple_mask_sample 130 130
zp_gate   eval_bfcl_live_simple_mask_sample
echo

# ================= 3. task_countdown_mask, 45 min =================
run diffgemma_fa.eval.run_tasks task_countdown_mask \
    --task countdown --variant mask --emission sample \
    --confidence mf --entropy-bound 0.1 --n 250 --seed 0 --max-new-tokens 256
zp_gate task_countdown_mask
echo

# ================= 4. task_sudoku_j1_sample, 45 min -- THE DECIDER ==========
# da1294a, not 2c9f9d9: j1+sample goes joint_draw -> up_sweep_log, which reads
# class_weights.  Published cs 250/250, solved 3/250, zero_partition 0.
run diffgemma_fa.eval.run_tasks task_sudoku_j1_sample \
    --task sudoku --variant j1 --emission sample \
    --confidence mf --entropy-bound 0.1 --n 250 --seed 0 --max-new-tokens 256

# ---- HARD STOP D ----
sj_cs=$(field task_sudoku_j1_sample cs)
sj_zp=$(field task_sudoku_j1_sample zero_partition)
sj_n=$(field task_sudoku_j1_sample n)
if [ "$sj_cs" = ERR ] || [ "$sj_zp" = ERR ] || [ "$sj_n" = ERR ]; then
  echo "!! task_sudoku_j1_sample fields unreadable. STOPPING."; exit 1
fi
if [ $((sj_cs + sj_zp)) -ne "$sj_n" ]; then
  echo "!! task_sudoku_j1_sample cs=$sj_cs zp=$sj_zp n=$sj_n."
  echo "!! On J1 a record either satisfies the automaton or died with Z==0."
  echo "!! cs + zero_partition != n is a GUARANTEE VIOLATION. STOPPING."
  exit 1
fi
echo "[ok] task_sudoku_j1_sample cs=$sj_cs zp=$sj_zp n=$sj_n"
echo "     DECIDER: published cs 250/250, solved 3/250 (0.012, and that one"
echo "     agrees with RESULTS.md's rescore). If solved is within seed noise"
echo "     of 3/250, task_sudoku_j0_sample and task_sudoku_j2 are CLEARED and"
echo "     need no GPU time. If it is not, queue all three -- roughly 1.5 h."
echo "     That call is a human one; this queue does not make it."
echo

# ================= 5. the mar sweep, ~10.2 h =================
# Emission unaffected (joint_map), but acceptance AND stopping route through
# prefix_suffix -> constrained_entropy_streamed, where a dead position returned
# H = -708.4 and was sorted first and always accepted.  Fresh baseline, not a
# rescore: the accept mask itself moved.
# --whitespace pretty --fence --ci-enums reproduces the published bundle, which
# is also the bundle of the mf control these arms exist to be compared against
# (exp_e5_grammar130_map -- numerically exempt, denominator-confounded, see
# section 2).
MAR="--task bfcl_live_simple --variant j0 --emission map --confidence mar \
--whitespace pretty --fence --ci-enums --dtype float64 --expand-wildcard \
--prompt-style stock --temp stock --seed 0 --max-new-tokens 256 --nonempty"

# shellcheck disable=SC2086
run diffgemma_fa.eval.run exp_marmap_b0.003 $MAR --entropy-bound 0.003 --offset 0   --n 130
require_n exp_marmap_b0.003 130 130
zp_gate   exp_marmap_b0.003
# shellcheck disable=SC2086
run diffgemma_fa.eval.run exp_marmap_b0.01  $MAR --entropy-bound 0.01  --offset 0   --n 130
require_n exp_marmap_b0.01 130 130
zp_gate   exp_marmap_b0.01
# shellcheck disable=SC2086
run diffgemma_fa.eval.run exp_marmap_b0.1   $MAR --entropy-bound 0.1   --offset 0   --n 130
require_n exp_marmap_b0.1 130 130
zp_gate   exp_marmap_b0.1
# shellcheck disable=SC2086
run diffgemma_fa.eval.run exp_marmap_b0.5   $MAR --entropy-bound 0.5   --offset 0   --n 130
require_n exp_marmap_b0.5 130 130
zp_gate   exp_marmap_b0.5
# shellcheck disable=SC2086
run diffgemma_fa.eval.run exp_marmap_b1.0   $MAR --entropy-bound 1.0   --offset 0   --n 130
require_n exp_marmap_b1.0 130 130
zp_gate   exp_marmap_b1.0
# shellcheck disable=SC2086
run diffgemma_fa.eval.run exp_h2_marmap     $MAR --entropy-bound 0.1   --offset 130 --n 128
require_n exp_h2_marmap 128 128
zp_gate   exp_h2_marmap
echo

# ================= 6. the da1294a residue, ~5.3 h =================
# NOT affected by 2c9f9d9 -- these are j0/j1 sample arms whose emission is
# joint_draw -> up_sweep_log and whose accept rule is mf.  They are here
# because da1294a's complement-algebra fix is not in them: each has provable
# pre-fix zero_partition records to recover.  Predicted 0 for all four.
# No zp gate: these run last, so stopping here buys nothing, and a residual
# Z==0 is a finding to write up rather than a reason to abandon artifacts
# already on disk.  The check prints instead.
zp_report () {                 # zp_report <tag> <published zp>
  local tag="$1" pub="$2" zp
  zp=$(field "$tag" zero_partition)
  if [ "$zp" = "0" ]; then
    echo "[ok] $tag zero_partition=0 (published $pub) -- da1294a confirmed"
  else
    echo "[!!] $tag zero_partition=$zp, published $pub, predicted 0."
    echo "     da1294a did not close it. Write this up in docs/LOG.md."
  fi
}

run diffgemma_fa.eval.run exp_h2_stock_j1 \
    --task bfcl_live_simple --variant j1 --emission sample \
    --confidence mf --entropy-bound 0.1 --whitespace stock --dtype float64 \
    --expand-wildcard --prompt-style stock --temp stock --seed 0 --offset 130 \
    --n 128 --max-new-tokens 256 --nonempty
require_n exp_h2_stock_j1 128 128
zp_report exp_h2_stock_j1 5

run diffgemma_fa.eval.run exp_h2_grammar_j1 \
    --task bfcl_live_simple --variant j1 --emission sample \
    --confidence mf --entropy-bound 0.1 \
    --whitespace pretty --fence --ci-enums --dtype float64 \
    --expand-wildcard --prompt-style stock --temp stock --seed 0 --offset 130 \
    --n 128 --max-new-tokens 256 --nonempty
require_n exp_h2_grammar_j1 128 128
zp_report exp_h2_grammar_j1 1

run diffgemma_fa.eval.run exp_oomfix_j0 \
    --task bfcl_live_simple --variant j0 --emission sample \
    --confidence mf --entropy-bound 0.1 \
    --whitespace pretty --fence --ci-enums --dtype float64 \
    --expand-wildcard --prompt-style stock --temp stock --seed 0 --offset 0 \
    --n 130 --max-new-tokens 256 --nonempty
require_n exp_oomfix_j0 130 130
zp_report exp_oomfix_j0 1

# LAST ON PURPOSE. See "THE ONE ARM TO KILL" in section 5: 105 min to recover
# two Z==0 records on an arm that cannot be compared to the row it replaces and
# has no paired partner. Kill this one first if the GPU is wanted back.
run diffgemma_fa.eval.run exp_f64_ws_only_j1 \
    --task bfcl_live_simple --variant j1 --emission sample \
    --confidence mf --entropy-bound 0.1 --whitespace pretty --dtype float64 \
    --expand-wildcard --prompt-style stock --temp stock --seed 0 --offset 0 \
    --n 130 --max-new-tokens 256 --nonempty
require_n exp_f64_ws_only_j1 130 130
zp_report exp_f64_ws_only_j1 2

check_tree
echo
echo "ALL DONE $(date -u '+%Y-%m-%d %H:%M') UTC"
echo "code base $EXPECT_COMMIT (diffgemma_fa/ identical); HEAD at launch"
echo "$HEAD_AT_LAUNCH, HEAD now $(git rev-parse HEAD)"
echo "tree fingerprint still $FP0"
echo
echo 'STILL INVALID AFTER THIS QUEUE -- do not cite:'
echo '  task_sudoku_j0_sample, task_sudoku_j2   (da1294a; gated on the'
echo '    task_sudoku_j1_sample decider above -- read its [ok] line)'
echo '  exp_oomfix_j1, exp_f64_grammar130_s1    (da1294a, j1/sample; not in'
echo '    this queue)'
echo '  exp_e5_grammar130_j0, exp_e5_grammar130_s1  (float32; NOT reproducible'
echo '    -- --dtype float32 now raises for sample, mask and mar. Withdraw'
echo '    them rather than re-running.)'
echo '  smoke_after_zdetector, smoke_post_da1294a, smoke_2c9f9d9_mask'
echo '    (n=4/8 smokes, never cited)'
echo '  exp_e{0,2,4,5,6}_*, exp_mar_ctrl_mf (n=30), phase4_e2e_sample,'
echo '    phase5_{calibrate_j1,sweep_fixed,final,header,nonempty,ws,diag_j1}'
echo '    -- the phase5 set is what calibrated entropy_bound=0.1, which every'
echo '    arm above inherits.'
echo
echo 'CPU FOLLOW-UP, before regenerating docs/RESULTS.md:'
echo '  * THE DENOMINATOR. Every arm above ran at the FIXED wildcard grammar;'
echo '    every arm NOT above ran at the broken one. The offset-0 bfcl arms in'
echo '    this queue now report n=130 where they published 128 (or a pre-gate'
echo '    130 whose records 117 and 122 were scored against a grammar that'
echo '    cannot emit a well-formed object). Discriminate on'
echo '    records_available, never on n. Before pairing a re-run row against'
echo '    an exempt one -- mask vs j0/j1/j2, the mar sweep vs'
echo '    exp_e5_grammar130_map -- either restrict both to the 128 common'
echo '    records using the per-record `rows` list in each artifact, or re-run'
echo '    the partner (~6.4 h for the five headline arms, 103 min for the mar'
echo '    control). Do not average two grammars into one table.'
echo '  * exp_h2_marmap vs exp_h2_mfmap is the one pair that needs none of'
echo '    that: records 130..257 contain zero wildcard schemas, so the grammar'
echo '    is identical and both are records_available=128.'
echo '  * scripts/phase5_report.py reads each artifact STORED accuracy fields'
echo '    and does not apply the 2026-08-08 rescoring, so the exempt rows still'
echo '    hold pre-audit numbers while every row above holds post-audit ones.'
echo '    Rescore the exempt artifacts from their own rows first, or the'
echo '    regenerated table silently mixes two scorers.'
echo '  * The Sudoku and Countdown solve_rate FIELDS in every pre-2026-08-08'
echo '    artifact are the old scorer. RESULTS.md carries the rescore. Compare'
echo '    new arms against RESULTS.md, not against the artifact they replace.'
echo '  * The mar sweep now covers 0.003 / 0.01 / 0.1 / 0.5 / 1.0. SPEC 3.4'
echo '    asks for 0.003 / 0.01 / 0.03 / 0.1 / 0.3 / 1.0 crossed with three'
echo '    entropy_threshold values on a 100-example dev slice. 0.03, 0.3 and'
echo '    the whole threshold axis are still unmeasured, and 0.5 is not on the'
echo '    ladder. Say so in the table rather than calling it the sweep.'
echo '  * n=128 means two different things: the pre-fix offset-0 arms (130'
echo '    minus two gate-refused records) and the offset-130 cut (all 128 ran).'
echo '    Do not pool them.'
