Artifacts and console logs replaced by the json_headline queue.

Stale for ONE reason: 15cdc47 (`type: any` -> a parenthesised anyOf). Both of
these arms are docs/RESULTS.md headline rows and both were measured PRE-GATE at
records_available=130 with no skips, so both INCLUDE live_simple_117-73-0 and
live_simple_122-78-0 scored under a grammar that rejects the correct object and
accepts a bare `1`. The constrained arm's single schema-validity failure IS
live_simple_122-78-0.

Their replacements in artifacts/ carry git_commit, code_base_commit,
git_tree_fingerprint and queue fields; these do not.

Two further reasons not to cite the numbers stored in these files even as a
baseline:
  * the stored accuracy fields predate the 2026-08-08 scorer (e2be58c).
    docs/RESULTS.md carries the rescore, and the rescore is exactly reproducible
    on CPU from each file's own `rows` list.
  * the replacement ran at HEAD, and four commits have touched the j0/map
    emission path since 2026-07-31 (da1294a, b58fd84, a41bf74, 1aa8ccf). The
    delta is therefore not attributable to the wildcard fix alone.
