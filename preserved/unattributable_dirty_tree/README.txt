Artifacts written from a working tree that was being edited mid-run.

`exp_marmap_b1.0` was launched 01:18:52 while a coder was rewriting
`infer/scans.py` (mtime 01:32:17, i.e. DURING the run) and had already
modified `infer/marginals.py` (mtime 00:59:02). The run was killed before it
wrote an artifact.

`exp_marmap_b0.003` and its smoke completed earlier in the same queue but are
quarantined for the same reason: they cannot be attributed to any commit, and
--confidence=mar consumes `u = a*b` from the very function being rewritten.

Do not cite. Do not pool with the b0.01/0.1/0.5 sweep points. The whole
five-point sweep must be re-run after the prefix_suffix fix lands, because
dead positions returned H = -708 nats, which `_accept_from_entropy` sorts
first and always accepts -- a directional corruption of `mar` acceptance.
