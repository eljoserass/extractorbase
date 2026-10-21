# Working in this repository

Keep the extraction methods independent of experiment orchestration. A method
accepts examples and a task specification, saves a portable artifact, and runs
inference without a synthesis client. Put dataset splits and benchmark matrices
under `experiments/`; keep clinical guidance in the task specification or an
explicit experiment prompt.

Write code for the next person reading it:

- Prefer descriptive names, explicit branches and small named functions.
- Use ordinary typed Python. Avoid dynamic tricks when a simple helper suffices.
- Separate model transport, learning decisions and inference when their concerns
  differ. Avoid mixing shell commands into rule or prompt construction.
- Make defaults, dependencies, selection metrics and data exposure explicit.
  Never silently substitute a different model or fall back from dev to train.
- Reuse dependency APIs and prompts. Do not edit installed packages or globally
  monkeypatch them. Document any local workaround and its reason.
- Keep existing artifacts loadable or explicitly version a breaking change.
- Test meaningful behavior: offsets, data separation, candidate acceptance,
  serialization and isolated inference. Avoid tests that merely restate code.

Preserve unrelated changes, private data, run artifacts and notebooks. Do not
commit private inputs or generated clinical outputs. Run focused checks before
starting expensive experiments; record the configuration and source snapshot.

Treat learned rules, lexicons and artifacts as private when fitted on private
data. Public result documentation may contain global metrics, aggregate counts,
configuration and resource measurements. Keep note excerpts, per-document
results, annotation IDs, private hashes, prompts and model responses local.
Use explicitly synthetic examples in public tests and documentation.
