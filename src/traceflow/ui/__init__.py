"""The delivery UI (plan.md §29, §30, §31, §61).

A local, read-only view over what a session already recorded. It computes nothing the
engine has not already concluded and stores nothing of its own: every page is rendered
from the artifacts on disk, so the dashboard can never disagree with `traceflow impact`.

Three constraints shape the package.

**It is a view, not a second analysis engine.** The moment the UI derived its own impact
or its own change set, the two could drift and the artifact would stop being the source
of truth (plan.md §35). It reads `session.json`, `baseline.json`, `changes.json`,
`symbols.json` and `impact.json` and renders them.

**It shows file contents, so it is the second place the secret policy applies.** Only the
diff view reads bytes that were not already read during analysis, and it refuses
sensitive paths outright rather than redacting them (plan.md §26).

**It serves one repository's sessions and nothing else.** The only paths it will ever
read are those named in a recorded session's own artifacts, so a request cannot reach
outside the session it belongs to (plan.md §46).
"""
