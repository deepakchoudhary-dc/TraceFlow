# TraceFlow

**Local-first, agent-independent change intelligence for AI-assisted software development.**

TraceFlow watches a repository while an AI coding agent works in it, notices when the
agent has stopped, and records what happened as a session you can inspect.

> The coding agent is the actor. TraceFlow is the observer.

---

## What this is, and what it is not

TraceFlow is **not** another coding agent, a generic codebase graph, or an
architecture-diagram generator. It does not read the agent's logs, its hooks, or its
telemetry, and it does not care which agent you use — Claude Code, Codex, Cursor,
Aider, a custom tool, or your own hands. The repository is the source of truth.

The question TraceFlow exists to answer is:

> What actually changed in my codebase, what does that change affect, and what
> evidence do we have about the resulting state of the system?

**Everything runs on your machine.** No account, no upload, no network calls, and no
LLM in the analysis engine.

---

## Status

**Phases 0–8 complete, and Phase 10 (TypeScript/JavaScript) is in.**

TraceFlow currently:

- discovers the git repository containing any path
- fingerprints the working tree and detects activity from it
- waits for the repository to settle before concluding anything
- records each settled episode as a session, with the baseline it started from
- reports precisely what changed during that session — and keeps changes that were
  already present out of the answer
- resolves that down to **symbols**: which functions, methods and classes were added,
  removed, re-declared, or rewritten
- analyses **TypeScript, TSX and JavaScript** beside Python: ES modules, CommonJS
  `require`, `import type`, dynamic `import()`, JSX, arrow functions, class members —
  with import resolution across relative paths, directory indexes and tsconfig
  `paths`/`baseUrl` aliases
- builds the repository's import graph on demand, in every supported language
- walks that graph from the changed symbols to everything they **reach**, separating
  obligations from possibilities and saying where static analysis is blind
- records the **task you gave the agent** and compares it with what actually changed —
  in careful words: *potentially related*, *potential scope expansion*, *no detected
  relationship* — never a verdict on correctness
- runs the tests a session reaches, when and only when the repository's configuration
  enables it, and records the result as evidence about the run — never as "the change
  is safe"
- serves a **local delivery view** of that session, with drill-downs to the graph, the
  diff, the evidence and the session record
- draws a **focused change map** of that session — a laid-out diagram of the change and
  what it reaches, with the before/after comparison beside it — and exports it to
  Excalidraw, JSON or SVG

Phase 9 (an optional AI explanation layer) and further languages are post-MVP.

---

## Quickstart

Requires Python 3.10 or newer and `git` on your `PATH`.

```bash
pip install -e ".[dev]"          # or: pip install -e .
traceflow init /path/to/repo     # prepare a repository
traceflow watch /path/to/repo    # observe until you press Ctrl-C
```

Then let your coding agent work. When it pauses, TraceFlow notices and records a
session:

```
TraceFlow 0.1.0
────────────────────────────────────────────────────────────────────
  Repository    shop-app
  Path          D:\Projects\shop-app
  Status        ● Watching
  Baseline      abc1234 — clean
  Quiet period  8s
────────────────────────────────────────────────────────────────────
Waiting for activity...

  ● activity detected — observing
  ◐ repository quieting…
  ✓ session 2026-09-25T14-52-43-9b3625 stabilised — 4 file(s), +57/-12
      1 pre-existing change(s) excluded from this session
      2 signature change(s), 1 body change(s), +1 import(s)
      impact: 3 needing review, 4 indirect, 2 test, 1 potential
```

Inspect what has been recorded:

```bash
traceflow status /path/to/repo     # repository and watcher state
traceflow sessions /path/to/repo   # list recorded sessions
traceflow graph /path/to/repo      # build and summarise the import graph
traceflow impact /path/to/repo     # what the last session's changes reach
traceflow analyze /path/to/repo    # analyse now, without waiting for quiescence
traceflow intent --task "..." /p   # attach the agent's task to a recorded session
traceflow ui /path/to/repo         # serve the delivery view on localhost
traceflow export /path/to/repo     # write the change map for another tool
```

`traceflow analyze --intent "Add rate limiting to the login endpoint."` records the task
at analysis time. `traceflow intent` attaches one afterwards — for a session the watcher
recorded live while you were away.

`traceflow init` creates `.traceflow/` for its own state, writes a `.traceflow.toml`,
and appends `.traceflow/` to the repository's `.gitignore`. That last step matters: the
state directory will hold diffs, so leaving it committable would risk committing a diff
of a `.env` file. `init` only ever appends to `.gitignore` and never rewrites it.

---

## How the watcher works

Three decisions shape the implementation, and each one is a deliberate trade.

**Git is the source of truth; the watcher only decides when to look.** TraceFlow
fingerprints the working tree by hashing `git status` output. A changed fingerprint
means activity. Because git holds the complete truth, a missed signal costs latency
and never correctness — which is what makes the watcher robust enough to be boring.

**Polling, not filesystem events.** `watchdog` would offer sub-second latency, but the
quiet period below makes sub-second latency worthless, and on Windows its
`ReadDirectoryChangesW` buffer can overflow and silently drop events under exactly the
burst load an AI agent produces. Polling also removes the need for a hand-maintained
ignore list, because `git status` already applies the repository's real `.gitignore`.
A filesystem-event source can be added later behind the same interface.

**Stability is a state machine, not a timer.** An agent does not make one edit; it makes
many, interleaved with test runs and failed attempts. Analysis only begins once the
repository has been untouched for the configured quiet period:

```
IDLE → ACTIVE → QUIETING → STABLE
```

Any activity during `QUIETING` restarts the clock, so a burst is one session rather
than one session per edit.

### How symbol analysis works

Files are not the useful unit. Knowing that `auth/service.py` changed tells you
nothing; knowing that the *signature* of `authenticate` changed tells you every caller
has to be re-examined. So every class, function and method is fingerprinted twice:

| Fingerprint | Covers | If it changes |
|---|---|---|
| **Signature** | name, parameters, defaults, annotations, decorators, base classes | the symbol is called differently — every caller is affected |
| **Body** | everything else inside it | what it does changed; no caller needs editing |

Both are computed from the syntax tree rather than the source text, so reformatting a
file, reindenting a block, or editing a comment registers as **nothing at all**. Only a
change to the program's structure counts. That is the difference between a tool you can
leave running and one that cries wolf every time a formatter touches a file.

Nested definitions are excluded from a parent's body fingerprint, so a change to a
method is reported once — against the method — rather than again against the class
containing it.

The declaration is stored alongside its fingerprint, so a reported signature change can
be *read* rather than taken on trust: `authenticate` was `(user, password)`, now
`(user, password, mfa=False)`.

Symbol comparison runs against the **baseline snapshot**, not the last commit. A file
that was already modified when the session began has no committed version matching what
was on disk, so comparing against the commit would credit this session with the earlier
edit — the file-level mistake of §14, one layer down.

### The dependency graph

`traceflow graph` parses every Python file and resolves the imports between them:

```
TraceFlow 0.1.0 — dependency graph
────────────────────────────────────────────────────────────────────
  Repository    shop-app
  Modules       4
  Import edges  1
  Unresolved    0

  Most imported
         1  auth/service.py
```

A file can be importable under more than one name (`src/traceflow/cli.py` is both
`src.traceflow.cli` and `traceflow.cli`), so both are registered rather than guessing
which layout a project uses. Imports that point outside the repository are recorded as
unresolved rather than dropped — silently discarding them is how a graph becomes
confidently incomplete.

Parsing is cached by content hash, so the second build re-parses only what changed.
Inside a watch loop the graph is built once and reused until a session changes the
repository's shape — a file added, removed or renamed, or an import added or dropped.
Building it parses every Python file, so doing it once per session rather than once per
run is the difference between a cost that is paid once and one that scales with the
repository on every quiet period.

### How impact analysis works

Knowing that `authenticate` changed is not the same as knowing what that costs you.
`traceflow impact` walks the import graph outward from the session's changed symbols
and reports what it reaches:

```
TraceFlow 0.1.0 — impact
────────────────────────────────────────────────────────────────────
  Session   2026-09-25T15-04-11-c4f19a
  Analyzer  python-1
  Changed   4 file(s) -> 7 file(s) impacted
  Review    3 node(s) requiring re-examination
  Depth     2

  STRUCTURE
      +1 module(s), -0 module(s), +2 relationship(s), -0 relationship(s)
      + auth/ratelimiter.py
      + auth/routes.py -> auth/ratelimiter.py

  DIRECT  (2)
      auth/service.py::authenticate  symbol_signature_changed (confirmed)
      auth/ratelimiter.py            file_added (confirmed)
  INDIRECT  (2)
      auth/routes.py::login  signature_changed (confirmed)  via auth/service.py
      auth/legacy.py         dangling_import (confirmed)
  TESTS  (1)
      tests/test_auth.py::test_login  signature_changed (confirmed)  via auth/service.py
  POTENTIAL  (2)
      auth/admin.py::admin_login  indirect_dependency (confirmed)  via auth/service.py -> auth/routes.py
```

**A signature change and a body change propagate differently.** This is the point of
fingerprinting symbols twice. If a symbol's *declaration* changed, every caller must be
re-examined, so those callers are reported as `INDIRECT` — an obligation. If only its
*body* changed, callers need no edit, though their behaviour may differ, so those callers
are reported as `POTENTIAL` — not an obligation. Collapsing the two into "affected" would
throw away the only distinction static analysis can make with confidence.

**Impact attenuates with distance.** At the first ring the classification reflects the
change itself. Past that, a node is reached through something that was merely *affected*
rather than changed, so its own signature never moved and its callers cannot be required
to act. Everything beyond the first ring is therefore a possibility, and the report says
so instead of inflating the count.

**Every node carries evidence.** `traceflow impact --json` emits, for each node, the
chain of files the traversal followed and the call site it went through. A node is not an
assertion that two components are connected; it is a location you can open.

**A symbol is reported once, with everything found about it.** A session can do two things
to the same symbol: change it, *and* reach it because something it calls moved. The node
keeps its own reason and carries the second as a `reaches` entry, so `auth/routes.py::login`
is both "body changed" and "calls a changed declaration". Two rows for one symbol would
invite you to count two components and would double-count it in every total; keeping only
the first is how a session with three call sites of a moved declaration came to report one
caller, with a list beneath the count that disagreed with it.

**The report states where it is blind.** `getattr`, `eval`, `importlib` and wildcard
imports defeat static analysis. They are collected into a `LIMITATIONS` section rather
than silently producing a smaller, more confident-looking graph (plan.md §70).

The walk is bounded by `analysis.impact_max_depth` (default 5). When the limit stops it,
the report records `truncated` rather than presenting a short walk as a complete one.

### The delivery view

`traceflow ui` serves a local dashboard for a recorded session on `127.0.0.1`:

```
TraceFlow delivery — shop-app
  http://127.0.0.1:8765/
  Ctrl-C to stop.
```

The first screen is plan.md §61's list — Session, Changes, Impact, Tests, Dependencies,
Potential concerns — followed by four drill-downs: **View Graph**, **View Diff**,
**View Evidence**, **View Session**. Sessions are listed at `/sessions`, so the history
from §36 is navigable without Git.

Four properties are worth stating, because each is a decision rather than an accident.

**It is a view, not a second engine.** Every page is rendered from the artifacts on disk.
Nothing is recomputed, so the dashboard cannot disagree with `traceflow impact`, and the
session record stays the source of truth. A test asserts that rendering leaves
`impact.json` byte-identical.

**It listens on the loopback interface and nothing else.** This serves the contents of
your source tree, so there is no `--host` option to get wrong. The only paths it will ever
read are the ones a recorded session named — no route takes a path from the request and
opens it, so a traversal attempt simply fails to match and 404s. That is a stronger
guarantee than sanitising input, and it is tested by attempting it.

**It shows file contents, so the secret policy applies a second time.** `View Diff` is
the only place bytes are read that analysis did not already read. A sensitive path is
refused outright — no redaction, because a file that was never read cannot leak.

**It says what it does not know.** When the repository has not enabled test execution,
the Tests section says exactly that, and still shows which tests the change reaches —
not whether they pass. Dependencies states that manifests are not parsed yet (§24), and
shows what *is* known — the relationships that moved — rather than implying a verdict
that phase has not been built. When a task was recorded, the Task section says plainly
that it compares wording, not correctness.

A diff is measured against the session's own baseline, never `HEAD`, for the same reason
the change set is. If the file has moved on since the session was recorded, the diff says
so: a diff that silently folds a later edit into a session's record is exactly the quiet
misattribution this project exists to prevent.

### The change map

`View Graph` opens on two diagrams, drawn as SVG in the page itself — no canvas, no
graph library, nothing that runs in your browser.

The **change map** is the session's own neighbourhood: every file it changed on the left,
and everything the change reaches to the right, with an arrow for each step the impact
walk actually took. The **before and after** diagram is plan.md §28's comparison: the
import relationships the session added and removed, and the modules they connect.

Five things are decisions rather than defaults.

**Colour is never the only indicator.** Each box carries its state as a word and a glyph —
`+ added`, `~ modified`, `- removed`, `→ affected`, `· unchanged` — and the legend repeats
them. A reader who cannot separate the red from the green reads the same map.

**Every arrow is a record.** A map is persuasive, so a line drawn because two files look
related would be the assertion this product exists to replace. Each edge is reconstructed
from the traversal's recorded chain, and carries the evidence of the step it draws: the
call that reaches back, or the import that no longer resolves. Hover a box to see why it
is there.

**One arrow, one meaning.** The change map runs in the impact direction ("a change in A
reaches B"); the before/after diagram runs in the import direction ("A imports B"). They
are never mixed in one picture, and each says which it is.

**It is focused.** The walk only pulls the change's neighbourhood (§49), and the diagram
stops at a fixed number of boxes rather than trying to draw a repository. When it stops,
it says how many boxes it left out — a picture invites the assumption that it is the whole
picture.

**The same records are also in words.** Below the diagrams, the traversal chains and the
structural change are listed as text. It is the same information, and it is what a screen
reader or a text browser gets.

Export the map to take it elsewhere:

```bash
traceflow export /path/to/repo                          # Excalidraw, change map
traceflow export /path/to/repo --view before-after      # the structural comparison
traceflow export /path/to/repo --format json            # the graph as data
traceflow export /path/to/repo --format svg             # a standalone drawing
traceflow export /path/to/repo --output ./map.excalidraw
```

Exports default into `.traceflow/exports/`, which `init` keeps out of version control.
That matters: TraceFlow writes into the repository it watches, and an export beside your
source would show up in the next session as an untracked file the agent never wrote.

The Excalidraw file is a normal `.excalidraw` document you can annotate and move around.
Its boxes carry the state words, and its caption, arrow meaning and limits travel with it,
because a drawing handed to someone else has no page around it to explain it. It is
reproducible too: Excalidraw's random seed and timestamp are derived from the data, so
exporting one session twice gives you the same bytes.

### Intent versus actual change

Record the task you gave the agent — at analysis time with
`traceflow analyze --intent "…"`, or afterwards with `traceflow intent --task "…"` — and
TraceFlow compares its words with what the session's own artifacts say changed. A file
counts as related when the task shares a word with its path, with a symbol the session
changed in it, or with a module it began or stopped importing; word matching forgives
inflections (`limiting` finds `rate_limit`).

The verdict uses plan.md §63's own words, and only three of them:

- **potentially related** — every changed file is covered by some word of the task
- **potential scope expansion** — some changed files are covered, some are not
- **no detected relationship** — no word of the task reaches any changed file

Two things it will never do. It will not say the agent did something wrong (§21): a
change the task did not name may be exactly what the task required, and the delivery
says so beside the verdict. And it will not compare when it cannot: a task that parses
to nothing but filler (`"please refactor the code"`) is recorded verbatim and marked
*not compared* rather than dressed up as a finding. Every match is listed — token, kind,
where it landed — so the comparison can be checked instead of trusted.

### Test evidence

Test execution is **off by default** (§22: running tests can be expensive or
destructive), and turns on only under `[tests]` in the repository's configuration. Even
then, TraceFlow runs only the tests the session's own impact analysis reaches — the test
files the session changed, plus the test files the walk arrived at — never the whole
suite.

The command is read from configuration only, split into a program and arguments, and
executed without a shell; paths follow a `--` separator so a filename can never become
an option. A run is bounded by `timeout_seconds` and stopped when it exceeds it.

The outcome is recorded as evidence about the run, never as a verdict on the change
(§23): the artifact keeps the status, the exact command, the exit code, the duration,
the summary counts, and the tail of the output — shown under Evidence. `Tests passed`
means the tests passed; it does not mean the change is safe. When the command is
missing, or the run exceeds its timeout, the record says that instead of a result.

### Sessions and baselines

A session is one coherent episode of change — deliberately not a commit and not a pull
request, because an agent may make dozens of edits before any commit exists.

A repository that is already dirty when the watcher starts has **pre-existing**
changes. Those are recorded as the session's baseline and are never attributed to the
session. This is a correctness requirement, not a nicety: without it, every session
would take credit for whatever was already lying around.

The baseline is built from two things:

- **The commit the session started from.** Diffing against that commit gives the
  session's changes to every file that was *clean* when it began. The recorded commit
  is used rather than `HEAD` on purpose — an agent that commits partway through a
  session moves `HEAD`, and diffing against the moved `HEAD` would silently lose
  everything it had already done.
- **A content snapshot of files git cannot supply a "before" for.** A file that was
  already modified, or already untracked, has no committed version matching what was
  on disk. Its bytes are copied into TraceFlow's own content-addressed store, and the
  session's changes to that file are measured against that snapshot.

### How change collection works

Three sources are combined, each used where it is the authority:

| File | Diffed by | Why |
|---|---|---|
| Clean at baseline | git, against the recorded commit | git's line counts, rename detection and binary handling are the reference implementation |
| Already modified or untracked at baseline | TraceFlow's snapshot, with `difflib` | git has no record of what the file looked like at that moment |
| Newly untracked during the session | counted directly | it did not exist, so all of it is an addition |

A file appears in exactly one of these, so no path is counted twice. Every changed
file is reported with its status, insertions, deletions, and — for renames — the path
it came from. Line counts are omitted rather than guessed at for binary files, and a
file whose contents were withheld is marked as such.

**Pre-existing changes are reported separately from the session's own.** They are
measured from the committed version to the baseline snapshot, never to the working
tree, because diffing to the working tree would fold the session's edits into the
pre-existing count and overstate it.

### Secrets

TraceFlow refuses to read files matching a sensitive-path policy — `.env`, `*.pem`,
`id_rsa`, `.ssh`, `.aws`, and similar. A value that was never read cannot leak through
a redaction bug, a log line, or a stack trace, so exclusion is a stronger guarantee
than masking.

For a *tracked* secret file, git still supplies status and line counts without anyone
reading the bytes, so it is reported normally with its contents withheld. For an
*untracked* one there is no such source, so it is recorded in the baseline with the
reason and left out of the change set. Reporting a change that could not be
determined would be a fabrication.

### Where state lives

```
.traceflow/
├── events.jsonl                    append-only event log, never rewritten
├── current-baseline.json           the baseline the last run recorded
├── blobs/                          content-addressed file snapshots
│   └── a1/a1b2c3…
├── derived/                        disposable — delete it and it rebuilds
│   └── analysis/python-1/          parse results, keyed by content hash
└── sessions/
    └── 2026-09-25T14-52-43-9b3625/
        ├── session.json            the session record
        ├── baseline.json           what it was measured against
        ├── changes.json            which files changed
        ├── symbols.json            what changed inside them
        ├── impact.json             what that reaches
        ├── intent.json             the task, when one was recorded (§63)
        └── tests.json              the test run, when the repository enabled one (§64)
```

The split between evidence and derived state is deliberate. `events.jsonl`, `blobs/` and
`baseline.json` are **evidence**: immutable, never rewritten, and the only record of what
the repository actually looked like. Everything under `derived/` is **computation**: when
an analyzer improves, deleting that directory is the whole migration, and no session is
left carrying conclusions drawn by an older, worse one.

Blobs and parse results are both content-addressed, so identical content is stored once
no matter how many paths or sessions refer to it. A rename costs nothing, and a baseline
snapshot is already parsed because the blob digest *is* the cache key.

---

## Configuration

`.traceflow.toml` at the repository root. Unknown keys are rejected rather than
ignored, so a typo fails loudly instead of silently doing nothing.

```toml
ignore = [".traceflow"]

[repository]
path = "."

[activity]
quiet_period_seconds = 8        # untouched this long ⇒ the session is stable
minimum_session_seconds = 2     # ignore bursts shorter than this
poll_interval_seconds = 2       # cadence while the repository is changing
idle_poll_interval_seconds = 5  # cadence while nothing is happening

[analysis]
max_file_size_mb = 5            # larger files are reported but never read
impact_max_depth = 5            # rings of dependents the impact walk expands

[secrets]
# Contents are never read for these. They still appear with their path, status and
# line counts; only their contents are withheld.
exclude_paths = [".env", "*.pem", "id_rsa", ".ssh", ".aws", …]

[tests]
enabled = false            # opt in explicitly (§22) — nothing runs by default
command = "pytest"         # split on words, run without a shell
timeout_seconds = 300      # the run is stopped and recorded as timed out
```

When `[tests]` is enabled, TraceFlow runs **only the tests the session's own impact
analysis reaches** — the test files it changed, plus the test files the walk arrived at
— never the whole suite. Test paths are passed after a `--` separator so a filename can
never become an option, and the result is recorded as evidence about the run:
`Tests passed`, not `Change is safe` (§23).

The `ignore` list is **not** a replacement for `.gitignore` — git already handles
ignored paths correctly. It exists so TraceFlow's own writes can never be mistaken for
repository activity.

`exclude_paths` matches on paths rather than content, deliberately. Path exclusion is
a guarantee; content scanning is a second line of defence and belongs with the phase
that renders diffs. Matching is case-insensitive and errs towards excluding: a false
positive costs one file's line statistics, a false negative costs a leaked credential.

---

## Development

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Windows
python .agent/eval/scripts/verify.py                # the completion gate
```

The gate runs ruff, ruff format, mypy (strict), and pytest, and refuses a vacuous pass:
tools exit zero when they have nothing to check, so it verifies that tests were actually
collected and that mypy actually checked files.

There are no runtime dependencies. Phase 1 needs the standard library, `git`, and
`tomli` on Python 3.10 (where `tomllib` is not yet in the standard library).

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Competitive landscape research | done |
| 1 | Repository watcher: activity, stability, sessions | done |
| 2 | Git evidence: baseline, change collection, line statistics | done |
| 3 | Python static analysis: AST, imports, symbols, changed symbols | done |
| 4 | Impact engine: direct/indirect impact, before/after comparison | **done** |
| 5 | Delivery: local dashboard | **done** |
| 6 | Visualization: focused change map, Excalidraw export | **done** |
| 7 | Intent vs actual change | **done** |
| 8 | Test evidence | **done** |
| 9 | Optional AI explanation layer | planned |
| 10 | Additional languages: **TypeScript/TSX/JavaScript done** — Go, Java, Rust, C# planned | **TypeScript done** |

---

## Design principles

- **Evidence before interpretation.** Prefer `auth/routes.py imports auth/service.py`
  over "this module appears to participate in authentication."
- **Explicit uncertainty.** Every claim is `Confirmed`, `Inferred`, `Possible`, or
  `Unknown`. `Unknown` beats a fabricated answer, and there is no numeric confidence
  score pretending to be a probability.
- **No verdicts.** TraceFlow reports what changed and what is connected. It does not
  tell you whether a change was good, and it does not claim to know what the agent
  intended.
- **Focused, not exhaustive.** The visualisation follows the change, not the
  repository. A 50,000-node graph is a worse experience than no graph.
- **Language adapters, not a pretend-universal parser.** Python is analysed with the
  standard library's `ast`. TypeScript/TSX/JavaScript are analysed with a
  scanner that understands strings, templates, comments and regex literals — a
  deliberate scope, stated in its docstring, rather than a claim to parse the whole
  language. `plan.md` §6 requires an interface rather than a claim, so
  `languages/base.py` defines what an analyzer must provide and each language
  implements it — a new language is a new subpackage plus one registry entry, not a
  rewrite.
- **Local by default.** Source code does not leave the machine.

---

## Known constraints

**Do not redirect TraceFlow's output into the repository it is watching.** TraceFlow's
own state directory is filtered out of activity detection, but a log file written
anywhere else in the repository is not — and because the watcher prints whenever the
state changes, writing its output into the watched tree creates a loop that never
settles. Redirect somewhere else:

```bash
traceflow watch . > /tmp/traceflow.log      # good
traceflow watch . > watch.log               # bad: watch.log is inside the repository
```

**TraceFlow reads file contents in two places.** The baseline snapshot, and line
counting for files git does not track. Both honour the sensitive-path policy, and
git's own reporting covers tracked files without anyone reading them. That is the
whole reason the policy is enforced by path rather than by scanning content.

**Static analysis has limits.** Later phases cannot resolve dynamic dispatch,
monkey-patching, or runtime-generated code. Where TraceFlow cannot determine
something, it says so rather than guessing — `Unknown` is a real answer.

---

## License

MIT
