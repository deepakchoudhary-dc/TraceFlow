"""Git as the primary evidence source (plan.md §6, §14, §15).

Git is used through subprocess calls rather than a wrapper library. The calls are
few, the output formats are stable and documented, and a subprocess keeps the
behaviour transparent — there is no library abstraction between TraceFlow and the
data it reports.
"""

from traceflow.git.baseline import (
    EMPTY_TREE,
    Baseline,
    CapturedFile,
    capture_baseline,
)
from traceflow.git.diff import (
    ChangeSet,
    ChangeStatus,
    FileChange,
    RawChange,
    collect_changes,
    count_line_changes,
    parse_name_status,
    parse_numstat,
)
from traceflow.git.repository import (
    GitError,
    ParsedStatus,
    Repository,
    StatusEntry,
    WorkingTreeState,
    parse_porcelain_v2,
)

__all__ = [
    "EMPTY_TREE",
    "Baseline",
    "CapturedFile",
    "ChangeSet",
    "ChangeStatus",
    "FileChange",
    "GitError",
    "ParsedStatus",
    "RawChange",
    "Repository",
    "StatusEntry",
    "WorkingTreeState",
    "capture_baseline",
    "collect_changes",
    "count_line_changes",
    "parse_name_status",
    "parse_numstat",
    "parse_porcelain_v2",
]
