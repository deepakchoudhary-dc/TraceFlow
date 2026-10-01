"""The C-family analyzer package: Go, Java, Rust and C# over one profile-driven engine.

The language-specific fact is a :class:`~traceflow.languages.cfamily.profile.LanguageProfile`;
the machinery is shared. Adding the next language is a new profile plus a registry
entry, which is the property plan.md §6 asked the architecture to have.
"""

from __future__ import annotations

from traceflow.languages.cfamily.analyzer import (
    ANALYZER_VERSION,
    CFamilyAnalyzer,
    csharp_profile,
    go_profile,
    java_profile,
    rust_profile,
)
from traceflow.languages.cfamily.graph import (
    CFamilyFiles,
    CFamilyIndex,
    cfamily_languages_for,
)
from traceflow.languages.cfamily.profile import LanguageProfile
from traceflow.languages.cfamily.scanner import SourceView

__all__ = [
    "ANALYZER_VERSION",
    "CFamilyAnalyzer",
    "CFamilyFiles",
    "CFamilyIndex",
    "LanguageProfile",
    "SourceView",
    "cfamily_languages_for",
    "csharp_profile",
    "go_profile",
    "java_profile",
    "rust_profile",
]
