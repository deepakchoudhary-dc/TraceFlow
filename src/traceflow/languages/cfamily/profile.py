"""What one C-family language contributes to the shared engine (plan.md §6, §66).

The engine — :mod:`traceflow.languages.cfamily.analyzer` — is one machine: it reads a
declaration backward from its brace, extracts import forms, and records call sites.
A :class:`LanguageProfile` is the set of knobs that machine turns per language. Every
knob exists because two of the languages genuinely disagree about it, and the profile
records the disagreement where the resolution happens rather than scattering
``if language == "go"`` through the engine.

Adding the ninth language is writing one of these — the engine is not touched. That
is the property plan.md §6 asked for and §66 gated Phase 10 on.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LanguageProfile:
    """The per-language facts the shared engine needs."""

    name: str
    extensions: tuple[str, ...]

    # ---- declaration keywords -------------------------------------------------
    type_keywords: frozenset[str]
    """Words that make the next name a composite type: ``struct/interface/…``."""

    function_keywords: frozenset[str]
    """Words that make the next name a function: ``func``, ``fn``, ``def``-analogs."""

    #: Words that sit between a declaration and its name (``impl Trait for Type``,
    #: ``public sealed class``): the backward climb steps over them looking for the
    #: real declaration head.
    modifier_words: frozenset[str] = field(default_factory=frozenset)

    #: Words that end a backward climb without declaring anything: a name preceded
    #: by one of these is an operand or a call, not a declaration head.
    stop_words: frozenset[str] = field(default_factory=frozenset)

    #: Words whose following brace is a **type-scope** (members are methods) rather
    #: than an anonymous block. Rust's ``impl`` is the case that forces this to be
    #: a set: ``impl Display for Wrapper {`` opens method scope with no name of
    #: its own adjacent to the brace.
    scope_openers: frozenset[str] = field(default_factory=frozenset)

    #: Words that turn the preceding ``for`` into part of a compound head
    #: (``impl Display for Wrapper``) so the declared name is one word further back.
    compound_heads: frozenset[str] = field(default_factory=frozenset)

    #: Words that end a call-like header in a *declaration* (``matches!(x, y)``);
    #: a name followed by ``!`` is a macro invocation, not a call site.
    macro_marker: str = "!"
    """Set to ``""`` when the language has no macro-call syntax."""

    # ---- statement keywords that are never symbols ----------------------------
    control_keywords: frozenset[str] = field(default_factory=frozenset)

    # ---- import extraction ----------------------------------------------------
    import_line_starts: tuple[str, ...] = ()
    """Line forms that begin an import: ``import``, ``from``, ``using``."""

    import_is_block: bool = False
    """Java/C# shapes: ``import x.y;`` / ``using x.y;`` single statements are the
    common case; the parenthesised multi-import form is parsed by the analyzer."""

    uses_colon_imports: bool = False
    """Rust shape: ``use a::b::c;`` — path separator resolution, not dots."""

    # ---- call modelling -------------------------------------------------------
    call_separators: tuple[str, ...] = (".",)
    """Tokens joining a call chain: ``.`` everywhere except Rust's ``::``."""

    method_call_marker: str = "."
    """The separator that marks a *method* call site (``obj.method(…)``)."""


def go_profile() -> LanguageProfile:
    """Go: ``func (r *Repo) Save(ctx) error {`` — receiver methods, capitalisation."""

    return LanguageProfile(
        name="go",
        extensions=(".go",),
        type_keywords=frozenset({"type", "struct", "interface"}),
        function_keywords=frozenset({"func"}),
        modifier_words=frozenset({"type", "map", "chan"}),
        stop_words=frozenset({"=", ":", "return", "go", "defer", "var", "const"}),
        scope_openers=frozenset(),
        control_keywords=frozenset(
            {"if", "for", "range", "switch", "select", "case", "else", "return", "go", "defer"}
        ),
        import_line_starts=("import",),
        # A call chain in Go is always dotted; a capitalised name is *exported*,
        # which the analyzer records as ordinary naming — no visibility guessing.
    )


def java_profile() -> LanguageProfile:
    """Java: ``public List<String> findAll() {`` — modifier walls, generics everywhere."""

    return LanguageProfile(
        name="java",
        extensions=(".java",),
        type_keywords=frozenset({"class", "interface", "enum", "record"}),
        function_keywords=frozenset(),
        modifier_words=frozenset(
            {
                "public",
                "private",
                "protected",
                "static",
                "final",
                "abstract",
                "synchronized",
                "native",
                "transient",
                "volatile",
                "strictfp",
                "default",
                "sealed",
                "non",
            }
        ),
        stop_words=frozenset({"=", "new", "return", "throw", "catch", "this", "super"}),
        control_keywords=frozenset(
            {
                "if",
                "for",
                "while",
                "switch",
                "case",
                "else",
                "do",
                "try",
                "catch",
                "finally",
                "return",
                "synchronized",
            }
        ),
        import_line_starts=("import",),
        import_is_block=True,
    )


def rust_profile() -> LanguageProfile:
    """Rust: ``pub fn save(&self, x: u32) -> Result<T, E> {`` — ``impl`` scopes, ``::``."""

    return LanguageProfile(
        name="rust",
        extensions=(".rs",),
        type_keywords=frozenset({"struct", "enum", "trait", "union"}),
        function_keywords=frozenset({"fn"}),
        modifier_words=frozenset(
            {
                "pub",
                "async",
                "const",
                "unsafe",
                "extern",
                "move",
                "dyn",
                "impl",
                "default",
                "crate",
                "self",
                "super",
            }
        ),
        stop_words=frozenset({"=", "let", "return", "match", "if", "in", "as"}),
        scope_openers=frozenset({"impl"}),
        compound_heads=frozenset({"for"}),
        macro_marker="!",
        control_keywords=frozenset(
            {
                "if",
                "for",
                "while",
                "loop",
                "match",
                "else",
                "return",
                "unsafe",
                "where",
                "in",
                "as",
                "use",
                "mod",
            }
        ),
        import_line_starts=("use",),
        uses_colon_imports=True,
        call_separators=(".", "::"),
        method_call_marker=".",
    )


def csharp_profile() -> LanguageProfile:
    """C#: ``public async Task<IActionResult> Get() {`` — Java's wall, attributes."""

    return LanguageProfile(
        name="csharp",
        extensions=(".cs",),
        type_keywords=frozenset({"class", "interface", "enum", "record", "struct"}),
        function_keywords=frozenset(),
        modifier_words=frozenset(
            {
                "public",
                "private",
                "protected",
                "internal",
                "static",
                "sealed",
                "abstract",
                "override",
                "virtual",
                "async",
                "readonly",
                "const",
                "partial",
                "extern",
                "new",
                "unsafe",
                "required",
            }
        ),
        stop_words=frozenset(
            {"=", "return", "throw", "this", "base", "is", "as", "var", "out", "ref", "in"}
        ),
        control_keywords=frozenset(
            {
                "if",
                "for",
                "foreach",
                "while",
                "switch",
                "case",
                "else",
                "do",
                "try",
                "catch",
                "finally",
                "return",
                "lock",
                "using",
                "when",
            }
        ),
        import_line_starts=("using",),
        import_is_block=True,
    )
