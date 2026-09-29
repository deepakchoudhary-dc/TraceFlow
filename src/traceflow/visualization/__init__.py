"""Turning a recorded session into a picture (plan.md §28, §31, §32, §49, §62).

The internal representation is the **evidence graph**: a set of files as nodes and a set
of recorded relationships as edges, every one of them traceable to an artifact. Everything
outside this package is a renderer of it — the dashboard draws it as SVG, and
:mod:`traceflow.visualization.excalidraw` and :mod:`traceflow.visualization.serializer`
write it out for other tools. plan.md §32 asks for exactly that split so a future Graphviz
or Mermaid target costs a module rather than a rewrite.

Two constraints shape the package.

**It is a projection, not an engine.** No module here parses source, resolves a call or
walks a graph. It reads what a session already recorded and arranges it. plan.md §80 is
explicit that the diagram is not the product, and plan.md §54 that TraceFlow must not
become a diagram generator; keeping the graph a pure projection of the artifacts is what
keeps those two statements true rather than aspirational.

**An edge exists only if an artifact recorded it.** A line drawn between two files because
they look related would be the one thing this product exists to replace. Where a
relationship was not recorded, it is absent, and the page says so — plan.md §69 prefers
"unknown" to a plausible invention.
"""
