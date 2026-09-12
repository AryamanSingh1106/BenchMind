"""
Single source of truth for the BenchMind version.

Kept at the repository root, outside any package, so every layer can import it
without creating a dependency cycle. Previously the version was written out in
four places and a bump silently left three of them stale, which a test caught
only by hardcoding the string itself.

Bump this when scoring, workloads or baselines change, and record what changed
in CHANGELOG.md. `BASELINE_VERSION` in benchmarks/cpu/common.py is separate and
tracks only the scoring baselines: two builds can share a baseline version and
be directly comparable while differing in application version.
"""

__version__ = "2.1.0"
