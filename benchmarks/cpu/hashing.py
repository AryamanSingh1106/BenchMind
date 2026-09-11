"""
Cryptographic hashing throughput (SHA-256 and BLAKE2b).

The payload is now built once in `setup` instead of being regenerated inside
every timed repetition. Validation checks the digests against values computed
independently after the clock stops, rather than just asserting that the hex
string is the right length.

What this really measures: your OpenSSL build. On x86-64 with SHA extensions,
SHA-256 runs several times faster than on a CPU without them, which is a real
and interesting hardware difference, but the OpenSSL version is recorded in
the fingerprint so scores are only compared like for like.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Tuple

from benchmarks.cpu.payloads import build_mixed_corpus

SIZE_MB = 16.0
PASSES = 4


def setup(scale: float = 1.0) -> Dict[str, Any]:
    # Buffer size fixed; `scale` changes the number of passes. A smaller
    # buffer would sit in cache and inflate the MB/s figure.
    data = build_mixed_corpus(SIZE_MB, seed=777)
    return {
        "data": data,
        "size": len(data),
        "passes": max(1, int(PASSES * scale)),
        "expected_sha256": hashlib.sha256(data).hexdigest(),
        "expected_blake2b": hashlib.blake2b(data).hexdigest(),
    }


def run(ctx: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    data, size, passes = ctx["data"], ctx["size"], ctx["passes"]

    sha_digest = ""
    blake_digest = ""
    for _ in range(passes):
        sha_digest = hashlib.sha256(data).hexdigest()
        blake_digest = hashlib.blake2b(data).hexdigest()

    total_bytes = 2.0 * size * passes
    return (
        {
            "sha256_digest": sha_digest,
            "blake2b_digest": blake_digest,
            "expected_sha256": ctx["expected_sha256"],
            "expected_blake2b": ctx["expected_blake2b"],
            "passes": passes,
        },
        total_bytes / (1024 * 1024),
    )


def validate(output: Dict[str, Any]) -> bool:
    if not isinstance(output, dict):
        return False

    digest = output.get("sha256_digest")
    blake = output.get("blake2b_digest")
    expected_sha = output.get("expected_sha256")
    expected_blake = output.get("expected_blake2b")

    # Presence check first. Without it, an empty dict passes because
    # None == None, which would let a workload that produced nothing at all
    # be scored as correct.
    if not all(isinstance(v, str) and v for v in
               (digest, blake, expected_sha, expected_blake)):
        return False
    if len(digest) != 64 or len(blake) != 128:
        return False

    return digest == expected_sha and blake == expected_blake
