"""
benchmarks/cpu/payloads.py

Deterministic payload generation for the compression and hashing workloads.

Why this file exists
--------------------
BenchMind 1.x built its compression payload by repeating a short byte pattern.
That buffer compresses at an absurd ratio, so the "compression" benchmark was
really measuring how fast the deflate match finder can skip over very long
runs. It was not measuring compression on anything resembling real data.

The generator below builds a mixed-entropy corpus instead, in fixed
proportions, so the measurement exercises literal coding, match finding and
entropy coding all at once:

    40%  English-like text          (highly compressible, long matches)
    25%  structured records         (JSON-ish, moderate redundancy)
    20%  binary floats              (low redundancy, poor matches)
    15%  incompressible random      (forces literal path)

All generation is seeded, so the same payload is produced on every machine and
every run. Payload construction is expensive; it belongs in `setup`, never in
the timed region.
"""

from __future__ import annotations

import hashlib
import numpy as np

_WORDS = (
    b"benchmark performance memory latency throughput cache pipeline vector "
    b"scalar integer floating branch predictor scheduler thermal frequency "
    b"kernel process thread affinity workload silicon package substrate "
)


def _text_block(nbytes: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    words = _WORDS.split()
    out = bytearray()
    while len(out) < nbytes:
        idx = rng.integers(0, len(words), size=256)
        out += b" ".join(words[i] for i in idx) + b".\n"
    return bytes(out[:nbytes])


def _record_block(nbytes: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    out = bytearray()
    i = 0
    while len(out) < nbytes:
        vals = rng.integers(0, 1_000_000, size=4)
        out += (
            b'{"id":%d,"cpu":%d,"ram":%d,"temp":%d,"tag":"node"}\n'
            % (i, vals[0], vals[1], vals[2])
        )
        i += 1
    return bytes(out[:nbytes])


def _float_block(nbytes: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    count = nbytes // 8 + 1
    arr = rng.normal(0.0, 1.0, size=count).astype(np.float64)
    return arr.tobytes()[:nbytes]


def _random_block(nbytes: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=nbytes, dtype=np.uint8).tobytes()


def build_mixed_corpus(size_mb: float, seed: int = 20260101) -> bytes:
    """
    Build a deterministic mixed-entropy corpus of approximately `size_mb`
    megabytes. Identical output for identical inputs on any platform.
    """
    total = int(size_mb * 1024 * 1024)
    parts = [
        _text_block(int(total * 0.40), seed + 1),
        _record_block(int(total * 0.25), seed + 2),
        _float_block(int(total * 0.20), seed + 3),
        _random_block(total - int(total * 0.40) - int(total * 0.25) - int(total * 0.20),
                      seed + 4),
    ]
    return b"".join(parts)


def corpus_digest(data: bytes) -> str:
    """SHA-256 of a corpus, used as the round-trip reference in validation."""
    return hashlib.sha256(data).hexdigest()
