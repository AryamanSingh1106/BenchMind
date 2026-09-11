"""
Compression and decompression throughput.

1.x built its payload by repeating a 1 KB pattern, which deflate chews through
at a ratio of several hundred to one. That measured the match finder skipping
long runs, not compression. It also ran three MD5 digests inside the timed
region and did not count those bytes in the throughput denominator, so the
MB/s figure was wrong in two directions at once.

2.0 uses the mixed-entropy corpus from payloads.py, built once in `setup`, and
times only the four compress/decompress calls. Round-trip verification moved
out of the timed region entirely: `run` returns the decompressed buffers and
`validate` hashes them afterwards.
"""

from __future__ import annotations

import bz2
import zlib
from typing import Any, Dict, Tuple

from benchmarks.cpu.payloads import build_mixed_corpus, corpus_digest

SIZE_MB = 8.0
ZLIB_LEVEL = 6
BZ2_LEVEL = 5


def setup(scale: float = 1.0) -> Dict[str, Any]:
    # Corpus size is fixed across modes. Compression throughput depends on the
    # payload's entropy profile, and a smaller slice of the corpus has a
    # different profile, so `scale` is deliberately ignored here.
    size_mb = SIZE_MB
    data = build_mixed_corpus(size_mb)
    return {
        "data": data,
        "size": len(data),
        "digest": corpus_digest(data),
    }


def run(ctx: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    data = ctx["data"]
    size = ctx["size"]

    zlib_compressed = zlib.compress(data, level=ZLIB_LEVEL)
    zlib_decompressed = zlib.decompress(zlib_compressed)

    bz2_compressed = bz2.compress(data, compresslevel=BZ2_LEVEL)
    bz2_decompressed = bz2.decompress(bz2_compressed)

    # Two compress passes plus two decompress passes over the payload.
    total_bytes = 4.0 * size

    return (
        {
            "reference_digest": ctx["digest"],
            "zlib_out": zlib_decompressed,
            "bz2_out": bz2_decompressed,
            "original_size": size,
            "zlib_size": len(zlib_compressed),
            "bz2_size": len(bz2_compressed),
        },
        total_bytes / (1024 * 1024),
    )


def validate(output: Dict[str, Any]) -> bool:
    """Full round-trip verification, performed after the clock has stopped."""
    if not isinstance(output, dict):
        return False
    ref = output.get("reference_digest")
    zlib_out = output.get("zlib_out")
    bz2_out = output.get("bz2_out")
    if not ref or zlib_out is None or bz2_out is None:
        return False
    if corpus_digest(zlib_out) != ref:
        return False
    if corpus_digest(bz2_out) != ref:
        return False
    # Sanity check the ratio: a mixed corpus should land well short of 10x.
    original = output.get("original_size", 0)
    zsize = output.get("zlib_size", 0)
    if original and zsize and (original / zsize) > 10.0:
        return False
    return True
