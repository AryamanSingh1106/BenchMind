import zlib
import bz2
import hashlib
from typing import Tuple, Dict, Any
from benchmarks.cpu.common import run_timed_subtest, SubtestResult


def _generate_deterministic_payload(size_mb: float = 10.0) -> bytes:
    """Generate deterministic payload buffer with mixed text & structured data."""
    target_bytes = int(size_mb * 1024 * 1024)
    pattern = (b"BenchMind CPU Compression Test Payload 1234567890 " * 20) + (bytes(range(256)) * 4)
    multiplier = (target_bytes // len(pattern)) + 1
    full_buffer = (pattern * multiplier)[:target_bytes]
    return full_buffer


def compression_workload(buffer_size_mb: float = 10.0) -> Tuple[Dict[str, Any], float]:
    """
    Compression & Decompression Workload:
    - zlib (Deflate level 6) compress & decompress
    - bz2 compress & decompress
    """
    data = _generate_deterministic_payload(buffer_size_mb)
    original_md5 = hashlib.md5(data).hexdigest()
    data_size = len(data)

    # 1. zlib compress & decompress
    zlib_compressed = zlib.compress(data, level=6)
    zlib_decompressed = zlib.decompress(zlib_compressed)

    # 2. bz2 compress & decompress
    bz2_compressed = bz2.compress(data, compresslevel=5)
    bz2_decompressed = bz2.decompress(bz2_compressed)

    total_bytes_processed = (data_size * 2) + (data_size * 2)  # 2x compress + 2x decompress

    output = {
        "original_md5": original_md5,
        "zlib_md5": hashlib.md5(zlib_decompressed).hexdigest(),
        "bz2_md5": hashlib.md5(bz2_decompressed).hexdigest(),
        "original_size": data_size,
        "zlib_size": len(zlib_compressed),
        "bz2_size": len(bz2_compressed)
    }

    # Scale processed bytes to Megabytes (MB)
    return output, total_bytes_processed / (1024 * 1024)


def validate_compression_workload(output: Dict[str, Any]) -> bool:
    """
    Validate compression output:
    100% round-trip match (zlib and bz2 decompressed MD5s match original MD5).
    """
    if not isinstance(output, dict):
        return False
    original_md5 = output.get("original_md5")
    zlib_md5 = output.get("zlib_md5")
    bz2_md5 = output.get("bz2_md5")

    if not original_md5 or not zlib_md5 or not bz2_md5:
        return False
    if original_md5 != zlib_md5 or original_md5 != bz2_md5:
        return False
    return True


def run_compression_subtest(target_duration: float = 1.0) -> SubtestResult:
    return run_timed_subtest(
        name="Data Compression & Decompression",
        category="compression",
        workload_profile="compression",
        raw_metric_name="MB/s",
        workload_fn=compression_workload,
        validate_fn=validate_compression_workload,
        target_duration=target_duration,
        min_reps=3,
        max_reps=5
    )
