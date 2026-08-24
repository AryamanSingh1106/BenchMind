import hashlib
from typing import Tuple, Dict, Any
from benchmarks.cpu.common import run_timed_subtest, SubtestResult


def _generate_hash_payload(size_mb: float = 16.0) -> bytes:
    target_bytes = int(size_mb * 1024 * 1024)
    seed = b"BenchMind Cryptographic Hashing Seed Buffer 0123456789 ABCDEF " * 16
    multiplier = (target_bytes // len(seed)) + 1
    return (seed * multiplier)[:target_bytes]


def hashing_workload(buffer_size_mb: float = 16.0, passes: int = 5) -> Tuple[Dict[str, Any], float]:
    """
    Cryptographic Hashing Workload:
    - Multi-pass SHA-256 and BLAKE2b hashing over 16MB buffer
    """
    data = _generate_hash_payload(buffer_size_mb)
    data_size = len(data)

    sha256_digests = []
    blake2b_digests = []

    for _ in range(passes):
        sha256_digests.append(hashlib.sha256(data).hexdigest())
        blake2b_digests.append(hashlib.blake2b(data).hexdigest())

    total_bytes_hashed = (data_size * passes) + (data_size * passes)

    output = {
        "sha256_digest": sha256_digests[-1],
        "blake2b_digest": blake2b_digests[-1],
        "sha256_passes": len(sha256_digests),
        "blake2b_passes": len(blake2b_digests),
        "buffer_size": data_size
    }

    # Scale total bytes to Megabytes (MB)
    return output, total_bytes_hashed / (1024 * 1024)


def validate_hashing_workload(output: Dict[str, Any]) -> bool:
    """
    Validate cryptographic hashing output:
    Verify expected length of hex digests and non-empty values.
    """
    if not isinstance(output, dict):
        return False
    sha256_digest = output.get("sha256_digest")
    blake2b_digest = output.get("blake2b_digest")

    if not sha256_digest or not blake2b_digest:
        return False
    if len(sha256_digest) != 64:  # SHA-256 is 64 hex chars
        return False
    if len(blake2b_digest) != 128:  # BLAKE2b is 128 hex chars
        return False
    return True


def run_hashing_subtest(target_duration: float = 1.0) -> SubtestResult:
    return run_timed_subtest(
        name="Cryptographic Hashing (SHA-256 & BLAKE2b)",
        category="hashing",
        workload_profile="crypto",
        raw_metric_name="MB/s",
        workload_fn=hashing_workload,
        validate_fn=validate_hashing_workload,
        target_duration=target_duration,
        min_reps=3,
        max_reps=5
    )
