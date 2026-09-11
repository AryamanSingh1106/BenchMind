"""
Dense matrix multiplication (BLAS dgemm).

1.x regenerated two 768x768 float64 matrices with two RNG streams inside the
timed function on every repetition. Filling ~1.2 million doubles is not free,
and it was being charged against the GFLOPS number.

2.0 allocates A, B and the output buffer once in `setup` and calls
`np.matmul(A, B, out=C)` in the timed region, so the measurement is dgemm and
nothing else.

Note honestly what this measures: your BLAS build. The same silicon will score
differently under OpenBLAS, MKL and the NumPy reference build. The BLAS vendor
and version are captured in the environment fingerprint for exactly this
reason.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

SIZE = 768
LOOPS = 3


def setup(scale: float = 1.0) -> Dict[str, Any]:
    # Matrix dimension is fixed. A 384x384 matmul and a 768x768 matmul sit in
    # different cache regimes and are not the same measurement, so `scale`
    # changes the repetition count rather than the size.
    size = SIZE
    rng_a = np.random.default_rng(12345)
    rng_b = np.random.default_rng(67890)
    return {
        "size": size,
        "loops": max(1, int(LOOPS * scale)),
        "A": np.ascontiguousarray(rng_a.uniform(0.1, 2.0, size=(size, size)), dtype=np.float64),
        "B": np.ascontiguousarray(rng_b.uniform(0.1, 2.0, size=(size, size)), dtype=np.float64),
        "C": np.empty((size, size), dtype=np.float64),
    }


def run(ctx: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    A, B, C = ctx["A"], ctx["B"], ctx["C"]
    size, loops = ctx["size"], ctx["loops"]

    for _ in range(loops):
        np.matmul(A, B, out=C)

    total_flops = float(loops) * 2.0 * (size ** 3)
    return (
        {
            "matrix_sum": float(np.sum(C)),
            "matrix_trace": float(np.trace(C)),
            "c_00": float(C[0, 0]),
            "c_mid": float(C[size // 2, size // 2]),
            "size": size,
        },
        total_flops / 1e9,
    )


def validate(output: Dict[str, Any]) -> bool:
    """
    Verify against an independently computed reference element rather than
    only checking for NaN. C[0,0] must equal dot(A[0,:], B[:,0]).
    """
    if not isinstance(output, dict):
        return False
    size = output.get("size")
    c_00 = output.get("c_00")
    matrix_sum = output.get("matrix_sum")
    if not size or c_00 is None or matrix_sum is None:
        return False
    if not np.isfinite(c_00) or not np.isfinite(matrix_sum) or matrix_sum <= 0:
        return False

    rng_a = np.random.default_rng(12345)
    rng_b = np.random.default_rng(67890)
    a_row = rng_a.uniform(0.1, 2.0, size=(size, size))[0, :]
    b_col = rng_b.uniform(0.1, 2.0, size=(size, size))[:, 0]
    expected = float(np.dot(a_row, b_col))

    return abs(c_00 - expected) <= max(1e-6, abs(expected) * 1e-9)
