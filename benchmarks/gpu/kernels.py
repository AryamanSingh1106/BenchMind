"""
benchmarks/gpu/kernels.py

OpenCL kernel sources for the GPU suite.

Every kernel is written so its result can be verified against a CPU reference.
The 1.x kernel never had its output read back at all, so a driver that silently
did nothing would have produced a high score.

The FMA kernels use a dependent chain (`x = x * y + c`) specifically so the
compiler cannot hoist the loop out. An independent chain would be optimized
away and the "benchmark" would measure the optimizer.
"""

FP32_FMA = """
__kernel void fp32_fma(__global const float *a,
                       __global const float *b,
                       __global float *out,
                       const int inner_loops)
{
    int gid = get_global_id(0);
    float x = a[gid];
    float y = b[gid];
    // Dependent chain: each iteration needs the previous result, so the
    // compiler cannot collapse the loop.
    for (int i = 0; i < inner_loops; i++) {
        x = fma(x, y, 0.001f);
    }
    out[gid] = x;
}
"""

FP64_FMA = """
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
__kernel void fp64_fma(__global const double *a,
                       __global const double *b,
                       __global double *out,
                       const int inner_loops)
{
    int gid = get_global_id(0);
    double x = a[gid];
    double y = b[gid];
    for (int i = 0; i < inner_loops; i++) {
        x = fma(x, y, 0.001);
    }
    out[gid] = x;
}
"""

# STREAM triad: out = a + scalar * b. One arithmetic op per 12 bytes moved,
# so this measures memory bandwidth rather than compute.
MEMORY_TRIAD = """
__kernel void triad(__global const float *a,
                    __global const float *b,
                    __global float *out,
                    const float scalar)
{
    int gid = get_global_id(0);
    out[gid] = a[gid] + scalar * b[gid];
}
"""

# Naive tiled matrix multiply. Not a competitive GEMM implementation, and not
# meant to be: it is a comparable fixed workload across vendors.
MATRIX_MULTIPLY = """
#define TILE 16
__kernel void matmul(__global const float *A,
                     __global const float *B,
                     __global float *C,
                     const int N)
{
    __local float Asub[TILE][TILE];
    __local float Bsub[TILE][TILE];

    int lrow = get_local_id(0);
    int lcol = get_local_id(1);
    int row = get_group_id(0) * TILE + lrow;
    int col = get_group_id(1) * TILE + lcol;

    float acc = 0.0f;
    int tiles = N / TILE;

    for (int t = 0; t < tiles; t++) {
        Asub[lrow][lcol] = A[row * N + (t * TILE + lcol)];
        Bsub[lrow][lcol] = B[(t * TILE + lrow) * N + col];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE; k++) {
            acc += Asub[lrow][k] * Bsub[k][lcol];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[row * N + col] = acc;
}
"""
