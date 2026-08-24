import pyopencl as cl
import numpy as np
import time


# ===== CLEAN GPU NAME =====
def clean_gpu_name(name):

    name = name.strip()

    if "RaptorLake" in name:
        return "Intel UHD Graphics"

    return name


# ===== BENCHMARK ONE GPU =====
def benchmark_device(device, size=10_000_000, loops=15):

    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)

    mf = cl.mem_flags

    a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    result_buf = cl.Buffer(context, mf.WRITE_ONLY, a.nbytes)

    program = cl.Program(context, """
    __kernel void heavy_compute(
        __global const float *a,
        __global const float *b,
        __global float *result)
    {
        int gid = get_global_id(0);

        float x = a[gid];
        float y = b[gid];

        for(int i=0; i<100; i++){
            x = x * y + 0.001f;
        }

        result[gid] = x;
    }
    """).build()

    # ===== GET KERNEL =====
    kernel = cl.Kernel(program, "heavy_compute")

    # ===== FIX: SET ARGUMENTS =====
    kernel.set_arg(0, a_buf)
    kernel.set_arg(1, b_buf)
    kernel.set_arg(2, result_buf)

    # ===== WARMUP =====
    cl.enqueue_nd_range_kernel(queue, kernel, a.shape, None)
    queue.finish()

    # ===== BENCHMARK =====
    start = time.time()

    for _ in range(loops):
        cl.enqueue_nd_range_kernel(queue, kernel, a.shape, None)

    queue.finish()

    elapsed = time.time() - start

    gpu_score = round((size * loops / elapsed) / 100000)

    return {
        "gpu_name": clean_gpu_name(device.name),
        "gpu_score": gpu_score,
        "execution_time": elapsed
    }


# ===== RUN ALL GPUS =====
def run_gpu_test():

    results = []

    for platform in cl.get_platforms():
        for device in platform.get_devices():

            try:
                print("Testing:", clean_gpu_name(device.name))

                result = benchmark_device(device)
                results.append(result)

            except Exception as e:
                print("Skipped:", device.name, "| Reason:", e)

    return results


# ===== RUN DIRECT =====
if __name__ == "__main__":
    print(run_gpu_test())