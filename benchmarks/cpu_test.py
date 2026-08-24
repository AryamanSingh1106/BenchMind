import time
import multiprocessing

multiprocessing.freeze_support()


# ===== WORKLOAD =====
def cpu_worker(iterations):
    result = 0
    for i in range(iterations):
        result += (i * i) % 97
    return result


# ===== WARM-UP =====
def warmup_cpu(seconds=2):
    end_time = time.time() + seconds
    while time.time() < end_time:
        cpu_worker(1_000_000)


# ===== SINGLE CORE =====
def run_single_core_test(iterations=80_000_000):

    start = time.time()

    result = cpu_worker(iterations)

    elapsed = time.time() - start

    raw_score = result / elapsed
    score = round(raw_score / 500000)

    return score, elapsed


# ===== MULTI CORE =====
def run_multi_core_test(total_iterations=1_000_000_000):

    logical_cores = multiprocessing.cpu_count()
    iterations_per_core = total_iterations // logical_cores

    start = time.time()

    with multiprocessing.Pool(processes=logical_cores) as pool:
        results = pool.map(
            cpu_worker,
            [iterations_per_core] * logical_cores
        )

    elapsed = time.time() - start

    raw_score = sum(results) / elapsed
    score = round(raw_score / 500000)

    return score, elapsed, logical_cores


# ===== MAIN CPU TEST =====
def run_cpu_test():

    # warm-up first
    warmup_cpu(seconds=2)

    single_score, single_time = run_single_core_test()
    multi_score, multi_time, cores = run_multi_core_test()

    return {
        "single_core_score": single_score,
        "multi_core_score": multi_score,
        "single_core_time": single_time,
        "multi_core_time": multi_time,
        "cores_used": cores,
    }


if __name__ == "__main__":
    print(run_cpu_test())