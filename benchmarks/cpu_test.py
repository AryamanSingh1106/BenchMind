import time

def run_cpu_test():
    start = time.time()

    for i in range(10000000):
        x = i*i

    end = time.time()

    return end - start

print(run_cpu_test())