from monitoring.live_monitor import TimelineCollector
from benchmarks.cpu_test import run_cpu_test


def main():
    collector = TimelineCollector(interval=1)

    print("Starting timeline collector...")
    collector.start()

    result = run_cpu_test()

    collector.stop()

    print("\nCPU RESULT:")
    print(result)

    print("\nTIMELINE LOGS:")
    print(collector.get_logs())


if __name__ == "__main__":
    main()