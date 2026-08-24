import numpy as np


def calculate_stability(cpu_log):
    if len(cpu_log) < 2:
        return 0

    cpu_array = np.array(cpu_log)

    # standard deviation = instability
    variation = np.std(cpu_array)

    # convert to score (0–100)
    stability_score = max(0, 100 - variation)

    return round(stability_score, 2)