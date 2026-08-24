from ai.stability_engine import calculate_stability

cpu_log = [0.0, 93.5]

score = calculate_stability(cpu_log)

print("Stability Score:", score)