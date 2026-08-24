def generate_summary(cpu_result, gpu_result, stability):

    summary = []

    # ===== CPU ANALYSIS =====
    multi = cpu_result.get("multi_core_score", 0)
    single = cpu_result.get("single_core_score", 0)

    if multi > 12000:
        summary.append("🔥 High-performance multi-core CPU detected.")
    elif multi > 7000:
        summary.append("⚡ Good multi-core performance.")
    else:
        summary.append("🟡 Entry-level multi-core performance.")

    if single > 1500:
        summary.append("🚀 Strong single-core responsiveness.")
    else:
        summary.append("🟡 Average single-core performance.")

    # ===== GPU ANALYSIS =====
    if gpu_result:

        best_gpu = max(gpu_result, key=lambda x: x["gpu_score"])
        best_score = best_gpu["gpu_score"]

        summary.append(f"🎮 Primary GPU: {best_gpu['gpu_name']}")

        if best_score > 100000:
            summary.append("🔥 Dedicated high-performance GPU detected.")
        elif best_score > 30000:
            summary.append("⚡ Mid-range GPU performance.")
        else:
            summary.append("🟡 Integrated or entry-level GPU performance.")

    # ===== STABILITY =====
    if stability >= 90:
        summary.append("🧊 Excellent stability during benchmark.")
    elif stability >= 70:
        summary.append("👍 System stability is good.")
    else:
        summary.append("⚠️ Performance fluctuations detected.")

    # ===== FINAL RATING =====
    overall = (multi / 1000) + (stability / 10)

    if overall > 25:
        rating = "🏆 High Performance System"
    elif overall > 15:
        rating = "⚡ Balanced Performance System"
    else:
        rating = "🟡 Basic Performance System"

    return {
        "rating": rating,
        "insights": summary
    }