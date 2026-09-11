#!/usr/bin/env python3
"""
run.py

BenchMind command line entry point.

Examples:

    python run.py bench                      # standard CPU + GPU run
    python run.py bench --mode full --scaling
    python run.py serve                      # API + dashboard on :8000
    python run.py check                      # is now a good time to benchmark?
    python run.py repeat --runs 3            # repeatability check
    python run.py history
    python run.py env

`run.py bench` prints a readable report and exits non-zero when the validity
gate marks the run unusable, so it can be wired into CI or a scheduled task.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import sys
import time
from typing import Any, Dict


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )


def _bar(fraction: float, width: int = 28) -> str:
    filled = int(fraction * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def cmd_bench(args: argparse.Namespace) -> int:
    from api.main import BenchmarkRequest, _execute_benchmark

    req = BenchmarkRequest(
        mode=args.mode,
        include_gpu=not args.no_gpu,
        include_scaling=args.scaling,
        skip_validity_gate=args.force,
        save_to_history=not args.no_save,
        llm_report=args.llm,
    )

    last = [""]

    def progress(stage: str, fraction: float) -> None:
        line = f"\r  {_bar(fraction)} {fraction:5.0%}  {stage:<28}"
        if line != last[0]:
            sys.stdout.write(line)
            sys.stdout.flush()
            last[0] = line

    print("BenchMind 2.0")
    started = time.time()
    result = _execute_benchmark(req, progress=progress)
    print(f"\r  {_bar(1.0)}  100%  done in {time.time() - started:.1f}s" + " " * 20)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    _print_report(result)
    return 0 if result.get("validity", {}).get("verdict") != "invalid" else 2


def _print_report(result: Dict[str, Any]) -> None:
    cpu = result.get("cpu_benchmark", {}) or {}
    system = result.get("system", {}) or {}
    summary = result.get("summary", {}) or {}
    env = result.get("environment", {}) or {}

    def rule(title: str = "") -> None:
        print("\n" + (f"-- {title} " + "-" * max(0, 58 - len(title)) if title else "-" * 62))

    rule("machine")
    print(f"  {system.get('cpu')}")
    print(f"  {system.get('logical_cores')} threads, {system.get('ram')} GB, {system.get('os')}")
    print(f"  environment {env.get('fingerprint_hash')}")

    rule("score")
    ci = cpu.get("cpu_index_ci_pct") or 0
    index = cpu.get("cpu_index") or 0
    print(f"  CPU Index        {index:>10,.0f}  +/- {ci:.1f}%")
    if ci:
        print(f"                   {'':>10}  95% CI {index * (1 - ci / 100):,.0f} to "
              f"{index * (1 + ci / 100):,.0f}")
    print(f"  Single core      {cpu.get('single_core_score', 0):>10,.0f}  "
          f"+/- {cpu.get('single_core_ci_pct', 0):.1f}%")
    print(f"  All cores        {cpu.get('multi_core_score', 0):>10,.0f}  "
          f"+/- {cpu.get('multi_core_ci_pct', 0):.1f}%")
    print("  1000 points = the BenchMind reference machine.")

    rule("by workload (single thread)")
    for name, data in (cpu.get("category_scores") or {}).items():
        if not isinstance(data, dict) or not data.get("single_core"):
            continue
        flag = "" if data.get("counted_in_index") else "  (not in index)"
        print(f"  {name:<16} {data['single_core']:>9,.0f}  "
              f"{data.get('single_core_raw', 0):>10,.1f} {data.get('raw_metric_name', ''):<10}{flag}")

    gpu = result.get("gpu_benchmark") or {}
    devices = gpu.get("devices") if isinstance(gpu, dict) else None
    if devices:
        rule("gpu")
        for d in devices:
            print(f"  {d.get('gpu_name')}: {d.get('gpu_score', 0):,.0f} "
                  f"(+/- {d.get('gpu_score_ci_pct', 0):.1f}%)")
            for w in d.get("workloads", []):
                if w.get("raw_metric_value"):
                    print(f"    {w.get('name', ''):<44} "
                          f"{w['raw_metric_value']:>9,.1f} {w.get('raw_metric_name', '')}")
    elif isinstance(gpu, dict) and gpu.get("reason"):
        rule("gpu")
        print(f"  {gpu['reason']}")

    rule("findings")
    for line in summary.get("insights", []):
        print(f"  - {line}")
    for warn in summary.get("warnings", []):
        print(f"  ! {warn}")

    regression = result.get("regression")
    if regression and regression.get("message"):
        rule("against your own history")
        print(f"  {regression['message']}")

    if result.get("llm_report"):
        rule("written analysis")
        for para in result["llm_report"].split("\n\n"):
            print(f"  {para}")

    print()


def cmd_check(args: argparse.Namespace) -> int:
    from monitoring.validity import check_run_validity, describe_verdict

    report = check_run_validity(sample_seconds=args.seconds)
    print(f"Verdict: {report.verdict}")
    print(describe_verdict(report))
    print(f"  background CPU   {report.background_cpu_pct}%")
    print(f"  free RAM         {report.free_ram_pct}%")
    print(f"  power            {'mains' if report.on_ac_power else 'BATTERY'}")
    print(f"  CPU temperature  {report.start_cpu_temp if report.start_cpu_temp else 'no sensor'}")
    for issue in report.issues:
        print(f"  [{issue.severity}] {issue.message}")
    for proc in report.top_processes:
        print(f"      {proc['name']} (pid {proc['pid']}): {proc['cpu_percent']}%")
    return 0 if report.verdict == "valid" else 1


def cmd_repeat(args: argparse.Namespace) -> int:
    from scripts.repeatability import run_repeatability

    result = run_repeatability(runs=args.runs, mode=args.mode, cooldown=args.cooldown)
    return 0 if result.get("verdict") in ("excellent", "good") else 1


def cmd_history(args: argparse.Namespace) -> int:
    from monitoring.environment import get_environment_fingerprint
    from storage.history import get_store

    fingerprint = None if args.all else get_environment_fingerprint()["fingerprint_hash"]
    runs = get_store().recent_runs(limit=args.limit, fingerprint_hash=fingerprint)
    if not runs:
        print("No stored runs" + ("." if args.all else " for this software stack."))
        return 0

    print(f"{'when':<20} {'mode':<9} {'index':>8} {'single':>8} {'multi':>8} {'peak C':>7}  state")
    for r in runs:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["created_at"]))
        peak = f"{r['peak_cpu_temp']:.0f}" if r.get("peak_cpu_temp") else "-"
        state = "throttled" if r.get("throttled") else r.get("validity_verdict", "")
        print(f"{when:<20} {r.get('mode', ''):<9} {r.get('cpu_index') or 0:>8,.0f} "
              f"{r.get('single_core_score') or 0:>8,.0f} {r.get('multi_core_score') or 0:>8,.0f} "
              f"{peak:>7}  {state}")
    return 0


def cmd_env(args: argparse.Namespace) -> int:
    from monitoring.environment import describe_fingerprint, get_environment_fingerprint

    fingerprint = get_environment_fingerprint(refresh=True)
    if args.json:
        print(json.dumps(fingerprint, indent=2, default=str))
        return 0

    print(describe_fingerprint(fingerprint))
    print(f"fingerprint {fingerprint['fingerprint_hash']}")
    print("\nScores are only comparable between runs with the same fingerprint.")
    print(f"  Python   {fingerprint['python']['version']} "
          f"({fingerprint['python']['implementation']})")
    print(f"  NumPy    {fingerprint['numpy'].get('version')}")
    for b in fingerprint.get("blas_backends", []):
        print(f"  BLAS     {b.get('internal_api')} {b.get('version')} "
              f"({b.get('num_threads')} threads)")
    print(f"  OpenSSL  {fingerprint.get('openssl')}")
    print(f"  zlib     {fingerprint.get('zlib')}")
    power = fingerprint.get("power", {})
    if power.get("has_battery"):
        print(f"  Power    {'mains' if power['on_ac_power'] else 'BATTERY - scores will be low'}")
    if fingerprint.get("windows_power_plan"):
        print(f"  Plan     {fingerprint['windows_power_plan']}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Install with: pip install -r requirements.txt")
        return 1

    print(f"Dashboard: http://{args.host}:{args.port}/dashboard")
    print(f"API docs:  http://{args.host}:{args.port}/docs")
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmind", description="BenchMind hardware benchmarking suite.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show log output.")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("bench", help="Run the benchmark.")
    bench.add_argument("--mode", choices=["quick", "standard", "full"], default="standard")
    bench.add_argument("--no-gpu", action="store_true", help="Skip the GPU suite.")
    bench.add_argument("--scaling", action="store_true", help="Sweep thread counts.")
    bench.add_argument("--force", action="store_true",
                       help="Run even when conditions are unusable.")
    bench.add_argument("--no-save", action="store_true", help="Do not write to local history.")
    bench.add_argument("--llm", action="store_true",
                       help="Ask a model for a narrative (needs ANTHROPIC_API_KEY).")
    bench.add_argument("--json", action="store_true", help="Emit raw JSON.")
    bench.set_defaults(func=cmd_bench)

    check = sub.add_parser("check", help="Check whether now is a good time to benchmark.")
    check.add_argument("--seconds", type=float, default=3.0)
    check.set_defaults(func=cmd_check)

    repeat = sub.add_parser("repeat", help="Run several times and report repeatability.")
    repeat.add_argument("--runs", type=int, default=3)
    repeat.add_argument("--mode", choices=["quick", "standard", "full"], default="standard")
    repeat.add_argument("--cooldown", type=float, default=90.0,
                        help="Seconds of idle between runs so heat does not carry over. "
                             "Raised from 30s in 2.0.1: on a thin laptop chassis 30s was "
                             "not enough, and scores drifted down across the session.")
    repeat.set_defaults(func=cmd_repeat)

    history = sub.add_parser("history", help="Show stored runs.")
    history.add_argument("--limit", type=int, default=20)
    history.add_argument("--all", action="store_true",
                         help="Include runs from other software stacks.")
    history.set_defaults(func=cmd_history)

    env = sub.add_parser("env", help="Show the environment fingerprint.")
    env.add_argument("--json", action="store_true")
    env.set_defaults(func=cmd_env)

    serve = sub.add_parser("serve", help="Start the API and dashboard.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    # Required before any ProcessPoolExecutor use when frozen on Windows.
    multiprocessing.freeze_support()
    sys.exit(main())
