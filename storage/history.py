"""
storage/history.py

Local benchmark history in SQLite.

Why local and not just Supabase: a benchmark is far more useful as a
diagnostic than as a leaderboard entry. "Your machine scores 4,800" is trivia.
"Your machine scores 12% lower than it did six weeks ago, and it now throttles
40 seconds earlier" is an actionable finding about thermal paste, a clogged
fan, a Windows power plan change or a driver regression.

That comparison is only valid between runs with the same environment
fingerprint and the same benchmark mode, so both are stored and both are
enforced on every query. A run marked `tainted` or `invalid` by the validity
gate is stored for the record but never used as a comparison baseline.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("BenchMind.History")

DEFAULT_DB_PATH = Path(
    os.getenv("BENCHMIND_HOME", Path.home() / ".benchmind")) / "history.db"

# A change larger than this, and outside both runs' confidence intervals,
# is treated as a real regression or improvement rather than noise.
SIGNIFICANT_CHANGE_PCT = 5.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        REAL    NOT NULL,
    benchmind_version TEXT,
    mode              TEXT    NOT NULL,
    fingerprint_hash  TEXT    NOT NULL,
    baseline_version  TEXT,
    validity_verdict  TEXT    NOT NULL DEFAULT 'valid',
    cpu_name          TEXT,
    os_name           TEXT,
    logical_cores     INTEGER,
    cpu_index         REAL,
    cpu_index_ci_pct  REAL,
    single_core_score REAL,
    multi_core_score  REAL,
    gpu_score         REAL,
    stability_score   REAL,
    throttled         INTEGER DEFAULT 0,
    peak_cpu_temp     REAL,
    payload           TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_fingerprint
    ON runs (fingerprint_hash, mode, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_created
    ON runs (created_at DESC);

CREATE TABLE IF NOT EXISTS category_scores (
    run_id            INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    category          TEXT    NOT NULL,
    single_core       REAL,
    multi_core        REAL,
    raw_metric_name   TEXT,
    single_core_raw   REAL,
    PRIMARY KEY (run_id, category)
);
"""


class HistoryStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    # ---------------- writes ----------------
    def save_run(self, result: Dict[str, Any]) -> int:
        """Persist one full benchmark result. Returns the local run id."""
        cpu = result.get("cpu_benchmark", {}) or {}
        system = result.get("system", {}) or {}
        env = result.get("environment", {}) or cpu.get("environment", {}) or {}
        validity = result.get("validity", {}) or {}
        throttle = result.get("throttling", {}) or {}
        stability = result.get("stability", {}) or {}

        gpu_score = 0.0
        gpu = result.get("gpu_benchmark") or {}
        devices = gpu.get("devices") if isinstance(gpu, dict) else gpu
        if devices:
            try:
                gpu_score = max(float(d.get("gpu_score") or 0) for d in devices)
            except (ValueError, TypeError):
                gpu_score = 0.0

        stability_score = (stability.get("overall_score")
                           if isinstance(stability, dict) else stability)

        row = (
            time.time(),
            env.get("benchmind_version", "unknown"),
            cpu.get("mode", "unknown"),
            env.get("fingerprint_hash", "unknown"),
            cpu.get("baseline_version"),
            validity.get("verdict", "valid"),
            system.get("cpu"),
            system.get("os"),
            system.get("logical_cores"),
            cpu.get("cpu_index"),
            cpu.get("cpu_index_ci_pct"),
            cpu.get("single_core_score"),
            cpu.get("multi_core_score"),
            gpu_score,
            stability_score,
            1 if throttle.get("throttling_detected") else 0,
            (cpu.get("telemetry_summary") or {}).get("max_cpu_temp"),
            json.dumps(_strip_bulk(result), default=str),
        )

        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO runs (
                       created_at, benchmind_version, mode, fingerprint_hash,
                       baseline_version, validity_verdict, cpu_name, os_name,
                       logical_cores, cpu_index, cpu_index_ci_pct,
                       single_core_score, multi_core_score, gpu_score,
                       stability_score, throttled, peak_cpu_temp, payload)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                row,
            )
            run_id = int(cur.lastrowid)

            for name, data in (cpu.get("category_scores") or {}).items():
                if not isinstance(data, dict):
                    continue
                conn.execute(
                    """INSERT OR REPLACE INTO category_scores
                       (run_id, category, single_core, multi_core,
                        raw_metric_name, single_core_raw)
                       VALUES (?,?,?,?,?,?)""",
                    (run_id, name, data.get("single_core"), data.get("multi_core"),
                     data.get("raw_metric_name"), data.get("single_core_raw")),
                )

        logger.info("Saved run %d (index=%s, verdict=%s)",
                    run_id, cpu.get("cpu_index"), validity.get("verdict"))
        return run_id

    # ---------------- reads ----------------
    def recent_runs(self, limit: int = 20,
                    fingerprint_hash: Optional[str] = None,
                    mode: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM runs"
        clauses, params = [], []
        if fingerprint_hash:
            clauses.append("fingerprint_hash = ?")
            params.append(fingerprint_hash)
        if mode:
            clauses.append("mode = ?")
            params.append(mode)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        data = _row_to_dict(row)
        try:
            data["payload"] = json.loads(row["payload"])
        except (json.JSONDecodeError, TypeError):
            data["payload"] = None
        return data

    def comparable_baseline(self, fingerprint_hash: str, mode: str,
                            exclude_run_id: Optional[int] = None,
                            window: int = 5) -> Optional[Dict[str, Any]]:
        """
        Median of the most recent VALID runs on the same fingerprint and mode.

        A median over several runs is a far more stable reference than the
        single previous run, which might itself have been a bad sample.
        """
        sql = ("SELECT * FROM runs WHERE fingerprint_hash = ? AND mode = ? "
               "AND validity_verdict = 'valid'")
        params: List[Any] = [fingerprint_hash, mode]
        if exclude_run_id is not None:
            sql += " AND id != ?"
            params.append(exclude_run_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(window)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        if not rows:
            return None

        indices = [r["cpu_index"] for r in rows if r["cpu_index"]]
        if not indices:
            return None

        return {
            "runs_considered": len(indices),
            "median_cpu_index": round(statistics.median(indices), 2),
            "oldest": min(r["created_at"] for r in rows),
            "newest": max(r["created_at"] for r in rows),
            "spread_pct": (round(statistics.stdev(indices) / statistics.fmean(indices) * 100, 2)
                           if len(indices) > 1 else 0.0),
        }

    def detect_regression(self, result: Dict[str, Any],
                          exclude_run_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Compare a fresh result against this machine's own history.

        Returns a verdict of: no_baseline, incomparable, stable, regression or
        improvement. A change is only called significant when it exceeds both
        the threshold and the measurement's own confidence interval.
        """
        cpu = result.get("cpu_benchmark", {}) or {}
        env = result.get("environment", {}) or cpu.get("environment", {}) or {}
        validity = result.get("validity", {}) or {}

        fingerprint = env.get("fingerprint_hash")
        mode = cpu.get("mode")
        current = cpu.get("cpu_index")
        current_ci = cpu.get("cpu_index_ci_pct") or 0.0

        if validity.get("verdict") not in (None, "valid"):
            return {
                "verdict": "incomparable",
                "message": ("This run was flagged by the validity gate, so it is not "
                            "compared against history. Fix the flagged conditions and "
                            "re-run."),
            }
        if not fingerprint or not mode or not current:
            return {"verdict": "no_baseline", "message": "Run is missing fingerprint or score."}

        baseline = self.comparable_baseline(fingerprint, mode, exclude_run_id)
        if baseline is None:
            return {
                "verdict": "no_baseline",
                "message": ("No earlier valid run on this exact software stack. "
                            "This run becomes the baseline for future comparisons."),
            }

        previous = baseline["median_cpu_index"]
        change_pct = ((current - previous) / previous) * 100.0 if previous else 0.0

        # Noise floor: the larger of the fixed threshold and the combined CI.
        combined_ci = (current_ci ** 2 + baseline["spread_pct"] ** 2) ** 0.5
        noise_floor = max(SIGNIFICANT_CHANGE_PCT, combined_ci)

        if abs(change_pct) <= noise_floor:
            verdict = "stable"
            message = (f"Within noise: {change_pct:+.1f}% against a baseline of "
                       f"{previous:,.0f} over {baseline['runs_considered']} run(s). "
                       f"Anything under {noise_floor:.1f}% is not distinguishable.")
        elif change_pct < 0:
            verdict = "regression"
            message = (f"Down {abs(change_pct):.1f}% against this machine's own baseline of "
                       f"{previous:,.0f}. Worth checking: background processes, power plan, "
                       "battery vs mains, dust and thermal paste, or a driver or library "
                       "update.")
        else:
            verdict = "improvement"
            message = (f"Up {change_pct:.1f}% against a baseline of {previous:,.0f}.")

        return {
            "verdict": verdict,
            "current_cpu_index": current,
            "baseline_cpu_index": previous,
            "change_pct": round(change_pct, 2),
            "noise_floor_pct": round(noise_floor, 2),
            "runs_in_baseline": baseline["runs_considered"],
            "baseline_spread_pct": baseline["spread_pct"],
            "message": message,
        }

    def score_timeline(self, fingerprint_hash: Optional[str] = None,
                       mode: str = "standard", limit: int = 50) -> List[Dict[str, Any]]:
        """Points for the history chart in the dashboard."""
        sql = ("SELECT created_at, cpu_index, cpu_index_ci_pct, single_core_score, "
               "multi_core_score, validity_verdict, throttled, peak_cpu_temp "
               "FROM runs WHERE mode = ?")
        params: List[Any] = [mode]
        if fingerprint_hash:
            sql += " AND fingerprint_hash = ?"
            params.append(fingerprint_hash)
        sql += " ORDER BY created_at ASC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def delete_run(self, run_id: int) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            conn.execute("DELETE FROM category_scores WHERE run_id = ?", (run_id,))
        return cur.rowcount > 0


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    data = {k: row[k] for k in row.keys() if k != "payload"}
    data["throttled"] = bool(data.get("throttled"))
    return data


def _strip_bulk(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drop the raw telemetry arrays before storing.

    A five-minute run at 5 Hz is 1,500 samples across six series. Keeping every
    sample for every run makes the database grow without bound for no analytic
    benefit; the derived summary and throttle analysis are retained.
    """
    trimmed = dict(result)
    cpu = dict(trimmed.get("cpu_benchmark") or {})
    cpu.pop("telemetry", None)
    trimmed["cpu_benchmark"] = cpu
    trimmed.pop("timeline", None)
    return trimmed


_default_store: Optional[HistoryStore] = None


def get_store(db_path: Optional[Path] = None) -> HistoryStore:
    """Lazily created module-level store, so importing this never touches disk."""
    global _default_store
    if _default_store is None or db_path is not None:
        _default_store = HistoryStore(db_path)
    return _default_store
