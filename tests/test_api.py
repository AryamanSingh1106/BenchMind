"""
tests/test_api.py

API surface tests using FastAPI's TestClient. No real benchmark is executed;
the pipeline is stubbed so these stay fast and deterministic in CI.
"""

from __future__ import annotations

import time
import unittest
from unittest import mock

try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    FASTAPI_AVAILABLE = False


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi/httpx not installed")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from api.main import app
        cls.app = app

    def test_root(self):
        with TestClient(self.app) as client:
            r = client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["version"], "2.0.0")

    def test_system_info_includes_fingerprint(self):
        with TestClient(self.app) as client:
            r = client.get("/api/system-info")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("environment", body)
        self.assertIn("fingerprint_hash", body["environment"])
        self.assertIn("logical_cores", body)

    def test_dashboard_reports_no_run(self):
        from api.jobs import job_manager
        with mock.patch.object(job_manager, "latest_finished", return_value=None):
            with TestClient(self.app) as client:
                r = client.get("/api/dashboard")
        self.assertEqual(r.json()["status"], "no_benchmark_run")

    def test_bad_mode_rejected(self):
        with TestClient(self.app) as client:
            r = client.post("/api/benchmark", json={"mode": "turbo"})
        self.assertIn(r.status_code, (400, 422))

    def test_benchmark_returns_job_id_immediately(self):
        import api.main as api_main

        def fake(req, progress=None):
            if progress:
                progress("working", 0.5)
            time.sleep(0.2)
            return {"cpu_benchmark": {"cpu_index": 1234}}

        with mock.patch.object(api_main, "_execute_benchmark", fake):
            with TestClient(self.app) as client:
                started = time.time()
                r = client.post("/api/benchmark",
                                json={"mode": "quick", "save_to_history": False})
                elapsed = time.time() - started

                self.assertEqual(r.status_code, 200)
                job_id = r.json()["job_id"]
                self.assertLess(elapsed, 0.15,
                                "POST must not block for the whole benchmark")

                for _ in range(60):
                    job = client.get(f"/api/jobs/{job_id}").json()
                    if job["status"] in ("finished", "failed"):
                        break
                    time.sleep(0.1)

        self.assertEqual(job["status"], "finished")
        self.assertEqual(job["result"]["cpu_benchmark"]["cpu_index"], 1234)

    def test_concurrent_benchmarks_rejected(self):
        import api.main as api_main

        def slow(req, progress=None):
            time.sleep(1.0)
            return {"cpu_benchmark": {}}

        with mock.patch.object(api_main, "_execute_benchmark", slow):
            with TestClient(self.app) as client:
                first = client.post("/api/benchmark", json={"mode": "quick"})
                second = client.post("/api/benchmark", json={"mode": "quick"})
                self.assertEqual(first.status_code, 200)
                self.assertEqual(second.status_code, 409,
                                 "two concurrent benchmarks would invalidate each other")
                for _ in range(30):
                    job = client.get(f"/api/jobs/{first.json()['job_id']}").json()
                    if job["status"] in ("finished", "failed"):
                        break
                    time.sleep(0.1)

    def test_unknown_job_is_404(self):
        with TestClient(self.app) as client:
            r = client.get("/api/jobs/doesnotexist")
        self.assertEqual(r.status_code, 404)

    def test_failed_job_reports_error(self):
        import api.main as api_main

        def boom(req, progress=None):
            raise ValueError("simulated failure")

        with mock.patch.object(api_main, "_execute_benchmark", boom):
            with TestClient(self.app) as client:
                job_id = client.post("/api/benchmark", json={"mode": "quick"}).json()["job_id"]
                for _ in range(30):
                    job = client.get(f"/api/jobs/{job_id}").json()
                    if job["status"] in ("finished", "failed"):
                        break
                    time.sleep(0.1)

        self.assertEqual(job["status"], "failed")
        self.assertIn("simulated failure", job["error"])


class TestDatabaseImportSafety(unittest.TestCase):
    """
    1.x raised ValueError at import time when SUPABASE_URL was missing, which
    crashed every fresh clone. Importing must never require credentials.
    """

    def test_import_without_credentials(self):
        import os
        from db import client as db_client

        db_client.reset_client()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(db_client.get_client())
            self.assertFalse(db_client.is_configured())

    def test_crud_returns_none_when_unconfigured(self):
        import os
        from db import client as db_client, crud

        db_client.reset_client()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(crud.save_system_profile({"cpu": "x"}))
            self.assertEqual(crud.get_recent_benchmarks(), [])
            self.assertFalse(crud.save_timeline_logs("id", {"elapsed": [1]}))


if __name__ == "__main__":
    unittest.main()
