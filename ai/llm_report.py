"""
ai/llm_report.py

Optional LLM-written analysis.

The `ai/` package in 1.x contained three empty files and a chain of if/else
statements with emoji. This module is the part that makes the folder name
honest: it hands the structured result to a language model and asks for a
narrative a human would actually want to read.

It is strictly optional. If no API key is configured, `generate_llm_report`
returns None and the deterministic summary from summary_engine is used
instead. BenchMind never blocks a benchmark run on a network call.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("BenchMind.LLMReport")

DEFAULT_MODEL = "claude-sonnet-4-6"
API_URL = "https://api.anthropic.com/v1/messages"

SYSTEM_PROMPT = """You are a hardware analyst writing a short report on a benchmark run.

Rules:
- Be specific and quantitative. Cite the actual numbers you were given.
- Every score has a confidence interval. Never claim a difference that is
  smaller than the interval.
- If the run was marked tainted or invalid, lead with that.
- Say what the machine is good at and what limits it. Do not pad.
- Four short paragraphs maximum. No bullet lists, no emoji, no headings.
"""


def _extract_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """Send a compact, relevant slice rather than the whole result blob."""
    cpu = result.get("cpu_benchmark", {}) or {}
    return {
        "system": result.get("system"),
        "environment_summary": (result.get("environment") or {}).get("fingerprint_hash"),
        "cpu_index": cpu.get("cpu_index"),
        "cpu_index_ci_pct": cpu.get("cpu_index_ci_pct"),
        "single_core_score": cpu.get("single_core_score"),
        "multi_core_score": cpu.get("multi_core_score"),
        "cores_used": cpu.get("cores_used"),
        "category_scores": cpu.get("category_scores"),
        "telemetry_summary": cpu.get("telemetry_summary"),
        "throttling": result.get("throttling"),
        "bottleneck": result.get("bottleneck"),
        "scaling": result.get("scaling_analysis"),
        "validity": result.get("validity"),
        "gpu": result.get("gpu_benchmark"),
    }


def generate_llm_report(result: Dict[str, Any],
                        model: str = DEFAULT_MODEL,
                        timeout: float = 45.0) -> Optional[str]:
    """
    Ask a model to write the narrative. Returns None when unavailable, so the
    caller always has a working fallback.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.info("ANTHROPIC_API_KEY not set; skipping LLM report.")
        return None

    try:
        import requests
    except ImportError:
        logger.info("requests not installed; skipping LLM report.")
        return None

    payload = _extract_payload(result)

    try:
        response = requests.post(
            API_URL,
            timeout=timeout,
            headers={
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": model,
                "max_tokens": 900,
                "system": SYSTEM_PROMPT,
                "messages": [{
                    "role": "user",
                    "content": (
                        "Write the report for this benchmark run.\n\n"
                        + json.dumps(payload, indent=2, default=str)
                    ),
                }],
            },
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM report request failed: %s", e)
        return None

    try:
        blocks = data.get("content", [])
        text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return text.strip() or None
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not parse LLM response: %s", e)
        return None
