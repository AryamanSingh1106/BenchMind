"""
db/client.py

Optional Supabase client for uploading results to a shared leaderboard.

1.x raised ValueError at import time when SUPABASE_URL or SUPABASE_KEY was
missing. That meant `import db.crud` crashed the whole application on any
machine without a .env file, including every fresh clone and every CI run.
Cloud sync is optional; a missing key is a normal state, not a fatal error.

2.0 initialises lazily and reports availability instead of raising.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger("BenchMind.DB")

_client: Optional[Any] = None
_checked = False


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def is_configured() -> bool:
    """True when both credentials are present. Never raises."""
    _load_env()
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"))


def get_client() -> Optional[Any]:
    """
    Return a Supabase client, or None when cloud sync is not configured.

    Callers must handle None. BenchMind works fully offline; the local SQLite
    history in storage/history.py is the primary store.
    """
    global _client, _checked

    if _client is not None:
        return _client
    if _checked:
        return None

    _checked = True
    _load_env()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        logger.info("Supabase not configured; cloud sync disabled. "
                    "Set SUPABASE_URL and SUPABASE_KEY in .env to enable it.")
        return None

    try:
        from supabase import create_client
    except ImportError:
        logger.info("supabase package not installed; cloud sync disabled. "
                    "Install with: pip install supabase")
        return None

    try:
        _client = create_client(url, key)
        logger.info("Supabase client ready.")
        return _client
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not create Supabase client: %s", e)
        return None


def reset_client() -> None:
    """Test hook."""
    global _client, _checked
    _client = None
    _checked = False
