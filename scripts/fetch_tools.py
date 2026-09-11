"""
scripts/fetch_tools.py

Download LibreHardwareMonitor instead of committing it.

BenchMind 1.x shipped roughly 10 MB of LibreHardwareMonitor DLLs and PDBs
inside the repository. Two problems with that:

  * every clone pays for binaries that only work on Windows
  * LibreHardwareMonitor is licensed under MPL-2.0, which carries source and
    notice obligations that a silently vendored binary does not satisfy

This script fetches the release on demand into tools/ and writes the licence
notice next to it. tools/ is gitignored.

Temperature reading is optional: BenchMind runs without it and reports
"no temperature source" rather than pretending the CPU is at 0 degrees.

Usage:
    python -m scripts.fetch_tools
"""

from __future__ import annotations

import io
import platform
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

# The release asset name has changed across versions; try each in turn.
# Older builds shipped as LibreHardwareMonitor-net472.zip and used the WinRing0
# kernel driver, which Windows now blocks by default under the vulnerable
# driver blocklist (CVE-2020-14979). On an affected machine the app launches
# and performance counters work, but every clock, temperature and power sensor
# reads as "-" because the driver never loaded. Newer builds replaced that
# driver, so on Windows 11 the current release is the one you want.
RELEASE_URLS = [
    ("https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/releases/"
     "latest/download/LibreHardwareMonitor-net8.0-windows.zip"),
    ("https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/releases/"
     "latest/download/LibreHardwareMonitor-net472.zip"),
]
TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools" / "LibreHardwareMonitor"

NOTICE = """LibreHardwareMonitor
--------------------
Downloaded by scripts/fetch_tools.py. Not authored by, and not part of,
BenchMind.

Licensed under the Mozilla Public License 2.0.
Source: https://github.com/LibreHardwareMonitor/LibreHardwareMonitor

BenchMind reads temperatures, per-core clocks and CPU package power from its
optional HTTP server. To enable:
  1. Run LibreHardwareMonitor.exe as Administrator
  2. Options -> Remote Web Server -> Run  (default port 8085)

Without it, BenchMind still runs; it reports "no temperature source" and
skips thermal throttle analysis rather than inventing numbers.

If every sensor shows "-" while the app is clearly running as Administrator,
the kernel driver is being blocked. Either use a current release (which no
longer relies on WinRing0) or check Core isolation -> Memory integrity.
"""


def fetch() -> int:
    if platform.system() != "Windows":
        print("LibreHardwareMonitor is Windows-only.")
        print("On Linux, BenchMind reads temperatures through psutil automatically.")
        print("On macOS, no temperature source is currently wired up.")
        return 0

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading LibreHardwareMonitor to {TOOLS_DIR} ...")

    payload = None
    for url in RELEASE_URLS:
        try:
            print(f"  trying {url.rsplit('/', 1)[-1]} ...")
            with urlopen(url, timeout=60) as response:
                payload = response.read()
            break
        except Exception as e:  # noqa: BLE001
            print(f"    not available: {e}")

    if payload is None:
        print("\nDownload failed. Get the latest Windows release manually from:")
        print("  https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/releases")
        print(f"and extract it into {TOOLS_DIR}")
        return 1

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(TOOLS_DIR)

    (TOOLS_DIR / "NOTICE.txt").write_text(NOTICE, encoding="utf-8")
    print(f"Done. {len(list(TOOLS_DIR.iterdir()))} items extracted.")
    print("\nNext: run LibreHardwareMonitor.exe as Administrator and enable")
    print("Options -> Remote Web Server -> Run (port 8085).")
    return 0


if __name__ == "__main__":
    sys.exit(fetch())
