from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Any


def environment_metadata() -> dict[str, Any]:
    packages = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "wfdb", "PennyLane"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "packages": packages,
    }

