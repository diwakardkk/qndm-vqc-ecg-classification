from __future__ import annotations

from pathlib import Path

from .io import atomic_write_text, ensure_dir


def marker_done(path: str | Path) -> bool:
    return Path(path).exists()


def write_marker(path: str | Path, text: str = "complete\n") -> None:
    ensure_dir(Path(path).parent)
    atomic_write_text(path, text)

