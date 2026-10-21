"""Consistent SQLite backups with the sqlite3 backup API, keeping the newest N."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

KEEP = 14


def sqlite_path(database_url: str) -> Path:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError("backups only support sqlite:/// URLs")
    return Path(database_url[len(prefix) :])


def backup(database_url: str, backup_dir: str, keep: int = KEEP) -> Path:
    src = sqlite_path(database_url)
    if not src.is_file():
        raise FileNotFoundError(src)
    out_dir = Path(backup_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    dest = out_dir / f"{src.stem}-{stamp}.db"
    s = sqlite3.connect(src)
    d = sqlite3.connect(dest)
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()
    dest.chmod(0o600)
    old = sorted(out_dir.glob(f"{src.stem}-*.db"))[:-keep] if keep > 0 else []
    for p in old:
        p.unlink()
    return dest
