"""Admin CLI: python -m app.cli {create-admin,import-event,seed-season,migrate,backup}.

No default credentials exist anywhere; the admin password is prompted (getpass)
or read from the ADMIN_PASSWORD environment variable for scripted use.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import sys

import httpx
from sqlalchemy import select

from app.audit import audit
from app.config import BACKEND_DIR, get_settings
from app.db import make_engine, make_session_factory
from app.models import User
from app.schemas import USERNAME_PATTERN
from app.security import hash_secret
from app.services import seasons, tba


def _session():
    s = get_settings()
    return s, make_session_factory(make_engine(s.database_url))()


def cmd_create_admin(args: argparse.Namespace) -> int:
    username = (args.username or input("Admin username: ")).strip().lower()
    if not re.match(USERNAME_PATTERN, username):
        print("Username must be 3-32 chars of letters, digits, _ . -", file=sys.stderr)
        return 2
    password = os.environ.get("ADMIN_PASSWORD")
    if not password:
        password = getpass.getpass("Admin password: ")
        if password != getpass.getpass("Confirm password: "):
            print("Passwords do not match", file=sys.stderr)
            return 2
    if not 10 <= len(password) <= 128:
        print("Password must be 10-128 characters", file=sys.stderr)
        return 2
    _, db = _session()
    with db:
        if db.scalar(select(User).where(User.username == username)):
            print("User already exists", file=sys.stderr)
            return 1
        user = User(username=username, password_hash=hash_secret(password), role="admin", active=1)
        db.add(user)
        db.flush()
        audit(db, user.id, "user_create", f"username={username} role=admin via=cli")
        db.commit()
    print(f"Created admin '{username}'")
    return 0


def cmd_seed_season(args: argparse.Namespace) -> int:
    _, db = _session()
    with db:
        season, created = seasons.seed_example_season(db, force=args.force)
    print(f"{'Seeded' if created else 'Already present'}: {season.year} ({season.game_name})")
    return 0


def cmd_import_event(args: argparse.Namespace) -> int:
    s, db = _session()
    with db, httpx.Client(base_url=s.tba_base_url, timeout=15.0) as client:
        try:
            summary = tba.import_event(db, client, s.tba_api_key.get_secret_value(), args.event_key)
        except tba.TbaError as e:
            db.rollback()
            print(f"Import failed: {e}", file=sys.stderr)
            return 1
        audit(db, None, "event_import", f"event={args.event_key} via=cli")
        db.commit()
    print(summary.model_dump_json())
    return 0


def cmd_migrate(_args: argparse.Namespace) -> int:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    command.upgrade(cfg, "head")
    return 0


def cmd_backup(args: argparse.Namespace) -> int:
    from app.services import backup

    s = get_settings()
    try:
        dest = backup.backup(s.database_url, args.dir or s.backup_dir, keep=args.keep)
    except (FileNotFoundError, ValueError) as e:
        print(f"Backup failed: {e}", file=sys.stderr)
        return 1
    print(f"Wrote {dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="app.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("create-admin")
    a.add_argument("--username")
    a.set_defaults(fn=cmd_create_admin)
    i = sub.add_parser("import-event")
    i.add_argument("event_key")
    i.set_defaults(fn=cmd_import_event)
    sd = sub.add_parser("seed-season")
    sd.add_argument("--force", action="store_true", help="overwrite the example season")
    sd.set_defaults(fn=cmd_seed_season)
    m = sub.add_parser("migrate")
    m.set_defaults(fn=cmd_migrate)
    b = sub.add_parser("backup", help="consistent copy of the database, keeps the newest 14")
    b.add_argument("--dir")
    b.add_argument("--keep", type=int, default=14)
    b.set_defaults(fn=cmd_backup)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
