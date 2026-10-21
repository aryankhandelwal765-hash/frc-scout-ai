import sqlite3

from app.services import backup


def make_db(path):
    c = sqlite3.connect(path)
    c.execute("create table t (x integer)")
    c.execute("insert into t values (7)")
    c.commit()
    c.close()


def test_backup_is_a_readable_copy_and_prunes(tmp_path, monkeypatch):
    src = tmp_path / "scout.db"
    make_db(src)
    url = f"sqlite:///{src}"
    stamps = iter(f"20260101-0000{i:02d}" for i in range(5))

    class FakeNow:
        @staticmethod
        def now(_tz):
            class D:
                @staticmethod
                def strftime(_f):
                    return next(stamps)

            return D

    monkeypatch.setattr(backup, "datetime", FakeNow)
    out = [backup.backup(url, str(tmp_path / "b"), keep=3) for _ in range(5)]
    left = sorted((tmp_path / "b").glob("scout-*.db"))
    assert left == out[-3:]
    assert sqlite3.connect(left[-1]).execute("select x from t").fetchone() == (7,)
    assert oct(left[-1].stat().st_mode & 0o777) == "0o600"


def test_missing_database_is_an_error(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        backup.backup(f"sqlite:///{tmp_path / 'nope.db'}", str(tmp_path / "b"))
