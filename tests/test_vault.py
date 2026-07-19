import tempfile
import os
from datetime import date
from psil.digest.vault import write_digest, ensure_daily_dir


def test_ensure_daily_dir_creates_directory():
    with tempfile.TemporaryDirectory() as tmp:
        vault_path = os.path.join(tmp, "vault")
        daily_dir = ensure_daily_dir(vault_path)
        assert os.path.isdir(daily_dir)
        assert daily_dir.endswith("daily")


def test_write_digest_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        vault_path = os.path.join(tmp, "vault")
        content = "# Test Digest\n\nContent here."
        path = write_digest(vault_path, date(2026, 5, 27), content)
        assert os.path.isfile(path)
        assert path.endswith("2026-05-27-signals.md")
        with open(path) as f:
            assert f.read() == content


def test_write_digest_overwrites_existing():
    with tempfile.TemporaryDirectory() as tmp:
        vault_path = os.path.join(tmp, "vault")
        write_digest(vault_path, date(2026, 5, 27), "first")
        path = write_digest(vault_path, date(2026, 5, 27), "second")
        with open(path) as f:
            assert f.read() == "second"
