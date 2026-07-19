import os
from datetime import date


def ensure_daily_dir(vault_path: str) -> str:
    daily_dir = os.path.join(vault_path, "daily")
    os.makedirs(daily_dir, exist_ok=True)
    return daily_dir


def write_digest(vault_path: str, digest_date: date, content: str) -> str:
    daily_dir = ensure_daily_dir(vault_path)
    filename = f"{digest_date.strftime('%Y-%m-%d')}-signals.md"
    filepath = os.path.join(daily_dir, filename)
    with open(filepath, "w") as f:
        f.write(content)
    return filepath
