"""Programmatic Alembic helpers (used by the CLI, Docker entrypoint and tests)."""

from __future__ import annotations

import os
from pathlib import Path

from alembic import command
from alembic.config import Config


def _backend_root() -> Path:
    """Locate the directory holding ``alembic.ini`` (source checkout, Docker image, or CWD)."""
    candidates = [
        os.environ.get("PAPERPILOT_ALEMBIC_ROOT"),
        str(Path(__file__).resolve().parents[3]),
        os.getcwd(),
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "alembic.ini").exists():
            return Path(candidate)
    raise FileNotFoundError("alembic.ini not found; set PAPERPILOT_ALEMBIC_ROOT")


def alembic_config(database_url: str | None = None) -> Config:
    root = _backend_root()
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    cfg.attributes["configure_logging"] = False
    if database_url:
        cfg.attributes["database_url"] = database_url
    return cfg


def upgrade(database_url: str | None = None, revision: str = "head") -> None:
    command.upgrade(alembic_config(database_url), revision)


def downgrade(database_url: str | None = None, revision: str = "base") -> None:
    command.downgrade(alembic_config(database_url), revision)
