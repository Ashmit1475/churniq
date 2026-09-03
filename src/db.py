"""Database helpers. One place that knows how to talk to SQL.

Uses SQLAlchemy when it is installed (required for Postgres). Falls back to the
stdlib `sqlite3` driver for local SQLite URLs so the pipeline still runs on a
bare Python install with only pandas available.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Union

import pandas as pd

from config import Config

try:
    from sqlalchemy import create_engine, text

    HAS_SQLALCHEMY = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_SQLALCHEMY = False

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlalchemy.engine import Engine

# Either half of the fallback: a SQLAlchemy Engine, or a stdlib sqlite3
# Connection. pandas accepts both, which is all this module asks of them.
Connectable = Union["Engine", sqlite3.Connection]


def _sqlite_path(url: str) -> str:
    return url.replace("sqlite:///", "", 1)


def get_engine(cfg: Config) -> Connectable:
    """Return a SQLAlchemy Engine, or a sqlite3 Connection as a fallback."""
    url = cfg.db_url
    if HAS_SQLALCHEMY:
        return create_engine(url, future=True)

    if not url.startswith("sqlite:///"):
        raise RuntimeError(
            "SQLAlchemy is required for non-SQLite databases.\n"
            "Install it with:  pip install sqlalchemy psycopg2-binary"
        )
    path = _sqlite_path(url)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path, check_same_thread=False)


def write_table(
    df: pd.DataFrame, table: str, engine: Connectable, chunksize: int = 1000
) -> None:
    """Replace `table` with `df`. Chunked so the insert's memory does not track
    the size of the frame - the predictions table is the whole scored base."""
    df.to_sql(table, engine, if_exists="replace", index=False, chunksize=chunksize)


def read_sql(query: str, engine: Connectable) -> pd.DataFrame:
    if HAS_SQLALCHEMY and not isinstance(engine, sqlite3.Connection):
        return pd.read_sql(text(query), engine)
    return pd.read_sql_query(query, engine)


def create_indexes(engine: Connectable, specs: list[tuple[str, str]]) -> None:
    """Create indexes if they do not already exist. specs: [(table, column), ...]"""
    statements = [
        f'CREATE INDEX IF NOT EXISTS idx_{t.lower()}_{c.lower()} ON {t} ("{c}")'
        for t, c in specs
    ]

    if HAS_SQLALCHEMY and not isinstance(engine, sqlite3.Connection):
        with engine.begin() as conn:
            for stmt in statements:
                try:
                    conn.execute(text(stmt))
                except Exception as exc:  # pragma: no cover - dialect differences
                    print(f"  ! index skipped: {exc}")
    else:
        cur = engine.cursor()
        for stmt in statements:
            try:
                cur.execute(stmt)
            except Exception as exc:  # pragma: no cover
                print(f"  ! index skipped: {exc}")
        engine.commit()
