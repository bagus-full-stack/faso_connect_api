# history/database.py
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlmodel import create_engine, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fasoconnect.db")

# ── Engine ────────────────────────────────────────────────────────────
# SQLite pour le dev, PostgreSQL pour la prod :
# DATABASE_URL=postgresql://user:pass@localhost:5432/fasoconnect

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    echo=False,           # True pour voir les requêtes SQL dans les logs
    connect_args=connect_args,
)

_ROOT = Path(__file__).resolve().parent.parent


def create_db_tables():
    """Applique les migrations Alembic (seule source du schéma)."""
    cfg = Config(str(_ROOT / "migrations" / "alembic.ini"))
    cfg.set_main_option("script_location", str(_ROOT / "migrations"))
    cfg.attributes["configure_logger"] = False  # garde le logging JSON de l'API
    tables = inspect(engine).get_table_names()
    if "history" in tables and "alembic_version" not in tables:
        # base créée avant Alembic (ancien create_all) : on la marque comme à jour
        command.stamp(cfg, "head")
    else:
        command.upgrade(cfg, "head")


def get_session():
    """Générateur de session SQLModel — utilisé via Depends() dans FastAPI."""
    with Session(engine) as session:
        yield session
