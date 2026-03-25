import sqlite3
from datetime import datetime

DB_NAME = "usuarios.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            nome         TEXT UNIQUE NOT NULL,
            documento    TEXT DEFAULT '',
            departamento TEXT DEFAULT '',
            cargo        TEXT DEFAULT '',
            email        TEXT DEFAULT '',
            data_cadastro TEXT DEFAULT ''
        )
    """)
    conn.commit()

    # Migração: adiciona colunas se não existirem (banco antigo)
    existing = {row[1] for row in c.execute("PRAGMA table_info(usuarios)")}
    for col in ("documento", "departamento", "cargo", "email", "data_cadastro"):
        if col not in existing:
            c.execute(f"ALTER TABLE usuarios ADD COLUMN {col} TEXT DEFAULT ''")
    conn.commit()
    conn.close()


def add_user(nome: str, documento: str = "", departamento: str = "",
             cargo: str = "", email: str = "") -> int:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        """INSERT OR IGNORE INTO usuarios
           (nome, documento, departamento, cargo, email, data_cadastro)
           VALUES (?,?,?,?,?,?)""",
        (nome, documento, departamento, cargo, email,
         datetime.now().strftime("%d/%m/%Y %H:%M"))
    )
    uid = c.lastrowid
    conn.commit()
    conn.close()
    return uid


def remove_user(nome: str) -> None:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM usuarios WHERE nome = ?", (nome,))
    conn.commit()
    conn.close()


def list_users() -> list[tuple[int, str]]:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, nome FROM usuarios ORDER BY nome")
    users = c.fetchall()
    conn.close()
    return users


def list_users_full() -> list[dict]:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM usuarios ORDER BY nome")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_user_info(nome: str) -> dict | None:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM usuarios WHERE nome = ?", (nome,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None
