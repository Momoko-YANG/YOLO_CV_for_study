# -*- coding: utf-8 -*-
"""
SQLite user management — mirrors QtFusion.manager.UserManager
Used by the FastAPI auth service (no Qt dependency).
"""
import sqlite3
import hashlib
import os
from pathlib import Path


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


class UserDatabase:
    MIN_PASSWORD_LEN = 6

    def __init__(self, db_path: str = "UserDatabase.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL,
                    avatar   TEXT DEFAULT ''
                )"""
            )

    # ------------------------------------------------------------------
    def verify_login(self, username: str, password: str) -> int:
        """0 = ok, -1 = user not found, -2 = wrong password."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT password FROM users WHERE username=?", (username,)
            ).fetchone()
        if row is None:
            return -1
        return 0 if row[0] == _hash(password) else -2

    def register(self, username: str, password: str, avatar: str = "") -> int:
        """0 = ok, -1 = already exists, -2 = password too short, -3 = no avatar."""
        if len(password) < self.MIN_PASSWORD_LEN:
            return -2
        if not avatar:
            return -3
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    "INSERT INTO users (username, password, avatar) VALUES (?,?,?)",
                    (username, _hash(password), avatar),
                )
                return 0
            except sqlite3.IntegrityError:
                return -1

    def delete_user(self, username: str, password: str) -> int:
        """0 = ok, -1 = not found, -2 = wrong password."""
        code = self.verify_login(username, password)
        if code != 0:
            return code
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM users WHERE username=?", (username,))
        return 0

    def change_password(self, username: str, new_password: str) -> int:
        """0 = ok, -1 = not found, -2 = password too short."""
        if len(new_password) < self.MIN_PASSWORD_LEN:
            return -2
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE users SET password=? WHERE username=?",
                (_hash(new_password), username),
            )
            return 0 if cur.rowcount else -1

    def change_avatar(self, username: str, password: str, avatar: str) -> int:
        """0 = ok, -1 = not found, -2 = wrong password, -3 = no avatar."""
        if not avatar:
            return -3
        code = self.verify_login(username, password)
        if code != 0:
            return code
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE users SET avatar=? WHERE username=?", (avatar, username)
            )
        return 0

    def get_avatar(self, username: str) -> str:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT avatar FROM users WHERE username=?", (username,)
            ).fetchone()
        return row[0] if row else ""
