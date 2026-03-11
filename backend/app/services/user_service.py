# -*- coding: utf-8 -*-
"""
Auth business logic — mirrors LoginWindow.py (do_login / do_reg / do_forget …)
No Qt imports; called by the FastAPI auth router.
"""
from backend.app.database.user_db import UserDatabase

_db = UserDatabase()

ERROR_MAP = {
    -1: "user_not_found",
    -2: "wrong_password",
    -3: "no_avatar",
}


def login(username: str, password: str) -> dict:
    code = _db.verify_login(username, password)
    if code == 0:
        avatar = _db.get_avatar(username)
        return {"ok": True, "username": username, "avatar_path": avatar}
    return {"ok": False, "error": ERROR_MAP.get(code, "unknown")}


def register(username: str, password: str, avatar: str, captcha_ok: bool) -> dict:
    if not captcha_ok:
        return {"ok": False, "error": "wrong_captcha"}
    code = _db.register(username, password, avatar)
    if code == 0:
        return {"ok": True}
    return {"ok": False, "error": ERROR_MAP.get(code, "unknown")}


def delete_account(username: str, password: str) -> dict:
    code = _db.delete_user(username, password)
    if code == 0:
        return {"ok": True}
    return {"ok": False, "error": ERROR_MAP.get(code, "unknown")}


def change_password(username: str, new_password: str) -> dict:
    code = _db.change_password(username, new_password)
    if code == 0:
        return {"ok": True}
    return {"ok": False, "error": ERROR_MAP.get(code, "unknown")}


def change_avatar(username: str, password: str, avatar: str) -> dict:
    code = _db.change_avatar(username, password, avatar)
    if code == 0:
        return {"ok": True, "avatar_path": avatar}
    return {"ok": False, "error": ERROR_MAP.get(code, "unknown")}
