# -*- coding: utf-8 -*-
"""
Auth router — /auth/*
Mirrors LoginWindow.py: do_login, do_reg, do_forget, do_avatar, delete account.
"""
import os
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from backend.app.models.user import (
    LoginRequest, RegisterRequest,
    ChangePasswordRequest, ChangeAvatarRequest, UserResponse,
)
from backend.app.services import user_service as svc

router = APIRouter(prefix="/auth", tags=["auth"])

AVATAR_DIR = "backend/static/avatars"
os.makedirs(AVATAR_DIR, exist_ok=True)


@router.post("/login", response_model=UserResponse)
def login(req: LoginRequest):
    result = svc.login(req.username, req.password)
    if not result["ok"]:
        raise HTTPException(status_code=401, detail=result["error"])
    return UserResponse(username=result["username"], avatar_path=result["avatar_path"])


@router.post("/register")
def register(
    username: str = Form(...),
    password: str = Form(...),
    captcha: str = Form(...),
    captcha_answer: str = Form(...),
    avatar: UploadFile = File(None),
):
    avatar_path = ""
    if avatar:
        dest = f"{AVATAR_DIR}/{username}_{avatar.filename}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(avatar.file, f)
        avatar_path = dest

    captcha_ok = captcha.lower() == captcha_answer.lower()
    result = svc.register(username, password, avatar_path, captcha_ok)
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return {"message": "registered"}


@router.post("/change-password")
def change_password(req: ChangePasswordRequest):
    # Captcha verification happens client-side for this endpoint
    result = svc.change_password(req.username, req.new_password)
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return {"message": "password changed"}


@router.post("/change-avatar")
def change_avatar(
    username: str = Form(...),
    password: str = Form(...),
    avatar: UploadFile = File(...),
):
    dest = f"{AVATAR_DIR}/{username}_{avatar.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(avatar.file, f)
    result = svc.change_avatar(username, password, dest)
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return {"avatar_path": result["avatar_path"]}


@router.delete("/account")
def delete_account(req: LoginRequest):
    result = svc.delete_account(req.username, req.password)
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return {"message": "account deleted"}
