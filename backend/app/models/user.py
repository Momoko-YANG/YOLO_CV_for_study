# -*- coding: utf-8 -*-
"""Pydantic schemas for auth endpoints."""
from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    captcha: str
    avatar_path: str = ""


class ChangePasswordRequest(BaseModel):
    username: str
    new_password: str
    captcha: str


class ChangeAvatarRequest(BaseModel):
    username: str
    password: str
    avatar_path: str


class UserResponse(BaseModel):
    username: str
    avatar_path: str
    message: str = "ok"
