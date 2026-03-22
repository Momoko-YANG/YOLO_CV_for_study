import os
import shutil
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.database import get_db
from core.dependencies import get_current_user
from core.security import hash_password, verify_password, create_access_token
from models.schemas import (
    UserRegister, UserLogin, UserResponse, TokenResponse, ChangePasswordRequest,
)
from models.user import User

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
async def register(data: UserRegister, db: AsyncSession = Depends(get_db)):
    if len(data.password) < 3:
        raise HTTPException(status_code=400, detail="Password too short")
    if len(data.username) > 10:
        raise HTTPException(status_code=400, detail="Username too long (max 10)")

    result = await db.execute(select(User).where(User.username == data.username))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already exists")

    user = User(
        username=data.username,
        password_hash=hash_password(data.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_access_token({"sub": user.username})
    return TokenResponse(
        access_token=token,
        user=UserResponse(id=user.id, username=user.username, avatar_url=user.avatar_path),
    )


@router.post("/login", response_model=TokenResponse)
async def login(data: UserLogin, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == data.username))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not registered")
    if not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect password")

    token = create_access_token({"sub": user.username})
    return TokenResponse(
        access_token=token,
        user=UserResponse(id=user.id, username=user.username, avatar_url=user.avatar_path),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)):
    return UserResponse(id=user.id, username=user.username, avatar_url=user.avatar_path)


@router.put("/change-password")
async def change_password(
    data: ChangePasswordRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not verify_password(data.old_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect old password")
    if len(data.new_password) < 3:
        raise HTTPException(status_code=400, detail="Password too short")

    user.password_hash = hash_password(data.new_password)
    await db.commit()
    return {"message": "Password changed successfully"}


@router.post("/avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    os.makedirs(settings.avatar_dir, exist_ok=True)
    ext = os.path.splitext(file.filename or "avatar.png")[1]
    filename = f"{user.username}_{uuid.uuid4().hex[:8]}{ext}"
    filepath = os.path.join(settings.avatar_dir, filename)

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    user.avatar_path = f"/static/avatars/{filename}"
    await db.commit()
    return {"avatar_url": user.avatar_path}


@router.delete("/account")
async def delete_account(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await db.delete(user)
    await db.commit()
    return {"message": "Account deleted successfully"}
