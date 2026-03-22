from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime

from core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(10), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    avatar_path = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
