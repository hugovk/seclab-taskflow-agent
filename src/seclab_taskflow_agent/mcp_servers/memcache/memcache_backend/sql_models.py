# SPDX-FileCopyrightText: 2025 GitHub
# SPDX-License-Identifier: MIT

from sqlalchemy import String, Text, Integer, ForeignKey, Column
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, relationship
from typing import Optional


class Base(DeclarativeBase):
    pass


class KeyValue(Base):
    __tablename__ = "key_value_store"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str]
    value: Mapped[str] = mapped_column(Text)

    def __repr__(self):
        return f"<KeyValue(key={self.key}, value={self.value})>"
