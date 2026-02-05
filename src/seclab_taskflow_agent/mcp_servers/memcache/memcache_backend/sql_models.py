# SPDX-FileCopyrightText: 2025 GitHub
# SPDX-License-Identifier: MIT

from sqlalchemy import Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class KeyValue(Base):
    __tablename__ = "key_value_store"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str]
    value: Mapped[str] = mapped_column(Text)

    def __repr__(self):
        return f"<KeyValue(key={self.key}, value={self.value})>"
