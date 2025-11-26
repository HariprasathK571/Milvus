from datetime import datetime
from sqlmodel import Column, Field, SQLModel
from sqlalchemy import String, DateTime, UniqueConstraint, func
import sqlalchemy.dialects.postgresql as pg


class UploadedFile(SQLModel, table=True):
    __tablename__ = "uploaded_files"
    
    id: int = Field(
        sa_column=Column(pg.INTEGER, primary_key=True, index=True, autoincrement=True)
    )
    file_hash: str = Field(
        sa_column=Column(String(64), nullable=False, unique=True, index=True)
    )
    original_filename: str = Field(
        sa_column=Column(String(512), nullable=False)
    )
    stored_filename: str = Field(
        sa_column=Column(String(512), nullable=False)
    )
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    
    __table_args__ = (
        UniqueConstraint("file_hash", name="uq_uploaded_files_file_hash"),
    )
    
    def __repr__(self):
        return f"<UploadedFile {self.original_filename}>"