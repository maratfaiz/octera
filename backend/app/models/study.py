import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base import Base


class Study(Base):
    __tablename__ = "studies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"))
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    eye: Mapped[str | None] = mapped_column(String(8), nullable=True)  # OD / OS
    status: Mapped[str] = mapped_column(String(32), default="uploaded")  # uploaded, processing, completed, failed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    patient: Mapped["Patient"] = relationship(back_populates="studies")
    result: Mapped["AnalysisResult | None"] = relationship(
        back_populates="study", cascade="all, delete-orphan", uselist=False
    )
