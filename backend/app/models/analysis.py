import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base import Base


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.id"), unique=True)

    quality_score: Mapped[float] = mapped_column(Float)
    quality_issues: Mapped[list] = mapped_column(JSON, default=list)

    segmentation_map_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    layer_thickness: Mapped[dict] = mapped_column(JSON, default=dict)

    diagnoses: Mapped[list] = mapped_column(JSON, default=list)  # [{code, label, probability}]

    report_text: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    study: Mapped["Study"] = relationship(back_populates="result")
