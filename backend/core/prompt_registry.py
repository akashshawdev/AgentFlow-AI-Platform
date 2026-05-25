"""
AgentFlow - Prompt Registry
Versioned prompt storage with A/B experimentation support.

Design: SQLite-backed (no extra infra), supports:
  - Create/update prompt versions with full history
  - Tag runs with prompt version for experiment tracking
  - A/B split: assign traffic % to each variant
  - Query performance by version to identify regressions
"""
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text, Boolean,
    create_engine, select, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

Base = declarative_base()


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    template = Column(Text, nullable=False)
    system_prompt = Column(Text, nullable=True)
    description = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True)
    traffic_split = Column(Float, default=1.0)  # 0.0-1.0
    created_at = Column(DateTime, default=datetime.utcnow)


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), nullable=False, unique=True)
    prompt_name = Column(String(100), nullable=False)
    prompt_version = Column(String(20), nullable=False)
    query = Column(Text, nullable=False)
    faithfulness = Column(Float, nullable=True)
    relevancy = Column(Float, nullable=True)
    hallucination = Column(Float, nullable=True)
    overall_score = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    verdict = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


@dataclass
class PromptSpec:
    name: str
    version: str
    template: str
    system_prompt: str = ""
    description: str = ""
    traffic_split: float = 1.0


class PromptRegistry:
    """
    Manages prompt versions and experiment tracking.

    A/B example:
        registry.register(PromptSpec("summarizer", "v1.1", ..., traffic_split=0.5))
        registry.register(PromptSpec("summarizer", "v1.2", ..., traffic_split=0.5))
        version = registry.select_version("summarizer")  # traffic-weighted random
    """

    def __init__(self, db_url: str | None = None):
        url = db_url or settings.database_url.replace("sqlite+aiosqlite", "sqlite")
        self.engine = create_engine(url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._seed_defaults()

    def register(self, spec: PromptSpec) -> None:
        """Register or update a prompt version."""
        with self.SessionLocal() as session:
            existing = session.execute(
                select(PromptVersion).where(
                    PromptVersion.name == spec.name,
                    PromptVersion.version == spec.version,
                )
            ).scalar_one_or_none()

            if existing:
                existing.template = spec.template
                existing.system_prompt = spec.system_prompt
                existing.traffic_split = spec.traffic_split
                existing.description = spec.description
                logger.info(f"Updated prompt: {spec.name}@{spec.version}")
            else:
                session.add(PromptVersion(
                    name=spec.name,
                    version=spec.version,
                    template=spec.template,
                    system_prompt=spec.system_prompt,
                    description=spec.description,
                    traffic_split=spec.traffic_split,
                ))
                logger.info(f"Registered new prompt: {spec.name}@{spec.version}")
            session.commit()

    def get(self, name: str, version: str) -> PromptSpec | None:
        with self.SessionLocal() as session:
            row = session.execute(
                select(PromptVersion).where(
                    PromptVersion.name == name,
                    PromptVersion.version == version,
                    PromptVersion.is_active == True,
                )
            ).scalar_one_or_none()
            if not row:
                return None
            return PromptSpec(
                name=row.name,
                version=row.version,
                template=row.template,
                system_prompt=row.system_prompt or "",
                description=row.description or "",
                traffic_split=row.traffic_split,
            )

    def select_version(self, name: str) -> str:
        """Traffic-weighted version selection for A/B experiments."""
        with self.SessionLocal() as session:
            rows = session.execute(
                select(PromptVersion).where(
                    PromptVersion.name == name,
                    PromptVersion.is_active == True,
                )
            ).scalars().all()

        if not rows:
            return "v1.0"

        total = sum(r.traffic_split for r in rows)
        rand = random.uniform(0, total)
        cumulative = 0.0
        for row in rows:
            cumulative += row.traffic_split
            if rand <= cumulative:
                return row.version
        return rows[-1].version

    def list_versions(self, name: str) -> list[dict]:
        with self.SessionLocal() as session:
            rows = session.execute(
                select(PromptVersion).where(PromptVersion.name == name)
            ).scalars().all()
            return [
                {
                    "name": r.name,
                    "version": r.version,
                    "description": r.description,
                    "traffic_split": r.traffic_split,
                    "is_active": r.is_active,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    def log_run(self, run_id: str, prompt_name: str, prompt_version: str,
                query: str, metrics: dict) -> None:
        with self.SessionLocal() as session:
            session.add(ExperimentRun(
                run_id=run_id,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                query=query,
                faithfulness=metrics.get("faithfulness"),
                relevancy=metrics.get("answer_relevancy"),
                hallucination=metrics.get("hallucination_score"),
                overall_score=metrics.get("overall_score"),
                latency_ms=metrics.get("latency_ms"),
                verdict=metrics.get("verdict"),
            ))
            session.commit()

    def get_experiment_summary(self, prompt_name: str) -> list[dict]:
        """Aggregate metrics by version for comparison dashboard."""
        with self.SessionLocal() as session:
            rows = session.execute(
                select(
                    ExperimentRun.prompt_version,
                    func.count(ExperimentRun.id).label("runs"),
                    func.avg(ExperimentRun.overall_score).label("avg_score"),
                    func.avg(ExperimentRun.faithfulness).label("avg_faithfulness"),
                    func.avg(ExperimentRun.hallucination).label("avg_hallucination"),
                    func.avg(ExperimentRun.latency_ms).label("avg_latency_ms"),
                ).where(ExperimentRun.prompt_name == prompt_name)
                .group_by(ExperimentRun.prompt_version)
            ).all()

            return [
                {
                    "version": r.prompt_version,
                    "runs": r.runs,
                    "avg_score": round(r.avg_score or 0, 3),
                    "avg_faithfulness": round(r.avg_faithfulness or 0, 3),
                    "avg_hallucination": round(r.avg_hallucination or 0, 3),
                    "avg_latency_ms": round(r.avg_latency_ms or 0, 1),
                }
                for r in rows
            ]

    def _seed_defaults(self) -> None:
        """Pre-populate default prompt versions."""
        defaults = [
            PromptSpec(
                name="summarizer",
                version="v1.0",
                template="Summarize the following context for the query: {query}\n\nContext: {context}",
                system_prompt="You are a precise technical summarizer.",
                description="Baseline summarizer prompt",
                traffic_split=0.3,
            ),
            PromptSpec(
                name="summarizer",
                version="v1.1",
                template="Given technical documentation chunks, answer: {query}\n\nChunks:\n{context}\n\nProvide a structured JSON response.",
                system_prompt="You are an expert technical writer who produces structured, grounded summaries.",
                description="Structured output with JSON enforcement",
                traffic_split=0.7,
            ),
        ]
        for spec in defaults:
            try:
                self.register(spec)
            except Exception:
                pass
