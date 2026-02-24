"""SQLite store for paper metadata."""
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Session


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content_hash = Column(String(64), unique=True, nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    title = Column(Text, nullable=True)
    authors = Column(Text, nullable=True)  # comma-separated
    year = Column(Integer, nullable=True)
    doi = Column(String(256), nullable=True)
    abstract = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)
    ingested_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    indexed = Column(Boolean, default=False)


class MetadataStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)

    def hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def is_ingested(self, content_hash: str) -> bool:
        with Session(self.engine) as session:
            return session.execute(
                select(Paper).where(Paper.content_hash == content_hash)
            ).first() is not None

    def add_paper(self, content_hash: str, filename: str, **kwargs) -> int:
        with Session(self.engine) as session:
            paper = Paper(content_hash=content_hash, filename=filename, **kwargs)
            session.add(paper)
            session.commit()
            session.refresh(paper)
            return paper.id

    def mark_indexed(self, paper_id: int, chunk_count: int):
        with Session(self.engine) as session:
            paper = session.get(Paper, paper_id)
            if paper:
                paper.indexed = True
                paper.chunk_count = chunk_count
                session.commit()

    def list_papers(self) -> list[Paper]:
        with Session(self.engine) as session:
            return list(session.execute(select(Paper)).scalars())
