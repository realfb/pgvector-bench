from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    String,
    Float,
    Text,
    ForeignKey,
    UniqueConstraint,
    Computed,
    Index,
    DDL,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import TSVECTOR, JSONB
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    """Base class for all database models"""

    pass


class User(Base):
    """User model"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

    # Relationships
    documents: Mapped[List["UserDocument"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class UserDocument(Base):
    """User document model - represents a Wikipedia article"""

    __tablename__ = "user_documents"
    __table_args__ = (
        UniqueConstraint("url", name="_document_url_uc"),  # URL must be unique across all users
        UniqueConstraint("wiki_id", name="_document_wiki_id_uc"),  # Wiki ID must be unique
        Index("idx_documents_user_id", "user_id"),
        Index("idx_documents_wiki_id", "wiki_id"),
        Index("idx_documents_meta_gin", "meta", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    wiki_id: Mapped[int] = mapped_column(index=True)
    url: Mapped[str] = mapped_column(String(500))
    title: Mapped[str] = mapped_column(String(500))
    views: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    langs: Mapped[Optional[int]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

    # Metadata column for efficient filtering and retrieval
    # Stores: category, subcategory, difficulty_level, content_type,
    # quality_score, last_updated, editor_count, reference_count, etc.
    meta: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")

    # Relationships
    user: Mapped["User"] = relationship(back_populates="documents")
    chunks: Mapped[List["UserDocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class UserDocumentChunk(Base):
    """Document chunk model - represents a paragraph/section of a document"""

    __tablename__ = "user_document_chunks"
    __table_args__ = (
        UniqueConstraint("user_document_id", "paragraph_id", name="_document_chunk_uc"),
        Index("idx_chunks_document_id", "user_document_id"),
        Index(
            "idx_chunks_text_search_vector_gin",
            "text_search_vector",
            postgresql_using="gin",
        ),
        Index(
            "idx_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 256},
            postgresql_ops={"embedding": "vector_ip_ops"},  # Changed to inner product
        ),
        Index("idx_chunks_meta_gin", "meta", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_document_id: Mapped[int] = mapped_column(ForeignKey("user_documents.id"), index=True)
    paragraph_id: Mapped[int] = mapped_column()
    text: Mapped[str] = mapped_column(Text)

    # Full-text search vector - stored as computed column for efficiency
    text_search_vector: Mapped[TSVECTOR] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )

    # Vector embedding column - 768 dimensions for Cohere embeddings
    embedding: Mapped[Vector] = mapped_column(Vector(768))

    # Metadata column for chunk-level filtering
    # Stores: section_type, position, word_count, has_code, has_math,
    # language_detected, sentiment, complexity_score, etc.
    meta: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")

    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

    # Relationships
    document: Mapped["UserDocument"] = relationship(back_populates="chunks")


# Database extensions and functions
# Create pgvector extension
create_extension = DDL("CREATE EXTENSION IF NOT EXISTS vector")
event.listen(Base.metadata, "before_create", create_extension)

# Create hybrid search function
create_hybrid_search = DDL("""
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text text,
    query_embedding vector(768),
    match_count int,
    search_user_id int DEFAULT NULL,
    full_text_weight float DEFAULT 1,
    semantic_weight float DEFAULT 1,
    rrf_k int DEFAULT 50
)
RETURNS TABLE(
    chunk_id int,
    document_id int,
    user_id int,
    title text,
    text text,
    meta jsonb,
    score float
)
LANGUAGE SQL
AS $$
WITH full_text AS (
    SELECT 
        c.id,
        -- Note: ts_rank_cd is not indexable but will only rank matches of the where clause
        -- which shouldn't be too big
        row_number() OVER (
            ORDER BY ts_rank_cd(c.text_search_vector, websearch_to_tsquery('english', query_text)) DESC
        ) as rank_ix
    FROM user_document_chunks c
    JOIN user_documents d ON c.user_document_id = d.id
    WHERE 
        c.text_search_vector @@ websearch_to_tsquery('english', query_text)
        AND (search_user_id IS NULL OR d.user_id = search_user_id)
    ORDER BY rank_ix
    LIMIT LEAST(match_count, 30) * 2
),
semantic AS (
    SELECT 
        c.id,
        row_number() OVER (ORDER BY c.embedding <#> query_embedding) as rank_ix
    FROM user_document_chunks c
    JOIN user_documents d ON c.user_document_id = d.id
    WHERE (search_user_id IS NULL OR d.user_id = search_user_id)
    ORDER BY rank_ix
    LIMIT LEAST(match_count, 30) * 2
)
SELECT 
    c.id as chunk_id,
    d.id as document_id,
    d.user_id,
    d.title,
    c.text,
    c.meta,
    COALESCE(1.0 / (rrf_k + full_text.rank_ix), 0.0) * full_text_weight +
    COALESCE(1.0 / (rrf_k + semantic.rank_ix), 0.0) * semantic_weight as score
FROM 
    full_text
    FULL OUTER JOIN semantic ON full_text.id = semantic.id
    JOIN user_document_chunks c ON COALESCE(full_text.id, semantic.id) = c.id
    JOIN user_documents d ON c.user_document_id = d.id
ORDER BY score DESC
LIMIT LEAST(match_count, 30)
$$;
""")
event.listen(Base.metadata, "after_create", create_hybrid_search)
