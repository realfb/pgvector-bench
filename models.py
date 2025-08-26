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
        UniqueConstraint("user_id", "url", name="_document_user_url_uc"),
        UniqueConstraint("user_id", "wiki_id", name="_document_user_wiki_id_uc"),
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
        
        # User-specific indexes for efficient filtering
        Index("idx_chunks_user_id", "user_id"),
        Index("idx_chunks_user_document_id", "user_document_id"),
        Index("idx_chunks_text_search_gin", "text_search_vector", postgresql_using="gin"),
        
        # HNSW index for vector search (single column only - HNSW doesn't support composite)
        # The user_id filtering will use idx_chunks_user_id index
        Index(
            "idx_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_ip_ops"},
        ),
        
        # Metadata index
        Index("idx_chunks_meta_gin", "meta", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_document_id: Mapped[int] = mapped_column(ForeignKey("user_documents.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    paragraph_id: Mapped[int] = mapped_column()
    text: Mapped[str] = mapped_column(Text)
    text_search_vector: Mapped[TSVECTOR] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )
    embedding: Mapped[Vector] = mapped_column(Vector(768))
    meta: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

    # Relationships
    document: Mapped["UserDocument"] = relationship(back_populates="chunks")


# Database extensions and functions
# Create pgvector extension
create_extension = DDL("CREATE EXTENSION IF NOT EXISTS vector")
event.listen(Base.metadata, "before_create", create_extension)


create_hybrid_search = DDL("""
CREATE OR REPLACE FUNCTION hybrid_search(
    query_user_id int,
    query_text text,
    query_embedding vector(768),
    match_count int,
    full_text_weight float DEFAULT 1,
    semantic_weight float DEFAULT 1,
    rrf_k int DEFAULT 50
)
RETURNS TABLE (
    id int,
    user_document_id int,
    paragraph_id int,
    text text,
    meta jsonb,
    created_at timestamp with time zone,
    k_score float,
    v_score float,
    score float
)
LANGUAGE SQL
AS $$
WITH full_text AS (
    SELECT 
        id,
        -- Note: ts_rank_cd is not indexable but will only rank matches of the where clause
        -- which shouldn't be too big when filtered by user
        row_number() OVER (
            ORDER BY ts_rank_cd(text_search_vector, websearch_to_tsquery(query_text)) DESC
        ) as rank_ix
    FROM user_document_chunks
    WHERE 
        user_id = query_user_id  -- Filter by user first
        AND text_search_vector @@ websearch_to_tsquery(query_text)
    ORDER BY rank_ix
    LIMIT LEAST(match_count, 30) * 2
),
semantic AS (
    SELECT 
        id,
        row_number() OVER (ORDER BY embedding <#> query_embedding) as rank_ix
    FROM user_document_chunks
    WHERE user_id = query_user_id  -- Filter by user first
    ORDER BY rank_ix
    LIMIT LEAST(match_count, 30) * 2
)
SELECT 
    user_document_chunks.id,
    user_document_chunks.user_document_id,
    user_document_chunks.paragraph_id,
    user_document_chunks.text,
    user_document_chunks.meta,
    user_document_chunks.created_at,
    COALESCE(1.0 / (rrf_k + full_text.rank_ix), 0.0) * full_text_weight AS k_score,
    COALESCE(1.0 / (rrf_k + semantic.rank_ix), 0.0) * semantic_weight AS v_score,
    COALESCE(1.0 / (rrf_k + full_text.rank_ix), 0.0) * full_text_weight +
    COALESCE(1.0 / (rrf_k + semantic.rank_ix), 0.0) * semantic_weight AS score
FROM 
    full_text
    FULL OUTER JOIN semantic ON full_text.id = semantic.id
    JOIN user_document_chunks ON COALESCE(full_text.id, semantic.id) = user_document_chunks.id
ORDER BY 
    score DESC
LIMIT LEAST(match_count, 30)
$$;
""")
event.listen(Base.metadata, "after_create", create_hybrid_search)
