from typing import Optional, List
from dataclasses import dataclass
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


# Dataset schemas
class DocumentItem(BaseModel):
    """Schema for Wikipedia document items from Cohere dataset"""

    id: int
    title: str
    text: str
    url: str
    wiki_id: int
    views: float
    paragraph_id: int
    langs: int
    emb: List[float] = Field(..., description="768-dimensional embedding vector")

    model_config = ConfigDict(from_attributes=True)


# Metadata schemas for JSONB columns
class DocumentMetadata(BaseModel):
    """Metadata schema for documents"""

    category: str = Field(..., description="Main category (e.g., Science, History, Technology)")
    subcategory: Optional[str] = Field(None, description="Subcategory")
    difficulty_level: str = Field(..., description="simple, intermediate, advanced")
    content_type: str = Field(..., description="article, tutorial, reference, biography")
    quality_score: float = Field(..., ge=0, le=10, description="Quality rating 0-10")
    last_updated: str = Field(..., description="ISO date of last update")
    editor_count: int = Field(..., ge=0, description="Number of editors")
    reference_count: int = Field(..., ge=0, description="Number of references")
    word_count: int = Field(..., ge=0, description="Total word count")
    image_count: int = Field(..., ge=0, description="Number of images")
    external_links: int = Field(..., ge=0, description="Number of external links")
    popularity_rank: int = Field(..., ge=0, description="Popularity ranking")
    is_featured: bool = Field(False, description="Is featured article")
    language: str = Field("en", description="Language code")
    tags: List[str] = Field(default_factory=list, description="Topic tags")

    model_config = ConfigDict(from_attributes=True)


class ChunkMetadata(BaseModel):
    """Metadata schema for document chunks"""

    section_type: str = Field(..., description="intro, body, conclusion, references")
    position: str = Field(..., description="beginning, middle, end")
    word_count: int = Field(..., ge=0, description="Word count in chunk")
    char_count: int = Field(..., ge=0, description="Character count")
    sentence_count: int = Field(..., ge=0, description="Number of sentences")
    has_code: bool = Field(False, description="Contains code snippets")
    has_math: bool = Field(False, description="Contains mathematical formulas")
    has_list: bool = Field(False, description="Contains bullet/numbered lists")
    has_table: bool = Field(False, description="Contains tables")
    language_detected: str = Field("en", description="Detected language")
    sentiment: str = Field("neutral", description="positive, negative, neutral")
    complexity_score: float = Field(..., ge=0, le=10, description="Text complexity 0-10")
    readability_score: float = Field(..., ge=0, le=100, description="Flesch reading ease")
    technical_density: float = Field(..., ge=0, le=1, description="Technical term density")
    named_entities: List[str] = Field(default_factory=list, description="Named entities found")
    key_phrases: List[str] = Field(default_factory=list, description="Key phrases extracted")

    model_config = ConfigDict(from_attributes=True)


# Search schemas
@dataclass
class SearchResult:
    """Represents a search result with scoring information"""

    chunk_id: int
    document_id: int
    title: str
    text: str
    score: float
    vector_rank: Optional[int] = None
    text_rank: Optional[int] = None
    k_score: Optional[float] = None  # Keyword search score
    v_score: Optional[float] = None  # Vector search score

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "text": self.text,
            "score": self.score,
            "vector_rank": self.vector_rank,
            "text_rank": self.text_rank,
            "k_score": self.k_score,
            "v_score": self.v_score,
        }


class SearchType(str, Enum):
    """Types of search methods available"""

    SEMANTIC = "semantic"  # Vector similarity search
    KEYWORD = "keyword"  # Full-text keyword search
    HYBRID = "hybrid"  # Combined semantic + keyword search


class SearchRequest(BaseModel):
    """Request model for search operations"""

    query_text: str = Field(..., description="Text query for search")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding for similarity search")
    search_type: SearchType = Field(SearchType.HYBRID, description="Type of search to perform")
    limit: int = Field(10, ge=1, le=100, description="Number of results to return")
    search_depth: int = Field(40, ge=10, le=200, description="Search depth for each method")
    rrf_k: int = Field(50, ge=1, le=200, description="RRF constant for hybrid scoring")
    user_id: Optional[int] = Field(None, description="Filter results by user ID")
    full_text_weight: float = Field(
        1.0, ge=0, description="Weight for full-text search results (1.0 = equal weight)"
    )
    semantic_weight: float = Field(
        1.0, ge=0, description="Weight for semantic search results (1.0 = equal weight)"
    )

    model_config = ConfigDict(from_attributes=True)


class SearchResponse(BaseModel):
    """Response model for search operations"""

    results: List[dict] = Field(..., description="List of search results")
    total_results: int = Field(..., description="Total number of results")
    search_type: SearchType = Field(..., description="Type of search performed")
    query_text: str = Field(..., description="Original query text")

    model_config = ConfigDict(from_attributes=True)


# Database configuration schemas
class DatabaseConfig(BaseModel):
    """Database configuration schema"""

    host: str = Field("localhost", description="Database host")
    port: int = Field(5432, description="Database port")
    database: str = Field("leo_pgvector", description="Database name")
    user: str = Field("postgres", description="Database user")
    password: str = Field("postgres", description="Database password")

    @property
    def url(self) -> str:
        """Generate PostgreSQL connection URL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    model_config = ConfigDict(from_attributes=True)


# Data ingestion schemas
class IngestionConfig(BaseModel):
    """Configuration for data ingestion"""

    num_users: int = Field(1000, ge=1, description="Number of users to create")
    max_documents_per_user: int = Field(10, ge=1, description="Max documents per user")
    dataset_split: str = Field("train[:5000]", description="Dataset split to load")
    batch_size: int = Field(100, ge=1, description="Batch size for ingestion")

    model_config = ConfigDict(from_attributes=True)


# Index configuration schemas
class IndexConfig(BaseModel):
    """Configuration for database indexes"""

    hnsw_m: int = Field(16, ge=2, description="HNSW M parameter")
    hnsw_ef_construction: int = Field(256, ge=16, description="HNSW ef_construction parameter")
    text_dictionary: str = Field("english", description="Dictionary for full-text search")

    model_config = ConfigDict(from_attributes=True)
