A production-ready implementation of hybrid search combining vector similarity search with full-text search using PostgreSQL, pgvector, and Reciprocal Rank Fusion (RRF).

- each user has a set of documents
- each document has a set of chunks whose embeddings are computed and stored in the document chunks table
- we want to do a very fast retrieval of the documents be it via semantic search, keyword search or hybrid search
- we need to always filter based for a specific user id (i.e. we narrow down the search space to documents uploaded by user abc and then perform the search methods - but we want this operation to very efficient)

## Features

- **Hybrid Search**: Combines semantic vector search with full-text search
- **RRF Scoring**: Reciprocal Rank Fusion for optimal result ranking
- **pgvector Integration**: Efficient vector similarity search with HNSW indexes
- **Full-Text Search**: PostgreSQL's built-in tsearch2 with GIN indexes
- **User Scoping**: Multi-tenant support with user-based filtering
- **Wikipedia Dataset**: Pre-configured for Cohere's Wikipedia embeddings
- **Document Fetching**: Optimized batch fetching of parent documents with deduplication

## Prerequisites

- Python 3.12+
- Docker and Docker Compose
- 4GB+ RAM for dataset processing
- Google API Key (for Gemini query generation)
- Cohere API Key (for embedding generation)

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd leo-pgvector
```

2. **Install dependencies**:
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

3. **Start PostgreSQL with pgvector**:
```bash
# Start the Docker container (PostgreSQL 17 with pgvector)
docker-compose up -d

# Verify the container is running
docker ps

# The database will be available at localhost:54320
```

4. **Configure environment**:
```bash
# Create .env file with your configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:54320/leo_pgvector
GOOGLE_API_KEY=your_google_api_key  # For Gemini query generation
COHERE_API_KEY=your_cohere_api_key  # For embeddings
```

## Project Structure

```
leo-pgvector/
...
```


## Additional Project Files

```
   bench.py                        # Performance benchmarking script
   pre-bench.py                    # Generate benchmark data with LLMs
   analyze_queries.py              # SQL EXPLAIN ANALYZE tool
   benchmark_ef_search.py          # Test different ef_search values
   benchmark_ef_search_thorough.py # Comprehensive ef_search analysis
   test_insert_performance.py      # Insert/update performance testing
   llm.py                          # Google Gemini client for query generation
   embedding.py                    # Cohere client for embeddings
   data/                           # Benchmark data storage
   sql-out/                        # Benchmark results
   sql-explain/                    # Query analysis results
```

## File Descriptions

### `models.py`
SQLAlchemy ORM models defining the database schema:
- `User`: User accounts
- `UserDocument`: Wikipedia articles metadata
- `UserDocumentChunk`: Text chunks with 768-dimensional embeddings

### `schemas.py`
Data validation and serialization schemas:
- `DocumentItem`: Dataset item validation
- `SearchRequest/Response`: API models for search
- `DatabaseConfig`: Database configuration
- `IngestionConfig`: Data ingestion settings
- `IndexConfig`: Index parameters

### `setup.py`
Database initialization and data ingestion:
- Creates pgvector extension and RRF function
- Sets up denormalized schema with user_id in chunks table
- Creates optimized indexes (HNSW, GIN, B-tree)
- Downloads Cohere Wikipedia dataset
- Populates database with users and documents

### `bench.py`
Comprehensive benchmarking tool:
- Tests vector, text, and hybrid search
- Evaluates with different filtering scenarios
- Measures latency and recall/precision
- Saves results to sql-out/ directory

### `analyze_queries.py`
SQL query performance analysis:
- Runs EXPLAIN ANALYZE BUFFERS on all query patterns
- Generates detailed execution plans
- Saves analysis to sql-explain/ directory
- Creates individual files for text, semantic, and hybrid queries

### `query.py`
Search implementation with three modes:
- **Semantic Search**: Vector similarity using negative inner product
- **Keyword Search**: Full-text search with ts_rank_cd
- **Hybrid Search**: RRF combination of both methods
- **User Filtering**: All searches require user_id for multi-tenant isolation

## Usage

### 1. Initialize Database

```bash
# Quick test with 1000 documents
python setup.py --drop --users 100 --max-docs 10 --dataset-split "train[:1000]"

# Full dataset (486K documents) - Takes ~30-60 minutes
python setup.py --drop --users 1000 --max-docs 500 --dataset-split "train"

# Options:
#   --drop           Drop existing tables before setup
#   --skip-data      Skip data ingestion (schema only)
#   --users N        Number of users to create (default: 1000)
#   --max-docs N     Max documents per user (default: 10)
#   --dataset-split  Dataset portion to load:
#                    "train[:1000]" for testing (1K docs)
#                    "train[:5000]" for development (5K docs)
#                    "train" for full dataset (486K docs)
```

### 2. Run Searches

```bash
# Run example search with sample data
python query.py --example

# Keyword search
python query.py "machine learning" --type keyword --limit 10

# Hybrid search (requires embedding)
python query.py "artificial intelligence" --type hybrid --limit 10

# User-scoped search
python query.py "science" --user-id 1 --type keyword

# Fetch full parent documents for search results
python query.py "machine learning" --type hybrid --fetch-docs

# Options:
#   --type {semantic,keyword,hybrid}  Search type (default: hybrid)
#   --limit N                    Results limit (default: 10)
#   --depth N                    Search depth per method (default: 40)
#   --rrf-k N                   RRF constant (default: 50)
#   --user-id N                 Filter by user ID
#   --fetch-docs                 Fetch full parent documents for results
```

### 3. Python API Usage

```python
from query import SearchEngine
from schemas import SearchRequest, SearchType

# Initialize engine
engine = SearchEngine()

# Create search request
request = SearchRequest(
    query_text="machine learning algorithms",
    embedding=your_embedding_vector,  # 768-dimensional
    search_type=SearchType.HYBRID,
    limit=10,
    search_depth=40,
    rrf_k=50,
    user_id=None  # Optional user filter
)

# Execute search
response = engine.search(request)

# Access results
for result in response.results:
    print(f"{result['title']}: {result['score']}")

# Fetch parent documents (2-query pattern)
from query import fetch_parent_documents, create_enriched_results

# After getting search results
results = engine.hybrid_search_function(
    embedding=embedding,
    query_text="machine learning",
    limit=10
)

# Fetch unique parent documents in single batch query
documents_map = fetch_parent_documents(engine.engine, results)

# Enrich chunks with their parent documents
enriched_results = create_enriched_results(results, documents_map)

# Access both chunk and full document
for item in enriched_results:
    chunk = item['chunk']
    document = item['document']  # Full parent document with all chunks
    print(f"Chunk from document: {document['title']}")
    print(f"Document has {document['chunk_count']} total chunks")
```

### 4. Performance Benchmarking

#### Generate Benchmark Data with LLMs
```bash
# Generate challenging queries using Google Gemini and Cohere embeddings
python pre-bench.py --users 25 --queries 2
# Creates: data/benchmark_TIMESTAMP.csv and .json with real embeddings
```

#### Run Performance Benchmark
```bash
# Run benchmark with generated data
uv run python bench.py data/benchmark_TIMESTAMP.csv --limit 10

# The benchmark tests:
#   - No Filter: Baseline performance without any filtering
#   - User Only: Performance with user_id filtering  
#   - User + JSONB: Additional filtering on JSONB metadata
#   - User + JSONB Complex: Complex JSONB filtering
#   - Hybrid Two-Query: Compares DB function vs separate vector+text queries
#
# Results saved to sql-out/benchmark_results_TIMESTAMP.csv
```

#### ef_search Optimization Testing
```bash
# Test different ef_search values for vector search performance
uv run python benchmark_ef_search.py data/benchmark_TIMESTAMP.csv --limit 10

# Comprehensive ef_search analysis with multiple iterations
uv run python benchmark_ef_search_thorough.py data/benchmark_TIMESTAMP.csv --iterations 3

# Results show ef_search=64 is optimal (not 100 as commonly suggested)
```

#### Analyze Query Performance
```bash
# Generate EXPLAIN ANALYZE reports for all query patterns
python analyze_queries.py

# Creates in sql-explain/:
#   - query_analysis_TIMESTAMP.txt (comprehensive report)
#   - text-search-explain.txt
#   - semantic-search-explain.txt
#   - hybrid-search-explain.txt
```

#### Recall/Precision Evaluation

Generate benchmark queries and evaluate search quality:

```bash
# Method 1: Generate simple benchmark queries
python generate_bench_data.py
# Creates: data/benchmark_queries_TIMESTAMP.csv and .json

# Method 2: Generate challenging queries with Google Gemini AI
export GOOGLE_API_KEY="your-api-key"  # Set your Google AI API key
python generate_bench_gemini.py --chunks 20 --queries 3
# Creates: data/gemini_benchmark_TIMESTAMP.csv and .json

# Run recall/precision evaluation on generated queries
python run_evaluation.py data/gemini_benchmark_TIMESTAMP.csv --limit 10

# Evaluation metrics include:
#   - Recall@1, @5, @10: Percentage of queries where ground truth appears in top K
#   - MRR (Mean Reciprocal Rank): Average of 1/rank for each query
#   - Average position when found
#   - Hits found: How many queries found their ground truth
#   - Average latency per search type
```

#### Using Gemini for Query Generation

The `llm.py` module provides intelligent query generation:

```python
from llm import GeminiQueryGenerator

# Initialize with API key
generator = GeminiQueryGenerator(api_key="your-key")

# Generate challenging queries from text
chunk_text = "Your document chunk text here..."
queries = generator.generate_queries(chunk_text, num_queries=5)

# Queries are ranked by difficulty (1 = most challenging)
for i, query in enumerate(queries, 1):
    print(f"Difficulty {i}: {query}")
```

### 5. Direct Database Setup

```python
from setup import DatabaseSetup, DataIngestion
from schemas import IngestionConfig

# Initialize database
db_setup = DatabaseSetup()
engine = db_setup.initialize(drop_existing=True)

# Ingest data
config = IngestionConfig(
    num_users=1000,
    max_documents_per_user=10,
    dataset_split="train[:5000]"
)
ingestion = DataIngestion(engine, config)
stats = ingestion.run()
```

## Database Schema

### Tables
- `users`: User accounts with names
- `user_documents`: Document metadata (title, URL, wiki_id)
- `user_document_chunks`: Text chunks with vector embeddings

### Indexes
- **HNSW Index**: Fast vector similarity search
- **GIN Index**: Full-text search on text content
- **B-tree Indexes**: Foreign key relationships

### Functions
- `hybrid_search()`: Stored function for efficient hybrid search with RRF scoring
- `fetch_parent_documents()`: Batch fetch documents with deduplication

## Search Methods

### Semantic Search
- Uses pgvector's inner product operator (`<#>`) for better performance
- HNSW index for efficient vector similarity search
- 768-dimensional embeddings (Cohere)
- Finds conceptually similar content

### Keyword Search
- PostgreSQL's tsearch2 with English dictionary
- GIN index on stored tsvector column
- Ranking with `ts_rank_cd`
- Finds exact keyword matches

### Hybrid Search
- Combines semantic and keyword search using RRF
- Formula: `score = 1/(semantic_rank + k) + 1/(keyword_rank + k)`
- Configurable k parameter (default: 50)
- Best of both worlds: concepts + keywords

### Document Fetching Pattern
- **2-Query Approach**: First search chunks, then batch fetch documents
- **Deduplication**: Extract unique document IDs to avoid redundant fetches
- **Batch Query**: Single SQL query retrieves all parent documents
- **O(1) Lookup**: Efficient mapping structure for chunk-to-document association
- **Performance**: Optimized for multiple chunks from same document (common in search results)

## Performance Optimization Results

### Denormalization Impact
Eliminated JOINs by adding user_id directly to chunks table:
- **5-35x faster** queries compared to JOIN-based approach
- No backward compatibility overhead - fresh schema design

### Current Performance Metrics (470K vectors, 25 users)
```
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Search Type ┃ Filter             ┃   Mean ┃ Median ┃ StdDev ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ Vector      │ No Filter          │ 165.48 │ 142.26 │  84.34 │
│ Vector      │ User Only          │  13.06 │   6.38 │  17.84 │
│ Vector      │ User + JSONB       │   0.68 │   0.65 │   0.23 │
│ Text        │ No Filter          │   8.52 │   4.53 │  16.42 │
│ Text        │ User Only          │   1.04 │   0.87 │   0.81 │
│ Hybrid      │ User Only          │   2.32 │   2.09 │   0.84 │
└─────────────┴────────────────────┴────────┴────────┴────────┘
```

### HNSW Index Optimization
After extensive benchmarking and tuning:
- Rebuilt index with `m = 16, ef_construction = 64`
- **Optimal ef_search = 64** (not 100 as initially configured)
- **Result**: Vector search latency of **2-3ms** with warm connections (vs initial 300ms)

### Index Parameters
```python
IndexConfig(
    hnsw_m=16,                  # HNSW connections
    hnsw_ef_construction=64,    # Build quality
    hnsw_ef_search=64,          # Optimal query-time quality (was 100)
    text_dictionary="english"   # Language dictionary
)
```

### ef_search Performance Analysis (485K vectors)
Comprehensive testing with multiple iterations shows:
```
ef_search  Cold Cache  Warm Cache  Recall@10  Recommendation
--------  ----------  ----------  ---------  --------------
32         54.8ms     3.9ms       80%        Too variable
40         7.9ms      4.2ms       80%        Good alternative
50         7.9ms      4.6ms       80%        Good alternative  
64         7.7ms      5.4ms       80%        ← OPTIMAL
80         7.9ms      6.4ms       80%        Diminishing returns
100        8.2ms      7.5ms       80%        Current (suboptimal)
128        8.0ms      22.0ms      80%        Cache pressure
150        8.0ms      19.3ms      80%        Cache pressure
200        8.3ms      28.9ms      80%        Poor performance
```

**Key Findings:**
- ef_search=64 provides best balance of speed and consistency
- All values 40-200 achieve same 80% recall (index quality limit)
- Higher ef_search values (>100) suffer from cache pressure
- First query on new connection: 5-600ms (connection initialization)
- Subsequent queries: 2-3ms with connection pooling

### Production Deployment Recommendations

**Connection Management (Critical):**
- **Use connection pooling** (pgbouncer/pgpool) - Essential for consistent 2-3ms latency
- **Without pooling**: Each new connection pays 5-600ms initialization cost
- **Serverless/Lambda**: Consider persistent connections or accept variable latency

**Cold Start Behavior:**
- First query after PostgreSQL restart: ~600ms
- First query on new connection: 5-600ms  
- Subsequent queries same connection: 2-3ms
- Mitigation: Warm connections in health checks

**Recommended Configuration:**
```sql
-- Apply optimal ef_search setting
ALTER DATABASE leo_pgvector SET hnsw.ef_search = 64;
```

### Key Performance Insights
- **User filtering**: 22x faster than no filter
- **JSONB filtering**: 240x faster than no filter (sub-millisecond)
- **Hybrid search**: Consistently ~2ms with user filtering
- **Text search**: Sub-millisecond with GIN index
- **Connection pooling**: 100-200x improvement for cold starts

### Search Parameters
- `search_depth`: Results per method before RRF (default: 40)
- `rrf_k`: Weight distribution (10-30 for top bias, 50-100 for balance)
- `limit`: Final results count

## Monitoring

```python
# Analyze query performance
from query import SearchEngine

engine = SearchEngine()
metrics = engine.analyze_query_performance(request)
```

## Dataset

Uses [Cohere/wikipedia-22-12-simple-embeddings](https://huggingface.co/datasets/Cohere/wikipedia-22-12-simple-embeddings):
- Wikipedia articles in simple English
- Pre-computed 768-dimensional embeddings
- ~485K documents with multiple chunks each

## Docker Management

```bash
# Start the database
docker-compose up -d

# Stop the database
docker-compose down

# View logs
docker-compose logs -f pgvector

# Connect to the database
docker exec -it leo-pgvector-db psql -U postgres -d leo_pgvector

# Reset everything (including data)
docker-compose down -v
docker-compose up -d
```

## Troubleshooting

### PostgreSQL Connection Issues
```bash
# Check if Docker container is running
docker ps | grep leo-pgvector-db

# Check container logs
docker logs leo-pgvector-db

# Test connection
psql postgresql://postgres:postgres@localhost:54320/leo_pgvector

# If port 54320 is in use, edit docker-compose.yml to use a different port
```

### pgvector Extension
```sql
-- Extension is automatically installed in the Docker image
-- Verify installation
docker exec -it leo-pgvector-db psql -U postgres -d leo_pgvector -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### Memory Issues
- Reduce dataset size: `--dataset-split "train[:1000]"`
- Lower batch size in IngestionConfig
- Increase PostgreSQL shared_buffers

## Development

### Running Tests
```bash
# Run example searches
python query.py --example

# Test database setup
python setup.py --drop --users 10 --dataset-split "train[:100]"
```

### Code Style
```bash
# Format code
black *.py

# Type checking
mypy *.py
```

## License

MIT License - See LICENSE file for details

## References

- [Hybrid Search With PostgreSQL and Pgvector](https://example.com/blog-post)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [PostgreSQL Full-Text Search](https://www.postgresql.org/docs/current/textsearch.html)
