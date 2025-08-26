"""
Query logic for hybrid search with PostgreSQL and pgvector
"""

import os
import time
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from rich import print
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv
import numpy as np

from models import User, UserDocument, UserDocumentChunk
from schemas import SearchResult, SearchType, SearchRequest, SearchResponse, DatabaseConfig

load_dotenv()


class SearchEngine:
    """
    Hybrid search engine combining vector similarity and full-text search.
    Implements best practices including RRF scoring.
    """

    def __init__(self, db_url: Optional[str] = None):
        if db_url is None:
            db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/leo_pgvector")

        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def hybrid_search_function(
        self,
        embedding: List[float],
        query_text: str,
        limit: int = 10,
        user_id: Optional[int] = None,
        full_text_weight: float = 1.0,
        semantic_weight: float = 1.0,
        rrf_k: int = 50,
    ) -> tuple[List[Dict[str, Any]], float]:
        """
        Execute hybrid search using stored database function.
        More efficient than building SQL in Python.

        Args:
            embedding: Query embedding vector (768-dimensional)
            query_text: Search query text
            limit: Maximum number of results
            user_id: Optional user ID filter
            full_text_weight: Weight for full-text search results
            semantic_weight: Weight for semantic search results
            rrf_k: RRF smoothing constant

        Returns:
            Tuple of (search results, latency in ms)
        """
        start_time = time.time()
        
        with self.SessionLocal() as session:
            # Convert embedding to PostgreSQL array format
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            result = session.execute(
                text("""
                    SELECT * FROM hybrid_search(
                        :query_text,
                        :embedding ::vector(768),
                        :limit,
                        :user_id,
                        :full_text_weight,
                        :semantic_weight,
                        :rrf_k
                    )
                """),
                {
                    "query_text": query_text,
                    "embedding": embedding_str,
                    "limit": limit,
                    "user_id": user_id,
                    "full_text_weight": full_text_weight,
                    "semantic_weight": semantic_weight,
                    "rrf_k": rrf_k,
                },
            )

            results = [
                {
                    "chunk_id": row[0],
                    "document_id": row[1],
                    "user_id": row[2],
                    "title": row[3],
                    "text": row[4],
                    "meta": row[5],
                    "score": row[6],
                }
                for row in result
            ]
            
        latency_ms = (time.time() - start_time) * 1000
        return results, latency_ms

    def vector_search(
        self, embedding: List[float], limit: int = 40, user_id: Optional[int] = None
    ) -> tuple[List[Dict[str, Any]], float]:
        """
        Perform semantic search using vector similarity (inner product).

        Args:
            embedding: Query embedding vector (768-dimensional for Cohere)
            limit: Maximum number of results
            user_id: Optional user ID filter

        Returns:
            Tuple of (search results, latency in ms)
        """
        start_time = time.time()
        
        with self.engine.connect() as conn:
            # Convert embedding to PostgreSQL vector format
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            # Simple query matching reference implementation style
            base_query = """
                SELECT 
                    c.id as chunk_id,
                    c.user_document_id as document_id,
                    c.text,
                    c.paragraph_id,
                    d.title,
                    d.url,
                    d.wiki_id,
                    row_number() OVER (ORDER BY c.embedding <#> :embedding ::vector) AS rank,
                    c.embedding <#> :embedding ::vector AS distance
                FROM user_document_chunks c
                JOIN user_documents d ON c.user_document_id = d.id
            """

            if user_id:
                query = text(
                    base_query
                    + """
                    WHERE d.user_id = :user_id
                    ORDER BY rank
                    LIMIT :limit
                """
                )
            else:
                query = text(
                    base_query
                    + """
                    ORDER BY rank
                    LIMIT :limit
                """
                )

            params = {"embedding": embedding_str, "limit": limit}
            if user_id:
                params["user_id"] = user_id

            result = conn.execute(query, params)
            results = [dict(row._mapping) for row in result]
        
        latency_ms = (time.time() - start_time) * 1000
        return results, latency_ms

    def text_search(
        self, query_text: str, limit: int = 40, user_id: Optional[int] = None
    ) -> tuple[List[Dict[str, Any]], float]:
        """
        Perform keyword search using PostgreSQL's full-text search (tsearch2).
        Uses websearch_to_tsquery for better user query handling.

        Args:
            query_text: Keyword query string
            limit: Maximum number of results
            user_id: Optional user ID filter

        Returns:
            Tuple of (search results, latency in ms)
        """
        start_time = time.time()
        
        with self.engine.connect() as conn:
            # Simple query matching reference implementation style
            base_query = """
                SELECT
                    c.id as chunk_id,
                    c.user_document_id as document_id,
                    c.text,
                    c.paragraph_id,
                    d.title,
                    d.url,
                    d.wiki_id,
                    row_number() OVER (
                        ORDER BY ts_rank_cd(
                            c.text_search_vector, 
                            websearch_to_tsquery('english', :query)
                        ) DESC
                    ) AS rank,
                    ts_rank_cd(
                        c.text_search_vector, 
                        websearch_to_tsquery('english', :query)
                    ) AS relevance
                FROM user_document_chunks c
                JOIN user_documents d ON c.user_document_id = d.id
                WHERE websearch_to_tsquery('english', :query) @@ c.text_search_vector
            """

            if user_id:
                query = text(
                    base_query
                    + """
                    AND d.user_id = :user_id
                    ORDER BY rank
                    LIMIT :limit
                """
                )
            else:
                query = text(
                    base_query
                    + """
                    ORDER BY rank
                    LIMIT :limit
                """
                )

            params = {"query": query_text, "limit": limit}
            if user_id:
                params["user_id"] = user_id

            result = conn.execute(query, params)
            results = [dict(row._mapping) for row in result]
        
        latency_ms = (time.time() - start_time) * 1000
        return results, latency_ms

    def hybrid_search_sql(
        self,
        embedding: List[float],
        query_text: str,
        limit: int = 10,
        search_depth: int = 40,
        rrf_k: int = 50,
        user_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        SQL-based hybrid search matching the article's exact approach.
        Uses UNION ALL to combine vector and text search with RRF scoring.
        """
        with self.engine.connect() as conn:
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            user_filter = ""
            if user_id:
                user_filter = f"AND d.user_id = {user_id}"

            # Using inline RRF calculation like Supabase article
            query = text(
                """
                WITH semantic AS (
                    SELECT
                        c.id,
                        row_number() OVER (ORDER BY c.embedding <#> :embedding ::vector) AS rank_ix
                    FROM user_document_chunks c
                    JOIN user_documents d ON c.user_document_id = d.id
                    WHERE 1=1 """
                + user_filter
                + """
                    ORDER BY rank_ix
                    LIMIT :search_depth
                ),
                full_text AS (
                    SELECT
                        c.id,
                        row_number() OVER (
                            ORDER BY ts_rank_cd(
                                c.text_search_vector,
                                websearch_to_tsquery('english', :query)
                            ) DESC
                        ) AS rank_ix
                    FROM user_document_chunks c
                    JOIN user_documents d ON c.user_document_id = d.id
                    WHERE 
                        websearch_to_tsquery('english', :query) @@ c.text_search_vector
                        """
                + user_filter
                + """
                    ORDER BY rank_ix
                    LIMIT :search_depth
                )
                SELECT
                    c.id as chunk_id,
                    d.id as document_id,
                    d.title,
                    c.text,
                    COALESCE(1.0 / (:rrf_k + semantic.rank_ix), 0.0) +
                    COALESCE(1.0 / (:rrf_k + full_text.rank_ix), 0.0) AS score
                FROM
                    semantic
                    FULL OUTER JOIN full_text ON semantic.id = full_text.id
                    JOIN user_document_chunks c ON COALESCE(semantic.id, full_text.id) = c.id
                    JOIN user_documents d ON c.user_document_id = d.id
                ORDER BY score DESC
                LIMIT :limit
            """
            )

            result = conn.execute(
                query,
                {
                    "embedding": embedding_str,
                    "query": query_text,
                    "search_depth": search_depth,
                    "rrf_k": rrf_k,
                    "limit": limit,
                },
            )

            return [dict(row._mapping) for row in result]

    def hybrid_search(self, request: SearchRequest) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword search with RRF scoring.

        Args:
            request: SearchRequest object with query parameters

        Returns:
            List of SearchResult objects sorted by combined RRF score
        """
        # Validate request
        if request.search_type == SearchType.SEMANTIC and not request.embedding:
            raise ValueError("Semantic search requires an embedding")

        if request.search_type == SearchType.KEYWORD and not request.query_text:
            raise ValueError("Keyword search requires query_text")

        if request.search_type == SearchType.HYBRID:
            if not request.embedding or not request.query_text:
                raise ValueError("Hybrid search requires both embedding and query_text")

        # Perform searches based on type
        if request.search_type == SearchType.SEMANTIC:
            vector_results, _ = self.vector_search(request.embedding, request.limit, request.user_id)
            return self._process_single_search_results(vector_results, "semantic")

        elif request.search_type == SearchType.KEYWORD:
            text_results, _ = self.text_search(request.query_text, request.limit, request.user_id)
            return self._process_single_search_results(text_results, "keyword")

        else:  # HYBRID
            # Perform both searches
            vector_results, _ = self.vector_search(request.embedding, request.search_depth, request.user_id)
            text_results, _ = self.text_search(request.query_text, request.search_depth, request.user_id)

            # Combine with RRF
            return self._combine_results_rrf(vector_results, text_results, request.rrf_k, request.limit)

    def _process_single_search_results(self, results: List[Dict], search_type: str) -> List[SearchResult]:
        """Process results from a single search method"""
        search_results = []

        for result in results:
            sr = SearchResult(
                chunk_id=result["chunk_id"],
                document_id=result["document_id"],
                title=result["title"],
                text=self._truncate_text(result["text"], 200),
                score=1.0 / (result["rank"] + 1),  # Simple scoring for single search
                vector_rank=result["rank"] if search_type == "semantic" else None,
                text_rank=result["rank"] if search_type == "keyword" else None,
            )
            search_results.append(sr)

        return search_results

    def _combine_results_rrf(
        self, vector_results: List[Dict], text_results: List[Dict], rrf_k: int, limit: int
    ) -> List[SearchResult]:
        """Combine search results using Reciprocal Rank Fusion"""
        combined_results = {}

        # Create lookup dictionaries
        vector_ranks = {r["chunk_id"]: r["rank"] for r in vector_results}
        text_ranks = {r["chunk_id"]: r["rank"] for r in text_results}

        # Process vector search results
        for result in vector_results:
            chunk_id = result["chunk_id"]
            vector_rank = result["rank"]
            text_rank = text_ranks.get(chunk_id)

            # Calculate RRF score
            score = 1.0 / (vector_rank + rrf_k)
            if text_rank:
                score += 1.0 / (text_rank + rrf_k)

            combined_results[chunk_id] = SearchResult(
                chunk_id=chunk_id,
                document_id=result["document_id"],
                title=result["title"],
                text=self._truncate_text(result["text"], 200),
                score=score,
                vector_rank=vector_rank,
                text_rank=text_rank,
            )

        # Process text search results not in vector results
        for result in text_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in combined_results:
                text_rank = result["rank"]
                score = 1.0 / (text_rank + rrf_k)

                combined_results[chunk_id] = SearchResult(
                    chunk_id=chunk_id,
                    document_id=result["document_id"],
                    title=result["title"],
                    text=self._truncate_text(result["text"], 200),
                    score=score,
                    vector_rank=None,
                    text_rank=text_rank,
                )

        # Sort by score and return top results
        sorted_results = sorted(combined_results.values(), key=lambda x: x.score, reverse=True)[:limit]

        return sorted_results

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length with ellipsis"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    def search(self, request: SearchRequest) -> SearchResponse:
        """
        Main search method that returns a SearchResponse.

        Args:
            request: SearchRequest with all parameters

        Returns:
            SearchResponse with results and metadata
        """
        results = self.hybrid_search(request)

        return SearchResponse(
            results=[r.to_dict() for r in results],
            total_results=len(results),
            search_type=request.search_type,
            query_text=request.query_text,
        )

    def display_results(self, results: List[SearchResult], title: str = "Search Results"):
        """Display search results in a formatted table"""
        console = Console()
        table = Table(title=title, show_header=True, header_style="bold magenta")

        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Title", style="yellow", width=30)
        table.add_column("Text", width=50)
        table.add_column("Score", style="green", width=12)
        table.add_column("V-Rank", style="blue", width=8)
        table.add_column("T-Rank", style="red", width=8)

        for idx, result in enumerate(results, 1):
            table.add_row(
                str(idx),
                self._truncate_text(result.title, 30),
                self._truncate_text(result.text, 50),
                f"{result.score:.6f}",
                str(result.vector_rank) if result.vector_rank else "-",
                str(result.text_rank) if result.text_rank else "-",
            )

        console.print(table)


def fetch_parent_documents(engine, search_results: list) -> tuple[dict, float]:
    """
    Fetch parent documents for search results using batch fetch with deduplication

    Args:
        engine: SQLAlchemy engine
        search_results: List of search results with document_id field

    Returns:
        Tuple of (documents map, latency in ms)
    """
    start_time = time.time()
    
    # Step 1: Extract unique document IDs (preserving order by best score)
    unique_doc_ids = []
    seen = set()
    for chunk in search_results:
        doc_id = chunk["document_id"]
        if doc_id not in seen:
            unique_doc_ids.append(doc_id)
            seen.add(doc_id)

    if not unique_doc_ids:
        return {}, 0.0

    # Step 2: Single batch query for all unique documents
    batch_query = text("""
        SELECT 
            d.id as document_id,
            d.user_id,
            d.title,
            d.url,
            d.wiki_id,
            d.views,
            d.langs,
            d.meta as document_meta,
            d.created_at,
            STRING_AGG(c.text, E'\n\n' ORDER BY c.paragraph_id) as full_text,
            COUNT(c.id)::int as chunk_count,
            jsonb_agg(
                jsonb_build_object(
                    'chunk_id', c.id,
                    'paragraph_id', c.paragraph_id,
                    'text', c.text,
                    'meta', c.meta
                ) ORDER BY c.paragraph_id
            ) as all_chunks
        FROM user_documents d
        JOIN user_document_chunks c ON d.id = c.user_document_id
        WHERE d.id = ANY(:doc_ids)
        GROUP BY d.id, d.user_id, d.title, d.url, d.wiki_id, 
                 d.views, d.langs, d.meta, d.created_at
    """)

    with engine.connect() as conn:
        result = conn.execute(batch_query, {"doc_ids": unique_doc_ids})

        # Step 3: Create efficient mapping structure
        documents_map = {}
        for row in result:
            doc_data = dict(row._mapping)
            documents_map[doc_data["document_id"]] = doc_data

    latency_ms = (time.time() - start_time) * 1000
    return documents_map, latency_ms


def create_enriched_results(search_results: list, documents_map: dict) -> list:
    """
    Enhance chunks with their parent documents

    Args:
        search_results: Original search results with chunks
        documents_map: Dictionary mapping document_id to full document

    Returns:
        List of enriched results with both chunk and document data
    """
    enriched_results = []
    for chunk in search_results:
        enriched_results.append(
            {"chunk": chunk, "document": documents_map.get(chunk["document_id"]), "score": chunk["score"]}
        )
    return enriched_results


def main():
    """Main query interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Query leo-pgvector database")
    parser.add_argument("query", nargs="?", help="Query text")
    parser.add_argument("--type", choices=["semantic", "keyword", "hybrid"], default="hybrid")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--rrf-k", type=int, default=50)
    parser.add_argument("--user-id", type=int, help="Filter by user ID")
    parser.add_argument("--example", action="store_true", help="Run with example query and embedding")
    parser.add_argument("--fetch-docs", action="store_true", help="Fetch full parent documents for results")

    args = parser.parse_args()

    if args.query or args.example:
        engine = SearchEngine()

        # Determine query text
        if args.example:
            query_text = "science technology innovation"
            print(f"[cyan]Running example search with query: '{query_text}'[/cyan]\n")
        else:
            query_text = args.query

        # Determine search type
        search_type_map = {
            "semantic": SearchType.SEMANTIC,
            "keyword": SearchType.KEYWORD,
            "hybrid": SearchType.HYBRID,
        }
        search_type = search_type_map[args.type]

        # Generate embedding if needed for semantic or hybrid search
        embedding = None
        if search_type in [SearchType.SEMANTIC, SearchType.HYBRID]:
            # Generate random 768-dimensional embedding for testing
            # In production, use an actual embedding model
            embedding = np.random.randn(768).tolist()
            print("[green]Using random embedding for testing[/green]\n")

        # Create request
        request = SearchRequest(
            query_text=query_text,
            embedding=embedding,
            search_type=search_type,
            limit=args.limit,
            search_depth=args.depth,
            rrf_k=args.rrf_k,
            user_id=args.user_id,
        )

        # Execute search based on type
        if search_type == SearchType.HYBRID and embedding:
            # Use the efficient hybrid search function
            results, search_latency = engine.hybrid_search_function(
                embedding=embedding,
                query_text=query_text,
                limit=args.limit,
                user_id=args.user_id,
                rrf_k=args.rrf_k,
            )
            
            print(f"[yellow]Search latency: {search_latency:.2f}ms[/yellow]\n")

            # Fetch parent documents if requested
            if args.fetch_docs:
                print("[cyan]Fetching parent documents...[/cyan]")
                documents_map, fetch_latency = fetch_parent_documents(engine.engine, results)
                enriched_results = create_enriched_results(results, documents_map)

                # Display document fetch summary
                unique_docs = len(documents_map)
                total_chunks = len(results)
                print(f"[green]Fetched {unique_docs} unique documents for {total_chunks} chunks[/green]")
                print(f"[yellow]Document fetch latency: {fetch_latency:.2f}ms[/yellow]")
                print(f"[bold cyan]Total latency: {(search_latency + fetch_latency):.2f}ms[/bold cyan]\n")

                # Display enriched results
                for idx, enriched in enumerate(enriched_results, 1):
                    chunk = enriched["chunk"]
                    doc = enriched["document"]
                    if doc:
                        print(f"[bold]#{idx} Document: {doc['title']}[/bold]")
                        print(f"  Chunk ID: {chunk['chunk_id']}, Score: {chunk['score']:.4f}")
                        print(f"  Document has {doc['chunk_count']} chunks, {len(doc['full_text'])} chars")
                        print(f"  Chunk text: {chunk['text'][:150]}...")
                        print()
            else:
                # Convert to SearchResult format
                search_results = []
                for r in results:
                    search_results.append(
                        SearchResult(
                            chunk_id=r["chunk_id"],
                            document_id=r["document_id"],
                            title=r["title"],
                            text=r["text"],
                            score=r["score"],
                        )
                    )

                engine.display_results(search_results)
        else:
            # Use standard search method
            start_time = time.time()
            response = engine.search(request)
            search_latency = (time.time() - start_time) * 1000
            results = [SearchResult(**r) for r in response.results]
            
            print(f"[yellow]Search latency: {search_latency:.2f}ms[/yellow]\n")

            # Fetch parent documents if requested
            if args.fetch_docs:
                print("[cyan]Fetching parent documents...[/cyan]")
                # Convert SearchResult to dict format
                results_dict = [
                    {
                        "document_id": r.document_id,
                        "chunk_id": r.chunk_id,
                        "text": r.text,
                        "title": r.title,
                        "score": r.score,
                    }
                    for r in results
                ]
                documents_map, fetch_latency = fetch_parent_documents(engine.engine, results_dict)
                enriched_results = create_enriched_results(results_dict, documents_map)

                # Display document fetch summary
                unique_docs = len(documents_map)
                total_chunks = len(results)
                print(f"[green]Fetched {unique_docs} unique documents for {total_chunks} chunks[/green]")
                print(f"[yellow]Document fetch latency: {fetch_latency:.2f}ms[/yellow]")
                print(f"[bold cyan]Total latency: {(search_latency + fetch_latency):.2f}ms[/bold cyan]\n")

                # Display enriched results
                for idx, enriched in enumerate(enriched_results, 1):
                    chunk = enriched["chunk"]
                    doc = enriched["document"]
                    if doc:
                        print(f"[bold]#{idx} Document: {doc['title']}[/bold]")
                        print(f"  Chunk ID: {chunk['chunk_id']}, Score: {chunk['score']:.4f}")
                        print(f"  Document has {doc['chunk_count']} chunks, {len(doc['full_text'])} chars")
                        print(f"  Chunk text: {chunk['text'][:150]}...")
                        print()
            else:
                engine.display_results(results)

        # Show search parameters
        print(
            f"\n[dim]Search type: {search_type.value}, Limit: {args.limit}, Depth: {args.depth}, RRF-k: {args.rrf_k}[/dim]"
        )
        if args.user_id:
            print(f"[dim]Filtered by user: {args.user_id}[/dim]")
        if args.fetch_docs:
            print("[dim]Parent documents fetched[/dim]")
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python query.py 'machine learning' --type hybrid")
        print("  python query.py 'python programming' --type keyword --limit 5")
        print("  python query.py --example --type hybrid")
        print("  python query.py 'science' --type hybrid --fetch-docs  # Fetch full parent documents")


if __name__ == "__main__":
    main()
