"""
Query logic for hybrid search with PostgreSQL and pgvector
"""

import os
import time
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, text
from rich import print
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv
import numpy as np

from schemas import SearchResult, SearchType, SearchRequest, SearchResponse

load_dotenv()


class SearchEngine:
    """
    Hybrid search engine combining vector similarity and full-text search.
    Implements best practices including RRF scoring.
    """

    def __init__(self, db_url: Optional[str] = None, use_pooling: bool = True):
        if db_url is None:
            db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/leo_pgvector")

        if use_pooling:
            # Use connection pooling for better performance
            self.engine = create_engine(
                db_url, 
                echo=False,
                pool_size=5,           # Number of connections to maintain in pool
                max_overflow=10,       # Maximum overflow connections
                pool_pre_ping=True,    # Test connections before using
                pool_recycle=3600     # Recycle connections after 1 hour
            )
        else:
            self.engine = create_engine(db_url, echo=False)

    def hybrid_search_function(
        self,
        embedding: List[float],
        query_text: str,
        limit: int = 10,
        full_text_weight: float = 1.0,
        semantic_weight: float = 1.0,
        rrf_k: int = 50,
    ) -> tuple[List[Dict[str, Any]], float]:
        """
        Execute hybrid search using the stored database function defined in models.py.
        This is the primary hybrid search implementation.

        Args:
            embedding: Query embedding vector (768-dimensional)
            query_text: Search query text
            limit: Maximum number of results
            full_text_weight: Weight for full-text search results
            semantic_weight: Weight for semantic search results
            rrf_k: RRF smoothing constant

        Returns:
            Tuple of (search results, latency in ms)
        """
        start_time = time.time()
        
        with self.engine.connect() as conn:
            # Convert embedding to PostgreSQL array format
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            result = conn.execute(
                text("""
                    SELECT * FROM hybrid_search(
                        :query_text,
                        :embedding ::vector(768),
                        :limit,
                        :full_text_weight,
                        :semantic_weight,
                        :rrf_k
                    )
                """),
                {
                    "query_text": query_text,
                    "embedding": embedding_str,
                    "limit": limit,
                    "full_text_weight": full_text_weight,
                    "semantic_weight": semantic_weight,
                    "rrf_k": rrf_k,
                },
            )

            results = [
                {
                    "chunk_id": row[0],  # id
                    "document_id": row[1],  # user_document_id
                    "paragraph_id": row[2],  # paragraph_id
                    "text": row[3],  # text
                    "meta": row[4],  # meta
                    "created_at": row[5],  # created_at
                    "k_score": row[6],  # keyword score
                    "v_score": row[7],  # vector score
                    "score": row[8],  # combined score
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

            # Simplified query - just get everything from chunks table
            if user_id:
                query = text("""
                    SELECT 
                        *,
                        embedding <#> :embedding ::vector AS distance
                    FROM user_document_chunks
                    WHERE user_document_id IN (
                        SELECT id FROM user_documents WHERE user_id = :user_id
                    )
                    ORDER BY embedding <#> :embedding ::vector
                    LIMIT :limit
                """)
            else:
                query = text("""
                    SELECT 
                        *,
                        embedding <#> :embedding ::vector AS distance
                    FROM user_document_chunks
                    ORDER BY embedding <#> :embedding ::vector
                    LIMIT :limit
                """)

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
            # Simplified query - just get everything from chunks table
            if user_id:
                query = text("""
                    SELECT
                        *,
                        ts_rank_cd(
                            text_search_vector, 
                            websearch_to_tsquery('english', :query)
                        ) AS relevance
                    FROM user_document_chunks
                    WHERE 
                        websearch_to_tsquery('english', :query) @@ text_search_vector
                        AND user_document_id IN (
                            SELECT id FROM user_documents WHERE user_id = :user_id
                        )
                    ORDER BY relevance DESC
                    LIMIT :limit
                """)
            else:
                query = text("""
                    SELECT
                        *,
                        ts_rank_cd(
                            text_search_vector, 
                            websearch_to_tsquery('english', :query)
                        ) AS relevance
                    FROM user_document_chunks
                    WHERE websearch_to_tsquery('english', :query) @@ text_search_vector
                    ORDER BY relevance DESC
                    LIMIT :limit
                """)

            params = {"query": query_text, "limit": limit}
            if user_id:
                params["user_id"] = user_id

            result = conn.execute(query, params)
            results = [dict(row._mapping) for row in result]
        
        latency_ms = (time.time() - start_time) * 1000
        return results, latency_ms

    def execute_search(self, request: SearchRequest) -> List[SearchResult]:
        """
        Execute search request by routing to appropriate search method.

        Args:
            request: SearchRequest object with query parameters

        Returns:
            List of SearchResult objects
        """
        # Route to appropriate search method based on type
        if request.search_type == SearchType.SEMANTIC:
            if not request.embedding:
                raise ValueError("Semantic search requires an embedding")
            vector_results, _ = self.vector_search(request.embedding, request.limit, request.user_id)
            return self._process_single_search_results(vector_results, "semantic")

        elif request.search_type == SearchType.KEYWORD:
            if not request.query_text:
                raise ValueError("Keyword search requires query_text")
            text_results, _ = self.text_search(request.query_text, request.limit, request.user_id)
            return self._process_single_search_results(text_results, "keyword")

        else:  # HYBRID
            if not request.embedding or not request.query_text:
                raise ValueError("Hybrid search requires both embedding and query_text")
            
            # Use the database hybrid search function directly
            results, _ = self.hybrid_search_function(
                embedding=request.embedding,
                query_text=request.query_text,
                limit=request.limit,
                user_id=request.user_id,
                rrf_k=request.rrf_k
            )
            
            # Process results into SearchResult objects
            search_results = []
            for r in results:
                search_results.append(SearchResult(
                    chunk_id=r['chunk_id'],
                    document_id=r['document_id'],
                    title="",  # Title fetched separately if needed
                    text=self._truncate_text(r['text'], 200),
                    score=r['score']
                ))
            return search_results

    def _process_single_search_results(self, results: List[Dict], search_type: str) -> List[SearchResult]:
        """Process results from a single search method"""
        search_results = []

        for idx, result in enumerate(results, 1):  # Position-based rank starting from 1
            sr = SearchResult(
                chunk_id=result["id"],  # Now using direct column name
                document_id=result["user_document_id"],
                title="",  # Title will be fetched with parent document if needed
                text=self._truncate_text(result["text"], 200),
                score=1.0 / idx,  # Simple scoring based on position
                vector_rank=idx if search_type == "semantic" else None,
                text_rank=idx if search_type == "keyword" else None,
            )
            search_results.append(sr)

        return search_results


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
        results = self.execute_search(request)

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
        table.add_column("Title", style="yellow", width=20)
        table.add_column("Text", width=40)
        table.add_column("Score", style="green", width=10)
        table.add_column("K-Score", style="red", width=10)
        table.add_column("V-Score", style="blue", width=10)

        for idx, result in enumerate(results, 1):
            table.add_row(
                str(idx),
                self._truncate_text(result.title, 20),
                self._truncate_text(result.text, 40),
                f"{result.score:.4f}",
                f"{result.k_score:.4f}" if result.k_score is not None else "-",
                f"{result.v_score:.4f}" if result.v_score is not None else "-",
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

    # Step 2: Simple batch query - just get the documents
    batch_query = text("""
        SELECT * 
        FROM user_documents 
        WHERE id = ANY(:doc_ids)
    """)

    with engine.connect() as conn:
        result = conn.execute(batch_query, {"doc_ids": unique_doc_ids})

        # Step 3: Create efficient mapping structure
        documents_map = {}
        for row in result:
            doc_data = dict(row._mapping)
            documents_map[doc_data["id"]] = doc_data

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
                        print(f"  Chunk ID: {chunk['chunk_id']}, Score: {chunk['score']:.4f} (K: {chunk.get('k_score', 0):.4f}, V: {chunk.get('v_score', 0):.4f})")
                        print(f"  Document ID: {doc['id']}")
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
                            title="",  # No title in chunks table
                            text=r["text"],
                            score=r["score"],
                            k_score=r.get("k_score"),
                            v_score=r.get("v_score"),
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
                        print(f"  Chunk ID: {chunk['chunk_id']}, Score: {chunk['score']:.4f} (K: {chunk.get('k_score', 0):.4f}, V: {chunk.get('v_score', 0):.4f})")
                        print(f"  Document ID: {doc['id']}")
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
