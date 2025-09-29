"""
Comprehensive benchmark script for testing search performance with different filtering scenarios
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from statistics import mean, median, stdev
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from datetime import datetime

# Import query module
from query import SearchEngine

load_dotenv()


@dataclass
class BenchmarkResult:
    """Store benchmark results for a single query"""

    query_id: int
    query_text: str
    latency_ms: float
    result_count: int
    filter_type: str
    found_ground_truth: bool
    ground_truth_rank: Optional[int]
    ground_truth_chunk_id: int


class SearchBenchmark:
    """Benchmark different search modes with various filtering scenarios"""

    def __init__(self, csv_file: str, json_file: str = None):
        self.console = Console()
        self.csv_file = csv_file
        self.json_file = json_file

        # Connect to database
        db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/leo_pgvector")
        self.engine = create_engine(db_url, echo=False)

        # Initialize SearchEngine
        self.search = SearchEngine(db_url)

        # Load benchmark data
        self.queries = self._load_benchmark_data()
        self.console.print(f"[green]Loaded {len(self.queries)} benchmark queries[/green]")

        # Get available user IDs for testing
        self.user_ids = list({q["ground_truth_user_id"] for q in self.queries})
        self.console.print(f"[green]Found {len(self.user_ids)} unique users in benchmark[/green]")

    def _load_benchmark_data(self) -> List[Dict]:
        """Load benchmark queries from CSV and embeddings from JSON"""
        queries = []

        # Load CSV data
        with open(self.csv_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                query_data = {
                    "query_id": int(row["query_id"]),
                    "query_text": row["query_text"],
                    "ground_truth_chunk_id": int(row["ground_truth_chunk_id"]),
                    "ground_truth_document_id": int(row["ground_truth_document_id"]),
                    "ground_truth_user_id": int(row["ground_truth_user_id"]),
                    "ground_truth_paragraph_id": int(row["ground_truth_paragraph_id"]),
                    "document_title": row["document_title"],
                    "chunk_text_preview": row["chunk_text_preview"],
                }

                # Parse metadata if present
                if "meta" in row and row["meta"]:
                    import ast

                    try:
                        query_data["meta"] = ast.literal_eval(row["meta"])
                    except:
                        query_data["meta"] = {}
                else:
                    query_data["meta"] = {}

                queries.append(query_data)

        # Load embeddings from JSON if provided
        if self.json_file and Path(self.json_file).exists():
            with open(self.json_file) as f:
                embeddings = json.load(f)
                for query in queries:
                    query_id_str = str(query["query_id"])
                    if query_id_str in embeddings:
                        # Handle both old and new JSON formats
                        if isinstance(embeddings[query_id_str], dict):
                            query["embedding"] = embeddings[query_id_str].get("query_embedding", [])
                            # Also load metadata from JSON if not in CSV
                            if "meta" in embeddings[query_id_str] and not query.get("meta"):
                                query["meta"] = embeddings[query_id_str]["meta"]
                        else:
                            query["embedding"] = embeddings[query_id_str]

        return queries

    def _format_embedding(self, embedding: List[float]) -> str:
        """Format embedding for PostgreSQL"""
        return "[" + ",".join(map(str, embedding)) + "]"

    def benchmark_vector_search(self, filter_type: str, limit: int = 10) -> List[BenchmarkResult]:
        """Benchmark vector/semantic search with different filters"""
        results = []

        with self.engine.connect() as conn:
            for query_data in self.queries:
                # Skip if no embedding available
                if "embedding" not in query_data:
                    continue

                user_id = query_data["ground_truth_user_id"]
                embedding = query_data["embedding"]
                embedding_str = self._format_embedding(embedding)
                meta = query_data.get("meta", {})

                # Use different filtering strategies
                start_time = time.time()
                if filter_type == "no_filter":
                    # Direct SQL query without any filters
                    query = text("""
                        SELECT id, user_document_id, embedding <#> :embedding ::vector AS distance
                        FROM user_document_chunks
                        ORDER BY embedding <#> :embedding ::vector
                        LIMIT :limit
                    """)
                    result = conn.execute(query, {"embedding": embedding_str, "limit": limit})
                    rows = result.fetchall()
                elif filter_type == "user_only":
                    # Filter by user only
                    rows, _ = self.search.vector_search(user_id, embedding, limit)
                elif filter_type == "user_jsonb":
                    # User + JSONB filters based on actual metadata
                    # Use common metadata attributes from the ground truth chunk
                    position = meta.get("position", "middle")
                    has_code = meta.get("has_code", False)

                    query = text("""
                        SELECT id, user_document_id, embedding <#> :embedding ::vector AS distance
                        FROM user_document_chunks
                        WHERE user_id = :user_id
                          AND meta->>'position' = :position
                          AND (meta->>'has_code')::boolean = :has_code
                        ORDER BY embedding <#> :embedding ::vector
                        LIMIT :limit
                    """)
                    result = conn.execute(
                        query,
                        {
                            "user_id": user_id,
                            "embedding": embedding_str,
                            "position": position,
                            "has_code": has_code,
                            "limit": limit,
                        },
                    )
                    rows = result.fetchall()
                elif filter_type == "user_jsonb_complex":
                    # User + Complex JSONB queries
                    complexity = meta.get("complexity_score", 5.0)

                    query = text("""
                        SELECT id, user_document_id, embedding <#> :embedding ::vector AS distance
                        FROM user_document_chunks
                        WHERE user_id = :user_id
                          AND (meta->>'complexity_score')::float >= :min_complexity
                          AND (meta->>'complexity_score')::float <= :max_complexity
                          AND meta->>'language_detected' = 'en'
                        ORDER BY embedding <#> :embedding ::vector
                        LIMIT :limit
                    """)
                    result = conn.execute(
                        query,
                        {
                            "user_id": user_id,
                            "embedding": embedding_str,
                            "min_complexity": max(0, complexity - 2),
                            "max_complexity": complexity + 2,
                            "limit": limit,
                        },
                    )
                    rows = result.fetchall()
                else:
                    continue

                latency_ms = (time.time() - start_time) * 1000

                # Check if ground truth was found
                if filter_type in ["no_filter", "user_jsonb", "user_jsonb_complex"]:
                    found_ids = [row[0] for row in rows]
                else:
                    found_ids = [row["id"] for row in rows]
                ground_truth_id = query_data["ground_truth_chunk_id"]
                found = ground_truth_id in found_ids
                rank = found_ids.index(ground_truth_id) + 1 if found else None

                results.append(
                    BenchmarkResult(
                        query_id=query_data["query_id"],
                        query_text=query_data["query_text"],
                        latency_ms=latency_ms,
                        result_count=len(rows),
                        filter_type=filter_type,
                        found_ground_truth=found,
                        ground_truth_rank=rank,
                        ground_truth_chunk_id=ground_truth_id,
                    )
                )

        return results

    def benchmark_text_search(self, filter_type: str, limit: int = 10) -> List[BenchmarkResult]:
        """Benchmark keyword/text search with different filters"""
        results = []

        with self.engine.connect() as conn:
            for query_data in self.queries:
                user_id = query_data["ground_truth_user_id"]
                query_text = query_data["query_text"]

                # Use SearchQuery class methods
                start_time = time.time()
                if filter_type == "no_filter":
                    # Direct SQL query without user filter
                    query = text("""
                        SELECT id, user_document_id,
                               ts_rank_cd(text_search_vector, websearch_to_tsquery('english', :query)) AS relevance
                        FROM user_document_chunks
                        WHERE websearch_to_tsquery('english', :query) @@ text_search_vector
                        ORDER BY relevance DESC
                        LIMIT :limit
                    """)
                    result = conn.execute(query, {"query": query_text, "limit": limit})
                    rows = result.fetchall()
                else:
                    # Use SearchEngine class with user filter
                    rows, _ = self.search.text_search(user_id, query_text, limit)

                latency_ms = (time.time() - start_time) * 1000

                # Check if ground truth was found
                if filter_type in ["no_filter", "user_jsonb", "user_jsonb_complex"]:
                    found_ids = [row[0] for row in rows]
                else:
                    found_ids = [row["id"] for row in rows]
                ground_truth_id = query_data["ground_truth_chunk_id"]
                found = ground_truth_id in found_ids
                rank = found_ids.index(ground_truth_id) + 1 if found else None

                results.append(
                    BenchmarkResult(
                        query_id=query_data["query_id"],
                        query_text=query_data["query_text"],
                        latency_ms=latency_ms,
                        result_count=len(rows),
                        filter_type=filter_type,
                        found_ground_truth=found,
                        ground_truth_rank=rank,
                        ground_truth_chunk_id=ground_truth_id,
                    )
                )

        return results

    def calculate_rrf_scores(
        self, vector_results: List[Dict], text_results: List[Dict], k: int = 60
    ) -> List[Dict]:
        """Calculate RRF scores in Python for two sets of results"""
        # Create dictionaries with ranks (use 'id' as the key)
        vector_ranks = {row["id"]: idx + 1 for idx, row in enumerate(vector_results)}
        text_ranks = {row["id"]: idx + 1 for idx, row in enumerate(text_results)}

        # Get all unique chunk IDs
        all_ids = set(vector_ranks.keys()) | set(text_ranks.keys())

        # Calculate RRF scores
        results = []
        for chunk_id in all_ids:
            v_rank = vector_ranks.get(chunk_id, 1000)  # High rank if not found
            t_rank = text_ranks.get(chunk_id, 1000)

            rrf_score = (1.0 / (k + v_rank)) + (1.0 / (k + t_rank))

            # Get the original row data from whichever result set has it
            if chunk_id in vector_ranks:
                row_data = vector_results[vector_ranks[chunk_id] - 1].copy()
            else:
                row_data = text_results[text_ranks[chunk_id] - 1].copy()

            row_data["rrf_score"] = rrf_score
            row_data["vector_rank"] = v_rank if v_rank < 1000 else None
            row_data["text_rank"] = t_rank if t_rank < 1000 else None
            row_data["chunk_id"] = chunk_id  # Add chunk_id for consistency
            results.append(row_data)

        return results

    def benchmark_hybrid_two_queries(self, filter_type: str, limit: int = 10) -> List[BenchmarkResult]:
        """Benchmark hybrid search using two separate queries combined in Python"""
        results = []

        for query_data in self.queries:
            # Skip if no embedding available
            if "embedding" not in query_data:
                continue

            user_id = query_data["ground_truth_user_id"]
            embedding = query_data["embedding"]
            query_text = query_data["query_text"]

            # Two-query approach always requires user_id
            if filter_type == "no_filter":
                continue

            # Measure total time and component times
            total_start = time.time()

            # Query 1: Vector search
            vector_start = time.time()
            vector_results, _ = self.search.vector_search(
                user_id=user_id,
                embedding=embedding,
                limit=40,  # Get more results for RRF
            )
            vector_time = (time.time() - vector_start) * 1000

            # Query 2: Text search
            text_start = time.time()
            text_results, _ = self.search.text_search(
                user_id=user_id,
                query_text=query_text,
                limit=40,  # Get more results for RRF
            )
            text_time = (time.time() - text_start) * 1000

            # RRF combination in Python
            rrf_start = time.time()
            combined = self.calculate_rrf_scores(vector_results, text_results, k=60)
            # Sort by RRF score and take top results
            combined.sort(key=lambda x: x["rrf_score"], reverse=True)
            final_results = combined[:limit]
            rrf_time = (time.time() - rrf_start) * 1000

            total_latency = (time.time() - total_start) * 1000

            # Check if ground truth was found (use 'chunk_id' or 'id')
            found_ids = [row.get("chunk_id", row.get("id")) for row in final_results]
            ground_truth_id = query_data["ground_truth_chunk_id"]
            found = ground_truth_id in found_ids
            rank = found_ids.index(ground_truth_id) + 1 if found else None

            # Store detailed timing info in the result
            result = BenchmarkResult(
                query_id=query_data["query_id"],
                query_text=query_data["query_text"],
                latency_ms=total_latency,
                result_count=len(final_results),
                filter_type=filter_type,
                found_ground_truth=found,
                ground_truth_rank=rank,
                ground_truth_chunk_id=ground_truth_id,
            )

            # Add component timing as attributes
            result.vector_ms = vector_time
            result.text_ms = text_time
            result.rrf_ms = rrf_time

            results.append(result)

        return results

    def benchmark_hybrid_search(self, filter_type: str, limit: int = 10) -> List[BenchmarkResult]:
        """Benchmark hybrid search combining vector and text search"""
        results = []

        for query_data in self.queries:
            # Skip if no embedding available
            if "embedding" not in query_data:
                continue

            user_id = query_data["ground_truth_user_id"]
            embedding = query_data["embedding"]
            query_text = query_data["query_text"]

            # Hybrid search always requires user_id
            if filter_type == "no_filter":
                continue

            # Use SearchEngine class hybrid search
            start_time = time.time()
            rows, _ = self.search.hybrid_search_function(
                user_id=user_id, embedding=embedding, query_text=query_text, limit=limit
            )
            latency_ms = (time.time() - start_time) * 1000

            # Check if ground truth was found
            found_ids = [row["chunk_id"] for row in rows]
            ground_truth_id = query_data["ground_truth_chunk_id"]
            found = ground_truth_id in found_ids
            rank = found_ids.index(ground_truth_id) + 1 if found else None

            results.append(
                BenchmarkResult(
                    query_id=query_data["query_id"],
                    query_text=query_data["query_text"],
                    latency_ms=latency_ms,
                    result_count=len(rows),
                    filter_type=filter_type,
                    found_ground_truth=found,
                    ground_truth_rank=rank,
                    ground_truth_chunk_id=ground_truth_id,
                )
            )

        return results

    def calculate_metrics(self, results: List[BenchmarkResult]) -> Dict:
        """Calculate metrics from results"""
        if not results:
            return {"mean": 0, "median": 0, "stdev": 0, "recall": 0, "precision_at_10": 0}

        latencies = [r.latency_ms for r in results]
        found_count = sum(1 for r in results if r.found_ground_truth)
        total = len(results)

        # Calculate precision at rank 10
        precision_count = sum(1 for r in results if r.found_ground_truth and (r.ground_truth_rank or 0) <= 10)

        return {
            "mean": mean(latencies),
            "median": median(latencies),
            "stdev": stdev(latencies) if len(latencies) > 1 else 0,
            "recall": (found_count / total * 100) if total > 0 else 0,
            "precision_at_10": (precision_count / total * 100) if total > 0 else 0,
            "total_queries": total,
            "found": found_count,
        }

    def run(self, limit: int = 10):
        """Run complete benchmark suite"""
        self.console.print("\n[bold cyan]Starting Comprehensive Search Benchmark[/bold cyan]\n")
        self.console.print(f"Running {len(self.queries)} queries for each test...\n")

        # Test scenarios
        scenarios = {
            "vector": ["no_filter", "user_only", "user_jsonb", "user_jsonb_complex"],
            "text": ["no_filter", "user_only"],
            "hybrid": ["user_only"],  # Hybrid requires user_id
            "hybrid_two_queries": ["user_only"],  # Two-query hybrid approach
        }

        all_results = {}

        # Run benchmarks for each search type
        for search_type, filter_types in scenarios.items():
            self.console.print(f"\n[cyan]Benchmarking {search_type.upper()} search...[/cyan]")
            all_results[search_type] = {}

            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=self.console
            ) as progress:
                task = progress.add_task(f"Running {search_type} benchmarks...", total=len(filter_types))

                for filter_type in filter_types:
                    if search_type == "vector":
                        results = self.benchmark_vector_search(filter_type, limit)
                    elif search_type == "text":
                        results = self.benchmark_text_search(filter_type, limit)
                    elif search_type == "hybrid":
                        results = self.benchmark_hybrid_search(filter_type, limit)
                    else:  # hybrid_two_queries
                        results = self.benchmark_hybrid_two_queries(filter_type, limit)

                    all_results[search_type][filter_type] = results
                    progress.advance(task)

        # Display results
        self.display_results(all_results)

        # Display hybrid comparison if both methods were run
        if "hybrid" in all_results and "hybrid_two_queries" in all_results:
            self.display_hybrid_comparison(all_results)

        # Save detailed results
        self.save_results(all_results)

    def display_results(self, all_results: Dict):
        """Display benchmark results in tables"""

        # Performance metrics table
        self.console.print("\n[bold cyan]Performance Metrics (Latency in ms)[/bold cyan]")
        perf_table = Table(show_header=True, header_style="bold magenta")
        perf_table.add_column("Search Type", style="cyan")
        perf_table.add_column("Filter", style="yellow")
        perf_table.add_column("Mean", justify="right")
        perf_table.add_column("Median", justify="right")
        perf_table.add_column("StdDev", justify="right")

        for search_type, filter_results in all_results.items():
            for filter_type, results in filter_results.items():
                metrics = self.calculate_metrics(results)
                perf_table.add_row(
                    search_type.title(),
                    filter_type.replace("_", " ").title(),
                    f"{metrics['mean']:.2f}",
                    f"{metrics['median']:.2f}",
                    f"{metrics['stdev']:.2f}",
                )

        self.console.print(perf_table)

        # Accuracy metrics table
        self.console.print("\n[bold cyan]Accuracy Metrics[/bold cyan]")
        acc_table = Table(show_header=True, header_style="bold magenta")
        acc_table.add_column("Search Type", style="cyan")
        acc_table.add_column("Filter", style="yellow")
        acc_table.add_column("Recall %", justify="right")
        acc_table.add_column("Precision@10 %", justify="right")
        acc_table.add_column("Found/Total", justify="right")

        for search_type, filter_results in all_results.items():
            for filter_type, results in filter_results.items():
                metrics = self.calculate_metrics(results)
                acc_table.add_row(
                    search_type.title(),
                    filter_type.replace("_", " ").title(),
                    f"{metrics['recall']:.1f}",
                    f"{metrics['precision_at_10']:.1f}",
                    f"{metrics['found']}/{metrics['total_queries']}",
                )

        self.console.print(acc_table)

    def display_hybrid_comparison(self, all_results: Dict):
        """Display detailed comparison of hybrid search approaches"""
        self.console.print("\n[bold cyan]Hybrid Search Method Comparison[/bold cyan]")

        # Get results for both approaches
        db_function_results = all_results["hybrid"]["user_only"]
        two_query_results = all_results["hybrid_two_queries"]["user_only"]

        # Calculate metrics for both
        db_metrics = self.calculate_metrics(db_function_results)
        tq_metrics = self.calculate_metrics(two_query_results)

        # Component timing for two-query approach
        vector_times = [r.vector_ms for r in two_query_results if hasattr(r, "vector_ms")]
        text_times = [r.text_ms for r in two_query_results if hasattr(r, "text_ms")]
        rrf_times = [r.rrf_ms for r in two_query_results if hasattr(r, "rrf_ms")]

        # Create comparison table
        comp_table = Table(show_header=True, header_style="bold magenta")
        comp_table.add_column("Method", style="cyan")
        comp_table.add_column("Mean (ms)", justify="right")
        comp_table.add_column("Median (ms)", justify="right")
        comp_table.add_column("StdDev", justify="right")
        comp_table.add_column("Recall %", justify="right")
        comp_table.add_column("Details", style="dim")

        comp_table.add_row(
            "DB Function (Single Query)",
            f"{db_metrics['mean']:.2f}",
            f"{db_metrics['median']:.2f}",
            f"{db_metrics['stdev']:.2f}",
            f"{db_metrics['recall']:.1f}",
            "CTE with RRF in PostgreSQL",
        )

        comp_table.add_row(
            "Two Separate Queries",
            f"{tq_metrics['mean']:.2f}",
            f"{tq_metrics['median']:.2f}",
            f"{tq_metrics['stdev']:.2f}",
            f"{tq_metrics['recall']:.1f}",
            f"Vector: {mean(vector_times) if vector_times else 0:.1f}ms, Text: {mean(text_times) if text_times else 0:.1f}ms, RRF: {mean(rrf_times) if rrf_times else 0:.1f}ms",
        )

        self.console.print(comp_table)

        # Show performance difference
        speedup = tq_metrics["mean"] / db_metrics["mean"] if db_metrics["mean"] > 0 else 1
        if speedup > 1:
            self.console.print(f"\n[green]✓ DB Function is {speedup:.2f}x faster than Two Queries[/green]")
        else:
            self.console.print(
                f"\n[yellow]⚠ Two Queries is {1 / speedup:.2f}x faster than DB Function[/yellow]"
            )

        # Component breakdown for two-query approach
        if vector_times and text_times and rrf_times:
            self.console.print(f"\n[dim]Two-Query Component Breakdown:[/dim]")
            self.console.print(
                f"  Vector Search: {mean(vector_times):.1f}ms ({mean(vector_times) / tq_metrics['mean'] * 100:.0f}%)"
            )
            self.console.print(
                f"  Text Search:   {mean(text_times):.1f}ms ({mean(text_times) / tq_metrics['mean'] * 100:.0f}%)"
            )
            self.console.print(
                f"  RRF Scoring:   {mean(rrf_times):.1f}ms ({mean(rrf_times) / tq_metrics['mean'] * 100:.0f}%)"
            )

    def save_results(self, all_results: Dict):
        """Save detailed results to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create sql-out directory if it doesn't exist
        output_dir = Path("sql-out")
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"benchmark_results_{timestamp}.csv"

        with open(output_file, "w", newline="") as f:
            fieldnames = [
                "search_type",
                "filter_type",
                "query_id",
                "query_text",
                "latency_ms",
                "found_ground_truth",
                "ground_truth_rank",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for search_type, filter_results in all_results.items():
                for filter_type, results in filter_results.items():
                    for r in results:
                        writer.writerow(
                            {
                                "search_type": search_type,
                                "filter_type": filter_type,
                                "query_id": r.query_id,
                                "query_text": r.query_text,
                                "latency_ms": r.latency_ms,
                                "found_ground_truth": r.found_ground_truth,
                                "ground_truth_rank": r.ground_truth_rank,
                            }
                        )

        self.console.print(f"\n[green]Detailed results saved to: {output_file}[/green]")


def main():
    """Main benchmark function"""
    parser = argparse.ArgumentParser(description="Benchmark search performance with various filters")
    parser.add_argument("csv_file", type=str, help="Path to benchmark CSV file")
    parser.add_argument("--json", type=str, help="Path to JSON file with embeddings")
    parser.add_argument("--limit", type=int, default=10, help="Result limit for searches (default: 10)")

    args = parser.parse_args()

    # Auto-detect JSON file if not provided
    json_file = args.json
    if not json_file:
        # Try to find matching JSON file
        csv_path = Path(args.csv_file)
        json_path = csv_path.with_suffix(".json")
        if json_path.exists():
            json_file = str(json_path)
            Console().print(f"[yellow]Auto-detected embeddings file: {json_file}[/yellow]")

    try:
        benchmark = SearchBenchmark(csv_file=args.csv_file, json_file=json_file)
        benchmark.run(limit=args.limit)
    except Exception as e:
        Console().print(f"[bold red]Error:[/bold red] {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
