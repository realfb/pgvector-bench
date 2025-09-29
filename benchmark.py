import os
import time
import json
import numpy as np
import psycopg2
from datetime import datetime
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from statistics import mean, median, stdev
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import argparse

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", 54320),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "database": os.getenv("DB_NAME", "leo_pgvector"),
}


class SearchType(Enum):
    VECTOR = "vector"
    TEXT = "text"
    HYBRID = "hybrid"


class FilterStrategy(Enum):
    NO_FILTER = "no_filter"
    USER_ID = "user_id"
    JSONB = "jsonb"
    COMPOSITE = "composite"


class SchemaDesign(Enum):
    DENORMALIZED = "denorm"
    NORMALIZED_JOIN = "join"


@dataclass
class BenchResult:
    config: str
    search_type: str
    filter_strategy: str
    schema_design: str
    avg_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_dev: float
    recall: float
    mrr: float
    qps: float


class ComprehensiveBenchmark:
    def __init__(self, queries: int = 30, limit: int = 10, verbose: bool = False):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.cur = self.conn.cursor(cursor_factory=RealDictCursor)
        self.queries = queries
        self.limit = limit
        self.verbose = verbose
        self.results = []

        # Register pgvector if available
        try:
            from pgvector.psycopg2 import register_vector

            register_vector(self.conn)
        except ImportError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cur.close()
        self.conn.close()

    def prepare_data(self) -> Dict:
        """Prepare comprehensive test data."""
        # Discover JSONB keys
        self.cur.execute("""
            SELECT jsonb_object_keys(meta) as key, COUNT(*) as count
            FROM user_document_chunks
            WHERE meta IS NOT NULL
            GROUP BY key ORDER BY count DESC LIMIT 10
        """)
        keys = self.cur.fetchall()

        jsonb_filter = None
        if keys:
            # Rotate through different keys for diversity
            import random
            # Use different keys based on test run to get variety
            key_idx = random.randint(0, min(2, len(keys) - 1))
            key = keys[key_idx]["key"]

            # Get common values for this key
            self.cur.execute(
                """
                SELECT meta->>%s as val, COUNT(*) as cnt
                FROM user_document_chunks
                WHERE meta ? %s AND meta->>%s IS NOT NULL
                GROUP BY val ORDER BY cnt DESC LIMIT 10
            """,
                (key, key, key),
            )
            values = self.cur.fetchall()

            if values:
                # Randomly select from top values for diversity
                # Weight towards more common values but allow variety
                top_values = [v for v in values if v["cnt"] > 100]
                if top_values:
                    selected = random.choice(top_values[:5])
                    jsonb_filter = {key: selected["val"]}
                    print(f"Selected JSONB filter: {jsonb_filter} (count: {selected['cnt']})")

        # Sample test data
        self.cur.execute(
            """
            WITH sampled AS (
                SELECT * FROM user_document_chunks
                TABLESAMPLE SYSTEM (2) LIMIT %s
            )
            SELECT * FROM sampled WHERE meta IS NOT NULL
            ORDER BY random() LIMIT %s
        """,
            (self.queries * 3, self.queries),
        )

        data = []
        for row in self.cur.fetchall():
            text = row["text"]
            lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 20]
            search_text = " ".join((lines[0] if lines else text).split()[:10])

            data.append(
                {
                    "id": row["id"],
                    "user_id": row["user_id"],
                    "embedding": row["embedding"],
                    "search_text": search_text,
                    "meta": row["meta"],
                    "user_document_id": row["user_document_id"],
                }
            )

        return {"samples": data, "jsonb_filter": jsonb_filter, "jsonb_key": keys[0]["key"] if keys else None}

    def _vector_str(self, embedding) -> str:
        """Convert embedding to PostgreSQL vector string."""
        if isinstance(embedding, str):
            return embedding
        elif isinstance(embedding, (list, tuple)):
            return "[" + ",".join(map(str, embedding)) + "]"
        elif hasattr(embedding, "tolist"):
            return "[" + ",".join(map(str, embedding.tolist())) + "]"
        else:
            return str(embedding)

    def _measure(self, query: str, params: Tuple) -> Tuple[float, List[int]]:
        """Execute query and measure latency."""
        start = time.perf_counter()
        self.cur.execute(query, params)
        results = self.cur.fetchall()
        latency = (time.perf_counter() - start) * 1000
        ids = [r["id"] if isinstance(r, dict) else r[0] for r in results]
        return latency, ids

    def _calc_metrics(self, latencies: List[float], recalls: List[float]) -> Dict:
        """Calculate comprehensive metrics."""
        if not latencies:
            return None

        sorted_lat = sorted(latencies)
        return {
            "avg_ms": mean(latencies),
            "median_ms": median(latencies),
            "p95_ms": sorted_lat[int(len(sorted_lat) * 0.95) - 1] if len(sorted_lat) > 1 else sorted_lat[0],
            "p99_ms": sorted_lat[int(len(sorted_lat) * 0.99) - 1]
            if len(sorted_lat) >= 100
            else sorted_lat[-1],
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "std_dev": stdev(latencies) if len(latencies) > 1 else 0,
            "recall": sum(1 for r in recalls if r > 0) / len(recalls) if recalls else 0,
            "mrr": mean(recalls) if recalls else 0,
            "qps": 1000 / mean(latencies) if mean(latencies) > 0 else 0,
        }

    def bench_vector(
        self, data: Dict, filter_strat: FilterStrategy, schema: SchemaDesign
    ) -> Optional[BenchResult]:
        """Benchmark vector search with all combinations."""
        latencies, recalls = [], []

        for sample in data["samples"]:
            vector = self._vector_str(sample["embedding"])
            user_id = sample["user_id"]

            # Build query based on schema and filter
            if schema == SchemaDesign.DENORMALIZED:
                base = "SELECT id FROM user_document_chunks"
                order = "ORDER BY embedding <#> %s::vector LIMIT %s"

                if filter_strat == FilterStrategy.NO_FILTER:
                    query = f"{base} {order}"
                    params = (vector, self.limit)
                elif filter_strat == FilterStrategy.USER_ID:
                    query = f"{base} WHERE user_id = %s {order}"
                    params = (user_id, vector, self.limit)
                elif filter_strat == FilterStrategy.JSONB and data["jsonb_filter"]:
                    query = f"{base} WHERE meta @> %s::jsonb {order}"
                    params = (json.dumps(data["jsonb_filter"]), vector, self.limit)
                elif filter_strat == FilterStrategy.COMPOSITE:
                    query = f"{base} WHERE user_id = %s AND created_at > '2024-01-01' {order}"
                    params = (user_id, vector, self.limit)
                else:
                    continue

            else:  # NORMALIZED_JOIN
                base = """SELECT c.id FROM user_document_chunks c
                         JOIN user_documents d ON c.user_document_id = d.id"""
                order = "ORDER BY c.embedding <#> %s::vector LIMIT %s"

                if filter_strat == FilterStrategy.NO_FILTER:
                    query = f"{base} {order}"
                    params = (vector, self.limit)
                elif filter_strat == FilterStrategy.USER_ID:
                    query = f"{base} WHERE d.user_id = %s {order}"
                    params = (user_id, vector, self.limit)
                elif filter_strat == FilterStrategy.JSONB and data["jsonb_filter"]:
                    query = f"{base} WHERE c.meta @> %s::jsonb {order}"
                    params = (json.dumps(data["jsonb_filter"]), vector, self.limit)
                else:
                    continue

            try:
                latency, ids = self._measure(query, params)
                latencies.append(latency)
                recalls.append(1.0 / (ids.index(sample["id"]) + 1) if sample["id"] in ids else 0.0)
            except Exception as e:
                if self.verbose:
                    print(f"Error in vector search: {e}")
                continue

        metrics = self._calc_metrics(latencies, recalls)
        if metrics:
            config = f"Vector/{schema.value}/{filter_strat.value}"
            return BenchResult(config, "vector", filter_strat.value, schema.value, **metrics)
        return None

    def bench_text(
        self, data: Dict, filter_strat: FilterStrategy, schema: SchemaDesign
    ) -> Optional[BenchResult]:
        """Benchmark text search with all combinations."""
        latencies, recalls = [], []

        for sample in data["samples"]:
            text = sample["search_text"]
            user_id = sample["user_id"]

            # Build query
            if schema == SchemaDesign.DENORMALIZED:
                base = "SELECT id FROM user_document_chunks"
                where_base = "text_search_vector @@ websearch_to_tsquery('english', %s)"
                order = "ORDER BY ts_rank_cd(text_search_vector, websearch_to_tsquery('english', %s)) DESC LIMIT %s"

                if filter_strat == FilterStrategy.NO_FILTER:
                    query = f"{base} WHERE {where_base} {order}"
                    params = (text, text, self.limit)
                elif filter_strat == FilterStrategy.USER_ID:
                    query = f"{base} WHERE user_id = %s AND {where_base} {order}"
                    params = (user_id, text, text, self.limit)
                elif filter_strat == FilterStrategy.JSONB and data["jsonb_filter"]:
                    query = f"{base} WHERE meta @> %s::jsonb AND {where_base} {order}"
                    params = (json.dumps(data["jsonb_filter"]), text, text, self.limit)
                elif filter_strat == FilterStrategy.COMPOSITE:
                    query = (
                        f"{base} WHERE user_id = %s AND created_at > '2024-01-01' AND {where_base} {order}"
                    )
                    params = (user_id, text, text, self.limit)
                else:
                    continue

            else:  # NORMALIZED_JOIN
                base = """SELECT c.id FROM user_document_chunks c
                         JOIN user_documents d ON c.user_document_id = d.id"""
                where_base = "c.text_search_vector @@ websearch_to_tsquery('english', %s)"
                order = "ORDER BY ts_rank_cd(c.text_search_vector, websearch_to_tsquery('english', %s)) DESC LIMIT %s"

                if filter_strat == FilterStrategy.NO_FILTER:
                    query = f"{base} WHERE {where_base} {order}"
                    params = (text, text, self.limit)
                elif filter_strat == FilterStrategy.USER_ID:
                    query = f"{base} WHERE d.user_id = %s AND {where_base} {order}"
                    params = (user_id, text, text, self.limit)
                elif filter_strat == FilterStrategy.JSONB and data["jsonb_filter"]:
                    query = f"{base} WHERE c.meta @> %s::jsonb AND {where_base} {order}"
                    params = (json.dumps(data["jsonb_filter"]), text, text, self.limit)
                else:
                    continue

            try:
                latency, ids = self._measure(query, params)
                latencies.append(latency)
                recalls.append(1.0 / (ids.index(sample["id"]) + 1) if sample["id"] in ids else 0.0)
            except Exception:
                continue

        metrics = self._calc_metrics(latencies, recalls)
        if metrics:
            config = f"Text/{schema.value}/{filter_strat.value}"
            return BenchResult(config, "text", filter_strat.value, schema.value, **metrics)
        return None

    def bench_hybrid(self, data: Dict, filter_strat: FilterStrategy) -> Optional[BenchResult]:
        """Benchmark hybrid search using database function."""
        latencies, recalls = [], []

        for sample in data["samples"]:
            vector = self._vector_str(sample["embedding"])
            text = sample["search_text"]
            user_id = sample["user_id"]

            # Hybrid function requires user_id
            if filter_strat == FilterStrategy.NO_FILTER:
                continue
            elif filter_strat == FilterStrategy.USER_ID:
                query = "SELECT * FROM hybrid_search(%s, %s, %s::vector(768), %s)"
                params = (user_id, text, vector, self.limit)
            elif filter_strat == FilterStrategy.JSONB and data["jsonb_filter"]:
                query = """SELECT * FROM hybrid_search(%s, %s, %s::vector(768), %s,
                          chunk_meta_filter := %s::jsonb)"""
                params = (user_id, text, vector, self.limit, json.dumps(data["jsonb_filter"]))
            elif filter_strat == FilterStrategy.COMPOSITE and data["jsonb_filter"]:
                query = """SELECT * FROM hybrid_search(%s, %s, %s::vector(768), %s,
                          chunk_meta_filter := %s::jsonb, doc_meta_filter := %s::jsonb)"""
                params = (
                    user_id,
                    text,
                    vector,
                    self.limit,
                    json.dumps(data["jsonb_filter"]),
                    json.dumps({"quality_score": "high"}),
                )
            else:
                continue

            try:
                latency, ids = self._measure(query, params)
                latencies.append(latency)
                recalls.append(1.0 / (ids.index(sample["id"]) + 1) if sample["id"] in ids else 0.0)
            except Exception:
                continue

        metrics = self._calc_metrics(latencies, recalls)
        if metrics:
            config = f"Hybrid/DB/{filter_strat.value}"
            return BenchResult(config, "hybrid", filter_strat.value, "function", **metrics)
        return None

    def bench_weight_variations(self, data: Dict) -> List[BenchResult]:
        """Test different weight configurations for hybrid search."""
        results = []
        weight_configs = [
            (1.0, 1.0, "balanced"),
            (2.0, 1.0, "text_heavy"),
            (1.0, 2.0, "vector_heavy"),
            (1.0, 0.0, "text_only"),
            (0.0, 1.0, "vector_only"),
        ]

        for ft_weight, sem_weight, name in weight_configs:
            latencies = []

            for sample in data["samples"][:10]:  # Smaller sample for weight testing
                vector = self._vector_str(sample["embedding"])

                query = """SELECT * FROM hybrid_search(%s, %s, %s::vector(768), %s,
                          full_text_weight := %s, semantic_weight := %s)"""
                params = (sample["user_id"], sample["search_text"], vector, self.limit, ft_weight, sem_weight)

                try:
                    latency, _ = self._measure(query, params)
                    latencies.append(latency)
                except Exception:
                    continue

            if latencies:
                metrics = self._calc_metrics(latencies, [])
                config = f"Hybrid/Weights/{name}"
                results.append(
                    BenchResult(
                        config,
                        "hybrid",
                        name,
                        "weights",
                        avg_ms=metrics["avg_ms"],
                        median_ms=metrics["median_ms"],
                        p95_ms=metrics["p95_ms"],
                        p99_ms=metrics["p99_ms"],
                        min_ms=metrics["min_ms"],
                        max_ms=metrics["max_ms"],
                        std_dev=metrics["std_dev"],
                        recall=0,
                        mrr=0,
                        qps=metrics["qps"],
                    )
                )

        return results


    def run_comprehensive(self) -> None:
        """Run all benchmark scenarios."""
        print("\n🚀 Comprehensive PGVector Benchmark")
        print("=" * 70)

        data = self.prepare_data()
        print(f"✓ Prepared {len(data['samples'])} test samples")
        if data["jsonb_filter"]:
            print(f"✓ Using JSONB filter: {data['jsonb_filter']}")

        # Test matrix
        test_matrix = []

        # Vector searches
        for schema in [SchemaDesign.DENORMALIZED, SchemaDesign.NORMALIZED_JOIN]:
            for filter_strat in FilterStrategy:
                test_matrix.append(("vector", filter_strat, schema))

        # Text searches
        for schema in [SchemaDesign.DENORMALIZED, SchemaDesign.NORMALIZED_JOIN]:
            for filter_strat in FilterStrategy:
                test_matrix.append(("text", filter_strat, schema))

        # Hybrid searches (function-based, no schema variations)
        for filter_strat in FilterStrategy:
            if filter_strat != FilterStrategy.NO_FILTER:  # Hybrid needs user_id
                test_matrix.append(("hybrid", filter_strat, None))

        # Run benchmarks
        print(f"\n📊 Running {len(test_matrix)} test configurations...")

        for search_type, filter_strat, schema in test_matrix:
            if search_type == "vector":
                result = self.bench_vector(data, filter_strat, schema)
            elif search_type == "text":
                result = self.bench_text(data, filter_strat, schema)
            elif search_type == "hybrid":
                result = self.bench_hybrid(data, filter_strat)
            else:
                continue

            if result:
                self.results.append(result)
                if self.verbose:
                    print(
                        f"  {result.config:30s}: {result.avg_ms:7.2f}ms (recall: {result.recall * 100:3.0f}%)"
                    )

        # Weight variations
        print("\n⚖️  Testing weight variations...")
        weight_results = self.bench_weight_variations(data)
        self.results.extend(weight_results)

        # Display and save results
        self.display_results()
        self.save_results()

    def display_results(self) -> None:
        """Display formatted results."""
        if not self.results:
            print("No results to display")
            return

        # Group results
        vectors = [r for r in self.results if r.search_type == "vector"]
        texts = [r for r in self.results if r.search_type == "text"]
        hybrids = [r for r in self.results if r.search_type == "hybrid" and "Weights" not in r.config]
        weights = [r for r in self.results if "Weights" in r.config]

        # Import rich components for better formatting
        try:
            from rich.table import Table
            from rich.console import Console

            console = Console()

            # Summary header
            console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
            console.print("[bold cyan]RESULTS SUMMARY[/bold cyan]")
            console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")

            # Best performers table
            if vectors or texts or hybrids:
                best_table = Table(title="📈 Best Performers by Category")
                best_table.add_column("Category", style="cyan", width=12)
                best_table.add_column("Configuration", style="yellow", width=30)
                best_table.add_column("Latency", justify="right", style="green")

                if vectors:
                    best_v = min(vectors, key=lambda x: x.avg_ms)
                    best_table.add_row("Vector", best_v.config, f"{best_v.avg_ms:.2f}ms")
                if texts:
                    best_t = min(texts, key=lambda x: x.avg_ms)
                    best_table.add_row("Text", best_t.config, f"{best_t.avg_ms:.2f}ms")
                if hybrids:
                    best_h = min(hybrids, key=lambda x: x.avg_ms)
                    best_table.add_row("Hybrid", best_h.config, f"{best_h.avg_ms:.2f}ms")

                console.print("\n")
                console.print(best_table)

            # Filter impact table
            filter_table = Table(title="🔍 Filter Impact (Denormalized Schema)")
            filter_table.add_column("Search Type", style="cyan")
            filter_table.add_column("User Filter", justify="center", style="green")
            filter_table.add_column("JSONB Filter", justify="center", style="yellow")

            for search_type in ["vector", "text"]:
                type_results = [
                    r for r in self.results if r.search_type == search_type and r.schema_design == "denorm"
                ]
                if type_results:
                    no_filter = next((r for r in type_results if r.filter_strategy == "no_filter"), None)
                    user_filter = next((r for r in type_results if r.filter_strategy == "user_id"), None)
                    jsonb_filter = next((r for r in type_results if r.filter_strategy == "jsonb"), None)

                    user_speedup = (
                        f"{no_filter.avg_ms / user_filter.avg_ms:.1f}x faster"
                        if no_filter and user_filter
                        else "N/A"
                    )
                    jsonb_speedup = (
                        f"{no_filter.avg_ms / jsonb_filter.avg_ms:.1f}x faster"
                        if no_filter and jsonb_filter
                        else "N/A"
                    )

                    filter_table.add_row(search_type.capitalize(), user_speedup, jsonb_speedup)

            if filter_table.rows:
                console.print("\n")
                console.print(filter_table)

        except ImportError:
            # Fallback to simple output if rich is not available
            print("\n" + "=" * 70)
            print("RESULTS SUMMARY")
            print("=" * 70)

            print("\n📈 Best Performers:")
            if vectors:
                best_v = min(vectors, key=lambda x: x.avg_ms)
                print(f"  Vector : {best_v.config:30s} {best_v.avg_ms:7.2f}ms")
            if texts:
                best_t = min(texts, key=lambda x: x.avg_ms)
                print(f"  Text   : {best_t.config:30s} {best_t.avg_ms:7.2f}ms")
            if hybrids:
                best_h = min(hybrids, key=lambda x: x.avg_ms)
                print(f"  Hybrid : {best_h.config:30s} {best_h.avg_ms:7.2f}ms")

            print("\n🔍 Filter Impact (Denormalized):")
            for search_type in ["vector", "text"]:
                type_results = [
                    r for r in self.results if r.search_type == search_type and r.schema_design == "denorm"
                ]
                if type_results:
                    no_filter = next((r for r in type_results if r.filter_strategy == "no_filter"), None)
                    user_filter = next((r for r in type_results if r.filter_strategy == "user_id"), None)
                    jsonb_filter = next((r for r in type_results if r.filter_strategy == "jsonb"), None)

                    if no_filter and user_filter:
                        speedup = no_filter.avg_ms / user_filter.avg_ms
                        print(f"  {search_type:7s}: User filter {speedup:5.1f}x faster")
                    if no_filter and jsonb_filter:
                        speedup = no_filter.avg_ms / jsonb_filter.avg_ms
                        print(f"  {search_type:7s}: JSONB filter {speedup:5.1f}x faster")

        # Schema comparison
        print("\n🏗️  Schema Impact:")
        for search_type in ["vector", "text"]:
            for filter_strat in ["no_filter", "user_id"]:
                denorm = next(
                    (
                        r
                        for r in self.results
                        if r.search_type == search_type
                        and r.schema_design == "denorm"
                        and r.filter_strategy == filter_strat
                    ),
                    None,
                )
                joined = next(
                    (
                        r
                        for r in self.results
                        if r.search_type == search_type
                        and r.schema_design == "join"
                        and r.filter_strategy == filter_strat
                    ),
                    None,
                )

                if denorm and joined:
                    speedup = joined.avg_ms / denorm.avg_ms
                    if speedup > 1.2:
                        print(f"  {search_type}/{filter_strat}: Denorm {speedup:5.1f}x faster than JOIN")

        # Weight variations and concurrency with rich tables
        try:
            from rich.table import Table
            from rich.console import Console

            console = Console()

            # Weight variations table
            if weights:
                weight_table = Table(title="⚙️ Hybrid Weight Variations")
                weight_table.add_column("Configuration", style="cyan")
                weight_table.add_column("Latency", justify="right", style="green")
                weight_table.add_column("QPS", justify="right", style="yellow")

                for w in sorted(weights, key=lambda x: x.avg_ms):
                    weight_table.add_row(
                        w.filter_strategy.replace("_", " ").title(), f"{w.avg_ms:.2f}ms", f"{w.qps:.0f}"
                    )

                console.print("\n")
                console.print(weight_table)

        except ImportError:
            # Fallback if rich is not available
            if weights:
                print("\n⚙️  Hybrid Weight Variations:")
                for w in sorted(weights, key=lambda x: x.avg_ms):
                    print(f"  {w.filter_strategy:12s}: {w.avg_ms:7.2f}ms")

            if conc_results:
                print("\n🚀 Throughput:")
                for name, metrics in conc_results.items():
                    print(f"  {name:10s}: {metrics['avg_ms']:7.2f}ms, {metrics['qps']:6.1f} QPS")

        # Full results table with rich formatting
        print("\n📊 Complete Results Table:")

        # Import here to avoid dependency if not needed
        try:
            from rich.table import Table
            from rich.console import Console

            console = Console()
            table = Table(title="Search Performance Results (sorted by latency)")
            table.add_column("Configuration", style="cyan", width=35)
            table.add_column("Mean", justify="right", style="green")
            table.add_column("Median", justify="right", style="blue")
            table.add_column("P95", justify="right", style="yellow")
            table.add_column("P99", justify="right", style="red")
            table.add_column("QPS", justify="right", style="white")
            table.add_column("Recall", justify="right", style="magenta")

            for r in sorted(self.results, key=lambda x: x.avg_ms):
                if "Weights" not in r.config:  # Skip weight variations in main table
                    table.add_row(
                        r.config,
                        f"{r.avg_ms:.1f}ms",
                        f"{r.median_ms:.1f}ms",
                        f"{r.p95_ms:.1f}ms",
                        f"{r.p99_ms:.1f}ms",
                        f"{r.qps:.0f}",
                        f"{r.recall * 100:.0f}%",
                    )

            console.print(table)

        except ImportError:
            # Fallback to simple table if rich is not available
            print(f"{'Configuration':<35} {'Mean':>8} {'P95':>8} {'P99':>8} {'Recall':>8}")
            print("-" * 70)

            for r in sorted(self.results, key=lambda x: x.avg_ms):
                if "Weights" not in r.config:  # Skip weight variations in main table
                    print(
                        f"{r.config:<35} {r.avg_ms:>7.1f}ms {r.p95_ms:>7.1f}ms "
                        f"{r.p99_ms:>7.1f}ms {r.recall * 100:>7.0f}%"
                    )

        # Add legend
        print("\n📖 Legend:")
        print("  Configuration format: SearchType/Schema/Filter")
        print("\n  Search Types:")
        print("    • Vector: Vector similarity search using embedding <#> operator")
        print("    • Text: Full-text search using tsvector @@ tsquery")
        print("    • Hybrid: Combined vector + text using hybrid_search() function")
        print("\n  Schema Designs:")
        print("    • denorm: Direct queries on user_document_chunks table")
        print("    • join: Queries with JOIN to user_documents table")
        print("    • DB: Database function (hybrid_search only)")
        print("\n  Filter Strategies:")
        print("    • no_filter: No filtering applied")
        print("    • user_id: Filter by user_id")
        print("    • jsonb: Filter by JSONB metadata (meta @> filter)")
        print("    • composite: Multiple filters (user_id + date/metadata)")

    def save_results(self) -> None:
        """Save comprehensive results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("sql-out", exist_ok=True)

        # Prepare output data
        output = {
            "timestamp": timestamp,
            "config": {"queries": self.queries, "limit": self.limit},
            "results": [asdict(r) for r in self.results],
            "summary": {
                "total_tests": len(self.results),
                "best_vector_ms": min(
                    [r.avg_ms for r in self.results if r.search_type == "vector"], default=0
                ),
                "best_text_ms": min([r.avg_ms for r in self.results if r.search_type == "text"], default=0),
                "best_hybrid_ms": min(
                    [r.avg_ms for r in self.results if r.search_type == "hybrid"], default=0
                ),
                "max_qps": max([r.qps for r in self.results], default=0),
            },
        }

        # JSON output
        json_file = f"sql-out/benchmark_comprehensive_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(output, f, indent=2)

        # CSV output
        csv_file = f"sql-out/benchmark_comprehensive_{timestamp}.csv"
        with open(csv_file, "w") as f:
            f.write(
                "config,search_type,filter_strategy,schema_design,avg_ms,median_ms,p95_ms,p99_ms,recall,mrr,qps\n"
            )
            for r in self.results:
                f.write(
                    f"{r.config},{r.search_type},{r.filter_strategy},{r.schema_design},"
                    f"{r.avg_ms:.2f},{r.median_ms:.2f},{r.p95_ms:.2f},{r.p99_ms:.2f},"
                    f"{r.recall:.3f},{r.mrr:.3f},{r.qps:.1f}\n"
                )

        print("\n✅ Results saved to:")
        print(f"  • {json_file}")
        print(f"  • {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive PGVector Benchmark")
    parser.add_argument("--queries", type=int, default=30, help="Number of test queries")
    parser.add_argument("--limit", type=int, default=10, help="Result limit per query")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    with ComprehensiveBenchmark(args.queries, args.limit, args.verbose) as bench:
        bench.run_comprehensive()
