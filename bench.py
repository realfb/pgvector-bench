"""
Benchmark and analysis tools for hybrid search performance
"""

import os
import time
import json
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from dotenv import load_dotenv
from datasets import load_dataset
import numpy as np

from models import User, UserDocument, UserDocumentChunk
from schemas import SearchRequest, SearchType
from query import SearchEngine

load_dotenv()
console = Console()


class SearchBenchmark:
    """Benchmark tool for analyzing search performance"""
    
    def __init__(self, db_url: Optional[str] = None):
        if db_url is None:
            db_url = os.getenv(
                'DATABASE_URL',
                'postgresql://postgres:postgres@localhost:54320/leo_pgvector'
            )
        
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.search_engine = SearchEngine(db_url)
    
    def analyze_query_performance(
        self, 
        request: SearchRequest
    ) -> Dict[str, Any]:
        """
        Analyze query performance and index usage with detailed metrics.
        
        Args:
            request: SearchRequest to analyze
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            'request': {
                'search_type': request.search_type.value,
                'limit': request.limit,
                'search_depth': request.search_depth,
                'rrf_k': request.rrf_k,
                'user_id': request.user_id
            },
            'performance': {}
        }
        
        # Analyze vector search if embedding provided
        if request.embedding:
            embedding_str = '[' + ','.join(map(str, request.embedding)) + ']'
            
            with self.engine.connect() as conn:
                # Get execution plan
                result = conn.execute(text("""
                    EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
                    SELECT c.id
                    FROM user_document_chunks c
                    JOIN user_documents d ON c.user_document_id = d.id
                    WHERE (:user_id IS NULL OR d.user_id = :user_id)
                    ORDER BY c.embedding <#> :embedding::vector
                    LIMIT :limit
                """), {
                    'embedding': embedding_str, 
                    'limit': request.search_depth,
                    'user_id': request.user_id
                })
                
                plan = json.loads(result.scalar())[0]
                metrics['performance']['vector_search'] = self._extract_metrics(plan)
        
        # Analyze text search if query text provided
        if request.query_text:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
                    SELECT c.id
                    FROM user_document_chunks c
                    JOIN user_documents d ON c.user_document_id = d.id
                    WHERE websearch_to_tsquery('english', :query) @@ c.text_search_vector
                    AND (:user_id IS NULL OR d.user_id = :user_id)
                    LIMIT :limit
                """), {
                    'query': request.query_text, 
                    'limit': request.search_depth,
                    'user_id': request.user_id
                })
                
                plan = json.loads(result.scalar())[0]
                metrics['performance']['text_search'] = self._extract_metrics(plan)
        
        # Analyze hybrid search if both provided
        if request.embedding and request.query_text and request.search_type == SearchType.HYBRID:
            embedding_str = '[' + ','.join(map(str, request.embedding)) + ']'
            
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
                    SELECT * FROM hybrid_search(
                        :query_text,
                        :embedding ::vector(768),
                        :limit,
                        :user_id,
                        :full_text_weight,
                        :semantic_weight,
                        :rrf_k
                    )
                """), {
                    'query_text': request.query_text,
                    'embedding': embedding_str,
                    'limit': request.limit,
                    'user_id': request.user_id,
                    'full_text_weight': request.full_text_weight,
                    'semantic_weight': request.semantic_weight,
                    'rrf_k': request.rrf_k
                })
                
                plan = json.loads(result.scalar())[0]
                metrics['performance']['hybrid_search'] = self._extract_metrics(plan)
        
        return metrics
    
    def _extract_metrics(self, plan: dict) -> dict:
        """Extract key metrics from query plan"""
        return {
            'execution_time_ms': plan['Execution Time'],
            'planning_time_ms': plan['Planning Time'],
            'total_time_ms': plan['Execution Time'] + plan['Planning Time'],
            'rows_returned': plan['Plan'].get('Actual Rows', 0),
            'shared_buffers': {
                'hits': plan['Plan'].get('Shared Hit Blocks', 0),
                'reads': plan['Plan'].get('Shared Read Blocks', 0)
            },
            'uses_index': self._check_index_usage(plan['Plan'])
        }
    
    def _check_index_usage(self, plan: dict) -> dict:
        """Check which indexes are being used"""
        index_usage = {
            'hnsw': False,
            'gin': False,
            'btree': False
        }
        
        def check_node(node):
            node_type = node.get('Node Type', '')
            index_name = node.get('Index Name', '').lower()
            
            if 'Index' in node_type:
                if 'hnsw' in index_name:
                    index_usage['hnsw'] = True
                elif 'gin' in index_name or 'text_search' in index_name:
                    index_usage['gin'] = True
                elif 'btree' in node_type.lower() or 'idx' in index_name:
                    index_usage['btree'] = True
            
            # Recursively check child nodes
            if 'Plans' in node:
                for child in node['Plans']:
                    check_node(child)
        
        check_node(plan)
        return index_usage
    
    def benchmark_search_methods(
        self,
        query_text: str,
        embedding: List[float],
        iterations: int = 5,
        limit: int = 10,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Benchmark different search methods with timing
        
        Args:
            query_text: Search query
            embedding: Query embedding vector
            iterations: Number of iterations for averaging
            limit: Result limit
            user_id: Optional user filter
        
        Returns:
            Benchmark results
        """
        results = {
            'iterations': iterations,
            'limit': limit,
            'methods': {}
        }
        
        # Test vector search
        vector_times = []
        for _ in range(iterations):
            start = time.time()
            self.search_engine.vector_search(embedding, limit, user_id)
            vector_times.append((time.time() - start) * 1000)
        
        results['methods']['vector'] = {
            'avg_ms': np.mean(vector_times),
            'min_ms': np.min(vector_times),
            'max_ms': np.max(vector_times),
            'std_ms': np.std(vector_times)
        }
        
        # Test text search
        text_times = []
        for _ in range(iterations):
            start = time.time()
            self.search_engine.text_search(query_text, limit, user_id)
            text_times.append((time.time() - start) * 1000)
        
        results['methods']['text'] = {
            'avg_ms': np.mean(text_times),
            'min_ms': np.min(text_times),
            'max_ms': np.max(text_times),
            'std_ms': np.std(text_times)
        }
        
        # Test hybrid search (SQL-based)
        hybrid_sql_times = []
        for _ in range(iterations):
            start = time.time()
            self.search_engine.hybrid_search_sql(
                embedding, query_text, limit, 
                search_depth=40, rrf_k=50, user_id=user_id
            )
            hybrid_sql_times.append((time.time() - start) * 1000)
        
        results['methods']['hybrid_sql'] = {
            'avg_ms': np.mean(hybrid_sql_times),
            'min_ms': np.min(hybrid_sql_times),
            'max_ms': np.max(hybrid_sql_times),
            'std_ms': np.std(hybrid_sql_times)
        }
        
        # Test hybrid search (function-based)
        hybrid_func_times = []
        for _ in range(iterations):
            start = time.time()
            self.search_engine.hybrid_search_function(
                embedding, query_text, limit, user_id
            )
            hybrid_func_times.append((time.time() - start) * 1000)
        
        results['methods']['hybrid_function'] = {
            'avg_ms': np.mean(hybrid_func_times),
            'min_ms': np.min(hybrid_func_times),
            'max_ms': np.max(hybrid_func_times),
            'std_ms': np.std(hybrid_func_times)
        }
        
        return results
    
    def display_benchmark_results(self, results: dict):
        """Display benchmark results in a formatted table"""
        table = Table(title="Search Method Benchmark Results")
        table.add_column("Method", style="cyan", width=20)
        table.add_column("Avg (ms)", style="green", width=12)
        table.add_column("Min (ms)", style="yellow", width=12)
        table.add_column("Max (ms)", style="yellow", width=12)
        table.add_column("Std Dev", style="magenta", width=12)
        
        for method, metrics in results['methods'].items():
            table.add_row(
                method.replace('_', ' ').title(),
                f"{metrics['avg_ms']:.2f}",
                f"{metrics['min_ms']:.2f}",
                f"{metrics['max_ms']:.2f}",
                f"{metrics['std_ms']:.2f}"
            )
        
        console.print(table)
        console.print(f"\n[dim]Iterations: {results['iterations']}, Limit: {results['limit']}[/dim]")
    
    def display_performance_analysis(self, metrics: dict):
        """Display detailed performance analysis"""
        console.print("\n[bold cyan]Query Performance Analysis[/bold cyan]\n")
        
        # Request details
        console.print("[bold]Request Parameters:[/bold]")
        for key, value in metrics['request'].items():
            console.print(f"  {key}: {value}")
        
        # Performance metrics
        if metrics['performance']:
            console.print("\n[bold]Performance Metrics:[/bold]")
            
            for search_type, perf in metrics['performance'].items():
                console.print(f"\n  [yellow]{search_type.replace('_', ' ').title()}:[/yellow]")
                console.print(f"    Total Time: {perf['total_time_ms']:.2f}ms")
                console.print(f"    Planning: {perf['planning_time_ms']:.2f}ms")
                console.print(f"    Execution: {perf['execution_time_ms']:.2f}ms")
                console.print(f"    Rows: {perf['rows_returned']}")
                console.print(f"    Buffer Hits: {perf['shared_buffers']['hits']}")
                console.print(f"    Buffer Reads: {perf['shared_buffers']['reads']}")
                
                console.print("    Index Usage:")
                for idx_type, used in perf['uses_index'].items():
                    status = "✓" if used else "✗"
                    color = "green" if used else "red"
                    console.print(f"      [{color}]{status}[/{color}] {idx_type.upper()}")
    
    def run_comprehensive_benchmark(self):
        """Run a comprehensive benchmark with sample data"""
        console.print("[bold magenta]Running Comprehensive Search Benchmark[/bold magenta]\n")
        
        # Load sample data for testing
        with Progress() as progress:
            task = progress.add_task("[cyan]Loading sample data...", total=1)
            dataset = load_dataset(
                "Cohere/wikipedia-22-12-simple-embeddings", 
                split="train[:5]"
            )
            sample = dataset[2]  # Pick a sample
            progress.update(task, advance=1)
        
        query_text = "science technology innovation"
        embedding = sample['emb']
        
        console.print(f"[bold]Test Query:[/bold] '{query_text}'\n")
        
        # Create request for analysis
        request = SearchRequest(
            query_text=query_text,
            embedding=embedding,
            search_type=SearchType.HYBRID,
            limit=10,
            search_depth=40,
            rrf_k=50
        )
        
        # Performance analysis
        console.print("[cyan]Analyzing query performance...[/cyan]")
        metrics = self.analyze_query_performance(request)
        self.display_performance_analysis(metrics)
        
        # Benchmark different methods
        console.print("\n[cyan]Benchmarking search methods...[/cyan]")
        benchmark_results = self.benchmark_search_methods(
            query_text=query_text,
            embedding=embedding,
            iterations=5,
            limit=10
        )
        self.display_benchmark_results(benchmark_results)
        
        # Compare result quality
        console.print("\n[bold cyan]Result Quality Comparison[/bold cyan]")
        self._compare_result_quality(query_text, embedding)
    
    def _compare_result_quality(self, query_text: str, embedding: List[float]):
        """Compare the quality of results from different search methods"""
        # Get results from each method
        vector_results = self.search_engine.vector_search(embedding, limit=5)
        text_results = self.search_engine.text_search(query_text, limit=5)
        hybrid_results = self.search_engine.hybrid_search_sql(
            embedding, query_text, limit=5
        )
        
        # Display comparison
        table = Table(title="Top 5 Results Comparison")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Vector Search", style="yellow", width=30)
        table.add_column("Text Search", style="green", width=30)
        table.add_column("Hybrid Search", style="magenta", width=30)
        
        for i in range(5):
            vector_title = vector_results[i]['title'][:27] + "..." if i < len(vector_results) else "-"
            text_title = text_results[i]['title'][:27] + "..." if i < len(text_results) else "-"
            hybrid_title = hybrid_results[i]['title'][:27] + "..." if i < len(hybrid_results) else "-"
            
            table.add_row(
                str(i + 1),
                vector_title,
                text_title,
                hybrid_title
            )
        
        console.print(table)


def main():
    """Main benchmark interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark leo-pgvector search")
    parser.add_argument("--comprehensive", action="store_true", 
                       help="Run comprehensive benchmark")
    parser.add_argument("--query", help="Query text for custom benchmark")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of iterations for benchmarking")
    parser.add_argument("--limit", type=int, default=10,
                       help="Result limit")
    
    args = parser.parse_args()
    
    benchmark = SearchBenchmark()
    
    if args.comprehensive:
        benchmark.run_comprehensive_benchmark()
    elif args.query:
        # For custom query, we need to generate or load an embedding
        console.print("[yellow]Loading sample embedding for testing...[/yellow]")
        dataset = load_dataset(
            "Cohere/wikipedia-22-12-simple-embeddings", 
            split="train[:1]"
        )
        embedding = dataset[0]['emb']
        
        console.print(f"\n[bold]Benchmarking query:[/bold] '{args.query}'")
        results = benchmark.benchmark_search_methods(
            query_text=args.query,
            embedding=embedding,
            iterations=args.iterations,
            limit=args.limit
        )
        benchmark.display_benchmark_results(results)
    else:
        console.print("[bold cyan]Leo PGVector Search Benchmark[/bold cyan]\n")
        console.print("Usage:")
        console.print("  python bench.py --comprehensive     Run full benchmark suite")
        console.print("  python bench.py --query 'text'      Benchmark specific query")
        console.print("\nOptions:")
        console.print("  --iterations N    Number of iterations (default: 5)")
        console.print("  --limit N         Result limit (default: 10)")


if __name__ == "__main__":
    main()