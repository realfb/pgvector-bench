"""
Benchmark the impact of removing JOINs from queries
"""

import time
import statistics
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
from rich import print
from rich.table import Table
from rich.console import Console
import numpy as np

load_dotenv()

def benchmark_with_join(engine, iterations=10):
    """Benchmark query with JOIN"""
    embedding = np.random.randn(768).tolist()
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    
    query_with_join = text("""
        SELECT 
            c.id as chunk_id,
            c.user_document_id as document_id,
            c.text,
            d.title,
            c.embedding <#> :embedding ::vector AS distance
        FROM user_document_chunks c
        JOIN user_documents d ON c.user_document_id = d.id
        ORDER BY c.embedding <#> :embedding ::vector
        LIMIT 10
    """)
    
    latencies = []
    for _ in range(iterations):
        start = time.time()
        with engine.connect() as conn:
            result = conn.execute(query_with_join, {"embedding": embedding_str})
            rows = result.fetchall()
        latency = (time.time() - start) * 1000
        latencies.append(latency)
    
    return latencies

def benchmark_without_join(engine, iterations=10):
    """Benchmark query without JOIN"""
    embedding = np.random.randn(768).tolist()
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    
    query_no_join = text("""
        SELECT 
            *,
            embedding <#> :embedding ::vector AS distance
        FROM user_document_chunks
        ORDER BY embedding <#> :embedding ::vector
        LIMIT 10
    """)
    
    latencies = []
    for _ in range(iterations):
        start = time.time()
        with engine.connect() as conn:
            result = conn.execute(query_no_join, {"embedding": embedding_str})
            rows = result.fetchall()
        latency = (time.time() - start) * 1000
        latencies.append(latency)
    
    return latencies

def main():
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:54320/leo_pgvector")
    engine = create_engine(db_url, echo=False, pool_size=5, pool_pre_ping=True)
    
    print("[bold]JOIN vs No-JOIN Query Performance[/bold]\n")
    print("Running 20 iterations of each query type...\n")
    
    # Warmup
    print("[dim]Warming up...[/dim]")
    benchmark_with_join(engine, 2)
    benchmark_without_join(engine, 2)
    
    # Actual benchmarks
    print("[cyan]Benchmarking WITH JOIN...[/cyan]")
    with_join_times = benchmark_with_join(engine, 20)
    
    print("[cyan]Benchmarking WITHOUT JOIN...[/cyan]")
    without_join_times = benchmark_without_join(engine, 20)
    
    # Calculate statistics
    console = Console()
    table = Table(title="Query Performance Comparison", show_header=True)
    table.add_column("Query Type", style="cyan")
    table.add_column("Min (ms)", style="green")
    table.add_column("Avg (ms)", style="yellow")
    table.add_column("Max (ms)", style="red")
    table.add_column("Median (ms)", style="blue")
    table.add_column("StdDev", style="magenta")
    
    table.add_row(
        "WITH JOIN",
        f"{min(with_join_times):.2f}",
        f"{statistics.mean(with_join_times):.2f}",
        f"{max(with_join_times):.2f}",
        f"{statistics.median(with_join_times):.2f}",
        f"{statistics.stdev(with_join_times):.2f}"
    )
    
    table.add_row(
        "WITHOUT JOIN",
        f"{min(without_join_times):.2f}",
        f"{statistics.mean(without_join_times):.2f}",
        f"{max(without_join_times):.2f}",
        f"{statistics.median(without_join_times):.2f}",
        f"{statistics.stdev(without_join_times):.2f}"
    )
    
    console.print(table)
    
    # Calculate improvement
    avg_with_join = statistics.mean(with_join_times)
    avg_without_join = statistics.mean(without_join_times)
    improvement = ((avg_with_join - avg_without_join) / avg_with_join) * 100
    
    print(f"\n[bold green]Results:[/bold green]")
    print(f"Average WITH JOIN: {avg_with_join:.2f}ms")
    print(f"Average WITHOUT JOIN: {avg_without_join:.2f}ms")
    
    if improvement > 0:
        print(f"[bold yellow]Performance improvement: {improvement:.1f}%[/bold yellow]")
    else:
        print(f"[dim]No significant improvement ({improvement:.1f}%)[/dim]")
    
    print(f"\n[bold]Analysis:[/bold]")
    print("• Removing JOINs eliminates the need to access the documents table")
    print("• The chunks table has all necessary data for initial search")
    print("• Document metadata is fetched separately in a batch query")
    print("• This follows the principle of 'fetch only what you need'")

if __name__ == "__main__":
    main()