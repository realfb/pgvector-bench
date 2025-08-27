"""
Analyze SQL queries using EXPLAIN ANALYZE and save results to sql-out/
"""

import os
import json
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()


class QueryAnalyzer:
    """Analyze different query patterns with EXPLAIN ANALYZE"""
    
    def __init__(self):
        self.console = Console()
        db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/leo_pgvector")
        self.engine = create_engine(db_url, echo=False)
        
        # Create output directory
        Path("sql-explain").mkdir(exist_ok=True)
        
        # Get sample data for testing
        self._load_sample_data()
    
    def _load_sample_data(self):
        """Load sample user_id and embedding for testing"""
        with self.engine.connect() as conn:
            # Get a sample user with chunks
            result = conn.execute(text("""
                SELECT user_id, COUNT(*) as chunk_count 
                FROM user_document_chunks 
                GROUP BY user_id 
                HAVING COUNT(*) > 100
                ORDER BY RANDOM() 
                LIMIT 1
            """))
            row = result.fetchone()
            self.sample_user_id = row[0] if row else 1
            
            # Get a sample embedding
            result = conn.execute(text("""
                SELECT embedding 
                FROM user_document_chunks 
                WHERE user_id = :user_id 
                LIMIT 1
            """), {"user_id": self.sample_user_id})
            row = result.fetchone()
            
            if row and row[0]:
                embedding = row[0]
                if isinstance(embedding, str):
                    embedding = embedding.strip('[]')
                    self.sample_embedding = [float(x) for x in embedding.split(',') if x]
                else:
                    self.sample_embedding = list(embedding)
            else:
                # Generate random 768-dim embedding
                import random
                self.sample_embedding = [random.random() for _ in range(768)]
            
            self.sample_embedding_str = "[" + ",".join(map(str, self.sample_embedding)) + "]"
            
            # Get sample metadata values
            result = conn.execute(text("""
                SELECT meta 
                FROM user_document_chunks 
                WHERE user_id = :user_id 
                  AND meta IS NOT NULL 
                LIMIT 1
            """), {"user_id": self.sample_user_id})
            row = result.fetchone()
            self.sample_meta = row[0] if row and row[0] else {}
            
            self.console.print(f"[green]Using sample user_id: {self.sample_user_id}[/green]")
            self.console.print(f"[green]Sample metadata keys: {list(self.sample_meta.keys())[:5]}...[/green]")
    
    def analyze_query(self, name: str, query: str, params: dict) -> dict:
        """Run EXPLAIN ANALYZE on a query and return results"""
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
        
        with self.engine.connect() as conn:
            result = conn.execute(text(explain_query), params)
            explain_json = result.fetchone()[0]
            
            # Extract key metrics
            plan = explain_json[0]
            execution_time = plan.get('Execution Time', 0)
            planning_time = plan.get('Planning Time', 0)
            total_time = execution_time + planning_time
            
            # Get the main plan node
            main_plan = plan.get('Plan', {})
            
            return {
                "name": name,
                "total_time_ms": total_time,
                "execution_time_ms": execution_time,
                "planning_time_ms": planning_time,
                "rows_returned": main_plan.get('Actual Rows', 0),
                "startup_cost": main_plan.get('Startup Cost', 0),
                "total_cost": main_plan.get('Total Cost', 0),
                "node_type": main_plan.get('Node Type', ''),
                "full_plan": explain_json
            }
    
    def format_explain_output(self, explain_json) -> str:
        """Format EXPLAIN output as readable text"""
        plan = explain_json[0]['Plan']
        
        def format_node(node, indent=0):
            """Recursively format plan nodes"""
            lines = []
            prefix = "  " * indent
            
            # Node header
            node_type = node.get('Node Type', 'Unknown')
            if 'Index Name' in node:
                node_type += f" using {node['Index Name']}"
            elif 'Relation Name' in node:
                node_type += f" on {node['Relation Name']}"
                
            actual_time = f"{node.get('Actual Startup Time', 0):.3f}..{node.get('Actual Total Time', 0):.3f}"
            actual_rows = node.get('Actual Rows', 0)
            loops = node.get('Actual Loops', 1)
            
            lines.append(f"{prefix}{node_type} (cost={node.get('Startup Cost', 0):.2f}..{node.get('Total Cost', 0):.2f} rows={node.get('Plan Rows', 0)} width={node.get('Plan Width', 0)}) (actual time={actual_time} rows={actual_rows} loops={loops})")
            
            # Additional details
            if 'Output' in node:
                lines.append(f"{prefix}  Output: {', '.join(node['Output'][:3])}...")
            if 'Index Cond' in node:
                lines.append(f"{prefix}  Index Cond: {node['Index Cond']}")
            if 'Filter' in node:
                lines.append(f"{prefix}  Filter: {node['Filter']}")
            if 'Rows Removed by Filter' in node:
                lines.append(f"{prefix}  Rows Removed by Filter: {node['Rows Removed by Filter']}")
            if 'Sort Key' in node:
                sort_key = str(node['Sort Key'][0])[:100] + "..." if len(str(node['Sort Key'][0])) > 100 else str(node['Sort Key'][0])
                lines.append(f"{prefix}  Sort Key: {sort_key}")
            if 'Sort Method' in node:
                lines.append(f"{prefix}  Sort Method: {node['Sort Method']}  Memory: {node.get('Sort Space Used', 0)}kB")
            
            # Buffer statistics
            shared_hit = node.get('Shared Hit Blocks', 0)
            shared_read = node.get('Shared Read Blocks', 0)
            if shared_hit > 0 or shared_read > 0:
                lines.append(f"{prefix}  Buffers: shared hit={shared_hit} read={shared_read}")
            
            # Process child nodes
            if 'Plans' in node:
                for child in node['Plans']:
                    lines.extend(format_node(child, indent + 1))
                    
            return lines
        
        output_lines = format_node(plan)
        
        # Add planning statistics
        planning = explain_json[0].get('Planning', {})
        output_lines.append("Planning:")
        output_lines.append(f"  Buffers: shared hit={planning.get('Shared Hit Blocks', 0)}")
        output_lines.append(f"Planning Time: {explain_json[0].get('Planning Time', 0):.3f} ms")
        output_lines.append(f"Execution Time: {explain_json[0].get('Execution Time', 0):.3f} ms")
        
        return "\n".join(output_lines)
    
    def run_analysis(self):
        """Run EXPLAIN ANALYZE on all query patterns"""
        self.console.print("\n[bold cyan]Running EXPLAIN ANALYZE on Query Patterns[/bold cyan]\n")
        
        # Track individual query type results for separate files
        text_search_results = []
        semantic_search_results = []
        hybrid_search_results = []
        
        queries = {
            "vector_no_filter": {
                "sql": """
                    SELECT id, user_document_id, embedding <#> :embedding ::vector AS distance
                    FROM user_document_chunks
                    ORDER BY embedding <#> :embedding ::vector
                    LIMIT :limit
                """,
                "params": {"embedding": self.sample_embedding_str, "limit": 10}
            },
            
            "vector_user_filter": {
                "sql": """
                    SELECT id, user_document_id, embedding <#> :embedding ::vector AS distance
                    FROM user_document_chunks
                    WHERE user_id = :user_id
                    ORDER BY embedding <#> :embedding ::vector
                    LIMIT :limit
                """,
                "params": {"user_id": self.sample_user_id, "embedding": self.sample_embedding_str, "limit": 10}
            },
            
            "vector_user_jsonb_simple": {
                "sql": """
                    SELECT id, user_document_id, embedding <#> :embedding ::vector AS distance
                    FROM user_document_chunks
                    WHERE user_id = :user_id
                      AND meta->>'position' = :position
                      AND (meta->>'has_code')::boolean = :has_code
                    ORDER BY embedding <#> :embedding ::vector
                    LIMIT :limit
                """,
                "params": {
                    "user_id": self.sample_user_id,
                    "embedding": self.sample_embedding_str,
                    "position": self.sample_meta.get('position', 'middle'),
                    "has_code": self.sample_meta.get('has_code', False),
                    "limit": 10
                }
            },
            
            "vector_user_jsonb_complex": {
                "sql": """
                    SELECT id, user_document_id, embedding <#> :embedding ::vector AS distance
                    FROM user_document_chunks
                    WHERE user_id = :user_id
                      AND (meta->>'complexity_score')::float >= :min_complexity
                      AND (meta->>'complexity_score')::float <= :max_complexity
                      AND meta->>'language_detected' = 'en'
                    ORDER BY embedding <#> :embedding ::vector
                    LIMIT :limit
                """,
                "params": {
                    "user_id": self.sample_user_id,
                    "embedding": self.sample_embedding_str,
                    "min_complexity": 3.0,
                    "max_complexity": 7.0,
                    "limit": 10
                }
            },
            
            "text_no_filter": {
                "sql": """
                    SELECT id, user_document_id,
                           ts_rank_cd(text_search_vector, websearch_to_tsquery('english', :query)) AS relevance
                    FROM user_document_chunks
                    WHERE websearch_to_tsquery('english', :query) @@ text_search_vector
                    ORDER BY relevance DESC
                    LIMIT :limit
                """,
                "params": {"query": "machine learning", "limit": 10}
            },
            
            "text_user_filter": {
                "sql": """
                    SELECT id, user_document_id,
                           ts_rank_cd(text_search_vector, websearch_to_tsquery('english', :query)) AS relevance
                    FROM user_document_chunks
                    WHERE user_id = :user_id
                      AND websearch_to_tsquery('english', :query) @@ text_search_vector
                    ORDER BY relevance DESC
                    LIMIT :limit
                """,
                "params": {"user_id": self.sample_user_id, "query": "machine learning", "limit": 10}
            },
            
            "hybrid_rrf": {
                "sql": """
                    WITH vector_search AS (
                        SELECT id, user_document_id, paragraph_id, text, meta, created_at,
                               ROW_NUMBER() OVER (ORDER BY embedding <#> :embedding ::vector) AS rank,
                               embedding <#> :embedding ::vector AS distance
                        FROM user_document_chunks
                        WHERE user_id = :user_id
                        ORDER BY embedding <#> :embedding ::vector
                        LIMIT 100
                    ),
                    keyword_search AS (
                        SELECT id, user_document_id, paragraph_id, text, meta, created_at,
                               ROW_NUMBER() OVER (ORDER BY ts_rank_cd(text_search_vector, websearch_to_tsquery('english', :query)) DESC) AS rank,
                               ts_rank_cd(text_search_vector, websearch_to_tsquery('english', :query)) AS relevance
                        FROM user_document_chunks
                        WHERE user_id = :user_id
                          AND websearch_to_tsquery('english', :query) @@ text_search_vector
                        ORDER BY relevance DESC
                        LIMIT 100
                    )
                    SELECT 
                        COALESCE(v.id, k.id) as id,
                        COALESCE(v.user_document_id, k.user_document_id) as user_document_id,
                        COALESCE(v.paragraph_id, k.paragraph_id) as paragraph_id,
                        COALESCE(v.text, k.text) as text,
                        COALESCE(v.meta, k.meta) as meta,
                        COALESCE(v.created_at, k.created_at) as created_at,
                        COALESCE(k.relevance, 0) AS k_score,
                        COALESCE(1.0 / (60 + v.rank), 0) AS v_score,
                        (COALESCE(1.0 / (60 + v.rank), 0) * :vector_weight + 
                         COALESCE(1.0 / (60 + k.rank), 0) * :text_weight) AS score
                    FROM vector_search v
                    FULL OUTER JOIN keyword_search k ON v.id = k.id
                    ORDER BY score DESC
                    LIMIT :limit
                """,
                "params": {
                    "user_id": self.sample_user_id,
                    "embedding": self.sample_embedding_str,
                    "query": "machine learning",
                    "vector_weight": 1.0,
                    "text_weight": 1.0,
                    "limit": 10
                }
            }
        }
        
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create single output file
        output_file = Path("sql-explain") / f"query_analysis_{timestamp}.txt"
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("POSTGRESQL QUERY PERFORMANCE ANALYSIS\n")
            f.write("="*80 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Database: {os.getenv('DATABASE_URL', 'localhost')}\n")
            f.write(f"Sample User ID: {self.sample_user_id}\n")
            f.write("\n")
        
        # Analyze each query
        for query_name, query_info in queries.items():
            self.console.print(f"[cyan]Analyzing: {query_name}...[/cyan]")
            
            try:
                result = self.analyze_query(query_name, query_info["sql"], query_info["params"])
                results.append(result)
                
                # Append to single output file
                with open(output_file, 'a') as f:
                    f.write("\n" + "="*80 + "\n")
                    f.write(f"=== {query_name.upper().replace('_', ' ')} ===\n")
                    f.write("="*80 + "\n\n")
                    
                    # Summary stats
                    f.write("PERFORMANCE SUMMARY:\n")
                    f.write(f"Total Time:     {result['total_time_ms']:.3f} ms\n")
                    f.write(f"Execution Time: {result['execution_time_ms']:.3f} ms\n")
                    f.write(f"Planning Time:  {result['planning_time_ms']:.3f} ms\n")
                    f.write(f"Rows Returned:  {result['rows_returned']}\n")
                    f.write(f"Total Cost:     {result['total_cost']:.2f}\n")
                    f.write("\n")
                    
                    # Query parameters
                    f.write("PARAMETERS:\n")
                    for param_name, param_value in query_info["params"].items():
                        if param_name == "embedding":
                            f.write(f"  {param_name}: [768-dimensional vector]\n")
                        elif isinstance(param_value, str) and len(param_value) > 50:
                            f.write(f"  {param_name}: {param_value[:50]}...\n")
                        else:
                            f.write(f"  {param_name}: {param_value}\n")
                    f.write("\n")
                    
                    # SQL Query
                    f.write("SQL QUERY:\n")
                    f.write("-"*40 + "\n")
                    # Clean up the SQL for better readability
                    sql_lines = query_info["sql"].strip().split('\n')
                    for line in sql_lines:
                        if line.strip():
                            f.write(line.strip() + "\n")
                    f.write("\n")
                    
                    # EXPLAIN output
                    f.write("EXPLAIN ANALYZE OUTPUT:\n")
                    f.write("-"*40 + "\n")
                    explain_output = self.format_explain_output(result["full_plan"])
                    f.write(explain_output + "\n")
                    f.write("\n")
                
                self.console.print(f"  [green]✓ {result['total_time_ms']:.2f}ms total time[/green]")
                
                # Categorize results for individual files
                if 'text' in query_name:
                    text_search_results.append((query_name, query_info, result))
                elif 'vector' in query_name or 'semantic' in query_name:
                    semantic_search_results.append((query_name, query_info, result))
                elif 'hybrid' in query_name:
                    hybrid_search_results.append((query_name, query_info, result))
                
            except Exception as e:
                self.console.print(f"  [red]✗ Error: {e}[/red]")
                results.append({
                    "name": query_name,
                    "error": str(e)
                })
                
                with open(output_file, 'a') as f:
                    f.write("\n" + "="*80 + "\n")
                    f.write(f"=== {query_name.upper().replace('_', ' ')} ===\n")
                    f.write("="*80 + "\n\n")
                    f.write(f"ERROR: {str(e)}\n\n")
        
        # Add summary section
        with open(output_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("=== PERFORMANCE COMPARISON SUMMARY ===\n")
            f.write("="*80 + "\n\n")
            
            # Sort by total time
            sorted_results = sorted([r for r in results if 'error' not in r], 
                                   key=lambda x: x['total_time_ms'])
            
            f.write("Query Type                      | Total (ms) | Exec (ms) | Plan (ms) | Rows | Cost\n")
            f.write("-"*85 + "\n")
            
            for result in sorted_results:
                name = result['name'].replace('_', ' ').title()
                f.write(f"{name:31} | {result['total_time_ms']:10.2f} | {result['execution_time_ms']:9.2f} | {result['planning_time_ms']:9.2f} | {result['rows_returned']:4} | {result['total_cost']:8.0f}\n")
            
            f.write("\n")
            f.write("KEY INSIGHTS:\n")
            f.write("-"*40 + "\n")
            
            # Calculate speedup factors
            no_filter = next((r for r in results if r['name'] == 'vector_no_filter'), None)
            user_filter = next((r for r in results if r['name'] == 'vector_user_filter'), None)
            jsonb_simple = next((r for r in results if r['name'] == 'vector_user_jsonb_simple'), None)
            
            if no_filter and user_filter:
                speedup = no_filter['total_time_ms'] / user_filter['total_time_ms']
                f.write(f"• User filtering is {speedup:.1f}x faster than no filter\n")
            
            if no_filter and jsonb_simple:
                speedup = no_filter['total_time_ms'] / jsonb_simple['total_time_ms']
                f.write(f"• User + JSONB filtering is {speedup:.0f}x faster than no filter\n")
            
            if user_filter and jsonb_simple:
                speedup = user_filter['total_time_ms'] / jsonb_simple['total_time_ms']
                f.write(f"• JSONB filtering adds {speedup:.1f}x speedup over user-only filter\n")
            
            f.write("\n")
        
        # Save individual files for text, semantic, and hybrid searches
        self._save_individual_file("text-search-explain.txt", text_search_results, "TEXT SEARCH")
        self._save_individual_file("semantic-search-explain.txt", semantic_search_results, "SEMANTIC/VECTOR SEARCH")
        self._save_individual_file("hybrid-search-explain.txt", hybrid_search_results, "HYBRID SEARCH")
        
        # Display summary table
        self.display_summary(results)
        
        self.console.print(f"\n[green]Full analysis saved to: {output_file}[/green]")
        self.console.print(f"[green]Individual files saved to sql-explain/[/green]")
    
    def _save_individual_file(self, filename, results_list, query_type):
        """Save individual query analysis to separate files"""
        if not results_list:
            return
            
        output_file = Path("sql-explain") / filename
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"{query_type} QUERY ANALYSIS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for query_name, query_info, result in results_list:
                f.write("\n" + "="*80 + "\n")
                f.write(f"Query: {query_name.replace('_', ' ').title()}\n")
                f.write("="*80 + "\n\n")
                
                # Summary stats
                f.write("PERFORMANCE SUMMARY:\n")
                f.write(f"Total Time:     {result['total_time_ms']:.3f} ms\n")
                f.write(f"Execution Time: {result['execution_time_ms']:.3f} ms\n")
                f.write(f"Planning Time:  {result['planning_time_ms']:.3f} ms\n")
                f.write(f"Rows Returned:  {result['rows_returned']}\n")
                f.write(f"Total Cost:     {result['total_cost']:.2f}\n")
                f.write("\n")
                
                # Query parameters
                f.write("PARAMETERS:\n")
                for param_name, param_value in query_info["params"].items():
                    if param_name == "embedding":
                        f.write(f"  {param_name}: [768-dimensional vector]\n")
                    elif isinstance(param_value, str) and len(param_value) > 50:
                        f.write(f"  {param_name}: {param_value[:50]}...\n")
                    else:
                        f.write(f"  {param_name}: {param_value}\n")
                f.write("\n")
                
                # SQL Query
                f.write("SQL QUERY:\n")
                f.write("-"*40 + "\n")
                sql_lines = query_info["sql"].strip().split('\n')
                for line in sql_lines:
                    if line.strip():
                        f.write(line.strip() + "\n")
                f.write("\n")
                
                # EXPLAIN output
                f.write("EXECUTION PLAN:\n")
                f.write("-"*40 + "\n")
                explain_output = self.format_explain_output(result["full_plan"])
                f.write(explain_output + "\n")
                f.write("\n")
    
    def display_summary(self, results):
        """Display summary table of all queries"""
        self.console.print("\n[bold cyan]Query Performance Summary[/bold cyan]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Query Type", style="cyan")
        table.add_column("Total Time (ms)", justify="right")
        table.add_column("Execution (ms)", justify="right")
        table.add_column("Planning (ms)", justify="right")
        table.add_column("Rows", justify="right")
        table.add_column("Cost", justify="right")
        
        for result in results:
            if "error" not in result:
                table.add_row(
                    result["name"],
                    f"{result['total_time_ms']:.2f}",
                    f"{result['execution_time_ms']:.2f}",
                    f"{result['planning_time_ms']:.2f}",
                    str(result['rows_returned']),
                    f"{result['total_cost']:.0f}"
                )
        
        self.console.print(table)


def main():
    """Run query analysis"""
    analyzer = QueryAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()