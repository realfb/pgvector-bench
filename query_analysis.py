import os
import psycopg2.extras
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

load_dotenv()


class QueryAnalyzer:
    def __init__(self):
        self.conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        self.cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        self.output_dir = Path("query-analyses")
        self.output_dir.mkdir(exist_ok=True)

        # Get sample data for testing
        self.cur.execute("""
            SELECT
                embedding,
                user_id,
                user_document_id
            FROM user_document_chunks
            LIMIT 1
        """)
        sample = self.cur.fetchone()
        self.embedding = sample["embedding"]
        self.user_id = sample["user_id"]
        self.text_query = "document"  # Simple text query for testing
        self.document_id = sample["user_document_id"]

        # Get JSONB keys
        self.cur.execute("""
            SELECT DISTINCT jsonb_object_keys(meta) as key
            FROM user_document_chunks
            LIMIT 5
        """)
        self.jsonb_keys = [row["key"] for row in self.cur.fetchall()]
        self.jsonb_key = self.jsonb_keys[0] if self.jsonb_keys else "source"

    def save_analysis(self, name: str, query: str, params: tuple, analysis: dict):
        """Save query analysis to text file."""
        output_file = self.output_dir / f"{name}.txt"

        # Format the query nicely
        formatted_query = query.strip()

        # Build the output text
        output_text = []
        output_text.append("=" * 80)
        output_text.append(f"Query: {name}")
        output_text.append(f"Timestamp: {datetime.now().isoformat()}")
        output_text.append("=" * 80)
        output_text.append("")
        output_text.append("QUERY:")
        output_text.append("-" * 40)
        output_text.append(formatted_query)
        output_text.append("")
        output_text.append(f"Parameters: {len(params)} provided")
        output_text.append("")
        output_text.append("EXECUTION PLAN:")
        output_text.append("-" * 40)

        # Add the plan text
        for row in analysis["plan_rows"]:
            output_text.append(row[0])

        output_text.append("")
        output_text.append("TIMING SUMMARY:")
        output_text.append("-" * 40)
        output_text.append(f"Planning Time: {analysis['planning_time']:.3f} ms")
        output_text.append(f"Execution Time: {analysis['execution_time']:.3f} ms")
        output_text.append(f"Total Time: {analysis['total_time']:.3f} ms")
        output_text.append("")

        # Write to file
        with open(output_file, "w") as f:
            f.write("\n".join(output_text))

        print(f"✓ {name}: {analysis['total_time']:.2f}ms")

    def analyze_query(self, name: str, query: str, params: tuple):
        """Run EXPLAIN ANALYZE BUFFERS on a query."""
        try:
            # Get JSON format for parsing
            json_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
            self.cur.execute(json_query, params)
            json_result = self.cur.fetchone()[0][0]

            # Get text format for readability
            text_query = f"EXPLAIN (ANALYZE, BUFFERS) {query}"
            self.cur.execute(text_query, params)
            text_result = self.cur.fetchall()

            analysis = {
                "plan_json": json_result,
                "plan_rows": text_result,
                "execution_time": json_result["Execution Time"],
                "planning_time": json_result["Planning Time"],
                "total_time": json_result["Execution Time"] + json_result["Planning Time"],
            }

            self.save_analysis(name, query, params, analysis)

        except Exception as e:
            print(f"✗ {name}: {str(e)}")

    def run_all_analyses(self):
        """Run all query analyses."""
        print("Starting Query Analysis...")
        print("=" * 50)

        # Vector Search Queries
        print("\n[Vector Search - Denormalized]")

        self.analyze_query(
            "vector_denorm_no_filter",
            """SELECT id FROM user_document_chunks
               ORDER BY embedding <#> %s::vector
               LIMIT 10""",
            (self.embedding,),
        )

        self.analyze_query(
            "vector_denorm_user_filter",
            """SELECT id FROM user_document_chunks
               WHERE user_id = %s
               ORDER BY embedding <#> %s::vector
               LIMIT 10""",
            (self.user_id, self.embedding),
        )

        self.analyze_query(
            "vector_denorm_jsonb_filter",
            """SELECT id FROM user_document_chunks
               WHERE meta ? %s
               ORDER BY embedding <#> %s::vector
               LIMIT 10""",
            (self.jsonb_key, self.embedding),
        )

        self.analyze_query(
            "vector_denorm_composite_filter",
            """SELECT id FROM user_document_chunks
               WHERE user_id = %s AND meta ? %s
               ORDER BY embedding <#> %s::vector
               LIMIT 10""",
            (self.user_id, self.jsonb_key, self.embedding),
        )

        # Vector Search with JOIN
        print("\n[Vector Search - JOIN]")

        self.analyze_query(
            "vector_join_no_filter",
            """SELECT c.id
               FROM user_document_chunks c
               JOIN user_documents d ON c.user_document_id = d.id
               ORDER BY c.embedding <#> %s::vector
               LIMIT 10""",
            (self.embedding,),
        )

        self.analyze_query(
            "vector_join_user_filter",
            """SELECT c.id
               FROM user_document_chunks c
               JOIN user_documents d ON c.user_document_id = d.id
               WHERE d.user_id = %s
               ORDER BY c.embedding <#> %s::vector
               LIMIT 10""",
            (self.user_id, self.embedding),
        )

        self.analyze_query(
            "vector_join_jsonb_filter",
            """SELECT c.id
               FROM user_document_chunks c
               JOIN user_documents d ON c.user_document_id = d.id
               WHERE c.meta ? %s
               ORDER BY c.embedding <#> %s::vector
               LIMIT 10""",
            (self.jsonb_key, self.embedding),
        )

        # Text Search Queries
        print("\n[Text Search - Denormalized]")

        self.analyze_query(
            "text_denorm_no_filter",
            """SELECT id FROM user_document_chunks
               WHERE text_search_vector @@ websearch_to_tsquery('english', %s)
               ORDER BY ts_rank_cd(text_search_vector, websearch_to_tsquery('english', %s)) DESC
               LIMIT 10""",
            (self.text_query, self.text_query),
        )

        self.analyze_query(
            "text_denorm_user_filter",
            """SELECT id FROM user_document_chunks
               WHERE user_id = %s
               AND text_search_vector @@ websearch_to_tsquery('english', %s)
               ORDER BY ts_rank_cd(text_search_vector, websearch_to_tsquery('english', %s)) DESC
               LIMIT 10""",
            (self.user_id, self.text_query, self.text_query),
        )

        self.analyze_query(
            "text_denorm_jsonb_filter",
            """SELECT id FROM user_document_chunks
               WHERE meta ? %s
               AND text_search_vector @@ websearch_to_tsquery('english', %s)
               ORDER BY ts_rank_cd(text_search_vector, websearch_to_tsquery('english', %s)) DESC
               LIMIT 10""",
            (self.jsonb_key, self.text_query, self.text_query),
        )

        # Text Search with JOIN
        print("\n[Text Search - JOIN]")

        self.analyze_query(
            "text_join_no_filter",
            """SELECT c.id
               FROM user_document_chunks c
               JOIN user_documents d ON c.user_document_id = d.id
               WHERE c.text_search_vector @@ websearch_to_tsquery('english', %s)
               ORDER BY ts_rank_cd(c.text_search_vector, websearch_to_tsquery('english', %s)) DESC
               LIMIT 10""",
            (self.text_query, self.text_query),
        )

        self.analyze_query(
            "text_join_user_filter",
            """SELECT c.id
               FROM user_document_chunks c
               JOIN user_documents d ON c.user_document_id = d.id
               WHERE d.user_id = %s
               AND c.text_search_vector @@ websearch_to_tsquery('english', %s)
               ORDER BY ts_rank_cd(c.text_search_vector, websearch_to_tsquery('english', %s)) DESC
               LIMIT 10""",
            (self.user_id, self.text_query, self.text_query),
        )

        # Hybrid Search Queries (Manual RRF)
        print("\n[Hybrid Search - Denormalized]")

        self.analyze_query(
            "hybrid_denorm_no_filter",
            """WITH vector_results AS (
                   SELECT id, row_number() OVER (
                       ORDER BY embedding <#> %s::vector
                   ) as rank
                   FROM user_document_chunks
                   ORDER BY embedding <#> %s::vector
                   LIMIT 40
               ),
               text_results AS (
                   SELECT id, row_number() OVER (
                       ORDER BY ts_rank_cd(text_search_vector, websearch_to_tsquery('english', %s)) DESC
                   ) as rank
                   FROM user_document_chunks
                   WHERE text_search_vector @@ websearch_to_tsquery('english', %s)
                   LIMIT 40
               )
               SELECT COALESCE(v.id, t.id) as id
               FROM vector_results v
               FULL OUTER JOIN text_results t ON v.id = t.id
               ORDER BY
                   COALESCE(1.0/(v.rank + 50), 0) +
                   COALESCE(1.0/(t.rank + 50), 0) DESC
               LIMIT 10""",
            (self.embedding, self.embedding, self.text_query, self.text_query),
        )

        self.analyze_query(
            "hybrid_denorm_user_filter",
            """WITH vector_results AS (
                   SELECT id, row_number() OVER (
                       ORDER BY embedding <#> %s::vector
                   ) as rank
                   FROM user_document_chunks
                   WHERE user_id = %s
                   ORDER BY embedding <#> %s::vector
                   LIMIT 40
               ),
               text_results AS (
                   SELECT id, row_number() OVER (
                       ORDER BY ts_rank_cd(text_search_vector, websearch_to_tsquery('english', %s)) DESC
                   ) as rank
                   FROM user_document_chunks
                   WHERE user_id = %s
                   AND text_search_vector @@ websearch_to_tsquery('english', %s)
                   LIMIT 40
               )
               SELECT COALESCE(v.id, t.id) as id
               FROM vector_results v
               FULL OUTER JOIN text_results t ON v.id = t.id
               ORDER BY
                   COALESCE(1.0/(v.rank + 50), 0) +
                   COALESCE(1.0/(t.rank + 50), 0) DESC
               LIMIT 10""",
            (self.embedding, self.user_id, self.embedding, self.text_query, self.user_id, self.text_query),
        )

        # Hybrid with JOIN
        print("\n[Hybrid Search - JOIN]")

        self.analyze_query(
            "hybrid_join_user_filter",
            """WITH vector_results AS (
                   SELECT c.id, row_number() OVER (
                       ORDER BY c.embedding <#> %s::vector
                   ) as rank
                   FROM user_document_chunks c
                   JOIN user_documents d ON c.user_document_id = d.id
                   WHERE d.user_id = %s
                   ORDER BY c.embedding <#> %s::vector
                   LIMIT 40
               ),
               text_results AS (
                   SELECT c.id, row_number() OVER (
                       ORDER BY ts_rank_cd(c.text_search_vector, websearch_to_tsquery('english', %s)) DESC
                   ) as rank
                   FROM user_document_chunks c
                   JOIN user_documents d ON c.user_document_id = d.id
                   WHERE d.user_id = %s
                   AND c.text_search_vector @@ websearch_to_tsquery('english', %s)
                   LIMIT 40
               )
               SELECT COALESCE(v.id, t.id) as id
               FROM vector_results v
               FULL OUTER JOIN text_results t ON v.id = t.id
               ORDER BY
                   COALESCE(1.0/(v.rank + 50), 0) +
                   COALESCE(1.0/(t.rank + 50), 0) DESC
               LIMIT 10""",
            (self.embedding, self.user_id, self.embedding, self.text_query, self.user_id, self.text_query),
        )

        # Special Cases
        print("\n[Special Cases]")

        # Test with disabled nested loop
        self.cur.execute("SET enable_nestloop = off")
        self.analyze_query(
            "vector_join_no_nestloop",
            """SELECT c.id
               FROM user_document_chunks c
               JOIN user_documents d ON c.user_document_id = d.id
               ORDER BY c.embedding <#> %s::vector
               LIMIT 10""",
            (self.embedding,),
        )
        self.cur.execute("SET enable_nestloop = on")

        # Test with forced index scan
        self.cur.execute("SET enable_seqscan = off")
        self.analyze_query(
            "vector_denorm_force_index",
            """SELECT id FROM user_document_chunks
               ORDER BY embedding <#> %s::vector
               LIMIT 10""",
            (self.embedding,),
        )
        self.cur.execute("SET enable_seqscan = on")

        print("\n" + "=" * 50)
        print("Analysis Complete!")
        print(f"Results saved in: {self.output_dir}/")

        # Generate summary
        self.generate_summary()

    def generate_summary(self):
        """Generate a summary of all analyses."""
        summary_data = []

        # Parse timing from txt files
        for txt_file in self.output_dir.glob("*.txt"):
            if txt_file.name == "summary.txt":
                continue

            with open(txt_file) as f:
                content = f.read()

            # Extract query name from filename
            query_name = txt_file.stem

            # Extract timings using simple parsing
            execution_time = None
            planning_time = None
            total_time = None

            for line in content.split("\n"):
                if line.startswith("Execution Time:"):
                    execution_time = float(line.split(":")[1].replace("ms", "").strip())
                elif line.startswith("Planning Time:"):
                    planning_time = float(line.split(":")[1].replace("ms", "").strip())
                elif line.startswith("Total Time:"):
                    total_time = float(line.split(":")[1].replace("ms", "").strip())

            if total_time is not None:
                summary_data.append(
                    {
                        "name": query_name,
                        "execution_time": execution_time,
                        "planning_time": planning_time,
                        "total_time": total_time,
                    }
                )

        # Sort by total time
        summary_data.sort(key=lambda x: x["total_time"])

        # Create summary text file
        summary_text = []
        summary_text.append("=" * 80)
        summary_text.append("QUERY ANALYSIS SUMMARY")
        summary_text.append(f"Generated: {datetime.now().isoformat()}")
        summary_text.append("=" * 80)
        summary_text.append("")
        summary_text.append("ALL QUERIES RANKED BY TOTAL TIME:")
        summary_text.append("-" * 40)

        for item in summary_data:
            summary_text.append(
                f"{item['name']:40} {item['total_time']:8.2f}ms "
                f"(exec: {item['execution_time']:7.2f}ms, plan: {item['planning_time']:6.2f}ms)"
            )

        if summary_data:
            summary_text.append("")
            summary_text.append("HIGHLIGHTS:")
            summary_text.append("-" * 40)
            summary_text.append(f"Fastest: {summary_data[0]['name']} ({summary_data[0]['total_time']:.2f}ms)")
            summary_text.append(
                f"Slowest: {summary_data[-1]['name']} ({summary_data[-1]['total_time']:.2f}ms)"
            )
            summary_text.append(
                f"Range: {summary_data[-1]['total_time'] / summary_data[0]['total_time']:.1f}x difference"
            )

        # Write summary
        with open(self.output_dir / "summary.txt", "w") as f:
            f.write("\n".join(summary_text))

        # Print summary
        print("\n[Performance Summary]")
        if summary_data:
            print(f"Fastest: {summary_data[0]['name']} ({summary_data[0]['total_time']:.2f}ms)")
            print(f"Slowest: {summary_data[-1]['name']} ({summary_data[-1]['total_time']:.2f}ms)")

    def close(self):
        """Close database connection."""
        self.cur.close()
        self.conn.close()


if __name__ == "__main__":
    analyzer = QueryAnalyzer()
    try:
        analyzer.run_all_analyses()
    finally:
        analyzer.close()
