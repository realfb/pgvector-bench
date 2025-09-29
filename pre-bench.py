import os
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from llm import GeminiClient
from embedding import EmbeddingClient
from pydantic import BaseModel

load_dotenv()


# Pydantic models for structured query generation
class BenchmarkQuery(BaseModel):
    query_text: str


class BenchmarkQueries(BaseModel):
    queries: List[str]  # List of query strings


class BenchmarkQueryGenerator:
    """Generate benchmark queries for search evaluation"""

    def __init__(self):
        self.console = Console()
        self.gemini = GeminiClient(model="gemini-2.0-flash-exp")  # Using faster model
        self.embedder = EmbeddingClient()  # For generating query embeddings

        # Connect to database
        db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/leo_pgvector")
        self.engine = create_engine(db_url, echo=False)

        # Create data directory
        Path("data").mkdir(exist_ok=True)

    def fetch_random_documents(self, n: int = 50) -> List[Dict]:
        """Fetch n random unique documents with their chunks"""
        self.console.print(f"[cyan]Fetching {n} random documents from database...[/cyan]")

        with self.engine.connect() as conn:
            # Get random unique documents with metadata
            query = text("""
                WITH random_docs AS (
                    SELECT DISTINCT ON (d.id)
                        d.id as document_id,
                        d.title as document_title,
                        d.user_id,
                        c.id as chunk_id,
                        c.paragraph_id,
                        c.text as chunk_text,
                        c.embedding,
                        c.meta
                    FROM user_documents d
                    JOIN user_document_chunks c ON d.id = c.user_document_id
                    WHERE LENGTH(c.text) > 300
                    ORDER BY d.id, RANDOM()
                )
                SELECT * FROM random_docs
                ORDER BY RANDOM()
                LIMIT :limit
            """)

            result = conn.execute(query, {"limit": n})

            documents = []
            for row in result:
                # Handle embedding conversion
                embedding = row[6]
                if isinstance(embedding, str):
                    embedding_str = embedding.strip("[]")
                    embedding_list = [float(x) for x in embedding_str.split(",") if x]
                else:
                    embedding_list = list(embedding) if embedding else []

                # Parse metadata
                meta = row[7] if row[7] else {}

                documents.append(
                    {
                        "document_id": row[0],
                        "document_title": row[1],
                        "user_id": row[2],
                        "chunk_id": row[3],
                        "paragraph_id": row[4],
                        "chunk_text": row[5],
                        "embedding": embedding_list,
                        "meta": meta,
                    }
                )

            return documents

    def generate_queries_batch(self, documents: List[Dict], queries_per_doc: int = 2) -> List[str]:
        """Generate queries for a batch of documents using Gemini"""

        # Prepare batch prompt
        prompt = f"""You are an expert at creating search queries that test information retrieval systems.

Generate challenging but realistic Google-style search queries for the following document chunks.

REQUIREMENTS:
- Create queries that someone might actually search for to find this content
- Use varied query styles: questions, keywords, concepts, problems
- Avoid exact phrase matches from the text (too easy)
- Mix different difficulty levels
- Think like different types of users (expert, novice, curious)
- Include typos occasionally for realism

For each document chunk below, generate exactly {queries_per_doc} search queries.

DOCUMENT CHUNKS:
"""

        # Add document chunks to prompt
        for i, doc in enumerate(documents[:10], 1):  # Process 10 at a time
            prompt += f"\n--- Document {i} ---\n"
            prompt += f"Title: {doc['document_title'][:100]}\n"
            prompt += f"Text: {doc['chunk_text'][:500]}...\n"

        total_queries = queries_per_doc * min(10, len(documents))
        prompt += f"\n\nGenerate {total_queries} queries total ({queries_per_doc} per document). Return as a JSON object with a 'queries' field containing a list of strings."

        # Generate queries using structured output
        try:
            result = self.gemini.generate_structured(
                prompt, response_schema=BenchmarkQueries, temperature=0.8, max_output_tokens=1500
            )

            if result and hasattr(result, "queries"):
                return result.queries
            else:
                self.console.print("[yellow]Warning: No queries generated[/yellow]")
                return []

        except Exception as e:
            self.console.print(f"[red]Error generating queries: {e}[/red]")
            return []

    def generate_benchmark_data(self, num_queries: int = 100) -> str:
        """Generate benchmark data with specified number of queries"""
        self.console.print(f"[bold cyan]Generating {num_queries} Benchmark Queries[/bold cyan]\n")

        # Calculate how many documents we need
        queries_per_doc = 2
        num_docs_needed = (num_queries + queries_per_doc - 1) // queries_per_doc

        # Fetch random documents
        documents = self.fetch_random_documents(num_docs_needed)

        if not documents:
            self.console.print("[red]No documents found in database![/red]")
            return ""

        self.console.print(f"[green]✓ Fetched {len(documents)} unique documents[/green]\n")

        # Generate queries in batches
        benchmark_data = []
        batch_size = 10

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Generating queries with Gemini...", total=len(documents))

            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]

                # Generate queries for this batch
                queries = self.generate_queries_batch(batch, queries_per_doc)

                # Map queries to documents
                for j, query_text in enumerate(queries):
                    if j < len(batch) * queries_per_doc:
                        doc_idx = j // queries_per_doc
                        if doc_idx < len(batch):
                            doc = batch[doc_idx]

                            benchmark_data.append(
                                {
                                    "query_id": len(benchmark_data) + 1,  # Simple integer ID
                                    "query_text": query_text,
                                    "ground_truth_chunk_id": doc["chunk_id"],
                                    "ground_truth_document_id": doc["document_id"],
                                    "ground_truth_user_id": doc["user_id"],
                                    "ground_truth_paragraph_id": doc["paragraph_id"],
                                    "document_title": doc["document_title"],
                                    "chunk_text_preview": doc["chunk_text"][:200] + "...",
                                    "original_embedding": doc["embedding"],  # Keep original for reference
                                    "meta": doc.get("meta", {}),  # Include metadata
                                }
                            )

                progress.advance(task, advance=len(batch))

                # Stop if we have enough queries
                if len(benchmark_data) >= num_queries:
                    break

        # Trim to exact number requested
        benchmark_data = benchmark_data[:num_queries]

        # Generate embeddings for all queries
        self.console.print("\n[cyan]Generating embeddings for queries...[/cyan]")
        query_texts = [item["query_text"] for item in benchmark_data]

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Computing embeddings with Cohere...", total=1)

            # Generate all query embeddings at once
            query_embeddings = self.embedder.embed_texts(query_texts, input_type="search_query")

            # Add embeddings to benchmark data
            for i, embedding in enumerate(query_embeddings):
                benchmark_data[i]["embedding"] = embedding

            progress.advance(task)

        self.console.print(f"[green]✓ Generated embeddings for {len(query_embeddings)} queries[/green]")

        # Save to files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = Path("data") / f"benchmark_{timestamp}.csv"
        json_file = Path("data") / f"benchmark_{timestamp}.json"

        # Save CSV (without embeddings)
        fieldnames = [
            "query_id",
            "query_text",
            "ground_truth_chunk_id",
            "ground_truth_document_id",
            "ground_truth_user_id",
            "ground_truth_paragraph_id",
            "document_title",
            "chunk_text_preview",
            "meta",
        ]

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in benchmark_data:
                csv_row = {k: row.get(k, "") for k in fieldnames}
                writer.writerow(csv_row)

        # Save embeddings and metadata to JSON
        embeddings_data = {}
        for row in benchmark_data:
            embeddings_data[str(row["query_id"])] = {
                "query_embedding": row["embedding"],
                "document_embedding": row.get("original_embedding", []),
                "meta": row.get("meta", {}),
            }
        with open(json_file, "w") as f:
            json.dump(embeddings_data, f)

        self.console.print(f"\n[green]✓ Generated {len(benchmark_data)} benchmark queries[/green]")
        self.console.print(f"[green]✓ Saved to:[/green]")
        self.console.print(f"  - {csv_file}")
        self.console.print(f"  - {json_file}")

        # Display sample queries
        self.console.print("\n[cyan]Sample queries:[/cyan]")
        for item in benchmark_data[:5]:
            self.console.print(f"  - '{item['query_text']}' → Doc: {item['document_title'][:50]}...")

        return str(csv_file)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark queries using Gemini")
    parser.add_argument(
        "--queries", type=int, default=100, help="Number of queries to generate (default: 100)"
    )

    args = parser.parse_args()

    try:
        generator = BenchmarkQueryGenerator()
        csv_file = generator.generate_benchmark_data(num_queries=args.queries)

        if csv_file:
            console = Console()
            console.print(f"\n[bold cyan]To evaluate these queries, run:[/bold cyan]")
            console.print(f"  python eval_queries.py {csv_file}")

    except Exception as e:
        Console().print(f"[bold red]Error:[/bold red] {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
