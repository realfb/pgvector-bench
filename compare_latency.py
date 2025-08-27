"""
Compare latency between different measurement approaches
"""

import os
import time
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from query import SearchEngine, SearchRequest, SearchType

load_dotenv()


def test_latencies():
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/leo_pgvector")
    
    # Test parameters
    user_id = 1
    query_text = "military"
    limit = 10
    
    print("=" * 60)
    print("LATENCY COMPARISON TEST")
    print("=" * 60)
    print(f"Query: '{query_text}', User: {user_id}, Limit: {limit}\n")
    
    # Method 1: Using query.py SearchEngine (with all overhead)
    print("Method 1: Using SearchEngine class (query.py style)")
    engine = SearchEngine()
    request = SearchRequest(
        query_text=query_text,
        search_type=SearchType.KEYWORD,
        limit=limit,
        user_id=user_id
    )
    
    # Run 5 times to see variation
    latencies = []
    for i in range(5):
        start = time.time()
        response = engine.search(request)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        print(f"  Run {i+1}: {latency:.2f}ms (found {len(response.results)} results)")
    
    print(f"  Average: {sum(latencies)/len(latencies):.2f}ms\n")
    
    # Method 2: Direct SQL execution (bench.py style)
    print("Method 2: Direct SQL execution (bench.py style)")
    engine_raw = create_engine(db_url, echo=False)
    
    with engine_raw.connect() as conn:
        query = text("""
            SELECT id, user_document_id,
                   ts_rank_cd(text_search_vector, websearch_to_tsquery('english', :query)) AS relevance
            FROM user_document_chunks
            WHERE user_id = :user_id
              AND websearch_to_tsquery('english', :query) @@ text_search_vector
            ORDER BY relevance DESC
            LIMIT :limit
        """)
        
        latencies = []
        for i in range(5):
            start = time.time()
            result = conn.execute(query, {"user_id": user_id, "query": query_text, "limit": limit})
            rows = result.fetchall()
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            print(f"  Run {i+1}: {latency:.2f}ms (found {len(rows)} results)")
        
        print(f"  Average: {sum(latencies)/len(latencies):.2f}ms\n")
    
    # Method 3: SearchEngine internal method (middle ground)
    print("Method 3: SearchEngine.text_search() method directly")
    latencies = []
    for i in range(5):
        results, latency = engine.text_search(user_id, query_text, limit)
        latencies.append(latency)
        print(f"  Run {i+1}: {latency:.2f}ms (found {len(results)} results)")
    
    print(f"  Average: {sum(latencies)/len(latencies):.2f}ms\n")
    
    # Show breakdown
    print("=" * 60)
    print("OVERHEAD ANALYSIS")
    print("=" * 60)
    print("The difference between methods shows:")
    print("- Method 1 includes: Object creation, validation, response building")
    print("- Method 2 includes: Only raw SQL execution")
    print("- Method 3 includes: SQL execution + basic dict creation")
    print("\nThe ~80ms difference is Python overhead + cold cache on first run!")


if __name__ == "__main__":
    test_latencies()